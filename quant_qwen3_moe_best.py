import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping
from llmcompressor.utils import dispatch_for_generation

# ================= 配置区 =================
# 1. 模型路径
MODEL_ID = "/root/autodl-tmp/models/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated" 
SAVE_DIR = "/root/autodl-tmp/models/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-AWQ-Int4-Best"

# 2. 校准数据集配置
# 使用混合数据集（Nemotron + girlfriend_calib）
# 来源：90% nvidia/Llama-Nemotron-Post-Training-Dataset + 10% girlfriend_calib.jsonl
# 配比：256个样本
#   - 99% (255个) Nemotron SFT/chat 样本（保持泛化性）
#   - 1% (1个) 数字人女友日常对话样本（贴近真实使用场景）
#   - 中文占比：50%（128个样本）
# 优势：
#   - 保留 Nemotron 多样性和泛化能力（参考 cyankiwi 实测验证）
#   - 适度注入真实使用场景，校准更贴近实际激活分布
#   - 本地文件，避免网络下载
DATA_PATH = "dataset_nemotron_calibration_chinese.jsonl"

# 3. 量化参数配置
# 参考 HF recipe.yaml 的质量优先配置，但可以适度调整样本数/长度来控制时间
NUM_CALIBRATION_SAMPLES = 256  # 可改为 128 或 64 加速
MAX_SEQUENCE_LENGTH = 512      # 可改为 256 或 384 加速

# ================= 1. 加载模型与分词器 =================
print(f"正在加载模型: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")

# ================= 2. 加载并处理数据集 =================
print(f"正在从本地文件 {DATA_PATH} 加载校准数据...")

# 从本地 JSONL 文件加载数据集
samples = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))

print(f"成功加载 {len(samples)} 个本地样本")
print(f"正在处理校准数据...")

# 预处理函数：将 Nemotron SFT 格式转换为模型可读的字符串
# Nemotron SFT 格式: {"input": [{"role": "user", "content": "..."}], "output": "..."}
def preprocess_fn(example):
    # 构建完整对话：input (用户消息列表) + output (助手回复)
    messages = example.get("input", [])
    if example.get("output"):
        messages = messages + [{"role": "assistant", "content": example["output"]}]
    
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    }

# 转换为 Dataset 格式
from datasets import Dataset
ds = Dataset.from_list(samples)

# 应用预处理
ds = ds.map(preprocess_fn)

# 只使用需要的样本数
if len(ds) > NUM_CALIBRATION_SAMPLES:
    ds = ds.select(range(NUM_CALIBRATION_SAMPLES))

print(f"成功处理 {len(ds)} 个样本用于校准")

# ================= 3. 配置量化策略 (基于 HF recipe.yaml 的高质量配置) =================
print("配置 AWQ 量化策略 (高质量模式)...")

# 参考 https://huggingface.co/cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit/raw/main/recipe.yaml
# 这是针对 Qwen3 30B MoE 的完整 AWQ 配置，包含：
# - 精细化的 ignore 列表（排除 embed、layernorm、lm_head、MoE gate 等敏感层）
# - 完整的 mappings（为 Qwen3 MoE 定制的 smoothing/balance 规则）
# - duo_scaling + MSE observer + offload_device 等质量优化选项

recipe = AWQModifier(
    # 目标：量化所有 Linear 层
    targets=["Linear"],
    
    # 忽略（不量化）的模块：保持 FP16 以维持稳定性
    ignore=[
        "lm_head",                           # 最终输出头
        "model.embed_tokens",                # 词嵌入层
        "re:.*input_layernorm$",             # 各层输入 LayerNorm
        "re:.*post_attention_layernorm$",    # 各层注意力后 LayerNorm
        "model.norm",                        # 顶层 norm
        "re:.*mlp.gate$",                    # MoE 路由 gate（关键！）
    ],
    
    # AWQ smoothing mappings（针对 Qwen3 MoE 结构的精确映射）
    mappings=[
        # 1. input_layernorm 平滑到 Q/K/V 投影
        AWQMapping(
            smooth_layer="re:.*input_layernorm$",
            balance_layers=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"]
        ),
        # 2. V 投影平滑到 O 投影
        AWQMapping(
            smooth_layer="re:.*v_proj$",
            balance_layers=["re:.*o_proj$"]
        ),
        # 3. post_attention_layernorm 平滑到 MoE MLP 的 gate_proj/up_proj
        AWQMapping(
            smooth_layer="re:.*post_attention_layernorm$",
            balance_layers=["re:.*mlp.experts.*.gate_proj$", "re:.*mlp.experts.*.up_proj$"]
        ),
        # 4. up_proj 平滑到 down_proj
        AWQMapping(
            smooth_layer="re:.*up_proj$",
            balance_layers=["re:.*down_proj$"]
        ),
    ],
    
    # 权重量化配置：INT4 per-group (group_size=32)
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "group_size": 32,         # 更细粒度，精度更好（相比默认的128）
                "strategy": "group",
                "block_structure": None,
                "dynamic": False,
                "actorder": None,
                "observer": "mse",        # MSE observer 比 minmax 稳定
                "observer_kwargs": {},
            },
            "input_activations": None,
            "output_activations": None,
        }
    },
    
    # 高质量选项
    duo_scaling=True,              # 同时使用激活和权重来确定 scale
    n_grid=20,                     # 网格搜索点数（越大越精确但越慢）
    # offload_device="cpu",          # 中间结果 offload 到 CPU 节省显存
)

# ================= 4. 执行量化 (Oneshot) =================
print("="*60)
print("开始量化校准...")
print(f"  - 校准样本数: {NUM_CALIBRATION_SAMPLES}")
print(f"  - 序列长度: {MAX_SEQUENCE_LENGTH}")
print(f"  - 量化方案: W4A16 (INT4 权重, group_size=32, MSE observer)")
print(f"  - AWQ 优化: duo_scaling=True, n_grid=20")
print("="*60)
print("预计耗时：2～3 小时（30B MoE 模型，视硬件而定）")
print("提示：可通过减少 NUM_CALIBRATION_SAMPLES 或 MAX_SEQUENCE_LENGTH 加速")
print("="*60)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ================= 5. 测试生成效果 =================
print("\n" + "="*20 + " 量化模型测试生成 " + "="*20)
dispatch_for_generation(model)

test_prompt = "亲爱的，今天辛苦了，晚饭想吃什么呀？"
inputs = [{"role": "user", "content": test_prompt}]
input_text = tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

output = model.generate(
    input_ids, 
    max_new_tokens=128,
    temperature=0.7,
    top_k=20,
    top_p=0.8
)

print(f"输入: {test_prompt}")
print(f"输出: {tokenizer.decode(output[0], skip_special_tokens=True).split('assistant')[-1].strip()}")
print("="*50 + "\n")

# ================= 6. 保存模型 =================
print(f"正在保存压缩后的模型至: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print("="*60)
print("保存完成！")
print(f"模型已保存到: {SAVE_DIR}")
print("可以使用 vLLM 或 Transformers 直接加载。")
print("="*60)
