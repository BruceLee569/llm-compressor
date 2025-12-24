import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

# ================= 配置区 =================
# 1. 模型路径
MODEL_ID = "/root/autodl-tmp/models/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated" 
SAVE_DIR = "/root/autodl-tmp/models/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-AWQ-Int4-RP"

# 2. 本地校准数据集路径 (jsonl格式)
# 建议该文件包含：{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
DATA_PATH = "girlfriend_calib.jsonl" 

# 3. 量化参数配置
NUM_CALIBRATION_SAMPLES = 256  # 校准样本数，建议128-512之间
MAX_SEQUENCE_LENGTH = 512     # 序列长度，数字人对话可以适当增长到1024-2048以捕捉上下文

# ================= 1. 加载模型与分词器 =================
print(f"正在加载模型: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")

# ================= 2. 加载并处理本地数据 =================
print(f"正在从 {DATA_PATH} 加载校准数据...")

# 加载本地 jsonl 文件
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 预处理函数：将对话格式转换为模型可读的字符串
def preprocess_fn(example):
    # 使用模型自带的 Chat Template 保持训练/推理一致性
    # 假设你的 jsonl 中每一行是 {"messages": [...]}
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

# 转换数据格式并洗牌（oneshot 内部会自动 tokenize，不需要手动调用）
ds = dataset.shuffle(seed=42).select(range(min(NUM_CALIBRATION_SAMPLES, len(dataset))))
ds = ds.map(preprocess_fn, remove_columns=dataset.column_names)

# ================= 3. 配置量化策略 (AWQ) =================
# 针对 Qwen MoE 模型优化的 AWQ 配置
# 说明：W4A16 表示 权重Int4，激活值FP16/BF16
recipe = [
    AWQModifier(
        # 排除非线性层及MoE路由层，以维持精度
        ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        scheme="W4A16", 
        targets=["Linear"],
    ),
]

# ================= 4. 执行量化 (Oneshot) =================
print("开始量化校准，这可能需要一段时间...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,  # 使用配置的样本数，而非 len(ds)
)

# ================= 5. 测试生成效果 =================
print("\n" + "="*20 + " 量化模型测试生成 " + "="*20)
dispatch_for_generation(model) # 优化生成速度的调度

test_prompt = "亲爱的，今天辛苦了，晚饭想吃什么呀？"
# 将输入包装成对话模板
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
print("保存完成！可以使用 vLLM 或 Transformers 直接加载。")
