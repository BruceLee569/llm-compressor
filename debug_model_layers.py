"""调试脚本：打印模型的所有层名称，用于检查 LayerNorm 命名"""
from transformers import AutoModelForCausalLM
import re

MODEL_ID = "/root/autodl-tmp/models/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated"

print(f"加载模型: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="cpu")

print("\n" + "="*80)
print("所有包含 'norm' 的层名：")
print("="*80)

norm_layers = []
for name, module in model.named_modules():
    if "norm" in name.lower():
        norm_layers.append((name, type(module).__name__))

for name, type_name in sorted(norm_layers):
    print(f"{name:60s} -> {type_name}")

print("\n" + "="*80)
print("所有包含 'proj' 的层名（前20个）：")
print("="*80)

proj_layers = []
for name, module in model.named_modules():
    if "proj" in name.lower():
        proj_layers.append((name, type(module).__name__))

for name, type_name in sorted(proj_layers)[:20]:
    print(f"{name:60s} -> {type_name}")

print("\n" + "="*80)
print("检查 AWQ mapping 正则匹配：")
print("="*80)

patterns = [
    "re:.*input_layernorm$",
    "re:.*post_attention_layernorm$",
    "re:.*q_proj$",
    "re:.*k_proj$",
    "re:.*v_proj$",
    "re:.*o_proj$",
    "re:.*mlp.gate$",
]

all_names = [name for name, _ in model.named_modules()]

for pattern in patterns:
    regex = pattern.replace("re:", "")
    matches = [name for name in all_names if re.match(regex, name)]
    print(f"\n模式: {pattern}")
    print(f"  匹配数量: {len(matches)}")
    if matches:
        print(f"  示例: {matches[0] if matches else '无'}")
    else:
        print(f"  ❌ 没有匹配！")
