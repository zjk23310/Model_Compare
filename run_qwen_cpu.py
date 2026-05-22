from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "/mnt/data/Models_Compare/Qwen-7B-Chat"  # 本地路径
prompt = "他知道我知道你知道他不知道吗？这句话里，到底谁不知道"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True  # Qwen 建议使用慢速 tokenizer
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 使用 float16 节省显存
    device_map="auto",  # 自动分配到可用设备
    low_cpu_mem_usage=True  # 节省 CPU 内存
).eval()

# 生成输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 创建流式输出
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 生成回复
print("模型回答：")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,  # 增加到 512
        temperature=0.7,     # 控制随机性
        top_p=0.8,           # 核采样
        do_sample=True,      # 启用采样
        repetition_penalty=1.05  # 减少重复
    )