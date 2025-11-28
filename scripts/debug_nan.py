import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"

print("Loading Teacher...")
model = AutoModelForCausalLM.from_pretrained(
    TEACHER_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager" 
)
tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)

texts = [
    "Hello, this is a test for UZR Garage.",
    "인공지능 모델의 지식을 증류하는 과정은 매우 흥미롭습니다.", 
    "Short.",
    "A very long sentence " * 50
]

print("Running inference...")
for i, text in enumerate(texts):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
        
    attn = out.attentions[-1][0] # (Head, S, S)
    
    if torch.isnan(attn).any():
        print(f"Sample {i}: NaN detected!")
        print(f"Max: {attn.max()}, Min: {attn.min()}")
    else:
        print(f"Sample {i}: OK. Max: {attn.max()}, Min: {attn.min()}")
