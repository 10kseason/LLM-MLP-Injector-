from transformers import AutoModelForCausalLM
import inspect

model_id = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
layer = model.model.layers[0]
attn = layer.self_attn
rotary = attn.rotary_emb

print(f"Rotary Class: {type(rotary)}")
print(f"Rotary Forward Sig: {inspect.signature(rotary.forward)}")
print(f"Attn Forward Sig: {inspect.signature(attn.forward)}")
