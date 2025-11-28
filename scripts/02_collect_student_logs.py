import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "student_logs")
os.makedirs(OUT_DIR, exist_ok=True)

class StudentLogger:
    def __init__(self, model_name, layer_to_hook=8, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            # load_in_4bit=True, # Optional for small model
            attn_implementation="eager"
        )
        self.layer_to_hook = layer_to_hook
        self.hidden_logs = []
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self):
        def fn(module, input, output):
            # output is usually (hidden_states, ) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # hidden: (B, S, D)
            self.hidden_logs.append(hidden.detach().cpu().numpy().astype(np.float16))
        return fn

    def _register_hooks(self):
        # Qwen2 structure: model.model.layers[i]
        # We want the output of the layer, or input?
        # User said: "Student의 히든 상태(또는 토큰 특징)" -> "Controller 입력으로 쓸 레이어"
        # Usually we take the output of layer N to predict Teacher layer M's attention.
        layer = self.model.model.layers[self.layer_to_hook]
        h = layer.register_forward_hook(self._hook_fn())
        self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def encode_with_logging(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.model.device)

        self.hidden_logs.clear()
        with torch.no_grad():
            _ = self.model(**enc)
        
        return enc, self.hidden_logs

def main():
    # User said "Qwen3 0.6B" -> likely Qwen2.5-0.5B
    model_name = "Qwen/Qwen2.5-0.5B" 
    print(f"Loading Student model: {model_name}")
    
    # Hooking layer 8 (arbitrary mid layer for now, as per plan)
    logger = StudentLogger(model_name, layer_to_hook=8)

    print("Loading dataset...")
    # Using same dataset/split as teacher logs to ensure alignment if we were doing strict matching
    # But here we just need to generate *some* logs. 
    # Ideally we should use the EXACT same texts.
    # For v0.1, let's just use the same dataset config.
    ds = load_dataset("code_search_net", "python", split="train[:100]", trust_remote_code=True)

    batch_texts = []
    batch_size = 8
    idx = 0

    print("Starting student log collection...")
    for item in tqdm(ds):
        text = item["func_code_string"]
        batch_texts.append(text)
        if len(batch_texts) == batch_size:
            enc, logs = logger.encode_with_logging(batch_texts)
            
            # logs is list of (B, S, D) - one per forward pass
            # We only did one forward pass
            hidden_states = logs[0] # (B, S, D)
            
            input_ids = enc["input_ids"].cpu().numpy()
            attn_mask = enc["attention_mask"].cpu().numpy()

            out_path = os.path.join(OUT_DIR, f"batch_{idx}.npz")
            np.savez_compressed(
                out_path,
                input_ids=input_ids,
                attention_mask=attn_mask,
                student_hidden=hidden_states
            )

            batch_texts = []
            idx += 1

    logger.remove_hooks()
    print("Done.")

if __name__ == "__main__":
    main()
