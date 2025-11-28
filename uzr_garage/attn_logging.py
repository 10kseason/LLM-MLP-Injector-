import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class AttentionLogger:
    def __init__(self, model_name, layers_to_hook=(8, 16), device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16,
            # load_in_4bit=True, # Windows compatibility issue with bitsandbytes/triton
            attn_implementation="eager"  # hook 위해 필요
        )
        self.layers_to_hook = set(layers_to_hook)
        self.attn_logs = []
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self, layer_idx):
        def fn(module, input, output):
            # output: (hidden_states, attn_weights) 형태일 가능성 높음
            attn_weights = output[1]  # (B, H, N, N)
            with torch.no_grad():
                attn = attn_weights  # (B,H,N,N)
                # 안전장치
                attn = attn.clamp(min=1e-9)

                # v0.1: Capture Mean Attention (B, S, S)
                # attn: (B, H, S, S)
                attn_mean = attn.mean(dim=1)  # (B, S, S)

                self.attn_logs.append({
                    "layer": layer_idx,
                    "attn_mean": attn_mean.detach().cpu().numpy().astype(np.float16), # Save space
                })
        return fn

    def _register_hooks(self):
        # Qwen/Mistral 구조: model.model.layers[i].self_attn
        for i, layer in enumerate(self.model.model.layers):
            if i in self.layers_to_hook:
                h = layer.self_attn.register_forward_hook(self._hook_fn(i))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def encode_with_logging(self, texts):
        # texts: list[str]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.model.device)

        self.attn_logs.clear()
        with torch.no_grad():
            _ = self.model(**enc, output_attentions=True)
        return enc, self.attn_logs
