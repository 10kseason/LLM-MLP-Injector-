import os
import sys
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uzr_garage.attn_logging import AttentionLogger

OUT_DIR = "data/teacher_logs"
# Ensure output directory exists relative to the script execution or project root
# Assuming script is run from project root or we use absolute paths
# Let's make it relative to this script for safety if run from scripts dir, 
# but usually we run from root. Let's use absolute path based on project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "teacher_logs")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # Note: User might need to login to HF or have the model cached.
    # If Qwen2.5-7B-Instruct is not available, user can change it.
    
    print(f"Loading model: {model_name}")
    logger = AttentionLogger(model_name, layers_to_hook=(8, 16))

    # 예: 코드 + 일반 텍스트 섞기 (dataset은 상황에 맞게)
    # Using a small subset for PoC
    print("Loading dataset...")
    ds = load_dataset("code_search_net", "python", split="train[:100]", trust_remote_code=True)

    batch_texts = []
    batch_size = 8
    idx = 0

    print(f"Dataset size: {len(ds)}")
    print("Starting log collection...")
    for i, item in enumerate(tqdm(ds)):
        text = item["func_code_string"]  # code_search_net uses func_code_string
        batch_texts.append(text)
        if len(batch_texts) == batch_size:
            print(f"Processing batch {idx}...")
            enc, logs = logger.encode_with_logging(batch_texts)
            print(f"Logs captured: {len(logs)}")

            input_ids = enc["input_ids"].cpu().numpy()
            attn_mask = enc["attention_mask"].cpu().numpy()

            # 레이어별로 모아서 저장
            out_path = os.path.join(OUT_DIR, f"batch_{idx}.npz")

            # logs: list of dicts (layer, attn_mean(B, S, S))
            layers = sorted(set(l["layer"] for l in logs))
            layer_attn = {l: [] for l in layers}

            for l in layers:
                attns = []
                for record in logs:
                    if record["layer"] == l:
                        attns.append(record["attn_mean"])   # (B, S, S)
                
                if attns:
                    # If multiple forward passes (unlikely here), stack/concat
                    # Here we assume one forward pass per batch
                    # attns is list of (B, S, S) - usually just one element if batch size matches
                    # But wait, logs are appended per layer per forward pass.
                    # If we did one forward pass with a batch of texts, we get one record per layer.
                    # So attns has 1 element: (B, S, S)
                    layer_attn[l].append(attns[0])

            # Stack layers: (L, B, S, S)
            # layer_attn[l] is list of (B, S, S) arrays.
            # We want to stack them.
            
            # First, ensure we have the data
            # layer_attn[l][0] is (B, S, S)
            attn_list = [layer_attn[l][0] for l in sorted(layers)]
            attn_arr = np.stack(attn_list, axis=0) # (L, B, S, S)

            np.savez_compressed(
                out_path,
                input_ids=input_ids,
                attention_mask=attn_mask,
                layer_ids=np.array(sorted(layers)),
                teacher_attn=attn_arr, # (L, B, S, S)
            )

            batch_texts = []
            idx += 1

    logger.remove_hooks()
    print("Done.")

if __name__ == "__main__":
    main()
