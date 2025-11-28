import os
import sys
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uzr_garage.attn_logging import AttentionLogger

OUT_DIR = "data/test_logs"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    # Use small model for testing logic
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading test model: {model_name}")
    logger = AttentionLogger(model_name, layers_to_hook=(8, 16))

    print("Loading dataset...")
    ds = load_dataset("code_search_net", "python", split="train[:10]", trust_remote_code=True)

    batch_texts = []
    batch_size = 2
    idx = 0

    print("Starting test log collection...")
    for item in tqdm(ds):
        text = item["func_code_string"]
        batch_texts.append(text)
        if len(batch_texts) == batch_size:
            enc, logs = logger.encode_with_logging(batch_texts)
            print(f"Logs captured: {len(logs)}")
            
            if len(logs) > 0:
                print(f"Sample log keys: {logs[0].keys()}")
                if "attn_mean" in logs[0]:
                    print(f"Attn mean shape: {logs[0]['attn_mean'].shape}")
                
                # Save to file
                out_path = os.path.join(OUT_DIR, f"batch_{idx}.npz")
                
                # Mock input_ids/attn_mask since we didn't capture them in return of encode_with_logging
                # Wait, encode_with_logging returns (enc, logs)
                input_ids = enc["input_ids"].cpu().numpy()
                attn_mask = enc["attention_mask"].cpu().numpy()
                
                # Prepare teacher_attn format
                # logs is list of dicts.
                # We need (L, B, S, S)
                # In test script we hooked 8 and 16.
                layers = sorted(set(l["layer"] for l in logs))
                layer_attn = {l: [] for l in layers}
                for l in layers:
                    attns = []
                    for record in logs:
                        if record["layer"] == l:
                            attns.append(record["attn_mean"])
                    if attns:
                        layer_attn[l].append(attns[0])
                
                attn_list = [layer_attn[l][0] for l in sorted(layers)]
                attn_arr = np.stack(attn_list, axis=0) # (L, B, S, S)

                np.savez_compressed(
                    out_path,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    layer_ids=np.array(sorted(layers)),
                    teacher_attn=attn_arr,
                )
                print(f"Saved to {out_path}")

            batch_texts = []
            idx += 1
            break # Run only one batch

    logger.remove_hooks()
    print("Test Done.")

if __name__ == "__main__":
    main()
