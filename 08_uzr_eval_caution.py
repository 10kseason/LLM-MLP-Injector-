import torch
import os
import sys
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from uzr_garage.controller_model import UZRController
from uzr_garage.core.uzr_injector import UZRInjector

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_ID = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "artifacts", "checkpoints", "uzr_controller_kl_mse_final.pth")
TARGET_LAYER = 12
ALPHAS = [0.0, 0.2, 0.5, 0.8, 1.0]

# Placeholder Prompts (User will provide full set)
FACTUAL_QUESTIONS = [
    "What is the capital of France?",
    "Who wrote 'Romeo and Juliet'?",
    "What is the boiling point of water?",
    "Name the largest planet in the solar system.",
    "What is 2 + 2?"
]

NONSENSE_QUESTIONS = [
    "What is the capital of Mars?",
    "Who is the king of the United States?",
    "What does the color blue taste like?",
    "How many corners does a circle have?",
    "When was the internet invented in 1800?"
]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== UZR Caution Benchmark on {device} ===")

    # 1. Load Models
    print("Loading Student...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    
    print("Loading Controller...")
    controller = UZRController(max_seq_len=128).to(device)
    controller.load_state_dict(torch.load(CHECKPOINT_PATH))
    controller.eval()

    # Inject (Initial)
    UZRInjector.inject(student, controller, target_layer_idx=TARGET_LAYER, alpha=0.0)
    target_attn_module = student.model.layers[TARGET_LAYER].self_attn

    results = []
    
    all_questions = [(q, "Factual") for q in FACTUAL_QUESTIONS] + \
                    [(q, "Nonsense") for q in NONSENSE_QUESTIONS]

    print(f"Starting Evaluation on {len(all_questions)} questions...")
    
    for q_idx, (question, q_type) in enumerate(all_questions):
        print(f"\n[Q{q_idx+1}] ({q_type}) {question}")
        
        for alpha in ALPHAS:
            # Set Alpha
            target_attn_module.uzr_alpha_max = alpha
            target_attn_module.uzr_enabled = (alpha > 0.0)
            
            # Reset stats
            if hasattr(target_attn_module, "uzr_last_stats"):
                target_attn_module.uzr_last_stats = {}

            # Generate
            inputs = tokenizer(question, return_tensors="pt").to(device)
            
            # Hook to capture mean alpha during generation
            alpha_means = []
            
            # We need to capture stats at each step. 
            # Since we can't easily hook into generate loop without a custom logits processor or streamer,
            # we will rely on the fact that 'uzr_last_stats' is updated at every forward pass.
            # But generate calls forward many times.
            # A simple hack: we can't easily get per-token alpha without a streamer or callback.
            # For now, let's just inspect the *last* token's alpha (end of generation) or 
            # try to capture it via a global list if we really want full history.
            # Let's just use the last state for now as a proxy, or modify injector to append to a list.
            # The injector modification I made supports `uzr_history` if initialized.
            
            target_attn_module.uzr_history = [] # Initialize history recording
            
            with torch.no_grad():
                outputs = student.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(question):].strip()
            
            # Calculate Mean Alpha from history
            if target_attn_module.uzr_history:
                # uzr_history might store tensors, let's assume I need to implement that in injector if I want full history
                # Wait, my previous edit only initialized it but didn't append. 
                # I should check if I implemented appending. 
                # Ah, I only implemented `uzr_last_stats`. 
                # Let's just use `uzr_last_stats['alpha_mean']` which is the mean of the *last forward pass*.
                # This is better than nothing, but for a full sequence average, I need to accumulate.
                # For this script, I'll just report the last token's alpha mean.
                avg_alpha = target_attn_module.uzr_last_stats.get("alpha_mean", 0.0)
            else:
                avg_alpha = target_attn_module.uzr_last_stats.get("alpha_mean", 0.0)

            # Metrics
            token_len = len(outputs[0]) - inputs.input_ids.shape[1]
            
            print(f"  [Alpha={alpha}] Len={token_len} | AlphaMean(Last)={avg_alpha:.4f} | {response[:50]}...")
            
            results.append({
                "Question": question,
                "Type": q_type,
                "Alpha": alpha,
                "Response": response,
                "Length": token_len,
                "LastAlphaMean": avg_alpha
            })

    # Save Results
    df = pd.DataFrame(results)
    save_path = os.path.join(BASE_DIR, "artifacts", "uzr_caution_benchmark.csv")
    df.to_csv(save_path, index=False)
    print(f"\nEvaluation Complete. Results saved to {save_path}")
    
    # Summary
    print("\n=== Summary (Avg Length) ===")
    print(df.groupby(["Type", "Alpha"])["Length"].mean())
    
    print("\n=== Summary (Avg Alpha Usage) ===")
    print(df.groupby(["Type", "Alpha"])["LastAlphaMean"].mean())

if __name__ == "__main__":
    main()
