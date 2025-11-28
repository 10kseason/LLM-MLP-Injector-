import torch
import os
import sys
import pandas as pd
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

QUESTIONS = [
    "Why is the sky blue?",
    "What is the capital of France?",
    "Explain quantum entanglement in simple terms.",
    "Write a short poem about a robot.",
    "Who wrote 'Romeo and Juliet'?",
    "What is the boiling point of water?",
    "How do you make a cup of tea?",
    "What is the largest planet in our solar system?",
    "Define 'artificial intelligence'.",
    "What is 2 + 2?",
    "Why do birds fly south for the winter?",
    "What is the speed of light?",
    "Name three primary colors.",
    "What is the powerhouse of the cell?",
    "Who painted the Mona Lisa?",
    "What is the currency of Japan?",
    "How many continents are there?",
    "What is the chemical symbol for gold?",
    "Who was the first person on the moon?",
    "What is the meaning of life?"
]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== UZR Evaluation Script on {device} ===")

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

    print(f"Starting Evaluation on {len(QUESTIONS)} questions...")
    
    for q_idx, question in enumerate(QUESTIONS):
        print(f"\n[Q{q_idx+1}] {question}")
        
        for alpha in ALPHAS:
            # Set Alpha
            target_attn_module.uzr_alpha = alpha
            target_attn_module.uzr_enabled = (alpha > 0.0)
            
            # Generate
            inputs = tokenizer(question, return_tensors="pt").to(device)
            
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
            
            # Metrics
            token_len = len(outputs[0]) - inputs.input_ids.shape[1]
            
            print(f"  [Alpha={alpha}] Len={token_len} | {response[:50]}...")
            
            results.append({
                "Question": question,
                "Alpha": alpha,
                "Response": response,
                "Length": token_len
            })

    # Save Results
    df = pd.DataFrame(results)
    save_path = os.path.join(BASE_DIR, "artifacts", "uzr_eval_results.csv")
    df.to_csv(save_path, index=False)
    print(f"\nEvaluation Complete. Results saved to {save_path}")
    
    # Summary
    print("\n=== Summary ===")
    summary = df.groupby("Alpha")["Length"].mean()
    print(summary)

if __name__ == "__main__":
    main()
