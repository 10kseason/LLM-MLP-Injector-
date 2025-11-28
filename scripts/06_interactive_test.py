import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from uzr_garage.controller_model import UZRController
from uzr_garage.core.uzr_injector import UZRInjector

# --- Config ---
STUDENT_ID = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_PATH = "./artifacts/checkpoints/uzr_controller_final.pth"
TARGET_LAYER = 12
ALPHA = 2.0  # 기본 강도

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=========================================")
    print(f"      UZR Interactive Test CLI           ")
    print(f"=========================================")
    print(f"Device: {device}")
    
    # 1. Load Models
    print("[1/2] Loading Student (Qwen 0.5B)...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)

    print("[2/2] Loading UZR Controller...")
    controller = UZRController(max_seq_len=128).to(device)
    controller.load_state_dict(torch.load(CHECKPOINT_PATH))
    controller.eval()

    print("\n[Ready] Enter your prompt below. (Type 'q' or 'exit' to quit)")
    print(f"Current Config: Layer={TARGET_LAYER}, Alpha={ALPHA}")
    
    while True:
        try:
            prompt = input("\n[User] > ")
        except EOFError:
            break
            
        if prompt.lower() in ['q', 'exit', 'quit']:
            print("Bye.")
            break
        
        if not prompt.strip():
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # --- 1. Pure Student ---
        print("\nGenerating Pure Output...", end="", flush=True)
        try:
            # Ensure clean state
            UZRInjector.eject(student, TARGET_LAYER) 
            
            with torch.no_grad():
                out_pure = student.generate(
                    **inputs, 
                    max_new_tokens=50, 
                    do_sample=True, 
                    temperature=0.7,
                    top_p=0.9
                )
            pure_text = tokenizer.decode(out_pure[0], skip_special_tokens=True)
            print(" Done.")
        except Exception as e:
            print(f" Error: {e}")
            pure_text = "Error during generation."

        # --- 2. UZR Injected ---
        print("Generating UZR Output...", end="", flush=True)
        try:
            UZRInjector.inject(student, controller, TARGET_LAYER, alpha=ALPHA)
            
            with torch.no_grad():
                out_uzr = student.generate(
                    **inputs, 
                    max_new_tokens=50, 
                    do_sample=True, 
                    temperature=0.7,
                    top_p=0.9
                )
            uzr_text = tokenizer.decode(out_uzr[0], skip_special_tokens=True)
            print(" Done.")
        except Exception as e:
            print(f" Error: {e}")
            uzr_text = "Error during generation."
            # Attempt to eject to restore state
            UZRInjector.eject(student, TARGET_LAYER)
        
        # --- Display ---
        print("-" * 60)
        print(f"[Pure Student]\n{pure_text}")
        print("-" * 60)
        print(f"[UZR Injected (Alpha={ALPHA})]\n{uzr_text}")
        print("-" * 60)

if __name__ == "__main__":
    main()
