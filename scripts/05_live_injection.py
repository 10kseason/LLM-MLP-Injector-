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
ALPHA = 2.0  # 좀 세게 줘보자. 티가 나야 하니까.

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[UZR] Live Injection Test on {device}")

    # 1. Load Models
    print("Loading Student...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)

    print("Loading Controller...")
    controller = UZRController(max_seq_len=128).to(device) # 학습때 쓴 Max Len
    controller.load_state_dict(torch.load(CHECKPOINT_PATH))
    controller.eval() # 추론 모드 (Dropout 끄기)

    # 2. Test Logic
    prompt = "The key to artificial general intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\n[Prompt] {prompt}")

    # --- Case 1: Pure Student ---
    print("\n--- [1] Student Pure Output ---")
    with torch.no_grad():
        out_pure = student.generate(**inputs, max_new_tokens=30, do_sample=False) # Greedy로 비교
    print(tokenizer.decode(out_pure[0], skip_special_tokens=True))

    # --- Case 2: UZR Injected ---
    print(f"\n--- [2] UZR Injected Output (Layer {TARGET_LAYER}, Alpha={ALPHA}) ---")
    
    # 뇌수술 집행
    UZRInjector.inject(student, controller, target_layer_idx=TARGET_LAYER, alpha=ALPHA)
    
    with torch.no_grad():
        out_uzr = student.generate(**inputs, max_new_tokens=30, do_sample=False)
    print(tokenizer.decode(out_uzr[0], skip_special_tokens=True))

    # --- Result Analysis ---
    if torch.equal(out_pure, out_uzr):
        print("\n[Result] Outputs are IDENTICAL. Injection might be too weak or ignored.")
    else:
        print("\n[Result] Outputs CHANGED! UZR is actively steering the generation.")

if __name__ == "__main__":
    main()
