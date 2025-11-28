import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from uzr_garage.controller_model import UZRController
from uzr_garage.core.uzr_injector import UZRInjector

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_ID = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "artifacts", "checkpoints", "uzr_controller_final.pth")
TARGET_LAYER = 12
DEFAULT_ALPHA = 1.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== UZR-Garage v0.1 CLI Interface on {device} ===")
    print("Initializing System...")

    # 1. Load Models
    print("[1/3] Loading Student (BF16)...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("[2/3] Loading Controller...")
    controller = UZRController(max_seq_len=128).to(device)
    controller.load_state_dict(torch.load(CHECKPOINT_PATH))
    controller.eval()

    print(f"[3/3] Injecting UZR Mechanism at Layer {TARGET_LAYER}...")
    # 일단 주입해놓고 시작 (Default: Enabled)
    UZRInjector.inject(student, controller, target_layer_idx=TARGET_LAYER, alpha=DEFAULT_ALPHA)
    
    # 제어용 핸들 가져오기
    target_attn_module = student.model.layers[TARGET_LAYER].self_attn

    print("\n" + "="*50)
    print(" [COMMANDS] ")
    print(" /toggle  : Turn UZR ON/OFF")
    print(" /alpha N : Set injection strength to N (e.g., /alpha 2.5)")
    print(" /exit    : Quit")
    print("="*50 + "\n")

    while True:
        # 상태 표시
        status = "ON" if getattr(target_attn_module, "uzr_enabled", False) else "OFF"
        alpha_val = getattr(target_attn_module, "uzr_alpha", 0.0)
        
        try:
            prompt_text = input(f"\n[UZR:{status} | α={alpha_val}] User: ")
        except EOFError:
            break

        if not prompt_text: continue

        # 명령어 처리
        if prompt_text.startswith("/exit"):
            break
        elif prompt_text.startswith("/toggle"):
            # Toggle logic
            current_status = getattr(target_attn_module, "uzr_enabled", False)
            target_attn_module.uzr_enabled = not current_status
            print(f">> UZR Toggled: {target_attn_module.uzr_enabled}")
            continue
        elif prompt_text.startswith("/alpha"):
            try:
                val = float(prompt_text.split()[1])
                target_attn_module.uzr_alpha = val
                print(f">> Alpha updated to {val}")
            except:
                print(">> Usage: /alpha 1.5")
            continue

        # 생성 (Generation)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        
        print(f"Assistant ({status}): ", end="", flush=True)
        try:
            with torch.no_grad():
                student.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    streamer=streamer, 
                    do_sample=True, # 창의성 테스트
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1 # 멍청한 모델 특유의 반복 방지
                )
        except Exception as e:
            print(f"\n[Error] Generation failed: {e}")
            print("Tip: Try a shorter prompt or check if UZR is causing instability.")
        print("")

if __name__ == "__main__":
    main()
