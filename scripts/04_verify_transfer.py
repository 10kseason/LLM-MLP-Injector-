import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from uzr_garage.controller_model import UZRController

# --- Config ---
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_ID = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_PATH = "./artifacts/checkpoints/uzr_controller_kl_mse_final.pth"
MAX_SEQ_LEN = 128
TARGET_LAYER = 12  # 학습 때 썼던 그 레이어

def verify():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[UZR] Mirror Test initializing on {device}...")

    # 1. Load Models
    # Teacher (BF16 to match training logic)
    print("[1/4] Loading Teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    t_tok = AutoTokenizer.from_pretrained(TEACHER_ID)

    # Student (BF16)
    print("[2/4] Loading Student...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    s_tok = AutoTokenizer.from_pretrained(STUDENT_ID)

    # Controller
    print("[3/4] Loading Controller...")
    controller = UZRController(max_seq_len=MAX_SEQ_LEN).to(device)
    controller.load_state_dict(torch.load(CHECKPOINT_PATH))
    controller.eval()

    # 2. Test Input
    # 학습 데이터에 없었던 새로운 문장
    text = "The future of AI lies not in scale, but in efficiency and pattern transfer."
    print(f"\n[Test Input] {text}")

    # 3. Harvest Real Data
    t_inputs = t_tok(text, return_tensors="pt").to(device)
    s_inputs = s_tok(text, return_tensors="pt").to(device)
    
    seq_len = min(t_inputs.input_ids.shape[1], MAX_SEQ_LEN)
    
    # Hook for Student Hidden
    hidden_cache = {}
    def get_hook(name):
        def hook(model, input, output):
            hidden_cache[name] = output[0].detach()
        return hook
    
    student.model.layers[TARGET_LAYER].register_forward_hook(get_hook("student"))

    with torch.no_grad():
        # Teacher Forward -> Get Real Attn
        t_out = teacher(**t_inputs, output_attentions=True)
        # Last Layer, Avg Heads
        real_attn = t_out.attentions[-1][0].mean(dim=0).float().cpu().numpy() # (S, S)
        
        # Student Forward -> Get Hidden
        student(**s_inputs)
        student_hidden = hidden_cache["student"].to(device) # (1, S, Hidden)

    # 4. Controller Prediction
    with torch.no_grad():
        # (1, S, Hidden) -> (1, S, Max_Seq) -> (S, S) slicing
        # Controller trained in FP32, input is BF16. Cast to float.
        pred_logits = controller(student_hidden.float()) 
        
        # Apply Softmax to get Probabilities (matching Teacher's scale)
        pred_probs = torch.softmax(pred_logits, dim=-1)
        
        pred_attn = pred_probs[0, :seq_len, :seq_len].float().cpu().numpy()

    # 5. Visualization
    print("[4/4] Generating Heatmaps...")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(real_attn[:seq_len, :seq_len], ax=axes[0], cmap="viridis")
    axes[0].set_title("Teacher's Real Attention (Target)")
    
    sns.heatmap(pred_attn, ax=axes[1], cmap="viridis")
    axes[1].set_title("Controller's Prediction (UZR)")
    
    plt.tight_layout()
    save_path = "uzr_mirror_test.png"
    plt.savefig(save_path)
    print(f"\n[Done] Result saved to {save_path}")
    print("Check the image. If patterns match, UZR is functional.")

if __name__ == "__main__":
    verify()
