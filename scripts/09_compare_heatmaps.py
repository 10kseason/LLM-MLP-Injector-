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
CHECKPOINT_OLD = "./artifacts/checkpoints/uzr_controller_final.pth"
CHECKPOINT_NEW = "./artifacts/checkpoints/uzr_controller_kl_mse_final.pth"
MAX_SEQ_LEN = 128
TARGET_LAYER = 12

def compare_heatmaps():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[UZR] Heatmap Comparison initializing on {device}...")

    # 1. Load Models
    print("[1/5] Loading Teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    t_tok = AutoTokenizer.from_pretrained(TEACHER_ID)

    print("[2/5] Loading Student...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    s_tok = AutoTokenizer.from_pretrained(STUDENT_ID)

    print("[3/5] Loading Controllers...")
    # Old Controller (Pre-KL)
    ctrl_old = UZRController(max_seq_len=MAX_SEQ_LEN).to(device)
    ctrl_old.load_state_dict(torch.load(CHECKPOINT_OLD))
    ctrl_old.eval()

    # New Controller (KL+Gating)
    ctrl_new = UZRController(max_seq_len=MAX_SEQ_LEN).to(device)
    ctrl_new.load_state_dict(torch.load(CHECKPOINT_NEW))
    ctrl_new.eval()

    # 2. Test Input
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
        real_attn = t_out.attentions[-1][0].mean(dim=0).float().cpu().numpy() # (S, S)
        
        # Student Forward -> Get Hidden
        student(**s_inputs)
        student_hidden = hidden_cache["student"].to(device) # (1, S, Hidden)

    # 4. Controller Predictions
    print("[4/5] Running Controllers...")
    with torch.no_grad():
        # Old Controller
        logits_old = ctrl_old(student_hidden.float())
        probs_old = torch.softmax(logits_old, dim=-1)
        attn_old = probs_old[0, :seq_len, :seq_len].float().cpu().numpy()

        # New Controller
        logits_new = ctrl_new(student_hidden.float())
        probs_new = torch.softmax(logits_new, dim=-1)
        attn_new = probs_new[0, :seq_len, :seq_len].float().cpu().numpy()

    # 5. Visualization
    print("[5/5] Generating Comparison Heatmaps...")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # Teacher
    sns.heatmap(real_attn[:seq_len, :seq_len], ax=axes[0], cmap="viridis")
    axes[0].set_title("Teacher (Target)")
    
    # Old Controller
    sns.heatmap(attn_old, ax=axes[1], cmap="viridis")
    axes[1].set_title("Old Controller (Pre-KL)")

    # New Controller
    sns.heatmap(attn_new, ax=axes[2], cmap="viridis")
    axes[2].set_title("New Controller (KL + Gating)")
    
    plt.tight_layout()
    save_path = "uzr_comparison_heatmap.png"
    plt.savefig(save_path)
    print(f"\n[Done] Result saved to {save_path}")

if __name__ == "__main__":
    compare_heatmaps()
