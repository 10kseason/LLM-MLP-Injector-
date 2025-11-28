import torch
import numpy as np
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from uzr_garage.controller_model import UZRController
from uzr_garage.models.gating_model import UZRGate
from uzr_garage.core.pattern_logger import PatternLogger

# --- Config ---
STUDENT_ID = "Qwen/Qwen2.5-0.5B"
CONTROLLER_PATH = "./artifacts/checkpoints/uzr_controller_kl_mse_final.pth"
GATING_PATH = "./artifacts/checkpoints/uzr_gate.pth"
TARGET_LAYER = 12
MAX_SEQ_LEN = 128
ALPHA_MAX = 1.0

def live_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[UZR] Live Injection with Gating on {device}...")

    # 1. Load Models
    print("Loading Student...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    
    print("Loading Controller...")
    controller = UZRController(max_seq_len=MAX_SEQ_LEN).to(device)
    controller.load_state_dict(torch.load(CONTROLLER_PATH))
    controller.eval()
    
    print("Loading Gating Model...")
    # Need to infer dims, but hardcoding for now based on knowns
    # Qwen-0.5B hidden=896, Extra=2
    gate_model = UZRGate(hidden_dim=896, extra_dim=2).to(device)
    gate_model.load_state_dict(torch.load(GATING_PATH))
    gate_model.eval()

    # 2. Monkey Patching Student
    # We need to inject logic into the forward pass of the target layer's attention
    # But since we need 'student_hidden' which is the INPUT to the layer, 
    # and we need to run Controller + Gate BEFORE the attention calculation...
    
    # Let's wrap the layer's forward method.
    original_layer_forward = student.model.layers[TARGET_LAYER].forward
    
    def custom_forward(hidden_states, attention_mask=None, position_ids=None, **kwargs):
        # hidden_states: (B, S, H)
        
        # --- UZR Logic Start ---
        # 1. Controller Inference
        # Cast to float for Controller
        with torch.no_grad():
            ctrl_logits = controller(hidden_states.float())
            ctrl_probs = torch.softmax(ctrl_logits, dim=-1) # (B, S, S)
            
            # 2. Feature Extraction
            # We need to do this per-batch item ideally, but let's assume B=1 for live test
            feats = PatternLogger.extract_student_features(hidden_states, ctrl_probs)
            
            hidden_mean = torch.tensor(feats["hidden_mean"]).to(device).unsqueeze(0) # (1, H)
            uzr_feats = torch.tensor([feats["uzr_entropy"], feats["uzr_max_attn"]]).to(device).unsqueeze(0) # (1, 2)
            
            # 3. Gating Inference
            gate_val = gate_model(hidden_mean, uzr_feats).item() # Scalar 0-1
            
            # 4. Calculate Alpha
            # Alpha = Alpha_Max * Gate * (1 - Normalized_Entropy)
            # Normalized Entropy: 0 (peaky) to 1 (uniform)
            # Max entropy for S=128 is log(128) ~ 4.85
            max_ent = np.log(hidden_states.shape[1])
            norm_ent = feats["uzr_entropy"] / (max_ent + 1e-9)
            norm_ent = min(max(norm_ent, 0), 1)
            
            alpha = ALPHA_MAX * gate_val * (1.0 - norm_ent)
            
            # Store alpha for visualization/logging
            if not hasattr(student, "uzr_logs"):
                student.uzr_logs = []
            student.uzr_logs.append({
                "gate": gate_val,
                "entropy": feats["uzr_entropy"],
                "alpha": alpha
            })
            
        # --- Injection ---
        # We need to pass this 'alpha' and 'ctrl_probs' to the attention mechanism.
        # The easiest way is to attach it to the module temporarily
        student.model.layers[TARGET_LAYER].self_attn.uzr_alpha = alpha
        student.model.layers[TARGET_LAYER].self_attn.uzr_pattern = ctrl_probs
        
        # Call original forward
        return original_layer_forward(hidden_states, attention_mask, position_ids, **kwargs)

    # Apply Patch
    student.model.layers[TARGET_LAYER].forward = custom_forward
    
    # Patch Attention Forward to use the injected pattern
    # This requires looking at Qwen2Attention code. 
    # For now, let's assume we can just print the calculated Alpha to verify the Gating logic works.
    # Actual injection into Attention requires copying the class code or using a hook that modifies output (but we need to modify attention weights internal).
    # Given the complexity, let's focus on verifying the Gating Logic (Alpha calculation) first.
    # If the user wants full injection, I'll need to patch `Qwen2Attention.forward`.
    
    print("\n[Ready] Type a prompt to test Gating (Ctrl+C to exit)")
    
    while True:
        try:
            prompt = input("\nUser: ")
            if not prompt: continue
            
            student.uzr_logs = [] # Reset logs
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate
            with torch.no_grad():
                _ = student.generate(**inputs, max_new_tokens=10)
            
            # Report Stats from the first step of generation (processing the prompt)
            # The logs will contain entries for every forward pass (prompt processing + generation steps)
            # We look at the last log entry which corresponds to the last token generated
            if hasattr(student, "uzr_logs") and student.uzr_logs:
                last_log = student.uzr_logs[-1]
                print(f"[UZR] Gate: {last_log['gate']:.4f} | Entropy: {last_log['entropy']:.2f} | Final Alpha: {last_log['alpha']:.4f}")
                
                if last_log['gate'] > 0.8:
                    print("=> CONFIDENT: Strong Injection")
                elif last_log['gate'] < 0.3:
                    print("=> CAUTIOUS: Weak Injection")
                else:
                    print("=> BALANCED")
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    live_test()
