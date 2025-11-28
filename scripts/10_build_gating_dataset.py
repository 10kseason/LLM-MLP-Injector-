import torch
import numpy as np
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from uzr_garage.controller_model import UZRController
from uzr_garage.core.pattern_logger import PatternLogger

# --- Config ---
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_ID = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_PATH = "./artifacts/checkpoints/uzr_controller_kl_mse_final.pth"
OUTPUT_PATH = "./artifacts/data/gating_dataset.npz"
MAX_SEQ_LEN = 128
TARGET_LAYER = 12

# Simple Prompt Set (Mix of Factual/Confident and Nonsense/Cautious)
PROMPTS = [
    # Factual (Should be Confident)
    "What is the capital of France?",
    "Who wrote 'Romeo and Juliet'?",
    "What is the boiling point of water?",
    "Name the largest planet in the solar system.",
    "What is 2 + 2?",
    "Who is the current president of the USA?",
    "What is the chemical symbol for Gold?",
    "How many continents are there?",
    "What is the speed of light?",
    "Who painted the Mona Lisa?",
    
    # Nonsense / Ambiguous (Should be Cautious or Waffle)
    "What is the capital of Mars?",
    "Who is the king of the United States?",
    "What does the color blue taste like?",
    "How many corners does a circle have?",
    "When was the internet invented in 1800?",
    "Can you eat a cloud?",
    "What is the meaning of life divided by zero?",
    "Who is the strongest Pokemon in real life?",
    "Why is the sky green at night?",
    "Do androids dream of electric sheep?"
]

def build_dataset():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[UZR] Building Gating Dataset on {device}...")

    # 1. Load Models
    print("Loading Teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    t_tok = AutoTokenizer.from_pretrained(TEACHER_ID)

    print("Loading Student...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    s_tok = AutoTokenizer.from_pretrained(STUDENT_ID)

    print("Loading Controller...")
    controller = UZRController(max_seq_len=MAX_SEQ_LEN).to(device)
    controller.load_state_dict(torch.load(CHECKPOINT_PATH))
    controller.eval()

    # Data Containers
    data_hidden = []
    data_uzr_feats = []
    data_labels = []
    data_prompts = []

    print(f"Processing {len(PROMPTS)} prompts...")
    
    for prompt in tqdm(PROMPTS):
        # --- Teacher Step ---
        t_inputs = t_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            t_out = teacher.generate(
                **t_inputs, 
                max_new_tokens=50, 
                return_dict_in_generate=True, 
                output_scores=True
            )
        
        # Extract Teacher Features (Logits -> Entropy -> Label)
        # t_out.scores is a tuple of (B, V) for each step
        # Stack them: (S_gen, B, V) -> (B, S_gen, V)
        generated_logits = torch.stack(t_out.scores, dim=1) 
        generated_ids = t_out.sequences[:, t_inputs.input_ids.shape[1]:]
        
        teacher_feats = PatternLogger.extract_teacher_features(
            generated_logits, generated_ids, t_tok
        )
        label = teacher_feats["label"]
        
        # --- Student + UZR Step ---
        s_inputs = s_tok(prompt, return_tensors="pt").to(device)
        
        # Hook for Student Hidden
        hidden_cache = {}
        def get_hook(name):
            def hook(model, input, output):
                hidden_cache[name] = output[0].detach()
            return hook
        
        handle = student.model.layers[TARGET_LAYER].register_forward_hook(get_hook("student"))
        
        with torch.no_grad():
            student(**s_inputs)
            student_hidden = hidden_cache["student"].to(device) # (1, S, H)
            
            # Controller Inference
            # Controller expects float32 input
            uzr_logits = controller(student_hidden.float())
            uzr_probs = torch.softmax(uzr_logits, dim=-1) # (1, S, S)
            
        handle.remove()
        
        # Extract Student/UZR Features
        student_feats = PatternLogger.extract_student_features(student_hidden, uzr_probs)
        
        # --- Collect Data ---
        data_hidden.append(student_feats["hidden_mean"]) # (H,)
        data_uzr_feats.append([student_feats["uzr_entropy"], student_feats["uzr_max_attn"]]) # (2,)
        data_labels.append(label)
        data_prompts.append(prompt)
        
        print(f"  Prompt: {prompt[:30]}... | Label: {label} | UZR Ent: {student_feats['uzr_entropy']:.2f}")

    # Save to NPZ
    print(f"Saving dataset to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        hidden=np.array(data_hidden),
        uzr_feats=np.array(data_uzr_feats),
        labels=np.array(data_labels),
        prompts=np.array(data_prompts)
    )
    print("Done.")

if __name__ == "__main__":
    build_dataset()
