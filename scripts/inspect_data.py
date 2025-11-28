import glob
import numpy as np
import os

DATA_DIR = "./data/harvested_logs"
files = glob.glob(os.path.join(DATA_DIR, "*.npz"))

print(f"Checking {len(files)} files...")

for f in files[:10]: # Check first 10
    data = np.load(f)
    s_hidden = data['student_hidden']
    t_attn = data['teacher_attn']
    
    if np.isnan(s_hidden).any() or np.isinf(s_hidden).any():
        print(f"[FAIL] {f} has NaN/Inf in Hidden")
    
    if np.isnan(t_attn).any() or np.isinf(t_attn).any():
        print(f"[FAIL] {f} has NaN/Inf in Attn")
        
    print(f"{f}: Hidden {s_hidden.shape}, Attn {t_attn.shape}, Max Attn: {t_attn.max()}, Min Attn: {t_attn.min()}")
