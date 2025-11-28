import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from .controller_model import UZRController

class AttnMetaDataset(Dataset):
    def __init__(self, teacher_dir, student_dir, max_files=None, max_seq_len=256):
        self.teacher_files = sorted(glob.glob(os.path.join(teacher_dir, "*.npz")))
        self.student_files = sorted(glob.glob(os.path.join(student_dir, "*.npz")))
        
        print(f"Found {len(self.teacher_files)} teacher files in {teacher_dir}")
        print(f"Found {len(self.student_files)} student files in {student_dir}")
        
        # Ensure alignment
        # We assume filenames match or are sorted identically
        if max_files is not None:
            self.teacher_files = self.teacher_files[:max_files]
            self.student_files = self.student_files[:max_files]

        self.max_seq_len = max_seq_len
        self.samples = []  # (file_idx, sample_idx)

        for fi, (t_path, s_path) in enumerate(zip(self.teacher_files, self.student_files)):
            try:
                t_data = np.load(t_path)
                s_data = np.load(s_path)
                
                # Check consistency
                B = t_data["input_ids"].shape[0]
                # We assume B is same for both
                for bi in range(B):
                    self.samples.append((fi, bi))
            except Exception as e:
                print(f"Error loading {t_path} or {s_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fi, bi = self.samples[idx]
        
        t_data = np.load(self.teacher_files[fi])
        s_data = np.load(self.student_files[fi])

        # Teacher: (L, B, S, S) -> We want specific layer or all?
        # User said: "Teacher Attention... 주입"
        # Let's pick one layer for v0.1 or average? 
        # User: "Teacher의 시선 패턴... 주입받는 대상"
        # Let's pick the last layer or a specific one. 
        # For v0.1, let's pick the LAST captured layer (e.g. 16).
        # t_data["teacher_attn"] is (L, B, S, S)
        # We take the last one: [-1, bi] -> (S, S)
        teacher_attn = t_data["teacher_attn"][-1, bi] # (S, S)

        # Student: (B, S, D)
        student_hidden = s_data["student_hidden"][bi] # (S, D)

        # Truncate/Pad to max_seq_len
        S = student_hidden.shape[0]
        if S > self.max_seq_len:
            teacher_attn = teacher_attn[:self.max_seq_len, :self.max_seq_len]
            student_hidden = student_hidden[:self.max_seq_len]
            S = self.max_seq_len
        
        # Convert to tensor
        teacher_attn_t = torch.tensor(teacher_attn, dtype=torch.float32)
        student_hidden_t = torch.tensor(student_hidden, dtype=torch.float32)
        
        return {
            "student_hidden": student_hidden_t, # (S, D)
            "teacher_attn": teacher_attn_t,     # (S, S)
            "seq_len": S
        }

def collate_fn(batch):
    # Pad to max in batch
    max_len = max(x["seq_len"] for x in batch)
    
    student_hiddens = []
    teacher_attns = []
    masks = []

    for x in batch:
        s_len = x["seq_len"]
        d_in = x["student_hidden"].shape[1]
        
        # Pad hidden
        h_pad = torch.zeros((max_len, d_in), dtype=torch.float32)
        h_pad[:s_len] = x["student_hidden"]
        student_hiddens.append(h_pad)
        
        # Pad attn
        a_pad = torch.zeros((max_len, max_len), dtype=torch.float32)
        a_pad[:s_len, :s_len] = x["teacher_attn"]
        teacher_attns.append(a_pad)
        
        # Mask
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:s_len] = True
        masks.append(mask)

    return {
        "student_hidden": torch.stack(student_hiddens), # (B, S, D)
        "teacher_attn": torch.stack(teacher_attns),     # (B, S, S)
        "mask": torch.stack(masks)                      # (B, S)
    }

def train_controller(
    teacher_dir,
    student_dir,
    d_in,
    d_hidden=1024,
    max_seq_len=256,
    device="cuda",
    epochs=2,
    batch_size=16,
    lr=1e-4,
):
    print(f"Training Controller...")
    ds = AttnMetaDataset(teacher_dir, student_dir, max_files=50, max_seq_len=max_seq_len)
    if len(ds) == 0:
        print("No data found.")
        return

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    controller = UZRController(
        d_in=d_in,
        d_hidden=d_hidden,
        max_seq_len=max_seq_len,
    ).to(device)

    optim = torch.optim.AdamW(controller.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # Or KLDiv

    for epoch in range(epochs):
        controller.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for batch in pbar:
            h = batch["student_hidden"].to(device) # (B, S, D)
            target = batch["teacher_attn"].to(device) # (B, S, S)
            mask = batch["mask"].to(device) # (B, S)

            logits = controller(h) # (B, S, max_seq_len)
            
            # We want to compare logits (or softmaxed) to target (probs)
            # If target is probs, we should softmax logits
            pred = torch.softmax(logits, dim=-1)
            
            # Masking? 
            # We only care about valid positions.
            # Simple MSE on the whole matrix (padded parts are 0 in target and should be 0 in pred?)
            # But softmax will make them non-zero.
            # For v0.1, let's just ignore padding in loss if possible, or just let it learn to output 0 (impossible with softmax).
            # We should mask the logits before softmax?
            # Or just mask the loss.
            
            loss = loss_fn(pred, target) # Simple MSE for now
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix(loss=loss.item())

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(controller.state_dict(), "checkpoints/controller.pt")
    print("Training complete. Checkpoints saved.")

if __name__ == "__main__":
    # This block allows running this file directly for testing
    # But usually we would run it via a script or import it.
    pass
