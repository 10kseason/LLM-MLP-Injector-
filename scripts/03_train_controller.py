import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 프로젝트 루트 경로
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from uzr_garage.controller_model import UZRController

# --- Config ---
DATA_DIR = "./data/harvested_logs"
SAVE_DIR = "./artifacts/checkpoints"

MAX_SEQ_LEN = 128
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3

# Loss 가중치
LAMBDA_KL = 1.0     # 분포 모양(Entropy) 맞추기
LAMBDA_MSE = 0.1    # 숫자 스케일도 살짝 맞춰주기

class UZRDataset(Dataset):
    def __init__(self, data_dir, max_len):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.max_len = max_len
        print(f"[Dataset] Found {len(self.files)} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        s_hidden = data["student_hidden"]  # (S, 896)
        t_attn   = data["teacher_attn"]    # (S, S)  이미 softmax된 prob에 가까움

        curr_len = min(s_hidden.shape[0], self.max_len)
        s_hidden = s_hidden[:curr_len]
        t_attn   = t_attn[:curr_len, :curr_len]

        return s_hidden, t_attn, curr_len


def collate_fn(batch):
    batch_s = []
    batch_t = []
    masks   = []

    for s, t, L in batch:
        # Hidden padding
        pad_s = np.zeros((MAX_SEQ_LEN, s.shape[1]), dtype=np.float32)
        pad_s[:L, :] = s
        batch_s.append(pad_s)

        # Attention padding
        pad_t = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=np.float32)
        pad_t[:L, :L] = t
        batch_t.append(pad_t)

        # Mask
        m = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=np.float32)
        m[:L, :L] = 1.0
        masks.append(m)

    return (
        torch.tensor(np.array(batch_s)),   # (B, S, D)
        torch.tensor(np.array(batch_t)),   # (B, S, S)
        torch.tensor(np.array(masks)),     # (B, S, S)
    )


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Training on {device}")

    dataset = UZRDataset(DATA_DIR, MAX_SEQ_LEN)
    if len(dataset) == 0:
        print("[Error] No data found. Run harvester.py first!")
        return

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = UZRController(max_seq_len=MAX_SEQ_LEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        steps = 0

        for batch_s, batch_t, mask in loader:
            batch_s = batch_s.to(device)  # (B, S, D)
            batch_t = batch_t.to(device)  # (B, S, S)
            mask    = mask.to(device)     # (B, S, S)

            optimizer.zero_grad()

            # 1) Controller 예측 logits
            pred_logits = model(batch_s)          # (B, S, Max_S)
            pred_logits = pred_logits[:, :MAX_SEQ_LEN, :MAX_SEQ_LEN]  # 안전빵

            # 2) Teacher 분포 / Controller 분포
            # Teacher는 이미 probs 비슷하지만, 한 번 더 정규화해서 분포로 고정
            teacher_probs = batch_t
            teacher_probs = teacher_probs.clamp_min(0.0)
            teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            ctrl_probs = F.softmax(pred_logits, dim=-1)  # (B, S, S)

            # 3) KL(T || C)  (분포 모양 공명)
            # kl = sum_i p_i * (log p_i - log q_i)
            kl_map = teacher_probs * (
                teacher_probs.clamp_min(1e-8).log()
                - ctrl_probs.clamp_min(1e-8).log()
            )
            kl_map = kl_map.sum(dim=-1)   # (B, S)

            # mask 적용
            token_mask = mask[..., 0]     # (B, S) 대각 기준
            kl_loss = (kl_map * token_mask).sum() / token_mask.sum().clamp_min(1e-8)

            # 4) MSE(probs) 약하게 추가 (스케일 보정용)
            mse_map = (ctrl_probs - teacher_probs) ** 2
            mse_map = (mse_map * mask).sum() / mask.sum().clamp_min(1e-8)

            loss = LAMBDA_KL * kl_loss + LAMBDA_MSE * mse_map
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            path = os.path.join(SAVE_DIR, f"uzr_ctrl_e{epoch+1}.pth")
            torch.save(model.state_dict(), path)
            print(f"  > Saved checkpoint: {path}")

    final_path = os.path.join(SAVE_DIR, "uzr_controller_kl_mse_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[Done] Training Finished. Saved: {final_path}")


if __name__ == "__main__":
    train()
