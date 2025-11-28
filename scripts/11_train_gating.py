import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from uzr_garage.models.gating_model import UZRGate

# --- Config ---
DATASET_PATH = "./artifacts/data/gating_dataset.npz"
CHECKPOINT_PATH = "./artifacts/checkpoints/uzr_gate.pth"
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-3

class GatingDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.hidden = torch.from_numpy(data['hidden']).float()
        self.uzr_feats = torch.from_numpy(data['uzr_feats']).float()
        self.labels = torch.from_numpy(data['labels']).long()
        
        # Normalize labels for Sigmoid training (0 or 1)
        # 0=Confident -> 1.0 (Trust UZR? No, wait.)
        # Logic:
        # If Teacher is CONFIDENT (0) -> We want UZR to be active? 
        # Wait, UZR is "Pattern Transfer". If Teacher is Confident, it means the pattern is strong.
        # But "Caution" means we should be careful.
        # Let's align with the user's "Alpha" logic:
        # alpha_final = alpha_ent * gate
        # If gate is high, UZR is active.
        # When do we want UZR active?
        # Ideally, when the Teacher *would have been* confident.
        # So if Label=0 (Confident), Target Gate = 1.0
        # If Label=1 (Cautious) or 2 (Waffle), Target Gate = 0.0 (or low)
        
        self.targets = (self.labels == 0).float().unsqueeze(1) # (B, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.hidden[idx], self.uzr_feats[idx], self.targets[idx]

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[UZR] Training Gating Model on {device}...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Run build_gating_dataset.py first.")
        return

    dataset = GatingDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    # Hidden dim from dataset (896 for Qwen-0.5B), Extra dim = 2
    model = UZRGate(hidden_dim=dataset.hidden.shape[1], extra_dim=dataset.uzr_feats.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss() # Binary Cross Entropy for Sigmoid output
    
    print(f"Dataset Size: {len(dataset)}")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for hidden, feats, target in dataloader:
            hidden, feats, target = hidden.to(device), feats.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(hidden, feats)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
            
    # Save
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Model saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()
