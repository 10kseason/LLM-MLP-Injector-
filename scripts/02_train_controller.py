import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uzr_garage.controller_train import train_controller

def main():
    # Configuration
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "teacher_logs")
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    EMBED_DIM = 768 # Qwen 7B might have different embed dim, but we use a separate embedding layer in controller_train.py
    # Wait, controller_train.py uses:
    # emb = torch.nn.Embedding(tokenizer.vocab_size, embed_dim)
    # So we are learning a new embedding from scratch for the controller, 
    # instead of using the teacher's embedding. This is what the user's code did.
    # User code: emb = torch.nn.Embedding(tokenizer.vocab_size, embed_dim).to(device)
    
    NUM_LAYERS = 2 # e.g. layer 8, 16
    
    train_controller(
        logs_dir=LOGS_DIR,
        model_name=MODEL_NAME,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        epochs=2,
        batch_size=8 # Small batch size for PoC
    )

if __name__ == "__main__":
    main()
