# UZR-Garage: Universal Zero-shot Reasoner (PoC)

**Goal:**
To validate the concept of transferring "Meta-Attention" patterns from a 7B Teacher model to a 0.5B Student model via a lightweight Controller (UZR). This allows the smaller model to mimic the reasoning patterns of the larger model.

---

## 🛠️ Installation

### 1. Prerequisites
*   Python 3.8+
*   CUDA-enabled GPU (Recommended: VRAM 12GB+)
*   [PyTorch](https://pytorch.org/) (CUDA version)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/your-username/uzr-garage.git
cd uzr-garage

# Create Virtual Environment (Optional)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install Dependencies
pip install torch transformers accelerate bitsandbytes seaborn matplotlib pandas tqdm numpy
```

---

## 🚀 Quick Start

If you have the trained checkpoints, you can run the live injection demo immediately.

```bash
python scripts/12_live_with_gating.py
```
*   Enter a prompt, and the **Gating Model** will predict the Teacher's confidence and adjust the injection strength (`Alpha`) accordingly.
*   Examples:
    *   `What is the capital of France?` -> **Confident** (Strong Injection)
    *   `What is the capital of Mars?` -> **Cautious** (Weak Injection)

---

## 🧪 Full Pipeline Guide

Step-by-step guide to training and running the UZR system from scratch.

### Phase 1: UZR Controller Training (Pattern Cloning)

Train the Controller to mimic the Teacher's attention patterns.

1.  **Collect Teacher Logs**:
    Run the Teacher model (Qwen2.5-7B) to harvest attention entropy and locality information.
    ```bash
    python scripts/01_collect_teacher_logs.py
    ```
    *   Output: `data/teacher_logs/*.npz`

2.  **Train Controller**:
    Train the Controller (MLP) using the collected logs.
    ```bash
    python scripts/03_train_controller.py
    ```
    *   Output: `artifacts/checkpoints/uzr_controller_kl_mse_final.pth`

3.  **Verify (Heatmap Comparison)**:
    Visually confirm that the Controller mimics the Teacher's patterns.
    ```bash
    python scripts/09_compare_heatmaps.py
    ```
    *   Output: `uzr_comparison_heatmap.png`

### Phase 2: Gating Pipeline (Context-Aware Injection)

Train the "Gating Model" to decide *when* to inject.

1.  **Build Gating Dataset**:
    Create a dataset mapping Teacher's confidence (Confident/Cautious) to Student's state.
    ```bash
    python scripts/10_build_gating_dataset.py
    ```
    *   Output: `artifacts/data/gating_dataset.npz`

2.  **Train Gating Model**:
    Train the Gating MLP using the dataset.
    ```bash
    python scripts/11_train_gating.py
    ```
    *   Output: `artifacts/checkpoints/uzr_gate.pth`

3.  **Live Test**:
    Run the full system combining Student + Controller + Gating.
    ```bash
    python scripts/12_live_with_gating.py
    ```

---

## 📂 Project Structure

```text
uzr_garage/
├── core/
│   ├── harvester.py          # Teacher/Student pattern harvester
│   ├── pattern_logger.py     # Feature extractor for gating
│   └── uzr_injector.py       # Injection logic (Monkey Patching)
├── models/
│   ├── controller.py         # UZR Controller (Attention Predictor)
│   └── gating_model.py       # Gating MLP (Confidence Judge)
├── scripts/
│   ├── 01_collect_teacher_logs.py # Phase 1: Log Collection
│   ├── 03_train_controller.py     # Phase 1: Controller Training
│   ├── 09_compare_heatmaps.py     # Verify: Heatmap Comparison
│   ├── 10_build_gating_dataset.py # Phase 2: Build Gating Dataset
│   ├── 11_train_gating.py         # Phase 2: Train Gating Model
│   └── 12_live_with_gating.py     # Demo: Live Injection
└── artifacts/
    ├── checkpoints/          # Model checkpoints
    └── data/                 # Datasets
```
