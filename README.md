# UZR-Garage: Universal Zero-shot Reasoner (PoC)



## What is this repo?

**LLM-MLP-Injector** is a small PoC in the UZR line of experiments.

The idea is:

- Treat a 7B LLM as a **Teacher**.
- Harvest its **“attention habits”** (entropy / locality over layers & heads).
- Train a lightweight **MLP Controller** that predicts those patterns.
- At runtime, **inject** the controller’s output into a tiny **0.5B Student** via monkey-patching.
- Add a **gating MLP** that decides *when* to inject strongly vs back off, based on the Teacher’s confidence and the Student’s state.

It’s not a polished library – it’s a research playground to see:
> “How far can a small model go if we literally pour a bigger model’s attention habits into it?”






## 이 저장소는 뭐 하는 곳인가요?

**LLM-MLP-Injector**는 UZR 계열 실험 중 하나인 작은 PoC입니다.

아이디어는 단순합니다:

- 7B LLM을 **Teacher**로 쓰고,
- 그 모델의 **“시선 습관(어텐션 엔트로피 / 로컬리티 패턴)”**을 수확해서,
- 그 패턴을 예측하는 **경량 MLP Controller**를 학습시키고,
- 실시간으로 작은 **0.5B Student**에 **Monkey Patching 방식으로 주입**합니다.
- 여기에 Teacher의 확신도와 Student 상태를 보고  
  *언제, 얼마나 세게 주입할지* 결정하는 **Gating MLP**를 하나 더 얹습니다.

프로덕션용 라이브러리가 아니라, 한마디로 이런 질문을 테스트해보는 **연구 놀이터**입니다:

> “큰 모델의 ‘시선 습관’을 그대로 부어주면,  
> 작은 모델은 어디까지 따라갈 수 있을까?”


**목표:**
7B Teacher 모델의 "시선 습관(Meta-Attention)"을 추출하여 경량화된 Controller(UZR)를 통해 0.5B Student 모델에 주입하는 기술을 검증합니다. 이를 통해 작은 모델이 큰 모델의 추론 패턴을 모방하도록 합니다.

---

## 🛠️ 설치 방법 (Installation)

### 1. 사전 요구 사항
*   Python 3.8 이상
*   CUDA 지원 GPU (권장: VRAM 12GB 이상)
*   [PyTorch](https://pytorch.org/) (CUDA 버전)

### 2. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/10kseason/LLM-MLP-Injector-.git
cd LLM-MLP-Injector-

# 가상환경 생성 (선택 사항)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 의존성 설치
pip install torch transformers accelerate bitsandbytes seaborn matplotlib pandas tqdm numpy
```

---

## 🚀 빠른 시작 (Quick Start)

이미 학습된 체크포인트가 있다면, 바로 실시간 주입 데모를 실행해볼 수 있습니다.

```bash
python scripts/12_live_with_gating.py
```
*   프롬프트를 입력하면, **게이팅 모델**이 Teacher의 확신도를 예측하여 주입 강도(`Alpha`)를 조절합니다.
*   예시:
    *   `What is the capital of France?` -> **Confident** (강한 주입)
    *   `What is the capital of Mars?` -> **Cautious** (약한 주입)

---

## 🧪 전체 파이프라인 실행 가이드

UZR 시스템을 바닥부터 학습시키고 실행하는 전체 과정입니다.

### Phase 1: UZR Controller 학습 (패턴 복제)

Teacher의 어텐션 패턴을 학습하여 Student에게 전달할 Controller를 만듭니다.

1.  **Teacher 로그 수집**:
    Teacher 모델(Qwen2.5-7B)을 실행하여 어텐션 엔트로피와 로컬리티 정보를 수집합니다.
    ```bash
    python scripts/01_collect_teacher_logs.py
    ```
    *   결과: `data/teacher_logs/*.npz`

2.  **Controller 학습**:
    수집된 로그를 바탕으로 Controller(MLP)를 학습시킵니다.
    ```bash
    python scripts/03_train_controller.py
    ```
    *   결과: `artifacts/checkpoints/uzr_controller_kl_mse_final.pth`

3.  **검증 (히트맵 비교)**:
    학습된 Controller가 Teacher의 패턴을 잘 모방하는지 시각적으로 확인합니다.
    ```bash
    python scripts/09_compare_heatmaps.py
    ```
    *   결과: `uzr_comparison_heatmap.png` 생성

### Phase 2: Gating Pipeline (상황 인지형 주입)

언제 주입할지 결정하는 "게이팅 모델"을 학습시킵니다.

1.  **게이팅 데이터셋 생성**:
    Teacher의 확신도(Confident/Cautious)와 Student의 상태를 매핑한 데이터셋을 만듭니다.
    ```bash
    python scripts/10_build_gating_dataset.py
    ```
    *   결과: `artifacts/data/gating_dataset.npz`

2.  **게이팅 모델 학습**:
    데이터셋을 사용하여 게이팅 MLP를 학습시킵니다.
    ```bash
    python scripts/11_train_gating.py
    ```
    *   결과: `artifacts/checkpoints/uzr_gate.pth`

3.  **실시간 테스트**:
    최종적으로 Student + Controller + Gating을 모두 결합하여 실행합니다.
    ```bash
    python scripts/12_live_with_gating.py
    ```

---

## 📂 프로젝트 구조

```text
uzr_garage/
├── core/
│   ├── harvester.py          # Teacher/Student 패턴 수확 모듈
│   ├── pattern_logger.py     # 게이팅 학습용 특징 추출기
│   └── uzr_injector.py       # 어텐션 주입 로직 (Monkey Patching)
├── models/
│   ├── controller.py         # UZR Controller (Attention Predictor)
│   └── gating_model.py       # Gating MLP (Confidence Judge)
├── scripts/
│   ├── 01_collect_teacher_logs.py # Phase 1: 로그 수집
│   ├── 03_train_controller.py     # Phase 1: 컨트롤러 학습
│   ├── 09_compare_heatmaps.py     # 검증: 히트맵 비교
│   ├── 10_build_gating_dataset.py # Phase 2: 게이팅 데이터 생성
│   ├── 11_train_gating.py         # Phase 2: 게이팅 학습
│   └── 12_live_with_gating.py     # 데모: 실시간 주입
└── artifacts/
    ├── checkpoints/          # 학습된 모델 저장소
    └── data/                 # 데이터셋 저장소
```



