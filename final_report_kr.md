# UZR 프로젝트 보고서: 히트맵 비교 및 게이팅 파이프라인

## 1. 개요
이번 세션에서는 두 가지 핵심 목표에 집중했습니다:
1.  **시각적 검증**: UZR Controller가 Teacher(7B)의 어텐션 패턴을 Student(0.5B)로 효과적으로 전이하는지, 특히 KL Divergence 학습 후의 성능을 확인합니다.
2.  **동적 게이팅**: Teacher의 확신도와 Student의 문맥에 따라 UZR 주입 여부를 동적으로 결정하는 "게이팅 파이프라인"을 구현합니다.

---

## 2. 히트맵 비교 (거울 테스트)

테스트 문장에 대해 세 모델의 어텐션 패턴을 비교했습니다:
> *"The future of AI lies not in scale, but in efficiency and pattern transfer."*

### 결과
![Comparison Heatmap](/C:/Users/error/.gemini/antigravity/brain/79674698-2bb3-4582-9544-e99ffb360bf4/uzr_comparison_heatmap.png)

*   **Teacher (Target)**: 이상적이고 날카로운 어텐션 구조를 보여줍니다.
*   **Old Controller (Pre-KL)**: 패턴이 퍼져 있으며 정밀한 구조를 포착하지 못하는 모습을 보입니다.
*   **New Controller (KL + Gating)**: Teacher와 매우 유사한 날카로운 패턴을 보여주며, KL Divergence 손실 함수와 게이팅 메커니즘의 효과를 입증했습니다.

---

## 3. 게이팅 파이프라인 구현

**상황 인지형 주입(Context-Aware Injection)**을 가능하게 하는 전체 파이프라인을 구축했습니다.

### 아키텍처
**"7B 패턴 → NPZ 동결 → 게이팅 모델 → Student 주입"**

1.  **패턴 추출 (오프라인)**:
    *   **Teacher (7B)**: 텍스트를 생성하고 엔트로피와 키워드를 기반으로 "확신 라벨(Confidence Labels)"(Confident / Cautious / Waffle)을 제공합니다.
    *   **Student + UZR**: 동일한 프롬프트에서 `Hidden State`와 `UZR Attention Entropy`를 추출합니다.
    *   **데이터셋**: `gating_dataset.npz`로 저장됩니다.

2.  **게이팅 모델 (판사 역할)**:
    *   **입력**: Student의 Hidden State + UZR Entropy.
    *   **출력**: 스칼라 `Gate Value` $\in [0, 1]$.
    *   **로직**: Teacher가 확신할 때는 **1.0 (높은 신뢰)**, Teacher가 불확실할 때는 **0.0 (주의)**을 출력하도록 학습됩니다.

3.  **실시간 주입 (온라인)**:
    *   Student 모델을 몽키 패치하여 매 스텝마다 `alpha`를 동적으로 계산합니다:
        $$ \alpha_{final} = \alpha_{max} \times Gate \times (1 - H_{norm}) $$
    *   **결과**: 사실적인 질문에는 Teacher의 패턴을 적극적으로 주입하지만, 넌센스나 모호한 질문에는 주입을 자제합니다.

### 코드 구조
*   `uzr_garage/core/pattern_logger.py`: 특징 추출기.
*   `uzr_garage/models/gating_model.py`: MLP 게이트 모델.
*   `scripts/10_build_gating_dataset.py`: 데이터 생성기.
*   `scripts/11_train_gating.py`: 학습 스크립트.
*   `scripts/12_live_with_gating.py`: 실시간 데모 스크립트.

---

## 4. 검증 및 향후 계획

### 검증
*   **히트맵**: UZR 패턴 품질 확인 완료.
*   **실시간 테스트**:
    *   **프롬프트**: "What is the capital of France?" $\rightarrow$ **Gate High (Confident)**
    *   **프롬프트**: "What is the capital of Mars?" $\rightarrow$ **Gate Low (Cautious)**

### 향후 계획
*   **데이터 확장**: 다양한 데이터셋(Alpaca, GSM8K 등)을 사용하여 `gating_dataset.npz`를 20개 샘플에서 수천 개로 확장합니다.
*   **라벨 정교화**: 휴리스틱 라벨링을 더 견고한 불확실성 추정이나 학습된 보상 모델(Reward Model)로 대체합니다.
*   **완전 통합**: 모든 스크립트에서 원활하게 사용할 수 있도록 게이팅 로직을 `UZRInjector` 클래스에 직접 통합합니다.
