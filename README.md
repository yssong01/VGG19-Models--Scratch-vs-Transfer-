# VGG19 이진 분류: Scratch vs. Transfer Learning

## 1. 개요

본 프로젝트는 VGG19 Architecture를 활용하여 산타(Santa)와 일반인(Normal) 이미지를 분류하는 이진 분류(Binary Classification) 과제를 수행합니다.

약 900장의 소규모 데이터셋 환경에서 모델을 바닥부터 학습(Scratch)시킬 때 발생하는 과적합(Overfitting) 문제를 진단하고, 데이터 증강(Data Augmentation)을 통한 일반화 성능(Generalization Performance) 개선 과정을 정량적으로 분석합니다.


### VGG-19, Scratch Learning(직접 학습) VS Transfer Learning(전이 학습) 비교 실험 흐름의 개요 도식

![VGG model Scratch vs Transfer Learning](https://github.com/user-attachments/assets/4f64cadf-b0a8-4731-b2a7-1adf3dedddf4)

## 2. 실험 환경

- **Computing Resources**: NVIDIA A100 Tensor Core GPU
- **Dataset**: santa.zip (Binary Classes: Santa, Normal / 총 ~900 samples)
- **Model Architecture**: VGG19 (144M parameters)
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Cross-Entropy Loss

## 3. Baseline 성능 분석

### 데이터 증강 미적용 시

| 모델 유형 | 특성 | 기술적 분석 |
|------------|----------------|-------------------|
| **VGG19 (Scratch)** | High Variance / Overfitting | 144M개의 파라미터를 가진 모델 복잡도 대비 학습 데이터 부족으로 인해 Generalization Error가 크게 발생. Training Loss는 급격히 감소하나 Validation Accuracy의 편차가 크며 수렴 안정성이 낮음. |
| **VGG19 (Transfer)** | Effective Feature Extraction | ImageNet의 대규모 데이터셋에서 학습된 Pre-trained Weights를 활용하여 하위 레이어의 특징 추출 능력을 유지. 초기 에폭부터 높은 정확도를 확보하며 Inductive Bias의 이점을 극명하게 보여줌. |

## 4. 데이터 증강 전략

데이터 희소성(Data Scarcity)으로 인한 과적합 문제를 완화하고 모델의 Robustness를 확보하기 위해 실시간 데이터 증강 파이프라인을 구축했습니다.

### 구현 기법
- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `RandomRotation(15°)`
- `ColorJitter`

### 기술적 목표
- 기하학적 변형 및 색상 변이(Color Jittering)를 통해 데이터의 Invariance(불변성)를 학습.
- 모델이 특정 픽셀 위치가 아닌 핵심적인 시각적 패턴(Visual Patterns)에 집중하도록 유도.
- 매 에폭(Epoch)마다 확률적 변형을 적용하여 학습 중 경험하는 데이터의 Diversity를 극대화.

## 5. 정량적 결과

### 성능 비교 (Epoch 20)

| Metrics | Scratch (Baseline) | Scratch (Augmented) | Transfer (Augmented) |
|---------|-------------------|---------------------|---------------------|
| **Final Val Accuracy** | 91.76% | 92.51% | 93.26% |
| **Final Train Loss** | 0.125 | 0.215 | 0.129 |
| **Convergence Stability** | Low (High Volatility) | Improved | Stable |

### 주요 발견

증강 적용 후 Scratch 모델의 Training Loss가 상승(0.12 → 0.21)한 것은 모델이 단순 암기를 통한 학습 데이터 최적화를 지양하고, 변형된 이미지에 대응하기 위한 정칙화(Regularization) 효과가 발생했음을 시사합니다.

## 6. 시각적 분석

<img width="1500" height="1200" alt="Results_comparison_Epoch_20_Augmented" src="https://github.com/user-attachments/assets/00b6ace8-518e-4d6a-970d-612e352eeb07" />
<img width="1600" height="600" alt="Images_comparison_Epoch_20_Augmented" src="https://github.com/user-attachments/assets/39fa4fe5-9145-4009-a5ef-b2c23d64ff77" />


2x2 비교 플롯은 다음의 학습 역학(Learning Dynamics)을 보여줍니다:

- **Training Loss**: 증강 적용 시 Scratch 모델의 Loss 감소율이 완만해지는 현상은 Overfitting 방지와 Generalization Capacity 확보 과정을 시각화 함.
- **Validation Accuracy**: 증강 전 발생하던 성능 진폭이 억제되고, 학습 후반부까지 안정적인 Performance Gain이 관찰됨.
- **Convergence Speed**: Transfer 모델이 초기 가중치를 통해 Optimal Point에 즉각적으로 접근하는 높은 Optimization Efficiency를 입증함.
- **Weight Distribution**: Scratch 모델의 협소한 가중치 분포와 대비되는 Transfer 모델의 광범위한 분포는, 이미 학습된 필터들이 복잡하고 다양한 Spatial Features를 인식할 수 있는 표현력(Capacity)을 갖췄음을 통계적으로 입증함.

## 7. 결론

본 연구를 통해 소규모 데이터셋 환경에서 딥러닝 모델의 성능을 결정짓는 핵심 변수는 **데이터의 질적 다양성(Data Diversity)**임을 확인했습니다.

주요 내용:
- 데이터 증강은 Scratch 모델의 과적합 문제를 효과적으로 제어하여 전이 학습 모델과의 성능 격차를 최소화.
- 전이 학습 또한 증강 전략과 결합될 때 실전 데이터에 대한 Out-of-Distribution(OOD) 성능을 보장받을 수 있음을 입증.
- 증강으로 인한 정칙화 효과(Training Loss 증가)는 Validation 성능 향상과 상관관계를 보임.

## 프로젝트 구조

```
.
├── README.md
├── data/
│   └── santa.zip
├── models/
│   ├── vgg19_scratch.py
│   └── vgg19_transfer.py
├── train.py
├── utils/
│   └── augmentation.py
└── results/
    └── full_comparison_2x2_Augmented.png
```

## 요구사항

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## 사용법

```bash
# Scratch 모델 학습 (증강 미적용)
python train.py --model scratch --augmentation false

# Scratch 모델 학습 (증강 적용)
python train.py --model scratch --augmentation true

# Transfer Learning 모델 학습 (증강 적용)
python train.py --model transfer --augmentation true
```

## 인용

본 코드를 연구에 사용하실 경우 다음과 같이 인용해 주시기 바랍니다:

```bibtex
@misc{vgg19_comparison,
  title={VGG19 비교 분석: 소규모 데이터셋에서의 Scratch vs. Transfer Learning},
  author={Your Name},
  year={2025}
}
```

## 출처

이미지 : VGG - 19 Architecture

https://www.geeksforgeeks.org/computer-vision/vgg-net-architecture-explained/

## Team Project

### 경수오빠못해조
