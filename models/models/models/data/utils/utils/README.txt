# BeyondResNet-CIFAR10

A systematic study demonstrating that a custom CNN architecture can outperform ResNet-18 on CIFAR-10 classification through methodical, step-by-step enhancements.

## Results

| Model | Accuracy | Parameters | Improvement |
|-------|----------|------------|-------------|
| Baseline CNN | 88.36% | 3,251,018 | - |
| + Attention | 88.84% | 3,259,482 | +0.48% |
| + Stochastic Depth | 89.90% | 3,259,482 | +1.06% |
| ResNet-18 (baseline) | 80.55% | 11,173,962 | - |

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
