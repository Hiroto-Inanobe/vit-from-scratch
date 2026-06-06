# vit-from-scratch  
## CNN vs Vision Transformer (ViT) on CIFAR-10: Inductive Bias Analysis in PyTorch

This repository implements and analyzes Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) from scratch in PyTorch, focusing on how architectural inductive biases affect learning dynamics, data efficiency, and representation formation.

Unlike a pure benchmark project, this work emphasizes **why performance differences arise**, not just accuracy comparison.

---

## 1. Objective

The goal of this project is to empirically study how model architecture influences:

- Convergence speed  
- Data efficiency  
- Training stability  
- Representation learning behavior  

in a low-data regime (CIFAR-10).

The central hypothesis is that **inductive bias plays a critical role in sample-efficient learning**, especially when data scale is limited.

---

## 2. Models

### CNN
- 2-layer convolutional network
- ReLU activation
- Max pooling
- Fully connected classifier

### Vision Transformer (ViT)
- Patch embedding
- Positional encoding
- Multi-head self-attention
- MLP head

Both models are implemented from scratch in PyTorch to ensure controlled comparison of architectural differences.

---

## 3. Dataset

- CIFAR-10 (10-class image classification)

CIFAR-10 is intentionally chosen as a small-scale dataset to highlight differences in inductive bias rather than maximize ViT performance.

---

## 4. Training Setup

- Optimizer: Adam  
- Loss: CrossEntropyLoss  
- Batch size: fixed across models  
- Epochs: identical training schedule for fair comparison  
- No architecture-specific hyperparameter tuning

---

## 5. Results

### Classification Accuracy

- CNN: 0.68 – 0.70  
- ViT: 0.58 – 0.61  

### Learning Behavior

- CNN converges faster and stabilizes early
- ViT shows slower but more gradual improvement

---

## 6. Analysis

### 6.1 Role of Inductive Bias

CNN demonstrates stronger performance in low-data regimes due to:

- Local receptive fields
- Translation invariance
- Hierarchical feature extraction

These biases reduce sample complexity and stabilize optimization.

---

### 6.2 Data Dependence of ViT

ViT lacks built-in locality bias and must learn spatial structure from data.

Observed behavior:
- Slower convergence in early training
- Higher variance in optimization
- Improved performance trend with longer training

This suggests ViT requires larger-scale data to fully realize its representational capacity.

---

### 6.3 Representation Learning Dynamics

Empirical observations:

- CNN: learns localized feature hierarchies early in training  
- ViT: gradually develops global dependency structures via attention  

This indicates fundamentally different representation formation pathways driven by architecture.

---

## 7. Limitations

- CIFAR-10 is insufficient for evaluating ViT at scale
- No exhaustive hyperparameter search was performed
- No formal representation analysis (e.g., attention rollout, embedding similarity metrics)
- Training scale is limited for large-model conclusions

---

## 8. Future Work

This project serves as a foundation for deeper architectural analysis.

Future directions include:

- Scaling experiments to ImageNet-level datasets  
- Representation analysis:
  - Attention visualization
  - Embedding space geometry
- Investigating CNN–ViT hybrid architectures  
- Studying scaling laws of architecture vs data size  
- Extending to Vision-Language Models (e.g., CLIP-style architectures)

---

## 9. Requirements

```bash
pip install -r requirements.txt
