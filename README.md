# vit-from-scratch

# CNN vs Vision Transformer on CIFAR-10

This repository compares Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) on the CIFAR-10 dataset using a fully implemented training pipeline in PyTorch.

---

## 1. Objective

To empirically analyze the differences in inductive bias between CNNs and Vision Transformers by evaluating:

- Convergence speed
- Final classification accuracy
- Training stability

---

## 2. Models

- Simple CNN with two convolutional layers
- Vision Transformer (ViT) with patch embedding and multi-head self-attention

---

## 3. Dataset

- CIFAR-10 (10-class image classification dataset)

---

## 4. Experimental Results

### Accuracy
- CNN: ~0.68–0.70
- ViT: ~0.58–0.61

### Observations

- CNN converges faster and achieves higher accuracy on small-scale datasets.
- ViT shows slower convergence but continues improving over epochs.

---

## 5. Key Findings

- CNN benefits from strong inductive bias (locality and translation invariance)
- ViT requires larger data scale for optimal performance
- On CIFAR-10, CNN is more data-efficient

---

## 6. Results Visualization

### CNN
- `results/CNN_loss.png`
- `results/CNN_acc.png`

### ViT
- `results/ViT_loss.png`
- `results/ViT_acc.png`

---

## 7. Future Work

- Scaling ViT to larger datasets (e.g., ImageNet)
- Introducing data augmentation and regularization
- Extending to Vision-Language Models (e.g., CLIP-style architectures)

---

## 8. Requirements

```bash
pip install -r requirements.txt
