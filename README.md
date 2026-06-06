# vit-from-scratch

A simple PyTorch implementation of Vision Transformer (ViT).

## Motivation

I am interested in Computer Vision and Vision-Language Models (VLMs).

This project is for understanding:

- How Vision Transformer works
- Difference between CNN and Transformer
- Self-Attention in vision tasks

## What is ViT?

Vision Transformer splits an image into patches and treats them as tokens.

Then it applies Transformer Encoder layers to learn relationships between patches.

## Architecture

Image → Patch Embedding → CLS Token + Position Embedding → Transformer Encoder → MLP Head → Output

## Self-Attention

Attention(Q,K,V) = softmax(QK^T / √d)V

## Why ViT?

- CNN: local feature extraction
- ViT: global attention between all patches

ViT is more flexible but requires more data.

## Future Work

- Train on CIFAR-10
- Compare with CNN
- Explore CLIP and VLMs
