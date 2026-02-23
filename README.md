# Fine-Grained Bird Species Classification: ResNet18 vs ViT on CUB-200-2011

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CUB--200--2011-green)](https://www.vision.caltech.edu/datasets/cub_200_2011/)

This project compares two popular approaches for **fine-grained image classification** on the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset:  
- Pretrained **ResNet-18** (with partial fine-tuning)  
- Pretrained **Vision Transformer (ViT-base)** from Hugging Face  

The goal is to classify 200 bird species using transfer learning, evaluate performance, and visualize training dynamics & confusion patterns.

## Dataset

- **Name**: Caltech-UCSD Birds-200-2011 (CUB-200-2011)  
- **Source**: Hugging Face → `bentrevett/caltech-ucsd-birds-200-2011`  
- **Classes**: 200 bird species  
- **Total Images**: ≈11,788  
- **Splits**: ~85% train (further split into train/val), ~15% test  

Images are resized to 224×224 and normalized using ImageNet statistics.

## Models & Approach

- **ResNet-18** (torchvision): Freeze early layers, fine-tune layer4 + fc head  
- **ViT-base-patch16-224-in21k** (google/vit-base-patch16-224-in21k): Full fine-tuning with small learning rate  
- **Training**: PyTorch + AMP (mixed precision), AdamW optimizer, CrossEntropyLoss  
- **Hyperparameters** (example):  
  - Batch size: 32  
  - Epochs: 20–30 (with early stopping)  
  - LR: ResNet 1e-3, ViT 5e-5 (or lower)  
  - Scheduler: ReduceLROnPlateau  
- **Augmentation**: RandomResizedCrop, HorizontalFlip, Rotation, Normalize  
