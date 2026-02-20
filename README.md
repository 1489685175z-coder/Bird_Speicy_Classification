# Fine-Grained Bird Species Classification on CUB-200-2011

A comparative study of transfer learning vs. from-scratch training for fine-grained visual classification using the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset.

**Pre-trained ResNet-18** significantly outperforms a **Simple CNN baseline** trained from scratch, demonstrating the critical role of ImageNet pre-training in data-limited fine-grained tasks.

## Project Overview

- **Task**: Classify images into one of 200 bird species  
- **Dataset**: CUB-200-2011 (11,788 images, 200 classes)  
- **Models Compared**:
  - ResNet-18 (pre-trained on ImageNet + fine-tuning)
  - SimpleCNN (custom 3-conv-layer model trained from scratch)
- **Framework**: PyTorch + torchvision
- **Environment**: Google Colab (GPU: T4/P100)

## Features & Implementation Highlights

- Data loading from Hugging Face (`bentrevett/caltech-ucsd-birds-200-2011`)
- Standard augmentation: RandomResizedCrop, HorizontalFlip, Rotation(15°)
- ResNet-18 fine-tuning: Freeze early layers, only train layer4 + classifier
- Baseline SimpleCNN: 3 conv layers + dropout
- Training pipeline: Adam + ReduceLROnPlateau scheduler
- Evaluation: Accuracy curves, confusion matrix, top-5 confused pairs, visualization of misclassified samples (with bird species names)

## How to Run

1. Open the notebook in Google Colab
2. Install dependencies (if needed):

```bash
!pip install datasets tqdm torch torchvision scikit-learn matplotlib seaborn
