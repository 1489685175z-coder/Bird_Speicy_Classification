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
