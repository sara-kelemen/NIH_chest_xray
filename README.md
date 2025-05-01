# CNN: NIH Chest XRays
---
## Project Overview
This project evaluates convolutional neural networks (CNNs)  to classify chest X-ray images as healthy or diseased based on the NIH ChestXray14 dataset. I focused on evaluating model performance using accuracy, precision, recall, F1-score, confusion matrices, and more.

Two models were explored:

- A custom ChestXrayCNN (simple CNN architecture)

- A transfer learning model based on ResNet18
---
## Dataset
- 112,120 frontal-view chest X-rays from >30,000 unique patients, 1024×1024
- Source: https://www.kaggle.com/datasets/nih-chest-xrays/data
- Resized to 224×224 pixels for faster training and efficiency 
- 14 diseases and 'no findings' as labels; generalized into no finding of disease vs diseased binary 
- Paper citation:
  
  Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers.
  ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly- Supervised
  Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471,2017
---
## CNN using Pytorch
Created ChestXrayCNN: 
- 2 × Convolutional layers (3×3 kernels, padding=1)
- MaxPooling (2×2) after each convolution
- Fully Connected Layer: 128 units → Output Layer
- ReLU activations
- Output: Single logit for binary classification
- 5 epochs
- 128 batch size
- CUDA device;  NVIDIA RTX 3060
- Adam optimizer
- Binary Cross-Entropy with Logits (BCEWithLogitsLoss) for loss function
---
## Resnet18 model
- Pretrained ResNet18 on ImageNet
- Final fully connected layer replaced for 2-class output
- Fine-tuned on Chest X-ray data
- 3 epochs, but same otherwise
---

