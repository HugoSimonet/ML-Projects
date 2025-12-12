# Medical Image Analysis

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Deep learning for medical image classification and segmentation with uncertainty quantification and explainability.

## Overview

This project implements computer vision models for analyzing medical images across multiple modalities (X-ray, MRI, CT, pathology). It includes pre-trained CNNs, Vision Transformers, U-Net for segmentation, uncertainty quantification, and explainability tools.

## Features

- Multi-modal medical imaging support (X-ray, MRI, CT, ultrasound, pathology)
- Classification and segmentation tasks
- Uncertainty quantification (MC Dropout, ensembles, Bayesian methods)
- Explainability (Grad-CAM, SHAP, attention visualization)
- DICOM image processing
- Transfer learning from ImageNet and RadImageNet

## Models

**Classification** - ResNet, DenseNet, EfficientNet, Vision Transformer
**Segmentation** - U-Net, U-Net++, Attention U-Net
**Uncertainty** - MC Dropout, Deep Ensembles, Bayesian CNN

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 1.9+, torchvision, pydicom, SimpleITK, scikit-image

## Quick Start

```bash
# Train chest X-ray classifier
python train.py \
    --task classification \
    --modality xray \
    --dataset chestxray14 \
    --model resnet50 \
    --pretrained \
    --epochs 50

# Segment brain MRI
python train.py \
    --task segmentation \
    --modality mri \
    --dataset brats \
    --model unet \
    --epochs 100

# Evaluate with uncertainty
python evaluate.py \
    --model_path checkpoints/best_model.pth \
    --uncertainty mc_dropout \
    --n_samples 20
```

## Usage

```python
from models import MedicalClassifier
from utils import load_dicom

# Load model
model = MedicalClassifier(
    backbone='resnet50',
    num_classes=14,
    pretrained=True
)

# Load and preprocess image
image = load_dicom('patient_xray.dcm')

# Predict with uncertainty
predictions, uncertainty = model.predict_with_uncertainty(image, n_samples=20)
```

## Explainability

```python
from explainability import GradCAM

grad_cam = GradCAM(model, target_layer='layer4')
heatmap = grad_cam.generate(image, target_class=3)
grad_cam.visualize(image, heatmap, save_path='explanation.png')
```

## Datasets

- **ChestX-ray14**: 14 thoracic diseases
- **CheXpert**: 14 observations, uncertainty labels
- **BraTS**: Brain tumor segmentation
- **ISIC**: Skin lesion classification
- **PathMNIST**: Pathology image classification
- **Custom DICOM**: Load from DICOM directories

## Metrics

**Classification**: Accuracy, AUC-ROC, sensitivity, specificity, F1-score
**Segmentation**: Dice coefficient, IoU, Hausdorff distance
**Uncertainty**: Expected calibration error, Brier score

## Project Structure

```
Medical-CV-Diagnosis/
├── models/              # Model architectures
├── preprocessing/       # DICOM processing, augmentation
├── training/            # Training loops
├── evaluation/          # Metrics and uncertainty
├── explainability/      # Grad-CAM, SHAP
├── utils/               # Data loading, helpers
├── configs/             # Configuration files
└── train.py             # Main training script
```

## Implementation Notes

Uses PyTorch with torchvision models. DICOM images loaded via pydicom and converted to arrays. Windowing applied for CT/MRI visualization. Augmentation includes rotation, translation, elastic deformation appropriate for medical images.

Uncertainty via MC Dropout (forward passes with dropout enabled) or ensembles (multiple models). Calibration measured using reliability diagrams.

## References

- He et al. "Deep Residual Learning for Image Recognition"
- Huang et al. "Densely Connected Convolutional Networks"
- Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Gal & Ghahramani "Dropout as a Bayesian Approximation"
- Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks"

## License

MIT License - see LICENSE file for details.
