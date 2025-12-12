# GAN Data Augmentation

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Generate high-quality synthetic training data using Generative Adversarial Networks to improve model performance and address data scarcity.

## Overview

This project implements multiple GAN architectures for data augmentation across different domains. It provides training frameworks, quality evaluation metrics, and integration tools for augmenting datasets with synthetic samples.

## Features

- Multiple GAN architectures (DCGAN, StyleGAN, CycleGAN, Conditional GAN)
- Progressive growing for high-resolution generation
- Comprehensive quality metrics (FID, IS, PPL, Precision/Recall)
- Flexible augmentation strategies (online, adaptive, mixup)
- Domain-specific implementations for images, text, and tabular data
- Visualization tools for latent space exploration
- Production-ready augmentation pipeline

## Architecture

### Generator Networks

**DCGAN** - Deep convolutional architecture with transposed convolutions and batch normalization for stable training.

**StyleGAN** - Advanced generator with style modulation layers and mapping network for fine-grained control.

**Conditional GAN** - Label-conditioned generation using embedding layers for class-specific synthesis.

**Progressive GAN** - Gradual resolution increase during training for stable high-resolution generation.

**CycleGAN** - Unpaired domain transfer using cycle consistency loss.

### Discriminator Networks

**PatchGAN** - Multi-scale discrimination for local realism assessment.

**Spectral Normalization** - Lipschitz constraint enforcement for training stability.

**Self-Attention** - Global consistency through attention mechanisms.

**Multi-Scale** - Operates at multiple resolutions for better quality assessment.

### Training Framework

**Loss Functions**: Vanilla GAN, LSGAN, Wasserstein (WGAN-GP), Hinge, perceptual loss, feature matching, cycle consistency.

**Regularization**: R1 gradient penalty, path length regularization, spectral normalization.

**Optimization**: Adam/RMSprop with beta tuning, separate learning rates for G/D, exponential moving average.

## Supported Data Types

- **Images**: Natural images (CIFAR-10, ImageNet), medical scans, satellite imagery
- **Text**: Sequence generation with embedding layers
- **Time Series**: 1D temporal data augmentation
- **Tabular Data**: Mixed data type synthesis

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

Requirements: Python 3.8+, PyTorch 1.9+, torchvision, numpy, pillow, scipy

## Quick Start

### Train DCGAN on CIFAR-10

```bash
python train_gan.py \
    --model dcgan \
    --dataset cifar10 \
    --epochs 100 \
    --batch_size 64 \
    --image_size 64 \
    --save_dir checkpoints/dcgan_cifar10
```

### Train Conditional GAN on MNIST

```bash
python train_gan.py \
    --model cgan \
    --dataset mnist \
    --epochs 50 \
    --num_classes 10 \
    --batch_size 128
```

### Generate Samples

```bash
python generate_samples.py \
    --model_path checkpoints/dcgan_cifar10/final_model.pth \
    --model_type dcgan \
    --num_samples 100 \
    --output_dir generated_samples
```

### Evaluate Quality

```bash
python evaluate.py \
    --model_path checkpoints/dcgan_cifar10/final_model.pth \
    --model_type dcgan \
    --dataset cifar10 \
    --metrics fid is ppl pr
```

## Programmatic Usage

### Basic Generation

```python
import torch
from models import DCGANGenerator

generator = DCGANGenerator(latent_dim=100, output_channels=3, image_size=64)
checkpoint = torch.load('checkpoints/final_model.pth')
generator.load_state_dict(checkpoint['generator_state_dict'])

# Generate samples
z = torch.randn(16, 100, 1, 1)
fake_images = generator(z)
```

### Data Augmentation

```python
from augmentation import ImageAugmenter
from torch.utils.data import DataLoader

# Load trained generator
generator = load_generator('checkpoints/final_model.pth')

# Create augmenter
augmenter = ImageAugmenter(generator, latent_dim=100, device='cuda')

# Augment dataset (50% synthetic)
augmented_dataset = augmenter.augment_dataset(
    original_dataset,
    augmentation_ratio=0.5
)

train_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True)
```

### Custom Training

```python
from models import DCGANGenerator, DCGANDiscriminator
from training import WGANTrainer

generator = DCGANGenerator(latent_dim=100, output_channels=3, image_size=64)
discriminator = DCGANDiscriminator(input_channels=3, image_size=64)

trainer = WGANTrainer(
    generator=generator,
    discriminator=discriminator,
    latent_dim=100,
    device='cuda',
    lr_g=0.0001,
    lr_d=0.0004
)

trainer.train(train_loader, epochs=100, save_dir='checkpoints')
```

## Configuration

YAML configuration files are available in `configs/`:

```yaml
model:
  type: dcgan
  latent_dim: 100
  image_size: 64
  base_channels: 64

training:
  epochs: 100
  batch_size: 64
  lr_g: 0.0002
  lr_d: 0.0002
  beta1: 0.5
  beta2: 0.999

data:
  dataset: cifar10
  num_workers: 4
  augment: true

evaluation:
  metrics: [fid, is, ppl]
  eval_frequency: 5
  num_samples: 10000
```

## Quality Metrics

**Inception Score (IS)** - Measures quality and diversity. Higher is better (typical range: 1-10).

**Fréchet Inception Distance (FID)** - Distribution similarity between real and generated. Lower is better (typical range: 0-300).

**Perceptual Path Length (PPL)** - Latent space smoothness. Lower indicates better disentanglement.

**Precision/Recall** - Precision measures quality (no outliers), recall measures coverage (mode diversity).

**Kernel Inception Distance (KID)** - Unbiased alternative to FID for smaller sample sizes.

## Augmentation Strategies

**Online Augmentation** - Generate samples on-the-fly during training for memory efficiency.

**Adaptive Augmentation** - Adjust augmentation ratio based on validation performance.

**Mixup Augmentation** - Blend real and synthetic samples for smooth distribution.

**Progressive Augmentation** - Gradually increase synthetic data ratio during training.

**Quality Filtering** - Filter generated samples based on discriminator confidence or perceptual quality.

## Visualization

```python
from visualization import GANVisualizer

visualizer = GANVisualizer(generator, latent_dim=100, device='cuda')

# Latent space interpolation
visualizer.interpolate_samples(num_frames=10, save_path='interpolation.gif')

# 2D latent space grid
visualizer.visualize_latent_grid(grid_size=8, save_path='latent_grid.png')

# Training progress
visualizer.plot_training_curves(log_file='training.log', save_path='curves.png')
```

## Project Structure

```
GAN-Data-Augmentation/
├── models/              # Generator and discriminator architectures
├── training/            # Training frameworks and loss functions
├── evaluation/          # Quality metrics (FID, IS, PPL, etc.)
├── augmentation/        # Augmentation strategies
├── utils/               # Sampling, data loading, helpers
├── visualization/       # Plotting and latent space exploration
├── configs/             # YAML configuration templates
├── train_gan.py         # Main training script
├── generate_samples.py  # Sample generation script
└── evaluate.py          # Evaluation script
```

## Implementation Notes

Models use PyTorch with optional CUDA acceleration. Training uses mixed precision (AMP) when available for performance. Progressive growing gradually increases resolution from 4x4 to target size.

Quality metrics require pre-computed statistics for real datasets. FID uses Inception-v3 features, IS uses class predictions. Large batch sizes (64-128) improve metric reliability.

Generated samples are clipped to valid range and converted to uint8 for saving. Latent space uses standard normal distribution (mean=0, std=1).

## Testing

```bash
# Run test suite
pytest tests/

# Quick verification
python test_installation.py
```

See `TEST_RESULTS.md` for validation results.

## References

- Goodfellow et al. "Generative Adversarial Networks" (2014)
- Radford et al. "Unsupervised Representation Learning with Deep Convolutional GANs" (DCGAN)
- Karras et al. "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
- Karras et al. "A Style-Based Generator Architecture for GANs" (StyleGAN)
- Arjovsky et al. "Wasserstein GAN"
- Gulrajani et al. "Improved Training of Wasserstein GANs" (WGAN-GP)
- Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (FID metric)

## License

MIT License - see LICENSE file for details.
