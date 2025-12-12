# Quick Start Guide

This guide will help you get started with the GAN Data Augmentation system quickly.

## Installation

```bash
# Clone the repository
cd GAN-Data-Augmentation

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Examples

### 1. Train a DCGAN on CIFAR-10

```bash
python train_gan.py \
    --model dcgan \
    --dataset cifar10 \
    --epochs 100 \
    --batch_size 64 \
    --image_size 64 \
    --save_dir checkpoints/dcgan_cifar10
```

Or using a config file:

```bash
python train_gan.py --config configs/dcgan_cifar10.yaml
```

### 2. Train a Conditional GAN on MNIST

```bash
python train_gan.py \
    --model cgan \
    --dataset mnist \
    --epochs 50 \
    --num_classes 10 \
    --batch_size 128
```

### 3. Generate Samples from Trained Model

```bash
python generate_samples.py \
    --model_path checkpoints/dcgan_cifar10/final_model.pth \
    --model_type dcgan \
    --num_samples 100 \
    --output_dir generated_samples
```

### 4. Evaluate GAN Quality

```bash
python evaluate.py \
    --model_path checkpoints/dcgan_cifar10/final_model.pth \
    --model_type dcgan \
    --dataset cifar10 \
    --metrics is fid ppl pr \
    --num_samples 10000 \
    --visualize
```

### 5. Use GAN for Data Augmentation

```python
import torch
from models import DCGANGenerator
from augmentation import ImageAugmenter

# Load trained generator
generator = DCGANGenerator(latent_dim=100, output_channels=3, image_size=64)
checkpoint = torch.load('checkpoints/dcgan_cifar10/final_model.pth')
generator.load_state_dict(checkpoint['generator_state_dict'])

# Create augmenter
augmenter = ImageAugmenter(generator, latent_dim=100, device='cuda')

# Augment dataset
augmented_dataset = augmenter.augment_dataset(
    original_dataset,
    augmentation_ratio=0.5  # 50% synthetic data
)

# Use augmented dataset for training
train_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True)
```

### 6. Programmatic Usage

```python
import torch
from models import DCGANGenerator, DCGANDiscriminator
from training import GANTrainer
from utils import get_dataloader
from visualization import GANVisualizer

# Setup
device = 'cuda'
latent_dim = 100

# Create models
generator = DCGANGenerator(latent_dim=100, output_channels=3, image_size=64)
discriminator = DCGANDiscriminator(input_channels=3, image_size=64)

# Create trainer
trainer = GANTrainer(
    generator, discriminator,
    g_lr=0.0002, d_lr=0.0002,
    device=device
)

# Load data
dataloader = get_dataloader('cifar10', batch_size=64)

# Train
for epoch in range(1, 101):
    losses = trainer.train_epoch(dataloader, epoch, latent_dim=latent_dim)
    print(f"Epoch {epoch}: G Loss = {losses['g_loss']:.4f}, D Loss = {losses['d_loss']:.4f}")

    # Generate samples every 10 epochs
    if epoch % 10 == 0:
        samples = trainer.generate(num_samples=64, latent_dim=latent_dim)
        visualizer = GANVisualizer()
        visualizer.plot_generated_samples(samples, save_path=f'samples_epoch_{epoch}.png')

# Save model
trainer.save_checkpoint('final_model.pth', epoch=100)
```

## Training Tips

### 1. **Choose the Right Architecture**
- **DCGAN**: Good baseline, works well on simple datasets
- **WGAN**: More stable training, better for complex datasets
- **Conditional GAN**: When you need class-specific generation
- **StyleGAN**: Best quality, requires more compute

### 2. **Hyperparameter Tuning**
- Start with learning rate 0.0002 for both G and D
- Use Adam optimizer with beta1=0.5, beta2=0.999
- Batch size: 64-128 for DCGAN, 16-32 for StyleGAN
- Train discriminator and generator at similar rates

### 3. **Monitor Training**
- Watch generator and discriminator losses
- Generate samples regularly to check quality
- Use gradient norm monitoring to detect issues
- Look for signs of mode collapse

### 4. **Common Issues**
- **Mode collapse**: Try WGAN or increase diversity loss
- **Training instability**: Reduce learning rates, use spectral normalization
- **Poor quality**: Increase model capacity, train longer
- **Slow training**: Use mixed precision, reduce image size

## Evaluation Metrics

- **Inception Score (IS)**: Measures quality and diversity (higher is better)
- **FID**: Measures similarity to real distribution (lower is better)
- **Precision/Recall**: Measures quality vs coverage
- **PPL**: Measures latent space smoothness (lower is better)

## Dataset Preparation

### Custom Dataset
Organize images in a directory:
```
data/
└── custom/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Then train:
```bash
python train_gan.py \
    --model dcgan \
    --dataset custom \
    --data_path data/custom \
    --image_size 64
```

### Conditional Dataset
Organize images by class:
```
data/
└── custom_conditional/
    ├── class_0/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── class_1/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── ...
```

## Next Steps

- Check out `notebooks/` for Jupyter notebook examples
- Read the full documentation in `README.md`
- Explore advanced features in the code
- Experiment with different architectures and datasets

## Getting Help

- Check the documentation
- Look at example configs in `configs/`
- Review the code examples
- Open an issue on GitHub for bugs or questions
