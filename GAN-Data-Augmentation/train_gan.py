"""
GAN Training Script
Main script for training various GAN architectures
"""

import argparse
import os
import torch
import yaml
from tqdm import tqdm

from models import (
    DCGANGenerator, DCGANDiscriminator,
    StyleGANGenerator, SpectralNormDiscriminator,
    ConditionalGenerator, ConditionalDiscriminator,
    ProgressiveGenerator, ProgressiveDiscriminator
)
from training import GANTrainer, WGANTrainer, ConditionalGANTrainer, ProgressiveGANTrainer
from utils import get_dataloader
from visualization import GANVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train GAN')
    parser.add_argument('--model', type=str, default='dcgan',
                       choices=['dcgan', 'wgan', 'stylegan', 'cgan', 'progan'],
                       help='GAN architecture to train')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist', 'celeba', 'custom'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image size')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels')

    # Optimization
    parser.add_argument('--g_lr', type=float, default=0.0002,
                       help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=0.0002,
                       help='Discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Adam beta2')

    # Conditional GAN
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes for conditional GAN')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--sample_interval', type=int, default=5,
                       help='Generate samples every N epochs')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    return parser.parse_args()


def create_model(args):
    """Create generator and discriminator based on model type"""

    if args.model == 'dcgan':
        generator = DCGANGenerator(
            latent_dim=args.latent_dim,
            output_channels=3,
            base_channels=args.base_channels,
            image_size=args.image_size
        )
        discriminator = DCGANDiscriminator(
            input_channels=3,
            base_channels=args.base_channels,
            image_size=args.image_size
        )

    elif args.model == 'wgan':
        generator = DCGANGenerator(
            latent_dim=args.latent_dim,
            output_channels=3,
            base_channels=args.base_channels,
            image_size=args.image_size
        )
        discriminator = SpectralNormDiscriminator(
            input_channels=3,
            base_channels=args.base_channels,
            image_size=args.image_size,
            use_attention=False
        )

    elif args.model == 'stylegan':
        generator = StyleGANGenerator(
            latent_dim=512,
            output_channels=3,
            base_channels=args.base_channels,
            image_size=args.image_size
        )
        discriminator = SpectralNormDiscriminator(
            input_channels=3,
            base_channels=args.base_channels,
            image_size=args.image_size,
            use_attention=True
        )
        args.latent_dim = 512

    elif args.model == 'cgan':
        generator = ConditionalGenerator(
            latent_dim=args.latent_dim,
            num_classes=args.num_classes,
            output_channels=3,
            base_channels=args.base_channels,
            image_size=args.image_size
        )
        discriminator = ConditionalDiscriminator(
            input_channels=3,
            num_classes=args.num_classes,
            base_channels=args.base_channels,
            image_size=args.image_size
        )

    elif args.model == 'progan':
        generator = ProgressiveGenerator(
            latent_dim=512,
            output_channels=3,
            base_channels=args.base_channels,
            max_resolution=args.image_size
        )
        discriminator = ProgressiveDiscriminator(
            input_channels=3,
            base_channels=args.base_channels,
            max_resolution=args.image_size
        )
        args.latent_dim = 512

    return generator, discriminator


def create_trainer(args, generator, discriminator):
    """Create trainer based on model type"""

    if args.model == 'wgan':
        trainer = WGANTrainer(
            generator, discriminator,
            g_lr=args.g_lr,
            d_lr=args.d_lr,
            beta1=args.beta1,
            beta2=args.beta2,
            device=args.device
        )

    elif args.model == 'cgan':
        trainer = ConditionalGANTrainer(
            generator, discriminator,
            num_classes=args.num_classes,
            g_lr=args.g_lr,
            d_lr=args.d_lr,
            beta1=args.beta1,
            beta2=args.beta2,
            device=args.device
        )

    elif args.model == 'progan':
        trainer = ProgressiveGANTrainer(
            generator, discriminator,
            g_lr=args.g_lr,
            d_lr=args.d_lr,
            beta1=args.beta1,
            beta2=args.beta2,
            device=args.device
        )

    else:
        trainer = GANTrainer(
            generator, discriminator,
            g_lr=args.g_lr,
            d_lr=args.d_lr,
            beta1=args.beta1,
            beta2=args.beta2,
            device=args.device
        )

    return trainer


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'samples'), exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create dataloader
    print(f'Loading {args.dataset} dataset...')
    dataloader = get_dataloader(
        args.dataset,
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f'Dataset size: {len(dataloader.dataset)}')

    # Create models
    print(f'Creating {args.model} model...')
    generator, discriminator = create_model(args)

    # Print model info
    total_params_g = sum(p.numel() for p in generator.parameters())
    total_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f'Generator parameters: {total_params_g:,}')
    print(f'Discriminator parameters: {total_params_d:,}')

    # Create trainer
    trainer = create_trainer(args, generator, discriminator)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f'Resumed from epoch {start_epoch}')

    # Create visualizer
    visualizer = GANVisualizer(save_dir=os.path.join(args.save_dir, 'visualizations'))

    # Training loop
    print('Starting training...')
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train epoch
        losses = trainer.train_epoch(dataloader, epoch, latent_dim=args.latent_dim)

        print(f"Epoch {epoch}/{args.epochs} - G Loss: {losses['g_loss']:.4f}, D Loss: {losses['d_loss']:.4f}")

        # Generate samples
        if epoch % args.sample_interval == 0:
            print('Generating samples...')
            samples = trainer.generate(num_samples=64, latent_dim=args.latent_dim)
            visualizer.plot_generated_samples(
                samples,
                save_path=os.path.join(args.save_dir, 'samples', f'epoch_{epoch:04d}.png')
            )

        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch:04d}.pth')
            trainer.save_checkpoint(checkpoint_path, epoch)
            print(f'Saved checkpoint: {checkpoint_path}')

    # Final checkpoint
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    trainer.save_checkpoint(final_path, args.epochs)
    print(f'Training complete! Final model saved to {final_path}')

    # Plot training curves
    print('Plotting training curves...')
    visualizer.plot_training_curves(
        trainer.g_losses,
        trainer.d_losses,
        save_path=os.path.join(args.save_dir, 'training_curves.png')
    )

    print('Done!')


if __name__ == '__main__':
    main()
