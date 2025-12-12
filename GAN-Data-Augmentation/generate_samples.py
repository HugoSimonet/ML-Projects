"""
Sample Generation Script
Generate synthetic samples from trained GAN
"""

import argparse
import os
import torch
import torchvision.utils as vutils

from models import (
    DCGANGenerator, StyleGANGenerator,
    ConditionalGenerator, ProgressiveGenerator
)
from utils import sample_latent, sample_truncated


def parse_args():
    parser = argparse.ArgumentParser(description='Generate samples from trained GAN')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='dcgan',
                       choices=['dcgan', 'stylegan', 'cgan', 'progan'],
                       help='Type of GAN model')

    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image size')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes (for conditional GAN)')

    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--truncation', type=float, default=None,
                       help='Truncation for sampling (None for no truncation)')

    # Conditional generation
    parser.add_argument('--class_label', type=int, default=None,
                       help='Class label for conditional generation')

    # Output
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                       help='Directory to save samples')
    parser.add_argument('--save_format', type=str, default='grid',
                       choices=['grid', 'individual'],
                       help='How to save samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    return parser.parse_args()


def load_generator(args):
    """Load generator from checkpoint"""

    if args.model_type == 'dcgan':
        generator = DCGANGenerator(
            latent_dim=args.latent_dim,
            output_channels=3,
            base_channels=args.base_channels,
            image_size=args.image_size
        )

    elif args.model_type == 'stylegan':
        generator = StyleGANGenerator(
            latent_dim=512,
            output_channels=3,
            base_channels=args.base_channels,
            image_size=args.image_size
        )
        args.latent_dim = 512

    elif args.model_type == 'cgan':
        generator = ConditionalGenerator(
            latent_dim=args.latent_dim,
            num_classes=args.num_classes,
            output_channels=3,
            base_channels=args.base_channels,
            image_size=args.image_size
        )

    elif args.model_type == 'progan':
        generator = ProgressiveGenerator(
            latent_dim=512,
            output_channels=3,
            base_channels=args.base_channels,
            max_resolution=args.image_size
        )
        args.latent_dim = 512

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=args.device)

    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)

    generator.to(args.device)
    generator.eval()

    return generator


def generate_samples(generator, args):
    """Generate samples"""
    samples = []

    print(f'Generating {args.num_samples} samples...')

    with torch.no_grad():
        for i in range(0, args.num_samples, args.batch_size):
            current_batch = min(args.batch_size, args.num_samples - i)

            # Sample latent vectors
            if args.truncation is not None:
                z = sample_truncated(current_batch, args.latent_dim, args.truncation, args.device)
            else:
                z = sample_latent(current_batch, args.latent_dim, args.device)

            # Generate
            if args.model_type == 'cgan':
                if args.class_label is not None:
                    labels = torch.full((current_batch,), args.class_label,
                                      dtype=torch.long, device=args.device)
                else:
                    labels = torch.randint(0, args.num_classes, (current_batch,),
                                         device=args.device)
                fake = generator(z, labels)
            else:
                fake = generator(z)

            samples.append(fake.cpu())

            print(f'Generated {min(i + args.batch_size, args.num_samples)}/{args.num_samples}')

    samples = torch.cat(samples, dim=0)

    # Denormalize
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    return samples


def save_samples(samples, args):
    """Save generated samples"""
    os.makedirs(args.output_dir, exist_ok=True)

    if args.save_format == 'grid':
        # Save as grid
        nrow = int(args.num_samples ** 0.5)
        grid = vutils.make_grid(samples, nrow=nrow, padding=2)
        output_path = os.path.join(args.output_dir, 'generated_grid.png')
        vutils.save_image(grid, output_path)
        print(f'Saved grid to {output_path}')

    else:
        # Save individual images
        for i, sample in enumerate(samples):
            output_path = os.path.join(args.output_dir, f'sample_{i:05d}.png')
            vutils.save_image(sample, output_path)

        print(f'Saved {len(samples)} images to {args.output_dir}')


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load generator
    print(f'Loading generator from {args.model_path}...')
    generator = load_generator(args)
    print('Generator loaded successfully')

    # Generate samples
    samples = generate_samples(generator, args)

    # Save samples
    save_samples(samples, args)

    print('Done!')


if __name__ == '__main__':
    main()
