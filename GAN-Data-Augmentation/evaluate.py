"""
GAN Evaluation Script
Evaluate trained GANs using various quality metrics
"""

import argparse
import os
import torch
import yaml
from tqdm import tqdm

from models import (
    DCGANGenerator, StyleGANGenerator,
    ConditionalGenerator, ProgressiveGenerator
)
from evaluation import (
    InceptionScore, FrechetInceptionDistance,
    PerceptualPathLength, PrecisionRecall, KernelInceptionDistance
)
from utils import get_dataloader, sample_latent
from visualization import GANVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GAN')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='dcgan',
                       choices=['dcgan', 'stylegan', 'cgan', 'progan'],
                       help='Type of GAN model')

    # Dataset for comparison
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset for comparison')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to dataset')

    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image size')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes (for conditional GAN)')

    # Evaluation parameters
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['is', 'fid'],
                       choices=['is', 'fid', 'ppl', 'pr', 'kid'],
                       help='Metrics to compute')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')

    # Output
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--num_viz_samples', type=int, default=64,
                       help='Number of samples for visualization')

    return parser.parse_args()


def load_model(args):
    """Load trained model from checkpoint"""

    # Create model
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

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device)

    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)

    generator.to(args.device)
    generator.eval()

    return generator


def generate_samples(generator, num_samples, latent_dim, batch_size, device):
    """Generate synthetic samples"""
    samples = []

    print('Generating samples...')
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size)):
            current_batch = min(batch_size, num_samples - i)
            z = sample_latent(current_batch, latent_dim, device)
            fake = generator(z)
            samples.append(fake.cpu())

    return torch.cat(samples, dim=0)


def load_real_samples(dataloader, num_samples):
    """Load real samples from dataset"""
    real_samples = []

    print('Loading real samples...')
    for batch in tqdm(dataloader):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        real_samples.append(images)

        if len(torch.cat(real_samples, dim=0)) >= num_samples:
            break

    real_samples = torch.cat(real_samples, dim=0)[:num_samples]
    return real_samples


def evaluate_metrics(args, generator, real_samples, fake_samples):
    """Evaluate selected metrics"""
    results = {}

    # Inception Score
    if 'is' in args.metrics:
        print('Computing Inception Score...')
        is_metric = InceptionScore(device=args.device)
        is_mean, is_std = is_metric.calculate(fake_samples)
        results['inception_score'] = {'mean': is_mean, 'std': is_std}
        print(f'Inception Score: {is_mean:.4f} ± {is_std:.4f}')

    # Fréchet Inception Distance
    if 'fid' in args.metrics:
        print('Computing FID...')
        fid_metric = FrechetInceptionDistance(device=args.device)
        fid_score = fid_metric.calculate(real_samples, fake_samples)
        results['fid'] = fid_score
        print(f'FID: {fid_score:.4f}')

    # Perceptual Path Length
    if 'ppl' in args.metrics:
        print('Computing Perceptual Path Length...')
        ppl_metric = PerceptualPathLength(device=args.device)
        ppl_score = ppl_metric.calculate(generator, num_samples=1000, latent_dim=args.latent_dim)
        results['ppl'] = ppl_score
        print(f'PPL: {ppl_score:.4f}')

    # Precision and Recall
    if 'pr' in args.metrics:
        print('Computing Precision and Recall...')
        pr_metric = PrecisionRecall(device=args.device)
        precision, recall = pr_metric.calculate(real_samples, fake_samples)
        results['precision'] = precision
        results['recall'] = recall
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')

    # Kernel Inception Distance
    if 'kid' in args.metrics:
        print('Computing KID...')
        kid_metric = KernelInceptionDistance(device=args.device)
        kid_score = kid_metric.calculate(real_samples, fake_samples)
        results['kid'] = kid_score
        print(f'KID: {kid_score:.6f}')

    return results


def visualize_results(args, generator, real_samples, fake_samples, visualizer):
    """Generate visualizations"""

    # Generated samples grid
    print('Visualizing generated samples...')
    visualizer.plot_generated_samples(
        fake_samples[:args.num_viz_samples],
        n_samples=args.num_viz_samples,
        save_path=os.path.join(args.output_dir, 'generated_samples.png')
    )

    # Comparison grid
    print('Visualizing comparison...')
    visualizer.plot_comparison_grid(
        real_samples[:8],
        fake_samples[:8],
        save_path=os.path.join(args.output_dir, 'comparison.png')
    )

    # Latent space interpolation
    print('Visualizing latent space interpolation...')
    visualizer.plot_latent_space_interpolation(
        generator,
        latent_dim=args.latent_dim,
        n_steps=10,
        device=args.device,
        save_path=os.path.join(args.output_dir, 'interpolation.png')
    )

    # 2D latent space
    print('Visualizing 2D latent space...')
    visualizer.plot_latent_space_2d(
        generator,
        latent_dim=args.latent_dim,
        n_points=10,
        device=args.device,
        save_path=os.path.join(args.output_dir, 'latent_space_2d.png')
    )


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    print(f'Loading model from {args.model_path}...')
    generator = load_model(args)
    print('Model loaded successfully')

    # Generate fake samples
    fake_samples = generate_samples(
        generator,
        args.num_samples,
        args.latent_dim,
        args.batch_size,
        device
    )
    print(f'Generated {len(fake_samples)} samples')

    # Load real samples
    print('Loading real samples...')
    dataloader = get_dataloader(
        args.dataset,
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False
    )
    real_samples = load_real_samples(dataloader, args.num_samples)
    print(f'Loaded {len(real_samples)} real samples')

    # Evaluate metrics
    print('\n' + '='*50)
    print('EVALUATION RESULTS')
    print('='*50)
    results = evaluate_metrics(args, generator, real_samples, fake_samples)

    # Save results
    results_path = os.path.join(args.output_dir, 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f'\nResults saved to {results_path}')

    # Visualize if requested
    if args.visualize:
        print('\n' + '='*50)
        print('GENERATING VISUALIZATIONS')
        print('='*50)
        visualizer = GANVisualizer(save_dir=args.output_dir)
        visualize_results(args, generator, real_samples, fake_samples, visualizer)
        print(f'Visualizations saved to {args.output_dir}')

    print('\nEvaluation complete!')


if __name__ == '__main__':
    main()
