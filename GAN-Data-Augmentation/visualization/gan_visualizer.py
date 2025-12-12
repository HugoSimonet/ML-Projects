"""
GAN Visualization Tools
Comprehensive visualization for training analysis and generation quality
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

# Optional plotly import
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class GANVisualizer:
    """
    Comprehensive GAN Visualizer
    Provides various visualization methods for GANs
    """

    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def plot_training_curves(self, g_losses, d_losses, save_path=None):
        """Plot generator and discriminator losses over time"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        epochs = range(1, len(g_losses) + 1)
        ax1.plot(epochs, g_losses, label='Generator Loss', linewidth=2)
        ax1.plot(epochs, d_losses, label='Discriminator Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Moving average
        window = min(10, len(g_losses) // 5)
        if window > 1:
            g_ma = np.convolve(g_losses, np.ones(window)/window, mode='valid')
            d_ma = np.convolve(d_losses, np.ones(window)/window, mode='valid')
            ma_epochs = range(window, len(g_losses) + 1)

            ax2.plot(ma_epochs, g_ma, label=f'G Loss (MA-{window})', linewidth=2)
            ax2.plot(ma_epochs, d_ma, label=f'D Loss (MA-{window})', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Smoothed Training Losses')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')

        plt.close()

    def plot_generated_samples(self, samples, n_samples=64, nrow=8, save_path=None):
        """Plot grid of generated samples"""
        if samples.size(0) > n_samples:
            samples = samples[:n_samples]

        # Denormalize if needed
        if samples.min() < 0:
            samples = (samples + 1) / 2

        samples = torch.clamp(samples, 0, 1)

        # Create grid
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        grid = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(15, 15))
        plt.imshow(grid)
        plt.axis('off')
        plt.title('Generated Samples', fontsize=16)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'generated_samples.png'), dpi=300, bbox_inches='tight')

        plt.close()

    def plot_latent_space_interpolation(self, generator, latent_dim=100, n_steps=10, device='cuda', save_path=None):
        """Visualize interpolation in latent space"""
        generator.eval()

        with torch.no_grad():
            # Sample two random points
            z1 = torch.randn(1, latent_dim, device=device)
            z2 = torch.randn(1, latent_dim, device=device)

            # Interpolate
            interpolations = []
            for alpha in np.linspace(0, 1, n_steps):
                z_interp = z1 * (1 - alpha) + z2 * alpha
                img = generator(z_interp)
                interpolations.append(img)

            interpolations = torch.cat(interpolations, dim=0)

        # Plot
        self.plot_generated_samples(interpolations, n_samples=n_steps, nrow=n_steps, save_path=save_path)

    def plot_latent_space_2d(self, generator, latent_dim=100, n_points=20, device='cuda', save_path=None):
        """Visualize 2D grid in latent space"""
        generator.eval()

        # Create grid
        z_range = np.linspace(-2, 2, n_points)
        samples = []

        with torch.no_grad():
            for i, z1_val in enumerate(z_range):
                for j, z2_val in enumerate(z_range):
                    z = torch.randn(1, latent_dim, device=device)
                    z[0, 0] = z1_val
                    z[0, 1] = z2_val
                    img = generator(z)
                    samples.append(img)

        samples = torch.cat(samples, dim=0)

        # Plot
        self.plot_generated_samples(samples, n_samples=len(samples), nrow=n_points, save_path=save_path)

    def plot_quality_metrics(self, metrics_history, save_path=None):
        """Plot quality metrics over time"""
        if not PLOTLY_AVAILABLE:
            print("Warning: plotly not available. Install with: pip install plotly")
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Inception Score', 'FID Score', 'Precision', 'Recall')
        )

        if 'is' in metrics_history:
            fig.add_trace(
                go.Scatter(y=metrics_history['is'], mode='lines+markers', name='IS'),
                row=1, col=1
            )

        if 'fid' in metrics_history:
            fig.add_trace(
                go.Scatter(y=metrics_history['fid'], mode='lines+markers', name='FID'),
                row=1, col=2
            )

        if 'precision' in metrics_history:
            fig.add_trace(
                go.Scatter(y=metrics_history['precision'], mode='lines+markers', name='Precision'),
                row=2, col=1
            )

        if 'recall' in metrics_history:
            fig.add_trace(
                go.Scatter(y=metrics_history['recall'], mode='lines+markers', name='Recall'),
                row=2, col=2
            )

        fig.update_xaxes(title_text='Epoch')
        fig.update_layout(height=800, showlegend=False, title_text='Quality Metrics Over Time')

        if save_path:
            fig.write_html(save_path)
        else:
            fig.write_html(os.path.join(self.save_dir, 'quality_metrics.html'))

    def plot_feature_space(self, real_features, fake_features, method='tsne', save_path=None):
        """Visualize feature space with dimensionality reduction"""
        # Combine features
        all_features = np.vstack([real_features, fake_features])
        labels = np.array(['Real'] * len(real_features) + ['Fake'] * len(fake_features))

        # Reduce dimensions
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)

        reduced = reducer.fit_transform(all_features)

        # Plot
        plt.figure(figsize=(10, 8))
        for label in ['Real', 'Fake']:
            mask = labels == label
            plt.scatter(
                reduced[mask, 0], reduced[mask, 1],
                label=label, alpha=0.6, s=30
            )

        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Feature Space Visualization ({method.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'feature_space.png'), dpi=300, bbox_inches='tight')

        plt.close()

    def plot_comparison_grid(self, real_samples, fake_samples, n_samples=8, save_path=None):
        """Plot side-by-side comparison of real and fake samples"""
        real_samples = real_samples[:n_samples]
        fake_samples = fake_samples[:n_samples]

        # Denormalize
        if real_samples.min() < 0:
            real_samples = (real_samples + 1) / 2
        if fake_samples.min() < 0:
            fake_samples = (fake_samples + 1) / 2

        real_samples = torch.clamp(real_samples, 0, 1)
        fake_samples = torch.clamp(fake_samples, 0, 1)

        # Create grids
        real_grid = make_grid(real_samples, nrow=n_samples, padding=2)
        fake_grid = make_grid(fake_samples, nrow=n_samples, padding=2)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))

        ax1.imshow(real_grid.permute(1, 2, 0).cpu().numpy())
        ax1.set_title('Real Samples', fontsize=14)
        ax1.axis('off')

        ax2.imshow(fake_grid.permute(1, 2, 0).cpu().numpy())
        ax2.set_title('Generated Samples', fontsize=14)
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'comparison_grid.png'), dpi=300, bbox_inches='tight')

        plt.close()

    def plot_conditional_samples(self, samples_by_class, class_names=None, save_path=None):
        """Plot conditional samples organized by class"""
        n_classes = len(samples_by_class)

        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]

        fig, axes = plt.subplots(n_classes, 1, figsize=(15, 3 * n_classes))

        if n_classes == 1:
            axes = [axes]

        for idx, (samples, class_name) in enumerate(zip(samples_by_class, class_names)):
            if samples.min() < 0:
                samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)

            grid = make_grid(samples[:8], nrow=8, padding=2)
            axes[idx].imshow(grid.permute(1, 2, 0).cpu().numpy())
            axes[idx].set_title(class_name, fontsize=12)
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'conditional_samples.png'), dpi=300, bbox_inches='tight')

        plt.close()

    def plot_gradient_norms(self, g_grad_norms, d_grad_norms, save_path=None):
        """Plot gradient norms to detect training issues"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(g_grad_norms, label='Generator', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.title('Generator Gradient Norms')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        plt.plot(d_grad_norms, label='Discriminator', linewidth=2, color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.title('Discriminator Gradient Norms')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'gradient_norms.png'), dpi=300, bbox_inches='tight')

        plt.close()


def plot_training_curves(g_losses, d_losses, save_path='training_curves.png'):
    """Standalone function to plot training curves"""
    visualizer = GANVisualizer()
    visualizer.plot_training_curves(g_losses, d_losses, save_path)


def plot_generated_samples(samples, n_samples=64, nrow=8, save_path='generated_samples.png'):
    """Standalone function to plot generated samples"""
    visualizer = GANVisualizer()
    visualizer.plot_generated_samples(samples, n_samples, nrow, save_path)


def plot_latent_space_interpolation(generator, latent_dim=100, n_steps=10, device='cuda', save_path='interpolation.png'):
    """Standalone function to plot latent space interpolation"""
    visualizer = GANVisualizer()
    visualizer.plot_latent_space_interpolation(generator, latent_dim, n_steps, device, save_path)
