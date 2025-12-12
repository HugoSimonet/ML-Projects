"""
Sampling Utilities
Functions for sampling from latent space and interpolation
"""

import torch
import numpy as np


def sample_latent(batch_size, latent_dim, device='cuda', distribution='normal'):
    """
    Sample from latent space
    Args:
        batch_size: Number of samples
        latent_dim: Dimension of latent space
        device: Device to create tensor on
        distribution: 'normal' or 'uniform'
    Returns:
        Latent vectors
    """
    if distribution == 'normal':
        z = torch.randn(batch_size, latent_dim, device=device)
    elif distribution == 'uniform':
        z = torch.rand(batch_size, latent_dim, device=device) * 2 - 1
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return z


def sample_truncated(batch_size, latent_dim, truncation=0.7, device='cuda'):
    """
    Sample from truncated normal distribution
    Improves quality at the cost of diversity
    Args:
        batch_size: Number of samples
        latent_dim: Dimension of latent space
        truncation: Truncation threshold
        device: Device to create tensor on
    Returns:
        Truncated latent vectors
    """
    z = torch.randn(batch_size, latent_dim, device=device)

    # Truncate values beyond threshold
    z = torch.clamp(z, -truncation * 2, truncation * 2)

    return z


def slerp(z1, z2, alpha):
    """
    Spherical linear interpolation between two vectors
    Better than linear interpolation for high-dimensional spaces
    Args:
        z1: First latent vector
        z2: Second latent vector
        alpha: Interpolation factor [0, 1]
    Returns:
        Interpolated vector
    """
    # Normalize vectors
    z1_norm = z1 / torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = z2 / torch.norm(z2, dim=-1, keepdim=True)

    # Compute angle
    omega = torch.acos(torch.clamp(torch.sum(z1_norm * z2_norm, dim=-1, keepdim=True), -1, 1))

    # Compute interpolation
    so = torch.sin(omega)
    if (so < 1e-8).any():
        # Fall back to linear interpolation for small angles
        return (1.0 - alpha) * z1 + alpha * z2

    s1 = torch.sin((1.0 - alpha) * omega) / so
    s2 = torch.sin(alpha * omega) / so

    return s1 * z1 + s2 * z2


def linear_interpolate(z1, z2, n_steps=10):
    """
    Linear interpolation between two latent vectors
    Args:
        z1: First latent vector
        z2: Second latent vector
        n_steps: Number of interpolation steps
    Returns:
        Interpolated vectors
    """
    alphas = torch.linspace(0, 1, n_steps, device=z1.device)
    interpolations = []

    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        interpolations.append(z_interp)

    return torch.cat(interpolations, dim=0)


def spherical_interpolate(z1, z2, n_steps=10):
    """
    Spherical interpolation between two latent vectors
    Args:
        z1: First latent vector
        z2: Second latent vector
        n_steps: Number of interpolation steps
    Returns:
        Interpolated vectors
    """
    alphas = torch.linspace(0, 1, n_steps, device=z1.device)
    interpolations = []

    for alpha in alphas:
        z_interp = slerp(z1, z2, alpha)
        interpolations.append(z_interp)

    return torch.cat(interpolations, dim=0)


def generate_grid(latent_dim, grid_size=10, range_val=2.0, device='cuda'):
    """
    Generate grid of latent vectors for visualization
    Varies first two dimensions while keeping others constant
    Args:
        latent_dim: Dimension of latent space
        grid_size: Size of grid (grid_size x grid_size)
        range_val: Range of values for grid
        device: Device to create tensor on
    Returns:
        Grid of latent vectors
    """
    # Create base vector
    base_z = torch.randn(1, latent_dim, device=device)

    # Create grid
    z_grid = []
    values = torch.linspace(-range_val, range_val, grid_size, device=device)

    for val1 in values:
        for val2 in values:
            z = base_z.clone()
            z[0, 0] = val1
            z[0, 1] = val2
            z_grid.append(z)

    return torch.cat(z_grid, dim=0)


def sample_neighborhood(z_center, radius=0.5, n_samples=8, latent_dim=100, device='cuda'):
    """
    Sample points in neighborhood of a center point
    Useful for exploring local latent space
    Args:
        z_center: Center latent vector
        radius: Radius of neighborhood
        n_samples: Number of samples
        latent_dim: Dimension of latent space
        device: Device to create tensor on
    Returns:
        Neighborhood samples
    """
    # Generate random directions
    directions = torch.randn(n_samples, latent_dim, device=device)
    directions = directions / torch.norm(directions, dim=1, keepdim=True)

    # Scale by radius
    offsets = directions * radius

    # Add to center
    samples = z_center + offsets

    return samples


def sample_from_distribution(batch_size, latent_dim, mean=0.0, std=1.0, device='cuda'):
    """
    Sample from custom Gaussian distribution
    Args:
        batch_size: Number of samples
        latent_dim: Dimension of latent space
        mean: Mean of distribution
        std: Standard deviation
        device: Device to create tensor on
    Returns:
        Sampled latent vectors
    """
    z = torch.randn(batch_size, latent_dim, device=device) * std + mean
    return z


def circular_interpolation(z1, z2, z3, n_steps=30):
    """
    Circular interpolation through three points
    Creates smooth circular path in latent space
    Args:
        z1, z2, z3: Three latent vectors
        n_steps: Number of interpolation steps
    Returns:
        Interpolated vectors
    """
    interpolations = []
    angles = torch.linspace(0, 2 * np.pi, n_steps, device=z1.device)

    for angle in angles:
        # Barycentric coordinates for circular path
        w1 = (1 + torch.cos(angle)) / 3
        w2 = (1 + torch.cos(angle + 2 * np.pi / 3)) / 3
        w3 = (1 + torch.cos(angle + 4 * np.pi / 3)) / 3

        z_interp = w1 * z1 + w2 * z2 + w3 * z3
        interpolations.append(z_interp)

    return torch.cat(interpolations, dim=0)


def random_walk(z_start, n_steps=100, step_size=0.1, latent_dim=100, device='cuda'):
    """
    Random walk in latent space
    Args:
        z_start: Starting latent vector
        n_steps: Number of steps
        step_size: Size of each step
        latent_dim: Dimension of latent space
        device: Device to create tensor on
    Returns:
        Path of latent vectors
    """
    path = [z_start]
    z_current = z_start.clone()

    for _ in range(n_steps - 1):
        # Random step
        step = torch.randn(1, latent_dim, device=device) * step_size
        z_current = z_current + step
        path.append(z_current.clone())

    return torch.cat(path, dim=0)


def attribute_vector(generator, attribute_classifier, positive_samples, negative_samples,
                     latent_dim=100, device='cuda'):
    """
    Find attribute direction in latent space
    Useful for controllable generation
    Args:
        generator: Generator model
        attribute_classifier: Classifier for attribute
        positive_samples: Samples with positive attribute
        negative_samples: Samples with negative attribute
        latent_dim: Dimension of latent space
        device: Device
    Returns:
        Attribute direction vector
    """
    generator.eval()
    attribute_classifier.eval()

    with torch.no_grad():
        # Generate latent codes for samples
        pos_latents = []
        neg_latents = []

        # This is simplified - in practice, you'd need to invert real images to latent space
        # or use labeled latent codes

        # Average difference
        pos_mean = torch.mean(torch.stack(pos_latents), dim=0)
        neg_mean = torch.mean(torch.stack(neg_latents), dim=0)

        attribute_dir = pos_mean - neg_mean

        # Normalize
        attribute_dir = attribute_dir / torch.norm(attribute_dir)

    return attribute_dir


def manipulate_latent(z, attribute_vector, strength=1.0):
    """
    Manipulate latent code along attribute direction
    Args:
        z: Latent vector
        attribute_vector: Direction vector for attribute
        strength: Strength of manipulation
    Returns:
        Manipulated latent vector
    """
    return z + strength * attribute_vector


def mix_styles(z1, z2, crossover_point=4):
    """
    Mix styles from two latent codes
    Used in StyleGAN for style mixing
    Args:
        z1: First latent vector
        z2: Second latent vector
        crossover_point: Point to switch between styles
    Returns:
        Mixed style codes
    """
    # This is simplified - actual implementation depends on generator architecture
    mixed = z1.clone()
    mixed[:, crossover_point:] = z2[:, crossover_point:]
    return mixed
