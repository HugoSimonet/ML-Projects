"""
Quality Assessment Metrics for GANs
Implements IS, FID, PPL, Precision/Recall, and other evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from scipy.stats import entropy
from tqdm import tqdm


class InceptionV3(nn.Module):
    """InceptionV3 network for feature extraction"""

    def __init__(self, output_blocks=[3], resize_input=True, normalize_input=True):
        super().__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)

        inception = models.inception_v3(pretrained=True, transform_input=False)

        # Block 0: input to maxpool1
        self.block0 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Block 1: maxpool1 to maxpool2
        self.block1 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Block 2: maxpool2 to aux classifier
        self.block2 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
        )

        # Block 3: aux classifier to final avgpool
        self.block3 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = []

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1

        x = self.block0(x)
        if 0 in self.output_blocks:
            output.append(x)

        x = self.block1(x)
        if 1 in self.output_blocks:
            output.append(x)

        x = self.block2(x)
        if 2 in self.output_blocks:
            output.append(x)

        x = self.block3(x)
        if 3 in self.output_blocks:
            output.append(x)

        return output


class InceptionScore:
    """
    Inception Score (IS)
    Measures quality and diversity of generated images
    Higher is better
    """

    def __init__(self, device='cuda', batch_size=32, splits=10):
        self.device = device
        self.batch_size = batch_size
        self.splits = splits

        # Load InceptionV3
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.eval()
        self.inception.to(device)

    def compute_predictions(self, images):
        """Compute class predictions for images"""
        preds = []

        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size].to(self.device)

                # Resize to 299x299
                if batch.shape[2] != 299 or batch.shape[3] != 299:
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

                pred = self.inception(batch)
                preds.append(F.softmax(pred, dim=1).cpu().numpy())

        return np.concatenate(preds, axis=0)

    def calculate(self, images):
        """
        Calculate Inception Score
        Args:
            images: torch.Tensor of shape (N, C, H, W) in range [0, 1]
        Returns:
            mean and std of IS across splits
        """
        preds = self.compute_predictions(images)

        # Calculate IS
        scores = []
        for i in range(self.splits):
            part = preds[i * (len(preds) // self.splits): (i + 1) * (len(preds) // self.splits)]
            py = np.mean(part, axis=0)
            scores.append(np.exp(np.mean([entropy(p, py) for p in part])))

        return np.mean(scores), np.std(scores)


class FrechetInceptionDistance:
    """
    Fréchet Inception Distance (FID)
    Measures similarity between real and generated image distributions
    Lower is better
    """

    def __init__(self, device='cuda', batch_size=32):
        self.device = device
        self.batch_size = batch_size

        # Load InceptionV3 for feature extraction
        self.inception = InceptionV3(output_blocks=[3]).to(device)
        self.inception.eval()

    def extract_features(self, images):
        """Extract features from images"""
        features = []

        with torch.no_grad():
            for i in tqdm(range(0, len(images), self.batch_size), desc='Extracting features'):
                batch = images[i:i + self.batch_size].to(self.device)

                feat = self.inception(batch)[0]
                feat = feat.squeeze(-1).squeeze(-1)
                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def calculate_statistics(self, features):
        """Calculate mean and covariance of features"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Fréchet distance between two Gaussians"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def calculate(self, real_images, fake_images):
        """
        Calculate FID between real and fake images
        Args:
            real_images: torch.Tensor of shape (N, C, H, W)
            fake_images: torch.Tensor of shape (N, C, H, W)
        Returns:
            FID score
        """
        # Extract features
        real_features = self.extract_features(real_images)
        fake_features = self.extract_features(fake_images)

        # Calculate statistics
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_fake, sigma_fake = self.calculate_statistics(fake_features)

        # Calculate FID
        fid = self.calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

        return fid


class PerceptualPathLength:
    """
    Perceptual Path Length (PPL)
    Measures smoothness of interpolation in latent space
    Lower is better
    """

    def __init__(self, device='cuda', epsilon=1e-4):
        self.device = device
        self.epsilon = epsilon

        # Load VGG for perceptual distance
        vgg = models.vgg16(pretrained=True).features
        self.perceptual_model = vgg.to(device)
        self.perceptual_model.eval()

        for param in self.perceptual_model.parameters():
            param.requires_grad = False

    def compute_perceptual_distance(self, img1, img2):
        """Compute perceptual distance between two images"""
        with torch.no_grad():
            feat1 = self.perceptual_model(img1)
            feat2 = self.perceptual_model(img2)
            distance = (feat1 - feat2).pow(2).mean()
        return distance

    def calculate(self, generator, num_samples=10000, latent_dim=512):
        """
        Calculate PPL for generator
        Args:
            generator: Generator model
            num_samples: Number of samples to evaluate
            latent_dim: Dimension of latent space
        Returns:
            PPL score
        """
        generator.eval()
        distances = []

        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc='Computing PPL'):
                # Sample two latent codes
                z1 = torch.randn(1, latent_dim, device=self.device)
                z2 = torch.randn(1, latent_dim, device=self.device)

                # Interpolate
                t = torch.rand(1, device=self.device)
                z_interp1 = z1 * (1 - t) + z2 * t
                z_interp2 = z1 * (1 - t + self.epsilon) + z2 * (t + self.epsilon)

                # Generate images
                img1 = generator(z_interp1)
                img2 = generator(z_interp2)

                # Compute perceptual distance
                dist = self.compute_perceptual_distance(img1, img2)
                distances.append(dist.item())

        return np.mean(distances) / self.epsilon


class PrecisionRecall:
    """
    Precision and Recall for Distributions
    Measures quality (precision) and coverage (recall)
    """

    def __init__(self, device='cuda', k=3, batch_size=32):
        self.device = device
        self.k = k
        self.batch_size = batch_size

        # Feature extractor
        self.inception = InceptionV3(output_blocks=[3]).to(device)
        self.inception.eval()

    def extract_features(self, images):
        """Extract features"""
        features = []

        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size].to(self.device)
                feat = self.inception(batch)[0]
                feat = feat.squeeze(-1).squeeze(-1)
                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def compute_pairwise_distances(self, X, Y):
        """Compute pairwise distances"""
        X_norm = np.sum(X ** 2, axis=1, keepdims=True)
        Y_norm = np.sum(Y ** 2, axis=1, keepdims=True)
        distances = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        return distances

    def compute_nearest_neighbor_distances(self, X, Y, k=3):
        """Compute k-nearest neighbor distances"""
        distances = self.compute_pairwise_distances(X, Y)
        sorted_distances = np.sort(distances, axis=1)
        return sorted_distances[:, :k]

    def calculate(self, real_images, fake_images):
        """
        Calculate precision and recall
        Args:
            real_images: torch.Tensor of real images
            fake_images: torch.Tensor of fake images
        Returns:
            precision, recall
        """
        # Extract features
        real_features = self.extract_features(real_images)
        fake_features = self.extract_features(fake_images)

        # Compute nearest neighbor distances
        real_nn_distances = self.compute_nearest_neighbor_distances(
            real_features, real_features, self.k + 1
        )[:, 1:]  # Exclude self
        fake_nn_distances = self.compute_nearest_neighbor_distances(
            fake_features, fake_features, self.k + 1
        )[:, 1:]

        # Manifold estimation
        real_manifold = real_nn_distances[:, -1]
        fake_manifold = fake_nn_distances[:, -1]

        # Precision: how many fake samples are close to real samples
        fake_to_real_distances = self.compute_nearest_neighbor_distances(
            fake_features, real_features, 1
        )
        precision = np.mean(fake_to_real_distances.ravel() <= real_manifold[np.newaxis].T)

        # Recall: how many real samples are close to fake samples
        real_to_fake_distances = self.compute_nearest_neighbor_distances(
            real_features, fake_features, 1
        )
        recall = np.mean(real_to_fake_distances.ravel() <= fake_manifold[np.newaxis].T)

        return precision, recall


class KernelInceptionDistance:
    """
    Kernel Inception Distance (KID)
    Alternative to FID, more unbiased for small sample sizes
    Lower is better
    """

    def __init__(self, device='cuda', batch_size=32):
        self.device = device
        self.batch_size = batch_size

        self.inception = InceptionV3(output_blocks=[3]).to(device)
        self.inception.eval()

    def extract_features(self, images):
        """Extract features"""
        features = []

        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size].to(self.device)
                feat = self.inception(batch)[0]
                feat = feat.squeeze(-1).squeeze(-1)
                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def polynomial_kernel(self, X, Y, degree=3, gamma=None, coef0=1):
        """Compute polynomial kernel"""
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        K = (gamma * np.dot(X, Y.T) + coef0) ** degree
        return K

    def calculate(self, real_images, fake_images, subset_size=1000):
        """
        Calculate KID
        Args:
            real_images: torch.Tensor of real images
            fake_images: torch.Tensor of fake images
            subset_size: Size of subsets for computation
        Returns:
            KID score
        """
        # Extract features
        real_features = self.extract_features(real_images)
        fake_features = self.extract_features(fake_images)

        # Subsample
        n_real = min(subset_size, len(real_features))
        n_fake = min(subset_size, len(fake_features))

        real_subset = real_features[np.random.choice(len(real_features), n_real, replace=False)]
        fake_subset = fake_features[np.random.choice(len(fake_features), n_fake, replace=False)]

        # Compute kernels
        K_rr = self.polynomial_kernel(real_subset, real_subset)
        K_ff = self.polynomial_kernel(fake_subset, fake_subset)
        K_rf = self.polynomial_kernel(real_subset, fake_subset)

        # Compute KID
        kid = np.mean(K_rr) + np.mean(K_ff) - 2 * np.mean(K_rf)

        return kid


class ModeScore:
    """
    Mode Score
    Evaluates mode coverage and quality
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.eval()
        self.inception.to(device)

    def calculate(self, fake_images, num_classes=1000):
        """Calculate mode score"""
        with torch.no_grad():
            fake_images = fake_images.to(self.device)

            if fake_images.shape[2] != 299:
                fake_images = F.interpolate(fake_images, size=(299, 299), mode='bilinear', align_corners=False)

            predictions = self.inception(fake_images)
            predictions = F.softmax(predictions, dim=1)

            # KL divergence
            py = predictions.mean(dim=0)
            kl_div = (predictions * (predictions.log() - py.log())).sum(dim=1).mean()

            return torch.exp(kl_div).item()
