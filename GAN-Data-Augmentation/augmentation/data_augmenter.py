"""
GAN-Based Data Augmentation Pipeline
Provides augmentation for various data types and downstream tasks
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from typing import Optional, Union, List
import random
from tqdm import tqdm


class GANAugmenter:
    """
    Base GAN Augmenter
    Generates synthetic data to augment training datasets
    """

    def __init__(
        self,
        generator,
        latent_dim=100,
        device='cuda',
        quality_threshold=None
    ):
        self.generator = generator.to(device)
        self.generator.eval()
        self.latent_dim = latent_dim
        self.device = device
        self.quality_threshold = quality_threshold

    def generate_samples(self, num_samples, batch_size=64):
        """Generate synthetic samples"""
        samples = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(z)
                samples.append(fake_data.cpu())

        return torch.cat(samples, dim=0)

    def augment_dataset(self, dataset, augmentation_ratio=0.5, batch_size=64):
        """
        Augment dataset with synthetic samples
        Args:
            dataset: Original dataset
            augmentation_ratio: Ratio of synthetic to real data
            batch_size: Batch size for generation
        Returns:
            Augmented dataset
        """
        num_synthetic = int(len(dataset) * augmentation_ratio)
        synthetic_samples = self.generate_samples(num_synthetic, batch_size)

        # Create synthetic dataset
        synthetic_dataset = SyntheticDataset(synthetic_samples)

        # Combine datasets
        augmented_dataset = ConcatDataset([dataset, synthetic_dataset])

        return augmented_dataset

    def save_samples(self, num_samples, save_path, batch_size=64):
        """Generate and save samples"""
        samples = self.generate_samples(num_samples, batch_size)
        torch.save(samples, save_path)
        return samples


class ImageAugmenter(GANAugmenter):
    """
    Image Augmentation with GANs
    Specialized for image data with quality filtering
    """

    def __init__(
        self,
        generator,
        latent_dim=100,
        device='cuda',
        quality_threshold=None,
        use_traditional_augmentation=True
    ):
        super().__init__(generator, latent_dim, device, quality_threshold)
        self.use_traditional_augmentation = use_traditional_augmentation

    def apply_traditional_augmentation(self, images):
        """Apply traditional augmentation techniques"""
        augmented = []

        for img in images:
            # Random horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])

            # Random rotation
            if random.random() > 0.5:
                angle = random.choice([0, 90, 180, 270])
                if angle == 90:
                    img = torch.rot90(img, k=1, dims=[1, 2])
                elif angle == 180:
                    img = torch.rot90(img, k=2, dims=[1, 2])
                elif angle == 270:
                    img = torch.rot90(img, k=3, dims=[1, 2])

            # Random brightness adjustment
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = torch.clamp(img * factor, 0, 1)

            augmented.append(img)

        return torch.stack(augmented)

    def generate_samples(self, num_samples, batch_size=64):
        """Generate samples with optional traditional augmentation"""
        samples = super().generate_samples(num_samples, batch_size)

        if self.use_traditional_augmentation:
            samples = self.apply_traditional_augmentation(samples)

        return samples


class ConditionalAugmenter(GANAugmenter):
    """
    Conditional GAN Augmenter
    Generates class-specific synthetic samples
    """

    def __init__(
        self,
        generator,
        num_classes,
        latent_dim=100,
        device='cuda',
        balance_classes=True
    ):
        super().__init__(generator, latent_dim, device)
        self.num_classes = num_classes
        self.balance_classes = balance_classes

    def generate_samples(self, num_samples, class_label=None, batch_size=64):
        """Generate conditional samples"""
        samples = []
        labels = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)

                if class_label is None:
                    batch_labels = torch.randint(
                        0, self.num_classes,
                        (current_batch_size,),
                        device=self.device
                    )
                else:
                    batch_labels = torch.full(
                        (current_batch_size,),
                        class_label,
                        dtype=torch.long,
                        device=self.device
                    )

                fake_data = self.generator(z, batch_labels)
                samples.append(fake_data.cpu())
                labels.append(batch_labels.cpu())

        return torch.cat(samples, dim=0), torch.cat(labels, dim=0)

    def augment_dataset(self, dataset, augmentation_ratio=0.5, batch_size=64):
        """Augment with class balancing"""
        if self.balance_classes:
            # Count samples per class
            class_counts = self._count_classes(dataset)

            # Generate samples to balance
            synthetic_samples = []
            synthetic_labels = []

            for class_idx in range(self.num_classes):
                num_to_generate = int(class_counts[class_idx] * augmentation_ratio)
                samples, labels = self.generate_samples(
                    num_to_generate,
                    class_label=class_idx,
                    batch_size=batch_size
                )
                synthetic_samples.append(samples)
                synthetic_labels.append(labels)

            synthetic_samples = torch.cat(synthetic_samples, dim=0)
            synthetic_labels = torch.cat(synthetic_labels, dim=0)
        else:
            num_synthetic = int(len(dataset) * augmentation_ratio)
            synthetic_samples, synthetic_labels = self.generate_samples(
                num_synthetic, batch_size=batch_size
            )

        # Create synthetic dataset
        synthetic_dataset = SyntheticDataset(synthetic_samples, synthetic_labels)

        # Combine datasets
        augmented_dataset = ConcatDataset([dataset, synthetic_dataset])

        return augmented_dataset

    def _count_classes(self, dataset):
        """Count samples per class"""
        class_counts = [0] * self.num_classes

        for _, label in dataset:
            class_counts[label] += 1

        return class_counts


class OnlineAugmenter:
    """
    Online GAN Augmenter
    Generates samples on-the-fly during training
    """

    def __init__(
        self,
        generator,
        latent_dim=100,
        device='cuda',
        augmentation_prob=0.5
    ):
        self.generator = generator.to(device)
        self.generator.eval()
        self.latent_dim = latent_dim
        self.device = device
        self.augmentation_prob = augmentation_prob

    def __call__(self, batch):
        """Augment batch with generated samples"""
        batch_size = batch.size(0)
        num_synthetic = int(batch_size * self.augmentation_prob)

        if num_synthetic > 0:
            # Generate synthetic samples
            with torch.no_grad():
                z = torch.randn(num_synthetic, self.latent_dim, device=self.device)
                synthetic = self.generator(z)

                # Combine real and synthetic
                augmented_batch = torch.cat([batch, synthetic], dim=0)

                # Shuffle
                indices = torch.randperm(augmented_batch.size(0))
                augmented_batch = augmented_batch[indices]

                return augmented_batch
        else:
            return batch


class AdaptiveAugmenter:
    """
    Adaptive GAN Augmenter
    Adjusts augmentation based on model performance
    """

    def __init__(
        self,
        generator,
        latent_dim=100,
        device='cuda',
        initial_ratio=0.5,
        min_ratio=0.1,
        max_ratio=0.9
    ):
        self.generator = generator.to(device)
        self.generator.eval()
        self.latent_dim = latent_dim
        self.device = device

        self.current_ratio = initial_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        self.performance_history = []

    def update_ratio(self, performance_metric):
        """Update augmentation ratio based on performance"""
        self.performance_history.append(performance_metric)

        if len(self.performance_history) >= 2:
            # If performance improved, increase augmentation
            if performance_metric > self.performance_history[-2]:
                self.current_ratio = min(self.max_ratio, self.current_ratio + 0.05)
            # If performance decreased, decrease augmentation
            else:
                self.current_ratio = max(self.min_ratio, self.current_ratio - 0.05)

    def generate_samples(self, num_samples, batch_size=64):
        """Generate samples with current ratio"""
        num_to_generate = int(num_samples * self.current_ratio)
        samples = []

        with torch.no_grad():
            for i in range(0, num_to_generate, batch_size):
                current_batch_size = min(batch_size, num_to_generate - i)
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(z)
                samples.append(fake_data.cpu())

        return torch.cat(samples, dim=0) if samples else torch.tensor([])


class SyntheticDataset(Dataset):
    """Dataset wrapper for synthetic samples"""

    def __init__(self, samples, labels=None):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.samples[idx], self.labels[idx]
        else:
            return self.samples[idx]


class MixupAugmenter:
    """
    Mixup Augmenter
    Combines real and synthetic samples with interpolation
    """

    def __init__(
        self,
        generator,
        latent_dim=100,
        device='cuda',
        alpha=0.2
    ):
        self.generator = generator.to(device)
        self.generator.eval()
        self.latent_dim = latent_dim
        self.device = device
        self.alpha = alpha

    def __call__(self, batch, labels=None):
        """Apply mixup between real and synthetic"""
        batch_size = batch.size(0)

        # Generate synthetic samples
        with torch.no_grad():
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            synthetic = self.generator(z).to(batch.device)

        # Mixup
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_batch = lam * batch + (1 - lam) * synthetic

        if labels is not None:
            # For classification, return both labels and lambda
            return mixed_batch, labels, lam
        else:
            return mixed_batch


class ProgressiveAugmenter:
    """
    Progressive Augmenter
    Gradually increases synthetic data ratio during training
    """

    def __init__(
        self,
        generator,
        latent_dim=100,
        device='cuda',
        initial_ratio=0.1,
        final_ratio=0.5,
        warmup_epochs=10
    ):
        self.generator = generator.to(device)
        self.generator.eval()
        self.latent_dim = latent_dim
        self.device = device

        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def get_current_ratio(self):
        """Calculate current augmentation ratio"""
        if self.current_epoch >= self.warmup_epochs:
            return self.final_ratio

        progress = self.current_epoch / self.warmup_epochs
        return self.initial_ratio + (self.final_ratio - self.initial_ratio) * progress

    def step_epoch(self):
        """Increment epoch counter"""
        self.current_epoch += 1

    def generate_samples(self, num_samples, batch_size=64):
        """Generate samples based on current epoch"""
        current_ratio = self.get_current_ratio()
        num_to_generate = int(num_samples * current_ratio)

        samples = []
        with torch.no_grad():
            for i in range(0, num_to_generate, batch_size):
                current_batch_size = min(batch_size, num_to_generate - i)
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(z)
                samples.append(fake_data.cpu())

        return torch.cat(samples, dim=0) if samples else torch.tensor([])
