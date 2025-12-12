"""
Data Loading Utilities
Functions for loading and preparing datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import os
import numpy as np


class ImageDataset(Dataset):
    """
    Custom Image Dataset
    Loads images from a directory
    """

    def __init__(self, root_dir, image_size=64, transform=None, extension='.jpg'):
        self.root_dir = root_dir
        self.image_size = image_size
        self.extension = extension

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # Get all image paths
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        """Get all image paths in directory"""
        image_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(self.extension) or file.endswith('.png'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


class LabeledImageDataset(Dataset):
    """
    Labeled Image Dataset
    Loads images with labels from directory structure
    """

    def __init__(self, root_dir, image_size=64, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # Get classes and image paths
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        """Create dataset samples"""
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloader(
    dataset_name,
    data_path='./data',
    image_size=64,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    augment=True
):
    """
    Get dataloader for common datasets
    Args:
        dataset_name: Name of dataset (cifar10, mnist, celeba, custom)
        data_path: Path to dataset
        image_size: Size to resize images
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation
    Returns:
        DataLoader
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    # Load dataset based on name
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name == 'mnist':
        # Expand grayscale to 3 channels
        transform_mnist = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transform_mnist
        )
    elif dataset_name == 'celeba':
        dataset = datasets.CelebA(
            root=data_path,
            split='train',
            download=True,
            transform=transform
        )
    elif dataset_name == 'custom':
        dataset = ImageDataset(data_path, image_size=image_size, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def prepare_dataset(images, image_size=64, normalize=True):
    """
    Prepare images for GAN training
    Args:
        images: Tensor or array of images
        image_size: Target size
        normalize: Whether to normalize to [-1, 1]
    Returns:
        Prepared images
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    # Ensure correct shape
    if images.dim() == 3:
        images = images.unsqueeze(1)

    # Resize if needed
    if images.shape[2] != image_size or images.shape[3] != image_size:
        images = torch.nn.functional.interpolate(
            images, size=(image_size, image_size), mode='bilinear', align_corners=False
        )

    # Normalize to [-1, 1]
    if normalize:
        if images.max() > 1.0:
            images = images / 255.0
        images = images * 2 - 1

    return images


class PairedDataset(Dataset):
    """
    Paired Dataset for domain transfer (CycleGAN)
    Loads paired images from two domains
    """

    def __init__(self, domain_a_path, domain_b_path, image_size=256, transform=None):
        self.domain_a_path = domain_a_path
        self.domain_b_path = domain_b_path
        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.12)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # Get image paths
        self.domain_a_images = self._get_images(domain_a_path)
        self.domain_b_images = self._get_images(domain_b_path)

    def _get_images(self, path):
        """Get all image paths in directory"""
        images = []
        for file in os.listdir(path):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(path, file))
        return sorted(images)

    def __len__(self):
        return max(len(self.domain_a_images), len(self.domain_b_images))

    def __getitem__(self, idx):
        # Wrap around if one domain has fewer images
        a_idx = idx % len(self.domain_a_images)
        b_idx = idx % len(self.domain_b_images)

        img_a = Image.open(self.domain_a_images[a_idx]).convert('RGB')
        img_b = Image.open(self.domain_b_images[b_idx]).convert('RGB')

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return img_a, img_b


class InfiniteDataLoader:
    """
    Infinite DataLoader
    Continuously cycles through dataset
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch


def create_dataloaders(config):
    """
    Create train and validation dataloaders from config
    Args:
        config: Configuration dict with dataset parameters
    Returns:
        train_loader, val_loader
    """
    train_transform = transforms.Compose([
        transforms.Resize(config.get('image_size', 64)),
        transforms.CenterCrop(config.get('image_size', 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.get('image_size', 64)),
        transforms.CenterCrop(config.get('image_size', 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Create datasets
    if config['dataset'] == 'custom':
        train_dataset = ImageDataset(
            config['train_path'],
            image_size=config['image_size'],
            transform=train_transform
        )
        val_dataset = ImageDataset(
            config['val_path'],
            image_size=config['image_size'],
            transform=val_transform
        ) if 'val_path' in config else None
    else:
        train_loader = get_dataloader(
            config['dataset'],
            data_path=config.get('data_path', './data'),
            image_size=config['image_size'],
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4)
        )
        return train_loader, None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    ) if val_dataset is not None else None

    return train_loader, val_loader
