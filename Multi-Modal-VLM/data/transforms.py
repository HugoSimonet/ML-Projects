"""
Image transformation and augmentation utilities
"""

import torch
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import random


class RandomAugmentation:
    """
    Random augmentation for vision-language training
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # Random horizontal flip
            if random.random() < 0.5:
                img = transforms.functional.hflip(img)

            # Random color jitter
            if random.random() < 0.5:
                jitter = transforms.ColorJitter(
                    brightness=0.4, contrast=0.4,
                    saturation=0.4, hue=0.1
                )
                img = jitter(img)

        return img


def get_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get training transforms with augmentation

    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Get validation transforms without augmentation

    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Denormalize image tensor for visualization

    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Create a dummy image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Test train transforms
    train_transform = get_train_transforms()
    transformed = train_transform(img)
    print(f"Train transform output shape: {transformed.shape}")

    # Test val transforms
    val_transform = get_val_transforms()
    transformed = val_transform(img)
    print(f"Val transform output shape: {transformed.shape}")

    # Test denormalize
    denorm = denormalize(transformed)
    print(f"Denormalized shape: {denorm.shape}")
    print(f"Denormalized range: [{denorm.min():.3f}, {denorm.max():.3f}]")
