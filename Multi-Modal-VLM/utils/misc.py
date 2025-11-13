"""
Miscellaneous utility functions
"""

import torch
import numpy as np
import random
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device

    Args:
        device: Device string ('cuda', 'cpu', or None for auto)

    Returns:
        torch.device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.device(device)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_model_size(model: torch.nn.Module) -> float:
    """
    Get model size in MB

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 ** 2

    return size_mb


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")

    # Test seed
    set_seed(42)
    print(f"Random number: {random.random()}")

    # Test device
    device = get_device()
    print(f"Device: {device}")

    # Test parameter counting
    model = torch.nn.Linear(10, 10)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params}")

    # Test time formatting
    print(f"Time: {format_time(3725)}")

    # Test model size
    size = get_model_size(model)
    print(f"Model size: {size:.2f} MB")
