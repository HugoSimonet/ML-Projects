"""
Helper Utility Functions
General utility functions for the project
"""

import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_file: Path to log file
        level: Logging level

    Returns:
        Logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_directories(directories: list):
    """
    Create directories if they don't exist

    Args:
        directories: List of directory paths
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get PyTorch device

    Args:
        gpu_id: GPU ID to use (None for automatic selection)

    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def print_model_summary(model: torch.nn.Module):
    """
    Print model summary

    Args:
        model: PyTorch model
    """
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(model)
    print("=" * 80)
    print(f"Total Parameters: {count_parameters(model):,}")
    print("=" * 80)


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute moving average

    Args:
        data: Input data [N]
        window_size: Window size

    Returns:
        Smoothed data
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def normalize_data(
    data: np.ndarray,
    method: str = 'minmax'
) -> tuple:
    """
    Normalize data

    Args:
        data: Input data
        method: Normalization method ('minmax', 'zscore')

    Returns:
        Normalized data, stats (for denormalization)
    """
    if method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        stats = {'min': min_val, 'max': max_val}

    elif method == 'zscore':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
        stats = {'mean': mean, 'std': std}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, stats


def denormalize_data(
    data: np.ndarray,
    stats: Dict[str, np.ndarray],
    method: str = 'minmax'
) -> np.ndarray:
    """
    Denormalize data

    Args:
        data: Normalized data
        stats: Statistics from normalization
        method: Normalization method used

    Returns:
        Original scale data
    """
    if method == 'minmax':
        return data * (stats['max'] - stats['min']) + stats['min']

    elif method == 'zscore':
        return data * stats['std'] + stats['mean']

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Check if tensor contains NaN or Inf values

    Args:
        tensor: Tensor to check
        name: Name for logging

    Returns:
        True if contains NaN or Inf
    """
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan:
        print(f"WARNING: {name} contains NaN values!")

    if has_inf:
        print(f"WARNING: {name} contains Inf values!")

    return has_nan or has_inf


def gradient_clipping(
    model: torch.nn.Module,
    max_norm: float = 1.0
):
    """
    Clip gradients

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def early_stopping_check(
    val_losses: list,
    patience: int = 10,
    min_delta: float = 0.0
) -> bool:
    """
    Check if early stopping criteria is met

    Args:
        val_losses: List of validation losses
        patience: Number of epochs to wait
        min_delta: Minimum change to qualify as improvement

    Returns:
        True if should stop
    """
    if len(val_losses) < patience + 1:
        return False

    best_loss = min(val_losses[:-patience])
    recent_best = min(val_losses[-patience:])

    return recent_best > best_loss - min_delta


def format_time(seconds: float) -> str:
    """
    Format time in seconds to readable string

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


def exponential_moving_average(
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    decay: float = 0.999
):
    """
    Update EMA model

    Args:
        model: Current model
        ema_model: EMA model
        decay: EMA decay rate
    """
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def split_train_val_test(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1
) -> tuple:
    """
    Split data into train, validation, and test sets

    Args:
        data: Input data
        train_ratio: Training data ratio
        val_ratio: Validation data ratio

    Returns:
        train_data, val_data, test_data
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


class AverageMeter:
    """
    Compute and store the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping handler
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop

        Args:
            score: Current score

        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False
