"""
Logging utilities for GNN training.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'gnn',
    log_dir: str = 'logs',
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name (auto-generated if None)
        level: Logging level

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create log file name
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'

    log_path = os.path.join(log_dir, log_file)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = 'gnn') -> logging.Logger:
    """
    Get existing logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
