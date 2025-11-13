"""Logging utility."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from .config_loader import load_config


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    Set up logger with file and console handlers.

    Args:
        log_file: Path to log file. If None, only logs to console.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        rotation: Log rotation policy.
        retention: Log retention period.
    """
    # Remove default handler
    logger.remove()

    # Load config
    try:
        config = load_config()
        log_format = config.get("logging", {}).get(
            "format",
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        level = config.get("logging", {}).get("level", level)
    except:
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # Add console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logger initialized with level: {level}")


def get_logger(name: str):
    """
    Get a logger instance.

    Args:
        name: Name of the logger.

    Returns:
        Logger instance.
    """
    return logger.bind(name=name)
