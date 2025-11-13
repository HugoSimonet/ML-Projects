"""Configuration loader utility."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default config.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        yaml.YAMLError: If configuration file is malformed.
    """
    if config_path is None:
        # Get default config path relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")


def get_data_path(data_type: str = "raw") -> Path:
    """
    Get path to data directory.

    Args:
        data_type: Type of data ('raw', 'processed', or 'external').

    Returns:
        Path to data directory.
    """
    project_root = Path(__file__).parent.parent.parent
    config = load_config()

    if data_type == "raw":
        return project_root / config["paths"]["data_raw"]
    elif data_type == "processed":
        return project_root / config["paths"]["data_processed"]
    elif data_type == "external":
        return project_root / config["paths"]["data_external"]
    else:
        raise ValueError(f"Invalid data type: {data_type}")


def get_models_path() -> Path:
    """
    Get path to models directory.

    Returns:
        Path to models directory.
    """
    project_root = Path(__file__).parent.parent.parent
    config = load_config()
    return project_root / config["paths"]["models"]
