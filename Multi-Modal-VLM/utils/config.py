"""
Configuration utilities
"""

import yaml
import json
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")

    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to file

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    with open(save_path, 'w') as f:
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        elif save_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {save_path}")


if __name__ == "__main__":
    # Test config loading
    config = load_config('configs/default_config.yaml')
    print("Config loaded successfully!")
    print(f"Model hidden dim: {config['model']['hidden_dim']}")
    print(f"Training batch size: {config['training']['batch_size']}")
