"""
Configuration Management
Handles model and training configurations
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = 'VanillaTransformer'  # 'VanillaTransformer', 'Informer', 'Autoformer'
    enc_in: int = 7  # Encoder input size
    dec_in: int = 7  # Decoder input size
    c_out: int = 7  # Output size
    seq_len: int = 96  # Input sequence length
    label_len: int = 48  # Start token length
    pred_len: int = 24  # Prediction length
    d_model: int = 512  # Model dimension
    n_heads: int = 8  # Number of attention heads
    e_layers: int = 2  # Number of encoder layers
    d_layers: int = 1  # Number of decoder layers
    d_ff: int = 2048  # Feed-forward dimension
    dropout: float = 0.1  # Dropout rate
    embed: str = 'fixed'  # Embedding type
    freq: str = 'h'  # Frequency
    activation: str = 'gelu'  # Activation function
    factor: int = 5  # ProbSparse attention factor (for Informer)
    distil: bool = True  # Use distilling (for Informer)
    moving_avg: int = 25  # Moving average window (for Autoformer)


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    criterion: str = 'mse'  # 'mse', 'mae', 'huber'
    scheduler: Optional[str] = 'cosine'  # 'cosine', 'step', 'plateau', None
    early_stopping_patience: int = 10
    gradient_clip: Optional[float] = 1.0
    use_amp: bool = False  # Automatic mixed precision
    num_workers: int = 4
    device: str = 'cuda'  # 'cuda' or 'cpu'
    gpu_id: Optional[int] = 0


@dataclass
class DataConfig:
    """Data configuration"""
    dataset: str = 'ETTh1'  # Dataset name
    root_path: str = './data/'  # Root path for datasets
    data_path: str = 'ETTh1.csv'  # Dataset file
    features: str = 'M'  # 'M': multivariate, 'S': univariate, 'MS': multivariate to univariate
    target: str = 'OT'  # Target column
    freq: str = 'h'  # Time frequency
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    scale: bool = True  # Scale data
    timeenc: int = 0  # Time encoding (0: raw, 1: continuous)


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    exp_name: str = 'exp_1'  # Experiment name
    seed: int = 42  # Random seed
    save_dir: str = './checkpoints/'  # Directory to save checkpoints
    log_dir: str = './logs/'  # Directory for logs
    result_dir: str = './results/'  # Directory for results
    save_best: bool = True  # Save best model
    save_last: bool = True  # Save last model
    log_interval: int = 1  # Logging interval (epochs)
    eval_interval: int = 1  # Evaluation interval (epochs)


@dataclass
class Config:
    """Complete configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file

        Args:
            yaml_path: Path to YAML file

        Returns:
            Config object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )

    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file

        Args:
            yaml_path: Path to save YAML file
        """
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def __str__(self) -> str:
        """String representation"""
        lines = ["=" * 80, "Configuration", "=" * 80]

        for section_name in ['model', 'training', 'data', 'experiment']:
            section = getattr(self, section_name)
            lines.append(f"\n{section_name.upper()}:")
            for key, value in section.__dict__.items():
                lines.append(f"  {key}: {value}")

        lines.append("=" * 80)
        return '\n'.join(lines)


def get_default_config(model_type: str = 'VanillaTransformer') -> Config:
    """
    Get default configuration for a model type

    Args:
        model_type: Model type

    Returns:
        Config object
    """
    config = Config()
    config.model.model_type = model_type

    # Model-specific defaults
    if model_type == 'Informer':
        config.model.e_layers = 3
        config.model.d_layers = 2
        config.model.d_ff = 512
        config.model.factor = 5
        config.model.distil = True

    elif model_type == 'Autoformer':
        config.model.e_layers = 2
        config.model.d_layers = 1
        config.model.moving_avg = 25

    return config


def create_config_template(save_path: str):
    """
    Create a template configuration file

    Args:
        save_path: Path to save template
    """
    config = Config()
    config.to_yaml(save_path)
    print(f"Configuration template saved to: {save_path}")
