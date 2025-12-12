"""
Configuration Management for Medical CV System
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import yaml
from enum import Enum


class TaskType(Enum):
    """Task types"""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"


class Modality(Enum):
    """Medical imaging modalities"""
    CT = "ct"
    MRI = "mri"
    XRAY = "xray"
    ULTRASOUND = "ultrasound"
    MAMMOGRAPHY = "mammography"


@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "./data"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    image_size: tuple = (224, 224)
    normalize_method: str = "min_max"
    augmentation: bool = True
    balanced_sampling: bool = False


@dataclass
class ModelConfig:
    """Model configuration"""
    architecture: str = "resnet50"
    num_classes: int = 2
    in_channels: int = 1
    pretrained: bool = True
    dropout: float = 0.5
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    loss_function: str = "focal"
    scheduler: str = "reduce_on_plateau"
    mixed_precision: bool = True
    early_stopping_patience: int = 15
    save_best: bool = True


@dataclass
class UncertaintyConfig:
    """Uncertainty quantification configuration"""
    enable: bool = False
    method: str = "mc_dropout"
    num_samples: int = 30


@dataclass
class ExplainabilityConfig:
    """Explainability configuration"""
    enable: bool = False
    methods: List[str] = field(default_factory=lambda: ["gradcam"])
    save_visualizations: bool = True


@dataclass
class MedicalConfig:
    """Main medical CV configuration"""
    task: TaskType = TaskType.CLASSIFICATION
    modality: Modality = Modality.CT
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    output_dir: str = "./output"
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MedicalConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
