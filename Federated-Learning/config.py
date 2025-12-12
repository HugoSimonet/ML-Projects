"""
Configuration module for Federated Learning System
Handles all configuration parameters for server, client, privacy, and communication
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import yaml
import os


class AggregationAlgorithm(Enum):
    """Supported aggregation algorithms"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"


class DatasetType(Enum):
    """Supported datasets"""
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    FEMNIST = "femnist"
    SHAKESPEARE = "shakespeare"
    CUSTOM = "custom"


class ClientSelectionStrategy(Enum):
    """Client selection strategies"""
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    IMPORTANCE_BASED = "importance_based"


@dataclass
class PrivacyConfig:
    """Privacy configuration for differential privacy"""
    enable_privacy: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    secure_aggregation: bool = False
    privacy_accounting: bool = True

    def __post_init__(self):
        if self.enable_privacy and self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.enable_privacy and not (0 <= self.delta <= 1):
            raise ValueError("Delta must be in [0, 1]")


@dataclass
class CommunicationConfig:
    """Communication layer configuration"""
    protocol: str = "grpc"  # grpc, http, websocket
    compression_enabled: bool = True
    compression_ratio: float = 0.1
    quantization_bits: int = 8
    use_ssl: bool = True
    timeout_seconds: int = 300
    max_retries: int = 3
    buffer_size: int = 1024 * 1024  # 1MB


@dataclass
class ServerConfig:
    """Central server configuration"""
    host: str = "localhost"
    port: int = 8080
    num_clients: int = 10
    min_clients_per_round: int = 5
    max_clients_per_round: int = 10
    client_selection_strategy: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM
    aggregation_algorithm: AggregationAlgorithm = AggregationAlgorithm.FEDAVG
    model_save_path: str = "./checkpoints"
    log_path: str = "./logs"
    checkpoint_frequency: int = 10  # Save every N rounds


@dataclass
class ClientConfig:
    """Client-side configuration"""
    client_id: Optional[int] = None
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = "sgd"  # sgd, adam, adamw
    momentum: float = 0.9
    weight_decay: float = 1e-4
    device: str = "cpu"  # cpu, cuda
    data_path: str = "./data"
    local_model_save: bool = False


@dataclass
class DataConfig:
    """Data configuration"""
    dataset: DatasetType = DatasetType.MNIST
    num_clients: int = 10
    iid: bool = False
    alpha: float = 0.5  # Dirichlet distribution parameter for non-IID
    train_test_split: float = 0.8
    validation_split: float = 0.1
    custom_data_path: Optional[str] = None
    download: bool = True
    normalization: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_rounds: int = 100
    evaluation_frequency: int = 1
    early_stopping: bool = False
    early_stopping_patience: int = 10
    target_accuracy: Optional[float] = None
    fedprox_mu: float = 0.01  # FedProx proximal term
    fednova_tau_eff: float = 1.0  # FedNova normalization


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_name: str = "simple_cnn"
    num_classes: int = 10
    input_channels: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.5
    pretrained: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    enable_wandb: bool = False
    wandb_project: str = "federated-learning"
    wandb_entity: Optional[str] = None
    enable_tensorboard: bool = True
    log_level: str = "INFO"
    metrics_save_path: str = "./results"
    save_client_metrics: bool = True


@dataclass
class FLConfig:
    """Main Federated Learning configuration"""
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'FLConfig':
        """Load configuration from YAML file"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FLConfig':
        """Create configuration from dictionary"""
        # Convert string enums to enum objects
        if 'server' in config_dict:
            if 'client_selection_strategy' in config_dict['server']:
                config_dict['server']['client_selection_strategy'] = ClientSelectionStrategy(
                    config_dict['server']['client_selection_strategy']
                )
            if 'aggregation_algorithm' in config_dict['server']:
                config_dict['server']['aggregation_algorithm'] = AggregationAlgorithm(
                    config_dict['server']['aggregation_algorithm']
                )

        if 'data' in config_dict and 'dataset' in config_dict['data']:
            config_dict['data']['dataset'] = DatasetType(config_dict['data']['dataset'])

        # Create sub-configs
        server_config = ServerConfig(**config_dict.get('server', {}))
        client_config = ClientConfig(**config_dict.get('client', {}))
        privacy_config = PrivacyConfig(**config_dict.get('privacy', {}))
        comm_config = CommunicationConfig(**config_dict.get('communication', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))

        return cls(
            server=server_config,
            client=client_config,
            privacy=privacy_config,
            communication=comm_config,
            data=data_config,
            training=training_config,
            model=model_config,
            monitoring=monitoring_config
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}

        for field_name in ['server', 'client', 'privacy', 'communication',
                          'data', 'training', 'model', 'monitoring']:
            field_value = getattr(self, field_name)
            if hasattr(field_value, '__dict__'):
                result[field_name] = {}
                for key, value in field_value.__dict__.items():
                    if isinstance(value, Enum):
                        result[field_name][key] = value.value
                    else:
                        result[field_name][key] = value

        return result

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(yaml_path) if os.path.dirname(yaml_path) else '.', exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []

        # Validate server config
        if self.server.min_clients_per_round > self.server.max_clients_per_round:
            errors.append("min_clients_per_round cannot exceed max_clients_per_round")

        if self.server.max_clients_per_round > self.server.num_clients:
            errors.append("max_clients_per_round cannot exceed total num_clients")

        # Validate client config
        if self.client.local_epochs < 1:
            errors.append("local_epochs must be at least 1")

        if self.client.batch_size < 1:
            errors.append("batch_size must be at least 1")

        # Validate data config
        if not (0 < self.data.train_test_split < 1):
            errors.append("train_test_split must be in (0, 1)")

        if not (0 <= self.data.validation_split < 1):
            errors.append("validation_split must be in [0, 1)")

        # Validate training config
        if self.training.num_rounds < 1:
            errors.append("num_rounds must be at least 1")

        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

        return True


def create_default_config() -> FLConfig:
    """Create a default configuration for quick start"""
    return FLConfig()


def create_privacy_config() -> FLConfig:
    """Create a configuration with privacy enabled"""
    config = FLConfig()
    config.privacy.enable_privacy = True
    config.privacy.epsilon = 1.0
    config.privacy.delta = 1e-5
    config.privacy.secure_aggregation = True
    return config


def create_heterogeneous_config() -> FLConfig:
    """Create a configuration for heterogeneous (non-IID) data"""
    config = FLConfig()
    config.data.iid = False
    config.data.alpha = 0.5
    config.training.fedprox_mu = 0.01
    config.server.aggregation_algorithm = AggregationAlgorithm.FEDPROX
    return config


if __name__ == "__main__":
    # Example: Create and save default configurations
    configs_dir = "./configs"
    os.makedirs(configs_dir, exist_ok=True)

    # Basic FL config
    basic_config = create_default_config()
    basic_config.to_yaml(f"{configs_dir}/basic_fl.yaml")
    print(f"Saved basic configuration to {configs_dir}/basic_fl.yaml")

    # Privacy-preserving FL config
    privacy_config = create_privacy_config()
    privacy_config.to_yaml(f"{configs_dir}/dp_fl.yaml")
    print(f"Saved privacy configuration to {configs_dir}/dp_fl.yaml")

    # Heterogeneous FL config
    hetero_config = create_heterogeneous_config()
    hetero_config.to_yaml(f"{configs_dir}/heterogeneous_fl.yaml")
    print(f"Saved heterogeneous configuration to {configs_dir}/heterogeneous_fl.yaml")
