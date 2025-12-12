"""
Main training script for federated learning
Coordinates server, clients, and training process
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import argparse
import logging
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import FLConfig, AggregationAlgorithm, DatasetType
from models import create_model
from server import CentralServer
from client import FederatedClient, FedProxClient, FedNovaClient
from algorithms import FedAvgAggregator, FedProxAggregator, FedNovaAggregator
from utils import NonIIDDataSplitter
from privacy import PrivacyEngine
from evaluation import FederatedMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('federated_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_dataset(config: FLConfig):
    """
    Load and prepare dataset

    Args:
        config: FL configuration

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    logger.info(f"Loading dataset: {config.data.dataset.value}")

    if config.data.dataset == DatasetType.MNIST:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(
            config.client.data_path,
            train=True,
            download=config.data.download,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            config.client.data_path,
            train=False,
            download=config.data.download,
            transform=transform
        )

    elif config.data.dataset == DatasetType.CIFAR10:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = datasets.CIFAR10(
            config.client.data_path,
            train=True,
            download=config.data.download,
            transform=transform_train
        )

        test_dataset = datasets.CIFAR10(
            config.client.data_path,
            train=False,
            download=config.data.download,
            transform=transform_test
        )

    else:
        raise ValueError(f"Unsupported dataset: {config.data.dataset}")

    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")

    return train_dataset, test_dataset


def create_clients(config: FLConfig, train_datasets, model_template):
    """
    Create federated clients

    Args:
        config: FL configuration
        train_datasets: List of client training datasets
        model_template: Model architecture template

    Returns:
        List of FederatedClient objects
    """
    logger.info(f"Creating {len(train_datasets)} federated clients...")

    clients = []

    for client_id, train_data in enumerate(train_datasets):
        # Create model copy
        model = create_model(
            config.model.model_name,
            num_classes=config.model.num_classes,
            input_channels=config.model.input_channels,
            hidden_dims=config.model.hidden_dims
        )

        # Create appropriate client type
        if config.server.aggregation_algorithm == AggregationAlgorithm.FEDPROX:
            client = FedProxClient(
                client_id=client_id,
                model=model,
                train_data=train_data,
                device=config.client.device,
                local_epochs=config.client.local_epochs,
                batch_size=config.client.batch_size,
                learning_rate=config.client.learning_rate,
                optimizer_type=config.client.optimizer,
                momentum=config.client.momentum,
                weight_decay=config.client.weight_decay,
                mu=config.training.fedprox_mu
            )
        elif config.server.aggregation_algorithm == AggregationAlgorithm.FEDNOVA:
            client = FedNovaClient(
                client_id=client_id,
                model=model,
                train_data=train_data,
                device=config.client.device,
                local_epochs=config.client.local_epochs,
                batch_size=config.client.batch_size,
                learning_rate=config.client.learning_rate,
                optimizer_type=config.client.optimizer,
                momentum=config.client.momentum,
                weight_decay=config.client.weight_decay
            )
        else:  # FedAvg or default
            client = FederatedClient(
                client_id=client_id,
                model=model,
                train_data=train_data,
                device=config.client.device,
                local_epochs=config.client.local_epochs,
                batch_size=config.client.batch_size,
                learning_rate=config.client.learning_rate,
                optimizer_type=config.client.optimizer,
                momentum=config.client.momentum,
                weight_decay=config.client.weight_decay
            )

        clients.append(client)

    logger.info(f"Created {len(clients)} clients successfully")

    return clients


def train_federated(config: FLConfig):
    """
    Main federated training function

    Args:
        config: FL configuration
    """
    logger.info("=" * 80)
    logger.info("FEDERATED LEARNING TRAINING")
    logger.info("=" * 80)

    # Validate configuration
    config.validate()

    # Load datasets
    train_dataset, test_dataset = load_dataset(config)

    # Split data across clients
    logger.info(f"Splitting data across clients (strategy: {'IID' if config.data.iid else 'non-IID'})...")
    data_splitter = NonIIDDataSplitter(
        dataset=train_dataset,
        num_clients=config.data.num_clients,
        strategy='iid' if config.data.iid else 'dirichlet',
        alpha=config.data.alpha,
        seed=42
    )
    client_datasets = data_splitter.split()

    # Create global model
    logger.info(f"Creating global model: {config.model.model_name}...")
    global_model = create_model(
        config.model.model_name,
        num_classes=config.model.num_classes,
        input_channels=config.model.input_channels,
        hidden_dims=config.model.hidden_dims
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    # Create aggregator
    logger.info(f"Creating aggregator: {config.server.aggregation_algorithm.value}...")
    if config.server.aggregation_algorithm == AggregationAlgorithm.FEDAVG:
        aggregator = FedAvgAggregator(device=config.client.device)
    elif config.server.aggregation_algorithm == AggregationAlgorithm.FEDPROX:
        aggregator = FedProxAggregator(
            device=config.client.device,
            mu=config.training.fedprox_mu
        )
    elif config.server.aggregation_algorithm == AggregationAlgorithm.FEDNOVA:
        aggregator = FedNovaAggregator(
            device=config.client.device,
            tau_eff_default=config.training.fednova_tau_eff
        )
    else:
        raise ValueError(f"Unknown aggregation algorithm: {config.server.aggregation_algorithm}")

    # Create privacy engine if enabled
    privacy_engine = None
    if config.privacy.enable_privacy:
        logger.info(f"Enabling differential privacy: ε={config.privacy.epsilon}, δ={config.privacy.delta}")
        privacy_engine = PrivacyEngine(
            epsilon=config.privacy.epsilon,
            delta=config.privacy.delta,
            max_grad_norm=config.privacy.max_grad_norm,
            noise_multiplier=config.privacy.noise_multiplier,
            sample_rate=config.server.max_clients_per_round / config.server.num_clients
        )

    # Create central server
    server = CentralServer(
        model=global_model,
        aggregator=aggregator,
        num_clients=config.server.num_clients,
        device=config.client.device,
        min_clients_per_round=config.server.min_clients_per_round,
        max_clients_per_round=config.server.max_clients_per_round,
        client_selection_strategy=config.server.client_selection_strategy.value,
        model_save_path=config.server.model_save_path,
        checkpoint_frequency=config.server.checkpoint_frequency,
        privacy_engine=privacy_engine
    )

    # Set test data for server evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.client.batch_size,
        shuffle=False
    )
    server.set_test_data(test_loader)

    # Create clients
    clients = create_clients(config, client_datasets, global_model)

    # Create metrics tracker
    metrics = FederatedMetrics()

    # Initialize TensorBoard
    log_dir = os.path.join('logs', f'{config.data.dataset.value}_{config.server.aggregation_algorithm.value}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logging to: {log_dir}")

    # Training loop
    logger.info("=" * 80)
    logger.info("STARTING FEDERATED TRAINING")
    logger.info("=" * 80)

    best_accuracy = 0.0

    for round_num in range(config.training.num_rounds):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"ROUND {round_num + 1}/{config.training.num_rounds}")
        logger.info(f"{'=' * 80}")

        # Execute federated round
        round_stats = server.federated_round(clients)

        # Update metrics
        global_metrics = {
            'test_accuracy': round_stats['global_test_accuracy'],
            'test_loss': round_stats['global_test_loss']
        }

        metrics.update(
            round_num=round_num,
            global_metrics=global_metrics,
            client_metrics=round_stats.get('client_metrics'),
            privacy_budget=(
                round_stats['privacy']['current_epsilon'],
                round_stats['privacy']['current_delta']
            ) if 'privacy' in round_stats else None
        )

        # Log to TensorBoard
        writer.add_scalar('Global/Test_Accuracy', round_stats['global_test_accuracy'], round_num)
        writer.add_scalar('Global/Test_Loss', round_stats['global_test_loss'], round_num)
        writer.add_scalar('Training/Num_Selected_Clients', round_stats.get('num_selected_clients', 0), round_num)

        # Log privacy budget if enabled
        if 'privacy' in round_stats:
            writer.add_scalar('Privacy/Epsilon', round_stats['privacy']['current_epsilon'], round_num)
            writer.add_scalar('Privacy/Budget_Remaining', round_stats['privacy']['budget_remaining'], round_num)

        # Log client-level metrics
        if 'client_metrics' in round_stats and round_stats['client_metrics']:
            avg_client_loss = sum(m['loss'] for m in round_stats['client_metrics']) / len(round_stats['client_metrics'])
            writer.add_scalar('Clients/Average_Loss', avg_client_loss, round_num)

        # Track best accuracy
        if round_stats['global_test_accuracy'] > best_accuracy:
            best_accuracy = round_stats['global_test_accuracy']
            logger.info(f"New best accuracy: {best_accuracy:.2f}%")

        # Early stopping
        if config.training.early_stopping:
            if config.training.target_accuracy and round_stats['global_test_accuracy'] >= config.training.target_accuracy:
                logger.info(f"Target accuracy reached: {round_stats['global_test_accuracy']:.2f}%")
                break

        # Check privacy budget
        if privacy_engine and not privacy_engine.check_privacy_budget():
            logger.warning("Privacy budget exceeded, stopping training")
            break

    # Training complete
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)

    # Save final model
    server.save_checkpoint("final_model.pt")
    server.save_training_log()

    # Print metrics summary
    metrics.print_summary()

    # Print training summary
    summary = server.get_training_summary()
    logger.info("\nTraining Summary:")
    for key, value in summary.items():
        if key != 'client_participation':
            logger.info(f"  {key}: {value}")

    # Log final summary to TensorBoard
    writer.add_hparams(
        {
            'num_clients': config.server.num_clients,
            'num_rounds': config.training.num_rounds,
            'local_epochs': config.client.local_epochs,
            'algorithm': config.server.aggregation_algorithm.value,
            'learning_rate': config.client.learning_rate,
            'batch_size': config.client.batch_size
        },
        {
            'best_accuracy': best_accuracy,
            'final_accuracy': summary.get('final_accuracy', 0.0)
        }
    )

    # Close TensorBoard writer
    writer.close()
    logger.info("TensorBoard logs saved")

    return server, clients, metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--num_clients', type=int, default=10,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of training rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Local training epochs per round')
    parser.add_argument('--algorithm', type=str, default='fedavg',
                       choices=['fedavg', 'fedprox', 'fednova'],
                       help='Aggregation algorithm')
    parser.add_argument('--privacy', action='store_true',
                       help='Enable differential privacy')
    parser.add_argument('--non_iid', action='store_true',
                       help='Use non-IID data distribution')

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = FLConfig.from_yaml(args.config)
    else:
        logger.info("Creating default configuration")
        config = FLConfig()

        # Override with command-line arguments
        config.data.dataset = DatasetType(args.dataset)
        config.server.num_clients = args.num_clients
        config.data.num_clients = args.num_clients
        config.training.num_rounds = args.num_rounds
        config.client.local_epochs = args.local_epochs
        config.server.aggregation_algorithm = AggregationAlgorithm(args.algorithm)
        config.privacy.enable_privacy = args.privacy
        config.data.iid = not args.non_iid

        # Adjust client selection based on num_clients
        if args.num_clients < config.server.max_clients_per_round:
            config.server.max_clients_per_round = args.num_clients
        if args.num_clients < config.server.min_clients_per_round:
            config.server.min_clients_per_round = max(1, args.num_clients // 2)

        # Adjust model based on dataset
        if args.dataset == 'mnist':
            config.model.model_name = 'mnist_net'
            config.model.input_channels = 1
            config.model.num_classes = 10
        elif args.dataset == 'cifar10':
            config.model.model_name = 'cifar10_net'
            config.model.input_channels = 3
            config.model.num_classes = 10

    # Run training
    server, clients, metrics = train_federated(config)

    logger.info("\nFederated learning completed successfully!")


if __name__ == "__main__":
    main()
