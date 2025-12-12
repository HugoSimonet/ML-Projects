"""
Central Server for Federated Learning
Coordinates federated training, client selection, and model aggregation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
import logging
import os
import random
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CentralServer:
    """
    Central aggregation server for federated learning

    Manages global model, client selection, and aggregation
    """

    def __init__(
        self,
        model: nn.Module,
        aggregator,
        num_clients: int,
        device: str = 'cpu',
        min_clients_per_round: int = 5,
        max_clients_per_round: int = 10,
        client_selection_strategy: str = 'random',
        model_save_path: str = './checkpoints',
        checkpoint_frequency: int = 10,
        privacy_engine: Optional[any] = None
    ):
        """
        Initialize central server

        Args:
            model: Global model
            aggregator: Aggregation algorithm (FedAvg, FedProx, etc.)
            num_clients: Total number of clients
            device: Computing device
            min_clients_per_round: Minimum clients per round
            max_clients_per_round: Maximum clients per round
            client_selection_strategy: Client selection strategy
            model_save_path: Path to save model checkpoints
            checkpoint_frequency: Frequency of checkpoints (in rounds)
            privacy_engine: Optional privacy engine for DP
        """
        self.global_model = model.to(device)
        self.aggregator = aggregator
        self.num_clients = num_clients
        self.device = device
        self.min_clients_per_round = min_clients_per_round
        self.max_clients_per_round = max_clients_per_round
        self.client_selection_strategy = client_selection_strategy
        self.model_save_path = Path(model_save_path)
        self.checkpoint_frequency = checkpoint_frequency
        self.privacy_engine = privacy_engine

        # Create directories
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_round = 0
        self.round_history = []
        self.client_participation = {i: 0 for i in range(num_clients)}

        # Test data (optional)
        self.test_loader = None

        logger.info(f"Initialized Central Server: "
                   f"{num_clients} clients, "
                   f"device={device}, "
                   f"strategy={client_selection_strategy}")

    def set_test_data(self, test_loader: DataLoader):
        """Set test data for global evaluation"""
        self.test_loader = test_loader
        logger.info("Test data set for global evaluation")

    def get_global_model_state(self) -> Dict[str, torch.Tensor]:
        """Get global model state dict"""
        return self.global_model.state_dict()

    def select_clients(
        self,
        num_clients: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[int]:
        """
        Select clients for current round

        Args:
            num_clients: Number of clients to select (uses random if None)
            round_num: Current round number (for deterministic strategies)

        Returns:
            List of selected client IDs
        """
        if num_clients is None:
            num_clients = random.randint(
                self.min_clients_per_round,
                min(self.max_clients_per_round, self.num_clients)
            )

        if self.client_selection_strategy == 'random':
            selected = random.sample(range(self.num_clients), num_clients)

        elif self.client_selection_strategy == 'round_robin':
            # Deterministic round-robin selection
            if round_num is None:
                round_num = self.current_round
            start_idx = (round_num * num_clients) % self.num_clients
            selected = [
                (start_idx + i) % self.num_clients
                for i in range(num_clients)
            ]

        elif self.client_selection_strategy == 'uniform':
            # Ensure uniform participation over time
            # Select clients with lowest participation count
            participation_counts = [
                (client_id, self.client_participation[client_id])
                for client_id in range(self.num_clients)
            ]
            participation_counts.sort(key=lambda x: x[1])
            selected = [client_id for client_id, _ in participation_counts[:num_clients]]

        else:
            logger.warning(f"Unknown selection strategy: {self.client_selection_strategy}, "
                          f"using random")
            selected = random.sample(range(self.num_clients), num_clients)

        # Update participation
        for client_id in selected:
            self.client_participation[client_id] += 1

        logger.info(f"Round {self.current_round}: Selected {len(selected)} clients: {selected}")

        return selected

    def aggregate_updates(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates

        Args:
            client_updates: List of client model states
            client_weights: Optional weights for each client
            **kwargs: Additional aggregation parameters

        Returns:
            Aggregated global model state
        """
        if len(client_updates) == 0:
            logger.warning("No client updates to aggregate")
            return self.get_global_model_state()

        # Apply privacy if enabled
        if self.privacy_engine is not None:
            logger.info("Applying differential privacy to updates")
            client_updates = [
                self.privacy_engine.privatize_update(update)
                for update in client_updates
            ]

        # Aggregate
        aggregated_state = self.aggregator.aggregate(
            client_updates,
            client_weights,
            global_model=self.get_global_model_state(),
            **kwargs
        )

        return aggregated_state

    def update_global_model(self, aggregated_state: Dict[str, torch.Tensor]):
        """
        Update global model with aggregated state

        Args:
            aggregated_state: Aggregated model state
        """
        self.global_model.load_state_dict(aggregated_state)
        logger.debug("Updated global model")

    def federated_round(
        self,
        clients: List[any],
        selected_client_ids: Optional[List[int]] = None
    ) -> Dict[str, any]:
        """
        Execute one federated learning round

        Args:
            clients: List of FederatedClient objects
            selected_client_ids: Optional pre-selected client IDs

        Returns:
            Round statistics
        """
        # Select clients
        if selected_client_ids is None:
            selected_client_ids = self.select_clients()

        # Get global model state
        global_model_state = self.get_global_model_state()

        # Collect client updates
        client_updates = []
        client_weights = []
        client_metrics = []

        for client_id in selected_client_ids:
            client = clients[client_id]

            # Local training
            logger.info(f"Training client {client_id}...")
            model_state, metrics = client.train(global_model_state)

            client_updates.append(model_state)
            client_weights.append(metrics['num_samples'])
            client_metrics.append(metrics)

        # Aggregate updates
        logger.info("Aggregating client updates...")
        aggregated_state = self.aggregate_updates(
            client_updates,
            client_weights
        )

        # Update global model
        self.update_global_model(aggregated_state)

        # Evaluate global model
        eval_metrics = self.evaluate() if self.test_loader else {}

        # Compute round statistics
        round_stats = {
            'round': self.current_round,
            'num_clients': len(selected_client_ids),
            'selected_clients': selected_client_ids,
            'avg_train_loss': np.mean([m['loss'] for m in client_metrics]),
            'avg_train_accuracy': np.mean([m['accuracy'] for m in client_metrics]),
            'global_test_loss': eval_metrics.get('test_loss', 0),
            'global_test_accuracy': eval_metrics.get('test_accuracy', 0),
            'client_metrics': client_metrics
        }

        # Add privacy metrics if available
        if self.privacy_engine is not None:
            privacy_report = self.privacy_engine.get_privacy_report()
            round_stats['privacy'] = privacy_report

        self.round_history.append(round_stats)

        # Save checkpoint
        if (self.current_round + 1) % self.checkpoint_frequency == 0:
            self.save_checkpoint()

        self.current_round += 1

        logger.info(
            f"Round {round_stats['round']} complete: "
            f"Train Loss={round_stats['avg_train_loss']:.4f}, "
            f"Train Acc={round_stats['avg_train_accuracy']:.2f}%, "
            f"Test Loss={round_stats['global_test_loss']:.4f}, "
            f"Test Acc={round_stats['global_test_accuracy']:.2f}%"
        )

        return round_stats

    def evaluate(
        self,
        data_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Evaluate global model

        Args:
            data_loader: Test data loader (uses self.test_loader if None)
            criterion: Loss function (uses CrossEntropyLoss if None)

        Returns:
            Evaluation metrics
        """
        if data_loader is None:
            if self.test_loader is None:
                logger.warning("No test data available for evaluation")
                return {}
            data_loader = self.test_loader

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        metrics = {
            'test_loss': test_loss / len(data_loader),
            'test_accuracy': 100. * correct / total,
            'num_samples': total
        }

        logger.info(f"Global evaluation: "
                   f"Loss={metrics['test_loss']:.4f}, "
                   f"Acc={metrics['test_accuracy']:.2f}%")

        return metrics

    def save_checkpoint(self, filename: Optional[str] = None):
        """
        Save model checkpoint

        Args:
            filename: Optional filename (uses default if None)
        """
        if filename is None:
            filename = f"global_model_round_{self.current_round}.pt"

        filepath = self.model_save_path / filename

        checkpoint = {
            'round': self.current_round,
            'model_state_dict': self.global_model.state_dict(),
            'round_history': self.round_history,
            'client_participation': self.client_participation
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

        # Also save as latest
        latest_path = self.model_save_path / "latest_model.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint

        Args:
            filename: Checkpoint filename
        """
        filepath = self.model_save_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.current_round = checkpoint['round']
        self.round_history = checkpoint.get('round_history', [])
        self.client_participation = checkpoint.get('client_participation', {})

        logger.info(f"Loaded checkpoint from {filepath}, round {self.current_round}")

    def get_training_summary(self) -> Dict[str, any]:
        """Get summary of training progress"""
        if not self.round_history:
            return {}

        return {
            'total_rounds': len(self.round_history),
            'current_round': self.current_round,
            'final_train_accuracy': self.round_history[-1]['avg_train_accuracy'],
            'final_test_accuracy': self.round_history[-1]['global_test_accuracy'],
            'best_test_accuracy': max(
                r['global_test_accuracy'] for r in self.round_history
                if 'global_test_accuracy' in r
            ),
            'client_participation': self.client_participation
        }

    def save_training_log(self, filename: str = "training_log.json"):
        """Save training history to JSON"""
        filepath = self.model_save_path / filename

        log_data = {
            'summary': self.get_training_summary(),
            'round_history': self.round_history
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved training log to {filepath}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    # Create server
    from algorithms import FedAvgAggregator

    model = SimpleModel()
    aggregator = FedAvgAggregator()

    server = CentralServer(
        model=model,
        aggregator=aggregator,
        num_clients=10,
        min_clients_per_round=3,
        max_clients_per_round=5
    )

    print(f"Server initialized: {server.num_clients} clients")
    print(f"Global model parameters: {sum(p.numel() for p in server.global_model.parameters())}")

    # Test client selection
    selected = server.select_clients(num_clients=5)
    print(f"Selected clients: {selected}")
