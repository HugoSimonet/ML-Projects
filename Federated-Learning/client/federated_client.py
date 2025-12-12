"""
Federated Learning Client Implementation
Handles local training, model updates, and communication with server
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Tuple, Callable
import logging
import copy
from collections import OrderedDict

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Base federated learning client

    Handles local training and model updates
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: Dataset,
        test_data: Optional[Dataset] = None,
        device: str = 'cpu',
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer_type: str = 'sgd',
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        criterion: Optional[nn.Module] = None
    ):
        """
        Initialize federated client

        Args:
            client_id: Unique client identifier
            model: Neural network model
            train_data: Training dataset
            test_data: Test dataset (optional)
            device: Computing device
            local_epochs: Number of local training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            optimizer_type: Optimizer ('sgd', 'adam', 'adamw')
            momentum: Momentum for SGD
            weight_decay: Weight decay
            criterion: Loss function
        """
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Model
        self.model = model.to(device)
        self.global_model_state = None

        # Data
        self.train_data = train_data
        self.test_data = test_data
        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        if test_data:
            self.test_loader = DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False
            )

        # Optimizer
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = self._create_optimizer()

        # Loss function
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()

        # Training stats
        self.num_samples = len(train_data)
        self.training_history = []
        self.local_steps = 0

        logger.info(f"Initialized client {client_id}: "
                   f"{self.num_samples} samples, "
                   f"epochs={local_epochs}, "
                   f"batch_size={batch_size}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.optimizer_type.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

    def set_model_parameters(self, model_state: Dict[str, torch.Tensor]):
        """
        Update local model with global model parameters

        Args:
            model_state: Global model state dict
        """
        self.model.load_state_dict(model_state)
        self.global_model_state = copy.deepcopy(model_state)
        logger.debug(f"Client {self.client_id}: Updated model parameters")

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters"""
        return self.model.state_dict()

    def train(
        self,
        global_model_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Perform local training

        Args:
            global_model_state: Global model parameters (if provided)

        Returns:
            Tuple of (updated model state, training metrics)
        """
        # Update model if global state provided
        if global_model_state is not None:
            self.set_model_parameters(global_model_state)

        self.model.train()
        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(self.local_epochs):
            batch_losses = []
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics
                batch_losses.append(loss.item())
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                self.local_steps += 1

            # Epoch statistics
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_acc = 100. * correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_acc)

            logger.debug(f"Client {self.client_id} Epoch {epoch}: "
                        f"Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

        # Training metrics
        metrics = {
            'loss': sum(epoch_losses) / len(epoch_losses),
            'accuracy': sum(epoch_accuracies) / len(epoch_accuracies),
            'num_samples': self.num_samples,
            'local_steps': self.local_steps
        }

        self.training_history.append(metrics)

        return self.get_model_parameters(), metrics

    def evaluate(
        self,
        data_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """
        Evaluate model

        Args:
            data_loader: DataLoader to use (uses test_loader if None)

        Returns:
            Evaluation metrics
        """
        if data_loader is None:
            if self.test_data is None:
                logger.warning(f"Client {self.client_id}: No test data available")
                return {}
            data_loader = self.test_loader

        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        metrics = {
            'test_loss': test_loss / len(data_loader),
            'test_accuracy': 100. * correct / total,
            'num_samples': total
        }

        logger.info(f"Client {self.client_id} Test: "
                   f"Loss={metrics['test_loss']:.4f}, "
                   f"Acc={metrics['test_accuracy']:.2f}%")

        return metrics

    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """
        Get model update (difference from global model)

        Returns:
            Model update (delta)
        """
        if self.global_model_state is None:
            return self.get_model_parameters()

        current_state = self.get_model_parameters()
        update = OrderedDict()

        for key in current_state.keys():
            update[key] = current_state[key] - self.global_model_state[key]

        return update


class FedProxClient(FederatedClient):
    """
    FedProx client with proximal term

    Adds proximal regularization to handle heterogeneous systems
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: Dataset,
        mu: float = 0.01,
        **kwargs
    ):
        """
        Initialize FedProx client

        Args:
            client_id: Client identifier
            model: Neural network model
            train_data: Training dataset
            mu: Proximal term coefficient
            **kwargs: Additional arguments for FederatedClient
        """
        super().__init__(client_id, model, train_data, **kwargs)
        self.mu = mu
        logger.info(f"Initialized FedProx client {client_id} with mu={mu}")

    def train(
        self,
        global_model_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Perform local training with proximal term

        Args:
            global_model_state: Global model parameters

        Returns:
            Tuple of (updated model state, training metrics)
        """
        # Update model if global state provided
        if global_model_state is not None:
            self.set_model_parameters(global_model_state)

        # Keep copy of global model for proximal term
        global_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        self.model.train()
        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(self.local_epochs):
            batch_losses = []
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Add proximal term: (mu/2) * ||w - w_global||^2
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    proximal_term += torch.sum((param - global_params[name]) ** 2)
                loss += (self.mu / 2) * proximal_term

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics (without proximal term for logging)
                with torch.no_grad():
                    output_eval = self.model(data)
                    eval_loss = self.criterion(output_eval, target)
                    batch_losses.append(eval_loss.item())
                    _, predicted = output_eval.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                self.local_steps += 1

            # Epoch statistics
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_acc = 100. * correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_acc)

            logger.debug(f"FedProx Client {self.client_id} Epoch {epoch}: "
                        f"Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

        # Training metrics
        metrics = {
            'loss': sum(epoch_losses) / len(epoch_losses),
            'accuracy': sum(epoch_accuracies) / len(epoch_accuracies),
            'num_samples': self.num_samples,
            'local_steps': self.local_steps,
            'mu': self.mu
        }

        self.training_history.append(metrics)

        return self.get_model_parameters(), metrics


class FedNovaClient(FederatedClient):
    """
    FedNova client with normalized updates

    Tracks effective local steps for normalized aggregation
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: Dataset,
        **kwargs
    ):
        """
        Initialize FedNova client

        Args:
            client_id: Client identifier
            model: Neural network model
            train_data: Training dataset
            **kwargs: Additional arguments for FederatedClient
        """
        super().__init__(client_id, model, train_data, **kwargs)
        self.tau_eff = 0.0
        logger.info(f"Initialized FedNova client {client_id}")

    def train(
        self,
        global_model_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Perform local training and compute tau_eff

        Args:
            global_model_state: Global model parameters

        Returns:
            Tuple of (updated model state, training metrics including tau_eff)
        """
        # Standard training
        model_state, metrics = super().train(global_model_state)

        # Compute effective local steps
        # For SGD: tau_eff = local_epochs * batches_per_epoch
        batches_per_epoch = len(self.train_loader)
        total_steps = self.local_epochs * batches_per_epoch

        # Account for momentum if using SGD
        if self.optimizer_type.lower() == 'sgd' and self.momentum > 0:
            # Simplified tau_eff computation
            self.tau_eff = (1 - self.momentum ** total_steps) / (1 - self.momentum)
        else:
            self.tau_eff = float(total_steps)

        metrics['tau_eff'] = self.tau_eff

        return model_state, metrics


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

    # Create dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return torch.randn(10), torch.randint(0, 2, (1,)).item()

    # Test FederatedClient
    model = SimpleModel()
    train_data = DummyDataset(100)
    test_data = DummyDataset(20)

    client = FederatedClient(
        client_id=0,
        model=model,
        train_data=train_data,
        test_data=test_data,
        local_epochs=3,
        batch_size=16
    )

    print(f"\nTraining client {client.client_id}...")
    model_state, metrics = client.train()
    print(f"Training metrics: {metrics}")

    # Evaluate
    eval_metrics = client.evaluate()
    print(f"Evaluation metrics: {eval_metrics}")

    # Test FedProx client
    fedprox_model = SimpleModel()
    fedprox_client = FedProxClient(
        client_id=1,
        model=fedprox_model,
        train_data=train_data,
        mu=0.01,
        local_epochs=3
    )

    print(f"\nTraining FedProx client {fedprox_client.client_id}...")
    model_state, metrics = fedprox_client.train()
    print(f"FedProx training metrics: {metrics}")
