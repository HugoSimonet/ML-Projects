"""
Time Series Forecasting Trainer
Comprehensive training pipeline for time series Transformer models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time


class ForecastingTrainer:
    """
    Trainer for time series forecasting models
    Supports multiple loss functions, learning rate scheduling, and early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: str = 'mse',
        optimizer: str = 'adam',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = './checkpoints',
        use_amp: bool = False
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function ('mse', 'mae', 'huber', 'quantile')
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            device: Device to train on
            save_dir: Directory to save checkpoints
            use_amp: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp

        # Loss function
        self.criterion = self._get_criterion(criterion)

        # Optimizer
        if optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        self.best_val_loss = float('inf')
        self.epochs_trained = 0

    def _get_criterion(self, criterion: str) -> Callable:
        """Get loss function"""
        if criterion == 'mse':
            return nn.MSELoss()
        elif criterion == 'mae':
            return nn.L1Loss()
        elif criterion == 'huber':
            return nn.HuberLoss()
        elif criterion == 'smooth_l1':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
            x_enc = x_enc.to(self.device)
            x_mark_enc = x_mark_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            x_mark_dec = x_mark_dec.to(self.device)
            y = y.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    loss = self.criterion(outputs, y)

                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = self.criterion(outputs, y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """
        Validate the model

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Unpack batch
                x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
                x_enc = x_enc.to(self.device)
                x_mark_enc = x_mark_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                x_mark_dec = x_mark_dec.to(self.device)
                y = y.to(self.device)

                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                        loss = self.criterion(outputs, y)
                else:
                    outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    loss = self.criterion(outputs, y)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(
        self,
        epochs: int,
        scheduler: Optional[object] = None,
        early_stopping_patience: int = 10,
        gradient_clip: Optional[float] = None,
        log_interval: int = 1,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """
        Main training loop

        Args:
            epochs: Number of epochs to train
            scheduler: Learning rate scheduler
            early_stopping_patience: Epochs to wait before early stopping
            gradient_clip: Gradient clipping value
            log_interval: Interval for logging
            save_best: Save best model based on validation loss

        Returns:
            Training history dictionary
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Logging
            if epoch % log_interval == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch}/{epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  LR: {current_lr:.6f}")
                print(f"  Time: {epoch_time:.2f}s")

            # Save best model
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, val_loss)
                print(f"  Saved best model with val_loss: {val_loss:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

            self.epochs_trained = epoch

        # Save final model
        self.save_checkpoint('final_model.pth', epochs, val_loss)

        return self.history

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_loss: float
    ):
        """
        Save model checkpoint

        Args:
            filename: Checkpoint filename
            epoch: Current epoch
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.save_dir / filename)

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint

        Args:
            filename: Checkpoint filename
        """
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.history = checkpoint.get('history', self.history)
        self.epochs_trained = checkpoint.get('epoch', 0)

        print(f"Loaded checkpoint from epoch {self.epochs_trained}")


class QuantileTrainer(ForecastingTrainer):
    """
    Trainer for quantile regression models
    Uses quantile loss for probabilistic forecasting
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        **kwargs
    ):
        """
        Args:
            model: Quantile forecasting model
            train_loader: Training data loader
            val_loader: Validation data loader
            quantiles: List of quantiles to predict
            **kwargs: Additional arguments for ForecastingTrainer
        """
        super().__init__(model, train_loader, val_loader, **kwargs)
        self.quantiles = quantiles

    def quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantile: float
    ) -> torch.Tensor:
        """
        Quantile loss (pinball loss)

        Args:
            predictions: Predicted values
            targets: Target values
            quantile: Quantile level

        Returns:
            Quantile loss
        """
        errors = targets - predictions
        loss = torch.max((quantile - 1) * errors, quantile * errors)
        return torch.mean(loss)

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with quantile loss"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
            x_enc = x_enc.to(self.device)
            x_mark_enc = x_mark_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            x_mark_dec = x_mark_dec.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Get quantile predictions [B, L, C, Q]
            outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # Calculate loss for each quantile
            loss = 0
            for i, q in enumerate(self.quantiles):
                q_loss = self.quantile_loss(outputs[..., i], y, q)
                loss += q_loss

            loss = loss / len(self.quantiles)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches


class DistributionalTrainer(ForecastingTrainer):
    """
    Trainer for distributional forecasting models
    Uses negative log-likelihood for training
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        distribution: str = 'gaussian',
        **kwargs
    ):
        """
        Args:
            model: Distributional forecasting model
            train_loader: Training data loader
            val_loader: Validation data loader
            distribution: Distribution type ('gaussian', 'student', 'negative_binomial')
            **kwargs: Additional arguments for ForecastingTrainer
        """
        super().__init__(model, train_loader, val_loader, **kwargs)
        self.distribution = distribution

    def nll_loss(
        self,
        params: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative log-likelihood loss

        Args:
            params: Distribution parameters
            targets: Target values

        Returns:
            NLL loss
        """
        if self.distribution == 'gaussian':
            mean = params['mean']
            std = params['std']
            dist = torch.distributions.Normal(mean, std)

        elif self.distribution == 'student':
            mean = params['mean']
            scale = params['scale']
            df = params['df']
            dist = torch.distributions.StudentT(df, mean, scale)

        elif self.distribution == 'negative_binomial':
            total_count = params['total_count']
            logits = params['logits']
            dist = torch.distributions.NegativeBinomial(total_count, logits=logits)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        # Negative log-likelihood
        nll = -dist.log_prob(targets)
        return torch.mean(nll)

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with NLL loss"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
            x_enc = x_enc.to(self.device)
            x_mark_enc = x_mark_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            x_mark_dec = x_mark_dec.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Get distribution parameters
            params = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # Calculate NLL loss
            loss = self.nll_loss(params, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches
