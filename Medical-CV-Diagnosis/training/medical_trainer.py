"""
Medical Model Trainer
Comprehensive training infrastructure for medical AI models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Optional, Dict, Callable
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MedicalTrainer:
    """Trainer for medical classification/segmentation models"""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        metrics_calculator: Callable,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        mixed_precision: bool = False
    ):
        """
        Initialize trainer

        Args:
            model: Model to train
            criterion: Loss function
            optimizer: Optimizer
            device: Device (cuda/cpu)
            metrics_calculator: Metrics calculation function
            scheduler: Learning rate scheduler
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
            mixed_precision: Use mixed precision training
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics_calculator = metrics_calculator
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.mixed_precision = mixed_precision

        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_metric': []}

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        self.metrics_calculator.reset()

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                # Update metrics
                predictions = outputs.argmax(dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                self.metrics_calculator.update(predictions, targets, probabilities)

        avg_loss = total_loss / len(val_loader)
        metrics = self.metrics_calculator.compute()

        return {'val_loss': avg_loss, **metrics}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Train model for multiple epochs

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_best: Save best model
            early_stopping_patience: Early stopping patience
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Log metrics
            self.writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)

            for metric_name, value in val_metrics.items():
                if metric_name != 'val_loss':
                    self.writer.add_scalar(f'Metrics/{metric_name}', value, epoch)

            # Update history
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # Save best model
            current_metric = val_metrics.get('accuracy', val_metrics.get('mean_dice', 0.0))
            if save_best and current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint('best_model.pth')
                logger.info(f"Best model saved with metric: {self.best_metric:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Log epoch summary
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Metric: {current_metric:.4f}"
            )

        self.writer.close()
        logger.info("Training completed")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'training_history': self.training_history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.training_history = checkpoint.get('training_history', {})

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Checkpoint loaded: {checkpoint_path}")


class SegmentationTrainer(MedicalTrainer):
    """Specialized trainer for segmentation tasks"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train epoch for segmentation"""
        self.model.train()
        total_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f'Epoch {self.current_epoch}'):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return {'train_loss': total_loss / len(train_loader)}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate segmentation model"""
        self.model.eval()
        total_loss = 0.0
        self.metrics_calculator.reset()

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                total_loss += loss.item()

                # Update metrics
                predictions = outputs.argmax(dim=1)
                self.metrics_calculator.update(predictions, masks)

        avg_loss = total_loss / len(val_loader)
        metrics = self.metrics_calculator.compute()

        return {'val_loss': avg_loss, **metrics}
