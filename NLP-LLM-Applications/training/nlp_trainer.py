"""
Training Pipeline for NLP Large Language Models.

This module provides a comprehensive training framework for NLP models:
- Training loop with validation
- Learning rate scheduling
- Gradient accumulation
- Mixed precision training
- Model checkpointing
- Early stopping
- Progress tracking and logging
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from abc import ABC, abstractmethod


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training parameters
    num_epochs: int = 10
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0

    # Optimization
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Learning rate scheduling
    lr_scheduler: Optional[str] = "linear"  # "linear", "cosine", "constant", None

    # Mixed precision training
    fp16: bool = False
    fp16_opt_level: str = "O1"

    # Logging and checkpointing
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    output_dir: str = "./output"

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0

    # Evaluation
    eval_strategy: str = "steps"  # "steps", "epoch", "no"
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

    # Other
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_num_workers: int = 0

    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class TrainingCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_train_begin(self, trainer: 'NLPTrainer'):
        """Called at the beginning of training."""
        pass

    @abstractmethod
    def on_train_end(self, trainer: 'NLPTrainer'):
        """Called at the end of training."""
        pass

    @abstractmethod
    def on_epoch_begin(self, trainer: 'NLPTrainer', epoch: int):
        """Called at the beginning of each epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: 'NLPTrainer', epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_step_begin(self, trainer: 'NLPTrainer', step: int):
        """Called at the beginning of each step."""
        pass

    @abstractmethod
    def on_step_end(self, trainer: 'NLPTrainer', step: int, loss: float):
        """Called at the end of each step."""
        pass


class EarlyStopping(TrainingCallback):
    """Early stopping to stop training when metric stops improving."""

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.0,
        mode: str = "min"
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            threshold: Minimum change to qualify as improvement
            mode: "min" or "max" for metric
        """
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def on_train_begin(self, trainer: 'NLPTrainer'):
        """Reset state at beginning of training."""
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def on_train_end(self, trainer: 'NLPTrainer'):
        """Nothing to do at end of training."""
        pass

    def on_epoch_begin(self, trainer: 'NLPTrainer', epoch: int):
        """Nothing to do at beginning of epoch."""
        pass

    def on_epoch_end(self, trainer: 'NLPTrainer', epoch: int, metrics: Dict[str, float]):
        """Check if should stop training."""
        metric_name = trainer.config.metric_for_best_model
        if metric_name not in metrics:
            return

        score = metrics[metric_name]

        if self.best_score is None:
            self.best_score = score
        else:
            if self.mode == "min":
                improvement = self.best_score - score
            else:
                improvement = score - self.best_score

            if improvement > self.threshold:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
                    print(f"Early stopping triggered after {epoch + 1} epochs")

    def on_step_begin(self, trainer: 'NLPTrainer', step: int):
        """Nothing to do at beginning of step."""
        pass

    def on_step_end(self, trainer: 'NLPTrainer', step: int, loss: float):
        """Nothing to do at end of step."""
        pass


class ModelCheckpoint(TrainingCallback):
    """Save model checkpoints during training."""

    def __init__(
        self,
        output_dir: str,
        save_best_only: bool = True,
        mode: str = "min"
    ):
        """
        Initialize model checkpoint.

        Args:
            output_dir: Directory to save checkpoints
            save_best_only: Only save when metric improves
            mode: "min" or "max" for metric
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_score = None

    def on_train_begin(self, trainer: 'NLPTrainer'):
        """Reset best score."""
        self.best_score = None

    def on_train_end(self, trainer: 'NLPTrainer'):
        """Save final model."""
        self._save_checkpoint(trainer, "final")

    def on_epoch_begin(self, trainer: 'NLPTrainer', epoch: int):
        """Nothing to do at beginning of epoch."""
        pass

    def on_epoch_end(self, trainer: 'NLPTrainer', epoch: int, metrics: Dict[str, float]):
        """Save checkpoint if metric improved."""
        metric_name = trainer.config.metric_for_best_model

        if metric_name in metrics:
            score = metrics[metric_name]
            should_save = False

            if not self.save_best_only:
                should_save = True
            elif self.best_score is None:
                should_save = True
                self.best_score = score
            else:
                if self.mode == "min" and score < self.best_score:
                    should_save = True
                    self.best_score = score
                elif self.mode == "max" and score > self.best_score:
                    should_save = True
                    self.best_score = score

            if should_save:
                self._save_checkpoint(trainer, f"epoch_{epoch + 1}")

    def on_step_begin(self, trainer: 'NLPTrainer', step: int):
        """Nothing to do at beginning of step."""
        pass

    def on_step_end(self, trainer: 'NLPTrainer', step: int, loss: float):
        """Nothing to do at end of step."""
        pass

    def _save_checkpoint(self, trainer: 'NLPTrainer', name: str):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_{name}.pt"

        checkpoint = {
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'global_step': trainer.global_step,
            'config': trainer.config
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


class LearningRateScheduler(TrainingCallback):
    """Learning rate scheduling callback."""

    def __init__(self, scheduler):
        """
        Initialize LR scheduler.

        Args:
            scheduler: PyTorch learning rate scheduler
        """
        self.scheduler = scheduler

    def on_train_begin(self, trainer: 'NLPTrainer'):
        """Nothing to do at beginning of training."""
        pass

    def on_train_end(self, trainer: 'NLPTrainer'):
        """Nothing to do at end of training."""
        pass

    def on_epoch_begin(self, trainer: 'NLPTrainer', epoch: int):
        """Nothing to do at beginning of epoch."""
        pass

    def on_epoch_end(self, trainer: 'NLPTrainer', epoch: int, metrics: Dict[str, float]):
        """Step the scheduler."""
        if hasattr(self.scheduler, 'step'):
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau needs a metric
                metric_name = trainer.config.metric_for_best_model
                if metric_name in metrics:
                    self.scheduler.step(metrics[metric_name])
            else:
                self.scheduler.step()

    def on_step_begin(self, trainer: 'NLPTrainer', step: int):
        """Nothing to do at beginning of step."""
        pass

    def on_step_end(self, trainer: 'NLPTrainer', step: int, loss: float):
        """Step scheduler if it's a per-step scheduler."""
        if hasattr(self.scheduler, 'step_update'):
            self.scheduler.step_update(step)


class NLPTrainer:
    """
    Comprehensive trainer for NLP models.

    Features:
    - Training and validation loops
    - Gradient accumulation
    - Mixed precision training
    - Learning rate scheduling
    - Checkpointing
    - Progress tracking
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        compute_metrics: Optional[Callable] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            optimizer: Optimizer (created if None)
            callbacks: List of training callbacks
            compute_metrics: Function to compute metrics
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks or []
        self.compute_metrics = compute_metrics

        # Move model to device
        self.model.to(config.device)

        # Create optimizer
        self.optimizer = optimizer or self._create_optimizer()

        # Create learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.training_history = []

        # Set random seed
        self._set_seed(config.seed)

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer from config."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_lr_scheduler(self):
        """Create learning rate scheduler from config."""
        if self.config.lr_scheduler is None or self.config.lr_scheduler == "constant":
            return None

        if self.train_dataloader is None:
            return None

        num_training_steps = len(self.train_dataloader) * self.config.num_epochs

        if self.config.lr_scheduler == "linear":
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(current_step: int):
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                return max(
                    0.0,
                    float(num_training_steps - current_step) /
                    float(max(1, num_training_steps - self.config.warmup_steps))
                )

            return LambdaLR(self.optimizer, lr_lambda)

        elif self.config.lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - self.config.warmup_steps
            )

        return None

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self) -> Dict[str, Any]:
        """
        Run training loop.

        Returns:
            Training history and metrics
        """
        # Callbacks: on_train_begin
        for callback in self.callbacks:
            callback.on_train_begin(self)

        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.config.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Callbacks: on_epoch_begin
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            # Train for one epoch
            train_metrics = self._train_epoch()

            # Evaluate
            eval_metrics = {}
            if self.eval_dataloader and self.config.eval_strategy == "epoch":
                eval_metrics = self.evaluate()

            # Combine metrics
            epoch_metrics = {**train_metrics, **eval_metrics}
            self.training_history.append(epoch_metrics)

            # Print progress
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Train loss: {train_metrics['train_loss']:.4f}")
            if eval_metrics:
                print(f"Eval loss: {eval_metrics.get('eval_loss', 0):.4f}")

            # Callbacks: on_epoch_end
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, epoch_metrics)

            # Check early stopping
            if any(isinstance(cb, EarlyStopping) and cb.should_stop for cb in self.callbacks):
                print("Early stopping triggered")
                break

        # Callbacks: on_train_end
        for callback in self.callbacks:
            callback.on_train_end(self)

        return {
            'history': self.training_history,
            'best_metric': self.best_metric
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for step, batch in enumerate(self.train_dataloader):
            # Callbacks: on_step_begin
            for callback in self.callbacks:
                callback.on_step_begin(self, self.global_step)

            # Move batch to device
            batch = self._prepare_batch(batch)

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # Gradient accumulation
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            self.global_step += 1

            # Callbacks: on_step_end
            for callback in self.callbacks:
                callback.on_step_end(self, self.global_step, loss.item())

            # Logging
            if self.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / num_batches
                print(f"Step {self.global_step}: loss = {avg_loss:.4f}")

            # Evaluation
            if (self.config.eval_strategy == "steps" and
                self.eval_dataloader and
                self.global_step % self.config.eval_steps == 0):
                eval_metrics = self.evaluate()
                print(f"Eval at step {self.global_step}: {eval_metrics}")
                self.model.train()

        return {'train_loss': total_loss / max(1, num_batches)}

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._prepare_batch(batch)

                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                total_loss += loss.item()
                num_batches += 1

                # Collect predictions and labels if compute_metrics is provided
                if self.compute_metrics:
                    if hasattr(outputs, 'logits'):
                        predictions = outputs.logits.argmax(dim=-1)
                        all_predictions.extend(predictions.cpu().numpy())

                    if 'labels' in batch:
                        all_labels.extend(batch['labels'].cpu().numpy())

        metrics = {'eval_loss': total_loss / max(1, num_batches)}

        # Compute custom metrics
        if self.compute_metrics and all_predictions and all_labels:
            custom_metrics = self.compute_metrics(all_predictions, all_labels)
            metrics.update(custom_metrics)

        return metrics

    def _prepare_batch(self, batch: Any) -> Any:
        """Move batch to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for v in batch]
        else:
            return batch.to(self.config.device) if isinstance(batch, torch.Tensor) else batch

    def save_model(self, path: str):
        """Save model to file."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from file."""
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def save_checkpoint(self, path: str):
        """Save full training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
            'training_history': self.training_history
        }

        if self.lr_scheduler:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load full training checkpoint."""
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', [])

        if self.lr_scheduler and 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from {path}")
