"""
Training pipeline for multi-modal vision-language models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
import os
import time
from tqdm import tqdm
import json
import logging

from .losses import CaptionLoss, VQALoss, RetrievalLoss, ContrastiveLoss, MultiTaskLoss
from .optimizers import create_optimizer, create_scheduler


class Trainer:
    """
    Main trainer for vision-language models
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task: str = 'caption',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_steps: int = 1000,
        scheduler_type: str = 'cosine',
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1,
        device: str = 'cuda',
        output_dir: str = './outputs',
        logging_steps: int = 100,
        eval_steps: int = 1000,
        save_steps: int = 1000,
        max_checkpoints: int = 3,
        resume_from: Optional[str] = None,
        mixed_precision: bool = True
    ):
        """
        Args:
            model: VLM model
            train_loader: Training data loader
            val_loader: Validation data loader
            task: Task type ('caption', 'vqa', 'retrieval', 'contrastive')
            learning_rate: Learning rate
            weight_decay: Weight decay
            num_epochs: Number of training epochs
            warmup_steps: Warmup steps
            scheduler_type: LR scheduler type
            gradient_clip: Gradient clipping value
            accumulation_steps: Gradient accumulation steps
            device: Device to train on
            output_dir: Output directory
            logging_steps: Steps between logging
            eval_steps: Steps between evaluations
            save_steps: Steps between checkpoints
            max_checkpoints: Maximum checkpoints to keep
            resume_from: Path to checkpoint to resume from
            mixed_precision: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task
        self.device = device
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_checkpoints = max_checkpoints
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Loss function
        self.criterion = self._get_criterion()

        # Optimizer
        self.optimizer = create_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        total_steps = len(train_loader) * num_epochs // accumulation_steps
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            num_training_steps=total_steps,
            num_warmup_steps=warmup_steps
        )

        # Mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if task in ['caption'] else 0.0
        self.checkpoints = []

        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)

    def _get_criterion(self) -> nn.Module:
        """Get loss criterion for task"""
        if self.task == 'caption':
            return CaptionLoss()
        elif self.task == 'vqa':
            return VQALoss()
        elif self.task == 'retrieval':
            return RetrievalLoss()
        elif self.task == 'contrastive':
            return ContrastiveLoss()
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.logger.info(f"Total steps: {len(self.train_loader) * self.num_epochs}")
        self.logger.info(f"Device: {self.device}")

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train one epoch
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # Log epoch metrics
            self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}")
            self.logger.info(f"Epoch {epoch + 1} - Val Loss: {val_metrics['loss']:.4f}")

            # Save checkpoint
            self.save_checkpoint(
                os.path.join(self.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            )

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                loss = self.compute_loss(batch)
                loss = loss / self.accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip > 0:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                # Optimizer step
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                # Logging
                if self.global_step % self.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.logger.info(
                        f"Step {self.global_step} - Loss: {loss.item() * self.accumulation_steps:.4f} - LR: {lr:.6f}"
                    )

                # Evaluation
                if self.global_step % self.eval_steps == 0:
                    val_metrics = self.validate()
                    self.logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")
                    self.model.train()

                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    checkpoint_path = os.path.join(
                        self.output_dir,
                        f'checkpoint_step_{self.global_step}.pt'
                    )
                    self.save_checkpoint(checkpoint_path)

            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1

            progress_bar.set_postfix({'loss': total_loss / num_batches})

        return {'loss': total_loss / num_batches}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute task-specific loss"""
        images = batch['image']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        if self.task == 'caption':
            # Shift target for teacher forcing
            target_ids = input_ids.clone()
            target_ids[:, :-1] = input_ids[:, 1:]

            # Forward pass
            outputs = self.model(
                images=images,
                input_ids=input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                target_ids=input_ids[:, :-1],
                task='caption'
            )

            loss = self.criterion(outputs['caption_logits'], target_ids[:, :-1])

        elif self.task == 'vqa':
            answer_labels = batch['answer_label']

            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                task='vqa'
            )

            loss = self.criterion(outputs['vqa_logits'], answer_labels)

        elif self.task == 'retrieval':
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                task='retrieval'
            )

            loss = self.criterion(
                outputs['similarity_matrix'],
                outputs['fine_grained_score']
            )

        elif self.task == 'contrastive':
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                task='contrastive'
            )

            loss = self.criterion(
                outputs['vision_embeds'],
                outputs['language_embeds'],
                outputs['logit_scale']
            )

        else:
            raise ValueError(f"Unknown task: {self.task}")

        return loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Compute loss
            loss = self.compute_loss(batch)

            total_loss += loss.item()
            num_batches += 1

        return {'loss': total_loss / num_batches}

    def save_checkpoint(self, path: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
        }

        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

        # Manage checkpoints
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']

        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.info(f"Checkpoint loaded from {path}")


class ContrastiveTrainer(Trainer):
    """
    Specialized trainer for contrastive learning
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, task='contrastive', **kwargs)


if __name__ == "__main__":
    print("Trainer module - use with VLM model and dataloaders")
    print("Example usage in examples/train_example.py")
