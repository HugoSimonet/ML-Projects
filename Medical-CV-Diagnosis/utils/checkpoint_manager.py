"""Checkpoint Manager for model saving/loading"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints"""

    def __init__(self, checkpoint_dir: str = './checkpoints', keep_last_n: int = 5):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        filename: Optional[str] = None,
        **kwargs
    ):
        """
        Save checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Metrics dictionary
            filename: Custom filename
            **kwargs: Additional data to save
        """
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            **kwargs
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Clean old checkpoints
        self._cleanup_old_checkpoints()

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: str = 'best_model.pth',
        device: str = 'cpu'
    ) -> Dict:
        """
        Load checkpoint

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            filename: Checkpoint filename
            device: Device to map tensors to

        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

        return checkpoint

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only last N"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))

        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                ckpt.unlink()
                logger.debug(f"Removed old checkpoint: {ckpt}")
