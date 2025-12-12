"""
Medical Loss Functions
Focal Loss, Dice Loss, Tversky Loss, and combined losses for medical imaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard examples
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss

        Args:
            alpha: Class weights (num_classes,)
            gamma: Focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, num_classes) or (B, C, H, W)
            targets: Ground truth (B,) or (B, H, W)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Directly optimizes Dice coefficient
    """

    def __init__(self, smooth: float = 1.0, apply_softmax: bool = True):
        """
        Initialize Dice Loss

        Args:
            smooth: Smoothing factor
            apply_softmax: Whether to apply softmax to inputs
        """
        super().__init__()
        self.smooth = smooth
        self.apply_softmax = apply_softmax

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, C, H, W)
            targets: Ground truth (B, H, W)
        """
        if self.apply_softmax:
            inputs = F.softmax(inputs, dim=1)

        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        # Flatten
        inputs_flat = inputs.contiguous().view(-1)
        targets_flat = targets_one_hot.contiguous().view(-1)

        # Dice coefficient
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss
    Controls false positives and false negatives
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0, apply_softmax: bool = True):
        """
        Initialize Tversky Loss

        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            apply_softmax: Whether to apply softmax
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.apply_softmax = apply_softmax

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, C, H, W)
            targets: Ground truth (B, H, W)
        """
        if self.apply_softmax:
            inputs = F.softmax(inputs, dim=1)

        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        # True positives, false positives, false negatives
        tp = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        fp = (inputs * (1 - targets_one_hot)).sum(dim=(0, 2, 3))
        fn = ((1 - inputs) * targets_one_hot).sum(dim=(0, 2, 3))

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1 - tversky.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted sum of multiple losses
    Useful for segmentation (e.g., Dice + CE)
    """

    def __init__(
        self,
        losses: list,
        weights: Optional[list] = None
    ):
        """
        Initialize combined loss

        Args:
            losses: List of loss functions
            weights: List of weights for each loss
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of losses"""
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        return total_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Cross Entropy with class weights for imbalanced data"""

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """
        Initialize weighted cross entropy

        Args:
            class_weights: Weight for each class
        """
        super().__init__()
        self.class_weights = class_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross entropy"""
        return F.cross_entropy(inputs, targets, weight=self.class_weights)
