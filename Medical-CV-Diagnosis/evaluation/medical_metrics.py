"""
Medical Evaluation Metrics
Comprehensive metrics for medical image classification and segmentation
Includes sensitivity, specificity, Dice, IoU, AUC-ROC, and more
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_sensitivity(tp: float, fn: float) -> float:
    """Sensitivity (Recall, True Positive Rate)"""
    return tp / (tp + fn + 1e-10)


def compute_specificity(tn: float, fp: float) -> float:
    """Specificity (True Negative Rate)"""
    return tn / (tn + fp + 1e-10)


def compute_precision(tp: float, fp: float) -> float:
    """Precision (Positive Predictive Value)"""
    return tp / (tp + fp + 1e-10)


def compute_npv(tn: float, fn: float) -> float:
    """Negative Predictive Value"""
    return tn / (tn + fn + 1e-10)


def compute_f1_score(precision: float, recall: float) -> float:
    """F1 Score"""
    return 2 * (precision * recall) / (precision + recall + 1e-10)


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    Dice Coefficient for segmentation

    Args:
        pred: Predicted segmentation (B, C, H, W) or (B, H, W)
        target: Ground truth (B, C, H, W) or (B, H, W)
        smooth: Smoothing factor

    Returns:
        Dice score
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    Intersection over Union (IoU / Jaccard Index)

    Args:
        pred: Predicted segmentation
        target: Ground truth
        smooth: Smoothing factor

    Returns:
        IoU score
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUC-ROC score

    Args:
        y_true: True labels
        y_score: Predicted scores/probabilities

    Returns:
        AUC-ROC score
    """
    try:
        return roc_auc_score(y_true, y_score)
    except Exception as e:
        logger.warning(f"Error computing AUC-ROC: {e}")
        return 0.0


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix"""
    return confusion_matrix(y_true, y_pred)


class ClassificationMetrics:
    """Comprehensive classification metrics for medical imaging"""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize classification metrics

        Args:
            num_classes: Number of classes
            class_names: Optional class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.all_targets = []
        self.all_predictions = []
        self.all_probabilities = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probabilities: Optional[torch.Tensor] = None):
        """
        Update metrics with batch results

        Args:
            predictions: Predicted class indices (B,)
            targets: True class indices (B,)
            probabilities: Class probabilities (B, num_classes)
        """
        self.all_targets.extend(targets.cpu().numpy())
        self.all_predictions.extend(predictions.cpu().numpy())

        if probabilities is not None:
            self.all_probabilities.extend(probabilities.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics

        Returns:
            Dictionary of metric names and values
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        metrics = {}

        # Overall accuracy
        metrics['accuracy'] = (y_true == y_pred).mean()

        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            # Binary classification metrics for each class
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            metrics[f'{class_name}_sensitivity'] = compute_sensitivity(tp, fn)
            metrics[f'{class_name}_specificity'] = compute_specificity(tn, fp)
            metrics[f'{class_name}_precision'] = compute_precision(tp, fp)
            metrics[f'{class_name}_npv'] = compute_npv(tn, fn)
            metrics[f'{class_name}_f1'] = compute_f1_score(
                metrics[f'{class_name}_precision'],
                metrics[f'{class_name}_sensitivity']
            )

        # Macro-averaged metrics
        metrics['macro_sensitivity'] = np.mean([metrics[f'{name}_sensitivity'] for name in self.class_names])
        metrics['macro_specificity'] = np.mean([metrics[f'{name}_specificity'] for name in self.class_names])
        metrics['macro_precision'] = np.mean([metrics[f'{name}_precision'] for name in self.class_names])
        metrics['macro_f1'] = np.mean([metrics[f'{name}_f1'] for name in self.class_names])

        # Cohen's Kappa
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # AUC-ROC if probabilities available
        if len(self.all_probabilities) > 0:
            y_prob = np.array(self.all_probabilities)

            if self.num_classes == 2:
                # Binary classification
                metrics['auc_roc'] = compute_auc_roc(y_true, y_prob[:, 1])
                metrics['auc_pr'] = average_precision_score(y_true, y_prob[:, 1])
            else:
                # Multi-class (one-vs-rest)
                try:
                    metrics['auc_roc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                    metrics['auc_roc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
                except:
                    pass

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(self.all_targets, self.all_predictions)

    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        return classification_report(
            self.all_targets,
            self.all_predictions,
            target_names=self.class_names
        )


class SegmentationMetrics:
    """Comprehensive segmentation metrics"""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize segmentation metrics

        Args:
            num_classes: Number of segmentation classes
            class_names: Optional class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.dice_scores = {name: [] for name in self.class_names}
        self.iou_scores = {name: [] for name in self.class_names}
        self.pixel_accuracies = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch results

        Args:
            predictions: Predicted segmentation (B, C, H, W) or (B, H, W)
            targets: Ground truth segmentation (B, C, H, W) or (B, H, W)
        """
        # Convert to one-hot if needed
        if predictions.ndim == 3:
            predictions = F.one_hot(predictions, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if targets.ndim == 3:
            targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Compute per-class metrics
        for i, class_name in enumerate(self.class_names):
            pred_class = predictions[:, i]
            target_class = targets[:, i]

            dice = compute_dice_score(pred_class, target_class)
            iou = compute_iou(pred_class, target_class)

            self.dice_scores[class_name].append(dice)
            self.iou_scores[class_name].append(iou)

        # Pixel accuracy
        pred_labels = predictions.argmax(dim=1)
        target_labels = targets.argmax(dim=1)
        pixel_acc = (pred_labels == target_labels).float().mean().item()
        self.pixel_accuracies.append(pixel_acc)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}

        # Per-class metrics
        for class_name in self.class_names:
            metrics[f'{class_name}_dice'] = np.mean(self.dice_scores[class_name])
            metrics[f'{class_name}_iou'] = np.mean(self.iou_scores[class_name])

        # Mean metrics
        metrics['mean_dice'] = np.mean([metrics[f'{name}_dice'] for name in self.class_names])
        metrics['mean_iou'] = np.mean([metrics[f'{name}_iou'] for name in self.class_names])
        metrics['pixel_accuracy'] = np.mean(self.pixel_accuracies)

        return metrics


class MedicalMetrics:
    """
    Unified medical metrics class
    Combines classification and segmentation metrics
    """

    def __init__(
        self,
        task: str = 'classification',
        num_classes: int = 2,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize medical metrics

        Args:
            task: 'classification' or 'segmentation'
            num_classes: Number of classes
            class_names: Optional class names
        """
        self.task = task

        if task == 'classification':
            self.metrics = ClassificationMetrics(num_classes, class_names)
        elif task == 'segmentation':
            self.metrics = SegmentationMetrics(num_classes, class_names)
        else:
            raise ValueError(f"Unknown task: {task}")

    def reset(self):
        """Reset metrics"""
        self.metrics.reset()

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probabilities: Optional[torch.Tensor] = None):
        """Update metrics"""
        if self.task == 'classification':
            self.metrics.update(predictions, targets, probabilities)
        else:
            self.metrics.update(predictions, targets)

    def compute(self) -> Dict[str, float]:
        """Compute metrics"""
        return self.metrics.compute()

    def print_summary(self):
        """Print metrics summary"""
        metrics = self.compute()

        logger.info("=" * 60)
        logger.info(f"Medical {self.task.capitalize()} Metrics")
        logger.info("=" * 60)

        for metric_name, value in sorted(metrics.items()):
            logger.info(f"{metric_name:.<40} {value:.4f}")

        logger.info("=" * 60)

        if self.task == 'classification':
            logger.info("\nClassification Report:")
            logger.info(self.metrics.get_classification_report())
