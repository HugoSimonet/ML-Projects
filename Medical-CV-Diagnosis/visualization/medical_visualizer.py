"""
Medical Visualization Tools
Comprehensive visualization for medical images, predictions, and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import cv2
from typing import List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class MedicalVisualizer:
    """Comprehensive medical imaging visualizer"""

    def __init__(self, output_dir: str = './visualizations'):
        """Initialize visualizer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_medical_image(
        self,
        image: np.ndarray,
        title: str = 'Medical Image',
        save_path: Optional[str] = None
    ):
        """Plot single medical image"""
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()

    def plot_segmentation_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        pred_mask: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        save_path: Optional[str] = None
    ):
        """Plot segmentation overlay"""
        n_plots = 3 if pred_mask is not None else 2

        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))

        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Ground truth
        axes[1].imshow(image, cmap='gray')
        axes[1].imshow(mask, alpha=alpha, cmap='jet')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # Prediction
        if pred_mask is not None:
            axes[2].imshow(image, cmap='gray')
            axes[2].imshow(pred_mask, alpha=alpha, cmap='jet')
            axes[2].set_title('Prediction')
            axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()

    def plot_grad_cam(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        save_path: Optional[str] = None
    ):
        """Plot Grad-CAM visualization"""
        # Resize heatmap to match image
        if heatmap.shape != image.shape:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Overlay
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        if len(image_uint8.shape) == 2:
            image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image_uint8

        overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap_colored, alpha, 0)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix"""
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()

    def plot_roc_curves(
        self,
        fpr_list: List[np.ndarray],
        tpr_list: List[np.ndarray],
        auc_list: List[float],
        class_names: List[str],
        save_path: Optional[str] = None
    ):
        """Plot ROC curves for multiple classes"""
        plt.figure(figsize=(10, 8))

        for fpr, tpr, auc, class_name in zip(fpr_list, tpr_list, auc_list, class_names):
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        val_metrics: Optional[List[float]] = None,
        metric_name: str = 'Accuracy',
        save_path: Optional[str] = None
    ):
        """Plot training curves"""
        epochs = range(1, len(train_losses) + 1)

        if val_metrics:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Metric curves
        if val_metrics:
            ax2.plot(epochs, val_metrics, 'g-', label=f'Validation {metric_name}', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel(metric_name)
            ax2.set_title(f'Validation {metric_name}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()


# Convenience functions
def plot_medical_image(image: np.ndarray, title: str = 'Medical Image', save_path: Optional[str] = None):
    """Convenience function for plotting medical image"""
    visualizer = MedicalVisualizer()
    visualizer.plot_medical_image(image, title, save_path)


def plot_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """Convenience function for segmentation overlay"""
    visualizer = MedicalVisualizer()
    visualizer.plot_segmentation_overlay(image, mask, pred_mask, save_path=save_path)


def plot_grad_cam(image: np.ndarray, heatmap: np.ndarray, save_path: Optional[str] = None):
    """Convenience function for Grad-CAM visualization"""
    visualizer = MedicalVisualizer()
    visualizer.plot_grad_cam(image, heatmap, save_path=save_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
    """Convenience function for confusion matrix"""
    visualizer = MedicalVisualizer()
    visualizer.plot_confusion_matrix(cm, class_names, save_path=save_path)


def plot_roc_curves(
    fpr_list: List[np.ndarray],
    tpr_list: List[np.ndarray],
    auc_list: List[float],
    class_names: List[str],
    save_path: Optional[str] = None
):
    """Convenience function for ROC curves"""
    visualizer = MedicalVisualizer()
    visualizer.plot_roc_curves(fpr_list, tpr_list, auc_list, class_names, save_path)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_metrics: Optional[List[float]] = None,
    save_path: Optional[str] = None
):
    """Convenience function for training curves"""
    visualizer = MedicalVisualizer()
    visualizer.plot_training_curves(train_losses, val_losses, val_metrics, save_path=save_path)
