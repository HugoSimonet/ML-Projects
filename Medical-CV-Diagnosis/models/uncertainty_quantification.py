"""
Uncertainty Quantification for Medical AI
Monte Carlo Dropout, Deep Ensembles, and uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MCDropoutModel(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation
    Enables dropout at test time for multiple stochastic forward passes
    """

    def __init__(self, base_model: nn.Module, num_samples: int = 30):
        """
        Initialize MC Dropout model

        Args:
            base_model: Base classification/segmentation model
            num_samples: Number of MC samples for uncertainty estimation
        """
        super().__init__()

        self.base_model = base_model
        self.num_samples = num_samples

        # Enable dropout during inference
        self._enable_dropout()

    def _enable_dropout(self):
        """Enable dropout layers during inference"""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass"""
        return self.base_model(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_all_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict with uncertainty estimation using MC Dropout

        Args:
            x: Input tensor (B, C, H, W)
            return_all_samples: Return all MC samples

        Returns:
            mean_prediction: Mean prediction (B, num_classes)
            uncertainty: Predictive uncertainty (B, num_classes)
            all_samples: All MC samples if return_all_samples=True
        """
        self.base_model.eval()
        self._enable_dropout()

        predictions = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.base_model(x)
                # Apply softmax for classification
                if output.dim() == 2:  # Classification
                    probs = F.softmax(output, dim=1)
                else:  # Segmentation
                    probs = F.softmax(output, dim=1)
                predictions.append(probs)

        predictions = torch.stack(predictions)  # (num_samples, B, num_classes, ...)

        # Calculate mean and variance
        mean_pred = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)

        # Total uncertainty (entropy)
        uncertainty = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=1)

        if return_all_samples:
            return mean_pred, uncertainty, predictions

        return mean_pred, uncertainty, None

    def get_epistemic_aleatoric_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose uncertainty into epistemic and aleatoric components

        Args:
            x: Input tensor

        Returns:
            epistemic_uncertainty: Model uncertainty (reducible with more data)
            aleatoric_uncertainty: Data uncertainty (irreducible)
        """
        mean_pred, _, all_samples = self.predict_with_uncertainty(x, return_all_samples=True)

        # Epistemic uncertainty (variance across predictions)
        epistemic = torch.var(all_samples, dim=0).mean(dim=1)

        # Aleatoric uncertainty (expected entropy)
        entropies = []
        for sample in all_samples:
            entropy = -torch.sum(sample * torch.log(sample + 1e-10), dim=1)
            entropies.append(entropy)

        aleatoric = torch.stack(entropies).mean(dim=0)

        return epistemic, aleatoric


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty estimation
    Multiple independently trained models for robust predictions
    """

    def __init__(self, models: List[nn.Module]):
        """
        Initialize deep ensemble

        Args:
            models: List of independently trained models
        """
        super().__init__()

        self.models = nn.ModuleList(models)
        self.num_models = len(models)

        logger.info(f"Created ensemble with {self.num_models} models")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through first model (for compatibility)"""
        return self.models[0](x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_all_predictions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Ensemble prediction with uncertainty

        Args:
            x: Input tensor
            return_all_predictions: Return predictions from all models

        Returns:
            mean_prediction: Mean prediction across ensemble
            uncertainty: Predictive variance
            all_predictions: Predictions from all models if requested
        """
        predictions = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(x)

                # Apply softmax for classification
                if output.dim() == 2:  # Classification
                    probs = F.softmax(output, dim=1)
                else:  # Segmentation
                    probs = F.softmax(output, dim=1)

                predictions.append(probs)

        predictions = torch.stack(predictions)  # (num_models, B, num_classes, ...)

        # Calculate mean and variance
        mean_pred = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)

        # Total uncertainty
        uncertainty = variance.mean(dim=1)  # Average across classes

        if return_all_predictions:
            return mean_pred, uncertainty, predictions

        return mean_pred, uncertainty, None

    def get_confidence_interval(
        self,
        x: torch.Tensor,
        confidence: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prediction with confidence intervals

        Args:
            x: Input tensor
            confidence: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            mean_prediction: Mean prediction
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
        """
        mean_pred, _, all_preds = self.predict_with_uncertainty(x, return_all_predictions=True)

        # Calculate percentiles
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = torch.quantile(all_preds, lower_percentile / 100, dim=0)
        upper_bound = torch.quantile(all_preds, upper_percentile / 100, dim=0)

        return mean_pred, lower_bound, upper_bound


class UncertaintyEstimator:
    """
    Comprehensive uncertainty estimation toolkit
    Combines multiple uncertainty quantification methods
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = 'mc_dropout',
        num_samples: int = 30
    ):
        """
        Initialize uncertainty estimator

        Args:
            model: Base model or ensemble
            method: Uncertainty method ('mc_dropout', 'ensemble')
            num_samples: Number of MC samples (for MC Dropout)
        """
        self.method = method

        if method == 'mc_dropout':
            self.uncertainty_model = MCDropoutModel(model, num_samples)
        elif method == 'ensemble':
            if isinstance(model, list):
                self.uncertainty_model = DeepEnsemble(model)
            else:
                raise ValueError("Ensemble method requires list of models")
        else:
            raise ValueError(f"Unknown method: {method}")

    def predict(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Make prediction with optional uncertainty

        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            prediction: Model prediction
            uncertainty: Uncertainty estimate (if requested)
        """
        if return_uncertainty:
            mean_pred, uncertainty, _ = self.uncertainty_model.predict_with_uncertainty(x)
            return mean_pred, uncertainty
        else:
            with torch.no_grad():
                pred = self.uncertainty_model(x)
                return pred, None

    def calibrate_uncertainty(
        self,
        val_loader,
        temperature: float = 1.0
    ) -> float:
        """
        Calibrate uncertainty using temperature scaling

        Args:
            val_loader: Validation data loader
            temperature: Initial temperature

        Returns:
            Optimal temperature
        """
        # Implement temperature scaling for calibration
        # This is a placeholder - full implementation would optimize temperature
        logger.info("Calibrating uncertainty estimates...")

        # Collect predictions and labels
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                logits = self.uncertainty_model(images)
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # Simple temperature scaling (could be optimized)
        optimal_temperature = temperature

        logger.info(f"Optimal temperature: {optimal_temperature:.3f}")

        return optimal_temperature

    def get_rejection_threshold(
        self,
        val_loader,
        target_accuracy: float = 0.95
    ) -> float:
        """
        Determine uncertainty threshold for prediction rejection

        Args:
            val_loader: Validation data loader
            target_accuracy: Target accuracy for accepted predictions

        Returns:
            Uncertainty threshold for rejection
        """
        uncertainties = []
        correct = []

        with torch.no_grad():
            for images, labels in val_loader:
                mean_pred, uncertainty, _ = self.uncertainty_model.predict_with_uncertainty(images)

                predicted = torch.argmax(mean_pred, dim=1)
                is_correct = (predicted == labels).float()

                uncertainties.extend(uncertainty.cpu().numpy())
                correct.extend(is_correct.cpu().numpy())

        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_correct = np.array(correct)[sorted_indices]

        # Find threshold that achieves target accuracy
        for i in range(len(sorted_correct)):
            accuracy = np.mean(sorted_correct[:i+1])
            if accuracy >= target_accuracy:
                threshold = uncertainties[sorted_indices[i]]
                logger.info(f"Rejection threshold: {threshold:.4f} (keeps {(i+1)/len(sorted_correct)*100:.1f}% of predictions)")
                return threshold

        return float('inf')

    def compute_calibration_error(
        self,
        val_loader,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE)

        Args:
            val_loader: Validation data loader
            num_bins: Number of bins for calibration

        Returns:
            Expected calibration error
        """
        confidences = []
        accuracies = []

        with torch.no_grad():
            for images, labels in val_loader:
                mean_pred, _, _ = self.uncertainty_model.predict_with_uncertainty(images)

                confidence, predicted = torch.max(mean_pred, dim=1)
                is_correct = (predicted == labels).float()

                confidences.extend(confidence.cpu().numpy())
                accuracies.extend(is_correct.cpu().numpy())

        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        # Bin confidences
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1

        ece = 0.0
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_confidence = confidences[mask].mean()
                bin_accuracy = accuracies[mask].mean()
                bin_size = mask.sum() / len(confidences)

                ece += bin_size * np.abs(bin_confidence - bin_accuracy)

        logger.info(f"Expected Calibration Error: {ece:.4f}")

        return ece


def compute_prediction_interval(
    predictions: torch.Tensor,
    confidence: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute prediction intervals from multiple samples

    Args:
        predictions: Stacked predictions (num_samples, batch, ...)
        confidence: Confidence level

    Returns:
        lower_bound, upper_bound
    """
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = torch.quantile(predictions, lower_percentile / 100, dim=0)
    upper_bound = torch.quantile(predictions, upper_percentile / 100, dim=0)

    return lower_bound, upper_bound
