"""
Forecasting Evaluation Metrics
Comprehensive metrics for time series forecasting evaluation
"""

import torch
import numpy as np
from typing import Union, List, Tuple, Dict
from scipy import stats


class ForecastingMetrics:
    """
    Collection of metrics for evaluating time series forecasting models
    """

    @staticmethod
    def mae(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Mean Absolute Error

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            MAE score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        return np.mean(np.abs(predictions - targets))

    @staticmethod
    def mse(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Mean Squared Error

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            MSE score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        return np.mean((predictions - targets) ** 2)

    @staticmethod
    def rmse(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Root Mean Squared Error

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            RMSE score
        """
        return np.sqrt(ForecastingMetrics.mse(predictions, targets))

    @staticmethod
    def mape(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        epsilon: float = 1e-8
    ) -> float:
        """
        Mean Absolute Percentage Error

        Args:
            predictions: Predicted values
            targets: Target values
            epsilon: Small value to avoid division by zero

        Returns:
            MAPE score (in percentage)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Avoid division by zero
        mask = np.abs(targets) > epsilon
        if not np.any(mask):
            return 0.0

        return np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100

    @staticmethod
    def smape(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        epsilon: float = 1e-8
    ) -> float:
        """
        Symmetric Mean Absolute Percentage Error

        Args:
            predictions: Predicted values
            targets: Target values
            epsilon: Small value to avoid division by zero

        Returns:
            SMAPE score (in percentage)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        denominator = (np.abs(targets) + np.abs(predictions)) / 2.0 + epsilon
        return np.mean(np.abs(targets - predictions) / denominator) * 100

    @staticmethod
    def mase(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        train_data: Union[torch.Tensor, np.ndarray],
        seasonal_period: int = 1
    ) -> float:
        """
        Mean Absolute Scaled Error

        Args:
            predictions: Predicted values
            targets: Target values
            train_data: Training data for scaling
            seasonal_period: Seasonal period for naive forecast

        Returns:
            MASE score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(train_data, torch.Tensor):
            train_data = train_data.detach().cpu().numpy()

        # Calculate MAE of forecast
        mae_forecast = np.mean(np.abs(targets - predictions))

        # Calculate MAE of naive forecast on training data
        naive_forecast = train_data[:-seasonal_period]
        naive_targets = train_data[seasonal_period:]
        mae_naive = np.mean(np.abs(naive_targets - naive_forecast))

        if mae_naive == 0:
            return 0.0

        return mae_forecast / mae_naive

    @staticmethod
    def r2_score(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        R² (coefficient of determination) score

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            R² score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    @staticmethod
    def quantile_loss(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        quantile: float
    ) -> float:
        """
        Quantile loss (pinball loss)

        Args:
            predictions: Predicted quantile values
            targets: Target values
            quantile: Quantile level (0-1)

        Returns:
            Quantile loss
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        errors = targets - predictions
        loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
        return np.mean(loss)

    @staticmethod
    def crps(
        predictions_samples: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Continuous Ranked Probability Score
        Measures the accuracy of probabilistic forecasts

        Args:
            predictions_samples: Samples from predictive distribution [num_samples, ...]
            targets: Target values

        Returns:
            CRPS score
        """
        if isinstance(predictions_samples, torch.Tensor):
            predictions_samples = predictions_samples.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Empirical CRPS calculation
        num_samples = predictions_samples.shape[0]

        # Term 1: Mean absolute error
        term1 = np.mean(np.abs(predictions_samples - targets), axis=0)

        # Term 2: Mean pairwise distance between samples
        term2 = 0
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                term2 += np.abs(predictions_samples[i] - predictions_samples[j])

        term2 = term2 / (num_samples * (num_samples - 1) / 2)

        crps = term1 - 0.5 * term2
        return np.mean(crps)

    @staticmethod
    def coverage(
        lower_bound: Union[torch.Tensor, np.ndarray],
        upper_bound: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Coverage: percentage of targets within prediction interval

        Args:
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval
            targets: Target values

        Returns:
            Coverage percentage (0-100)
        """
        if isinstance(lower_bound, torch.Tensor):
            lower_bound = lower_bound.detach().cpu().numpy()
        if isinstance(upper_bound, torch.Tensor):
            upper_bound = upper_bound.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        within_interval = (targets >= lower_bound) & (targets <= upper_bound)
        return np.mean(within_interval) * 100

    @staticmethod
    def sharpness(
        lower_bound: Union[torch.Tensor, np.ndarray],
        upper_bound: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Sharpness: average width of prediction intervals

        Args:
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval

        Returns:
            Average interval width
        """
        if isinstance(lower_bound, torch.Tensor):
            lower_bound = lower_bound.detach().cpu().numpy()
        if isinstance(upper_bound, torch.Tensor):
            upper_bound = upper_bound.detach().cpu().numpy()

        return np.mean(upper_bound - lower_bound)

    @staticmethod
    def winkler_score(
        lower_bound: Union[torch.Tensor, np.ndarray],
        upper_bound: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        alpha: float = 0.05
    ) -> float:
        """
        Winkler Score: combines coverage and sharpness

        Args:
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval
            targets: Target values
            alpha: Significance level (default: 0.05 for 95% interval)

        Returns:
            Winkler score (lower is better)
        """
        if isinstance(lower_bound, torch.Tensor):
            lower_bound = lower_bound.detach().cpu().numpy()
        if isinstance(upper_bound, torch.Tensor):
            upper_bound = upper_bound.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        width = upper_bound - lower_bound
        penalty_lower = (2 / alpha) * (lower_bound - targets) * (targets < lower_bound)
        penalty_upper = (2 / alpha) * (targets - upper_bound) * (targets > upper_bound)

        score = width + penalty_lower + penalty_upper
        return np.mean(score)


class AnomalyDetectionMetrics:
    """
    Metrics for anomaly detection evaluation
    """

    @staticmethod
    def precision(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Precision: TP / (TP + FP)

        Args:
            predictions: Binary predictions (0 or 1)
            targets: Binary targets (0 or 1)

        Returns:
            Precision score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        predictions = predictions.astype(bool)
        targets = targets.astype(bool)

        tp = np.sum(predictions & targets)
        fp = np.sum(predictions & ~targets)

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    @staticmethod
    def recall(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Recall: TP / (TP + FN)

        Args:
            predictions: Binary predictions (0 or 1)
            targets: Binary targets (0 or 1)

        Returns:
            Recall score
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        predictions = predictions.astype(bool)
        targets = targets.astype(bool)

        tp = np.sum(predictions & targets)
        fn = np.sum(~predictions & targets)

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    @staticmethod
    def f1_score(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        F1 Score: harmonic mean of precision and recall

        Args:
            predictions: Binary predictions (0 or 1)
            targets: Binary targets (0 or 1)

        Returns:
            F1 score
        """
        prec = AnomalyDetectionMetrics.precision(predictions, targets)
        rec = AnomalyDetectionMetrics.recall(predictions, targets)

        if prec + rec == 0:
            return 0.0

        return 2 * (prec * rec) / (prec + rec)

    @staticmethod
    def auroc(
        scores: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Area Under ROC Curve

        Args:
            scores: Anomaly scores (continuous)
            targets: Binary targets (0 or 1)

        Returns:
            AUROC score
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        from sklearn.metrics import roc_auc_score
        return roc_auc_score(targets, scores)

    @staticmethod
    def auprc(
        scores: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Area Under Precision-Recall Curve

        Args:
            scores: Anomaly scores (continuous)
            targets: Binary targets (0 or 1)

        Returns:
            AUPRC score
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        from sklearn.metrics import average_precision_score
        return average_precision_score(targets, scores)


def evaluate_forecasting(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    train_data: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Comprehensive forecasting evaluation

    Args:
        predictions: Predicted values
        targets: Target values
        train_data: Training data (for MASE)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['mae'] = ForecastingMetrics.mae(predictions, targets)
    metrics['mse'] = ForecastingMetrics.mse(predictions, targets)
    metrics['rmse'] = ForecastingMetrics.rmse(predictions, targets)
    metrics['mape'] = ForecastingMetrics.mape(predictions, targets)
    metrics['smape'] = ForecastingMetrics.smape(predictions, targets)
    metrics['r2'] = ForecastingMetrics.r2_score(predictions, targets)

    # MASE if training data provided
    if train_data is not None:
        metrics['mase'] = ForecastingMetrics.mase(predictions, targets, train_data)

    return metrics


def evaluate_probabilistic_forecasting(
    predictions_quantiles: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    quantiles: List[float],
    predictions_samples: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Evaluate probabilistic forecasting

    Args:
        predictions_quantiles: Predicted quantiles [..., num_quantiles]
        targets: Target values
        quantiles: List of quantile levels
        predictions_samples: Samples from predictive distribution [num_samples, ...]

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Quantile scores
    for i, q in enumerate(quantiles):
        q_pred = predictions_quantiles[..., i]
        metrics[f'quantile_loss_{q}'] = ForecastingMetrics.quantile_loss(
            q_pred, targets, q
        )

    # Prediction intervals
    if 0.1 in quantiles and 0.9 in quantiles:
        idx_lower = quantiles.index(0.1)
        idx_upper = quantiles.index(0.9)
        lower = predictions_quantiles[..., idx_lower]
        upper = predictions_quantiles[..., idx_upper]

        metrics['coverage_80'] = ForecastingMetrics.coverage(lower, upper, targets)
        metrics['sharpness_80'] = ForecastingMetrics.sharpness(lower, upper)
        metrics['winkler_80'] = ForecastingMetrics.winkler_score(
            lower, upper, targets, alpha=0.2
        )

    if 0.05 in quantiles and 0.95 in quantiles:
        idx_lower = quantiles.index(0.05)
        idx_upper = quantiles.index(0.95)
        lower = predictions_quantiles[..., idx_lower]
        upper = predictions_quantiles[..., idx_upper]

        metrics['coverage_90'] = ForecastingMetrics.coverage(lower, upper, targets)
        metrics['sharpness_90'] = ForecastingMetrics.sharpness(lower, upper)
        metrics['winkler_90'] = ForecastingMetrics.winkler_score(
            lower, upper, targets, alpha=0.1
        )

    # CRPS if samples provided
    if predictions_samples is not None:
        metrics['crps'] = ForecastingMetrics.crps(predictions_samples, targets)

    return metrics


def evaluate_anomaly_detection(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    scores: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Evaluate anomaly detection

    Args:
        predictions: Binary predictions (0 or 1)
        targets: Binary targets (0 or 1)
        scores: Anomaly scores (continuous, optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Classification metrics
    metrics['precision'] = AnomalyDetectionMetrics.precision(predictions, targets)
    metrics['recall'] = AnomalyDetectionMetrics.recall(predictions, targets)
    metrics['f1'] = AnomalyDetectionMetrics.f1_score(predictions, targets)

    # ROC metrics if scores provided
    if scores is not None:
        metrics['auroc'] = AnomalyDetectionMetrics.auroc(scores, targets)
        metrics['auprc'] = AnomalyDetectionMetrics.auprc(scores, targets)

    return metrics
