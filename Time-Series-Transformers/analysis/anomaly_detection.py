"""
Anomaly Detection for Time Series
Multiple methods for detecting anomalies in time series data
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, Optional, List
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Base class for anomaly detection in time series
    """

    def __init__(
        self,
        method: str = 'threshold',
        threshold: float = 3.0,
        contamination: float = 0.1
    ):
        """
        Args:
            method: Detection method
                - 'threshold': Statistical threshold (z-score)
                - 'iqr': Interquartile range
                - 'isolation_forest': Isolation Forest
                - 'reconstruction': Reconstruction error
            threshold: Threshold for z-score method
            contamination: Expected proportion of anomalies
        """
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, data: np.ndarray):
        """
        Fit the anomaly detector

        Args:
            data: Training data (normal samples) [N, features]
        """
        if self.method == 'threshold':
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)

        elif self.method == 'iqr':
            self.q1 = np.percentile(data, 25, axis=0)
            self.q3 = np.percentile(data, 75, axis=0)
            self.iqr = self.q3 - self.q1

        elif self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            self.model.fit(data)

        self.fitted = True

    def detect(
        self,
        data: np.ndarray,
        return_scores: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect anomalies

        Args:
            data: Data to check for anomalies [N, features]
            return_scores: Whether to return anomaly scores

        Returns:
            anomalies: Binary array (1 for anomaly, 0 for normal)
            scores (optional): Anomaly scores
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted first")

        if self.method == 'threshold':
            # Z-score method
            z_scores = np.abs((data - self.mean) / (self.std + 1e-8))
            scores = np.max(z_scores, axis=1) if data.ndim > 1 else z_scores
            anomalies = (scores > self.threshold).astype(int)

        elif self.method == 'iqr':
            # IQR method
            lower_bound = self.q1 - 1.5 * self.iqr
            upper_bound = self.q3 + 1.5 * self.iqr
            outliers = (data < lower_bound) | (data > upper_bound)
            anomalies = np.any(outliers, axis=1).astype(int) if data.ndim > 1 else outliers.astype(int)
            scores = np.max(np.abs(data - self.mean) / (self.iqr + 1e-8), axis=1) if data.ndim > 1 else np.abs(data - self.mean) / (self.iqr + 1e-8)

        elif self.method == 'isolation_forest':
            # Isolation Forest
            predictions = self.model.predict(data)
            anomalies = (predictions == -1).astype(int)
            scores = -self.model.score_samples(data)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        if return_scores:
            return anomalies, scores
        return anomalies


class ReconstructionAnomalyDetector(nn.Module):
    """
    Anomaly detection using reconstruction error from autoencoder
    """

    def __init__(
        self,
        model: nn.Module,
        threshold_percentile: float = 95.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Trained forecasting or autoencoder model
            threshold_percentile: Percentile for threshold (default: 95th percentile)
            device: Device to run on
        """
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    def compute_reconstruction_error(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction error

        Args:
            data: Input data

        Returns:
            Reconstruction errors
        """
        self.model.eval()
        with torch.no_grad():
            # Get reconstruction
            reconstruction = self.model(data)

            # Compute error (MSE per sample)
            errors = torch.mean((data - reconstruction) ** 2, dim=-1)

            if errors.dim() > 1:
                errors = torch.mean(errors, dim=-1)

        return errors

    def fit_threshold(self, normal_data: torch.Tensor):
        """
        Fit threshold on normal data

        Args:
            normal_data: Normal (non-anomalous) data
        """
        errors = self.compute_reconstruction_error(normal_data)
        errors_np = errors.cpu().numpy()

        self.threshold = np.percentile(errors_np, self.threshold_percentile)

    def detect(
        self,
        data: torch.Tensor,
        return_scores: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Detect anomalies

        Args:
            data: Data to check
            return_scores: Return anomaly scores

        Returns:
            anomalies: Binary tensor (1 for anomaly)
            scores (optional): Reconstruction errors
        """
        if self.threshold is None:
            raise ValueError("Threshold must be fitted first")

        scores = self.compute_reconstruction_error(data)
        anomalies = (scores > self.threshold).long()

        if return_scores:
            return anomalies, scores
        return anomalies


class PredictionErrorAnomalyDetector:
    """
    Anomaly detection based on prediction error
    Uses forecasting model to predict next values and detect anomalies
    """

    def __init__(
        self,
        model: nn.Module,
        threshold_method: str = 'percentile',
        threshold_value: float = 95.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Trained forecasting model
            threshold_method: Method for setting threshold ('percentile', 'std')
            threshold_value: Threshold value (percentile or number of std devs)
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.threshold = None

    def compute_prediction_errors(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction errors

        Args:
            x_enc: Encoder input
            x_mark_enc: Encoder time features
            x_dec: Decoder input
            x_mark_dec: Decoder time features
            y_true: True values

        Returns:
            Prediction errors
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # Compute MAE per sample
            errors = torch.mean(torch.abs(predictions - y_true), dim=(1, 2))

        return errors

    def fit_threshold(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        y_true: torch.Tensor
    ):
        """
        Fit threshold on normal data

        Args:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true: Normal data
        """
        errors = self.compute_prediction_errors(
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true
        )
        errors_np = errors.cpu().numpy()

        if self.threshold_method == 'percentile':
            self.threshold = np.percentile(errors_np, self.threshold_value)
        elif self.threshold_method == 'std':
            mean = np.mean(errors_np)
            std = np.std(errors_np)
            self.threshold = mean + self.threshold_value * std
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")

    def detect(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        y_true: torch.Tensor,
        return_scores: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Detect anomalies

        Args:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true: Test data
            return_scores: Return anomaly scores

        Returns:
            anomalies: Binary tensor (1 for anomaly)
            scores (optional): Prediction errors
        """
        if self.threshold is None:
            raise ValueError("Threshold must be fitted first")

        scores = self.compute_prediction_errors(
            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true
        )
        anomalies = (scores > self.threshold).long()

        if return_scores:
            return anomalies, scores
        return anomalies


class AttentionBasedAnomalyDetector:
    """
    Anomaly detection using attention weights
    Detects anomalies based on unusual attention patterns
    """

    def __init__(
        self,
        model: nn.Module,
        threshold_percentile: float = 95.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Transformer model with attention mechanisms
            threshold_percentile: Threshold percentile
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    def extract_attention_scores(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract attention scores from model

        Args:
            x_enc, x_mark_enc, x_dec, x_mark_dec: Input data

        Returns:
            Attention scores
        """
        self.model.eval()

        # This is a placeholder - actual implementation depends on model architecture
        # You would need to modify the model to return attention weights
        with torch.no_grad():
            _ = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # attention_weights = self.model.get_attention_weights()

        # For now, return a placeholder
        # In practice, you'd compute entropy or variance of attention weights
        attention_scores = torch.zeros(x_enc.size(0)).to(self.device)

        return attention_scores

    def fit_threshold(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor
    ):
        """Fit threshold on normal data"""
        scores = self.extract_attention_scores(x_enc, x_mark_enc, x_dec, x_mark_dec)
        scores_np = scores.cpu().numpy()
        self.threshold = np.percentile(scores_np, self.threshold_percentile)

    def detect(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        return_scores: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Detect anomalies based on attention patterns"""
        if self.threshold is None:
            raise ValueError("Threshold must be fitted first")

        scores = self.extract_attention_scores(x_enc, x_mark_enc, x_dec, x_mark_dec)
        anomalies = (scores > self.threshold).long()

        if return_scores:
            return anomalies, scores
        return anomalies


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple methods
    """

    def __init__(
        self,
        detectors: List,
        voting_strategy: str = 'majority'
    ):
        """
        Args:
            detectors: List of anomaly detectors
            voting_strategy: 'majority', 'unanimous', or 'any'
        """
        self.detectors = detectors
        self.voting_strategy = voting_strategy

    def detect(
        self,
        *args,
        return_scores: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect anomalies using ensemble

        Args:
            *args, **kwargs: Arguments for detector
            return_scores: Return aggregated scores

        Returns:
            anomalies: Binary array
            scores (optional): Aggregated scores
        """
        all_anomalies = []
        all_scores = []

        for detector in self.detectors:
            if return_scores:
                anomalies, scores = detector.detect(*args, return_scores=True, **kwargs)
                all_scores.append(scores)
            else:
                anomalies = detector.detect(*args, **kwargs)

            # Convert to numpy if tensor
            if isinstance(anomalies, torch.Tensor):
                anomalies = anomalies.cpu().numpy()

            all_anomalies.append(anomalies)

        # Stack predictions
        all_anomalies = np.stack(all_anomalies, axis=0)

        # Voting
        if self.voting_strategy == 'majority':
            final_anomalies = (np.mean(all_anomalies, axis=0) > 0.5).astype(int)
        elif self.voting_strategy == 'unanimous':
            final_anomalies = (np.mean(all_anomalies, axis=0) == 1.0).astype(int)
        elif self.voting_strategy == 'any':
            final_anomalies = (np.max(all_anomalies, axis=0) > 0).astype(int)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

        if return_scores:
            # Average scores
            final_scores = np.mean(np.stack(all_scores, axis=0), axis=0)
            return final_anomalies, final_scores

        return final_anomalies


def detect_anomalies_sliding_window(
    data: np.ndarray,
    window_size: int = 100,
    method: str = 'threshold',
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using sliding window approach

    Args:
        data: Time series data [N]
        window_size: Window size for local statistics
        method: Detection method
        threshold: Threshold value

    Returns:
        anomalies: Binary array
        scores: Anomaly scores
    """
    n = len(data)
    anomalies = np.zeros(n, dtype=int)
    scores = np.zeros(n)

    detector = AnomalyDetector(method=method, threshold=threshold)

    for i in range(window_size, n):
        # Get window
        window = data[i - window_size:i].reshape(-1, 1)

        # Fit on window
        detector.fit(window)

        # Detect on current point
        current = data[i:i+1].reshape(-1, 1)
        anomaly, score = detector.detect(current, return_scores=True)

        anomalies[i] = anomaly[0]
        scores[i] = score[0]

    return anomalies, scores


def detect_contextual_anomalies(
    data: np.ndarray,
    seasonal_period: int = 24,
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect contextual anomalies considering seasonal patterns

    Args:
        data: Time series data [N]
        seasonal_period: Seasonal period
        threshold: Threshold for z-score

    Returns:
        anomalies: Binary array
        scores: Anomaly scores
    """
    n = len(data)
    anomalies = np.zeros(n, dtype=int)
    scores = np.zeros(n)

    # Compute seasonal baseline
    for i in range(seasonal_period, n):
        # Get seasonal context (same position in previous cycles)
        seasonal_indices = np.arange(i % seasonal_period, i, seasonal_period)
        if len(seasonal_indices) == 0:
            continue

        seasonal_values = data[seasonal_indices]

        # Compute z-score
        mean = np.mean(seasonal_values)
        std = np.std(seasonal_values)

        if std > 0:
            score = np.abs((data[i] - mean) / std)
        else:
            score = 0.0

        scores[i] = score
        anomalies[i] = int(score > threshold)

    return anomalies, scores
