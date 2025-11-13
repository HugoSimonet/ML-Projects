"""Drift detection module for detecting data and model drift."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection."""
    feature: str
    drift_detected: bool
    drift_score: float
    threshold: float
    test_statistic: float
    p_value: Optional[float]
    drift_type: str  # 'data_drift' or 'concept_drift'
    timestamp: str


class DriftDetector:
    """Detects data drift and concept drift."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize drift detector.

        Args:
            config: Configuration dictionary containing:
                - drift_threshold: Threshold for drift detection
                - statistical_test: Statistical test to use
                - window_size: Size of reference window
        """
        self.config = config
        self.drift_threshold = config.get('drift_threshold', 0.05)
        self.statistical_test = config.get('statistical_test', 'ks')
        self.window_size = config.get('window_size', 1000)

        # Store reference data
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_predictions: Optional[np.ndarray] = None

        logger.info("Initialized DriftDetector")

    def set_reference_data(self, data: pd.DataFrame,
                          predictions: Optional[np.ndarray] = None):
        """Set reference data for drift detection.

        Args:
            data: Reference data (typically training data)
            predictions: Reference predictions (optional)
        """
        self.reference_data = data.copy()
        self.reference_predictions = predictions

        logger.info(f"Set reference data with {len(data)} samples")

    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> List[DriftResult]:
        """Detect data drift in features.

        Args:
            current_data: Current/production data
            features: Features to check (None for all)

        Returns:
            List of drift results for each feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")

        logger.info(f"Detecting data drift for {len(current_data)} samples")

        # Determine features to check
        if features is None:
            features = list(self.reference_data.columns)

        results = []

        for feature in features:
            if feature not in current_data.columns:
                logger.warning(f"Feature {feature} not in current data")
                continue

            if feature not in self.reference_data.columns:
                logger.warning(f"Feature {feature} not in reference data")
                continue

            # Get feature values
            reference_values = self.reference_data[feature].dropna()
            current_values = current_data[feature].dropna()

            # Check if numeric or categorical
            if pd.api.types.is_numeric_dtype(reference_values):
                drift_result = self._detect_numeric_drift(
                    feature, reference_values, current_values
                )
            else:
                drift_result = self._detect_categorical_drift(
                    feature, reference_values, current_values
                )

            results.append(drift_result)

        # Log summary
        drifted_features = [r.feature for r in results if r.drift_detected]
        if drifted_features:
            logger.warning(f"Drift detected in features: {drifted_features}")

        return results

    def _detect_numeric_drift(
        self,
        feature: str,
        reference_values: pd.Series,
        current_values: pd.Series
    ) -> DriftResult:
        """Detect drift in numeric feature."""

        if self.statistical_test == 'ks':
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(reference_values, current_values)
            drift_detected = p_value < self.drift_threshold

        elif self.statistical_test == 'mann_whitney':
            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                reference_values, current_values, alternative='two-sided'
            )
            drift_detected = p_value < self.drift_threshold

        elif self.statistical_test == 'wasserstein':
            # Wasserstein distance
            statistic = stats.wasserstein_distance(reference_values, current_values)
            p_value = None

            # Normalize by standard deviation
            drift_score = statistic / (reference_values.std() + 1e-10)
            drift_detected = drift_score > 1.0

        else:
            raise ValueError(f"Unknown statistical test: {self.statistical_test}")

        return DriftResult(
            feature=feature,
            drift_detected=drift_detected,
            drift_score=statistic,
            threshold=self.drift_threshold,
            test_statistic=statistic,
            p_value=p_value,
            drift_type='data_drift',
            timestamp=datetime.now().isoformat()
        )

    def _detect_categorical_drift(
        self,
        feature: str,
        reference_values: pd.Series,
        current_values: pd.Series
    ) -> DriftResult:
        """Detect drift in categorical feature."""

        # Get value distributions
        ref_dist = reference_values.value_counts(normalize=True)
        curr_dist = current_values.value_counts(normalize=True)

        # Align distributions
        all_categories = set(ref_dist.index) | set(curr_dist.index)
        ref_probs = np.array([ref_dist.get(cat, 0) for cat in all_categories])
        curr_probs = np.array([curr_dist.get(cat, 0) for cat in all_categories])

        # Chi-square test
        if self.statistical_test == 'chi2':
            # Create contingency table
            ref_counts = reference_values.value_counts()
            curr_counts = current_values.value_counts()

            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_c = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_c = [curr_counts.get(cat, 0) for cat in all_categories]

            contingency = np.array([ref_c, curr_c])
            statistic, p_value, _, _ = stats.chi2_contingency(contingency)
            drift_detected = p_value < self.drift_threshold

        else:
            # Jensen-Shannon divergence
            statistic = self._jensen_shannon_divergence(ref_probs, curr_probs)
            p_value = None
            drift_detected = statistic > 0.1  # Threshold for JS divergence

        return DriftResult(
            feature=feature,
            drift_detected=drift_detected,
            drift_score=statistic,
            threshold=self.drift_threshold,
            test_statistic=statistic,
            p_value=p_value,
            drift_type='data_drift',
            timestamp=datetime.now().isoformat()
        )

    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # Calculate JS divergence
        m = (p + q) / 2
        js_div = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)

        return js_div

    def detect_concept_drift(
        self,
        current_predictions: np.ndarray,
        current_labels: np.ndarray
    ) -> DriftResult:
        """Detect concept drift (change in model performance).

        Args:
            current_predictions: Current model predictions
            current_labels: Current ground truth labels

        Returns:
            Drift result
        """
        if self.reference_predictions is None:
            logger.warning("No reference predictions available for concept drift detection")
            return DriftResult(
                feature='model_predictions',
                drift_detected=False,
                drift_score=0.0,
                threshold=self.drift_threshold,
                test_statistic=0.0,
                p_value=None,
                drift_type='concept_drift',
                timestamp=datetime.now().isoformat()
            )

        logger.info("Detecting concept drift")

        # Calculate error rates
        current_errors = (current_predictions != current_labels).astype(int)

        # Use proportion test
        current_error_rate = current_errors.mean()

        # Simple threshold-based detection
        # In production, you might use more sophisticated methods
        drift_detected = abs(current_error_rate - 0.1) > 0.05  # Example threshold

        return DriftResult(
            feature='model_predictions',
            drift_detected=drift_detected,
            drift_score=current_error_rate,
            threshold=0.05,
            test_statistic=current_error_rate,
            p_value=None,
            drift_type='concept_drift',
            timestamp=datetime.now().isoformat()
        )

    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray
    ) -> DriftResult:
        """Detect drift in prediction distribution.

        Args:
            current_predictions: Current model predictions

        Returns:
            Drift result
        """
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not set")

        logger.info("Detecting prediction drift")

        # Compare prediction distributions
        if np.issubdtype(current_predictions.dtype, np.number):
            statistic, p_value = stats.ks_2samp(
                self.reference_predictions,
                current_predictions
            )
            drift_detected = p_value < self.drift_threshold
        else:
            # Categorical predictions
            ref_dist = pd.Series(self.reference_predictions).value_counts(normalize=True)
            curr_dist = pd.Series(current_predictions).value_counts(normalize=True)

            all_classes = set(ref_dist.index) | set(curr_dist.index)
            ref_probs = np.array([ref_dist.get(c, 0) for c in all_classes])
            curr_probs = np.array([curr_dist.get(c, 0) for c in all_classes])

            statistic = self._jensen_shannon_divergence(ref_probs, curr_probs)
            p_value = None
            drift_detected = statistic > 0.1

        return DriftResult(
            feature='predictions',
            drift_detected=drift_detected,
            drift_score=statistic,
            threshold=self.drift_threshold,
            test_statistic=statistic,
            p_value=p_value,
            drift_type='prediction_drift',
            timestamp=datetime.now().isoformat()
        )

    def get_drift_summary(self, results: List[DriftResult]) -> Dict[str, Any]:
        """Get summary of drift detection results.

        Args:
            results: List of drift results

        Returns:
            Summary dictionary
        """
        drifted = [r for r in results if r.drift_detected]

        summary = {
            'total_features': len(results),
            'drifted_features': len(drifted),
            'drift_percentage': len(drifted) / len(results) if results else 0,
            'drifted_feature_names': [r.feature for r in drifted],
            'max_drift_score': max([r.drift_score for r in results]) if results else 0,
            'avg_drift_score': np.mean([r.drift_score for r in results]) if results else 0,
            'timestamp': datetime.now().isoformat()
        }

        return summary

    def get_feature_importance_shift(
        self,
        model: Any,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Detect shifts in feature importance.

        Args:
            model: Model with feature_importances_ attribute
            reference_data: Reference data
            current_data: Current data

        Returns:
            Dictionary of feature importance shifts
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_")
            return {}

        logger.info("Detecting feature importance shift")

        # This is a simplified version
        # In production, you would retrain on current data and compare
        importance = model.feature_importances_
        feature_names = reference_data.columns

        shifts = {}
        for i, feature in enumerate(feature_names):
            if i < len(importance):
                shifts[feature] = float(importance[i])

        return shifts
