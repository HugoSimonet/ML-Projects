"""A/B testing framework for model comparison and experimentation."""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from scipy import stats
import random


logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class ABTestResult:
    """Result of A/B test."""
    test_id: str
    control_model: str
    treatment_model: str
    metric_name: str
    control_metric: float
    treatment_metric: float
    improvement: float
    improvement_pct: float
    p_value: float
    is_significant: bool
    confidence_level: float
    sample_size_control: int
    sample_size_treatment: int
    test_duration: str
    winner: Optional[str]


@dataclass
class ExperimentMetrics:
    """Metrics collected during experiment."""
    model_version: str
    predictions: List[Any] = field(default_factory=list)
    ground_truth: List[Any] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    errors: List[bool] = field(default_factory=list)
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)


class ABTest:
    """A/B testing framework for model comparison."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize A/B test.

        Args:
            config: Configuration dictionary containing:
                - test_id: Unique test identifier
                - control_model: Version of control model
                - treatment_model: Version of treatment model
                - traffic_split: Traffic split ratio (default 0.5)
                - success_metric: Metric to optimize
                - confidence_level: Confidence level for significance (default 0.95)
                - min_sample_size: Minimum sample size per variant
        """
        self.config = config
        self.test_id = config.get('test_id', f'test_{datetime.now().strftime("%Y%m%d%H%M%S")}')
        self.control_model = config['control_model']
        self.treatment_model = config['treatment_model']
        self.traffic_split = config.get('traffic_split', 0.5)
        self.success_metric = config['success_metric']
        self.confidence_level = config.get('confidence_level', 0.95)
        self.min_sample_size = config.get('min_sample_size', 100)

        # Test state
        self.status = TestStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Metrics collection
        self.control_metrics = ExperimentMetrics(model_version=self.control_model)
        self.treatment_metrics = ExperimentMetrics(model_version=self.treatment_model)

        logger.info(f"Initialized A/B test {self.test_id}: "
                   f"{self.control_model} vs {self.treatment_model}")

    def start(self):
        """Start the A/B test."""
        if self.status == TestStatus.RUNNING:
            logger.warning("Test already running")
            return

        self.status = TestStatus.RUNNING
        self.start_time = datetime.now()

        logger.info(f"Started A/B test {self.test_id}")

    def stop(self):
        """Stop the A/B test."""
        if self.status != TestStatus.RUNNING:
            logger.warning("Test not running")
            return

        self.status = TestStatus.STOPPED
        self.end_time = datetime.now()

        logger.info(f"Stopped A/B test {self.test_id}")

    def assign_variant(self, user_id: Optional[str] = None) -> str:
        """Assign a user to control or treatment.

        Args:
            user_id: User identifier (for consistent assignment)

        Returns:
            Variant name ('control' or 'treatment')
        """
        if user_id:
            # Deterministic assignment based on user_id
            hash_val = hash(user_id)
            assignment_prob = (hash_val % 1000) / 1000
        else:
            # Random assignment
            assignment_prob = random.random()

        variant = 'control' if assignment_prob < self.traffic_split else 'treatment'
        return variant

    def record_prediction(
        self,
        variant: str,
        prediction: Any,
        ground_truth: Optional[Any] = None,
        latency: Optional[float] = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """Record a prediction for a variant.

        Args:
            variant: Variant name ('control' or 'treatment')
            prediction: Model prediction
            ground_truth: Actual label (if available)
            latency: Prediction latency
            custom_metrics: Additional custom metrics
        """
        metrics = (self.control_metrics if variant == 'control'
                  else self.treatment_metrics)

        metrics.predictions.append(prediction)

        if ground_truth is not None:
            metrics.ground_truth.append(ground_truth)
            error = prediction != ground_truth
            metrics.errors.append(error)

        if latency is not None:
            metrics.latencies.append(latency)

        if custom_metrics:
            for metric_name, value in custom_metrics.items():
                if metric_name not in metrics.custom_metrics:
                    metrics.custom_metrics[metric_name] = []
                metrics.custom_metrics[metric_name].append(value)

    def get_results(self) -> ABTestResult:
        """Calculate and return A/B test results.

        Returns:
            Test results with statistical analysis
        """
        logger.info(f"Calculating results for A/B test {self.test_id}")

        # Calculate metric values
        control_value = self._calculate_metric(self.control_metrics)
        treatment_value = self._calculate_metric(self.treatment_metrics)

        # Calculate improvement
        improvement = treatment_value - control_value
        improvement_pct = (improvement / control_value * 100) if control_value != 0 else 0

        # Perform statistical test
        p_value = self._statistical_test(self.control_metrics, self.treatment_metrics)

        # Determine significance
        alpha = 1 - self.confidence_level
        is_significant = p_value < alpha

        # Determine winner
        winner = None
        if is_significant:
            winner = ('treatment' if improvement > 0 else 'control')

        # Calculate duration
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        elif self.start_time:
            duration = datetime.now() - self.start_time
        else:
            duration = timedelta(0)

        result = ABTestResult(
            test_id=self.test_id,
            control_model=self.control_model,
            treatment_model=self.treatment_model,
            metric_name=self.success_metric,
            control_metric=control_value,
            treatment_metric=treatment_value,
            improvement=improvement,
            improvement_pct=improvement_pct,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            sample_size_control=len(self.control_metrics.predictions),
            sample_size_treatment=len(self.treatment_metrics.predictions),
            test_duration=str(duration),
            winner=winner
        )

        logger.info(f"Results: {self.control_model}={control_value:.4f}, "
                   f"{self.treatment_model}={treatment_value:.4f}, "
                   f"improvement={improvement_pct:.2f}%, "
                   f"significant={is_significant}")

        return result

    def _calculate_metric(self, metrics: ExperimentMetrics) -> float:
        """Calculate the success metric value.

        Args:
            metrics: Experiment metrics

        Returns:
            Metric value
        """
        if self.success_metric == 'accuracy':
            if not metrics.ground_truth:
                return 0.0
            correct = sum(1 for p, gt in zip(metrics.predictions, metrics.ground_truth)
                         if p == gt)
            return correct / len(metrics.ground_truth)

        elif self.success_metric == 'error_rate':
            if not metrics.errors:
                return 0.0
            return sum(metrics.errors) / len(metrics.errors)

        elif self.success_metric == 'latency':
            if not metrics.latencies:
                return 0.0
            return np.mean(metrics.latencies)

        elif self.success_metric in metrics.custom_metrics:
            values = metrics.custom_metrics[self.success_metric]
            return np.mean(values) if values else 0.0

        else:
            raise ValueError(f"Unknown metric: {self.success_metric}")

    def _statistical_test(
        self,
        control_metrics: ExperimentMetrics,
        treatment_metrics: ExperimentMetrics
    ) -> float:
        """Perform statistical significance test.

        Args:
            control_metrics: Control group metrics
            treatment_metrics: Treatment group metrics

        Returns:
            p-value
        """
        # Extract metric values
        control_values = self._get_metric_values(control_metrics)
        treatment_values = self._get_metric_values(treatment_metrics)

        if len(control_values) == 0 or len(treatment_values) == 0:
            return 1.0  # No significance if no data

        # Perform t-test
        try:
            statistic, p_value = stats.ttest_ind(control_values, treatment_values)
            return p_value
        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return 1.0

    def _get_metric_values(self, metrics: ExperimentMetrics) -> np.ndarray:
        """Get metric values as array.

        Args:
            metrics: Experiment metrics

        Returns:
            Array of metric values
        """
        if self.success_metric == 'accuracy':
            if not metrics.ground_truth:
                return np.array([])
            correct = [1 if p == gt else 0
                      for p, gt in zip(metrics.predictions, metrics.ground_truth)]
            return np.array(correct)

        elif self.success_metric == 'error_rate':
            return np.array(metrics.errors, dtype=float)

        elif self.success_metric == 'latency':
            return np.array(metrics.latencies)

        elif self.success_metric in metrics.custom_metrics:
            return np.array(metrics.custom_metrics[self.success_metric])

        else:
            return np.array([])

    def is_ready_for_decision(self) -> bool:
        """Check if test has enough data for decision.

        Returns:
            True if ready for decision
        """
        control_size = len(self.control_metrics.predictions)
        treatment_size = len(self.treatment_metrics.predictions)

        return (control_size >= self.min_sample_size and
                treatment_size >= self.min_sample_size)

    def calculate_sample_size(
        self,
        baseline_conversion: float,
        minimum_detectable_effect: float,
        power: float = 0.8
    ) -> int:
        """Calculate required sample size for test.

        Args:
            baseline_conversion: Baseline conversion rate
            minimum_detectable_effect: Minimum effect to detect
            power: Statistical power (default 0.8)

        Returns:
            Required sample size per variant
        """
        alpha = 1 - self.confidence_level

        # Calculate effect size (Cohen's h)
        p1 = baseline_conversion
        p2 = baseline_conversion * (1 + minimum_detectable_effect)

        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        # Calculate sample size using power analysis
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    def get_summary(self) -> Dict[str, Any]:
        """Get test summary.

        Returns:
            Summary dictionary
        """
        summary = {
            'test_id': self.test_id,
            'status': self.status.value,
            'control_model': self.control_model,
            'treatment_model': self.treatment_model,
            'traffic_split': self.traffic_split,
            'success_metric': self.success_metric,
            'control_samples': len(self.control_metrics.predictions),
            'treatment_samples': len(self.treatment_metrics.predictions),
            'is_ready': self.is_ready_for_decision(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }

        return summary


class MultiVariantTest(ABTest):
    """Multi-variant testing (A/B/n testing)."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-variant test.

        Args:
            config: Configuration with variants list
        """
        super().__init__(config)

        self.variants = config.get('variants', [])
        self.variant_metrics = {
            variant: ExperimentMetrics(model_version=variant)
            for variant in self.variants
        }

        logger.info(f"Initialized multi-variant test with {len(self.variants)} variants")

    def assign_variant(self, user_id: Optional[str] = None) -> str:
        """Assign user to a variant.

        Args:
            user_id: User identifier

        Returns:
            Assigned variant
        """
        if not self.variants:
            raise ValueError("No variants defined")

        if user_id:
            hash_val = hash(user_id)
            idx = hash_val % len(self.variants)
        else:
            idx = random.randint(0, len(self.variants) - 1)

        return self.variants[idx]

    def record_prediction(
        self,
        variant: str,
        prediction: Any,
        ground_truth: Optional[Any] = None,
        latency: Optional[float] = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """Record prediction for variant."""
        if variant not in self.variant_metrics:
            raise ValueError(f"Unknown variant: {variant}")

        metrics = self.variant_metrics[variant]

        metrics.predictions.append(prediction)

        if ground_truth is not None:
            metrics.ground_truth.append(ground_truth)
            error = prediction != ground_truth
            metrics.errors.append(error)

        if latency is not None:
            metrics.latencies.append(latency)

        if custom_metrics:
            for metric_name, value in custom_metrics.items():
                if metric_name not in metrics.custom_metrics:
                    metrics.custom_metrics[metric_name] = []
                metrics.custom_metrics[metric_name].append(value)

    def get_results(self) -> Dict[str, Any]:
        """Get results for all variants."""
        results = {}

        for variant in self.variants:
            metrics = self.variant_metrics[variant]
            metric_value = self._calculate_metric(metrics)

            results[variant] = {
                'metric_value': metric_value,
                'sample_size': len(metrics.predictions)
            }

        # Find best variant
        best_variant = max(results.keys(), key=lambda v: results[v]['metric_value'])
        results['best_variant'] = best_variant

        return results
