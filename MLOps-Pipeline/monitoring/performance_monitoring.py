"""Performance monitoring module for tracking model and system metrics."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import time
from prometheus_client import Gauge, Counter, Histogram, Summary


logger = logging.getLogger(__name__)


# Prometheus metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy', ['model_version'])
MODEL_LATENCY = Histogram('model_latency_seconds', 'Model prediction latency', ['model_version'])
MODEL_THROUGHPUT = Gauge('model_throughput', 'Predictions per second', ['model_version'])
MODEL_ERROR_RATE = Gauge('model_error_rate', 'Model error rate', ['model_version'])
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_mb', 'Memory usage in MB')
FEATURE_MISSING_RATE = Gauge('feature_missing_rate', 'Feature missing value rate', ['feature'])


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: str
    model_version: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput: Optional[float] = None
    error_rate: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """Container for system metrics."""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    network_io_mbps: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


class PerformanceMonitor:
    """Monitors model and system performance."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize performance monitor.

        Args:
            config: Configuration dictionary containing:
                - window_size: Size of monitoring window
                - alert_thresholds: Thresholds for alerts
                - metrics_storage: Storage for metrics
        """
        self.config = config
        self.window_size = config.get('window_size', 1000)
        self.alert_thresholds = config.get('alert_thresholds', {})

        # Metrics storage
        self.performance_history: deque = deque(maxlen=self.window_size)
        self.system_history: deque = deque(maxlen=self.window_size)
        self.prediction_history: deque = deque(maxlen=self.window_size)

        # Tracking
        self.start_time = time.time()
        self.total_predictions = 0
        self.total_errors = 0

        logger.info("Initialized PerformanceMonitor")

    def record_prediction(
        self,
        prediction: Any,
        ground_truth: Optional[Any] = None,
        latency_ms: Optional[float] = None,
        model_version: str = 'unknown'
    ):
        """Record a prediction for monitoring.

        Args:
            prediction: Model prediction
            ground_truth: Actual label (if available)
            latency_ms: Prediction latency
            model_version: Version of the model
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'latency_ms': latency_ms,
            'error': ground_truth is not None and prediction != ground_truth
        }

        self.prediction_history.append(record)
        self.total_predictions += 1

        # Update Prometheus metrics
        if latency_ms is not None:
            MODEL_LATENCY.labels(model_version=model_version).observe(latency_ms / 1000)

        # Update error tracking
        if ground_truth is not None and prediction != ground_truth:
            self.total_errors += 1

    def record_performance_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics.

        Args:
            metrics: Performance metrics to record
        """
        self.performance_history.append(metrics)

        # Update Prometheus metrics
        if metrics.accuracy is not None:
            MODEL_ACCURACY.labels(model_version=metrics.model_version).set(metrics.accuracy)

        if metrics.throughput is not None:
            MODEL_THROUGHPUT.labels(model_version=metrics.model_version).set(metrics.throughput)

        if metrics.error_rate is not None:
            MODEL_ERROR_RATE.labels(model_version=metrics.model_version).set(metrics.error_rate)

        logger.debug(f"Recorded performance metrics: {metrics}")

    def record_system_metrics(self, metrics: SystemMetrics):
        """Record system metrics.

        Args:
            metrics: System metrics to record
        """
        self.system_history.append(metrics)

        # Update Prometheus metrics
        SYSTEM_CPU_USAGE.set(metrics.cpu_usage_percent)
        SYSTEM_MEMORY_USAGE.set(metrics.memory_usage_mb)

        logger.debug(f"Recorded system metrics: CPU={metrics.cpu_usage_percent}%, "
                    f"Memory={metrics.memory_usage_mb}MB")

    def get_performance_summary(
        self,
        time_window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get performance summary.

        Args:
            time_window_minutes: Time window for summary (None for all)

        Returns:
            Performance summary
        """
        # Filter by time window if specified
        predictions = self._filter_by_time(
            self.prediction_history,
            time_window_minutes
        )

        if not predictions:
            return {
                'total_predictions': 0,
                'error_rate': 0.0,
                'avg_latency_ms': 0.0
            }

        # Calculate metrics
        total = len(predictions)
        errors = sum(1 for p in predictions if p.get('error', False))
        latencies = [p['latency_ms'] for p in predictions if p.get('latency_ms')]

        summary = {
            'total_predictions': total,
            'error_rate': errors / total if total > 0 else 0.0,
            'avg_latency_ms': np.mean(latencies) if latencies else 0.0,
            'p50_latency_ms': np.percentile(latencies, 50) if latencies else 0.0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0.0,
            'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0.0,
            'throughput': total / ((time_window_minutes or 1) * 60) if time_window_minutes else 0.0
        }

        return summary

    def get_system_summary(
        self,
        time_window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get system metrics summary.

        Args:
            time_window_minutes: Time window for summary

        Returns:
            System metrics summary
        """
        metrics = self._filter_by_time(
            self.system_history,
            time_window_minutes
        )

        if not metrics:
            return {
                'avg_cpu_usage': 0.0,
                'avg_memory_usage': 0.0
            }

        cpu_usage = [m.cpu_usage_percent for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics]

        summary = {
            'avg_cpu_usage': np.mean(cpu_usage),
            'max_cpu_usage': np.max(cpu_usage),
            'avg_memory_usage': np.mean(memory_usage),
            'max_memory_usage': np.max(memory_usage)
        }

        return summary

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check if any alert conditions are met.

        Returns:
            List of triggered alerts
        """
        alerts = []

        # Get recent performance
        summary = self.get_performance_summary(time_window_minutes=5)

        # Check error rate
        error_threshold = self.alert_thresholds.get('error_rate', 0.1)
        if summary['error_rate'] > error_threshold:
            alerts.append({
                'type': 'error_rate',
                'severity': 'high',
                'message': f"Error rate {summary['error_rate']:.2%} exceeds "
                          f"threshold {error_threshold:.2%}",
                'value': summary['error_rate'],
                'threshold': error_threshold,
                'timestamp': datetime.now().isoformat()
            })

        # Check latency
        latency_threshold = self.alert_thresholds.get('latency_ms', 1000)
        if summary['p95_latency_ms'] > latency_threshold:
            alerts.append({
                'type': 'latency',
                'severity': 'medium',
                'message': f"P95 latency {summary['p95_latency_ms']:.2f}ms exceeds "
                          f"threshold {latency_threshold}ms",
                'value': summary['p95_latency_ms'],
                'threshold': latency_threshold,
                'timestamp': datetime.now().isoformat()
            })

        # Check system metrics
        system_summary = self.get_system_summary(time_window_minutes=5)

        cpu_threshold = self.alert_thresholds.get('cpu_usage', 80)
        if system_summary.get('avg_cpu_usage', 0) > cpu_threshold:
            alerts.append({
                'type': 'cpu_usage',
                'severity': 'medium',
                'message': f"CPU usage {system_summary['avg_cpu_usage']:.1f}% exceeds "
                          f"threshold {cpu_threshold}%",
                'value': system_summary['avg_cpu_usage'],
                'threshold': cpu_threshold,
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def get_model_accuracy(
        self,
        model_version: str,
        time_window_minutes: Optional[int] = None
    ) -> float:
        """Calculate model accuracy over time window.

        Args:
            model_version: Version of model to check
            time_window_minutes: Time window

        Returns:
            Accuracy score
        """
        predictions = self._filter_by_time(
            self.prediction_history,
            time_window_minutes
        )

        # Filter by model version
        predictions = [p for p in predictions if p.get('model_version') == model_version]

        # Only consider predictions with ground truth
        predictions = [p for p in predictions if p.get('ground_truth') is not None]

        if not predictions:
            return 0.0

        correct = sum(1 for p in predictions
                     if p['prediction'] == p['ground_truth'])

        return correct / len(predictions)

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics.

        Returns:
            Current system metrics
        """
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()

            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_io_mbps=net_io.bytes_sent / (1024 * 1024)
            )

            # Try to get GPU metrics if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.gpu_usage_percent = gpu.load * 100
                    metrics.gpu_memory_mb = gpu.memoryUsed
            except ImportError:
                pass

            return metrics

        except ImportError:
            logger.warning("psutil not available, returning dummy metrics")
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                disk_usage_percent=0.0,
                network_io_mbps=0.0
            )

    def _filter_by_time(
        self,
        data: deque,
        time_window_minutes: Optional[int]
    ) -> List[Any]:
        """Filter data by time window.

        Args:
            data: Data to filter
            time_window_minutes: Time window in minutes

        Returns:
            Filtered data
        """
        if time_window_minutes is None:
            return list(data)

        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

        filtered = []
        for item in data:
            timestamp_str = None

            if isinstance(item, dict):
                timestamp_str = item.get('timestamp')
            elif hasattr(item, 'timestamp'):
                timestamp_str = item.timestamp

            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp >= cutoff_time:
                    filtered.append(item)

        return filtered

    def export_metrics(self, filepath: str):
        """Export metrics to file.

        Args:
            filepath: Path to export file
        """
        import json

        metrics = {
            'performance_history': [
                {
                    'timestamp': m.timestamp,
                    'model_version': m.model_version,
                    'accuracy': m.accuracy,
                    'latency_ms': m.latency_ms,
                    'throughput': m.throughput
                }
                for m in self.performance_history
            ],
            'summary': self.get_performance_summary(),
            'system_summary': self.get_system_summary()
        }

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Exported metrics to {filepath}")
