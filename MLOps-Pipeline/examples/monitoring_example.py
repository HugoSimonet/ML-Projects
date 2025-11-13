"""Monitoring and drift detection example."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring import PerformanceMonitor, DriftDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate monitoring and drift detection."""
    logger.info("="*60)
    logger.info("Monitoring and Drift Detection Example")
    logger.info("="*60)

    # Configuration
    monitor_config = {
        'window_size': 1000,
        'alert_thresholds': {
            'error_rate': 0.1,
            'latency_ms': 1000,
            'cpu_usage': 80
        }
    }

    drift_config = {
        'drift_threshold': 0.05,
        'statistical_test': 'ks',
        'window_size': 1000
    }

    # Initialize monitoring
    logger.info("\n1. Initializing Monitoring Components...")
    monitor = PerformanceMonitor(monitor_config)
    drift_detector = DriftDetector(drift_config)

    # Generate sample data
    logger.info("\n2. Generating Sample Data...")
    np.random.seed(42)

    # Training data
    X_train = pd.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'feature_3': np.random.randn(1000)
    })
    y_train = np.random.randint(0, 2, 1000)

    # Production data (slightly different distribution)
    X_prod = pd.DataFrame({
        'feature_1': np.random.randn(500) + 0.5,  # Shifted distribution
        'feature_2': np.random.randn(500),
        'feature_3': np.random.randn(500) * 1.5  # Different variance
    })
    y_prod = np.random.randint(0, 2, 500)

    # Train a simple model
    logger.info("\n3. Training Model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Set reference data for drift detection
    logger.info("\n4. Setting Reference Data...")
    drift_detector.set_reference_data(X_train, y_train)

    # Simulate production predictions
    logger.info("\n5. Simulating Production Predictions...")
    for i in range(len(X_prod)):
        # Make prediction
        pred = model.predict([X_prod.iloc[i]])[0]
        true_label = y_prod[i]

        # Simulate latency
        latency = np.random.uniform(10, 100)

        # Record prediction
        monitor.record_prediction(
            prediction=pred,
            ground_truth=true_label,
            latency_ms=latency,
            model_version='v1'
        )

    # Get performance summary
    logger.info("\n6. Performance Summary...")
    summary = monitor.get_performance_summary(time_window_minutes=None)

    logger.info(f"Total Predictions: {summary['total_predictions']}")
    logger.info(f"Error Rate: {summary['error_rate']:.4f}")
    logger.info(f"Average Latency: {summary['avg_latency_ms']:.2f}ms")
    logger.info(f"P50 Latency: {summary['p50_latency_ms']:.2f}ms")
    logger.info(f"P95 Latency: {summary['p95_latency_ms']:.2f}ms")
    logger.info(f"P99 Latency: {summary['p99_latency_ms']:.2f}ms")

    # Check for alerts
    logger.info("\n7. Checking for Alerts...")
    alerts = monitor.check_alerts()

    if alerts:
        logger.warning(f"Found {len(alerts)} alert(s):")
        for alert in alerts:
            logger.warning(f"  - {alert['type']}: {alert['message']}")
    else:
        logger.info("No alerts triggered")

    # Detect data drift
    logger.info("\n8. Detecting Data Drift...")
    drift_results = drift_detector.detect_data_drift(
        current_data=X_prod,
        features=['feature_1', 'feature_2', 'feature_3']
    )

    logger.info("\nDrift Detection Results:")
    for result in drift_results:
        status = "DETECTED" if result.drift_detected else "OK"
        logger.info(f"  {result.feature}: {status} (score={result.drift_score:.4f})")

    # Get drift summary
    drift_summary = drift_detector.get_drift_summary(drift_results)
    logger.info(f"\nDrift Summary:")
    logger.info(f"  Features Checked: {drift_summary['total_features']}")
    logger.info(f"  Drifted Features: {drift_summary['drifted_features']}")
    logger.info(f"  Drift Percentage: {drift_summary['drift_percentage']:.2%}")

    if drift_summary['drifted_features'] > 0:
        logger.warning(f"  Features with drift: {drift_summary['drifted_feature_names']}")

    # Collect system metrics
    logger.info("\n9. Collecting System Metrics...")
    system_metrics = monitor.collect_system_metrics()

    logger.info(f"System Metrics:")
    logger.info(f"  CPU Usage: {system_metrics.cpu_usage_percent:.1f}%")
    logger.info(f"  Memory Usage: {system_metrics.memory_usage_mb:.1f} MB")
    logger.info(f"  Disk Usage: {system_metrics.disk_usage_percent:.1f}%")

    # Export metrics
    logger.info("\n10. Exporting Metrics...")
    monitor.export_metrics('monitoring_metrics.json')
    logger.info("Metrics exported to monitoring_metrics.json")

    logger.info("\n" + "="*60)
    logger.info("Monitoring Example Completed!")
    logger.info("="*60)

    # Recommendations
    logger.info("\nRecommendations:")
    if drift_summary['drift_percentage'] > 0.3:
        logger.warning("  - High drift detected! Consider retraining the model.")
    if summary['error_rate'] > 0.1:
        logger.warning("  - Error rate is high! Investigate model performance.")
    if summary['p95_latency_ms'] > 500:
        logger.warning("  - High latency detected! Consider optimization.")

    logger.info("\n  - Continue monitoring for production deployment")
    logger.info("  - Set up automated alerts for critical metrics")
    logger.info("  - Schedule regular drift detection checks")


if __name__ == "__main__":
    main()
