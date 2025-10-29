"""
Demonstration examples for the anomaly detection system
"""

import numpy as np
from typing import Tuple

from core.system import AnomalyDetectionSystem
from models.statistical_methods import IsolationForestDetector, LocalOutlierFactor, StatisticalProcessControl
from models.ensemble_methods import EnsembleDetector
from streaming.real_time_detection import StreamingAnomalyDetector
from evaluation.anomaly_metrics import AnomalyMetrics
from visualization.anomaly_visualizer import AnomalyVisualizer


def generate_synthetic_data(n_samples: int = 1000, 
                          n_features: int = 20,
                          contamination: float = 0.1,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with anomalies"""
    np.random.seed(random_state)
    
    # Normal data
    n_normal = int(n_samples * (1 - contamination))
    X_normal = np.random.randn(n_normal, n_features)
    y_normal = np.zeros(n_normal)
    
    # Anomalies (shifted distribution)
    n_anomalies = n_samples - n_normal
    X_anomalies = np.random.randn(n_anomalies, n_features) * 3 + 5
    y_anomalies = np.ones(n_anomalies)
    
    # Combine and shuffle
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([y_normal, y_anomalies])
    
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def demo_isolation_forest():
    """Demonstrate Isolation Forest"""
    print("\n" + "=" * 70)
    print("DEMO: ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 70)
    
    # Generate data
    X_train, _ = generate_synthetic_data(n_samples=500, contamination=0.1)
    X_test, y_test = generate_synthetic_data(n_samples=200, contamination=0.15)
    
    # Create feature names
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    # Initialize system
    system = AnomalyDetectionSystem(
        detector_type='isolation_forest',
        contamination=0.1,
        n_estimators=100
    )
    
    # Fit
    print("\nTraining Isolation Forest...")
    system.fit(X_train, feature_names=feature_names)
    
    # Detect
    print("Detecting anomalies...")
    result = system.detect(X_test, explain_anomalies=True)
    
    # Visualize
    visualizer = AnomalyVisualizer()
    visualizer.print_detection_summary(result)
    
    # Evaluate
    print("\nEvaluating performance...")
    metrics = system.evaluate(X_test, y_test)
    visualizer.print_evaluation_metrics(metrics)
    visualizer.plot_score_distribution(result.scores, result.labels)


def demo_lof():
    """Demonstrate Local Outlier Factor"""
    print("\n" + "=" * 70)
    print("DEMO: LOCAL OUTLIER FACTOR (LOF)")
    print("=" * 70)
    
    # Generate data
    X_train, _ = generate_synthetic_data(n_samples=300, n_features=10)
    X_test, y_test = generate_synthetic_data(n_samples=100, n_features=10)
    
    # Initialize system
    system = AnomalyDetectionSystem(
        detector_type='lof',
        contamination=0.1,
        n_neighbors=20
    )
    
    # Fit and detect
    print("\nTraining LOF detector...")
    system.fit(X_train)
    result = system.detect(X_test)
    
    # Results
    visualizer = AnomalyVisualizer()
    visualizer.print_detection_summary(result)
    
    metrics = system.evaluate(X_test, y_test)
    visualizer.print_evaluation_metrics(metrics)


def demo_autoencoder():
    """Demonstrate Autoencoder-based detection"""
    print("\n" + "=" * 70)
    print("DEMO: AUTOENCODER ANOMALY DETECTION")
    print("=" * 70)
    
    # Generate data
    X_train, _ = generate_synthetic_data(n_samples=400, n_features=15)
    X_test, y_test = generate_synthetic_data(n_samples=100, n_features=15)
    
    # Initialize system
    system = AnomalyDetectionSystem(
        detector_type='autoencoder',
        contamination=0.1,
        hidden_dims=[32, 16, 8],
        epochs=50
    )
    
    # Fit and detect
    print("\nTraining Autoencoder (this may take a moment)...")
    system.fit(X_train)
    result = system.detect(X_test)
    
    # Results
    visualizer = AnomalyVisualizer()
    visualizer.print_detection_summary(result)
    
    metrics = system.evaluate(X_test, y_test)
    visualizer.print_evaluation_metrics(metrics)


def demo_ensemble():
    """Demonstrate Ensemble detection"""
    print("\n" + "=" * 70)
    print("DEMO: ENSEMBLE ANOMALY DETECTION")
    print("=" * 70)
    
    # Generate data
    X_train, _ = generate_synthetic_data(n_samples=500, n_features=12)
    X_test, y_test = generate_synthetic_data(n_samples=150, n_features=12)
    
    # Create ensemble
    detectors = [
        IsolationForestDetector(contamination=0.1, n_estimators=50),
        LocalOutlierFactor(contamination=0.1, n_neighbors=15),
        StatisticalProcessControl(contamination=0.1)
    ]
    
    ensemble = EnsembleDetector(
        detectors=detectors,
        combination_method='average',
        contamination=0.1
    )
    
    # Fit and predict
    print("\nTraining ensemble of 3 detectors...")
    ensemble.fit(X_train)
    
    scores = ensemble.score_samples(X_test)
    labels = ensemble.predict(X_test)
    
    # Results
    metrics = AnomalyMetrics()
    prf = metrics.precision_recall_f1(y_test, labels)
    auc = metrics.auc_roc(y_test, scores)
    
    print(f"\nEnsemble Results:")
    print(f"  Precision: {prf['precision']:.4f}")
    print(f"  Recall: {prf['recall']:.4f}")
    print(f"  F1-Score: {prf['f1_score']:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")


def demo_streaming():
    """Demonstrate streaming detection"""
    print("\n" + "=" * 70)
    print("DEMO: REAL-TIME STREAMING ANOMALY DETECTION")
    print("=" * 70)
    
    # Generate streaming data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    # Create streaming detector
    streaming = StreamingAnomalyDetector(
        window_size=100,
        detector=IsolationForestDetector(contamination=0.1),
        update_frequency=50
    )
    
    print("\nProcessing streaming data...")
    print("-" * 70)
    
    anomalies_detected = []
    
    for i in range(n_samples):
        # Generate sample (mostly normal, some anomalies)
        if i % 20 == 0:  # Inject anomaly
            sample = np.random.randn(n_features) * 3 + 5
        else:
            sample = np.random.randn(n_features)
        
        score, is_anomaly = streaming.process_sample(sample)
        
        if is_anomaly:
            anomalies_detected.append(i)
            print(f"Sample {i:3d}: ANOMALY DETECTED (score: {score:.4f})")
    
    print("-" * 70)
    print(f"\nTotal anomalies detected: {len(anomalies_detected)}")
    print(f"Anomaly indices: {anomalies_detected}")


def demo_high_dimensional():
    """Demonstrate high-dimensional anomaly detection"""
    print("\n" + "=" * 70)
    print("DEMO: HIGH-DIMENSIONAL ANOMALY DETECTION")
    print("=" * 70)
    
    # Generate high-dimensional data
    X_train, _ = generate_synthetic_data(n_samples=500, n_features=100)
    X_test, y_test = generate_synthetic_data(n_samples=150, n_features=100)
    
    print(f"\nOriginal dimensions: {X_train.shape[1]}")
    
    # Initialize system with dimensionality reduction
    system = AnomalyDetectionSystem(
        detector_type='isolation_forest',
        contamination=0.1
    )
    
    # Fit with dimensionality reduction
    print("Applying dimensionality reduction...")
    system.fit(X_train, dimensionality_reduction=30)
    
    # Detect
    result = system.detect(X_test)
    
    # Evaluate
    metrics = system.evaluate(X_test, y_test)
    
    visualizer = AnomalyVisualizer()
    visualizer.print_detection_summary(result)
    visualizer.print_evaluation_metrics(metrics)


def run_all_demos():
    """Run all demonstration examples"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ANOMALY DETECTION SYSTEM")
    print("High-Dimensional Data with Multiple Detection Methods")
    print("=" * 70)
    
    # Run all demonstrations
    demo_isolation_forest()
    demo_lof()
    demo_autoencoder()
    demo_ensemble()
    demo_streaming()
    demo_high_dimensional()
    
    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETED")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Isolation Forest for efficient detection")
    print("  ✓ Local Outlier Factor for density-based detection")
    print("  ✓ Autoencoder for deep learning-based detection")
    print("  ✓ Ensemble methods for robust detection")
    print("  ✓ Real-time streaming detection")
    print("  ✓ High-dimensional data processing")
    print("  ✓ Interpretability and explanations")
    print("  ✓ Comprehensive evaluation metrics")
    print("\nSystem Ready for Production Use!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_demos()