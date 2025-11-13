"""
Complete example: Testing anomaly detection on real datasets
Run this after setting up the project
"""

import numpy as np
import sys
sys.path.append('..')

from core.system import AnomalyDetectionSystem
from models.statistical_methods import IsolationForestDetector, LocalOutlierFactor
from models.ensemble_methods import EnsembleDetector
from evaluation.anomaly_metrics import AnomalyMetrics
from visualization.anomaly_visualizer import AnomalyVisualizer


# Import the dataset loader (copy the dataset_loader.py content to datasets/ folder)
# Or use the DatasetLoader class directly


def quick_test_iot_sensors():
    """Quick test with IoT sensor data (easiest to start with)"""
    
    print("\n" + "="*80)
    print("QUICK TEST: IoT SENSOR ANOMALY DETECTION")
    print("="*80)
    
    # Generate IoT sensor data
    print("\n1. Generating IoT sensor data...")
    np.random.seed(42)
    n_samples = 10000
    
    time = np.arange(n_samples)
    
    # Normal sensor readings
    temperature = 20 + 5 * np.sin(time * 2 * np.pi / 1000) + np.random.randn(n_samples) * 0.5
    humidity = 60 + 10 * np.cos(time * 2 * np.pi / 1500) + np.random.randn(n_samples) * 2
    pressure = 1013 + 3 * np.sin(time * 2 * np.pi / 800) + np.random.randn(n_samples) * 1
    vibration = 0.5 + 0.2 * np.sin(time * 2 * np.pi / 500) + np.random.randn(n_samples) * 0.1
    power = 100 + 20 * np.sin(time * 2 * np.pi / 2000) + np.random.randn(n_samples) * 5
    
    X = np.column_stack([
        temperature, humidity, pressure, vibration, power,
        np.gradient(temperature),
        np.gradient(humidity),
        np.gradient(pressure),
        np.gradient(vibration),
        np.gradient(power)
    ])
    
    # Inject anomalies
    y = np.zeros(n_samples)
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.randint(0, 5)
        if anomaly_type == 0:
            X[idx, 0] += np.random.uniform(15, 25)  # Temperature spike
        elif anomaly_type == 1:
            X[idx, 1] -= np.random.uniform(20, 40)  # Humidity drop
        elif anomaly_type == 2:
            X[idx, 2] += np.random.uniform(10, 20) * np.random.choice([-1, 1])
        elif anomaly_type == 3:
            X[idx, 3] += np.random.uniform(2, 5)  # Vibration spike
        else:
            X[idx, 4] += np.random.uniform(50, 100)  # Power surge
        y[idx] = 1
    
    feature_names = [
        'temperature', 'humidity', 'pressure', 'vibration', 'power',
        'temp_rate', 'humid_rate', 'press_rate', 'vibr_rate', 'power_rate'
    ]
    
    print(f"   ✓ Generated {n_samples:,} samples with {X.shape[1]} features")
    print(f"   ✓ Anomalies: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
    
    # Split data
    print("\n2. Splitting data (70% train, 30% test)...")
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   ✓ Train: {len(X_train):,} samples")
    print(f"   ✓ Test: {len(X_test):,} samples")
    
    # Test different detectors
    print("\n3. Testing different detectors...")
    print("-" * 80)
    
    detectors = {
        'Isolation Forest': 'isolation_forest',
        'LOF': 'lof',
        'Autoencoder': 'autoencoder'
    }
    
    results = {}
    
    for name, detector_type in detectors.items():
        print(f"\n{name}:")
        
        system = AnomalyDetectionSystem(
            detector_type=detector_type,
            contamination=np.mean(y_train),
            n_estimators=100 if detector_type == 'isolation_forest' else None,
            n_neighbors=20 if detector_type == 'lof' else None,
            hidden_dims=[32, 16, 8] if detector_type == 'autoencoder' else None,
            epochs=30 if detector_type == 'autoencoder' else None
        )
        
        print(f"  Training...")
        system.fit(X_train, feature_names=feature_names)
        
        print(f"  Detecting...")
        result = system.detect(X_test, explain_anomalies=True)
        
        print(f"  Evaluating...")
        metrics = system.evaluate(X_test, y_test)
        results[name] = metrics
        
        print(f"  ✓ Precision: {metrics['precision']:.4f}")
        print(f"  ✓ Recall: {metrics['recall']:.4f}")
        print(f"  ✓ F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ✓ AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # Compare results
    print("\n4. Results Comparison:")
    print("="*80)
    print(f"{'Detector':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC-ROC':<12}")
    print("-"*80)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {metrics['auc_roc']:<12.4f}")
    print("="*80)
    
    # Best detector
    best_detector = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\n✓ Best Detector: {best_detector} (F1-Score: {results[best_detector]['f1_score']:.4f})")
    
    # Show some detected anomalies
    print("\n5. Sample Detected Anomalies:")
    print("-" * 80)
    
    system = AnomalyDetectionSystem(detector_type='isolation_forest', contamination=0.05)
    system.fit(X_train, feature_names=feature_names)
    result = system.detect(X_test, explain_anomalies=True)
    
    visualizer = AnomalyVisualizer()
    
    for i, idx in enumerate(result.anomaly_indices[:5]):
        print(f"\nAnomaly #{i+1} (Index: {idx}):")
        print(f"  Score: {result.scores[idx]:.4f}")
        print(f"  Actual: {'ANOMALY' if y_test[idx] == 1 else 'NORMAL'}")
        
        if idx in result.metadata['explanations']:
            exp = result.metadata['explanations'][idx]
            print(f"  {exp['explanation']}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)


def compare_all_detectors_comprehensive():
    """Comprehensive comparison of all detection methods"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DETECTOR COMPARISON")
    print("="*80)
    
    # Generate data
    np.random.seed(42)
    n_samples = 5000
    n_features = 15
    
    # Normal data
    X_normal = np.random.randn(int(n_samples * 0.9), n_features)
    y_normal = np.zeros(int(n_samples * 0.9))
    
    # Anomalies
    X_anomaly = np.random.randn(int(n_samples * 0.1), n_features) * 3 + 4
    y_anomaly = np.ones(int(n_samples * 0.1))
    
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([y_normal, y_anomaly])
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    # Split
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\nDataset: {len(X):,} samples, {n_features} features")
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"Anomaly rate: {np.mean(y)*100:.2f}%\n")
    
    # Test configurations
    configs = [
        ('Isolation Forest (100 trees)', 'isolation_forest', {'n_estimators': 100}),
        ('Isolation Forest (200 trees)', 'isolation_forest', {'n_estimators': 200}),
        ('LOF (k=20)', 'lof', {'n_neighbors': 20}),
        ('LOF (k=50)', 'lof', {'n_neighbors': 50}),
        ('SPC (3 sigma)', 'spc', {'n_sigma': 3.0}),
        ('Autoencoder (fast)', 'autoencoder', {'hidden_dims': [32, 16], 'epochs': 30}),
        ('Autoencoder (deep)', 'autoencoder', {'hidden_dims': [64, 32, 16], 'epochs': 50}),
    ]
    
    results = []
    
    for name, detector_type, params in configs:
        print(f"Testing {name}...")
        
        try:
            system = AnomalyDetectionSystem(
                detector_type=detector_type,
                contamination=np.mean(y_train),
                **params
            )
            
            system.fit(X_train)
            metrics = system.evaluate(X_test, y_test)
            
            results.append({
                'name': name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1_score'],
                'auc': metrics['auc_roc']
            })
            
            print(f"  ✓ F1: {metrics['f1_score']:.4f}, AUC: {metrics['auc_roc']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Display results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"{'Detector':<35} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-"*80)
    
    for r in sorted(results, key=lambda x: x['f1'], reverse=True):
        print(f"{r['name']:<35} {r['precision']:<12.4f} {r['recall']:<12.4f} "
              f"{r['f1']:<12.4f} {r['auc']:<12.4f}")
    
    print("="*80)


def real_world_scenario_network_monitoring():
    """Simulate real-world network monitoring scenario"""
    
    print("\n" + "="*80)
    print("REAL-WORLD SCENARIO: NETWORK TRAFFIC MONITORING")
    print("="*80)
    
    print("\nSimulating 24 hours of network traffic...")
    
    # Generate realistic network traffic
    np.random.seed(42)
    n_minutes = 24 * 60  # 24 hours
    
    # Normal traffic patterns (daily cycle)
    hour_of_day = (np.arange(n_minutes) / 60) % 24
    
    # Traffic increases during business hours
    business_hour_factor = 1 + 0.5 * np.sin((hour_of_day - 6) * np.pi / 12)
    business_hour_factor = np.clip(business_hour_factor, 0.5, 1.5)
    
    # Features
    requests_per_min = np.random.poisson(100 * business_hour_factor)
    bandwidth_mbps = np.random.gamma(2, 10 * business_hour_factor)
    avg_response_ms = np.random.gamma(2, 50 / business_hour_factor)
    error_rate = np.random.beta(1, 100, n_minutes) * 100
    unique_ips = np.random.poisson(50 * business_hour_factor)
    cpu_usage = np.random.beta(2, 5, n_minutes) * 100
    memory_usage = np.random.beta(3, 4, n_minutes) * 100
    active_connections = np.random.poisson(200 * business_hour_factor)
    
    X = np.column_stack([
        requests_per_min, bandwidth_mbps, avg_response_ms, error_rate,
        unique_ips, cpu_usage, memory_usage, active_connections
    ])
    
    feature_names = [
        'requests_per_min', 'bandwidth_mbps', 'avg_response_ms', 'error_rate',
        'unique_ips', 'cpu_usage', 'memory_usage', 'active_connections'
    ]
    
    # Inject realistic attacks/anomalies
    y = np.zeros(n_minutes)
    
    # DDoS attack (high traffic)
    ddos_start = 500
    for i in range(ddos_start, ddos_start + 30):
        X[i, 0] *= 10  # 10x requests
        X[i, 4] *= 5   # 5x unique IPs
        X[i, 7] *= 8   # High connections
        y[i] = 1
    
    # Server crash (high error rate)
    crash_start = 800
    for i in range(crash_start, crash_start + 15):
        X[i, 3] = np.random.uniform(50, 90)  # High error rate
        X[i, 2] *= 5  # Slow response
        y[i] = 1
    
    # Memory leak (gradual increase)
    leak_start = 1200
    for i in range(leak_start, leak_start + 60):
        X[i, 6] = min(99, 70 + (i - leak_start) * 0.5)
        if X[i, 6] > 90:
            y[i] = 1
    
    print(f"  ✓ Generated {n_minutes:,} minutes of data")
    print(f"  ✓ Injected anomalies: DDoS attack, server crash, memory leak")
    print(f"  ✓ Total anomalies: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
    
    # Train on first 12 hours, test on last 12 hours
    split = n_minutes // 2
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\nTraining on first 12 hours (assuming normal)...")
    
    system = AnomalyDetectionSystem(
        detector_type='isolation_forest',
        contamination=0.05,
        n_estimators=150
    )
    
    system.fit(X_train, feature_names=feature_names)
    
    print("Monitoring last 12 hours...")
    result = system.detect(X_test, explain_anomalies=True)
    
    print(f"\n  Anomalies detected: {result.metadata['n_anomalies']}")
    
    # Show timeline of detections
    print("\nDetection Timeline:")
    print("-" * 80)
    
    for idx in result.anomaly_indices[:10]:
        actual_minute = split + idx
        hour = actual_minute // 60
        minute = actual_minute % 60
        
        is_true_anomaly = y_test[idx] == 1
        status = "✓ TRUE POSITIVE" if is_true_anomaly else "✗ FALSE POSITIVE"
        
        print(f"  Time: {hour:02d}:{minute:02d} | Score: {result.scores[idx]:.4f} | {status}")
        
        if idx in result.metadata['explanations']:
            exp = result.metadata['explanations'][idx]
            print(f"    Reason: {exp['explanation']}")
    
    # Evaluate
    metrics = system.evaluate(X_test, y_test)
    
    print(f"\n" + "="*80)
    print("MONITORING RESULTS")
    print("="*80)
    print(f"Precision: {metrics['precision']:.4f} (How many alerts were real?)")
    print(f"Recall: {metrics['recall']:.4f} (Did we catch all incidents?)")
    print(f"F1-Score: {metrics['f1_score']:.4f} (Overall performance)")
    print("="*80)


if __name__ == "__main__":
    print("\nChoose a test:")
    print("1. Quick IoT Sensor Test (Recommended)")
    print("2. Comprehensive Detector Comparison")
    print("3. Real-World Network Monitoring Scenario")
    print("4. Run All Tests")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        quick_test_iot_sensors()
    elif choice == '2':
        compare_all_detectors_comprehensive()
    elif choice == '3':
        real_world_scenario_network_monitoring()
    elif choice == '4':
        quick_test_iot_sensors()
        compare_all_detectors_comprehensive()
        real_world_scenario_network_monitoring()
    else:
        print("Invalid choice. Running quick test...")
        quick_test_iot_sensors()