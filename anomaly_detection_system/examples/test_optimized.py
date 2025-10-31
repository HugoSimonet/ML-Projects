"""
Optimized Dataset Testing - Using diagnosed optimal settings
Run this for best results on all datasets
"""

import numpy as np
import sys
sys.path.append('..')

from core.system import AnomalyDetectionSystem
from data.datasets.data_loader import DatasetLoader
from visualization.anomaly_visualizer import AnomalyVisualizer
from models.statistical_methods import IsolationForestDetector, LocalOutlierFactor
from models.ensemble_methods import EnsembleDetector


# Optimal settings from diagnostic results
OPTIMIZED_CONFIG = {
    'Credit Card Fraud': {
        'contamination': 0.0035,
        'n_estimators': 200,
        'notes': 'Extreme imbalance - very rare fraud (0.17%)'
    },
    'Network Intrusion': {
        'contamination': 0.3992,
        'n_estimators': 150,
        'notes': 'High attack rate (79.83%) - inverted distribution'
    },
    'IoT Sensors': {
        'contamination': 0.0250,
        'n_estimators': 100,
        'notes': 'Moderate sensor faults (5%)'
    },
    'Manufacturing': {
        'contamination': 0.0300,
        'n_estimators': 100,
        'notes': 'Rare defects (3%) - works perfectly'
    },
    'Server Logs': {
        'contamination': 0.0400,
        'n_estimators': 120,
        'notes': 'Server incidents (8%)'
    }
}


def test_with_optimal_settings():
    """Test all datasets with optimized settings"""
    
    loader = DatasetLoader()
    visualizer = AnomalyVisualizer()
    
    loaders = {
        'Credit Card Fraud': loader.load_creditcard_fraud,
        'Network Intrusion': loader.load_kdd_cup,
        'IoT Sensors': loader.load_sensor_data,
        'Manufacturing': loader.load_manufacturing_data,
        'Server Logs': loader.load_server_logs
    }
    
    print("\n" + "="*80)
    print("TESTING ALL DATASETS WITH OPTIMIZED SETTINGS")
    print("="*80)
    
    results = {}
    
    for name, load_func in loaders.items():
        print(f"\n{'='*80}")
        print(f"{name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Load data
            X, y, feature_names = load_func()
            config = OPTIMIZED_CONFIG[name]
            
            print(f"Dataset info:")
            print(f"  Samples: {len(X):,}")
            print(f"  Features: {X.shape[1]}")
            print(f"  Actual anomaly rate: {np.mean(y)*100:.2f}%")
            print(f"  Using contamination: {config['contamination']:.4f}")
            print(f"  Note: {config['notes']}")
            
            # Split train/test
            split_idx = int(len(X) * 0.7)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train with optimal settings
            print(f"\nTraining Isolation Forest...")
            system = AnomalyDetectionSystem(
                detector_type='isolation_forest',
                contamination=config['contamination'],
                n_estimators=config['n_estimators']
            )
            
            system.fit(X_train, feature_names=feature_names)
            
            # Detect and evaluate
            print(f"Detecting anomalies...")
            result = system.detect(X_test, explain_anomalies=True)
            
            print(f"Evaluating performance...")
            metrics = system.evaluate(X_test, y_test)
            results[name] = metrics
            
            # Show results
            print(f"\nResults:")
            print(f"  Detected: {result.metadata['n_anomalies']} anomalies")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            
            # Show sample detections
            if result.metadata['explanations']:
                print(f"\nSample Detected Anomalies:")
                for i, (idx, exp_data) in enumerate(list(result.metadata['explanations'].items())[:3]):
                    actual = "✓ TRUE" if y_test[idx] == 1 else "✗ FALSE"
                    print(f"  #{i+1}: Score={result.scores[idx]:.4f} {actual}")
                    print(f"      {exp_data['explanation']}")
            
        except Exception as e:
            print(f"⚠️  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Dataset':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-"*80)
    
    for name, metrics in results.items():
        status = "🏆" if metrics['f1_score'] > 0.7 else "⚠️" if metrics['f1_score'] > 0.3 else "❌"
        print(f"{name:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {metrics['auc_roc']:<12.4f} {status}")
    
    print("="*80)
    
    # Interpretations
    print("\n📊 INTERPRETATIONS:")
    print("-"*80)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        
        if metrics['f1_score'] > 0.8:
            print(f"  ✅ EXCELLENT performance (F1={metrics['f1_score']:.4f})")
        elif metrics['f1_score'] > 0.5:
            print(f"  ✓ GOOD performance (F1={metrics['f1_score']:.4f})")
        elif metrics['f1_score'] > 0.2:
            print(f"  ⚠️ MODERATE performance (F1={metrics['f1_score']:.4f})")
        else:
            print(f"  ❌ POOR performance (F1={metrics['f1_score']:.4f})")
        
        # Specific advice
        if name == 'Credit Card Fraud' and metrics['f1_score'] < 0.5:
            print(f"     → Extreme imbalance makes this challenging")
            print(f"     → Consider: Ensemble methods, SMOTE oversampling, or cost-sensitive learning")
        
        if name == 'Network Intrusion' and metrics['auc_roc'] < 0.5:
            print(f"     → Low AUC suggests labels might be inverted")
            print(f"     → Normal traffic labeled as attacks, or vice versa")
        
        if name == 'IoT Sensors' and metrics['f1_score'] < 0.5:
            print(f"     → Anomalies may not be distinct enough")
            print(f"     → Try: LOF or Autoencoder for different patterns")
        
        if name == 'Server Logs' and metrics['f1_score'] < 0.5:
            print(f"     → Consider: Time-series features or windowing")
            print(f"     → Server anomalies often have temporal patterns")
    
    print("\n" + "="*80)


def compare_detectors_on_best_dataset():
    """Compare different detectors on the best-performing dataset (Manufacturing)"""
    
    print("\n" + "="*80)
    print("DETECTOR COMPARISON: Manufacturing Dataset")
    print("="*80)
    
    loader = DatasetLoader()
    X, y, feature_names = loader.load_manufacturing_data()
    
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Anomaly rate: {np.mean(y)*100:.2f}%\n")
    
    detectors = [
        ('Isolation Forest', 'isolation_forest', {'n_estimators': 100}),
        ('Isolation Forest (200 trees)', 'isolation_forest', {'n_estimators': 200}),
        ('LOF (k=20)', 'lof', {'n_neighbors': 20}),
        ('LOF (k=50)', 'lof', {'n_neighbors': 50}),
        ('SPC (3 sigma)', 'spc', {'n_sigma': 3.0}),
        ('Autoencoder', 'autoencoder', {'hidden_dims': [32, 16, 8], 'epochs': 50}),
    ]
    
    results = []
    
    for name, detector_type, params in detectors:
        print(f"Testing {name}...")
        
        try:
            system = AnomalyDetectionSystem(
                detector_type=detector_type,
                contamination=0.03,
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
    
    # Display comparison
    print("\n" + "="*80)
    print("DETECTOR COMPARISON RESULTS")
    print("="*80)
    print(f"{'Detector':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-"*80)
    
    for r in sorted(results, key=lambda x: x['f1'], reverse=True):
        best_marker = " 🏆" if r == results[0] else ""
        print(f"{r['name']:<30} {r['precision']:<12.4f} {r['recall']:<12.4f} "
              f"{r['f1']:<12.4f} {r['auc']:<12.4f}{best_marker}")
    
    print("="*80)


def test_ensemble_on_difficult_dataset():
    """Test ensemble methods on the most challenging dataset (Credit Card Fraud)"""
    
    print("\n" + "="*80)
    print("ENSEMBLE TEST: Credit Card Fraud (Challenging)")
    print("="*80)
    
    loader = DatasetLoader()
    X, y, feature_names = loader.load_creditcard_fraud()
    
    print(f"\nDataset: {len(X):,} samples")
    print(f"Extreme imbalance: {np.mean(y)*100:.4f}% anomalies\n")
    
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Test individual detectors
    print("Testing individual detectors:")
    
    detectors = [
        IsolationForestDetector(contamination=0.0035, n_estimators=200),
        LocalOutlierFactor(contamination=0.0035, n_neighbors=50),
    ]
    
    individual_results = []
    
    for i, detector in enumerate(detectors):
        name = type(detector).__name__
        print(f"  {name}...", end=" ")
        
        detector.fit(X_train)
        y_pred = detector.predict(X_test)
        scores = detector.score_samples(X_test)
        
        from evaluation.anomaly_metrics import AnomalyMetrics
        metrics = AnomalyMetrics()
        prf = metrics.precision_recall_f1(y_test, y_pred)
        auc = metrics.auc_roc(y_test, scores)
        
        print(f"F1={prf['f1_score']:.4f}, AUC={auc:.4f}")
        individual_results.append((name, prf['f1_score'], auc))
    
    # Test ensemble
    print(f"\nTesting ensemble (average combination):")
    
    ensemble = EnsembleDetector(
        detectors=detectors,
        combination_method='average',
        contamination=0.0035
    )
    
    ensemble.fit(X_train)
    y_pred = ensemble.predict(X_test)
    scores = ensemble.score_samples(X_test)
    
    from evaluation.anomaly_metrics import AnomalyMetrics
    metrics = AnomalyMetrics()
    prf = metrics.precision_recall_f1(y_test, y_pred)
    auc = metrics.auc_roc(y_test, scores)
    
    print(f"  Ensemble: F1={prf['f1_score']:.4f}, AUC={auc:.4f}")
    
    # Summary
    print("\n" + "="*80)
    print("ENSEMBLE COMPARISON")
    print("="*80)
    print(f"{'Method':<30} {'F1-Score':<12} {'AUC-ROC':<12}")
    print("-"*80)
    
    for name, f1, auc_score in individual_results:
        print(f"{name:<30} {f1:<12.4f} {auc_score:<12.4f}")
    
    print(f"{'Ensemble (Average)':<30} {prf['f1_score']:<12.4f} {auc:<12.4f}")
    print("="*80)
    
    improvement = prf['f1_score'] - max([r[1] for r in individual_results])
    if improvement > 0:
        print(f"\n✓ Ensemble improved F1 by {improvement:.4f}")
    else:
        print(f"\n→ Individual detector performed better")


if __name__ == "__main__":
    print("\nChoose a test:")
    print("1. Test all datasets with optimized settings (RECOMMENDED)")
    print("2. Compare different detectors on Manufacturing dataset")
    print("3. Test ensemble on Credit Card Fraud dataset")
    print("4. Run all tests")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        test_with_optimal_settings()
    elif choice == '2':
        compare_detectors_on_best_dataset()
    elif choice == '3':
        test_ensemble_on_difficult_dataset()
    elif choice == '4':
        test_with_optimal_settings()
        compare_detectors_on_best_dataset()
        test_ensemble_on_difficult_dataset()
    else:
        print("Invalid choice. Running main test...")
        test_with_optimal_settings()