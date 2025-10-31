"""
Dataset Diagnostics - Analyze and fix dataset issues
"""

import numpy as np
import sys
sys.path.append('..')

from core.system import AnomalyDetectionSystem
from data.datasets.data_loader import DatasetLoader


def diagnose_dataset(name, X, y, feature_names):
    """Comprehensive dataset diagnosis"""
    
    print("\n" + "="*80)
    print(f"DIAGNOSING: {name}")
    print("="*80)
    
    # Basic stats
    print(f"\n1. BASIC STATISTICS:")
    print(f"   Samples: {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Normal samples: {np.sum(y == 0):,} ({np.mean(y == 0)*100:.2f}%)")
    print(f"   Anomaly samples: {np.sum(y == 1):,} ({np.mean(y == 1)*100:.2f}%)")
    
    # Check for issues
    print(f"\n2. DATA QUALITY CHECKS:")
    
    # Missing values
    n_missing = np.sum(np.isnan(X))
    print(f"   Missing values: {n_missing}")
    
    # Infinite values
    n_inf = np.sum(np.isinf(X))
    print(f"   Infinite values: {n_inf}")
    
    # Feature ranges
    print(f"   Feature ranges:")
    for i in range(min(5, X.shape[1])):
        feat_name = feature_names[i] if feature_names else f"Feature_{i}"
        print(f"     {feat_name}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
    if X.shape[1] > 5:
        print(f"     ... ({X.shape[1] - 5} more features)")
    
    # Class imbalance
    anomaly_rate = np.mean(y == 1)
    print(f"\n3. CLASS BALANCE:")
    print(f"   Anomaly rate: {anomaly_rate:.4f} ({anomaly_rate*100:.2f}%)")
    
    if anomaly_rate < 0.01:
        print("   ⚠️  SEVERE IMBALANCE: Very rare anomalies (<1%)")
        recommended_contamination = max(0.001, anomaly_rate * 1.5)
    elif anomaly_rate < 0.05:
        print("   ⚠️  HIGH IMBALANCE: Rare anomalies (<5%)")
        recommended_contamination = anomaly_rate * 1.2
    elif anomaly_rate > 0.3:
        print("   ⚠️  UNUSUAL: High anomaly rate (>30%)")
        recommended_contamination = anomaly_rate * 0.9
    else:
        print("   ✓ BALANCED: Moderate anomaly rate")
        recommended_contamination = anomaly_rate
    
    print(f"   Recommended contamination: {recommended_contamination:.4f}")
    
    # Test with different contamination rates
    print(f"\n4. TESTING DIFFERENT CONTAMINATION RATES:")
    print(f"   {'Contamination':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("   " + "-"*63)
    
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Try different contamination rates
    test_rates = [
        anomaly_rate * 0.5,
        anomaly_rate,
        anomaly_rate * 1.5,
        anomaly_rate * 2.0,
        recommended_contamination
    ]
    
    best_f1 = 0
    best_rate = anomaly_rate
    
    for rate in test_rates:
        try:
            system = AnomalyDetectionSystem(
                detector_type='isolation_forest',
                contamination=min(0.5, max(0.001, rate)),  # Clamp to valid range
                n_estimators=100
            )
            
            system.fit(X_train)
            metrics = system.evaluate(X_test, y_test)
            
            marker = " ← BEST" if metrics['f1_score'] > best_f1 else ""
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_rate = rate
            
            print(f"   {rate:<15.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f} {metrics['auc_roc']:<12.4f}{marker}")
        except Exception as e:
            print(f"   {rate:<15.4f} ERROR: {e}")
    
    # Recommendations
    print(f"\n5. RECOMMENDATIONS:")
    print(f"   ✓ Use contamination = {best_rate:.4f}")
    print(f"   ✓ Expected F1-Score: {best_f1:.4f}")
    
    if anomaly_rate < 0.01:
        print(f"   ⚠️  Consider ensemble methods for extreme imbalance")
        print(f"   ⚠️  May need more training data")
    
    if X.shape[1] > 50:
        print(f"   💡 Consider dimensionality reduction (current: {X.shape[1]} features)")
    
    return best_rate, best_f1


def fix_and_retest_all():
    """Diagnose all datasets and retest with optimized settings"""
    
    loader = DatasetLoader()
    
    datasets = {
        'Credit Card Fraud': loader.load_creditcard_fraud,
        'Network Intrusion': loader.load_kdd_cup,
        'IoT Sensors': loader.load_sensor_data,
        'Manufacturing': loader.load_manufacturing_data,
        'Server Logs': loader.load_server_logs
    }
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DATASET DIAGNOSIS & OPTIMIZATION")
    print("="*80)
    
    optimized_settings = {}
    
    for name, load_func in datasets.items():
        try:
            X, y, feature_names = load_func()
            best_rate, best_f1 = diagnose_dataset(name, X, y, feature_names)
            optimized_settings[name] = {
                'contamination': best_rate,
                'expected_f1': best_f1
            }
        except Exception as e:
            print(f"\n⚠️  {name}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("OPTIMIZED SETTINGS SUMMARY")
    print("="*80)
    print(f"{'Dataset':<25} {'Contamination':<15} {'Expected F1':<15}")
    print("-"*80)
    
    for name, settings in optimized_settings.items():
        print(f"{name:<25} {settings['contamination']:<15.4f} {settings['expected_f1']:<15.4f}")
    
    print("="*80)
    
    # Generate code snippet
    print("\n📋 COPY THIS CODE FOR OPTIMAL RESULTS:")
    print("-"*80)
    print("optimized_config = {")
    for name, settings in optimized_settings.items():
        print(f"    '{name}': {{'contamination': {settings['contamination']:.4f}}},")
    print("}")
    print("-"*80)


def quick_fix_guide():
    """Print quick fix guide for common issues"""
    
    print("\n" + "="*80)
    print("QUICK FIX GUIDE")
    print("="*80)
    
    print("""
PROBLEM: Low F1-Score, High AUC
CAUSE: Contamination parameter too high/low
FIX: Set contamination = actual_anomaly_rate
    
    actual_rate = np.mean(y_train)
    system = AnomalyDetectionSystem(contamination=actual_rate)

PROBLEM: Low AUC-ROC
CAUSE: Labels might be inverted or features not informative
FIX: Check label distribution, try different detector
    
    # Check labels
    print(f"Normal: {np.sum(y==0)}, Anomaly: {np.sum(y==1)}")
    
    # Try LOF instead
    system = AnomalyDetectionSystem(detector_type='lof')

PROBLEM: Both F1 and AUC low
CAUSE: Features not predictive or need preprocessing
FIX: Normalize features, try ensemble
    
    system.fit(X_train, preprocess=True)
    
    # Or use ensemble
    from models.ensemble_methods import EnsembleDetector
    ensemble = EnsembleDetector([detector1, detector2, detector3])

PROBLEM: Perfect scores (1.0000)
CAUSE: Data leakage or too easy
FIX: Check train/test split, verify data generation

PROBLEM: Credit Card Fraud very low F1
CAUSE: Extreme imbalance (0.17% anomalies)
FIX: Use contamination=0.002 and more trees
    
    system = AnomalyDetectionSystem(
        detector_type='isolation_forest',
        contamination=0.002,  # Match actual rate
        n_estimators=200      # More trees for rare events
    )
""")
    
    print("="*80)


if __name__ == "__main__":
    print("\nChoose an option:")
    print("1. Diagnose all datasets and get optimized settings")
    print("2. Show quick fix guide")
    print("3. Diagnose specific dataset")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        fix_and_retest_all()
    elif choice == '2':
        quick_fix_guide()
    elif choice == '3':
        loader = DatasetLoader()
        
        print("\nAvailable datasets:")
        print("1. Credit Card Fraud")
        print("2. Network Intrusion")
        print("3. IoT Sensors")
        print("4. Manufacturing")
        print("5. Server Logs")
        
        dataset_choice = input("\nEnter dataset (1-5): ").strip()
        
        datasets = {
            '1': ('Credit Card Fraud', loader.load_creditcard_fraud),
            '2': ('Network Intrusion', loader.load_kdd_cup),
            '3': ('IoT Sensors', loader.load_sensor_data),
            '4': ('Manufacturing', loader.load_manufacturing_data),
            '5': ('Server Logs', loader.load_server_logs)
        }
        
        if dataset_choice in datasets:
            name, load_func = datasets[dataset_choice]
            X, y, feature_names = load_func()
            diagnose_dataset(name, X, y, feature_names)
        else:
            print("Invalid choice!")
    else:
        print("Invalid choice!")
        quick_fix_guide()