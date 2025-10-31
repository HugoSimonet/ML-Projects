"""
Dataset loaders for anomaly detection testing
Includes multiple real-world datasets with different characteristics
"""

import numpy as np
import pandas as pd
from typing import Tuple
import urllib.request
import os


class DatasetLoader:
    """Load various datasets for anomaly detection"""
    
    @staticmethod
    def create_data_directory():
        """Create data directory if it doesn't exist"""
        if not os.path.exists('data/datasets'):
            os.makedirs('data/datasets')
    
    # ========================================================================
    # OPTION 1: Credit Card Fraud (BEST FOR FINANCIAL ANOMALIES)
    # ========================================================================
    
    @staticmethod
    def load_creditcard_fraud() -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load Credit Card Fraud Detection Dataset
        
        Features: 30 (28 PCA components + Time + Amount)
        Samples: ~284,807
        Anomalies: ~0.17%
        Use case: Financial fraud detection
        
        Dataset from Kaggle (requires download):
        https://www.kaggle.com/mlg-ulb/creditcardfraud
        
        Returns:
            X: Feature matrix
            y: Labels (0=normal, 1=fraud)
            feature_names: List of feature names
        """
        DatasetLoader.create_data_directory()
        
        print("\n" + "="*70)
        print("CREDIT CARD FRAUD DATASET")
        print("="*70)
        
        filepath = 'data/datasets/creditcard.csv'
        
        if not os.path.exists(filepath):
            print("\n⚠️  Dataset not found!")
            print("\nDownload from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
            print(f"Save to: {filepath}")
            print("\nGenerating synthetic similar data instead...")
            return DatasetLoader._generate_synthetic_creditcard()
        
        df = pd.read_csv(filepath)
        
        # Separate features and labels
        X = df.drop('Class', axis=1).values
        y = df['Class'].values
        feature_names = df.drop('Class', axis=1).columns.tolist()
        
        print(f"✓ Loaded successfully!")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Anomalies: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
        
        return X, y, feature_names
    
    @staticmethod
    def _generate_synthetic_creditcard() -> Tuple[np.ndarray, np.ndarray, list]:
        """Generate synthetic credit card-like data"""
        np.random.seed(42)
        
        n_normal = 5000
        n_fraud = 50
        n_features = 30
        
        # Normal transactions
        X_normal = np.random.randn(n_normal, n_features)
        y_normal = np.zeros(n_normal)
        
        # Fraudulent transactions (different distribution)
        X_fraud = np.random.randn(n_fraud, n_features) * 2 + 3
        y_fraud = np.ones(n_fraud)
        
        X = np.vstack([X_normal, X_fraud])
        y = np.hstack([y_normal, y_fraud])
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        
        feature_names = [f'V{i}' for i in range(28)] + ['Time', 'Amount']
        
        print(f"✓ Generated synthetic data")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Anomalies: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
        
        return X, y, feature_names
    
    # ========================================================================
    # OPTION 2: Network Intrusion (BEST FOR CYBERSECURITY)
    # ========================================================================
    
    @staticmethod
    def load_kdd_cup() -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load KDD Cup 99 Network Intrusion Dataset (10% subset)
        
        Features: 41 (network traffic features)
        Samples: ~494,021
        Anomalies: ~19.7%
        Use case: Network intrusion detection
        
        Auto-downloads from UCI repository
        
        Returns:
            X: Feature matrix
            y: Labels (0=normal, 1=attack)
            feature_names: List of feature names
        """
        DatasetLoader.create_data_directory()
        
        print("\n" + "="*70)
        print("KDD CUP 99 NETWORK INTRUSION DATASET")
        print("="*70)
        
        filepath = 'data/datasets/kddcup.data_10_percent'
        
        if not os.path.exists(filepath):
            print("Downloading dataset... (may take a minute)")
            url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
            try:
                import gzip
                import shutil
                
                urllib.request.urlretrieve(url, filepath + '.gz')
                with gzip.open(filepath + '.gz', 'rb') as f_in:
                    with open(filepath, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(filepath + '.gz')
                print("✓ Download complete!")
            except Exception as e:
                print(f"⚠️  Download failed: {e}")
                print("Generating synthetic network data instead...")
                return DatasetLoader._generate_synthetic_network()
        
        # Column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
        ]
        
        # Load data
        df = pd.read_csv(filepath, names=columns)
        
        # Sample 10,000 rows for faster processing
        df = df.sample(n=10000, random_state=42)
        
        # Convert categorical to numeric
        for col in ['protocol_type', 'service', 'flag']:
            df[col] = pd.Categorical(df[col]).codes
        
        # Create binary labels
        df['anomaly'] = (df['label'] != 'normal.').astype(int)
        
        X = df.drop(['label', 'anomaly'], axis=1).values
        y = df['anomaly'].values
        feature_names = df.drop(['label', 'anomaly'], axis=1).columns.tolist()
        
        print(f"✓ Loaded successfully!")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Anomalies: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
        
        return X, y, feature_names
    
    @staticmethod
    def _generate_synthetic_network() -> Tuple[np.ndarray, np.ndarray, list]:
        """Generate synthetic network traffic data"""
        np.random.seed(42)
        
        n_normal = 8000
        n_attack = 2000
        n_features = 20
        
        # Normal traffic
        X_normal = np.random.randn(n_normal, n_features)
        y_normal = np.zeros(n_normal)
        
        # Attack traffic
        X_attack = np.random.randn(n_attack, n_features) * 1.5 + 2
        y_attack = np.ones(n_attack)
        
        X = np.vstack([X_normal, X_attack])
        y = np.hstack([y_normal, y_attack])
        
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        
        feature_names = [
            'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'num_root', 'num_file_creations', 'count',
            'srv_count', 'serror_rate', 'rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count'
        ]
        
        print(f"✓ Generated synthetic network data")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Anomalies: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
        
        return X, y, feature_names
    
    # ========================================================================
    # OPTION 3: IoT Sensor Data (BEST FOR TIME SERIES)
    # ========================================================================
    
    @staticmethod
    def load_sensor_data() -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Generate realistic IoT sensor data with anomalies
        
        Features: 10 (temperature, humidity, pressure, etc.)
        Samples: 10,000
        Anomalies: ~5%
        Use case: IoT monitoring, predictive maintenance
        
        Returns:
            X: Feature matrix
            y: Labels (0=normal, 1=anomaly)
            feature_names: List of feature names
        """
        print("\n" + "="*70)
        print("IOT SENSOR DATASET")
        print("="*70)
        
        np.random.seed(42)
        n_samples = 10000
        
        # Generate time series sensor data
        time = np.arange(n_samples)
        
        # Normal sensor readings with patterns
        temperature = 20 + 5 * np.sin(time * 2 * np.pi / 1000) + np.random.randn(n_samples) * 0.5
        humidity = 60 + 10 * np.cos(time * 2 * np.pi / 1500) + np.random.randn(n_samples) * 2
        pressure = 1013 + 3 * np.sin(time * 2 * np.pi / 800) + np.random.randn(n_samples) * 1
        vibration = 0.5 + 0.2 * np.sin(time * 2 * np.pi / 500) + np.random.randn(n_samples) * 0.1
        power = 100 + 20 * np.sin(time * 2 * np.pi / 2000) + np.random.randn(n_samples) * 5
        
        # Create feature matrix
        X = np.column_stack([
            temperature, humidity, pressure, vibration, power,
            np.gradient(temperature),  # Rate of change
            np.gradient(humidity),
            np.gradient(pressure),
            np.gradient(vibration),
            np.gradient(power)
        ])
        
        # Inject anomalies (5%)
        y = np.zeros(n_samples)
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        
        for idx in anomaly_indices:
            # Random anomaly type
            anomaly_type = np.random.randint(0, 5)
            
            if anomaly_type == 0:  # Temperature spike
                X[idx, 0] += np.random.uniform(15, 25)
            elif anomaly_type == 1:  # Humidity drop
                X[idx, 1] -= np.random.uniform(20, 40)
            elif anomaly_type == 2:  # Pressure anomaly
                X[idx, 2] += np.random.uniform(10, 20) * np.random.choice([-1, 1])
            elif anomaly_type == 3:  # Vibration spike
                X[idx, 3] += np.random.uniform(2, 5)
            else:  # Power surge
                X[idx, 4] += np.random.uniform(50, 100)
            
            y[idx] = 1
        
        feature_names = [
            'temperature', 'humidity', 'pressure', 'vibration', 'power_consumption',
            'temp_rate_change', 'humid_rate_change', 'press_rate_change',
            'vibr_rate_change', 'power_rate_change'
        ]
        
        print(f"✓ Generated IoT sensor data")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Anomalies: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
        
        return X, y, feature_names
    
    # ========================================================================
    # OPTION 4: Manufacturing Quality (BEST FOR INDUSTRIAL)
    # ========================================================================
    
    @staticmethod
    def load_manufacturing_data() -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Generate manufacturing process data with defects
        
        Features: 15 (process parameters)
        Samples: 5,000
        Anomalies: ~3%
        Use case: Quality control, defect detection
        
        Returns:
            X: Feature matrix
            y: Labels (0=normal, 1=defect)
            feature_names: List of feature names
        """
        print("\n" + "="*70)
        print("MANUFACTURING QUALITY DATASET")
        print("="*70)
        
        np.random.seed(42)
        n_samples = 5000
        
        # Normal manufacturing parameters
        machine_temp = np.random.normal(75, 3, n_samples)
        rpm = np.random.normal(1500, 50, n_samples)
        feed_rate = np.random.normal(500, 20, n_samples)
        cutting_force = np.random.normal(200, 15, n_samples)
        torque = np.random.normal(50, 5, n_samples)
        tool_wear = np.random.uniform(0, 100, n_samples)
        vibration_x = np.random.normal(0.5, 0.1, n_samples)
        vibration_y = np.random.normal(0.5, 0.1, n_samples)
        vibration_z = np.random.normal(0.5, 0.1, n_samples)
        acoustic_noise = np.random.normal(80, 5, n_samples)
        power_consumption = np.random.normal(2.5, 0.3, n_samples)
        coolant_temp = np.random.normal(25, 2, n_samples)
        coolant_pressure = np.random.normal(10, 1, n_samples)
        material_hardness = np.random.normal(150, 10, n_samples)
        surface_roughness = np.random.normal(1.5, 0.3, n_samples)
        
        X = np.column_stack([
            machine_temp, rpm, feed_rate, cutting_force, torque,
            tool_wear, vibration_x, vibration_y, vibration_z,
            acoustic_noise, power_consumption, coolant_temp,
            coolant_pressure, material_hardness, surface_roughness
        ])
        
        # Inject defects (3%)
        y = np.zeros(n_samples)
        defect_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
        
        for idx in defect_indices:
            # Defect causes parameter deviations
            X[idx, 0] += np.random.uniform(10, 20)  # Overheating
            X[idx, 3] += np.random.uniform(50, 100)  # Excessive force
            X[idx, 6:9] *= np.random.uniform(2, 4)  # High vibration
            X[idx, 14] += np.random.uniform(1, 3)  # Poor surface finish
            y[idx] = 1
        
        feature_names = [
            'machine_temp', 'rpm', 'feed_rate', 'cutting_force', 'torque',
            'tool_wear', 'vibration_x', 'vibration_y', 'vibration_z',
            'acoustic_noise', 'power_consumption', 'coolant_temp',
            'coolant_pressure', 'material_hardness', 'surface_roughness'
        ]
        
        print(f"✓ Generated manufacturing data")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Anomalies: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
        
        return X, y, feature_names
    
    # ========================================================================
    # OPTION 5: Server Logs (BEST FOR IT/DEVOPS)
    # ========================================================================
    
    @staticmethod
    def load_server_logs() -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Generate server performance metrics with anomalies
        
        Features: 12 (CPU, memory, network, etc.)
        Samples: 8,000
        Anomalies: ~8%
        Use case: Server monitoring, DevOps
        
        Returns:
            X: Feature matrix
            y: Labels (0=normal, 1=anomaly)
            feature_names: List of feature names
        """
        print("\n" + "="*70)
        print("SERVER LOGS DATASET")
        print("="*70)
        
        np.random.seed(42)
        n_samples = 8000
        
        # Normal server metrics
        cpu_usage = np.random.beta(2, 5, n_samples) * 100
        memory_usage = np.random.beta(3, 4, n_samples) * 100
        disk_io_read = np.random.gamma(2, 50, n_samples)
        disk_io_write = np.random.gamma(2, 30, n_samples)
        network_in = np.random.gamma(3, 100, n_samples)
        network_out = np.random.gamma(3, 80, n_samples)
        active_connections = np.random.poisson(100, n_samples)
        response_time = np.random.gamma(2, 50, n_samples)
        error_rate = np.random.beta(1, 50, n_samples) * 100
        request_rate = np.random.poisson(500, n_samples)
        cache_hit_rate = np.random.beta(8, 2, n_samples) * 100
        queue_length = np.random.poisson(10, n_samples)
        
        X = np.column_stack([
            cpu_usage, memory_usage, disk_io_read, disk_io_write,
            network_in, network_out, active_connections, response_time,
            error_rate, request_rate, cache_hit_rate, queue_length
        ])
        
        # Inject anomalies (8%)
        y = np.zeros(n_samples)
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.08), replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.randint(0, 4)
            
            if anomaly_type == 0:  # CPU spike
                X[idx, 0] = np.random.uniform(90, 100)
            elif anomaly_type == 1:  # Memory leak
                X[idx, 1] = np.random.uniform(85, 99)
            elif anomaly_type == 2:  # Network issue
                X[idx, 4:6] *= np.random.uniform(5, 10)
            else:  # High error rate
                X[idx, 8] = np.random.uniform(10, 50)
                X[idx, 7] *= np.random.uniform(3, 8)  # Slow response
            
            y[idx] = 1
        
        feature_names = [
            'cpu_usage', 'memory_usage', 'disk_io_read', 'disk_io_write',
            'network_in_mbps', 'network_out_mbps', 'active_connections',
            'avg_response_time_ms', 'error_rate_percent', 'requests_per_sec',
            'cache_hit_rate', 'queue_length'
        ]
        
        print(f"✓ Generated server logs data")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Anomalies: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
        
        return X, y, feature_names


# =============================================================================
# EXAMPLE USAGE SCRIPT
# =============================================================================

def test_all_datasets():
    """Test all available datasets with optimized contamination rates"""
    from core.system import AnomalyDetectionSystem
    from visualization.anomaly_visualizer import AnomalyVisualizer
    
    loader = DatasetLoader()
    visualizer = AnomalyVisualizer()
    
    # Dataset-specific configurations
    datasets = {
        'Credit Card Fraud': {
            'loader': loader.load_creditcard_fraud,
            'contamination': 0.002,  # Very rare anomalies
            'n_estimators': 200
        },
        'Network Intrusion': {
            'loader': loader.load_kdd_cup,
            'contamination': 0.2,  # Common attacks
            'n_estimators': 150
        },
        'IoT Sensors': {
            'loader': loader.load_sensor_data,
            'contamination': 0.05,  # 5% sensor faults
            'n_estimators': 100
        },
        'Manufacturing': {
            'loader': loader.load_manufacturing_data,
            'contamination': 0.03,  # 3% defects
            'n_estimators': 100
        },
        'Server Logs': {
            'loader': loader.load_server_logs,
            'contamination': 0.08,  # 8% incidents
            'n_estimators': 100
        }
    }
    
    print("\n" + "="*70)
    print("TESTING ALL DATASETS (OPTIMIZED)")
    print("="*70)
    
    results = {}
    
    for name, config in datasets.items():
        try:
            # Load data
            X, y, feature_names = config['loader']()
            
            # Calculate actual anomaly rate
            actual_contamination = np.mean(y)
            
            # Split train/test
            split_idx = int(len(X) * 0.7)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"\n{name}:")
            print(f"  Actual anomaly rate: {actual_contamination:.4f} ({actual_contamination*100:.2f}%)")
            print(f"  Using contamination: {config['contamination']:.4f}")
            
            # Train detector with optimized settings
            system = AnomalyDetectionSystem(
                detector_type='isolation_forest',
                contamination=config['contamination'],
                n_estimators=config['n_estimators']
            )
            
            print(f"  Training...")
            system.fit(X_train, feature_names=feature_names)
            
            # Evaluate
            print(f"  Evaluating...")
            metrics = system.evaluate(X_test, y_test)
            results[name] = metrics
            
            print(f"  ✓ Precision: {metrics['precision']:.4f}")
            print(f"  ✓ Recall: {metrics['recall']:.4f}")
            print(f"  ✓ F1-Score: {metrics['f1_score']:.4f}")
            print(f"  ✓ AUC-ROC: {metrics['auc_roc']:.4f}")
            
        except Exception as e:
            print(f"\n⚠️  {name}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Dataset':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-"*70)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {metrics['auc_roc']:<12.4f}")
    print("="*70)
    
    # Highlight best performers
    if results:
        best_f1 = max(results, key=lambda x: results[x]['f1_score'])
        best_auc = max(results, key=lambda x: results[x]['auc_roc'])
        
        print(f"\n🏆 Best F1-Score: {best_f1} ({results[best_f1]['f1_score']:.4f})")
        print(f"🏆 Best AUC-ROC: {best_auc} ({results[best_auc]['auc_roc']:.4f})")
    print("="*70)


if __name__ == "__main__":
    # Quick test with one dataset
    loader = DatasetLoader()
    
    print("\nChoose a dataset:")
    print("1. Credit Card Fraud (Financial)")
    print("2. Network Intrusion (Cybersecurity)")
    print("3. IoT Sensors (Time Series)")
    print("4. Manufacturing (Quality Control)")
    print("5. Server Logs (IT/DevOps)")
    print("6. Test All Datasets")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        X, y, names = loader.load_creditcard_fraud()
    elif choice == '2':
        X, y, names = loader.load_kdd_cup()
    elif choice == '3':
        X, y, names = loader.load_sensor_data()
    elif choice == '4':
        X, y, names = loader.load_manufacturing_data()
    elif choice == '5':
        X, y, names = loader.load_server_logs()
    elif choice == '6':
        test_all_datasets()
    else:
        print("Invalid choice!")
        X, y, names = loader.load_sensor_data()  # Default
    
    if choice != '6':
        print(f"\n✓ Dataset loaded! Shape: {X.shape}")
        print(f"  Feature names: {names[:5]}... (showing first 5)")