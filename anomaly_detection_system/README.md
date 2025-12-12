# Anomaly Detection System

A Python implementation of multiple anomaly detection algorithms for high-dimensional data, including statistical methods (Isolation Forest, One-Class SVM, LOF), deep learning approaches (Autoencoders, VAE), and ensemble techniques.

## Overview

This project provides a unified framework for anomaly detection across different data types and domains. It implements both classical statistical methods and modern deep learning techniques, with built-in evaluation tools and visualization capabilities.

## Features

- **Statistical Methods**: Isolation Forest, One-Class SVM, Local Outlier Factor (LOF)
- **Deep Learning**: Autoencoders and Variational Autoencoders for reconstruction-based detection
- **Ensemble Detection**: Combine multiple algorithms with voting or stacking
- **Evaluation Framework**: Precision, recall, F1-score, AUC-ROC, AUC-PR metrics
- **Visualization Tools**: Anomaly score distributions, 2D projections, feature importance

## Architecture

### Core Components

**models/**: Detection algorithms
- `isolation_forest.py` - Isolation Forest implementation
- `one_class_svm.py` - One-Class SVM wrapper
- `autoencoder.py` - Autoencoder and VAE for anomaly detection
- `lof.py` - Local Outlier Factor
- `ensemble.py` - Ensemble methods

**data/**: Dataset handling
- `dataset_loader.py` - Load and preprocess datasets
- `preprocessing.py` - Data normalization and transformation

**evaluation/**: Metrics and analysis
- `metrics.py` - Detection performance metrics
- `evaluator.py` - Unified evaluation interface

**visualization/**: Plotting utilities
- `anomaly_viz.py` - Visualization tools for scores and predictions

## Installation

```bash
# Clone repository
git clone <repository-url>
cd anomaly_detection_system

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional anomaly detection libraries
pip install pyod scikit-learn
```

**Requirements:**
- Python 3.8+
- PyTorch 1.9+ (for deep learning methods)
- scikit-learn
- numpy, pandas, matplotlib

## Usage

### Basic Example

```python
from models import IsolationForest
from data import load_dataset

# Load data
X_train, X_test, y_test = load_dataset('credit_card')

# Train detector
detector = IsolationForest(contamination=0.1)
detector.fit(X_train)

# Detect anomalies
predictions = detector.predict(X_test)
scores = detector.decision_function(X_test)
```

### Using Deep Learning Methods

```python
from models import AnomalyAutoencoder
from training import AnomalyTrainer

# Initialize autoencoder
model = AnomalyAutoencoder(
    input_dim=X_train.shape[1],
    hidden_dims=[128, 64, 32],
    latent_dim=16
)

# Train
trainer = AnomalyTrainer()
trainer.train(model, X_train, epochs=100)

# Get reconstruction errors as anomaly scores
scores = model.get_reconstruction_error(X_test)
```

### Ensemble Detection

```python
from models import EnsembleAnomalyDetector

# Combine multiple detectors
ensemble = EnsembleAnomalyDetector([
    ('if', IsolationForest()),
    ('svm', OneClassSVM()),
    ('ae', AnomalyAutoencoder())
])

ensemble.fit(X_train)
predictions, uncertainty = ensemble.predict_with_uncertainty(X_test)
```

## Evaluation

The framework includes standard anomaly detection metrics:

- **Precision/Recall/F1**: Classification performance
- **AUC-ROC**: Ranking quality across thresholds
- **AUC-PR**: Useful for highly imbalanced datasets

```python
from evaluation import AnomalyEvaluator

evaluator = AnomalyEvaluator()
metrics = evaluator.evaluate(y_test, predictions, scores)
print(f"F1-Score: {metrics['f1']:.3f}")
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
```

## Supported Datasets

The system has been tested on:
- Credit card fraud detection
- Manufacturing defect detection
- Network intrusion detection
- IoT sensor anomalies
- Server log analysis

See `docs/UNDERSTANDING_RESULTS.md` for performance benchmarks and interpretation guidelines.

## Project Structure

```
anomaly_detection_system/
├── models/           # Detection algorithms
├── data/             # Data loading and preprocessing
├── evaluation/       # Metrics and evaluation tools
├── visualization/    # Plotting utilities
├── examples/         # Usage examples
└── docs/             # Additional documentation
```

## Testing

```bash
# Run example detection on sample dataset
python examples/demos.py

# Run all examples
python examples/demos.py --all
```

## References

- Liu, F. T., et al. "Isolation Forest" (2008)
- Schölkopf, B., et al. "Estimating the Support of a High-Dimensional Distribution" (2001)
- Breunig, M. M., et al. "LOF: Identifying Density-Based Local Outliers" (2000)

## License

MIT License - see LICENSE file for details.
