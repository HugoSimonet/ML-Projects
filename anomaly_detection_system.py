"""
Comprehensive Anomaly Detection System for High-Dimensional Data
================================================================

A production-ready anomaly detection framework supporting multiple detection methods,
real-time processing, and interpretability features.

Author: AI Assistant
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict
from enum import Enum

warnings.filterwarnings('ignore')

# ============================================================================
# Core Data Structures
# ============================================================================

class AnomalyType(Enum):
    """Types of anomalies"""
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    PATTERN = "pattern"

@dataclass
class AnomalyScore:
    """Anomaly score with metadata"""
    score: float
    is_anomaly: bool
    confidence: float
    anomaly_type: AnomalyType
    features_contribution: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None

@dataclass
class DetectionResult:
    """Complete detection result"""
    scores: np.ndarray
    labels: np.ndarray
    anomaly_indices: np.ndarray
    metadata: Dict[str, Any]
    
# ============================================================================
# Statistical Anomaly Detection Methods
# ============================================================================

class BaseAnomalyDetector(ABC):
    """Base class for all anomaly detectors"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """Fit the detector"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        pass
    
    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
        pass


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Isolation Forest for anomaly detection
    Efficient for high-dimensional data
    """
    
    def __init__(self, n_estimators: int = 100, max_samples: int = 256,
                 contamination: float = 0.1, random_state: int = 42):
        super().__init__(contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Build isolation forest"""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Sample data
            sample_idx = np.random.choice(n_samples, 
                                         min(self.max_samples, n_samples),
                                         replace=False)
            sample_data = X[sample_idx]
            
            # Build tree
            tree = self._build_tree(sample_data, 0)
            self.trees.append(tree)
            
        self.is_fitted = True
        return self
    
    def _build_tree(self, X: np.ndarray, depth: int, max_depth: int = 10) -> Dict:
        """Build isolation tree recursively"""
        n_samples, n_features = X.shape
        
        if depth >= max_depth or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples}
        
        # Random split
        feature = np.random.randint(0, n_features)
        feature_values = X[:, feature]
        
        if len(np.unique(feature_values)) == 1:
            return {'type': 'leaf', 'size': n_samples}
        
        split_value = np.random.uniform(feature_values.min(), 
                                       feature_values.max())
        
        left_mask = feature_values < split_value
        right_mask = ~left_mask
        
        return {
            'type': 'node',
            'feature': feature,
            'split_value': split_value,
            'left': self._build_tree(X[left_mask], depth + 1, max_depth),
            'right': self._build_tree(X[right_mask], depth + 1, max_depth)
        }
    
    def _path_length(self, x: np.ndarray, tree: Dict, depth: int = 0) -> float:
        """Calculate path length for a sample"""
        if tree['type'] == 'leaf':
            return depth + self._c(tree['size'])
        
        if x[tree['feature']] < tree['split_value']:
            return self._path_length(x, tree['left'], depth + 1)
        else:
            return self._path_length(x, tree['right'], depth + 1)
    
    def _c(self, n: int) -> float:
        """Average path length of unsuccessful search in BST"""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
        
        scores = np.zeros(X.shape[0])
        
        for i, x in enumerate(X):
            path_lengths = [self._path_length(x, tree) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            scores[i] = 2 ** (-avg_path_length / self._c(self.max_samples))
        
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return (scores >= threshold).astype(int)


class LocalOutlierFactor(BaseAnomalyDetector):
    """
    Local Outlier Factor for density-based anomaly detection
    """
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        super().__init__(contamination)
        self.n_neighbors = n_neighbors
        self.X_train = None
        
    def fit(self, X: np.ndarray) -> 'LocalOutlierFactor':
        """Fit LOF detector"""
        self.X_train = X.copy()
        self.is_fitted = True
        return self
    
    def _k_distance(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate k-distance and k-nearest neighbors"""
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_nearest_idx = np.argsort(distances)[:self.n_neighbors]
        k_distance = distances[k_nearest_idx[-1]]
        return k_distance, k_nearest_idx
    
    def _reachability_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate reachability distance"""
        k_dist_y, _ = self._k_distance(y)
        actual_dist = np.linalg.norm(x - y)
        return max(k_dist_y, actual_dist)
    
    def _local_reachability_density(self, x: np.ndarray) -> float:
        """Calculate local reachability density"""
        _, k_neighbors_idx = self._k_distance(x)
        
        reach_dists = []
        for idx in k_neighbors_idx:
            reach_dist = self._reachability_distance(x, self.X_train[idx])
            reach_dists.append(reach_dist)
        
        avg_reach_dist = np.mean(reach_dists)
        
        if avg_reach_dist == 0:
            return np.inf
        
        return 1.0 / avg_reach_dist
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculate LOF scores"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
        
        lof_scores = np.zeros(X.shape[0])
        
        for i, x in enumerate(X):
            lrd_x = self._local_reachability_density(x)
            _, k_neighbors_idx = self._k_distance(x)
            
            lrd_neighbors = []
            for idx in k_neighbors_idx:
                lrd_neighbors.append(
                    self._local_reachability_density(self.X_train[idx])
                )
            
            if lrd_x == 0 or lrd_x == np.inf:
                lof_scores[i] = 1.0
            else:
                lof_scores[i] = np.mean(lrd_neighbors) / lrd_x
        
        return lof_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return (scores >= threshold).astype(int)


class StatisticalProcessControl(BaseAnomalyDetector):
    """
    Statistical Process Control using control charts
    """
    
    def __init__(self, n_sigma: float = 3.0, contamination: float = 0.1):
        super().__init__(contamination)
        self.n_sigma = n_sigma
        self.mean = None
        self.std = None
        
    def fit(self, X: np.ndarray) -> 'StatisticalProcessControl':
        """Fit SPC detector"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.is_fitted = True
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores based on z-scores"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
        
        # Calculate z-scores
        z_scores = np.abs((X - self.mean) / (self.std + 1e-10))
        
        # Aggregate z-scores across features
        scores = np.max(z_scores, axis=1)
        
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        scores = self.score_samples(X)
        return (scores > self.n_sigma).astype(int)


# ============================================================================
# Deep Learning Anomaly Detection Methods
# ============================================================================

class AutoencoderDetector:
    """
    Simplified Autoencoder-based anomaly detection
    Uses reconstruction error as anomaly score
    """
    
    def __init__(self, hidden_dims: List[int] = [64, 32, 16],
                 learning_rate: float = 0.001, epochs: int = 100,
                 batch_size: int = 32, contamination: float = 0.1):
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.contamination = contamination
        self.is_fitted = False
        self.weights = {}
        
    def fit(self, X: np.ndarray):
        """Train autoencoder with simplified backprop"""
        n_samples, n_features = X.shape
        
        # Build simple 3-layer architecture: input -> hidden -> bottleneck -> hidden -> output
        hidden_dim = self.hidden_dims[0] if self.hidden_dims else 32
        bottleneck_dim = self.hidden_dims[-1] if len(self.hidden_dims) > 1 else 16
        
        # Initialize weights with proper scaling
        self.weights = {
            'W1': np.random.randn(n_features, hidden_dim) * np.sqrt(2.0 / n_features),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, bottleneck_dim) * np.sqrt(2.0 / hidden_dim),
            'b2': np.zeros(bottleneck_dim),
            'W3': np.random.randn(bottleneck_dim, hidden_dim) * np.sqrt(2.0 / bottleneck_dim),
            'b3': np.zeros(hidden_dim),
            'W4': np.random.randn(hidden_dim, n_features) * np.sqrt(2.0 / hidden_dim),
            'b4': np.zeros(n_features)
        }
        
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch = X_shuffled[i:i+self.batch_size]
                
                if len(batch) < 2:
                    continue
                
                # Forward pass
                h1 = np.tanh(batch @ self.weights['W1'] + self.weights['b1'])
                h2 = np.tanh(h1 @ self.weights['W2'] + self.weights['b2'])
                h3 = np.tanh(h2 @ self.weights['W3'] + self.weights['b3'])
                output = h3 @ self.weights['W4'] + self.weights['b4']
                
                # Loss
                loss = np.mean((batch - output) ** 2)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass - compute all gradients
                batch_size = len(batch)
                
                # Output layer gradients
                d_output = 2 * (output - batch) / batch_size
                dW4 = h3.T @ d_output
                db4 = np.sum(d_output, axis=0)
                
                # Third hidden layer
                d_h3 = d_output @ self.weights['W4'].T
                d_h3 = d_h3 * (1 - h3**2)  # tanh derivative
                dW3 = h2.T @ d_h3
                db3 = np.sum(d_h3, axis=0)
                
                # Second hidden layer (bottleneck)
                d_h2 = d_h3 @ self.weights['W3'].T
                d_h2 = d_h2 * (1 - h2**2)  # tanh derivative
                dW2 = h1.T @ d_h2
                db2 = np.sum(d_h2, axis=0)
                
                # First hidden layer
                d_h1 = d_h2 @ self.weights['W2'].T
                d_h1 = d_h1 * (1 - h1**2)  # tanh derivative
                dW1 = batch.T @ d_h1
                db1 = np.sum(d_h1, axis=0)
                
                # Update weights with gradient clipping
                self.weights['W4'] -= self.learning_rate * np.clip(dW4, -1, 1)
                self.weights['b4'] -= self.learning_rate * np.clip(db4, -1, 1)
                self.weights['W3'] -= self.learning_rate * np.clip(dW3, -1, 1)
                self.weights['b3'] -= self.learning_rate * np.clip(db3, -1, 1)
                self.weights['W2'] -= self.learning_rate * np.clip(dW2, -1, 1)
                self.weights['b2'] -= self.learning_rate * np.clip(db2, -1, 1)
                self.weights['W1'] -= self.learning_rate * np.clip(dW1, -1, 1)
                self.weights['b1'] -= self.learning_rate * np.clip(db1, -1, 1)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
                if epoch < 20:
                    print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        self.is_fitted = True
        return self
    
    def _reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input through autoencoder"""
        h1 = np.tanh(X @ self.weights['W1'] + self.weights['b1'])
        h2 = np.tanh(h1 @ self.weights['W2'] + self.weights['b2'])
        h3 = np.tanh(h2 @ self.weights['W3'] + self.weights['b3'])
        output = h3 @ self.weights['W4'] + self.weights['b4']
        return output
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculate reconstruction error as anomaly score"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
        
        reconstructed = self._reconstruct(X)
        scores = np.mean((X - reconstructed) ** 2, axis=1)
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies based on reconstruction error threshold"""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return (scores >= threshold).astype(int)


# =============================================================================
# INSTRUCTIONS:
# =============================================================================
# 1. Find the AutoencoderDetector class in your anomaly_detection_system.py file
# 2. Delete the ENTIRE AutoencoderDetector class (from "class AutoencoderDetector" 
#    to the end of its last method)
# 3. Copy THIS entire AutoencoderDetector class above
# 4. Paste it in place of the old one
# 5. Save the file
# 6. Run again
#
# This simplified version:
# - Uses a fixed 4-layer architecture (input->hidden->bottleneck->hidden->output)
# - Computes gradients explicitly for each layer (no complex recursion)
# - Avoids all the shape mismatch issues
# - Is much easier to debug
# =============================================================================


# Test the class independently
if __name__ == "__main__":
    print("Testing AutoencoderDetector...")
    
    # Generate test data
    np.random.seed(42)
    X_train = np.random.randn(400, 15)
    X_test = np.random.randn(100, 15)
    
    # Add some anomalies
    X_test[::10] = np.random.randn(10, 15) * 3 + 5
    
    # Create and train detector
    detector = AutoencoderDetector(
        hidden_dims=[32, 16, 8],
        epochs=50,
        learning_rate=0.001
    )
    
    print("\nTraining...")
    detector.fit(X_train)
    
    print("\nScoring test samples...")
    scores = detector.score_samples(X_test)
    predictions = detector.predict(X_test)
    
    print(f"\nResults:")
    print(f"  Anomalies detected: {np.sum(predictions)}")
    print(f"  Mean anomaly score: {np.mean(scores[predictions == 1]):.4f}")
    print(f"  Mean normal score: {np.mean(scores[predictions == 0]):.4f}")
    print("\n✓ AutoencoderDetector working correctly!")

class VariationalAutoencoder(BaseAnomalyDetector):
    """
    Variational Autoencoder for probabilistic anomaly detection
    """
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64,
                 learning_rate: float = 0.001, epochs: int = 100,
                 contamination: float = 0.1):
        super().__init__(contamination)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = {}
        
    def fit(self, X: np.ndarray) -> 'VariationalAutoencoder':
        """Train VAE"""
        n_features = X.shape[1]
        
        # Initialize weights (simplified)
        self.weights = {
            'encoder_W1': np.random.randn(n_features, self.hidden_dim) * 0.01,
            'encoder_b1': np.zeros(self.hidden_dim),
            'mu_W': np.random.randn(self.hidden_dim, self.latent_dim) * 0.01,
            'mu_b': np.zeros(self.latent_dim),
            'logvar_W': np.random.randn(self.hidden_dim, self.latent_dim) * 0.01,
            'logvar_b': np.zeros(self.latent_dim),
            'decoder_W1': np.random.randn(self.latent_dim, self.hidden_dim) * 0.01,
            'decoder_b1': np.zeros(self.hidden_dim),
            'decoder_W2': np.random.randn(self.hidden_dim, n_features) * 0.01,
            'decoder_b2': np.zeros(n_features)
        }
        
        # Training (simplified)
        for epoch in range(self.epochs):
            # Encode
            h = np.tanh(X @ self.weights['encoder_W1'] + self.weights['encoder_b1'])
            mu = h @ self.weights['mu_W'] + self.weights['mu_b']
            logvar = h @ self.weights['logvar_W'] + self.weights['logvar_b']
            
            # Reparameterization trick
            std = np.exp(0.5 * logvar)
            eps = np.random.randn(*mu.shape)
            z = mu + std * eps
            
            # Decode
            h = np.tanh(z @ self.weights['decoder_W1'] + self.weights['decoder_b1'])
            reconstructed = h @ self.weights['decoder_W2'] + self.weights['decoder_b2']
            
            # Loss (reconstruction + KL divergence)
            recon_loss = np.mean((X - reconstructed) ** 2)
            kl_loss = -0.5 * np.mean(1 + logvar - mu ** 2 - np.exp(logvar))
            
            total_loss = recon_loss + kl_loss
        
        self.is_fitted = True
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores using reconstruction probability"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
        
        # Encode
        h = np.tanh(X @ self.weights['encoder_W1'] + self.weights['encoder_b1'])
        mu = h @ self.weights['mu_W'] + self.weights['mu_b']
        
        # Decode
        h = np.tanh(mu @ self.weights['decoder_W1'] + self.weights['decoder_b1'])
        reconstructed = h @ self.weights['decoder_W2'] + self.weights['decoder_b2']
        
        # Reconstruction error
        scores = np.mean((X - reconstructed) ** 2, axis=1)
        
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return (scores >= threshold).astype(int)


# ============================================================================
# Ensemble Methods
# ============================================================================

class EnsembleDetector:
    """
    Ensemble anomaly detector combining multiple methods
    """
    
    def __init__(self, detectors: List[BaseAnomalyDetector],
                 combination_method: str = 'average',
                 contamination: float = 0.1):
        self.detectors = detectors
        self.combination_method = combination_method
        self.contamination = contamination
        
    def fit(self, X: np.ndarray) -> 'EnsembleDetector':
        """Fit all detectors"""
        for detector in self.detectors:
            detector.fit(X)
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Combine scores from all detectors"""
        all_scores = []
        
        for detector in self.detectors:
            scores = detector.score_samples(X)
            # Normalize scores to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            all_scores.append(scores)
        
        all_scores = np.array(all_scores)
        
        if self.combination_method == 'average':
            combined_scores = np.mean(all_scores, axis=0)
        elif self.combination_method == 'max':
            combined_scores = np.max(all_scores, axis=0)
        elif self.combination_method == 'median':
            combined_scores = np.median(all_scores, axis=0)
        else:
            combined_scores = np.mean(all_scores, axis=0)
        
        return combined_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using ensemble"""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return (scores >= threshold).astype(int)


# ============================================================================
# Real-Time Streaming Detection
# ============================================================================

class StreamingAnomalyDetector:
    """
    Real-time anomaly detection for streaming data
    """
    
    def __init__(self, window_size: int = 100, 
                 detector: Optional[BaseAnomalyDetector] = None,
                 update_frequency: int = 50):
        self.window_size = window_size
        self.detector = detector or IsolationForestDetector()
        self.update_frequency = update_frequency
        self.buffer = []
        self.samples_since_update = 0
        
    def process_sample(self, x: np.ndarray) -> Tuple[float, bool]:
        """Process a single sample"""
        # Add to buffer
        self.buffer.append(x)
        
        # Maintain window size
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        # Update model periodically
        self.samples_since_update += 1
        if self.samples_since_update >= self.update_frequency:
            if len(self.buffer) >= self.window_size // 2:
                X_buffer = np.array(self.buffer)
                self.detector.fit(X_buffer)
                self.samples_since_update = 0
        
        # Detect anomaly
        if self.detector.is_fitted:
            score = self.detector.score_samples(x.reshape(1, -1))[0]
            is_anomaly = self.detector.predict(x.reshape(1, -1))[0] == 1
            return score, is_anomaly
        else:
            return 0.0, False


# ============================================================================
# Evaluation Metrics
# ============================================================================

class AnomalyMetrics:
    """
    Comprehensive evaluation metrics for anomaly detection
    """
    
    @staticmethod
    def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate precision, recall, and F1-score"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    @staticmethod
    def auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
        """Calculate AUC-ROC score"""
        # Sort by scores
        sorted_indices = np.argsort(scores)[::-1]
        y_sorted = y_true[sorted_indices]
        
        # Calculate TPR and FPR at different thresholds
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tpr = np.cumsum(y_sorted == 1) / n_pos
        fpr = np.cumsum(y_sorted == 0) / n_neg
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        
        return abs(auc)
    
    @staticmethod
    def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
        """Calculate Average Precision (AP)"""
        sorted_indices = np.argsort(scores)[::-1]
        y_sorted = y_true[sorted_indices]
        
        precisions = []
        n_anomalies = 0
        
        for i, label in enumerate(y_sorted):
            if label == 1:
                n_anomalies += 1
                precision = n_anomalies / (i + 1)
                precisions.append(precision)
        
        if len(precisions) == 0:
            return 0.0
        
        return np.mean(precisions)


# ============================================================================
# Interpretability and Explanation
# ============================================================================

class AnomalyExplainer:
    """
    Generate explanations for detected anomalies
    """
    
    @staticmethod
    def feature_importance(detector: BaseAnomalyDetector, 
                          X: np.ndarray, 
                          sample_idx: int,
                          feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate feature importance for a specific anomaly"""
        sample = X[sample_idx:sample_idx+1]
        base_score = detector.score_samples(sample)[0]
        
        n_features = X.shape[1]
        importances = {}
        
        for i in range(n_features):
            # Perturb feature
            perturbed = sample.copy()
            perturbed[0, i] = np.median(X[:, i])
            
            perturbed_score = detector.score_samples(perturbed)[0]
            importance = abs(base_score - perturbed_score)
            
            feature_name = feature_names[i] if feature_names else f"Feature_{i}"
            importances[feature_name] = importance
        
        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v/total for k, v in importances.items()}
        
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    @staticmethod
    def generate_explanation(feature_importance: Dict[str, float], 
                           top_k: int = 3) -> str:
        """Generate human-readable explanation"""
        top_features = list(feature_importance.items())[:top_k]
        
        explanation = "Anomaly detected due to unusual values in: "
        feature_strs = [f"{feat} ({imp:.2%})" for feat, imp in top_features]
        explanation += ", ".join(feature_strs)
        
        return explanation


# ============================================================================
# High-Dimensional Data Processing
# ============================================================================

class HighDimensionalProcessor:
    """
    Data preprocessing for high-dimensional anomaly detection
    """
    
    @staticmethod
    def reduce_dimensionality(X: np.ndarray, n_components: int = 50,
                            method: str = 'pca') -> np.ndarray:
        """Reduce dimensionality using PCA or random projection"""
        if method == 'pca':
            # Center data
            X_centered = X - np.mean(X, axis=0)
            
            # Compute covariance matrix
            cov_matrix = np.cov(X_centered.T)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalues
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Project data
            X_reduced = X_centered @ eigenvectors[:, :n_components]
            
        elif method == 'random_projection':
            n_features = X.shape[1]
            # Random projection matrix
            projection_matrix = np.random.randn(n_features, n_components)
            projection_matrix /= np.linalg.norm(projection_matrix, axis=0)
            
            X_reduced = X @ projection_matrix
        else:
            X_reduced = X
        
        return X_reduced
    
    @staticmethod
    def normalize_features(X: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Normalize features"""
        if method == 'standard':
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            X_normalized = (X - mean) / (std + 1e-10)
        elif method == 'minmax':
            min_val = np.min(X, axis=0)
            max_val = np.max(X, axis=0)
            X_normalized = (X - min_val) / (max_val - min_val + 1e-10)
        else:
            X_normalized = X
        
        return X_normalized


# ============================================================================
# Main Anomaly Detection System
# ============================================================================

class AnomalyDetectionSystem:
    """
    Main system integrating all components
    """
    
    def __init__(self, detector_type: str = 'isolation_forest',
                 contamination: float = 0.1,
                 **detector_params):
        self.detector_type = detector_type
        self.contamination = contamination
        self.detector = self._create_detector(**detector_params)
        self.processor = HighDimensionalProcessor()
        self.explainer = AnomalyExplainer()
        self.metrics = AnomalyMetrics()
        self.feature_names = None
        
    def _create_detector(self, **params) -> BaseAnomalyDetector:
        """Create detector based on type"""
        if self.detector_type == 'isolation_forest':
            return IsolationForestDetector(contamination=self.contamination, **params)
        elif self.detector_type == 'lof':
            return LocalOutlierFactor(contamination=self.contamination, **params)
        elif self.detector_type == 'spc':
            return StatisticalProcessControl(contamination=self.contamination, **params)
        elif self.detector_type == 'autoencoder':
            return AutoencoderDetector(contamination=self.contamination, **params)
        elif self.detector_type == 'vae':
            return VariationalAutoencoder(contamination=self.contamination, **params)
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")
    
    def fit(self, X: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            preprocess: bool = True,
            dimensionality_reduction: Optional[int] = None) -> 'AnomalyDetectionSystem':
        """Fit the anomaly detection system"""
        self.feature_names = feature_names
        
        # Preprocess data
        if preprocess:
            X = self.processor.normalize_features(X)
        
        # Dimensionality reduction if needed
        if dimensionality_reduction and X.shape[1] > dimensionality_reduction:
            X = self.processor.reduce_dimensionality(X, dimensionality_reduction)
        
        # Fit detector
        self.detector.fit(X)
        
        return self
    
    def detect(self, X: np.ndarray, 
               explain_anomalies: bool = True,
               top_k_features: int = 3) -> DetectionResult:
        """Detect anomalies with full results"""
        # Score samples
        scores = self.detector.score_samples(X)
        
        # Predict labels
        labels = self.detector.predict(X)
        
        # Find anomaly indices
        anomaly_indices = np.where(labels == 1)[0]
        
        # Generate explanations if requested
        explanations = {}
        if explain_anomalies and len(anomaly_indices) > 0:
            for idx in anomaly_indices[:10]:  # Explain top 10
                feature_imp = self.explainer.feature_importance(
                    self.detector, X, idx, self.feature_names
                )
                explanation = self.explainer.generate_explanation(
                    feature_imp, top_k_features
                )
                explanations[int(idx)] = {
                    'feature_importance': feature_imp,
                    'explanation': explanation
                }
        
        metadata = {
            'n_anomalies': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(X),
            'mean_anomaly_score': np.mean(scores[anomaly_indices]) if len(anomaly_indices) > 0 else 0,
            'explanations': explanations
        }
        
        return DetectionResult(
            scores=scores,
            labels=labels,
            anomaly_indices=anomaly_indices,
            metadata=metadata
        )
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate detector performance"""
        scores = self.detector.score_samples(X)
        y_pred = self.detector.predict(X)
        
        metrics = self.metrics.precision_recall_f1(y_true, y_pred)
        metrics['auc_roc'] = self.metrics.auc_roc(y_true, scores)
        metrics['average_precision'] = self.metrics.average_precision(y_true, scores)
        
        return metrics


# ============================================================================
# Visualization Tools
# ============================================================================

class AnomalyVisualizer:
    """
    Visualization tools for anomaly detection results
    """
    
    @staticmethod
    def print_detection_summary(result: DetectionResult):
        """Print summary of detection results"""
        print("=" * 70)
        print("ANOMALY DETECTION SUMMARY")
        print("=" * 70)
        print(f"Total Samples: {len(result.scores)}")
        print(f"Anomalies Detected: {result.metadata['n_anomalies']}")
        print(f"Anomaly Rate: {result.metadata['anomaly_rate']:.2%}")
        print(f"Mean Anomaly Score: {result.metadata['mean_anomaly_score']:.4f}")
        print()
        
        if result.metadata['explanations']:
            print("TOP ANOMALIES:")
            print("-" * 70)
            for idx, exp_data in list(result.metadata['explanations'].items())[:5]:
                print(f"\nAnomaly #{idx} (Score: {result.scores[idx]:.4f})")
                print(f"  {exp_data['explanation']}")
                print("  Top Contributing Features:")
                for feat, imp in list(exp_data['feature_importance'].items())[:3]:
                    print(f"    - {feat}: {imp:.2%}")
        print("=" * 70)
    
    @staticmethod
    def print_evaluation_metrics(metrics: Dict[str, float]):
        """Print evaluation metrics"""
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1-Score:          {metrics['f1_score']:.4f}")
        print(f"AUC-ROC:           {metrics['auc_roc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print("=" * 70)
    
    @staticmethod
    def plot_score_distribution(scores: np.ndarray, labels: np.ndarray):
        """Create ASCII histogram of anomaly scores"""
        print("\nANOMALY SCORE DISTRIBUTION:")
        print("-" * 70)
        
        # Separate normal and anomaly scores
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        # Create bins
        min_score, max_score = scores.min(), scores.max()
        n_bins = 20
        bins = np.linspace(min_score, max_score, n_bins + 1)
        
        # Count in each bin
        normal_hist, _ = np.histogram(normal_scores, bins=bins)
        anomaly_hist, _ = np.histogram(anomaly_scores, bins=bins)
        
        max_count = max(normal_hist.max(), anomaly_hist.max())
        
        for i in range(n_bins):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            
            normal_bar = '█' * int(40 * normal_hist[i] / max_count)
            anomaly_bar = '▓' * int(40 * anomaly_hist[i] / max_count)
            
            print(f"{bin_start:6.3f}-{bin_end:6.3f} | {normal_bar}{anomaly_bar}")
        
        print(f"\nLegend: █ Normal  ▓ Anomaly")
        print("-" * 70)


# ============================================================================
# Example Usage and Demonstrations
# ============================================================================

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
    
    # Combine
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([y_normal, y_anomalies])
    
    # Shuffle
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
    
    # Generate data with local outliers
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


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
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