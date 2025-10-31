import numpy as np
from typing import Dict, Tuple
from models.base import BaseAnomalyDetector

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