import numpy as np

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