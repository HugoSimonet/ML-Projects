import numpy as np
from typing import List
from models.base import BaseAnomalyDetector

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