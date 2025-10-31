from abc import ABC, abstractmethod
import numpy as np

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