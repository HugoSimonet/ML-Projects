import numpy as np
from typing import Tuple, Optional
from models.base import BaseAnomalyDetector
from models.statistical_methods import IsolationForestDetector

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