"""
Ensemble anomaly detection methods
"""

import numpy as np
from typing import List
from models.base import BaseAnomalyDetector


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