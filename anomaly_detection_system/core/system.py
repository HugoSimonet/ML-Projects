import numpy as np
from typing import Dict, List, Optional
from models.base import BaseAnomalyDetector
from models.statistical_methods import (
    IsolationForestDetector, LocalOutlierFactor, StatisticalProcessControl
)
from models.deep_learning_methods import AutoencoderDetector, VariationalAutoencoder
from data.high_dimensional_data import HighDimensionalProcessor
from evaluation.anomaly_metrics import AnomalyMetrics
from utils.explainer import AnomalyExplainer
from core.data_structures import DetectionResult

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