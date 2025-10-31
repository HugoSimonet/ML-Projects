import numpy as np
from typing import Dict

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