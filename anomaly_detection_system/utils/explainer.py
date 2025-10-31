import numpy as np
from typing import Dict, List, Optional
from models.base import BaseAnomalyDetector

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