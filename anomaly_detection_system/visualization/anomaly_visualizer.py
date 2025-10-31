import numpy as np
from typing import Dict
from core.data_structures import DetectionResult

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