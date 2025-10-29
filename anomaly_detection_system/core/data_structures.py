"""
Core data structures for anomaly detection system
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum
import numpy as np


class AnomalyType(Enum):
    """Types of anomalies"""
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    PATTERN = "pattern"


@dataclass
class AnomalyScore:
    """Anomaly score with metadata"""
    score: float
    is_anomaly: bool
    confidence: float
    anomaly_type: AnomalyType
    features_contribution: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None


@dataclass
class DetectionResult:
    """Complete detection result"""
    scores: np.ndarray
    labels: np.ndarray
    anomaly_indices: np.ndarray
    metadata: Dict[str, Any]