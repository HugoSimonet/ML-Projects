"""MLOps Pipeline - Production-ready ML deployment and monitoring."""

__version__ = "1.0.0"
__author__ = "MLOps Team"
__description__ = "Comprehensive MLOps pipeline for ML model deployment"

from pipeline import MLOpsPipeline
from registry import ModelRegistry
from deployment import ModelServer, DeploymentManager
from monitoring import PerformanceMonitor, DriftDetector
from testing import ABTest

__all__ = [
    'MLOpsPipeline',
    'ModelRegistry',
    'ModelServer',
    'DeploymentManager',
    'PerformanceMonitor',
    'DriftDetector',
    'ABTest'
]
