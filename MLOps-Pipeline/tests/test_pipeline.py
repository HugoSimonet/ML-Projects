"""Tests for MLOps Pipeline."""

import pytest
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import MLOpsPipeline
from data import DataIngestion, DataValidator, FeatureEngineer
from registry import ModelRegistry
from deployment import DeploymentManager, DeploymentConfig, DeploymentStrategy
from monitoring import PerformanceMonitor, DriftDetector
from testing import ABTest
from security import SecurityManager


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'data': {
            'ingestion': {'source_type': 'local', 'source_path': './data'},
            'validation': {'thresholds': {'max_missing_pct': 0.1}},
            'features': {'numeric_features': ['feature_1', 'feature_2'], 'scaling_method': 'standard'},
            'versioning': {'version_dir': './test_data_versions'}
        },
        'registry': {'registry_dir': './test_model_registry', 'storage_backend': 'local'},
        'deployment': {'platform': 'local', 'namespace': 'test'},
        'monitoring': {'window_size': 100},
        'drift_detection': {'drift_threshold': 0.05},
        'infrastructure': {'kubernetes': {'namespace': 'test'}},
        'security': {'enable_rbac': True, 'enable_audit': True}
    }


@pytest.fixture
def sample_data():
    """Generate sample data."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })


def test_data_ingestion(config):
    """Test data ingestion."""
    ingestion = DataIngestion(config['data']['ingestion'])
    data = ingestion.ingest_data('test_dataset')

    assert data is not None
    assert len(data) > 0
    assert 'feature_1' in data.columns


def test_data_validation(config, sample_data):
    """Test data validation."""
    validator = DataValidator(config['data']['validation'])
    result = validator.validate(sample_data)

    assert result is not None
    assert hasattr(result, 'passed')


def test_feature_engineering(config, sample_data):
    """Test feature engineering."""
    engineer = FeatureEngineer(config['data']['features'])
    transformed = engineer.fit_transform(sample_data)

    assert transformed is not None
    assert len(transformed) == len(sample_data)


def test_model_registry(config):
    """Test model registry."""
    registry = ModelRegistry(config['registry'])

    # Train simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)

    # Register model
    version_id = registry.register_model(
        model=model,
        model_name='test_model',
        metadata={'metrics': {'accuracy': 0.85}},
        tags=['test']
    )

    assert version_id is not None
    assert 'test_model:v1' == version_id

    # Retrieve model
    retrieved_model, metadata = registry.get_model('test_model')
    assert retrieved_model is not None
    assert metadata['model_name'] == 'test_model'


def test_deployment_manager(config):
    """Test deployment manager."""
    manager = DeploymentManager(config['deployment'])

    deployment_config = DeploymentConfig(
        model_version='test_model:v1',
        strategy=DeploymentStrategy.BLUE_GREEN,
        replicas=2
    )

    deployment = manager.deploy_model(deployment_config)

    assert deployment is not None
    assert deployment.model_version == 'test_model:v1'
    assert deployment.replicas == 2


def test_performance_monitor(config):
    """Test performance monitoring."""
    monitor = PerformanceMonitor(config['monitoring'])

    # Record some predictions
    for i in range(10):
        monitor.record_prediction(
            prediction=i % 2,
            ground_truth=i % 2,
            latency_ms=10.0,
            model_version='v1'
        )

    summary = monitor.get_performance_summary()

    assert summary is not None
    assert summary['total_predictions'] == 10
    assert summary['error_rate'] == 0.0


def test_drift_detector(config, sample_data):
    """Test drift detection."""
    detector = DriftDetector(config['drift_detection'])

    # Set reference data
    detector.set_reference_data(sample_data)

    # Create slightly different data
    current_data = sample_data.copy()
    current_data['feature_1'] += 0.1

    # Detect drift
    results = detector.detect_data_drift(current_data)

    assert results is not None
    assert len(results) > 0
    assert hasattr(results[0], 'drift_detected')


def test_ab_testing(config):
    """Test A/B testing framework."""
    test_config = {
        'control_model': 'v1',
        'treatment_model': 'v2',
        'success_metric': 'accuracy',
        'traffic_split': 0.5
    }

    ab_test = ABTest(test_config)

    # Record some predictions
    for i in range(50):
        variant = 'control' if i < 25 else 'treatment'
        ab_test.record_prediction(
            variant=variant,
            prediction=i % 2,
            ground_truth=i % 2
        )

    # Check if ready
    is_ready = ab_test.is_ready_for_decision()

    assert is_ready == False  # Need min_sample_size


def test_security_manager(config):
    """Test security manager."""
    security = SecurityManager(config['security'])

    # Create user
    user = security.create_user(
        username='test_user',
        role='data_scientist'
    )

    assert user is not None
    assert user['username'] == 'test_user'
    assert 'api_key' in user

    # Test authentication
    authenticated = security.authenticate(user['api_key'])
    assert authenticated == 'test_user'


def test_mlops_pipeline(config):
    """Test complete MLOps pipeline."""
    pipeline = MLOpsPipeline(config)

    # Test pipeline initialization
    assert pipeline is not None
    assert pipeline.data_ingestion is not None
    assert pipeline.model_registry is not None
    assert pipeline.deployment_manager is not None

    # Test pipeline status
    status = pipeline.get_pipeline_status()
    assert status is not None
    assert 'timestamp' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
