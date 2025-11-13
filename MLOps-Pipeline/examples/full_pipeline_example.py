"""Complete MLOps pipeline example."""

import sys
import logging
from pathlib import Path
import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import MLOpsPipeline


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = '../configs/pipeline_config.yaml'):
    """Load pipeline configuration."""
    # Handle relative paths from examples directory
    if not Path(config_path).exists():
        # Try from project root
        config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Run complete MLOps pipeline example."""
    logger.info("="*60)
    logger.info("MLOps Pipeline - Complete Example")
    logger.info("="*60)

    # Load configuration
    config = load_config()

    # Initialize pipeline
    logger.info("\n1. Initializing MLOps Pipeline...")
    pipeline = MLOpsPipeline(config)

    # Create a user for auditing
    user = "data_scientist"
    pipeline.security_manager.create_user(
        username=user,
        role="data_scientist",
        metadata={"team": "ml_team"}
    )

    # Step 1: Run data pipeline
    logger.info("\n2. Running Data Pipeline...")
    data_results = pipeline.run_data_pipeline(
        dataset_name="sample_dataset",
        user=user
    )
    logger.info(f"Data version: {data_results['version_id']}")
    logger.info(f"Rows: {data_results['num_rows']}, Features: {data_results['num_features']}")

    # Get the transformed data
    data = data_results['transformed_data']

    # Step 2: Train a model
    logger.info("\n3. Training Model...")

    # Prepare data - select only numeric features for training
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['target', 'id']]
    X = data[feature_cols].fillna(0)
    y = data['target']

    logger.info(f"Using features: {feature_cols}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
    }

    logger.info(f"Model Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Step 3: Register model
    logger.info("\n4. Registering Model...")
    version_id = pipeline.train_and_register_model(
        model=model,
        model_name="random_forest_classifier",
        data_version=data_results['version_id'],
        metrics=metrics,
        parameters={
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        user=user
    )
    logger.info(f"Model registered: {version_id}")

    # Step 4: Grant deployment access
    logger.info("\n5. Granting Deployment Access...")
    from security.security_manager import ResourceType, AccessLevel

    pipeline.security_manager.grant_access(
        user=user,
        resource_type=ResourceType.DEPLOYMENT,
        resource_id=version_id,
        access_level=AccessLevel.WRITE
    )

    # Step 5: Deploy model
    logger.info("\n6. Deploying Model...")
    deployment_info = pipeline.deploy_model(
        model_name="random_forest_classifier",
        model_version="v1",
        strategy="blue_green",
        replicas=3,
        user=user
    )
    logger.info(f"Deployment ID: {deployment_info['deployment_id']}")
    logger.info(f"Status: {deployment_info['status']}")

    # Step 6: Setup monitoring
    logger.info("\n7. Setting up Monitoring...")

    # Set reference data for drift detection
    pipeline.drift_detector.set_reference_data(X_train)

    # Simulate some predictions for monitoring
    for i in range(50):
        prediction = model.predict([X_test.iloc[i]])[0]
        ground_truth = y_test.iloc[i]

        pipeline.performance_monitor.record_prediction(
            prediction=prediction,
            ground_truth=ground_truth,
            latency_ms=np.random.uniform(10, 100),
            model_version="v1"
        )

    # Get monitoring results
    logger.info("\n8. Checking Monitoring Results...")
    monitoring_results = pipeline.monitor_deployment(
        deployment_id=deployment_info['deployment_id'],
        check_drift=True
    )

    logger.info(f"Performance Summary:")
    for key, value in monitoring_results['performance'].items():
        logger.info(f"  {key}: {value}")

    if monitoring_results['alerts']:
        logger.warning(f"Alerts: {len(monitoring_results['alerts'])}")
        for alert in monitoring_results['alerts']:
            logger.warning(f"  {alert['type']}: {alert['message']}")
    else:
        logger.info("No alerts triggered")

    # Step 7: Check data drift
    logger.info("\n9. Checking for Data Drift...")
    drift_results = pipeline.drift_detector.detect_data_drift(
        current_data=X_test.head(100),
        features=feature_cols[:5]  # Check first 5 features
    )

    drifted_features = [r.feature for r in drift_results if r.drift_detected]
    if drifted_features:
        logger.warning(f"Drift detected in: {drifted_features}")
    else:
        logger.info("No drift detected")

    # Step 8: Setup A/B test
    logger.info("\n10. Setting up A/B Test...")
    ab_test_config = {
        'test_id': 'rf_v1_vs_v2',
        'control_model': 'v1',
        'treatment_model': 'v2',
        'success_metric': 'accuracy',
        'traffic_split': 0.5,
        'confidence_level': 0.95,
        'min_sample_size': 100
    }

    ab_test_results = pipeline.run_ab_test(ab_test_config, user=user)
    logger.info(f"A/B Test: {ab_test_results['test_id']}")
    logger.info(f"Control: {ab_test_results['control_model']}")
    logger.info(f"Treatment: {ab_test_results['treatment_model']}")
    logger.info(f"Status: {ab_test_results['status']}")

    # Step 9: Get pipeline status
    logger.info("\n11. Pipeline Status...")
    status = pipeline.get_pipeline_status()
    logger.info(f"Registered Models: {status['registered_models']}")
    logger.info(f"Active Deployments: {status['active_deployments']}")
    logger.info(f"Total Predictions: {status['total_predictions']}")

    # Step 10: Generate compliance report
    logger.info("\n12. Generating Compliance Report...")
    compliance = pipeline.get_compliance_report(user=user)
    logger.info(f"RBAC Enabled: {compliance['rbac_enabled']}")
    logger.info(f"Audit Enabled: {compliance['audit_enabled']}")
    logger.info(f"Total Audit Logs: {compliance['total_audit_logs']}")

    # Export audit logs
    pipeline.security_manager.export_audit_logs('audit_logs.json')
    logger.info("Audit logs exported to audit_logs.json")

    logger.info("\n" + "="*60)
    logger.info("MLOps Pipeline Example Completed Successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
