"""Example: Deploy Your Own Custom Model."""

import sys
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import MLOpsPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Deploy a custom model through the pipeline."""

    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize pipeline
    pipeline = MLOpsPipeline(config)

    # Create user
    user = "ml_engineer"
    pipeline.security_manager.create_user(username=user, role="ml_engineer")

    logger.info("="*60)
    logger.info("Custom Model Deployment Example")
    logger.info("="*60)

    # 1. Generate custom dataset
    logger.info("\n1. Generating custom dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    logger.info(f"Created dataset: {df.shape[0]} samples, {df.shape[1]-1} features")

    # 2. Train your custom model
    logger.info("\n2. Training custom Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(df[feature_names], df['target'])

    # Evaluate
    accuracy = model.score(df[feature_names], df['target'])
    logger.info(f"Model accuracy: {accuracy:.4f}")

    # 3. Register the model
    logger.info("\n3. Registering model in registry...")
    version_id = pipeline.train_and_register_model(
        model=model,
        model_name="custom_logistic_regression",
        data_version="v1.0",
        metrics={
            'accuracy': accuracy,
            'n_features': len(feature_names),
            'n_samples': len(df)
        },
        parameters={
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 42
        },
        user=user
    )
    logger.info(f"Model registered: {version_id}")

    # 4. Deploy the model
    logger.info("\n4. Deploying model with canary strategy...")
    from security.security_manager import ResourceType, AccessLevel

    # Extract version from version_id (format: "model_name:version")
    model_version = version_id.split(':')[1]

    pipeline.security_manager.grant_access(
        user=user,
        resource_type=ResourceType.DEPLOYMENT,
        resource_id=version_id,
        access_level=AccessLevel.WRITE
    )

    deployment = pipeline.deploy_model(
        model_name="custom_logistic_regression",
        model_version=model_version,
        strategy="canary",  # Try canary deployment!
        replicas=3,
        user=user
    )

    logger.info(f"Deployment successful!")
    logger.info(f"  Deployment ID: {deployment['deployment_id']}")
    logger.info(f"  Strategy: {deployment['strategy']}")
    logger.info(f"  Status: {deployment['status']}")

    # 5. Test predictions
    logger.info("\n5. Testing model predictions...")
    test_sample = df[feature_names].iloc[0:5]
    predictions = model.predict(test_sample)

    logger.info("Sample predictions:")
    for i, pred in enumerate(predictions):
        logger.info(f"  Sample {i+1}: {pred}")

    # 6. Monitor
    logger.info("\n6. Setting up monitoring...")
    for i in range(100):
        idx = np.random.randint(0, len(df))
        pred = model.predict([df[feature_names].iloc[idx]])[0]
        true = df['target'].iloc[idx]

        pipeline.performance_monitor.record_prediction(
            prediction=pred,
            ground_truth=true,
            latency_ms=np.random.uniform(5, 50),
            model_version="v1"
        )

    summary = pipeline.performance_monitor.get_performance_summary()
    logger.info(f"\nMonitoring Summary:")
    logger.info(f"  Total Predictions: {summary['total_predictions']}")
    logger.info(f"  Error Rate: {summary['error_rate']:.2%}")
    logger.info(f"  Avg Latency: {summary['avg_latency_ms']:.2f}ms")

    logger.info("\n" + "="*60)
    logger.info("Custom Model Deployment Completed!")
    logger.info("="*60)

    # Next steps
    logger.info("\n📝 Next Steps:")
    logger.info("  1. Try different models (XGBoost, Neural Networks)")
    logger.info("  2. Compare models with A/B testing")
    logger.info("  3. Deploy to Kubernetes")
    logger.info("  4. Set up real-time monitoring with Grafana")


if __name__ == "__main__":
    main()
