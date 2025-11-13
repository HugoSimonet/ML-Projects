"""Simple deployment example."""

import sys
from pathlib import Path
import yaml
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import MLOpsPipeline
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Simple deployment workflow."""
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize pipeline
    pipeline = MLOpsPipeline(config)

    # Create user
    user = "deployer"
    pipeline.security_manager.create_user(username=user, role="deployer")

    # Create and train a simple model
    model = RandomForestClassifier(n_estimators=10)

    # Register model
    logger.info("Registering model...")
    version_id = pipeline.train_and_register_model(
        model=model,
        model_name="simple_model",
        data_version="v1",
        metrics={'accuracy': 0.85},
        parameters={'n_estimators': 10},
        user=user
    )

    logger.info(f"Model registered: {version_id}")

    # Grant deployment access
    from security.security_manager import ResourceType, AccessLevel
    pipeline.security_manager.grant_access(
        user=user,
        resource_type=ResourceType.DEPLOYMENT,
        resource_id=version_id,
        access_level=AccessLevel.WRITE
    )

    # Deploy model
    logger.info("Deploying model...")
    model_version = version_id.split(':')[1]
    deployment = pipeline.deploy_model(
        model_name="simple_model",
        model_version=model_version,
        strategy="blue_green",
        replicas=2,
        user=user
    )

    logger.info(f"Deployment successful: {deployment['deployment_id']}")
    logger.info(f"Status: {deployment['status']}")


if __name__ == "__main__":
    main()
