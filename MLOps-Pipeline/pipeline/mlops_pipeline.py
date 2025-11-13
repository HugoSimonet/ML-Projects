"""Main MLOps pipeline orchestration module."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Import all components
from data import DataIngestion, DataValidator, FeatureEngineer, DataVersioning
from registry import ModelRegistry
from deployment import DeploymentManager, DeploymentConfig, DeploymentStrategy
from monitoring import PerformanceMonitor, DriftDetector
from testing import ABTest
from infrastructure import KubernetesManager
from security import SecurityManager


logger = logging.getLogger(__name__)


class MLOpsPipeline:
    """Main MLOps pipeline that orchestrates all components."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize MLOps pipeline.

        Args:
            config: Configuration dictionary containing:
                - data: Data pipeline configuration
                - registry: Model registry configuration
                - deployment: Deployment configuration
                - monitoring: Monitoring configuration
                - security: Security configuration
        """
        self.config = config

        # Initialize all components
        logger.info("Initializing MLOps Pipeline components...")

        # Data pipeline
        self.data_ingestion = DataIngestion(config.get('data', {}).get('ingestion', {}))
        self.data_validator = DataValidator(config.get('data', {}).get('validation', {}))
        self.feature_engineer = FeatureEngineer(config.get('data', {}).get('features', {}))
        self.data_versioning = DataVersioning(config.get('data', {}).get('versioning', {}))

        # Model registry
        self.model_registry = ModelRegistry(config.get('registry', {}))

        # Deployment
        self.deployment_manager = DeploymentManager(config.get('deployment', {}))

        # Monitoring
        self.performance_monitor = PerformanceMonitor(config.get('monitoring', {}))
        self.drift_detector = DriftDetector(config.get('drift_detection', {}))

        # Infrastructure
        self.kubernetes_manager = KubernetesManager(
            config.get('infrastructure', {}).get('kubernetes', {})
        )

        # Security
        self.security_manager = SecurityManager(config.get('security', {}))

        logger.info("MLOps Pipeline initialized successfully")

    def run_data_pipeline(
        self,
        dataset_name: str,
        user: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the data pipeline.

        Args:
            dataset_name: Name of dataset to process
            user: User running the pipeline

        Returns:
            Data pipeline results
        """
        logger.info(f"Running data pipeline for dataset: {dataset_name}")

        # Audit log
        if user:
            self.security_manager.audit_action(
                user=user,
                action='run_data_pipeline',
                resource_type='data',
                resource_id=dataset_name,
                status='started'
            )

        try:
            # Step 1: Ingest data
            logger.info("Step 1: Data ingestion")
            data = self.data_ingestion.ingest_data(dataset_name)
            data_info = self.data_ingestion.get_data_info(data)

            # Step 2: Validate data
            logger.info("Step 2: Data validation")
            validation_result = self.data_validator.validate(data)

            if not validation_result.passed:
                logger.error(f"Data validation failed: {validation_result.errors}")
                raise ValueError("Data validation failed")

            # Step 3: Feature engineering
            logger.info("Step 3: Feature engineering")
            transformed_data = self.feature_engineer.fit_transform(data)

            # Step 4: Version data
            logger.info("Step 4: Data versioning")
            version_id = self.data_versioning.create_version(
                transformed_data,
                {
                    'dataset_name': dataset_name,
                    'validation_passed': True,
                    'num_features': len(transformed_data.columns)
                }
            )

            results = {
                'dataset_name': dataset_name,
                'version_id': version_id,
                'num_rows': len(transformed_data),
                'num_features': len(transformed_data.columns),
                'validation_passed': True,
                'data_info': data_info,
                'transformed_data': transformed_data
            }

            # Audit log success
            if user:
                self.security_manager.audit_action(
                    user=user,
                    action='run_data_pipeline',
                    resource_type='data',
                    resource_id=dataset_name,
                    status='success',
                    details={'version_id': version_id}
                )

            logger.info(f"Data pipeline completed successfully: {version_id}")
            return results

        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")

            if user:
                self.security_manager.audit_action(
                    user=user,
                    action='run_data_pipeline',
                    resource_type='data',
                    resource_id=dataset_name,
                    status='failed',
                    details={'error': str(e)}
                )

            raise

    def train_and_register_model(
        self,
        model: Any,
        model_name: str,
        data_version: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        user: Optional[str] = None
    ) -> str:
        """Train and register a model.

        Args:
            model: Trained model object
            model_name: Name of the model
            data_version: Version of data used for training
            metrics: Training metrics
            parameters: Model parameters
            user: User registering the model

        Returns:
            Model version ID
        """
        logger.info(f"Registering model: {model_name}")

        # Audit log
        if user:
            self.security_manager.audit_action(
                user=user,
                action='register_model',
                resource_type='model',
                resource_id=model_name,
                status='started'
            )

        try:
            # Register model
            metadata = {
                'metrics': metrics,
                'parameters': parameters,
                'dataset_version': data_version,
                'framework': type(model).__module__.split('.')[0]
            }

            version_id = self.model_registry.register_model(
                model=model,
                model_name=model_name,
                metadata=metadata,
                tags=['trained', f'data:{data_version}']
            )

            # Audit log success
            if user:
                self.security_manager.audit_action(
                    user=user,
                    action='register_model',
                    resource_type='model',
                    resource_id=model_name,
                    status='success',
                    details={'version_id': version_id}
                )

            logger.info(f"Model registered successfully: {version_id}")
            return version_id

        except Exception as e:
            logger.error(f"Model registration failed: {e}")

            if user:
                self.security_manager.audit_action(
                    user=user,
                    action='register_model',
                    resource_type='model',
                    resource_id=model_name,
                    status='failed',
                    details={'error': str(e)}
                )

            raise

    def deploy_model(
        self,
        model_name: str,
        model_version: str,
        strategy: str = 'blue_green',
        replicas: int = 3,
        user: Optional[str] = None
    ) -> Dict[str, Any]:
        """Deploy a model to production.

        Args:
            model_name: Name of the model
            model_version: Version to deploy
            strategy: Deployment strategy
            replicas: Number of replicas
            user: User deploying the model

        Returns:
            Deployment information
        """
        logger.info(f"Deploying model {model_name}:{model_version}")

        # Check authorization
        if user and self.security_manager.enable_rbac:
            from security.security_manager import ResourceType, AccessLevel

            authorized = self.security_manager.authorize(
                user=user,
                resource_type=ResourceType.DEPLOYMENT,
                resource_id=f"{model_name}:{model_version}",
                access_level=AccessLevel.WRITE
            )

            if not authorized:
                raise PermissionError(f"User {user} not authorized to deploy models")

        # Audit log
        if user:
            self.security_manager.audit_action(
                user=user,
                action='deploy_model',
                resource_type='deployment',
                resource_id=f"{model_name}:{model_version}",
                status='started'
            )

        try:
            # Create deployment configuration
            deployment_config = DeploymentConfig(
                model_version=f"{model_name}:{model_version}",
                strategy=DeploymentStrategy[strategy.upper()],
                replicas=replicas
            )

            # Deploy
            deployment = self.deployment_manager.deploy_model(deployment_config)

            # Create Kubernetes manifests
            self.kubernetes_manager.export_manifests(
                model_name=model_name,
                model_version=model_version,
                output_dir=f"./k8s/{model_name}/{model_version}"
            )

            # Audit log success
            if user:
                self.security_manager.audit_action(
                    user=user,
                    action='deploy_model',
                    resource_type='deployment',
                    resource_id=f"{model_name}:{model_version}",
                    status='success',
                    details={'deployment_id': deployment.deployment_id}
                )

            logger.info(f"Model deployed successfully: {deployment.deployment_id}")

            return {
                'deployment_id': deployment.deployment_id,
                'model_version': f"{model_name}:{model_version}",
                'strategy': strategy,
                'replicas': replicas,
                'status': deployment.status.value
            }

        except Exception as e:
            logger.error(f"Deployment failed: {e}")

            if user:
                self.security_manager.audit_action(
                    user=user,
                    action='deploy_model',
                    resource_type='deployment',
                    resource_id=f"{model_name}:{model_version}",
                    status='failed',
                    details={'error': str(e)}
                )

            raise

    def monitor_deployment(
        self,
        deployment_id: str,
        check_drift: bool = True
    ) -> Dict[str, Any]:
        """Monitor a deployment.

        Args:
            deployment_id: Deployment to monitor
            check_drift: Whether to check for drift

        Returns:
            Monitoring results
        """
        logger.info(f"Monitoring deployment: {deployment_id}")

        # Get performance summary
        performance = self.performance_monitor.get_performance_summary(
            time_window_minutes=60
        )

        # Check alerts
        alerts = self.performance_monitor.check_alerts()

        monitoring_results = {
            'deployment_id': deployment_id,
            'performance': performance,
            'alerts': [
                {
                    'type': alert['type'],
                    'severity': alert['severity'],
                    'message': alert['message']
                }
                for alert in alerts
            ],
            'timestamp': datetime.now().isoformat()
        }

        # Check drift if requested
        if check_drift:
            # This would need current data to check drift
            # For now, just indicate that drift checking is enabled
            monitoring_results['drift_detection_enabled'] = True

        return monitoring_results

    def run_ab_test(
        self,
        test_config: Dict[str, Any],
        user: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run an A/B test between models.

        Args:
            test_config: A/B test configuration
            user: User running the test

        Returns:
            A/B test results
        """
        logger.info(f"Running A/B test: {test_config.get('test_id', 'unknown')}")

        # Audit log
        if user:
            self.security_manager.audit_action(
                user=user,
                action='run_ab_test',
                resource_type='experiment',
                resource_id=test_config.get('test_id', 'unknown'),
                status='started'
            )

        try:
            # Create A/B test
            ab_test = ABTest(test_config)

            # Start test
            ab_test.start()

            # Note: In production, this would run for a configured duration
            # For now, we return the test configuration
            results = {
                'test_id': ab_test.test_id,
                'status': ab_test.status.value,
                'control_model': ab_test.control_model,
                'treatment_model': ab_test.treatment_model,
                'success_metric': ab_test.success_metric
            }

            # Audit log success
            if user:
                self.security_manager.audit_action(
                    user=user,
                    action='run_ab_test',
                    resource_type='experiment',
                    resource_id=test_config.get('test_id', 'unknown'),
                    status='success'
                )

            logger.info(f"A/B test started successfully")
            return results

        except Exception as e:
            logger.error(f"A/B test failed: {e}")

            if user:
                self.security_manager.audit_action(
                    user=user,
                    action='run_ab_test',
                    resource_type='experiment',
                    resource_id=test_config.get('test_id', 'unknown'),
                    status='failed',
                    details={'error': str(e)}
                )

            raise

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status.

        Returns:
            Pipeline status
        """
        # Get counts from various components
        models = self.model_registry.list_models()
        deployments = self.deployment_manager.list_deployments()
        performance = self.performance_monitor.get_performance_summary()

        status = {
            'timestamp': datetime.now().isoformat(),
            'registered_models': len(models),
            'active_deployments': len([d for d in deployments
                                      if d.status.value == 'completed']),
            'total_predictions': performance.get('total_predictions', 0),
            'avg_latency_ms': performance.get('avg_latency_ms', 0),
            'error_rate': performance.get('error_rate', 0)
        }

        return status

    def get_compliance_report(self, user: str) -> Dict[str, Any]:
        """Generate compliance report.

        Args:
            user: User requesting the report

        Returns:
            Compliance report
        """
        logger.info(f"Generating compliance report for user: {user}")

        # Get security compliance
        compliance = self.security_manager.get_compliance_report()

        # Add pipeline-specific information
        compliance['pipeline_status'] = self.get_pipeline_status()

        return compliance
