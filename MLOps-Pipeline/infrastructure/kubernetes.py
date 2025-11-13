"""Kubernetes management module for ML model deployment."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import yaml
import json


logger = logging.getLogger(__name__)


@dataclass
class KubernetesDeployment:
    """Represents a Kubernetes deployment."""
    name: str
    namespace: str
    replicas: int
    image: str
    labels: Dict[str, str]
    resources: Dict[str, Any]
    created_at: str


class KubernetesManager:
    """Manages Kubernetes deployments for ML models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Kubernetes manager.

        Args:
            config: Configuration dictionary containing:
                - namespace: Kubernetes namespace
                - registry_url: Container registry URL
                - service_account: Service account name
                - resource_limits: Default resource limits
        """
        self.config = config
        self.namespace = config.get('namespace', 'default')
        self.registry_url = config.get('registry_url', '')
        self.service_account = config.get('service_account', 'default')
        self.resource_limits = config.get('resource_limits', {})

        logger.info(f"Initialized KubernetesManager for namespace: {self.namespace}")

    def create_deployment(
        self,
        model_name: str,
        model_version: str,
        replicas: int = 3,
        resources: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a Kubernetes deployment for a model.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            replicas: Number of replicas
            resources: Resource requests and limits

        Returns:
            Deployment manifest
        """
        logger.info(f"Creating deployment for {model_name}:{model_version}")

        # Generate deployment name
        deployment_name = f"{model_name}-{model_version}".replace('_', '-').lower()

        # Prepare resource configuration
        if resources is None:
            resources = self.resource_limits

        # Create deployment manifest
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': deployment_name,
                'namespace': self.namespace,
                'labels': {
                    'app': model_name,
                    'version': model_version,
                    'component': 'model-server'
                }
            },
            'spec': {
                'replicas': replicas,
                'selector': {
                    'matchLabels': {
                        'app': model_name,
                        'version': model_version
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': model_name,
                            'version': model_version,
                            'component': 'model-server'
                        }
                    },
                    'spec': {
                        'serviceAccountName': self.service_account,
                        'containers': [{
                            'name': 'model-server',
                            'image': f"{self.registry_url}/{model_name}:{model_version}",
                            'ports': [{
                                'name': 'http',
                                'containerPort': 8000,
                                'protocol': 'TCP'
                            }],
                            'env': [
                                {
                                    'name': 'MODEL_NAME',
                                    'value': model_name
                                },
                                {
                                    'name': 'MODEL_VERSION',
                                    'value': model_version
                                }
                            ],
                            'resources': resources,
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 10,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }

        logger.info(f"Created deployment manifest for {deployment_name}")
        return deployment

    def create_service(
        self,
        model_name: str,
        model_version: str,
        service_type: str = 'ClusterIP'
    ) -> Dict[str, Any]:
        """Create a Kubernetes service for a model.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            service_type: Type of service (ClusterIP, NodePort, LoadBalancer)

        Returns:
            Service manifest
        """
        service_name = f"{model_name}-{model_version}".replace('_', '-').lower()

        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': service_name,
                'namespace': self.namespace,
                'labels': {
                    'app': model_name,
                    'version': model_version
                }
            },
            'spec': {
                'type': service_type,
                'selector': {
                    'app': model_name,
                    'version': model_version
                },
                'ports': [{
                    'name': 'http',
                    'protocol': 'TCP',
                    'port': 80,
                    'targetPort': 8000
                }]
            }
        }

        logger.info(f"Created service manifest for {service_name}")
        return service

    def create_hpa(
        self,
        model_name: str,
        model_version: str,
        min_replicas: int = 2,
        max_replicas: int = 10,
        target_cpu_utilization: int = 70
    ) -> Dict[str, Any]:
        """Create Horizontal Pod Autoscaler.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas
            target_cpu_utilization: Target CPU utilization percentage

        Returns:
            HPA manifest
        """
        deployment_name = f"{model_name}-{model_version}".replace('_', '-').lower()

        hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{deployment_name}-hpa",
                'namespace': self.namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': deployment_name
                },
                'minReplicas': min_replicas,
                'maxReplicas': max_replicas,
                'metrics': [{
                    'type': 'Resource',
                    'resource': {
                        'name': 'cpu',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': target_cpu_utilization
                        }
                    }
                }]
            }
        }

        logger.info(f"Created HPA manifest for {deployment_name}")
        return hpa

    def create_ingress(
        self,
        model_name: str,
        hostname: str,
        path: str = '/'
    ) -> Dict[str, Any]:
        """Create Ingress for external access.

        Args:
            model_name: Name of the model
            hostname: Hostname for ingress
            path: Path prefix

        Returns:
            Ingress manifest
        """
        ingress_name = f"{model_name}-ingress"

        ingress = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': ingress_name,
                'namespace': self.namespace,
                'annotations': {
                    'kubernetes.io/ingress.class': 'nginx',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                }
            },
            'spec': {
                'rules': [{
                    'host': hostname,
                    'http': {
                        'paths': [{
                            'path': path,
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': model_name,
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }],
                'tls': [{
                    'hosts': [hostname],
                    'secretName': f"{model_name}-tls"
                }]
            }
        }

        logger.info(f"Created ingress manifest for {model_name}")
        return ingress

    def create_configmap(
        self,
        name: str,
        data: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create ConfigMap for configuration.

        Args:
            name: ConfigMap name
            data: Configuration data

        Returns:
            ConfigMap manifest
        """
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': name,
                'namespace': self.namespace
            },
            'data': data
        }

        return configmap

    def create_secret(
        self,
        name: str,
        data: Dict[str, str],
        secret_type: str = 'Opaque'
    ) -> Dict[str, Any]:
        """Create Secret for sensitive data.

        Args:
            name: Secret name
            data: Secret data (will be base64 encoded)
            secret_type: Type of secret

        Returns:
            Secret manifest
        """
        import base64

        # Base64 encode data
        encoded_data = {
            key: base64.b64encode(value.encode()).decode()
            for key, value in data.items()
        }

        secret = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': name,
                'namespace': self.namespace
            },
            'type': secret_type,
            'data': encoded_data
        }

        return secret

    def create_service_monitor(
        self,
        model_name: str,
        model_version: str
    ) -> Dict[str, Any]:
        """Create ServiceMonitor for Prometheus.

        Args:
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            ServiceMonitor manifest
        """
        service_name = f"{model_name}-{model_version}".replace('_', '-').lower()

        service_monitor = {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': f"{service_name}-monitor",
                'namespace': self.namespace,
                'labels': {
                    'app': model_name,
                    'version': model_version
                }
            },
            'spec': {
                'selector': {
                    'matchLabels': {
                        'app': model_name,
                        'version': model_version
                    }
                },
                'endpoints': [{
                    'port': 'http',
                    'path': '/metrics',
                    'interval': '30s'
                }]
            }
        }

        return service_monitor

    def export_manifests(
        self,
        model_name: str,
        model_version: str,
        output_dir: str
    ):
        """Export all Kubernetes manifests to files.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            output_dir: Directory to save manifests
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Create all manifests
        deployment = self.create_deployment(model_name, model_version)
        service = self.create_service(model_name, model_version)
        hpa = self.create_hpa(model_name, model_version)

        # Save to files
        with open(f"{output_dir}/deployment.yaml", 'w') as f:
            yaml.dump(deployment, f, default_flow_style=False)

        with open(f"{output_dir}/service.yaml", 'w') as f:
            yaml.dump(service, f, default_flow_style=False)

        with open(f"{output_dir}/hpa.yaml", 'w') as f:
            yaml.dump(hpa, f, default_flow_style=False)

        logger.info(f"Exported manifests to {output_dir}")

    def apply_manifest(self, manifest: Dict[str, Any]) -> bool:
        """Apply a Kubernetes manifest.

        Args:
            manifest: Kubernetes manifest

        Returns:
            True if successful
        """
        # Placeholder - would use kubernetes client in production
        logger.info(f"Applying manifest: {manifest['kind']} "
                   f"{manifest['metadata']['name']}")
        return True

    def delete_deployment(self, deployment_name: str) -> bool:
        """Delete a Kubernetes deployment.

        Args:
            deployment_name: Name of deployment to delete

        Returns:
            True if successful
        """
        logger.info(f"Deleting deployment: {deployment_name}")
        # Placeholder - would use kubernetes client
        return True

    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get status of a deployment.

        Args:
            deployment_name: Name of deployment

        Returns:
            Deployment status
        """
        # Placeholder - would query Kubernetes API
        return {
            'name': deployment_name,
            'replicas': 3,
            'available_replicas': 3,
            'status': 'Running'
        }
