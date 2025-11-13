"""Model registry for versioning and managing ML models."""

import logging
import json
import pickle
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import shutil


logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model versioning, storage, and metadata."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize model registry.

        Args:
            config: Configuration dictionary containing:
                - registry_dir: Directory for storing models
                - storage_backend: Storage backend (local, s3, mlflow)
                - enable_staging: Enable staging/production stages
        """
        self.config = config
        self.registry_dir = Path(config.get('registry_dir', './model_registry'))
        self.storage_backend = config.get('storage_backend', 'local')
        self.enable_staging = config.get('enable_staging', True)

        # Create registry directory
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.metadata_file = self.registry_dir / 'models.json'
        self.models = self._load_models()

        logger.info("Initialized ModelRegistry")

    def register_model(
        self,
        model: Any,
        model_name: str,
        metadata: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """Register a new model.

        Args:
            model: Model object to register
            model_name: Name of the model
            metadata: Metadata about the model
            tags: Optional tags for the model

        Returns:
            Version ID
        """
        # Generate version
        version = self._generate_version(model_name)
        version_id = f"{model_name}:{version}"

        logger.info(f"Registering model: {version_id}")

        # Create model directory
        model_dir = self.registry_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Compute model hash
        model_hash = self._compute_hash(model_path)

        # Store metadata
        model_metadata = {
            'model_name': model_name,
            'version': version,
            'version_id': version_id,
            'model_hash': model_hash,
            'model_path': str(model_path),
            'model_dir': str(model_dir),
            'registered_at': datetime.now().isoformat(),
            'stage': 'none' if self.enable_staging else 'production',
            'tags': tags or [],
            'metrics': metadata.get('metrics', {}),
            'parameters': metadata.get('parameters', {}),
            'framework': metadata.get('framework', 'unknown'),
            'description': metadata.get('description', ''),
            'dataset_version': metadata.get('dataset_version', 'unknown')
        }

        # Initialize model entry if doesn't exist
        if model_name not in self.models:
            self.models[model_name] = {
                'name': model_name,
                'created_at': datetime.now().isoformat(),
                'versions': {}
            }

        # Store version metadata
        self.models[model_name]['versions'][version] = model_metadata
        self._save_models()

        logger.info(f"Registered model {version_id} with hash {model_hash[:8]}")
        return version_id

    def get_model(self, model_name: str, version: Optional[str] = None,
                  stage: Optional[str] = None) -> tuple:
        """Retrieve a model.

        Args:
            model_name: Name of the model
            version: Specific version (optional)
            stage: Stage to retrieve (staging/production) (optional)

        Returns:
            Tuple of (model, metadata)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")

        # Get the appropriate version
        if version:
            model_metadata = self.models[model_name]['versions'].get(version)
            if not model_metadata:
                raise ValueError(f"Version {version} not found for {model_name}")
        elif stage:
            model_metadata = self._get_model_by_stage(model_name, stage)
        else:
            # Get latest version
            model_metadata = self._get_latest_version(model_name)

        # Load model
        model_path = Path(model_metadata['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Loaded model {model_metadata['version_id']}")
        return model, model_metadata

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models.

        Returns:
            List of model information
        """
        models_list = []

        for model_name, model_data in self.models.items():
            latest_version = self._get_latest_version(model_name)

            models_list.append({
                'name': model_name,
                'created_at': model_data['created_at'],
                'num_versions': len(model_data['versions']),
                'latest_version': latest_version['version'],
                'latest_registered_at': latest_version['registered_at'],
                'tags': latest_version.get('tags', [])
            })

        return models_list

    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model.

        Args:
            model_name: Name of the model

        Returns:
            List of version information
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")

        versions = []
        for version, metadata in self.models[model_name]['versions'].items():
            versions.append({
                'version': version,
                'version_id': metadata['version_id'],
                'registered_at': metadata['registered_at'],
                'stage': metadata.get('stage', 'none'),
                'metrics': metadata.get('metrics', {}),
                'tags': metadata.get('tags', [])
            })

        # Sort by version number (descending)
        versions.sort(key=lambda x: int(x['version'].replace('v', '')), reverse=True)
        return versions

    def transition_model_stage(self, model_name: str, version: str, stage: str):
        """Transition a model to a different stage.

        Args:
            model_name: Name of the model
            version: Version to transition
            stage: Target stage (staging, production, archived)
        """
        if not self.enable_staging:
            raise ValueError("Model staging is not enabled")

        valid_stages = ['none', 'staging', 'production', 'archived']
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        if version not in self.models[model_name]['versions']:
            raise ValueError(f"Version {version} not found")

        # If transitioning to production, move current production to staging
        if stage == 'production':
            for v, meta in self.models[model_name]['versions'].items():
                if meta['stage'] == 'production':
                    meta['stage'] = 'staging'
                    logger.info(f"Moved version {v} to staging")

        # Update stage
        self.models[model_name]['versions'][version]['stage'] = stage
        self.models[model_name]['versions'][version]['stage_updated_at'] = \
            datetime.now().isoformat()

        self._save_models()
        logger.info(f"Transitioned {model_name}:{version} to {stage}")

    def delete_model_version(self, model_name: str, version: str):
        """Delete a specific version of a model.

        Args:
            model_name: Name of the model
            version: Version to delete
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        if version not in self.models[model_name]['versions']:
            raise ValueError(f"Version {version} not found")

        # Get metadata
        metadata = self.models[model_name]['versions'][version]

        # Don't delete production models
        if metadata.get('stage') == 'production':
            raise ValueError("Cannot delete production model. Transition to another stage first.")

        # Delete model files
        model_dir = Path(metadata['model_dir'])
        if model_dir.exists():
            shutil.rmtree(model_dir)

        # Remove from metadata
        del self.models[model_name]['versions'][version]

        # If no versions left, delete model entry
        if not self.models[model_name]['versions']:
            del self.models[model_name]

        self._save_models()
        logger.info(f"Deleted {model_name}:{version}")

    def search_models(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for models based on criteria.

        Args:
            query: Search criteria (tags, metrics, etc.)

        Returns:
            List of matching models
        """
        results = []

        for model_name, model_data in self.models.items():
            for version, metadata in model_data['versions'].items():
                match = True

                # Check tags
                if 'tags' in query:
                    required_tags = set(query['tags'])
                    model_tags = set(metadata.get('tags', []))
                    if not required_tags.issubset(model_tags):
                        match = False

                # Check metrics
                if 'min_metrics' in query:
                    for metric, min_value in query['min_metrics'].items():
                        if metric not in metadata.get('metrics', {}):
                            match = False
                        elif metadata['metrics'][metric] < min_value:
                            match = False

                # Check stage
                if 'stage' in query:
                    if metadata.get('stage') != query['stage']:
                        match = False

                if match:
                    results.append(metadata)

        return results

    def compare_models(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two model versions.

        Args:
            version_id1: First version ID (format: model_name:version)
            version_id2: Second version ID

        Returns:
            Comparison results
        """
        model_name1, version1 = version_id1.split(':')
        model_name2, version2 = version_id2.split(':')

        if model_name1 != model_name2:
            raise ValueError("Can only compare versions of the same model")

        meta1 = self.models[model_name1]['versions'][version1]
        meta2 = self.models[model_name2]['versions'][version2]

        comparison = {
            'model_name': model_name1,
            'version_1': version1,
            'version_2': version2,
            'metrics_comparison': {},
            'parameters_comparison': {},
        }

        # Compare metrics
        for metric in set(meta1.get('metrics', {}).keys()) | \
                      set(meta2.get('metrics', {}).keys()):
            val1 = meta1.get('metrics', {}).get(metric, 0)
            val2 = meta2.get('metrics', {}).get(metric, 0)
            comparison['metrics_comparison'][metric] = {
                'version_1': val1,
                'version_2': val2,
                'diff': val2 - val1,
                'pct_change': ((val2 - val1) / val1 * 100) if val1 != 0 else float('inf')
            }

        # Compare parameters
        params1 = set(meta1.get('parameters', {}).keys())
        params2 = set(meta2.get('parameters', {}).keys())

        comparison['parameters_comparison'] = {
            'common': list(params1 & params2),
            'only_in_v1': list(params1 - params2),
            'only_in_v2': list(params2 - params1),
        }

        return comparison

    def _generate_version(self, model_name: str) -> str:
        """Generate next version number for a model."""
        if model_name not in self.models:
            return 'v1'

        versions = self.models[model_name]['versions'].keys()
        if not versions:
            return 'v1'

        # Extract version numbers
        version_nums = [int(v.replace('v', '')) for v in versions]
        next_version = max(version_nums) + 1

        return f'v{next_version}'

    def _get_latest_version(self, model_name: str) -> Dict[str, Any]:
        """Get the latest version of a model."""
        versions = self.models[model_name]['versions']

        if not versions:
            raise ValueError(f"No versions found for {model_name}")

        # Get version with highest number
        latest_version = max(versions.keys(),
                           key=lambda v: int(v.replace('v', '')))

        return versions[latest_version]

    def _get_model_by_stage(self, model_name: str, stage: str) -> Dict[str, Any]:
        """Get model by stage."""
        for version, metadata in self.models[model_name]['versions'].items():
            if metadata.get('stage') == stage:
                return metadata

        raise ValueError(f"No model in {stage} stage for {model_name}")

    def _compute_hash(self, file_path: Path) -> str:
        """Compute hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _load_models(self) -> Dict[str, Dict[str, Any]]:
        """Load model metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_models(self):
        """Save model metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.models, f, indent=2)
