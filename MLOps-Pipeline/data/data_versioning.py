"""Data versioning module for tracking data changes."""

import logging
import hashlib
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import pandas as pd


logger = logging.getLogger(__name__)


class DataVersioning:
    """Handles data versioning and lineage tracking."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data versioning.

        Args:
            config: Configuration dictionary containing:
                - version_dir: Directory for storing version metadata
                - storage_backend: Storage backend (local, s3, gcs)
        """
        self.config = config
        self.version_dir = Path(config.get('version_dir', './data_versions'))
        self.storage_backend = config.get('storage_backend', 'local')

        # Create version directory if it doesn't exist
        self.version_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.metadata_file = self.version_dir / 'versions.json'
        self.versions = self._load_versions()

        logger.info("Initialized DataVersioning")

    def create_version(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> str:
        """Create a new version of the data.

        Args:
            data: DataFrame to version
            metadata: Metadata about the data version

        Returns:
            Version ID
        """
        # Generate version ID based on data hash
        data_hash = self._compute_hash(data)
        version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{data_hash[:8]}"

        logger.info(f"Creating data version: {version_id}")

        # Store data
        data_path = self.version_dir / f"{version_id}.parquet"
        data.to_parquet(data_path, index=False)

        # Store metadata
        version_metadata = {
            'version_id': version_id,
            'data_hash': data_hash,
            'num_rows': len(data),
            'num_columns': len(data.columns),
            'columns': list(data.columns),
            'created_at': datetime.now().isoformat(),
            'data_path': str(data_path),
            **metadata
        }

        self.versions[version_id] = version_metadata
        self._save_versions()

        logger.info(f"Created version {version_id} with {len(data)} rows")
        return version_id

    def get_version(self, version_id: str) -> pd.DataFrame:
        """Retrieve a specific version of the data.

        Args:
            version_id: Version identifier

        Returns:
            DataFrame for that version
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        metadata = self.versions[version_id]
        data_path = Path(metadata['data_path'])

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info(f"Loading version {version_id}")
        return pd.read_parquet(data_path)

    def get_version_metadata(self, version_id: str) -> Dict[str, Any]:
        """Get metadata for a specific version.

        Args:
            version_id: Version identifier

        Returns:
            Version metadata
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        return self.versions[version_id]

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all data versions.

        Returns:
            List of version metadata
        """
        return list(self.versions.values())

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two data versions.

        Args:
            version_id1: First version ID
            version_id2: Second version ID

        Returns:
            Comparison results
        """
        logger.info(f"Comparing versions {version_id1} and {version_id2}")

        data1 = self.get_version(version_id1)
        data2 = self.get_version(version_id2)
        meta1 = self.versions[version_id1]
        meta2 = self.versions[version_id2]

        comparison = {
            'version_1': version_id1,
            'version_2': version_id2,
            'row_count_diff': len(data2) - len(data1),
            'column_count_diff': len(data2.columns) - len(data1.columns),
            'added_columns': list(set(data2.columns) - set(data1.columns)),
            'removed_columns': list(set(data1.columns) - set(data2.columns)),
            'common_columns': list(set(data1.columns) & set(data2.columns)),
            'data_hash_1': meta1['data_hash'],
            'data_hash_2': meta2['data_hash'],
            'identical': meta1['data_hash'] == meta2['data_hash']
        }

        # Compare statistics for common columns
        common_numeric = [col for col in comparison['common_columns']
                         if col in data1.select_dtypes(include=['number']).columns]

        stats_comparison = {}
        for col in common_numeric:
            stats_comparison[col] = {
                'mean_diff': float(data2[col].mean() - data1[col].mean()),
                'std_diff': float(data2[col].std() - data1[col].std()),
                'min_diff': float(data2[col].min() - data1[col].min()),
                'max_diff': float(data2[col].max() - data1[col].max())
            }

        comparison['statistics_comparison'] = stats_comparison

        return comparison

    def get_lineage(self, version_id: str) -> Dict[str, Any]:
        """Get data lineage for a version.

        Args:
            version_id: Version identifier

        Returns:
            Lineage information
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        metadata = self.versions[version_id]

        lineage = {
            'version_id': version_id,
            'created_at': metadata['created_at'],
            'source': metadata.get('source', 'unknown'),
            'transformations': metadata.get('transformations', []),
            'parent_version': metadata.get('parent_version'),
        }

        # If there's a parent version, recursively get its lineage
        if lineage['parent_version']:
            parent_lineage = self.get_lineage(lineage['parent_version'])
            lineage['parent_lineage'] = parent_lineage

        return lineage

    def _compute_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of the data."""
        # Convert dataframe to string representation and hash it
        data_str = pd.util.hash_pandas_object(data).values.tobytes()
        return hashlib.sha256(data_str).hexdigest()

    def _load_versions(self) -> Dict[str, Dict[str, Any]]:
        """Load version metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_versions(self):
        """Save version metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

    def delete_version(self, version_id: str):
        """Delete a data version.

        Args:
            version_id: Version to delete
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        # Delete data file
        metadata = self.versions[version_id]
        data_path = Path(metadata['data_path'])
        if data_path.exists():
            data_path.unlink()

        # Remove from metadata
        del self.versions[version_id]
        self._save_versions()

        logger.info(f"Deleted version {version_id}")

    def cleanup_old_versions(self, keep_latest: int = 10):
        """Clean up old versions, keeping only the latest N versions.

        Args:
            keep_latest: Number of latest versions to keep
        """
        if len(self.versions) <= keep_latest:
            return

        # Sort versions by creation time
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1]['created_at'],
            reverse=True
        )

        # Delete old versions
        for version_id, _ in sorted_versions[keep_latest:]:
            try:
                self.delete_version(version_id)
            except Exception as e:
                logger.error(f"Error deleting version {version_id}: {e}")

        logger.info(f"Cleaned up old versions, kept latest {keep_latest}")
