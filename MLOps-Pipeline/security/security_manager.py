"""Security and compliance management for MLOps pipeline."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import secrets
import json


logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access levels for RBAC."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class ResourceType(Enum):
    """Resource types."""
    MODEL = "model"
    DATA = "data"
    DEPLOYMENT = "deployment"
    EXPERIMENT = "experiment"


@dataclass
class AuditLog:
    """Audit log entry."""
    timestamp: str
    user: str
    action: str
    resource_type: str
    resource_id: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPolicy:
    """Access control policy."""
    policy_id: str
    user: str
    resource_type: ResourceType
    resource_id: str
    access_level: AccessLevel
    created_at: str
    expires_at: Optional[str] = None


class SecurityManager:
    """Manages security and compliance for MLOps pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize security manager.

        Args:
            config: Configuration dictionary containing:
                - enable_rbac: Enable role-based access control
                - enable_audit: Enable audit logging
                - enable_encryption: Enable data encryption
                - secret_key: Secret key for encryption
        """
        self.config = config
        self.enable_rbac = config.get('enable_rbac', True)
        self.enable_audit = config.get('enable_audit', True)
        self.enable_encryption = config.get('enable_encryption', True)
        self.secret_key = config.get('secret_key', self._generate_secret_key())

        # Storage
        self.access_policies: Dict[str, AccessPolicy] = {}
        self.audit_logs: List[AuditLog] = []
        self.users: Dict[str, Dict[str, Any]] = {}

        logger.info("Initialized SecurityManager")

    def create_user(self, username: str, role: str,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new user.

        Args:
            username: Username
            role: User role
            metadata: Additional metadata

        Returns:
            User information
        """
        if username in self.users:
            raise ValueError(f"User {username} already exists")

        api_key = self._generate_api_key()

        user = {
            'username': username,
            'role': role,
            'api_key': api_key,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self.users[username] = user

        logger.info(f"Created user: {username} with role {role}")
        return user

    def authenticate(self, api_key: str) -> Optional[str]:
        """Authenticate user by API key.

        Args:
            api_key: API key to authenticate

        Returns:
            Username if authenticated, None otherwise
        """
        for username, user in self.users.items():
            if user['api_key'] == api_key:
                return username

        return None

    def authorize(
        self,
        user: str,
        resource_type: ResourceType,
        resource_id: str,
        access_level: AccessLevel
    ) -> bool:
        """Check if user has access to resource.

        Args:
            user: Username
            resource_type: Type of resource
            resource_id: Resource identifier
            access_level: Required access level

        Returns:
            True if authorized
        """
        if not self.enable_rbac:
            return True

        # Check for specific policy
        policy_key = f"{user}:{resource_type.value}:{resource_id}"

        if policy_key in self.access_policies:
            policy = self.access_policies[policy_key]

            # Check if policy expired
            if policy.expires_at:
                expires = datetime.fromisoformat(policy.expires_at)
                if datetime.now() > expires:
                    return False

            # Check access level
            if policy.access_level == AccessLevel.ADMIN:
                return True
            elif policy.access_level == AccessLevel.WRITE:
                return access_level in [AccessLevel.READ, AccessLevel.WRITE]
            elif policy.access_level == AccessLevel.READ:
                return access_level == AccessLevel.READ

        # Check user role
        if user in self.users:
            user_role = self.users[user]['role']
            if user_role == 'admin':
                return True

        return False

    def grant_access(
        self,
        user: str,
        resource_type: ResourceType,
        resource_id: str,
        access_level: AccessLevel,
        expires_at: Optional[str] = None
    ) -> AccessPolicy:
        """Grant access to a resource.

        Args:
            user: Username
            resource_type: Type of resource
            resource_id: Resource identifier
            access_level: Access level to grant
            expires_at: Optional expiration time

        Returns:
            Access policy
        """
        policy_id = f"{user}:{resource_type.value}:{resource_id}"

        policy = AccessPolicy(
            policy_id=policy_id,
            user=user,
            resource_type=resource_type,
            resource_id=resource_id,
            access_level=access_level,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at
        )

        self.access_policies[policy_id] = policy

        logger.info(f"Granted {access_level.value} access to {user} "
                   f"for {resource_type.value}:{resource_id}")

        return policy

    def revoke_access(
        self,
        user: str,
        resource_type: ResourceType,
        resource_id: str
    ):
        """Revoke access to a resource.

        Args:
            user: Username
            resource_type: Type of resource
            resource_id: Resource identifier
        """
        policy_id = f"{user}:{resource_type.value}:{resource_id}"

        if policy_id in self.access_policies:
            del self.access_policies[policy_id]
            logger.info(f"Revoked access for {user} "
                       f"to {resource_type.value}:{resource_id}")

    def audit_action(
        self,
        user: str,
        action: str,
        resource_type: str,
        resource_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log an action for audit.

        Args:
            user: User performing action
            action: Action performed
            resource_type: Type of resource
            resource_id: Resource identifier
            status: Status of action
            details: Additional details
        """
        if not self.enable_audit:
            return

        log = AuditLog(
            timestamp=datetime.now().isoformat(),
            user=user,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            status=status,
            details=details or {}
        )

        self.audit_logs.append(log)

        logger.debug(f"Audit log: {user} {action} {resource_type}:{resource_id} - {status}")

    def get_audit_logs(
        self,
        user: Optional[str] = None,
        resource_type: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs with optional filters.

        Args:
            user: Filter by user
            resource_type: Filter by resource type
            action: Filter by action
            limit: Maximum number of logs to return

        Returns:
            List of audit logs
        """
        logs = self.audit_logs

        # Apply filters
        if user:
            logs = [log for log in logs if log.user == user]

        if resource_type:
            logs = [log for log in logs if log.resource_type == resource_type]

        if action:
            logs = [log for log in logs if log.action == action]

        # Return most recent logs
        return logs[-limit:]

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data (hex encoded)
        """
        if not self.enable_encryption:
            return data

        # Simple XOR encryption for demonstration
        # In production, use proper encryption (e.g., Fernet, AES)
        key_bytes = self.secret_key.encode()
        data_bytes = data.encode()

        encrypted = bytearray()
        for i, byte in enumerate(data_bytes):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])

        return encrypted.hex()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt encrypted data.

        Args:
            encrypted_data: Encrypted data (hex encoded)

        Returns:
            Decrypted data
        """
        if not self.enable_encryption:
            return encrypted_data

        # Decrypt using XOR
        key_bytes = self.secret_key.encode()
        encrypted_bytes = bytes.fromhex(encrypted_data)

        decrypted = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            decrypted.append(byte ^ key_bytes[i % len(key_bytes)])

        return decrypted.decode()

    def hash_data(self, data: str) -> str:
        """Hash data for integrity checking.

        Args:
            data: Data to hash

        Returns:
            SHA256 hash (hex)
        """
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_hash(self, data: str, hash_value: str) -> bool:
        """Verify data integrity using hash.

        Args:
            data: Original data
            hash_value: Expected hash

        Returns:
            True if hash matches
        """
        computed_hash = self.hash_data(data)
        return computed_hash == hash_value

    def scan_for_pii(self, data: Dict[str, Any]) -> List[str]:
        """Scan data for potential PII.

        Args:
            data: Data to scan

        Returns:
            List of fields that may contain PII
        """
        pii_patterns = [
            'email', 'phone', 'ssn', 'credit_card',
            'name', 'address', 'passport', 'license'
        ]

        pii_fields = []

        for key in data.keys():
            key_lower = key.lower()
            for pattern in pii_patterns:
                if pattern in key_lower:
                    pii_fields.append(key)
                    break

        if pii_fields:
            logger.warning(f"Potential PII found in fields: {pii_fields}")

        return pii_fields

    def anonymize_data(self, data: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """Anonymize specified fields.

        Args:
            data: Data to anonymize
            fields: Fields to anonymize

        Returns:
            Anonymized data
        """
        anonymized = data.copy()

        for field in fields:
            if field in anonymized:
                # Hash the value for anonymization
                original = str(anonymized[field])
                anonymized[field] = self.hash_data(original)[:16]

        return anonymized

    def export_audit_logs(self, filepath: str):
        """Export audit logs to file.

        Args:
            filepath: Path to export file
        """
        logs_data = [
            {
                'timestamp': log.timestamp,
                'user': log.user,
                'action': log.action,
                'resource_type': log.resource_type,
                'resource_id': log.resource_id,
                'status': log.status,
                'details': log.details
            }
            for log in self.audit_logs
        ]

        with open(filepath, 'w') as f:
            json.dump(logs_data, f, indent=2)

        logger.info(f"Exported {len(logs_data)} audit logs to {filepath}")

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)

    def _generate_secret_key(self) -> str:
        """Generate a secret key for encryption."""
        return secrets.token_urlsafe(32)

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report.

        Returns:
            Compliance report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'rbac_enabled': self.enable_rbac,
            'audit_enabled': self.enable_audit,
            'encryption_enabled': self.enable_encryption,
            'total_users': len(self.users),
            'total_policies': len(self.access_policies),
            'total_audit_logs': len(self.audit_logs),
            'recent_actions': [
                {
                    'user': log.user,
                    'action': log.action,
                    'timestamp': log.timestamp
                }
                for log in self.audit_logs[-10:]
            ]
        }

        return report
