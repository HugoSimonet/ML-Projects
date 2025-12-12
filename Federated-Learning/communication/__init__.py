"""Communication layer for federated learning"""

from .secure_channels import (
    SecureChannel,
    GRPCChannel,
    HTTPChannel,
    MessageSerializer,
    CompressionHandler
)

__all__ = [
    'SecureChannel',
    'GRPCChannel',
    'HTTPChannel',
    'MessageSerializer',
    'CompressionHandler'
]
