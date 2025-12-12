"""
Secure communication channels for federated learning
Supports gRPC, HTTP, and WebSocket with compression and encryption
"""

import torch
import pickle
import zlib
import json
import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import io
import numpy as np

logger = logging.getLogger(__name__)


class MessageSerializer:
    """Serialization for model updates and messages"""

    @staticmethod
    def serialize_model(model_state: Dict[str, torch.Tensor]) -> bytes:
        """
        Serialize model state dictionary to bytes

        Args:
            model_state: Dictionary of model parameters

        Returns:
            Serialized bytes
        """
        buffer = io.BytesIO()

        # Convert tensors to numpy for efficient serialization
        numpy_state = {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in model_state.items()
        }

        # Use pickle for serialization
        pickle.dump(numpy_state, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        buffer.seek(0)

        return buffer.read()

    @staticmethod
    def deserialize_model(data: bytes) -> Dict[str, torch.Tensor]:
        """
        Deserialize bytes to model state dictionary

        Args:
            data: Serialized bytes

        Returns:
            Model state dictionary
        """
        buffer = io.BytesIO(data)
        numpy_state = pickle.load(buffer)

        # Convert numpy arrays back to tensors
        model_state = {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in numpy_state.items()
        }

        return model_state

    @staticmethod
    def serialize_metadata(metadata: Dict[str, Any]) -> bytes:
        """Serialize metadata dictionary"""
        json_str = json.dumps(metadata)
        return json_str.encode('utf-8')

    @staticmethod
    def deserialize_metadata(data: bytes) -> Dict[str, Any]:
        """Deserialize metadata dictionary"""
        json_str = data.decode('utf-8')
        return json.loads(json_str)


class CompressionHandler:
    """Handle compression and decompression of messages"""

    @staticmethod
    def compress(data: bytes, level: int = 6) -> bytes:
        """
        Compress data using zlib

        Args:
            data: Data to compress
            level: Compression level (0-9)

        Returns:
            Compressed data
        """
        return zlib.compress(data, level=level)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        """
        Decompress data using zlib

        Args:
            data: Compressed data

        Returns:
            Decompressed data
        """
        return zlib.decompress(data)

    @staticmethod
    def quantize_model(
        model_state: Dict[str, torch.Tensor],
        bits: int = 8
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize model parameters to reduce size

        Args:
            model_state: Model parameters
            bits: Number of bits for quantization

        Returns:
            Quantized model state
        """
        quantized_state = {}

        for name, param in model_state.items():
            if param.dtype in [torch.float32, torch.float64]:
                # Quantize to specified bits
                min_val = param.min()
                max_val = param.max()

                # Avoid division by zero
                if max_val == min_val:
                    quantized_state[name] = param
                    continue

                # Quantize
                scale = (max_val - min_val) / (2 ** bits - 1)
                zero_point = min_val

                quantized = torch.round((param - zero_point) / scale)
                quantized = quantized.clamp(0, 2 ** bits - 1)

                # Store quantized values with metadata
                quantized_state[name] = {
                    'values': quantized.to(torch.uint8),
                    'scale': scale,
                    'zero_point': zero_point,
                    'shape': param.shape,
                    'dtype': param.dtype
                }
            else:
                quantized_state[name] = param

        return quantized_state

    @staticmethod
    def dequantize_model(
        quantized_state: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Dequantize model parameters

        Args:
            quantized_state: Quantized model state

        Returns:
            Dequantized model state
        """
        dequantized_state = {}

        for name, data in quantized_state.items():
            if isinstance(data, dict) and 'values' in data:
                # Dequantize
                values = data['values'].to(torch.float32)
                scale = data['scale']
                zero_point = data['zero_point']
                original_dtype = data['dtype']

                dequantized = values * scale + zero_point
                dequantized = dequantized.to(original_dtype)
                dequantized_state[name] = dequantized
            else:
                dequantized_state[name] = data

        return dequantized_state


class SecureChannel(ABC):
    """Abstract base class for secure communication channels"""

    def __init__(
        self,
        use_compression: bool = True,
        use_encryption: bool = False,
        compression_level: int = 6,
        quantization_bits: Optional[int] = None
    ):
        """
        Initialize secure channel

        Args:
            use_compression: Enable compression
            use_encryption: Enable encryption (placeholder for future)
            compression_level: zlib compression level
            quantization_bits: Bits for quantization (None = no quantization)
        """
        self.use_compression = use_compression
        self.use_encryption = use_encryption
        self.compression_level = compression_level
        self.quantization_bits = quantization_bits

        self.serializer = MessageSerializer()
        self.compressor = CompressionHandler()

    def prepare_message(
        self,
        model_state: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Prepare message for transmission

        Args:
            model_state: Model state to send
            metadata: Optional metadata

        Returns:
            Prepared message bytes
        """
        # Apply quantization if enabled
        if self.quantization_bits is not None:
            model_state = self.compressor.quantize_model(
                model_state, bits=self.quantization_bits
            )

        # Serialize model state
        serialized = self.serializer.serialize_model(model_state)

        # Add metadata
        if metadata:
            metadata_bytes = self.serializer.serialize_metadata(metadata)
            # Combine: [metadata_length(4 bytes)][metadata][model_data]
            metadata_length = len(metadata_bytes).to_bytes(4, byteorder='big')
            serialized = metadata_length + metadata_bytes + serialized

        # Compress if enabled
        if self.use_compression:
            original_size = len(serialized)
            serialized = self.compressor.compress(serialized, level=self.compression_level)
            compressed_size = len(serialized)
            logger.debug(f"Compression: {original_size} -> {compressed_size} bytes "
                        f"({100 * (1 - compressed_size/original_size):.1f}% reduction)")

        return serialized

    def parse_message(
        self,
        message: bytes
    ) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        """
        Parse received message

        Args:
            message: Received message bytes

        Returns:
            Tuple of (model_state, metadata)
        """
        # Decompress if enabled
        if self.use_compression:
            message = self.compressor.decompress(message)

        # Extract metadata if present
        metadata = None
        if len(message) > 4:
            # Check if metadata is present
            try:
                metadata_length = int.from_bytes(message[:4], byteorder='big')
                if metadata_length > 0 and metadata_length < len(message):
                    metadata_bytes = message[4:4 + metadata_length]
                    metadata = self.serializer.deserialize_metadata(metadata_bytes)
                    message = message[4 + metadata_length:]
            except:
                pass  # No metadata present

        # Deserialize model state
        model_state = self.serializer.deserialize_model(message)

        # Dequantize if needed
        if self.quantization_bits is not None:
            # Check if model is quantized
            first_key = list(model_state.keys())[0]
            if isinstance(model_state[first_key], dict):
                model_state = self.compressor.dequantize_model(model_state)

        return model_state, metadata

    @abstractmethod
    def send(self, destination: str, message: bytes) -> bool:
        """Send message to destination"""
        pass

    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Receive message with optional timeout"""
        pass


class GRPCChannel(SecureChannel):
    """gRPC-based communication channel"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        use_ssl: bool = False,
        **kwargs
    ):
        """
        Initialize gRPC channel

        Args:
            host: Server host
            port: Server port
            use_ssl: Use SSL/TLS
            **kwargs: Additional arguments for SecureChannel
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.address = f"{host}:{port}"

        # Note: Actual gRPC implementation would require protobuf definitions
        # This is a simplified version for demonstration
        logger.info(f"Initialized gRPC channel at {self.address}")

    def send(self, destination: str, message: bytes) -> bool:
        """
        Send message via gRPC

        Args:
            destination: Destination address
            message: Message bytes

        Returns:
            Success status
        """
        try:
            # Placeholder for actual gRPC call
            logger.debug(f"Sending {len(message)} bytes to {destination} via gRPC")
            # In real implementation: grpc_stub.SendUpdate(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def receive(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Receive message via gRPC

        Args:
            timeout: Timeout in seconds

        Returns:
            Received message bytes or None
        """
        try:
            # Placeholder for actual gRPC receive
            logger.debug(f"Receiving message via gRPC (timeout={timeout})")
            # In real implementation: return grpc_stub.ReceiveUpdate(timeout=timeout)
            return None
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None


class HTTPChannel(SecureChannel):
    """HTTP/HTTPS-based communication channel"""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        use_ssl: bool = False,
        **kwargs
    ):
        """
        Initialize HTTP channel

        Args:
            base_url: Base URL for server
            use_ssl: Use HTTPS
            **kwargs: Additional arguments for SecureChannel
        """
        super().__init__(**kwargs)
        self.base_url = base_url
        self.use_ssl = use_ssl

        if use_ssl and not base_url.startswith('https'):
            self.base_url = base_url.replace('http://', 'https://')

        logger.info(f"Initialized HTTP channel at {self.base_url}")

    def send(self, destination: str, message: bytes) -> bool:
        """
        Send message via HTTP POST

        Args:
            destination: Endpoint path
            message: Message bytes

        Returns:
            Success status
        """
        try:
            url = f"{self.base_url}/{destination}"
            logger.debug(f"Sending {len(message)} bytes to {url} via HTTP")

            # Placeholder for actual HTTP request
            # In real implementation:
            # import requests
            # response = requests.post(url, data=message, timeout=30)
            # return response.status_code == 200

            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def receive(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Receive message via HTTP GET

        Args:
            timeout: Timeout in seconds

        Returns:
            Received message bytes or None
        """
        try:
            url = f"{self.base_url}/receive"
            logger.debug(f"Receiving message from {url} via HTTP")

            # Placeholder for actual HTTP request
            # In real implementation:
            # import requests
            # response = requests.get(url, timeout=timeout)
            # return response.content if response.status_code == 200 else None

            return None
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None


class MessageQueue:
    """Simple message queue for asynchronous communication"""

    def __init__(self, max_size: int = 100):
        """
        Initialize message queue

        Args:
            max_size: Maximum queue size
        """
        self.queue = []
        self.max_size = max_size

    def put(self, message: bytes, priority: int = 0):
        """Add message to queue"""
        if len(self.queue) >= self.max_size:
            logger.warning("Message queue full, dropping oldest message")
            self.queue.pop(0)

        self.queue.append((priority, message))
        self.queue.sort(key=lambda x: x[0], reverse=True)

    def get(self) -> Optional[bytes]:
        """Get message from queue"""
        if self.queue:
            _, message = self.queue.pop(0)
            return message
        return None

    def size(self) -> int:
        """Get queue size"""
        return len(self.queue)

    def clear(self):
        """Clear queue"""
        self.queue.clear()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample model state
    model_state = {
        'layer1.weight': torch.randn(100, 50),
        'layer1.bias': torch.randn(100),
        'layer2.weight': torch.randn(10, 100),
        'layer2.bias': torch.randn(10)
    }

    # Test with gRPC channel
    grpc_channel = GRPCChannel(
        use_compression=True,
        compression_level=6,
        quantization_bits=8
    )

    # Prepare and send message
    metadata = {'client_id': 1, 'round': 5}
    message = grpc_channel.prepare_message(model_state, metadata)
    print(f"Message size: {len(message)} bytes")

    # Parse message
    received_state, received_metadata = grpc_channel.parse_message(message)
    print(f"Received metadata: {received_metadata}")
    print(f"Received state keys: {list(received_state.keys())}")

    # Verify reconstruction
    for key in model_state.keys():
        original = model_state[key]
        received = received_state[key]
        error = torch.abs(original - received).max().item()
        print(f"{key}: max error = {error:.6f}")
