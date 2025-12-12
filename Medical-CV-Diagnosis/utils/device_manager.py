"""Device Manager for GPU/CPU handling"""

import torch
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manage device (GPU/CPU) selection and information"""

    @staticmethod
    def get_device(prefer_gpu: bool = True, gpu_id: int = 0) -> torch.device:
        """
        Get appropriate device

        Args:
            prefer_gpu: Prefer GPU if available
            gpu_id: GPU device ID

        Returns:
            torch.device
        """
        if prefer_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")

        return device

    @staticmethod
    def print_gpu_info():
        """Print GPU information"""
        if torch.cuda.is_available():
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")
                logger.info(f"  Compute Capability: {props.major}.{props.minor}")
        else:
            logger.info("No GPU available")

    @staticmethod
    def clear_cache():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")


def get_device(prefer_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    """Convenience function to get device"""
    return DeviceManager.get_device(prefer_gpu, gpu_id)
