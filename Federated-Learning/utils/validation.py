"""
Input validation decorators for federated learning
Provides decorators to validate function parameters and improve code robustness
"""

import functools
import torch
from typing import Any, Callable, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def validate_positive(param_name: str):
    """
    Decorator to validate that a parameter is positive

    Args:
        param_name: Name of the parameter to validate

    Example:
        @validate_positive('learning_rate')
        def train(learning_rate: float):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get parameter value
            sig = func.__code__.co_varnames
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param_name in sig:
                idx = sig.index(param_name)
                if idx < len(args):
                    value = args[idx]
                else:
                    return func(*args, **kwargs)  # Parameter not provided
            else:
                return func(*args, **kwargs)

            # Validate
            if value is not None and value <= 0:
                raise ValueError(f"{param_name} must be positive, got {value}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_range(param_name: str, min_val: Optional[float] = None, max_val: Optional[float] = None):
    """
    Decorator to validate that a parameter is within a range

    Args:
        param_name: Name of the parameter to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Example:
        @validate_range('epsilon', min_val=0.0, max_val=10.0)
        def set_privacy(epsilon: float):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get parameter value
            sig = func.__code__.co_varnames
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param_name in sig:
                idx = sig.index(param_name)
                if idx < len(args):
                    value = args[idx]
                else:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

            # Validate
            if value is not None:
                if min_val is not None and value < min_val:
                    raise ValueError(f"{param_name} must be >= {min_val}, got {value}")
                if max_val is not None and value > max_val:
                    raise ValueError(f"{param_name} must be <= {max_val}, got {value}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_type(param_name: str, expected_type: Union[type, tuple]):
    """
    Decorator to validate parameter type

    Args:
        param_name: Name of the parameter to validate
        expected_type: Expected type or tuple of types

    Example:
        @validate_type('model', torch.nn.Module)
        def train(model):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get parameter value
            sig = func.__code__.co_varnames
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param_name in sig:
                idx = sig.index(param_name)
                if idx < len(args):
                    value = args[idx]
                else:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

            # Validate
            if value is not None and not isinstance(value, expected_type):
                raise TypeError(
                    f"{param_name} must be of type {expected_type}, got {type(value)}"
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_not_none(*param_names: str):
    """
    Decorator to validate that parameters are not None

    Args:
        *param_names: Names of parameters to validate

    Example:
        @validate_not_none('model', 'optimizer')
        def train(model, optimizer):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = func.__code__.co_varnames

            for param_name in param_names:
                value = None

                # Get parameter value
                if param_name in kwargs:
                    value = kwargs[param_name]
                elif param_name in sig:
                    idx = sig.index(param_name)
                    if idx < len(args):
                        value = args[idx]

                # Validate
                if value is None:
                    raise ValueError(f"{param_name} cannot be None")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_tensor_shape(param_name: str, expected_dims: Optional[int] = None,
                         expected_shape: Optional[tuple] = None):
    """
    Decorator to validate tensor shape

    Args:
        param_name: Name of the tensor parameter to validate
        expected_dims: Expected number of dimensions
        expected_shape: Expected shape (use None for flexible dimensions)

    Example:
        @validate_tensor_shape('input', expected_dims=4)  # Batch of images
        def forward(input):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get parameter value
            sig = func.__code__.co_varnames
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param_name in sig:
                idx = sig.index(param_name)
                if idx < len(args):
                    value = args[idx]
                else:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

            # Validate
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{param_name} must be a torch.Tensor, got {type(value)}")

                if expected_dims is not None and value.dim() != expected_dims:
                    raise ValueError(
                        f"{param_name} must have {expected_dims} dimensions, "
                        f"got {value.dim()} (shape: {value.shape})"
                    )

                if expected_shape is not None:
                    for i, (expected, actual) in enumerate(zip(expected_shape, value.shape)):
                        if expected is not None and expected != actual:
                            raise ValueError(
                                f"{param_name} shape mismatch at dimension {i}: "
                                f"expected {expected}, got {actual} (shape: {value.shape})"
                            )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_dict_keys(param_name: str, required_keys: List[str],
                       optional_keys: Optional[List[str]] = None):
    """
    Decorator to validate dictionary keys

    Args:
        param_name: Name of the dict parameter to validate
        required_keys: List of required keys
        optional_keys: List of optional keys (if None, any additional keys allowed)

    Example:
        @validate_dict_keys('config', required_keys=['model', 'data'])
        def setup(config: dict):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get parameter value
            sig = func.__code__.co_varnames
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param_name in sig:
                idx = sig.index(param_name)
                if idx < len(args):
                    value = args[idx]
                else:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

            # Validate
            if value is not None:
                if not isinstance(value, dict):
                    raise TypeError(f"{param_name} must be a dict, got {type(value)}")

                # Check required keys
                missing_keys = set(required_keys) - set(value.keys())
                if missing_keys:
                    raise ValueError(
                        f"{param_name} missing required keys: {missing_keys}"
                    )

                # Check for unexpected keys if optional_keys is specified
                if optional_keys is not None:
                    allowed_keys = set(required_keys) | set(optional_keys)
                    unexpected_keys = set(value.keys()) - allowed_keys
                    if unexpected_keys:
                        raise ValueError(
                            f"{param_name} has unexpected keys: {unexpected_keys}"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_device(param_name: str = 'device'):
    """
    Decorator to validate PyTorch device

    Args:
        param_name: Name of the device parameter (default: 'device')

    Example:
        @validate_device()
        def train(device: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get parameter value
            sig = func.__code__.co_varnames
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param_name in sig:
                idx = sig.index(param_name)
                if idx < len(args):
                    value = args[idx]
                else:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

            # Validate
            if value is not None:
                valid_devices = ['cpu', 'cuda', 'mps']
                if isinstance(value, str):
                    device_type = value.split(':')[0]
                    if device_type not in valid_devices:
                        raise ValueError(
                            f"Invalid device: {value}. Must be one of {valid_devices}"
                        )
                elif isinstance(value, torch.device):
                    if value.type not in valid_devices:
                        raise ValueError(
                            f"Invalid device type: {value.type}. Must be one of {valid_devices}"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_validation_warning(message: str):
    """
    Decorator to log validation warnings instead of raising exceptions
    Useful for soft validation where you want to warn but not fail

    Args:
        message: Warning message template (can use {param_name} and {value})

    Example:
        @log_validation_warning("Learning rate {value} is unusually high")
        def train(learning_rate: float):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.warning(message)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience function to chain multiple validators
def validate_all(*decorators):
    """
    Chain multiple validation decorators

    Args:
        *decorators: Validation decorators to apply

    Example:
        @validate_all(
            validate_positive('learning_rate'),
            validate_range('epsilon', min_val=0, max_val=10),
            validate_not_none('model')
        )
        def train(model, learning_rate, epsilon):
            ...
    """
    def decorator(func: Callable) -> Callable:
        result = func
        for dec in reversed(decorators):
            result = dec(result)
        return result
    return decorator


if __name__ == "__main__":
    # Example usage and tests

    @validate_positive('lr')
    @validate_range('epsilon', min_val=0.0, max_val=10.0)
    def example_train(lr: float, epsilon: float):
        print(f"Training with lr={lr}, epsilon={epsilon}")

    # Valid call
    example_train(lr=0.01, epsilon=1.0)

    # Invalid calls (uncomment to test)
    # example_train(lr=-0.01, epsilon=1.0)  # Should raise ValueError
    # example_train(lr=0.01, epsilon=15.0)  # Should raise ValueError

    @validate_tensor_shape('input', expected_dims=4)
    def example_forward(input: torch.Tensor):
        print(f"Processing tensor of shape {input.shape}")

    # Valid call
    example_forward(torch.randn(32, 3, 224, 224))

    # Invalid call (uncomment to test)
    # example_forward(torch.randn(32, 3))  # Should raise ValueError

    print("All validation examples passed!")
