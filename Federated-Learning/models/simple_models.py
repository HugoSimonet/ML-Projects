"""
Simple neural network models for federated learning
Includes models for MNIST, CIFAR-10, and other common datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        hidden_dims: List[int] = [32, 64]
    ):
        """
        Initialize Simple CNN

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            hidden_dims: List of hidden dimensions for conv layers
        """
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_dims[0], kernel_size=5)
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened size (assuming 28x28 input for MNIST)
        self.fc1 = nn.Linear(hidden_dims[1] * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MNISTNet(nn.Module):
    """Optimized CNN for MNIST dataset"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: 28x28x1
        x = F.relu(self.conv1(x))  # 28x28x32
        x = self.pool(x)           # 14x14x32
        x = F.relu(self.conv2(x))  # 14x14x64
        x = self.pool(x)           # 7x7x64
        x = F.relu(self.conv3(x))  # 7x7x64
        x = self.pool(x)           # 3x3x64

        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # Flatten to 576

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class CIFAR10Net(nn.Module):
    """CNN optimized for CIFAR-10 dataset"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input: 32x32x3
        x = F.relu(self.bn1(self.conv1(x)))  # 32x32x64
        x = self.pool(x)                      # 16x16x64

        x = F.relu(self.bn2(self.conv2(x)))  # 16x16x128
        x = self.pool(x)                      # 8x8x128

        x = F.relu(self.bn3(self.conv3(x)))  # 8x8x256
        x = F.relu(self.bn4(self.conv4(x)))  # 8x8x256
        x = self.pool(x)                      # 4x4x256

        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron for simple datasets"""

    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        hidden_dims: List[int] = [128, 64]
    ):
        """
        Initialize MLP

        Args:
            input_dim: Input dimension (flattened)
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.model(x)


class ResidualBlock(nn.Module):
    """Residual block for ResNet"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18FL(nn.Module):
    """ResNet-18 adapted for federated learning"""

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def create_model(model_name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Factory function to create models

    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    models = {
        'simple_cnn': SimpleCNN,
        'mnist_net': MNISTNet,
        'cifar10_net': CIFAR10Net,
        'mlp': MLP,
        'resnet18': ResNet18FL
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    # MNISTNet and CIFAR10Net only accept num_classes
    if model_name.lower() in ['mnist_net', 'cifar10_net']:
        return models[model_name.lower()](num_classes=num_classes)

    return models[model_name.lower()](num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test models
    print("Testing models...")

    # Test MNIST Net
    mnist_model = MNISTNet(num_classes=10)
    mnist_input = torch.randn(4, 1, 28, 28)
    mnist_output = mnist_model(mnist_input)
    print(f"MNIST Net: Input {mnist_input.shape} -> Output {mnist_output.shape}")
    print(f"MNIST Net parameters: {sum(p.numel() for p in mnist_model.parameters()):,}")

    # Test CIFAR10 Net
    cifar_model = CIFAR10Net(num_classes=10)
    cifar_input = torch.randn(4, 3, 32, 32)
    cifar_output = cifar_model(cifar_input)
    print(f"\nCIFAR10 Net: Input {cifar_input.shape} -> Output {cifar_output.shape}")
    print(f"CIFAR10 Net parameters: {sum(p.numel() for p in cifar_model.parameters()):,}")

    # Test MLP
    mlp_model = MLP(input_dim=784, num_classes=10, hidden_dims=[256, 128])
    mlp_input = torch.randn(4, 784)
    mlp_output = mlp_model(mlp_input)
    print(f"\nMLP: Input {mlp_input.shape} -> Output {mlp_output.shape}")
    print(f"MLP parameters: {sum(p.numel() for p in mlp_model.parameters()):,}")

    # Test ResNet18
    resnet_model = ResNet18FL(num_classes=10)
    resnet_input = torch.randn(4, 3, 32, 32)
    resnet_output = resnet_model(resnet_input)
    print(f"\nResNet18: Input {resnet_input.shape} -> Output {resnet_output.shape}")
    print(f"ResNet18 parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")

    # Test model factory
    model = create_model('mnist_net', num_classes=10)
    print(f"\nFactory-created MNIST Net: {type(model).__name__}")
