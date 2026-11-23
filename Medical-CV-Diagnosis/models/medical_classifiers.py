"""
Medical CNN Classifiers
ResNet, DenseNet, EfficientNet, and Vision Transformer implementations
Optimized for medical imaging with modifications for grayscale/multi-modal inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MedicalResNet(nn.Module):
    """
    ResNet architecture adapted for medical imaging
    Supports grayscale and multi-channel inputs
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        architecture: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize Medical ResNet

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            architecture: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before final layer
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Load pretrained ResNet
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif architecture == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feat_dim = 512
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feat_dim = 2048
        elif architecture == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feat_dim = 2048
        elif architecture == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Modify first conv layer for grayscale input
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove original FC layer
        self.backbone.fc = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification"""
        return self.backbone(x)


class MedicalDenseNet(nn.Module):
    """
    DenseNet architecture adapted for medical imaging
    Efficient feature reuse, good for limited data
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        architecture: str = 'densenet121',
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize Medical DenseNet

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            architecture: DenseNet variant ('densenet121', 'densenet169', 'densenet201')
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Load pretrained DenseNet
        if architecture == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            feat_dim = 1024
        elif architecture == 'densenet169':
            self.backbone = models.densenet169(pretrained=pretrained)
            feat_dim = 1664
        elif architecture == 'densenet201':
            self.backbone = models.densenet201(pretrained=pretrained)
            feat_dim = 1920
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Modify first conv for grayscale
        if in_channels != 3:
            self.backbone.features.conv0 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone.features(x)
        features = F.relu(features, inplace=True)
        output = self.classifier(features)
        return output


class MedicalEfficientNet(nn.Module):
    """
    EfficientNet architecture for medical imaging
    State-of-the-art efficiency and accuracy trade-off
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        architecture: str = 'efficientnet_b0',
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize Medical EfficientNet

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            architecture: EfficientNet variant ('efficientnet_b0' to 'efficientnet_b7')
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Load pretrained EfficientNet using timm
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove global pooling
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 224, 224)
            feat_dim = self.backbone(dummy_input).shape[1]

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class MedicalViT(nn.Module):
    """
    Vision Transformer adapted for medical imaging
    Excellent for capturing long-range dependencies
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        image_size: int = 224,
        architecture: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Initialize Medical Vision Transformer

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            image_size: Input image size
            architecture: ViT variant ('vit_tiny', 'vit_small', 'vit_base', 'vit_large')
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Load pretrained ViT
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=image_size,
            num_classes=0  # Remove head
        )

        # Get embedding dimension
        embed_dim = self.backbone.embed_dim

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Get features from ViT (CLS token)
        features = self.backbone(x)
        output = self.head(features)
        return output


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple medical classifiers
    Combines predictions from different architectures
    """

    def __init__(
        self,
        models_list: List[nn.Module],
        ensemble_method: str = 'average'
    ):
        """
        Initialize ensemble classifier

        Args:
            models_list: List of trained models
            ensemble_method: 'average', 'max', or 'weighted'
        """
        super().__init__()

        self.models = nn.ModuleList(models_list)
        self.ensemble_method = ensemble_method
        self.num_models = len(models_list)

        if ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ensembling"""
        outputs = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                outputs.append(output)

        outputs = torch.stack(outputs)  # (num_models, batch, num_classes)

        if self.ensemble_method == 'average':
            return torch.mean(outputs, dim=0)
        elif self.ensemble_method == 'max':
            return torch.max(outputs, dim=0)[0]
        elif self.ensemble_method == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            weighted = outputs * weights.view(-1, 1, 1)
            return torch.sum(weighted, dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


def create_medical_classifier(
    architecture: str,
    num_classes: int,
    in_channels: int = 1,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create medical classifier

    Args:
        architecture: Model architecture name
        num_classes: Number of classes
        in_channels: Input channels
        pretrained: Use pretrained weights
        **kwargs: Additional model-specific arguments

    Returns:
        Medical classifier model
    """
    architecture = architecture.lower()

    if 'resnet' in architecture:
        model = MedicalResNet(
            num_classes=num_classes,
            in_channels=in_channels,
            architecture=architecture,
            pretrained=pretrained,
            **kwargs
        )
    elif 'densenet' in architecture:
        model = MedicalDenseNet(
            num_classes=num_classes,
            in_channels=in_channels,
            architecture=architecture,
            pretrained=pretrained,
            **kwargs
        )
    elif 'efficientnet' in architecture:
        model = MedicalEfficientNet(
            num_classes=num_classes,
            in_channels=in_channels,
            architecture=architecture,
            pretrained=pretrained,
            **kwargs
        )
    elif 'vit' in architecture:
        model = MedicalViT(
            num_classes=num_classes,
            in_channels=in_channels,
            architecture=architecture,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    logger.info(f"Created {architecture} with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Total parameters and trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params
