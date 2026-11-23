"""
Vision Encoder Module
Implements ResNet-50 and Vision Transformer (ViT) backbones for image feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class VisionEncoder(nn.Module):
    """
    Vision Encoder using ResNet-50 backbone
    Extracts spatial features from images for cross-modal processing
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        output_dim: int = 768,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            feature_dim: Dimension of ResNet-50 features (2048)
            output_dim: Output dimension for projection
            pretrained: Use pretrained ImageNet weights
            freeze_backbone: Freeze backbone weights during training
        """
        super(VisionEncoder, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)

        # Remove final FC layer and avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Adaptive pooling to get fixed size features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Projection layer to output_dim
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Positional encoding for spatial features
        self.register_buffer(
            'pos_encoding',
            self._create_positional_encoding(49, output_dim)
        )

        self.output_dim = output_dim

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            features: Projected features [B, 49, output_dim]
            pooled_features: Global pooled features [B, output_dim]
        """
        batch_size = images.size(0)

        # Extract features from backbone
        x = self.backbone(images)  # [B, 2048, H/32, W/32]

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # [B, 2048, 7, 7]

        # Reshape to sequence
        x = x.view(batch_size, 2048, -1).transpose(1, 2)  # [B, 49, 2048]

        # Project to output dimension
        features = self.projection(x)  # [B, 49, output_dim]

        # Add positional encoding
        features = features + self.pos_encoding.unsqueeze(0)

        # Global pooled features
        pooled_features = features.mean(dim=1)  # [B, output_dim]

        return features, pooled_features


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder
    Alternative to ResNet-50 for better performance on vision-language tasks
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        pretrained: bool = True
    ):
        """
        Args:
            image_size: Input image size (assumes square)
            patch_size: Size of image patches
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension
            mlp_dim: MLP dimension in transformer
            dropout: Dropout rate
            pretrained: Use pretrained weights if available
        """
        super(ViTEncoder, self).__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_dim = hidden_dim

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            3, hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            features: Patch features [B, num_patches+1, hidden_dim]
            pooled_features: CLS token features [B, hidden_dim]
        """
        batch_size = images.size(0)

        # Extract patches
        x = self.patch_embedding(images)  # [B, hidden_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, hidden_dim]

        # Add positional embedding
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Transformer encoding
        x = self.transformer(x)  # [B, num_patches+1, hidden_dim]
        x = self.layer_norm(x)

        # CLS token as global representation
        pooled_features = x[:, 0]  # [B, hidden_dim]

        return x, pooled_features


if __name__ == "__main__":
    # Test ResNet encoder
    encoder = VisionEncoder(output_dim=768, pretrained=False)
    images = torch.randn(2, 3, 224, 224)
    features, pooled = encoder(images)
    print(f"ResNet Features: {features.shape}, Pooled: {pooled.shape}")

    # Test ViT encoder
    vit_encoder = ViTEncoder(pretrained=False)
    features, pooled = vit_encoder(images)
    print(f"ViT Features: {features.shape}, Pooled: {pooled.shape}")
