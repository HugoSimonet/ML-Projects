"""
Medical Segmentation Models
U-Net, U-Net++, and Attention U-Net implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block: Conv -> BatchNorm -> ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batchnorm: bool = True
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_batchnorm)
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()

        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, use_batchnorm=use_batchnorm),
            ConvBlock(out_channels, out_channels, use_batchnorm=use_batchnorm)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batchnorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        use_batchnorm: bool = True
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, use_batchnorm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_batchnorm)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad x1 to match x2 size if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for Attention U-Net"""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from coarser scale
            x: Skip connection from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample g1 to match x1 size
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UNet(nn.Module):
    """
    U-Net architecture for medical image segmentation
    https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        features: List[int] = [64, 128, 256, 512],
        bilinear: bool = True,
        use_batchnorm: bool = True
    ):
        """
        Initialize U-Net

        Args:
            in_channels: Number of input channels
            num_classes: Number of segmentation classes
            features: List of feature dimensions for each level
            bilinear: Use bilinear upsampling instead of transposed conv
            use_batchnorm: Use batch normalization
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(in_channels, features[0], use_batchnorm)
        self.down1 = Down(features[0], features[1], use_batchnorm)
        self.down2 = Down(features[1], features[2], use_batchnorm)
        self.down3 = Down(features[2], features[3], use_batchnorm)

        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor, use_batchnorm)

        # Decoder
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear, use_batchnorm)
        self.up2 = Up(features[3], features[2] // factor, bilinear, use_batchnorm)
        self.up3 = Up(features[2], features[1] // factor, bilinear, use_batchnorm)
        self.up4 = Up(features[1], features[0], bilinear, use_batchnorm)

        # Output
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass"""
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits


class AttentionUNet(nn.Module):
    """
    Attention U-Net for medical image segmentation
    Attention gates help focus on relevant regions
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        features: List[int] = [64, 128, 256, 512],
        bilinear: bool = True,
        use_batchnorm: bool = True
    ):
        """
        Initialize Attention U-Net

        Args:
            in_channels: Number of input channels
            num_classes: Number of segmentation classes
            features: List of feature dimensions
            bilinear: Use bilinear upsampling
            use_batchnorm: Use batch normalization
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.inc = DoubleConv(in_channels, features[0], use_batchnorm)
        self.down1 = Down(features[0], features[1], use_batchnorm)
        self.down2 = Down(features[1], features[2], use_batchnorm)
        self.down3 = Down(features[2], features[3], use_batchnorm)

        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor, use_batchnorm)

        # Attention gates
        self.att1 = AttentionGate(features[3] * 2 // factor, features[3], features[3] // 2)
        self.att2 = AttentionGate(features[3] // factor, features[2], features[2] // 2)
        self.att3 = AttentionGate(features[2] // factor, features[1], features[1] // 2)
        self.att4 = AttentionGate(features[1] // factor, features[0], features[0] // 2)

        # Decoder
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear, use_batchnorm)
        self.up2 = Up(features[3], features[2] // factor, bilinear, use_batchnorm)
        self.up3 = Up(features[2], features[1] // factor, bilinear, use_batchnorm)
        self.up4 = Up(features[1], features[0], bilinear, use_batchnorm)

        # Output
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass with attention"""
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with attention gates
        x4_att = self.att1(x5, x4)
        x = self.up1(x5, x4_att)

        x3_att = self.att2(x, x3)
        x = self.up2(x, x3_att)

        x2_att = self.att3(x, x2)
        x = self.up3(x, x2_att)

        x1_att = self.att4(x, x1)
        x = self.up4(x, x1_att)

        # Output
        logits = self.outc(x)
        return logits


class NestedBlock(nn.Module):
    """Nested dense block for U-Net++"""

    def __init__(self, in_channels: List[int], out_channels: int, use_batchnorm: bool = True):
        super().__init__()

        self.conv = DoubleConv(sum(in_channels), out_channels, use_batchnorm)

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) architecture
    Dense skip connections for better gradient flow
    https://arxiv.org/abs/1807.10165
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        features: int = 32,
        deep_supervision: bool = False,
        use_batchnorm: bool = True
    ):
        """
        Initialize U-Net++

        Args:
            in_channels: Number of input channels
            num_classes: Number of segmentation classes
            features: Base number of features
            deep_supervision: Use deep supervision
            use_batchnorm: Use batch normalization
        """
        super().__init__()

        self.deep_supervision = deep_supervision

        # Encoder
        self.conv0_0 = DoubleConv(in_channels, features, use_batchnorm)
        self.conv1_0 = Down(features, features * 2, use_batchnorm)
        self.conv2_0 = Down(features * 2, features * 4, use_batchnorm)
        self.conv3_0 = Down(features * 4, features * 8, use_batchnorm)
        self.conv4_0 = Down(features * 8, features * 16, use_batchnorm)

        # Nested connections
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_1 = NestedBlock([features, features * 2], features, use_batchnorm)
        self.conv1_1 = NestedBlock([features * 2, features * 4], features * 2, use_batchnorm)
        self.conv2_1 = NestedBlock([features * 4, features * 8], features * 4, use_batchnorm)
        self.conv3_1 = NestedBlock([features * 8, features * 16], features * 8, use_batchnorm)

        self.conv0_2 = NestedBlock([features, features, features * 2], features, use_batchnorm)
        self.conv1_2 = NestedBlock([features * 2, features * 2, features * 4], features * 2, use_batchnorm)
        self.conv2_2 = NestedBlock([features * 4, features * 4, features * 8], features * 4, use_batchnorm)

        self.conv0_3 = NestedBlock([features, features, features, features * 2], features, use_batchnorm)
        self.conv1_3 = NestedBlock([features * 2, features * 2, features * 2, features * 4], features * 2, use_batchnorm)

        self.conv0_4 = NestedBlock([features, features, features, features, features * 2], features, use_batchnorm)

        # Output
        if deep_supervision:
            self.out1 = nn.Conv2d(features, num_classes, kernel_size=1)
            self.out2 = nn.Conv2d(features, num_classes, kernel_size=1)
            self.out3 = nn.Conv2d(features, num_classes, kernel_size=1)
            self.out4 = nn.Conv2d(features, num_classes, kernel_size=1)
        else:
            self.out = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass"""
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)

        # Nested decoder
        x0_1 = self.conv0_1(x0_0, self.up(x1_0))
        x1_1 = self.conv1_1(x1_0, self.up(x2_0))
        x2_1 = self.conv2_1(x2_0, self.up(x3_0))
        x3_1 = self.conv3_1(x3_0, self.up(x4_0))

        x0_2 = self.conv0_2(x0_0, x0_1, self.up(x1_1))
        x1_2 = self.conv1_2(x1_0, x1_1, self.up(x2_1))
        x2_2 = self.conv2_2(x2_0, x2_1, self.up(x3_1))

        x0_3 = self.conv0_3(x0_0, x0_1, x0_2, self.up(x1_2))
        x1_3 = self.conv1_3(x1_0, x1_1, x1_2, self.up(x2_2))

        x0_4 = self.conv0_4(x0_0, x0_1, x0_2, x0_3, self.up(x1_3))

        # Output
        if self.deep_supervision:
            out1 = self.out1(x0_1)
            out2 = self.out2(x0_2)
            out3 = self.out3(x0_3)
            out4 = self.out4(x0_4)
            return [out1, out2, out3, out4]
        else:
            return self.out(x0_4)


def create_segmentation_model(
    architecture: str,
    in_channels: int = 1,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create segmentation model

    Args:
        architecture: Model architecture ('unet', 'attention_unet', 'unet_plusplus')
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        **kwargs: Additional model-specific arguments

    Returns:
        Segmentation model
    """
    architecture = architecture.lower()

    if architecture == 'unet':
        model = UNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif architecture == 'attention_unet':
        model = AttentionUNet(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif architecture == 'unet_plusplus' or architecture == 'unetplusplus':
        model = UNetPlusPlus(in_channels=in_channels, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    logger.info(f"Created {architecture} with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model
