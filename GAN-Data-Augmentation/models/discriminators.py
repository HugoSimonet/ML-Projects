"""
Discriminator Network Implementations
Includes DCGAN, PatchGAN, Multi-Scale, and Spectral Normalization architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np


class DiscriminatorBlock(nn.Module):
    """Basic discriminator block with downsampling"""

    def __init__(self, in_channels, out_channels, stride=2, normalize=True, use_spectral_norm=False):
        super().__init__()

        conv = nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False)

        if use_spectral_norm:
            conv = spectral_norm(conv)

        layers = [conv]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    """Self-attention layer for capturing long-range dependencies"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Query, key, value projections
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)

        # Attention weights
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        # Residual connection with learnable weight
        out = self.gamma * out + x

        return out


class DCGANDiscriminator(nn.Module):
    """
    Deep Convolutional GAN Discriminator
    Standard architecture with strided convolutions
    """

    def __init__(self, input_channels=3, base_channels=64, image_size=64):
        super().__init__()
        self.input_channels = input_channels

        # Calculate number of downsampling layers
        n_downsampling = int(np.log2(image_size)) - 2

        # Initial layer (no normalization)
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Progressive downsampling blocks
        layers = []
        in_ch = base_channels
        for i in range(n_downsampling - 1):
            out_ch = min(base_channels * 8, in_ch * 2)
            layers.append(DiscriminatorBlock(in_ch, out_ch, stride=2, normalize=True))
            in_ch = out_ch

        self.main = nn.Sequential(*layers)

        # Final classification layer
        self.final = nn.Sequential(
            nn.Conv2d(in_ch, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        x = self.main(x)
        x = self.final(x)
        return x.view(x.size(0), -1)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator
    Outputs a feature map instead of single value for local realism
    """

    def __init__(self, input_channels=3, base_channels=64, n_layers=3):
        super().__init__()

        # Initial layer
        layers = [
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Intermediate layers
        in_ch = base_channels
        for i in range(1, n_layers):
            out_ch = min(base_channels * 8, in_ch * 2)
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_ch = out_ch

        # Final layer
        layers.extend([
            nn.Conv2d(in_ch, in_ch * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch * 2, 1, 4, 1, 1)
        ])

        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator
    Operates at multiple image resolutions for better quality
    """

    def __init__(self, input_channels=3, base_channels=64, n_scales=3):
        super().__init__()
        self.n_scales = n_scales

        # Create discriminators at different scales
        self.discriminators = nn.ModuleList()
        for _ in range(n_scales):
            self.discriminators.append(
                PatchGANDiscriminator(input_channels, base_channels, n_layers=3)
            )

        # Downsampling for multi-scale
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        features = []

        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)

            output = discriminator(x)
            outputs.append(output)

        return outputs


class SpectralNormDiscriminator(nn.Module):
    """
    Discriminator with Spectral Normalization
    Improved training stability through weight normalization
    """

    def __init__(self, input_channels=3, base_channels=64, image_size=64, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # Calculate number of downsampling layers
        n_downsampling = int(np.log2(image_size)) - 2

        # Initial layer
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Progressive downsampling with spectral norm
        layers = []
        in_ch = base_channels
        for i in range(n_downsampling - 1):
            out_ch = min(base_channels * 8, in_ch * 2)
            layers.append(
                nn.Sequential(
                    spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1)),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_ch = out_ch

        self.main = nn.Sequential(*layers)

        # Self-attention layer
        if use_attention:
            self.attention = SelfAttention(in_ch)

        # Final classification
        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, 1, 4, 1, 0))
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        x = self.main(x)

        if self.use_attention:
            x = self.attention(x)

        x = self.final(x)
        return x.view(x.size(0), -1)


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator
    Discriminates based on both image and class label
    """

    def __init__(self, input_channels=3, num_classes=10, base_channels=64,
                 image_size=64, embedding_dim=100):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # Projection to image dimensions
        self.label_proj = nn.Sequential(
            nn.Linear(embedding_dim, image_size * image_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Calculate number of downsampling layers
        n_downsampling = int(np.log2(image_size)) - 2

        # Initial layer (concatenate image and label)
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels + 1, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Downsampling blocks
        layers = []
        in_ch = base_channels
        for i in range(n_downsampling - 1):
            out_ch = min(base_channels * 8, in_ch * 2)
            layers.append(DiscriminatorBlock(in_ch, out_ch, stride=2, normalize=True))
            in_ch = out_ch

        self.main = nn.Sequential(*layers)

        # Final classification
        self.final = nn.Sequential(
            nn.Conv2d(in_ch, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, labels):
        # Embed and project labels
        label_embed = self.label_embedding(labels)
        label_map = self.label_proj(label_embed)

        # Use actual image size from input
        actual_size = x.size(2)
        label_map = label_map.view(x.size(0), 1, self.image_size, self.image_size)

        # Resize label map if needed to match input size
        if actual_size != self.image_size:
            label_map = F.interpolate(label_map, size=(actual_size, actual_size), mode='nearest')

        # Concatenate image and label map
        x = torch.cat([x, label_map], dim=1)

        x = self.initial(x)
        x = self.main(x)
        x = self.final(x)

        return x.view(x.size(0), -1)


class ProgressiveDiscriminator(nn.Module):
    """
    Progressive Growing Discriminator
    Matches progressive generator architecture
    """

    def __init__(self, input_channels=3, base_channels=64, max_resolution=256):
        super().__init__()
        self.input_channels = input_channels
        self.max_resolution = max_resolution

        # From RGB layers for different resolutions
        n_blocks = int(np.log2(max_resolution)) - 2
        self.from_rgb_layers = nn.ModuleList()

        in_ch = base_channels
        for i in range(n_blocks):
            self.from_rgb_layers.append(
                nn.Conv2d(input_channels, in_ch * (2 ** min(i, 3)), 1)
            )

        # Progressive blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            in_ch = base_channels * (2 ** min(n_blocks - i - 1, 3))
            out_ch = base_channels * (2 ** min(n_blocks - i, 3))
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        # Final 4x4 block
        final_ch = base_channels * (2 ** min(3, n_blocks))
        self.final = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(final_ch, final_ch, 4, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(final_ch, 1, 1)
        )

        self.current_resolution = 4
        self.alpha = 1.0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def set_alpha(self, alpha):
        """Set blending factor for progressive growing"""
        self.alpha = alpha

    def grow(self):
        """Increase resolution"""
        if self.current_resolution < self.max_resolution:
            self.current_resolution *= 2

    def forward(self, x):
        # Determine which blocks to use
        n_blocks_to_use = int(np.log2(self.current_resolution)) - 2

        # If in transition, blend between resolutions
        if self.alpha < 1.0 and n_blocks_to_use > 0:
            # Higher resolution path
            x_high = self.from_rgb_layers[n_blocks_to_use](x)

            # Lower resolution path
            x_low = F.avg_pool2d(x, 2)
            x_low = self.from_rgb_layers[n_blocks_to_use - 1](x_low)

            # Blend
            x = self.alpha * x_high + (1 - self.alpha) * x_low
        else:
            x = self.from_rgb_layers[n_blocks_to_use](x)

        # Apply blocks in reverse
        for i in range(n_blocks_to_use - 1, -1, -1):
            x = self.blocks[i](x)

        x = self.final(x)
        return x.view(x.size(0), -1)


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer for better diversity"""

    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        group_size = min(batch_size, self.group_size)

        # Compute std dev per group
        y = x.view(group_size, -1, channels, height, width)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0) + 1e-8)
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(group_size, 1, height, width)

        return torch.cat([x, y], dim=1)
