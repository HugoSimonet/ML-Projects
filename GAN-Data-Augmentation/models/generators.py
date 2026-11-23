"""
Generator Network Implementations
Includes DCGAN, StyleGAN, Conditional, and Progressive Growing architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GeneratorBlock(nn.Module):
    """Basic generator block with upsampling"""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, final=False):
        super().__init__()
        self.final = final

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )

        if not final:
            self.bn = nn.BatchNorm2d(out_channels)
            self.activation = nn.ReLU(True)
        else:
            self.activation = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        if not self.final:
            x = self.bn(x)
        x = self.activation(x)
        return x


class DCGANGenerator(nn.Module):
    """
    Deep Convolutional GAN Generator
    Standard architecture with transposed convolutions
    """

    def __init__(self, latent_dim=100, output_channels=3, base_channels=64, image_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.image_size = image_size

        # Calculate number of upsampling layers needed
        self.n_upsampling = int(np.log2(image_size)) - 2

        # Initial projection
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True)
        )

        # Progressive upsampling blocks
        layers = []
        in_ch = base_channels * 8
        for i in range(self.n_upsampling):
            out_ch = in_ch // 2 if i < self.n_upsampling - 1 else base_channels
            layers.append(GeneratorBlock(in_ch, out_ch))
            in_ch = out_ch

        self.main = nn.Sequential(*layers)

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_channels, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)

        x = self.initial(z)
        x = self.main(x)
        x = self.final(x)
        return x


class StyleModulation(nn.Module):
    """Style modulation layer for StyleGAN"""

    def __init__(self, latent_dim, channels):
        super().__init__()
        self.style_scale = nn.Linear(latent_dim, channels)
        self.style_bias = nn.Linear(latent_dim, channels)

    def forward(self, x, style):
        if style is None:
            return x

        scale = self.style_scale(style).unsqueeze(-1).unsqueeze(-1)
        bias = self.style_bias(style).unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias


class StyleBlock(nn.Module):
    """StyleGAN generator block with style modulation"""

    def __init__(self, in_channels, out_channels, latent_dim, upsample=True):
        super().__init__()
        self.upsample = upsample

        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.style_mod1 = StyleModulation(latent_dim, out_channels)
        self.style_mod2 = StyleModulation(latent_dim, out_channels)

        self.noise1 = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.noise2 = nn.Parameter(torch.randn(1, 1, 1, 1))

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style):
        if self.upsample:
            x = self.up(x)

        # First convolution
        x = self.conv1(x)
        x = self.style_mod1(x, style)
        x = x + self.noise1 * torch.randn_like(x)
        x = self.activation(x)

        # Second convolution
        x = self.conv2(x)
        x = self.style_mod2(x, style)
        x = x + self.noise2 * torch.randn_like(x)
        x = self.activation(x)

        return x


class MappingNetwork(nn.Module):
    """Mapping network to transform latent code"""

    def __init__(self, latent_dim, num_layers=8):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


class StyleGANGenerator(nn.Module):
    """
    StyleGAN Generator
    High-quality generation with style modulation
    """

    def __init__(self, latent_dim=512, output_channels=3, base_channels=64, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        # Mapping network
        self.mapping = MappingNetwork(latent_dim)

        # Constant input
        self.constant = nn.Parameter(torch.randn(1, base_channels * 8, 4, 4))

        # Calculate number of blocks
        n_blocks = int(np.log2(image_size)) - 2

        # Style blocks
        self.blocks = nn.ModuleList()
        in_ch = base_channels * 8
        for i in range(n_blocks):
            out_ch = max(base_channels, in_ch // 2)
            self.blocks.append(StyleBlock(in_ch, out_ch, latent_dim, upsample=True))
            in_ch = out_ch

        # To RGB
        self.to_rgb = nn.Conv2d(in_ch, output_channels, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z, return_latents=False):
        # Map to intermediate latent space
        w = self.mapping(z)

        # Start from constant input
        x = self.constant.repeat(z.size(0), 1, 1, 1)

        # Apply style blocks
        for block in self.blocks:
            x = block(x, w)

        # Convert to RGB
        x = self.to_rgb(x)
        x = torch.tanh(x)

        if return_latents:
            return x, w
        return x


class ConditionalGenerator(nn.Module):
    """
    Conditional GAN Generator
    Generates images conditioned on class labels
    """

    def __init__(self, latent_dim=100, num_classes=10, output_channels=3,
                 base_channels=64, image_size=64, embedding_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Class embedding
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # Combined input dimension
        self.input_dim = latent_dim + embedding_dim

        # Calculate number of upsampling layers
        n_upsampling = int(np.log2(image_size)) - 2

        # Initial projection
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True)
        )

        # Upsampling blocks
        layers = []
        in_ch = base_channels * 8
        for i in range(n_upsampling):
            out_ch = in_ch // 2 if i < n_upsampling - 1 else base_channels
            layers.append(GeneratorBlock(in_ch, out_ch))
            in_ch = out_ch

        self.main = nn.Sequential(*layers)

        # Final output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_channels, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z, labels):
        # Embed labels
        label_embed = self.label_embedding(labels)

        # Concatenate noise and label embedding
        z_combined = torch.cat([z, label_embed], dim=1)
        z_combined = z_combined.view(z_combined.size(0), z_combined.size(1), 1, 1)

        x = self.initial(z_combined)
        x = self.main(x)
        x = self.final(x)
        return x


class ProgressiveBlock(nn.Module):
    """Progressive growing block"""

    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        self.upsample = upsample

        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        return x


class ProgressiveGenerator(nn.Module):
    """
    Progressive Growing GAN Generator
    Supports training at multiple resolutions
    """

    def __init__(self, latent_dim=512, output_channels=3, base_channels=64,
                 max_resolution=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.max_resolution = max_resolution

        # Initial 4x4 block
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Progressive blocks
        n_blocks = int(np.log2(max_resolution)) - 2
        self.blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()

        # Initial to_rgb for 4x4 resolution
        in_ch = base_channels * 8
        self.to_rgb_layers.append(nn.Conv2d(in_ch, output_channels, 1))

        # Progressive blocks
        for i in range(n_blocks):
            out_ch = max(base_channels, in_ch // 2)
            self.blocks.append(ProgressiveBlock(in_ch, out_ch, upsample=True))
            self.to_rgb_layers.append(nn.Conv2d(out_ch, output_channels, 1))
            in_ch = out_ch

        self.current_resolution = 4
        self.alpha = 1.0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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

    def forward(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)

        x = self.initial(z)

        # Determine which blocks to use based on current resolution
        n_blocks_to_use = int(np.log2(self.current_resolution)) - 2

        for i in range(n_blocks_to_use):
            x = self.blocks[i](x)

        # If in transition, blend between resolutions
        if self.alpha < 1.0 and n_blocks_to_use > 0:
            # Lower resolution path (before adding new layer)
            x_low = self.to_rgb_layers[n_blocks_to_use](x)
            x_low = F.interpolate(x_low, scale_factor=2, mode='nearest')

            # Higher resolution path (after adding new layer)
            x = self.blocks[n_blocks_to_use](x)
            x_high = self.to_rgb_layers[n_blocks_to_use + 1](x)

            # Blend
            x = self.alpha * x_high + (1 - self.alpha) * x_low
        else:
            # Normal path: use to_rgb at current resolution
            # Index is n_blocks_to_use because we have initial to_rgb at index 0
            x = self.to_rgb_layers[n_blocks_to_use](x)

        return torch.tanh(x)


class ResidualBlock(nn.Module):
    """Residual block for generator"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.activation(out)

        return out


class CycleGANGenerator(nn.Module):
    """
    CycleGAN Generator for domain transfer
    Uses ResNet architecture
    """

    def __init__(self, input_channels=3, output_channels=3, base_channels=64, n_residual_blocks=9):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, 1, 3, padding_mode='reflect'),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        residual_blocks = []
        for _ in range(n_residual_blocks):
            residual_blocks.append(ResidualBlock(base_channels * 4))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, 7, 1, 3, padding_mode='reflect'),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        x = self.downsample(x)
        x = self.residual_blocks(x)
        x = self.upsample(x)
        x = self.output(x)
        return x
