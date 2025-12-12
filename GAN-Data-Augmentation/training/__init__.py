"""
Training Framework Package
Contains GAN training logic and loss functions
"""

from .gan_trainer import (
    GANTrainer,
    WGANTrainer,
    ProgressiveGANTrainer,
    ConditionalGANTrainer
)

from .losses import (
    GANLoss,
    WassersteinLoss,
    PerceptualLoss,
    FeatureMatchingLoss,
    GradientPenalty
)

__all__ = [
    'GANTrainer',
    'WGANTrainer',
    'ProgressiveGANTrainer',
    'ConditionalGANTrainer',
    'GANLoss',
    'WassersteinLoss',
    'PerceptualLoss',
    'FeatureMatchingLoss',
    'GradientPenalty'
]
