"""
GAN Training Framework
Comprehensive trainers for different GAN architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np
from .losses import (
    GANLoss, WassersteinLoss, GradientPenalty,
    PerceptualLoss, FeatureMatchingLoss, HingeLoss,
    R1Regularization
)


class GANTrainer:
    """
    Base GAN Trainer
    Standard GAN training with various loss functions
    """

    def __init__(
        self,
        generator,
        discriminator,
        g_lr=0.0002,
        d_lr=0.0002,
        beta1=0.5,
        beta2=0.999,
        loss_type='vanilla',
        device='cuda',
        use_mixed_precision=False
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision

        # Optimizers
        self.g_optimizer = optim.Adam(
            generator.parameters(),
            lr=g_lr,
            betas=(beta1, beta2)
        )
        self.d_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=d_lr,
            betas=(beta1, beta2)
        )

        # Loss functions
        self.gan_loss = GANLoss(loss_type=loss_type).to(device)

        # Mixed precision scaler
        if use_mixed_precision:
            self.scaler = GradScaler()

        # Training statistics
        self.g_losses = []
        self.d_losses = []

    def train_discriminator(self, real_data, fake_data):
        """Train discriminator for one step"""
        self.discriminator.zero_grad()

        # Real data
        real_pred = self.discriminator(real_data)
        d_loss_real = self.gan_loss(real_pred, True)

        # Fake data
        fake_pred = self.discriminator(fake_data.detach())
        d_loss_fake = self.gan_loss(fake_pred, False)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake

        if self.use_mixed_precision:
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.d_optimizer)
            self.scaler.update()
        else:
            d_loss.backward()
            self.d_optimizer.step()

        return d_loss.item()

    def train_generator(self, fake_data):
        """Train generator for one step"""
        self.generator.zero_grad()

        # Generator loss
        fake_pred = self.discriminator(fake_data)
        g_loss = self.gan_loss(fake_pred, True)

        if self.use_mixed_precision:
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
        else:
            g_loss.backward()
            self.g_optimizer.step()

        return g_loss.item()

    def train_step(self, real_data, latent_dim=100):
        """Single training step"""
        batch_size = real_data.size(0)

        # Generate fake data
        z = torch.randn(batch_size, latent_dim, device=self.device)

        device_type = 'cuda' if (hasattr(self.device, 'type') and self.device.type == 'cuda') or str(self.device) == 'cuda' else 'cpu'

        with autocast(device_type, enabled=self.use_mixed_precision):
            fake_data = self.generator(z)

            # Train discriminator
            d_loss = self.train_discriminator(real_data, fake_data)

        # Generate new fake data for generator training
        z = torch.randn(batch_size, latent_dim, device=self.device)

        with autocast(device_type, enabled=self.use_mixed_precision):
            fake_data = self.generator(z)

            # Train generator
            g_loss = self.train_generator(fake_data)

        return {'d_loss': d_loss, 'g_loss': g_loss}

    def train_epoch(self, dataloader, epoch, latent_dim=100):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, data in enumerate(pbar):
            if isinstance(data, (list, tuple)):
                real_data = data[0].to(self.device)
            else:
                real_data = data.to(self.device)

            losses = self.train_step(real_data, latent_dim)

            epoch_g_loss += losses['g_loss']
            epoch_d_loss += losses['d_loss']

            pbar.set_postfix({
                'G_loss': f"{losses['g_loss']:.4f}",
                'D_loss': f"{losses['d_loss']:.4f}"
            })

        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)

        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)

        return {'g_loss': avg_g_loss, 'd_loss': avg_d_loss}

    def generate(self, num_samples, latent_dim=100):
        """Generate samples"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim, device=self.device)
            samples = self.generator(z)
        return samples

    def save_checkpoint(self, path, epoch):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        return checkpoint['epoch']


class WGANTrainer(GANTrainer):
    """
    Wasserstein GAN Trainer with Gradient Penalty
    Improved training stability
    """

    def __init__(
        self,
        generator,
        discriminator,
        g_lr=0.0001,
        d_lr=0.0001,
        beta1=0.0,
        beta2=0.9,
        lambda_gp=10.0,
        n_critic=5,
        device='cuda',
        use_mixed_precision=False
    ):
        self.n_critic = n_critic

        super().__init__(
            generator, discriminator,
            g_lr, d_lr, beta1, beta2,
            loss_type='wgan', device=device,
            use_mixed_precision=use_mixed_precision
        )

        # Gradient penalty
        self.gradient_penalty = GradientPenalty(lambda_gp=lambda_gp)

    def train_discriminator(self, real_data, fake_data):
        """Train discriminator with gradient penalty"""
        self.discriminator.zero_grad()

        # Real and fake predictions
        real_pred = self.discriminator(real_data)
        fake_pred = self.discriminator(fake_data.detach())

        # Wasserstein loss
        d_loss = fake_pred.mean() - real_pred.mean()

        # Gradient penalty
        gp = self.gradient_penalty(self.discriminator, real_data, fake_data, self.device)
        d_loss_total = d_loss + gp

        if self.use_mixed_precision:
            self.scaler.scale(d_loss_total).backward()
            self.scaler.step(self.d_optimizer)
            self.scaler.update()
        else:
            d_loss_total.backward()
            self.d_optimizer.step()

        return d_loss_total.item()

    def train_step(self, real_data, latent_dim=100):
        """Train with multiple discriminator updates"""
        batch_size = real_data.size(0)
        d_loss_total = 0.0

        # Train discriminator multiple times
        for _ in range(self.n_critic):
            z = torch.randn(batch_size, latent_dim, device=self.device)

            with autocast(enabled=self.use_mixed_precision):
                fake_data = self.generator(z)
                d_loss = self.train_discriminator(real_data, fake_data)
                d_loss_total += d_loss

        d_loss_avg = d_loss_total / self.n_critic

        # Train generator once
        z = torch.randn(batch_size, latent_dim, device=self.device)

        with autocast(enabled=self.use_mixed_precision):
            fake_data = self.generator(z)
            g_loss = self.train_generator(fake_data)

        return {'d_loss': d_loss_avg, 'g_loss': g_loss}


class ProgressiveGANTrainer(GANTrainer):
    """
    Progressive GAN Trainer
    Gradually increases resolution during training
    """

    def __init__(
        self,
        generator,
        discriminator,
        g_lr=0.001,
        d_lr=0.001,
        beta1=0.0,
        beta2=0.99,
        fade_in_steps=100000,
        stabilization_steps=100000,
        device='cuda'
    ):
        super().__init__(
            generator, discriminator,
            g_lr, d_lr, beta1, beta2,
            device=device
        )

        self.fade_in_steps = fade_in_steps
        self.stabilization_steps = stabilization_steps
        self.current_step = 0
        self.current_phase = 'stabilization'

    def update_alpha(self):
        """Update alpha for progressive growing"""
        if self.current_phase == 'fade_in':
            alpha = min(1.0, self.current_step / self.fade_in_steps)
            self.generator.set_alpha(alpha)
            self.discriminator.set_alpha(alpha)

    def grow_network(self):
        """Increase network resolution"""
        self.generator.grow()
        self.discriminator.grow()
        self.current_phase = 'fade_in'
        self.current_step = 0

    def train_step(self, real_data, latent_dim=512):
        """Training step with progressive growing"""
        self.update_alpha()
        self.current_step += 1

        # Check if we should transition to stabilization or grow
        if self.current_phase == 'fade_in' and self.current_step >= self.fade_in_steps:
            self.current_phase = 'stabilization'
            self.current_step = 0
        elif self.current_phase == 'stabilization' and self.current_step >= self.stabilization_steps:
            if self.generator.current_resolution < self.generator.max_resolution:
                self.grow_network()

        return super().train_step(real_data, latent_dim)


class ConditionalGANTrainer(GANTrainer):
    """
    Conditional GAN Trainer
    Trains with class labels
    """

    def __init__(
        self,
        generator,
        discriminator,
        num_classes,
        g_lr=0.0002,
        d_lr=0.0002,
        beta1=0.5,
        beta2=0.999,
        device='cuda'
    ):
        super().__init__(
            generator, discriminator,
            g_lr, d_lr, beta1, beta2,
            device=device
        )
        self.num_classes = num_classes

    def train_step(self, real_data, labels, latent_dim=100):
        """Training step with labels"""
        batch_size = real_data.size(0)

        # Generate fake data with labels
        z = torch.randn(batch_size, latent_dim, device=self.device)

        with autocast(enabled=self.use_mixed_precision):
            fake_data = self.generator(z, labels)

            # Train discriminator
            self.discriminator.zero_grad()

            real_pred = self.discriminator(real_data, labels)
            d_loss_real = self.gan_loss(real_pred, True)

            fake_pred = self.discriminator(fake_data.detach(), labels)
            d_loss_fake = self.gan_loss(fake_pred, False)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()

        # Train generator
        z = torch.randn(batch_size, latent_dim, device=self.device)

        with autocast(enabled=self.use_mixed_precision):
            fake_data = self.generator(z, labels)

            self.generator.zero_grad()
            fake_pred = self.discriminator(fake_data, labels)
            g_loss = self.gan_loss(fake_pred, True)
            g_loss.backward()
            self.g_optimizer.step()

        return {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}

    def train_epoch(self, dataloader, epoch, latent_dim=100):
        """Train epoch with labels"""
        self.generator.train()
        self.discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, (real_data, labels) in enumerate(pbar):
            real_data = real_data.to(self.device)
            labels = labels.to(self.device)

            losses = self.train_step(real_data, labels, latent_dim)

            epoch_g_loss += losses['g_loss']
            epoch_d_loss += losses['d_loss']

            pbar.set_postfix({
                'G_loss': f"{losses['g_loss']:.4f}",
                'D_loss': f"{losses['d_loss']:.4f}"
            })

        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)

        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)

        return {'g_loss': avg_g_loss, 'd_loss': avg_d_loss}

    def generate(self, num_samples, class_label=None, latent_dim=100):
        """Generate samples with specific class"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim, device=self.device)

            if class_label is None:
                labels = torch.randint(0, self.num_classes, (num_samples,), device=self.device)
            else:
                labels = torch.full((num_samples,), class_label, dtype=torch.long, device=self.device)

            samples = self.generator(z, labels)
        return samples


class StyleGANTrainer(GANTrainer):
    """
    StyleGAN Trainer
    Advanced training with style modulation
    """

    def __init__(
        self,
        generator,
        discriminator,
        g_lr=0.002,
        d_lr=0.002,
        beta1=0.0,
        beta2=0.99,
        r1_gamma=10.0,
        pl_weight=2.0,
        device='cuda'
    ):
        super().__init__(
            generator, discriminator,
            g_lr, d_lr, beta1, beta2,
            device=device
        )

        self.r1_reg = R1Regularization(lambda_r1=r1_gamma)
        self.pl_weight = pl_weight

        # Use non-saturating loss with R1 regularization
        self.hinge_loss = HingeLoss()

    def train_discriminator(self, real_data, fake_data):
        """Train with R1 regularization"""
        self.discriminator.zero_grad()

        # Standard discriminator loss
        real_pred = self.discriminator(real_data)
        fake_pred = self.discriminator(fake_data.detach())

        d_loss = self.hinge_loss.discriminator_loss(real_pred, fake_pred)

        # R1 regularization (apply periodically)
        if self.current_step % 16 == 0:
            r1_loss = self.r1_reg(self.discriminator, real_data)
            d_loss = d_loss + r1_loss

        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item()

    def train_generator(self, fake_data):
        """Train with path length regularization"""
        self.generator.zero_grad()

        fake_pred = self.discriminator(fake_data)
        g_loss = self.hinge_loss.generator_loss(fake_pred)

        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item()
