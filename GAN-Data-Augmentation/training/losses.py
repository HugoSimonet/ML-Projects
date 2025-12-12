"""
Loss Functions for GAN Training
Includes various GAN losses, perceptual losses, and regularization terms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GANLoss(nn.Module):
    """
    Standard GAN Loss
    Supports multiple loss types: vanilla, lsgan, wgan
    """

    def __init__(self, loss_type='vanilla', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.loss_type = loss_type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if loss_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif loss_type == 'wgan':
            self.loss = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        if self.loss_type == 'wgan':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)


class WassersteinLoss(nn.Module):
    """Wasserstein GAN Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target_is_real):
        if target_is_real:
            return -prediction.mean()
        else:
            return prediction.mean()


class GradientPenalty(nn.Module):
    """
    Gradient Penalty for WGAN-GP
    Enforces Lipschitz constraint
    """

    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def forward(self, discriminator, real_data, fake_data, device):
        batch_size = real_data.size(0)

        # Random interpolation between real and fake
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_(True)

        # Get discriminator output
        d_interpolates = discriminator(interpolates)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Flatten gradients
        gradients = gradients.view(batch_size, -1)

        # Compute gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

        return gradient_penalty


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG features
    Measures similarity in feature space
    """

    def __init__(self, layers=None, weights=None):
        super().__init__()

        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

        if weights is None:
            weights = [1.0 / len(layers)] * len(layers)

        self.layers = layers
        self.weights = weights

        # Load pre-trained VGG
        vgg = models.vgg16(pretrained=True).features
        self.vgg_layers = nn.ModuleDict()

        layer_names = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'
        }

        for name, layer in vgg.named_children():
            self.vgg_layers[name] = layer
            if name in layer_names:
                self.vgg_layers[layer_names[name]] = layer

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def extract_features(self, x):
        x = self.normalize(x)
        features = {}
        for name, layer in self.vgg_layers.items():
            x = layer(x)
            if name in self.layers:
                features[name] = x
        return features

    def forward(self, pred, target):
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        loss = 0.0
        for layer, weight in zip(self.layers, self.weights):
            loss += weight * self.criterion(pred_features[layer], target_features[layer])

        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss
    Matches discriminator feature statistics
    """

    def __init__(self, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.criterion = nn.L1Loss()

    def forward(self, real_features, fake_features):
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += self.criterion(fake_feat.mean(dim=0), real_feat.mean(dim=0))
        return loss / len(real_features)


class SpectralNormLoss(nn.Module):
    """Spectral Norm Regularization"""

    def __init__(self, lambda_sn=1.0):
        super().__init__()
        self.lambda_sn = lambda_sn

    def forward(self, model):
        loss = 0.0
        for module in model.modules():
            if hasattr(module, 'weight_u'):
                weight = module.weight
                u = module.weight_u
                v = module.weight_v

                weight_mat = weight.view(weight.size(0), -1)
                sigma = torch.dot(u, torch.mv(weight_mat, v))
                loss += sigma

        return loss * self.lambda_sn


class CycleLoss(nn.Module):
    """
    Cycle Consistency Loss for CycleGAN
    Ensures forward and backward transformations are consistent
    """

    def __init__(self, lambda_cycle=10.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.criterion = nn.L1Loss()

    def forward(self, real, reconstructed):
        return self.criterion(reconstructed, real) * self.lambda_cycle


class IdentityLoss(nn.Module):
    """
    Identity Loss for CycleGAN
    Encourages generators to preserve color composition
    """

    def __init__(self, lambda_identity=5.0):
        super().__init__()
        self.lambda_identity = lambda_identity
        self.criterion = nn.L1Loss()

    def forward(self, real, same):
        return self.criterion(same, real) * self.lambda_identity


class DiversityLoss(nn.Module):
    """
    Diversity Loss to prevent mode collapse
    Encourages different outputs for different inputs
    """

    def __init__(self, lambda_diversity=1.0):
        super().__init__()
        self.lambda_diversity = lambda_diversity

    def forward(self, z1, z2, x1, x2):
        # Cosine similarity in latent space
        z_sim = F.cosine_similarity(z1, z2, dim=1).mean()

        # L1 distance in output space
        x_dist = torch.abs(x1 - x2).mean()

        # Encourage diversity: maximize output distance relative to latent similarity
        loss = -torch.log(x_dist + 1e-8) * z_sim

        return loss * self.lambda_diversity


class HingeLoss(nn.Module):
    """
    Hinge Loss for GAN training
    Used in Spectral Normalization GAN
    """

    def __init__(self):
        super().__init__()

    def discriminator_loss(self, real_pred, fake_pred):
        real_loss = F.relu(1.0 - real_pred).mean()
        fake_loss = F.relu(1.0 + fake_pred).mean()
        return real_loss + fake_loss

    def generator_loss(self, fake_pred):
        return -fake_pred.mean()


class R1Regularization(nn.Module):
    """
    R1 Regularization
    Penalizes discriminator gradients
    """

    def __init__(self, lambda_r1=10.0):
        super().__init__()
        self.lambda_r1 = lambda_r1

    def forward(self, discriminator, real_data):
        real_data.requires_grad_(True)
        real_pred = discriminator(real_data)

        gradients = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True
        )[0]

        r1_penalty = gradients.pow(2).view(gradients.size(0), -1).sum(1).mean()

        return r1_penalty * self.lambda_r1 / 2


class PathLengthRegularization(nn.Module):
    """
    Path Length Regularization for StyleGAN
    Encourages smooth latent space
    """

    def __init__(self, pl_decay=0.01, pl_weight=2.0):
        super().__init__()
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.register_buffer('pl_mean', torch.zeros(1))

    def forward(self, fake_img, latents):
        pl_noise = torch.randn_like(fake_img) / np.sqrt(fake_img.shape[2] * fake_img.shape[3])

        outputs = (fake_img * pl_noise).sum()

        pl_grads = torch.autograd.grad(
            outputs=outputs,
            inputs=latents,
            create_graph=True,
            retain_graph=True
        )[0]

        pl_lengths = torch.sqrt(pl_grads.pow(2).sum(dim=1).mean())

        # Update moving average
        self.pl_mean.copy_(
            self.pl_decay * self.pl_mean + (1 - self.pl_decay) * pl_lengths.mean()
        )

        pl_penalty = (pl_lengths - self.pl_mean).pow(2).mean()

        return pl_penalty * self.pl_weight
