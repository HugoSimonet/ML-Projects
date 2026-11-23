"""
Explainability and Interpretability for Medical AI
Grad-CAM, Grad-CAM++, SHAP, LIME, and Integrated Gradients implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Callable
import cv2
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes important regions for model predictions
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM

        Args:
            model: The model to explain
            target_layer: The layer to compute gradients from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Save activations"""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Save gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap

        Args:
            input_image: Input image tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)

        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        input_image.requires_grad = True

        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        # ReLU (only positive influence)
        cam = F.relu(cam)

        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def __del__(self):
        """Remove hooks"""
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ - Improved version with better localization
    Weights gradients more carefully
    """

    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap

        Args:
            input_image: Input image tensor
            target_class: Target class index

        Returns:
            CAM heatmap
        """
        self.model.eval()
        input_image.requires_grad = True

        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Grad-CAM++ weighting
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)

        alpha = grad_2 / (2 * grad_2 + (grad_3 * activations).sum(dim=(1, 2), keepdim=True) + 1e-8)

        # ReLU on gradients
        relu_grads = F.relu(class_score.exp() * gradients)

        # Weights
        weights = (alpha * relu_grads).sum(dim=(1, 2))

        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


class IntegratedGradients:
    """
    Integrated Gradients for attribution
    Attributes prediction to input features
    """

    def __init__(self, model: nn.Module):
        """
        Initialize Integrated Gradients

        Args:
            model: The model to explain
        """
        self.model = model

    def generate_attribution(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        num_steps: int = 50
    ) -> np.ndarray:
        """
        Generate integrated gradients attribution

        Args:
            input_image: Input image tensor (1, C, H, W)
            target_class: Target class index
            baseline: Baseline image (default: zeros)
            num_steps: Number of integration steps

        Returns:
            Attribution map
        """
        self.model.eval()

        if baseline is None:
            baseline = torch.zeros_like(input_image)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, num_steps, device=input_image.device)
        interpolated_inputs = []

        for alpha in alphas:
            interpolated = baseline + alpha * (input_image - baseline)
            interpolated_inputs.append(interpolated)

        interpolated_inputs = torch.cat(interpolated_inputs, dim=0)
        interpolated_inputs.requires_grad = True

        # Forward pass
        outputs = self.model(interpolated_inputs)

        if target_class is None:
            target_class = outputs[0].argmax().item()

        # Get target scores
        target_scores = outputs[:, target_class]

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=target_scores,
            inputs=interpolated_inputs,
            grad_outputs=torch.ones_like(target_scores),
            create_graph=False
        )[0]

        # Average gradients
        avg_gradients = gradients.mean(dim=0, keepdim=True)

        # Multiply by input difference
        integrated_grads = (input_image - baseline) * avg_gradients

        # Convert to numpy
        attribution = integrated_grads[0].cpu().detach().numpy()

        # Sum across channels for visualization
        if attribution.shape[0] > 1:
            attribution = attribution.sum(axis=0)
        else:
            attribution = attribution[0]

        return attribution


class SHAP_Explainer:
    """
    SHAP (SHapley Additive exPlanations) for model explanations
    Simplified implementation for image classification
    """

    def __init__(self, model: nn.Module, background_samples: torch.Tensor):
        """
        Initialize SHAP explainer

        Args:
            model: The model to explain
            background_samples: Background dataset samples for reference
        """
        self.model = model
        self.background_samples = background_samples

    def explain(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
        num_samples: int = 100
    ) -> np.ndarray:
        """
        Generate SHAP values (simplified kernel SHAP)

        Args:
            input_image: Input image to explain
            target_class: Target class
            num_samples: Number of samples for approximation

        Returns:
            SHAP values
        """
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_image)
            if target_class is None:
                target_class = output.argmax().item()

        # Create masked versions
        shap_values = torch.zeros_like(input_image)

        # Simplified: Use sliding window masking
        # Full SHAP would require proper sampling
        mask_size = 16

        for i in range(0, input_image.shape[2], mask_size):
            for j in range(0, input_image.shape[3], mask_size):
                # Create masked version
                masked_input = input_image.clone()
                masked_input[:, :, i:i+mask_size, j:j+mask_size] = 0

                with torch.no_grad():
                    masked_output = self.model(masked_input)
                    masked_score = masked_output[0, target_class]
                    original_score = output[0, target_class]

                # Attribution
                attribution = (original_score - masked_score).abs()
                shap_values[:, :, i:i+mask_size, j:j+mask_size] = attribution

        shap_values = shap_values[0].cpu().numpy()

        if shap_values.shape[0] > 1:
            shap_values = shap_values.sum(axis=0)
        else:
            shap_values = shap_values[0]

        return shap_values


class LIME_Explainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations)
    Simplified implementation for images
    """

    def __init__(self, model: nn.Module):
        """
        Initialize LIME explainer

        Args:
            model: The model to explain
        """
        self.model = model

    def explain(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
        num_samples: int = 100,
        num_features: int = 64
    ) -> np.ndarray:
        """
        Generate LIME explanation

        Args:
            input_image: Input image
            target_class: Target class
            num_samples: Number of samples
            num_features: Number of superpixels

        Returns:
            LIME explanation map
        """
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_image)
            if target_class is None:
                target_class = output.argmax().item()

        # Create superpixels (simplified: grid)
        h, w = input_image.shape[2:]
        seg_h = h // int(np.sqrt(num_features))
        seg_w = w // int(np.sqrt(num_features))

        # Generate perturbations
        explanations = torch.zeros(h, w)

        for _ in range(num_samples):
            # Random mask
            mask = torch.rand(int(np.sqrt(num_features)), int(np.sqrt(num_features))) > 0.5
            mask = mask.float()

            # Upsample mask
            mask_upsampled = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='nearest'
            )[0, 0]

            # Apply mask
            perturbed = input_image * mask_upsampled.unsqueeze(0).unsqueeze(0).to(input_image.device)

            with torch.no_grad():
                perturbed_output = self.model(perturbed)
                perturbed_score = perturbed_output[0, target_class]
                original_score = output[0, target_class]

            # Attribution
            diff = (original_score - perturbed_score).abs()
            explanations += mask_upsampled.cpu() * diff.cpu().item()

        explanations = explanations.numpy()
        explanations = (explanations - explanations.min()) / (explanations.max() - explanations.min() + 1e-8)

        return explanations


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on original image

    Args:
        image: Original image (H, W) or (H, W, C)
        heatmap: Heatmap (H, W)
        alpha: Blending factor
        colormap: OpenCV colormap

    Returns:
        Overlayed image
    """
    # Resize heatmap to match image size
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize heatmap to [0, 255]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Normalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Blend
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlayed


def get_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """
    Get the target layer for Grad-CAM based on architecture

    Args:
        model: The model
        architecture: Architecture name

    Returns:
        Target layer
    """
    architecture = architecture.lower()

    if 'resnet' in architecture:
        return model.backbone.layer4[-1]
    elif 'densenet' in architecture:
        return model.backbone.features.denseblock4
    elif 'efficientnet' in architecture:
        return model.backbone.blocks[-1]
    elif 'vit' in architecture:
        return model.backbone.blocks[-1]
    else:
        # Try to get last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module

    raise ValueError(f"Could not find target layer for {architecture}")
