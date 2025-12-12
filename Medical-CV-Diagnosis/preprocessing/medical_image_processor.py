"""
Medical Image Processor
Comprehensive medical image preprocessing and augmentation
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union
from PIL import Image
import cv2
from scipy import ndimage
from skimage import exposure, filters
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageNormalizer:
    """Medical image normalization techniques"""

    @staticmethod
    def min_max_normalize(image: np.ndarray, clip_percentile: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Min-max normalization with optional percentile clipping

        Args:
            image: Input image
            clip_percentile: Optional percentile range for clipping (e.g., (1, 99))

        Returns:
            Normalized image in range [0, 1]
        """
        if clip_percentile is not None:
            p_low, p_high = clip_percentile
            v_min, v_max = np.percentile(image, (p_low, p_high))
            image = np.clip(image, v_min, v_max)
        else:
            v_min, v_max = image.min(), image.max()

        if v_max - v_min > 0:
            return (image - v_min) / (v_max - v_min)
        return image

    @staticmethod
    def z_score_normalize(image: np.ndarray, clip_std: Optional[float] = None) -> np.ndarray:
        """
        Z-score (standardization) normalization

        Args:
            image: Input image
            clip_std: Optional standard deviation clipping value

        Returns:
            Normalized image with zero mean and unit variance
        """
        mean = np.mean(image)
        std = np.std(image)

        if std > 0:
            normalized = (image - mean) / std

            if clip_std is not None:
                normalized = np.clip(normalized, -clip_std, clip_std)

            return normalized
        return image

    @staticmethod
    def windowing(image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
        """
        Apply CT/MRI windowing (level/width)

        Args:
            image: Input image
            window_center: Window center (level)
            window_width: Window width

        Returns:
            Windowed image
        """
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2

        windowed = np.clip(image, window_min, window_max)
        return ImageNormalizer.min_max_normalize(windowed)

    @staticmethod
    def adaptive_histogram_equalization(image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

        Args:
            image: Input image
            clip_limit: Clipping limit for contrast

        Returns:
            Equalized image
        """
        # Convert to uint8 for CLAHE
        image_normalized = ImageNormalizer.min_max_normalize(image)
        image_uint8 = (image_normalized * 255).astype(np.uint8)

        # Apply CLAHE
        equalized = exposure.equalize_adapthist(image_uint8, clip_limit=clip_limit)
        return equalized


class MedicalImageProcessor:
    """
    Comprehensive medical image preprocessing pipeline
    Handles various medical imaging modalities (CT, MRI, X-Ray, etc.)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_method: str = 'min_max',
        clip_percentile: Optional[Tuple[float, float]] = (1, 99),
        apply_clahe: bool = False,
        window_settings: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize medical image processor

        Args:
            target_size: Target image size (height, width)
            normalize_method: Normalization method ('min_max', 'z_score', 'window')
            clip_percentile: Percentile clipping for min_max normalization
            apply_clahe: Whether to apply CLAHE
            window_settings: Dictionary of window settings for different modalities
        """
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.clip_percentile = clip_percentile
        self.apply_clahe = apply_clahe
        self.window_settings = window_settings or {}

        self.normalizer = ImageNormalizer()

    def preprocess(
        self,
        image: np.ndarray,
        modality: Optional[str] = None,
        mask: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Preprocess medical image

        Args:
            image: Input image (2D or 3D)
            modality: Image modality (CT, MRI, XRAY, etc.)
            mask: Optional segmentation mask

        Returns:
            Preprocessed image, and mask if provided
        """
        # Handle 3D images (take middle slice or process all)
        if image.ndim == 3:
            # For 3D, take middle slice as default
            image = image[image.shape[0] // 2]

        # Ensure 2D
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.ndim}D")

        # Apply modality-specific windowing
        if modality and modality.upper() in self.window_settings:
            window_center, window_width = self.window_settings[modality.upper()]
            image = self.normalizer.windowing(image, window_center, window_width)
        else:
            # Normalize based on method
            if self.normalize_method == 'min_max':
                image = self.normalizer.min_max_normalize(image, self.clip_percentile)
            elif self.normalize_method == 'z_score':
                image = self.normalizer.z_score_normalize(image, clip_std=3.0)
            else:
                image = self.normalizer.min_max_normalize(image)

        # Apply CLAHE if requested
        if self.apply_clahe:
            image = self.normalizer.adaptive_histogram_equalization(image)

        # Resize to target size
        image_resized = cv2.resize(
            image,
            self.target_size[::-1],  # OpenCV uses (width, height)
            interpolation=cv2.INTER_LINEAR
        )

        # Process mask if provided
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[mask.shape[0] // 2]

            mask_resized = cv2.resize(
                mask.astype(np.float32),
                self.target_size[::-1],
                interpolation=cv2.INTER_NEAREST
            )

            return image_resized, mask_resized

        return image_resized

    def denoise(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Denoise medical image

        Args:
            image: Input image
            method: Denoising method ('bilateral', 'gaussian', 'median', 'nlm')

        Returns:
            Denoised image
        """
        # Convert to uint8 for some methods
        image_uint8 = (self.normalizer.min_max_normalize(image) * 255).astype(np.uint8)

        if method == 'bilateral':
            denoised = cv2.bilateralFilter(image_uint8, d=9, sigmaColor=75, sigmaSpace=75)
        elif method == 'gaussian':
            denoised = cv2.GaussianBlur(image_uint8, (5, 5), 0)
        elif method == 'median':
            denoised = cv2.medianBlur(image_uint8, 5)
        elif method == 'nlm':
            denoised = cv2.fastNlMeansDenoising(image_uint8, h=10)
        else:
            raise ValueError(f"Unknown denoising method: {method}")

        return denoised.astype(np.float32) / 255.0

    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance edges in medical image

        Args:
            image: Input image

        Returns:
            Edge-enhanced image
        """
        # Apply unsharp masking
        blurred = ndimage.gaussian_filter(image, sigma=1.0)
        sharpened = image + 0.5 * (image - blurred)
        return np.clip(sharpened, 0, 1)

    def remove_background(
        self,
        image: np.ndarray,
        threshold_method: str = 'otsu'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove background from medical image

        Args:
            image: Input image
            threshold_method: Thresholding method ('otsu', 'adaptive')

        Returns:
            Foreground-only image and binary mask
        """
        image_uint8 = (self.normalizer.min_max_normalize(image) * 255).astype(np.uint8)

        if threshold_method == 'otsu':
            _, binary = cv2.threshold(
                image_uint8,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                image_uint8,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Apply mask to image
        mask = binary.astype(np.float32) / 255.0
        foreground = image * mask

        return foreground, mask


class MedicalAugmentation:
    """
    Medical-specific data augmentation
    Carefully designed to preserve medical validity
    """

    @staticmethod
    def get_training_augmentation(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
        """
        Get training augmentation pipeline

        Args:
            image_size: Target image size

        Returns:
            Albumentations composition
        """
        return A.Compose([
            # Geometric transformations (small ranges for medical images)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),

            # Elastic deformation (simulates tissue deformation)
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ),

            # Grid distortion
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),

            # Brightness and contrast (small changes)
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),

            # Gamma adjustment
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),

            # Gaussian noise (simulates acquisition noise)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

            # Blur (simulates motion or acquisition blur)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.2),

            # Final resize
            A.Resize(height=image_size[0], width=image_size[1]),

            # Normalize to [0, 1]
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),

            # Convert to tensor
            ToTensorV2()
        ])

    @staticmethod
    def get_validation_augmentation(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
        """
        Get validation augmentation pipeline (minimal transforms)

        Args:
            image_size: Target image size

        Returns:
            Albumentations composition
        """
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),
            ToTensorV2()
        ])

    @staticmethod
    def get_segmentation_augmentation(
        image_size: Tuple[int, int] = (256, 256)
    ) -> A.Compose:
        """
        Get segmentation-specific augmentation (applies same transforms to image and mask)

        Args:
            image_size: Target image size

        Returns:
            Albumentations composition
        """
        return A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),

            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ),

            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),

            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),

            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),
            ToTensorV2()
        ])


# Preset window settings for different modalities
WINDOW_PRESETS = {
    'CT_BRAIN': (40, 80),
    'CT_LUNG': (-600, 1500),
    'CT_LIVER': (30, 150),
    'CT_BONE': (400, 1800),
    'CT_ABDOMEN': (50, 350),
    'CT_MEDIASTINUM': (50, 350),
}


def create_medical_processor(
    modality: str = 'CT',
    target_size: Tuple[int, int] = (224, 224),
    **kwargs
) -> MedicalImageProcessor:
    """
    Factory function to create medical image processor with preset configurations

    Args:
        modality: Medical imaging modality
        target_size: Target image size
        **kwargs: Additional arguments for MedicalImageProcessor

    Returns:
        Configured MedicalImageProcessor
    """
    # Set window settings based on modality
    window_settings = {}
    if modality.upper().startswith('CT'):
        window_settings = WINDOW_PRESETS

    return MedicalImageProcessor(
        target_size=target_size,
        window_settings=window_settings,
        **kwargs
    )
