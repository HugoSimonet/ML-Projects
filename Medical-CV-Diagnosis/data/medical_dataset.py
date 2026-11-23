"""
Medical Dataset Classes
Comprehensive dataset implementations for medical imaging
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Optional, Dict, List, Callable, Union
from pathlib import Path
import pandas as pd
from PIL import Image
import cv2
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class MedicalImageDataset(Dataset):
    """
    Base medical image dataset for classification tasks
    Supports various medical imaging formats and preprocessing
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        labels_file: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        image_processor: Optional[Callable] = None,
        modality: Optional[str] = None,
        file_extension: str = '.png',
        return_path: bool = False
    ):
        """
        Initialize medical image dataset

        Args:
            data_dir: Directory containing images
            labels_file: CSV file with image_path and label columns
            transform: Albumentations or torchvision transforms
            image_processor: Medical image preprocessor
            modality: Medical imaging modality (CT, MRI, XRAY, etc.)
            file_extension: Image file extension
            return_path: Whether to return file path along with image
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_processor = image_processor
        self.modality = modality
        self.return_path = return_path

        # Load dataset
        if labels_file is not None:
            self._load_from_csv(labels_file)
        else:
            self._load_from_directory(file_extension)

        logger.info(f"Loaded {len(self)} images from {data_dir}")
        if hasattr(self, 'labels'):
            label_counts = Counter(self.labels)
            logger.info(f"Label distribution: {dict(label_counts)}")

    def _load_from_csv(self, labels_file: Union[str, Path]):
        """Load dataset from CSV file with image paths and labels"""
        df = pd.read_csv(labels_file)

        if 'image_path' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'image_path' and 'label' columns")

        self.image_paths = [self.data_dir / path for path in df['image_path']]
        self.labels = df['label'].tolist()

        # Load additional metadata if present
        self.metadata = {}
        for col in df.columns:
            if col not in ['image_path', 'label']:
                self.metadata[col] = df[col].tolist()

    def _load_from_directory(self, file_extension: str):
        """Load dataset from directory structure: data_dir/class_name/image.ext"""
        self.image_paths = []
        self.labels = []

        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if not class_dirs:
            # No class subdirectories, load all images
            self.image_paths = sorted(self.data_dir.glob(f'*{file_extension}'))
            self.labels = [0] * len(self.image_paths)
        else:
            # Load from class subdirectories
            self.class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

            for class_dir in class_dirs:
                label = self.class_to_idx[class_dir.name]
                images = sorted(class_dir.glob(f'*{file_extension}'))

                self.image_paths.extend(images)
                self.labels.extend([label] * len(images))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str]]:
        """
        Get item by index

        Args:
            idx: Index

        Returns:
            (image, label) or (image, label, path) if return_path=True
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = self._load_image(image_path)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return blank image on error
            image = np.zeros((224, 224), dtype=np.float32)

        # Apply medical preprocessing
        if self.image_processor is not None:
            image = self.image_processor.preprocess(image, modality=self.modality)

        # Ensure image is in correct format for transforms
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # Apply transforms
        if self.transform is not None:
            # Check if albumentations or torchvision
            if hasattr(self.transform, '__call__'):
                try:
                    # Try albumentations format
                    transformed = self.transform(image=image)
                    image = transformed['image']
                except:
                    # Try torchvision format
                    image = self.transform(image)

        # Ensure tensor format
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()

        # Ensure channel-first format (C, H, W)
        if image.ndim == 3 and image.shape[-1] in [1, 3]:
            image = image.permute(2, 0, 1)

        if self.return_path:
            return image, label, str(image_path)

        return image, label

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image file"""
        if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            return image.astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        if not hasattr(self, 'labels'):
            return None

        label_counts = Counter(self.labels)
        total = len(self.labels)

        # Inverse frequency weighting
        weights = torch.zeros(len(label_counts))
        for label, count in label_counts.items():
            weights[label] = total / (count * len(label_counts))

        return weights


class DICOMDataset(Dataset):
    """
    Dataset for DICOM medical images
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        labels_file: Union[str, Path],
        dicom_processor,
        transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        """
        Initialize DICOM dataset

        Args:
            data_dir: Directory containing DICOM files
            labels_file: CSV with dicom_path and label columns
            dicom_processor: DICOMProcessor instance
            transform: Transforms to apply
            return_metadata: Whether to return DICOM metadata
        """
        self.data_dir = Path(data_dir)
        self.dicom_processor = dicom_processor
        self.transform = transform
        self.return_metadata = return_metadata

        # Load labels
        df = pd.read_csv(labels_file)
        self.dicom_paths = [self.data_dir / path for path in df['dicom_path']]
        self.labels = df['label'].tolist()

        logger.info(f"Loaded {len(self)} DICOM files")

    def __len__(self) -> int:
        return len(self.dicom_paths)

    def __getitem__(self, idx: int):
        """Get DICOM image and label"""
        dicom_path = self.dicom_paths[idx]
        label = self.labels[idx]

        # Process DICOM
        try:
            if self.return_metadata:
                image, metadata = self.dicom_processor.process_dicom(
                    dicom_path,
                    return_metadata=True
                )
            else:
                image = self.dicom_processor.process_dicom(
                    dicom_path,
                    return_metadata=False
                )
        except Exception as e:
            logger.error(f"Error processing DICOM {dicom_path}: {e}")
            image = np.zeros((512, 512), dtype=np.float32)
            metadata = {} if self.return_metadata else None

        # Add channel dimension
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        if self.return_metadata:
            return image, label, metadata

        return image, label


class SegmentationDataset(Dataset):
    """
    Dataset for medical image segmentation
    Returns image and corresponding segmentation mask
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        image_processor: Optional[Callable] = None,
        mask_suffix: str = '_mask',
        num_classes: int = 2
    ):
        """
        Initialize segmentation dataset

        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing segmentation masks
            transform: Albumentations transforms (applied to both image and mask)
            image_processor: Medical image preprocessor
            mask_suffix: Suffix for mask files
            num_classes: Number of segmentation classes
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_processor = image_processor
        self.mask_suffix = mask_suffix
        self.num_classes = num_classes

        # Load image and mask paths
        self.image_paths = sorted(self.images_dir.glob('*.png'))
        self.mask_paths = []

        for img_path in self.image_paths:
            mask_name = img_path.stem + mask_suffix + img_path.suffix
            mask_path = self.masks_dir / mask_name

            if not mask_path.exists():
                logger.warning(f"Mask not found for {img_path.name}, skipping")
                continue

            self.mask_paths.append(mask_path)

        # Keep only images with corresponding masks
        self.image_paths = self.image_paths[:len(self.mask_paths)]

        logger.info(f"Loaded {len(self)} image-mask pairs")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and mask"""
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)

        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.int64)

        # Apply preprocessing
        if self.image_processor is not None:
            image, mask = self.image_processor.preprocess(image, mask=mask)

        # Add channel dimension
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # Apply transforms (must support both image and mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Ensure proper format
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return image, mask


class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal medical imaging
    Combines multiple imaging modalities (e.g., T1, T2, FLAIR MRI)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        modalities: List[str],
        labels_file: Union[str, Path],
        transform: Optional[Callable] = None,
        align_modalities: bool = True
    ):
        """
        Initialize multi-modal dataset

        Args:
            data_dir: Base directory
            modalities: List of modality names (subdirectories)
            labels_file: CSV with sample_id and label
            transform: Transforms to apply
            align_modalities: Whether to ensure spatial alignment
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.transform = transform
        self.align_modalities = align_modalities

        # Load labels
        df = pd.read_csv(labels_file)
        self.sample_ids = df['sample_id'].tolist()
        self.labels = df['label'].tolist()

        logger.info(f"Loaded {len(self)} multi-modal samples with {len(modalities)} modalities")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get multi-modal image stack and label"""
        sample_id = self.sample_ids[idx]
        label = self.labels[idx]

        # Load all modalities
        images = []
        for modality in self.modalities:
            img_path = self.data_dir / modality / f"{sample_id}.png"

            if not img_path.exists():
                logger.warning(f"Missing {modality} for sample {sample_id}")
                # Use zeros for missing modality
                images.append(np.zeros((224, 224), dtype=np.float32))
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                images.append(img.astype(np.float32))

        # Stack modalities
        multi_modal_image = np.stack(images, axis=-1)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=multi_modal_image)
            multi_modal_image = transformed['image']

        return multi_modal_image, label


class BalancedMedicalDataset(Dataset):
    """
    Wrapper for balanced sampling from imbalanced medical datasets
    """

    def __init__(self, dataset: Dataset):
        """
        Initialize balanced dataset wrapper

        Args:
            dataset: Base medical dataset
        """
        self.dataset = dataset

        # Calculate class distribution
        if hasattr(dataset, 'labels'):
            self.labels = dataset.labels
        else:
            # Extract labels by iterating (slower)
            self.labels = [dataset[i][1] for i in range(len(dataset))]

        self.class_counts = Counter(self.labels)
        self.num_classes = len(self.class_counts)

        logger.info(f"Class distribution: {dict(self.class_counts)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Create weighted random sampler for balanced training

        Returns:
            WeightedRandomSampler
        """
        # Calculate sample weights
        class_weights = {}
        total = len(self.labels)

        for label, count in self.class_counts.items():
            class_weights[label] = total / (count * self.num_classes)

        sample_weights = [class_weights[label] for label in self.labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return sampler


def create_medical_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    balanced: bool = False
) -> DataLoader:
    """
    Create medical image dataloader with optional balanced sampling

    Args:
        dataset: Medical dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        balanced: Use balanced sampling for imbalanced datasets

    Returns:
        DataLoader
    """
    sampler = None

    if balanced and hasattr(dataset, 'get_balanced_sampler'):
        sampler = dataset.get_balanced_sampler()
        shuffle = False  # Cannot shuffle with sampler

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return dataloader


def split_medical_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split medical dataset into train/val/test with stratification

    Args:
        dataset: Medical dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility

    Returns:
        Train, validation, and test datasets
    """
    from sklearn.model_selection import train_test_split

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    # Get indices and labels
    indices = list(range(len(dataset)))
    labels = dataset.labels if hasattr(dataset, 'labels') else [dataset[i][1] for i in indices]

    # Split train and temp (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_seed
    )

    # Split temp into val and test
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=random_seed
    )

    # Create subset datasets
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    logger.info(f"Split dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset
