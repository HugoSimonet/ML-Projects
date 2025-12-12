"""
DICOM Processor
Handles DICOM file reading, metadata extraction, and conversion
"""

import numpy as np
import pydicom
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from typing import Dict, Optional, List, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DICOMMetadataExtractor:
    """Extract and parse DICOM metadata"""

    # Standard DICOM tags to extract
    STANDARD_TAGS = {
        'PatientID': (0x0010, 0x0020),
        'PatientName': (0x0010, 0x0010),
        'PatientBirthDate': (0x0010, 0x0030),
        'PatientSex': (0x0010, 0x0040),
        'StudyDate': (0x0008, 0x0020),
        'StudyTime': (0x0008, 0x0030),
        'StudyDescription': (0x0008, 0x1030),
        'SeriesDescription': (0x0008, 0x103e),
        'Modality': (0x0008, 0x0060),
        'Manufacturer': (0x0008, 0x0070),
        'InstitutionName': (0x0008, 0x0080),
        'SliceThickness': (0x0018, 0x0050),
        'PixelSpacing': (0x0028, 0x0030),
        'Rows': (0x0028, 0x0010),
        'Columns': (0x0028, 0x0011),
        'WindowCenter': (0x0028, 0x1050),
        'WindowWidth': (0x0028, 0x1051),
        'RescaleIntercept': (0x0028, 0x1052),
        'RescaleSlope': (0x0028, 0x1053),
    }

    @staticmethod
    def extract_metadata(dicom_data: pydicom.Dataset) -> Dict:
        """
        Extract standard metadata from DICOM dataset

        Args:
            dicom_data: PyDICOM dataset

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        for tag_name, tag_address in DICOMMetadataExtractor.STANDARD_TAGS.items():
            try:
                value = dicom_data[tag_address].value
                # Convert to JSON-serializable format
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                elif isinstance(value, pydicom.multival.MultiValue):
                    value = list(value)
                elif hasattr(value, 'decode'):
                    value = str(value)

                metadata[tag_name] = value
            except (KeyError, AttributeError):
                metadata[tag_name] = None

        return metadata

    @staticmethod
    def extract_patient_info(dicom_data: pydicom.Dataset) -> Dict:
        """
        Extract patient information

        Args:
            dicom_data: PyDICOM dataset

        Returns:
            Dictionary of patient information
        """
        patient_info = {}

        try:
            patient_info['id'] = str(dicom_data.PatientID) if hasattr(dicom_data, 'PatientID') else None
            patient_info['name'] = str(dicom_data.PatientName) if hasattr(dicom_data, 'PatientName') else None
            patient_info['birth_date'] = str(dicom_data.PatientBirthDate) if hasattr(dicom_data, 'PatientBirthDate') else None
            patient_info['sex'] = str(dicom_data.PatientSex) if hasattr(dicom_data, 'PatientSex') else None
            patient_info['age'] = str(dicom_data.PatientAge) if hasattr(dicom_data, 'PatientAge') else None
        except Exception as e:
            logger.warning(f"Error extracting patient info: {e}")

        return patient_info

    @staticmethod
    def extract_study_info(dicom_data: pydicom.Dataset) -> Dict:
        """
        Extract study information

        Args:
            dicom_data: PyDICOM dataset

        Returns:
            Dictionary of study information
        """
        study_info = {}

        try:
            study_info['date'] = str(dicom_data.StudyDate) if hasattr(dicom_data, 'StudyDate') else None
            study_info['time'] = str(dicom_data.StudyTime) if hasattr(dicom_data, 'StudyTime') else None
            study_info['description'] = str(dicom_data.StudyDescription) if hasattr(dicom_data, 'StudyDescription') else None
            study_info['id'] = str(dicom_data.StudyInstanceUID) if hasattr(dicom_data, 'StudyInstanceUID') else None
            study_info['modality'] = str(dicom_data.Modality) if hasattr(dicom_data, 'Modality') else None
        except Exception as e:
            logger.warning(f"Error extracting study info: {e}")

        return study_info

    @staticmethod
    def extract_image_info(dicom_data: pydicom.Dataset) -> Dict:
        """
        Extract image acquisition information

        Args:
            dicom_data: PyDICOM dataset

        Returns:
            Dictionary of image information
        """
        image_info = {}

        try:
            image_info['rows'] = int(dicom_data.Rows) if hasattr(dicom_data, 'Rows') else None
            image_info['columns'] = int(dicom_data.Columns) if hasattr(dicom_data, 'Columns') else None
            image_info['pixel_spacing'] = list(dicom_data.PixelSpacing) if hasattr(dicom_data, 'PixelSpacing') else None
            image_info['slice_thickness'] = float(dicom_data.SliceThickness) if hasattr(dicom_data, 'SliceThickness') else None
            image_info['bits_allocated'] = int(dicom_data.BitsAllocated) if hasattr(dicom_data, 'BitsAllocated') else None
            image_info['bits_stored'] = int(dicom_data.BitsStored) if hasattr(dicom_data, 'BitsStored') else None
        except Exception as e:
            logger.warning(f"Error extracting image info: {e}")

        return image_info


class DICOMConverter:
    """Convert DICOM pixel data to numpy arrays with proper scaling"""

    @staticmethod
    def get_pixel_array(dicom_data: pydicom.Dataset) -> np.ndarray:
        """
        Get pixel array from DICOM with proper type conversion

        Args:
            dicom_data: PyDICOM dataset

        Returns:
            Pixel array as numpy array
        """
        try:
            pixel_array = dicom_data.pixel_array
            return pixel_array
        except Exception as e:
            logger.error(f"Error reading pixel array: {e}")
            raise

    @staticmethod
    def apply_modality_lut(pixel_array: np.ndarray, dicom_data: pydicom.Dataset) -> np.ndarray:
        """
        Apply modality LUT transformation (rescale slope and intercept)

        Args:
            pixel_array: Raw pixel array
            dicom_data: PyDICOM dataset containing rescale parameters

        Returns:
            Rescaled pixel array in Hounsfield Units (for CT) or appropriate units
        """
        intercept = getattr(dicom_data, 'RescaleIntercept', 0)
        slope = getattr(dicom_data, 'RescaleSlope', 1)

        # Apply rescale: HU = pixel_value * slope + intercept
        if slope != 1 or intercept != 0:
            pixel_array = pixel_array.astype(np.float32)
            pixel_array = pixel_array * slope + intercept

        return pixel_array

    @staticmethod
    def apply_voi_lut(
        pixel_array: np.ndarray,
        dicom_data: pydicom.Dataset,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply VOI LUT (Value of Interest - windowing)

        Args:
            pixel_array: Pixel array (after modality LUT)
            dicom_data: PyDICOM dataset
            window_center: Override window center
            window_width: Override window width

        Returns:
            Windowed pixel array normalized to [0, 1]
        """
        # Get window parameters
        if window_center is None:
            if hasattr(dicom_data, 'WindowCenter'):
                wc = dicom_data.WindowCenter
                window_center = wc[0] if isinstance(wc, (list, tuple)) else wc
            else:
                window_center = np.median(pixel_array)

        if window_width is None:
            if hasattr(dicom_data, 'WindowWidth'):
                ww = dicom_data.WindowWidth
                window_width = ww[0] if isinstance(ww, (list, tuple)) else ww
            else:
                window_width = np.percentile(pixel_array, 95) - np.percentile(pixel_array, 5)

        # Apply windowing
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2

        windowed = np.clip(pixel_array, window_min, window_max)

        # Normalize to [0, 1]
        if window_max - window_min > 0:
            windowed = (windowed - window_min) / (window_max - window_min)

        return windowed

    @staticmethod
    def convert_to_hounsfield(pixel_array: np.ndarray, dicom_data: pydicom.Dataset) -> np.ndarray:
        """
        Convert pixel array to Hounsfield Units (for CT scans)

        Args:
            pixel_array: Raw pixel array
            dicom_data: PyDICOM dataset

        Returns:
            Pixel array in Hounsfield Units
        """
        return DICOMConverter.apply_modality_lut(pixel_array, dicom_data)


class DICOMProcessor:
    """
    Main DICOM processor class
    Handles reading, processing, and converting DICOM files
    """

    def __init__(
        self,
        apply_modality_lut: bool = True,
        apply_voi_lut: bool = False,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        force_read: bool = False
    ):
        """
        Initialize DICOM processor

        Args:
            apply_modality_lut: Apply rescale slope/intercept
            apply_voi_lut: Apply windowing
            window_center: Window center for VOI LUT
            window_width: Window width for VOI LUT
            force_read: Force reading even if file appears invalid
        """
        self.apply_modality_lut = apply_modality_lut
        self.apply_voi_lut = apply_voi_lut
        self.window_center = window_center
        self.window_width = window_width
        self.force_read = force_read

        self.metadata_extractor = DICOMMetadataExtractor()
        self.converter = DICOMConverter()

    def read_dicom(self, file_path: Union[str, Path]) -> Optional[pydicom.Dataset]:
        """
        Read DICOM file

        Args:
            file_path: Path to DICOM file

        Returns:
            PyDICOM dataset or None if reading fails
        """
        try:
            dicom_data = dcmread(
                file_path,
                force=self.force_read,
                stop_before_pixels=False
            )
            return dicom_data
        except InvalidDicomError as e:
            logger.error(f"Invalid DICOM file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading DICOM file {file_path}: {e}")
            return None

    def process_dicom(
        self,
        file_path: Union[str, Path],
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Process DICOM file to numpy array

        Args:
            file_path: Path to DICOM file
            return_metadata: Whether to return metadata along with image

        Returns:
            Processed image array, and optionally metadata dictionary
        """
        # Read DICOM
        dicom_data = self.read_dicom(file_path)
        if dicom_data is None:
            raise ValueError(f"Failed to read DICOM file: {file_path}")

        # Get pixel array
        pixel_array = self.converter.get_pixel_array(dicom_data)

        # Apply modality LUT
        if self.apply_modality_lut:
            pixel_array = self.converter.apply_modality_lut(pixel_array, dicom_data)

        # Apply VOI LUT (windowing)
        if self.apply_voi_lut:
            pixel_array = self.converter.apply_voi_lut(
                pixel_array,
                dicom_data,
                self.window_center,
                self.window_width
            )

        # Extract metadata if requested
        if return_metadata:
            metadata = {
                'standard': self.metadata_extractor.extract_metadata(dicom_data),
                'patient': self.metadata_extractor.extract_patient_info(dicom_data),
                'study': self.metadata_extractor.extract_study_info(dicom_data),
                'image': self.metadata_extractor.extract_image_info(dicom_data)
            }
            return pixel_array, metadata

        return pixel_array

    def batch_process_dicom(
        self,
        file_paths: List[Union[str, Path]],
        return_metadata: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[Dict]]]:
        """
        Process multiple DICOM files

        Args:
            file_paths: List of DICOM file paths
            return_metadata: Whether to return metadata

        Returns:
            List of processed images, and optionally list of metadata
        """
        images = []
        metadata_list = []

        for file_path in file_paths:
            try:
                if return_metadata:
                    image, metadata = self.process_dicom(file_path, return_metadata=True)
                    images.append(image)
                    metadata_list.append(metadata)
                else:
                    image = self.process_dicom(file_path, return_metadata=False)
                    images.append(image)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        if return_metadata:
            return images, metadata_list
        return images

    def save_metadata_to_json(
        self,
        file_path: Union[str, Path],
        output_path: Union[str, Path]
    ):
        """
        Extract and save DICOM metadata to JSON

        Args:
            file_path: Path to DICOM file
            output_path: Path to output JSON file
        """
        _, metadata = self.process_dicom(file_path, return_metadata=True)

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Metadata saved to {output_path}")


def load_dicom_series(
    directory: Union[str, Path],
    sort_by: str = 'InstanceNumber'
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load a complete DICOM series from a directory

    Args:
        directory: Directory containing DICOM files
        sort_by: DICOM tag to sort slices by

    Returns:
        3D numpy array (slices, height, width) and list of metadata
    """
    directory = Path(directory)
    dicom_files = sorted(directory.glob('*.dcm'))

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {directory}")

    processor = DICOMProcessor()

    # Read all files
    dicom_datasets = []
    for file_path in dicom_files:
        dicom_data = processor.read_dicom(file_path)
        if dicom_data is not None:
            dicom_datasets.append((file_path, dicom_data))

    # Sort by instance number or slice location
    if sort_by == 'InstanceNumber':
        dicom_datasets.sort(key=lambda x: int(getattr(x[1], 'InstanceNumber', 0)))
    elif sort_by == 'SliceLocation':
        dicom_datasets.sort(key=lambda x: float(getattr(x[1], 'SliceLocation', 0)))

    # Process all slices
    images = []
    metadata_list = []

    for file_path, dicom_data in dicom_datasets:
        image, metadata = processor.process_dicom(file_path, return_metadata=True)
        images.append(image)
        metadata_list.append(metadata)

    # Stack into 3D array
    volume = np.stack(images, axis=0)

    return volume, metadata_list
