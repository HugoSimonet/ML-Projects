"""
Test Installation Script
Verifies that all components are properly installed and importable
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all module imports"""
    print("Testing imports...")

    try:
        # Preprocessing
        from preprocessing import MedicalImageProcessor, DICOMProcessor
        print("✓ Preprocessing modules imported successfully")

        # Data
        from data import MedicalImageDataset, DICOMDataset, SegmentationDataset
        print("✓ Data modules imported successfully")

        # Models
        from models import (
            create_medical_classifier,
            create_segmentation_model,
            MCDropoutModel,
            DeepEnsemble,
            GradCAM
        )
        print("✓ Model modules imported successfully")

        # Training
        from training import MedicalTrainer, FocalLoss, DiceLoss
        print("✓ Training modules imported successfully")

        # Evaluation
        from evaluation import MedicalMetrics
        print("✓ Evaluation modules imported successfully")

        # Visualization
        from visualization import MedicalVisualizer
        print("✓ Visualization modules imported successfully")

        # Utils
        from utils import Logger, CheckpointManager, get_device
        print("✓ Utility modules imported successfully")

        print("\n✓ All imports successful!")
        return True

    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        return False


def test_dependencies():
    """Test required dependencies"""
    print("\nTesting dependencies...")

    dependencies = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'cv2',
        'pydicom',
        'albumentations',
        'timm'
    ]

    missing = []

    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} - NOT FOUND")
            missing.append(dep)

    if missing:
        print(f"\n✗ Missing dependencies: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠ GPU not available - will use CPU")
            return True

    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")

    try:
        import torch
        from models import create_medical_classifier

        # Create a simple model
        model = create_medical_classifier(
            architecture='resnet18',
            num_classes=2,
            in_channels=1,
            pretrained=False
        )

        # Test forward pass
        x = torch.randn(2, 1, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 2), "Output shape mismatch"

        print("✓ Model creation and forward pass successful")
        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Medical CV Diagnosis System - Installation Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("Dependency Test", test_dependencies()))
    results.append(("GPU Test", test_gpu()))
    results.append(("Functionality Test", test_basic_functionality()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ Installation successful! System is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
