"""
Medical Deep Learning Models Module
CNN architectures, segmentation models, uncertainty quantification, and explainability
"""

from .medical_classifiers import (
    MedicalResNet,
    MedicalDenseNet,
    MedicalEfficientNet,
    MedicalViT,
    create_medical_classifier
)
from .medical_segmentation import (
    UNet,
    UNetPlusPlus,
    AttentionUNet,
    create_segmentation_model
)
from .uncertainty_quantification import (
    MCDropoutModel,
    DeepEnsemble,
    UncertaintyEstimator
)
from .explainability import (
    GradCAM,
    GradCAMPlusPlus,
    SHAP_Explainer,
    LIME_Explainer,
    IntegratedGradients
)

__all__ = [
    'MedicalResNet',
    'MedicalDenseNet',
    'MedicalEfficientNet',
    'MedicalViT',
    'create_medical_classifier',
    'UNet',
    'UNetPlusPlus',
    'AttentionUNet',
    'create_segmentation_model',
    'MCDropoutModel',
    'DeepEnsemble',
    'UncertaintyEstimator',
    'GradCAM',
    'GradCAMPlusPlus',
    'SHAP_Explainer',
    'LIME_Explainer',
    'IntegratedGradients'
]
