"""
Multi-Modal Vision-Language Model (VLM)
A comprehensive implementation supporting image captioning, VQA, and cross-modal retrieval
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from models import VLMModel
from inference import ImageCaptioner, VQASystem, RetrievalSystem
from training import Trainer, ContrastiveTrainer
from evaluation import Evaluator
from data import create_dataloaders, get_tokenizer

__all__ = [
    'VLMModel',
    'ImageCaptioner',
    'VQASystem',
    'RetrievalSystem',
    'Trainer',
    'ContrastiveTrainer',
    'Evaluator',
    'create_dataloaders',
    'get_tokenizer'
]
