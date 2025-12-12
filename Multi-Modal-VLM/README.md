# Multi-Modal Vision-Language Model

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Vision-language model for image captioning, visual question answering, and cross-modal retrieval.

## Overview

This project implements a multi-modal transformer model that processes both images and text. It supports image captioning, visual question answering (VQA), image-text retrieval, and zero-shot classification.

## Features

- Image captioning with transformer decoder
- Visual question answering
- Cross-modal retrieval (image-to-text, text-to-image)
- Zero-shot image classification
- Attention visualization
- CLIP-style contrastive learning

## Architecture

**Vision Encoder** - ResNet-50 or Vision Transformer (ViT) backbone for image feature extraction

**Text Encoder** - BERT or GPT-style transformer for text encoding

**Cross-Modal Fusion** - Multi-head cross-attention layers for vision-language alignment

**Task Heads** - Decoder for captioning, classifier for VQA, similarity scorer for retrieval

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 1.9+, torchvision, transformers, pillow

## Quick Start

### Image Captioning

```bash
python caption.py \
    --image path/to/image.jpg \
    --model checkpoints/caption_model.pth
```

### Visual Question Answering

```bash
python vqa.py \
    --image path/to/image.jpg \
    --question "What color is the car?" \
    --model checkpoints/vqa_model.pth
```

### Training

```bash
python train.py \
    --task captioning \
    --dataset coco \
    --vision-model vit \
    --text-model bert \
    --epochs 30
```

## Usage

```python
from models import VisionLanguageModel
from PIL import Image

# Load model
model = VisionLanguageModel.from_pretrained('checkpoints/model.pth')

# Image captioning
image = Image.open('photo.jpg')
caption = model.generate_caption(image)
print(f"Caption: {caption}")

# Visual question answering
question = "How many people are in the image?"
answer = model.answer_question(image, question)
print(f"Answer: {answer}")

# Image-text similarity
text = "A dog playing in the park"
similarity = model.compute_similarity(image, text)
print(f"Similarity: {similarity:.3f}")
```

## Training

```python
from models import VisionLanguageModel
from data import COCODataset
from torch.utils.data import DataLoader

# Create model
model = VisionLanguageModel(
    vision_backbone='vit',
    text_backbone='bert',
    hidden_dim=768,
    num_heads=12,
    num_layers=6
)

# Load data
dataset = COCODataset(root='data/coco', split='train')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(30):
    for images, captions in dataloader:
        loss = model.compute_loss(images, captions)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Datasets

- **COCO Captions**: 330K images with 5 captions each
- **VQA v2.0**: 200K images with 1.1M questions
- **Flickr30K**: 31K images with captions
- **Conceptual Captions**: 3.3M image-caption pairs

## Tasks

### Image Captioning

Generate natural language descriptions of images using transformer decoder with cross-attention to image features.

### Visual Question Answering

Answer questions about image content using classification head over fused vision-language features.

### Cross-Modal Retrieval

Find images given text queries (or vice versa) using contrastive learning and similarity scoring.

### Zero-Shot Classification

Classify images using text descriptions of categories without task-specific training.

## Model Architecture

```python
class VisionLanguageModel(nn.Module):
    def __init__(self, vision_backbone, text_backbone, hidden_dim):
        self.vision_encoder = VisionEncoder(vision_backbone)
        self.text_encoder = TextEncoder(text_backbone)
        self.cross_attention = MultiHeadCrossAttention(hidden_dim, num_heads=8)
        self.fusion = TransformerLayer(hidden_dim)

    def forward(self, images, text):
        # Extract features
        img_features = self.vision_encoder(images)
        text_features = self.text_encoder(text)

        # Cross-modal fusion
        fused = self.cross_attention(text_features, img_features)
        output = self.fusion(fused)

        return output
```

## Attention Visualization

```python
from visualization import visualize_attention

# Get attention weights
attention_weights = model.get_attention_weights(image, text)

# Visualize
visualize_attention(
    image=image,
    text=text,
    attention=attention_weights,
    save_path='attention_map.png'
)
```

## Configuration

```yaml
model:
  vision_backbone: vit-base
  text_backbone: bert-base
  hidden_dim: 768
  num_heads: 12
  num_layers: 6
  dropout: 0.1

training:
  task: captioning
  dataset: coco
  batch_size: 32
  epochs: 30
  learning_rate: 1e-4
  warmup_steps: 1000

data:
  image_size: 224
  max_text_length: 128
  augmentation: true
```

## Metrics

**Captioning**: BLEU, METEOR, CIDEr, SPICE
**VQA**: Accuracy, per-answer-type accuracy
**Retrieval**: Recall@K, mean reciprocal rank
**Classification**: Accuracy, top-5 accuracy

## Project Structure

```
Multi-Modal-VLM/
├── models/              # Model architectures
├── data/                # Dataset loaders
├── training/            # Training loops
├── evaluation/          # Metrics
├── visualization/       # Attention visualization
├── configs/             # Configuration files
└── train.py             # Main training script
```

## Implementation Notes

Uses PyTorch with torchvision and transformers. Vision encoder extracts spatial features from images. Text encoder processes tokenized text. Cross-attention aligns visual and textual representations.

For captioning, uses autoregressive decoder with cross-attention to image features. For VQA, classifies over answer vocabulary. For retrieval, computes cosine similarity in joint embedding space.

Pre-training uses contrastive learning (CLIP-style) to align image and text embeddings.

## References

- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
- Li et al. "BLIP: Bootstrapping Language-Image Pre-training"
- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition" (ViT)
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers"

## License

MIT License - see LICENSE file for details.
