# NLP with Large Language Models

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Natural language processing applications using pre-trained LLMs including GPT, BERT, and T5 with fine-tuning and prompt engineering.

## Overview

This project implements NLP applications using transformer-based language models. It covers text classification, question answering, summarization, and generation with both full fine-tuning and parameter-efficient methods (LoRA, adapters).

## Features

- Pre-trained models: GPT-2, BERT, T5, RoBERTa
- Tasks: Classification, QA, summarization, generation, NER
- Fine-tuning: Full fine-tuning and LoRA/adapter methods
- Prompt engineering: Few-shot learning, chain-of-thought
- Evaluation metrics: BLEU, ROUGE, perplexity, F1

## Models

**BERT** - Masked language modeling for classification and token tasks
**GPT-2** - Autoregressive generation and completion
**T5** - Text-to-text framework for all NLP tasks
**RoBERTa** - Robust BERT variant

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 1.9+, transformers, datasets, tokenizers

## Quick Start

```bash
# Text classification
python train.py \
    --task classification \
    --model bert-base-uncased \
    --dataset imdb \
    --epochs 3

# Text summarization
python train.py \
    --task summarization \
    --model t5-small \
    --dataset cnn_dailymail \
    --max_length 512

# Question answering
python train.py \
    --task qa \
    --model bert-base-uncased \
    --dataset squad \
    --epochs 2
```

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Classify text
inputs = tokenizer("This movie is great!", return_tensors='pt')
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1)
```

## Fine-Tuning with LoRA

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['query', 'value'],
    lora_dropout=0.1,
    bias='none',
    task_type='SEQ_CLS'
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Shows only ~1% trainable
```

## Prompt Engineering

```python
# Few-shot learning
prompt = """
Classify sentiment: positive or negative

Text: Great product!
Sentiment: positive

Text: Terrible experience.
Sentiment: negative

Text: Amazing quality!
Sentiment:"""

output = model.generate(tokenizer(prompt, return_tensors='pt').input_ids)
```

## Datasets

- **GLUE**: General Language Understanding (SST-2, MRPC, QQP, etc.)
- **SQuAD**: Question Answering
- **CNN/DailyMail**: Summarization
- **IMDB**: Sentiment classification
- **CoNLL**: Named Entity Recognition
- **Custom**: Load from CSV/JSON

## Configuration

```yaml
model:
  name: bert-base-uncased
  num_labels: 2
  max_length: 512

training:
  batch_size: 16
  epochs: 3
  learning_rate: 2e-5
  warmup_steps: 500
  weight_decay: 0.01

lora:
  enabled: true
  r: 8
  lora_alpha: 32
  target_modules: [query, value]
```

## Metrics

**Classification**: Accuracy, Precision, Recall, F1-score
**Generation**: BLEU, ROUGE-L, perplexity
**QA**: Exact Match, F1-score
**NER**: Token-level F1, entity-level F1

## Project Structure

```
NLP-LLM-Applications/
├── models/              # Model wrappers and custom architectures
├── training/            # Training loops and fine-tuning
├── evaluation/          # Metrics and evaluation
├── prompts/             # Prompt templates
├── utils/               # Data loading, tokenization
├── configs/             # Configuration files
└── train.py             # Main training script
```

## Implementation Notes

Uses Hugging Face transformers library for models and tokenizers. Training uses AdamW optimizer with linear warmup. Gradient accumulation for large batches on limited GPU memory.

LoRA reduces trainable parameters to ~1% of model size while maintaining performance. Adapter layers are alternative parameter-efficient method.

For generation tasks, use beam search or nucleus sampling. Temperature controls randomness in sampling.

## References

- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers"
- Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Raffel et al. "Exploring the Limits of Transfer Learning with T5"
- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models"
- Brown et al. "Language Models are Few-Shot Learners" (GPT-3)

## License

MIT License - see LICENSE file for details.
