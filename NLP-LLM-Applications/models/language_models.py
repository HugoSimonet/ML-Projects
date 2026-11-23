"""
Language Model Architectures: GPT, BERT, and T5 implementations.

This module provides implementations of various large language model architectures
optimized for different NLP tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel, GPT2Config, GPT2Tokenizer,
    BertModel, BertConfig, BertTokenizer,
    T5ForConditionalGeneration, T5Config, T5Tokenizer,
    AutoModel, AutoTokenizer
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        self.scale = self.scale.to(query.device)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.d_model)

        return self.W_o(x), attention


class GPTModel(nn.Module):
    """
    GPT-style language model for text generation tasks.

    Features:
    - Autoregressive generation
    - Causal attention masking
    - Support for pre-trained and custom configurations
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        use_pretrained: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.use_pretrained = use_pretrained

        if use_pretrained:
            # Load pre-trained model
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # Build custom model
            config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head
            )
            self.model = GPT2LMHeadModel(config)
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = self.model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens with highest probability
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling (vs greedy)

        Returns:
            List of generated text sequences
        """
        self.model.eval()

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return generated_texts

    def get_perplexity(self, text: str) -> float:
        """Calculate perplexity of text."""
        self.model.eval()

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity


class BERTModel(nn.Module):
    """
    BERT-style model for understanding and classification tasks.

    Features:
    - Bidirectional context understanding
    - Masked language modeling
    - Support for classification, NER, QA tasks
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        hidden_size: int = 768,
        use_pretrained: bool = True,
        task_type: str = "classification"
    ):
        super().__init__()
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels

        if use_pretrained:
            self.bert = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            config = BertConfig(hidden_size=hidden_size)
            self.bert = BertModel(config)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.hidden_size = self.bert.config.hidden_size

        # Task-specific heads
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.qa_outputs = nn.Linear(self.hidden_size, 2)  # For QA: start/end positions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None
    ):
        """Forward pass through BERT."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        if self.task_type == "classification":
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return {'loss': loss, 'logits': logits, 'hidden_states': outputs.last_hidden_state}

        elif self.task_type == "qa":
            sequence_output = outputs.last_hidden_state
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            loss = None
            if start_positions is not None and end_positions is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

            return {
                'loss': loss,
                'start_logits': start_logits,
                'end_logits': end_logits,
                'hidden_states': outputs.last_hidden_state
            }

        return outputs

    def encode(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """Encode texts to embeddings."""
        self.bert.eval()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(next(self.bert.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert(**inputs)
            embeddings = outputs.pooler_output

        return embeddings

    def classify(self, text: str) -> Tuple[int, float]:
        """Classify a single text."""
        self.eval()

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.forward(**inputs)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_label].item()

        return pred_label, confidence


class T5Model(nn.Module):
    """
    T5 model for text-to-text tasks.

    Features:
    - Unified text-to-text framework
    - Support for translation, summarization, QA
    - Encoder-decoder architecture
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        use_pretrained: bool = True
    ):
        super().__init__()
        self.model_name = model_name

        if use_pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        else:
            config = T5Config()
            self.model = T5ForConditionalGeneration(config)
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")

        self.config = self.model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """Forward pass through T5."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate_text(
        self,
        input_text: str,
        task_prefix: str = "",
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> str:
        """
        Generate text using T5.

        Args:
            input_text: Input text
            task_prefix: Task-specific prefix (e.g., "summarize:", "translate English to German:")
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated text
        """
        self.model.eval()

        # Add task prefix
        if task_prefix:
            full_input = f"{task_prefix} {input_text}"
        else:
            full_input = input_text

        # Tokenize
        inputs = self.tokenizer(
            full_input,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                early_stopping=True
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def summarize(self, text: str, max_length: int = 150) -> str:
        """Summarize text."""
        return self.generate_text(text, task_prefix="summarize:", max_length=max_length)

    def translate(self, text: str, source_lang: str = "English", target_lang: str = "German") -> str:
        """Translate text."""
        task_prefix = f"translate {source_lang} to {target_lang}:"
        return self.generate_text(text, task_prefix=task_prefix)

    def answer_question(self, context: str, question: str) -> str:
        """Answer a question given context."""
        input_text = f"question: {question} context: {context}"
        return self.generate_text(input_text, max_length=100)


class CustomTransformer(nn.Module):
    """
    Custom transformer model for specialized NLP tasks.

    Features:
    - Flexible architecture
    - Custom attention mechanisms
    - Task-specific modifications
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ):
        """Forward pass through custom transformer."""
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)

        tgt = self.embedding(tgt) * np.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        memory = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        output = self.fc_out(output)

        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Utility functions

def load_model(
    model_type: str = "gpt2",
    model_name: str = None,
    **kwargs
) -> nn.Module:
    """
    Load a language model.

    Args:
        model_type: Type of model ('gpt', 'bert', 't5', 'custom')
        model_name: Specific model name (e.g., 'gpt2-medium')
        **kwargs: Additional arguments for model initialization

    Returns:
        Initialized model
    """
    model_type = model_type.lower()

    if model_type in ['gpt', 'gpt2']:
        model_name = model_name or "gpt2"
        return GPTModel(model_name=model_name, **kwargs)

    elif model_type == 'bert':
        model_name = model_name or "bert-base-uncased"
        return BERTModel(model_name=model_name, **kwargs)

    elif model_type == 't5':
        model_name = model_name or "t5-base"
        return T5Model(model_name=model_name, **kwargs)

    elif model_type == 'custom':
        return CustomTransformer(**kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(model: nn.Module, num_layers: int = 0):
    """Freeze specified number of layers from the beginning."""
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model

    layers = list(base_model.children())
    for i, layer in enumerate(layers[:num_layers]):
        for param in layer.parameters():
            param.requires_grad = False
