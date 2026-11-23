"""
Language Encoder Module
Implements BERT-style transformer for text encoding
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class LanguageEncoder(nn.Module):
    """
    BERT-style Language Encoder for text processing
    Provides bidirectional context understanding
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        max_length: int = 77,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_dim: MLP dimension in transformer
            dropout: Dropout rate
            pad_token_id: Padding token ID
        """
        super(LanguageEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)

        # Positional embedding
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        # Token type embedding (for sentence pairs)
        self.token_type_embedding = nn.Embedding(2, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        self.token_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.position_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.token_type_embedding.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            token_type_ids: Token type IDs for sentence pairs [B, seq_len]

        Returns:
            features: Token features [B, seq_len, hidden_dim]
            pooled_features: CLS token or mean pooled features [B, hidden_dim]
        """
        batch_size, seq_len = input_ids.size()

        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Token type IDs
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        token_type_embeds = self.token_type_embedding(token_type_ids)

        # Combine embeddings
        embeddings = token_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Create padding mask for transformer
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # Convert to boolean mask (True for positions to ignore)
        padding_mask = (attention_mask == 0)

        # Transformer encoding
        features = self.transformer(
            embeddings,
            src_key_padding_mask=padding_mask
        )
        features = self.final_layer_norm(features)

        # Pooled features (mean pooling over non-padded tokens)
        if attention_mask is not None:
            # Expand mask for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).expand(features.size()).float()
            sum_embeddings = torch.sum(features * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_features = sum_embeddings / sum_mask
        else:
            pooled_features = features.mean(dim=1)

        return features, pooled_features


class BERTEncoder(nn.Module):
    """
    Wrapper for using pretrained BERT models from transformers library
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        hidden_dim: int = 768,
        freeze_encoder: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model name
            hidden_dim: Output hidden dimension
            freeze_encoder: Freeze BERT weights
        """
        super(BERTEncoder, self).__init__()

        try:
            from transformers import BertModel, BertTokenizer
            self.bert = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        except ImportError:
            raise ImportError(
                "transformers library not found. Install with: pip install transformers"
            )

        self.hidden_dim = hidden_dim

        # Freeze BERT if requested
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Projection layer if needed
        if self.bert.config.hidden_size != hidden_dim:
            self.projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            token_type_ids: Token type IDs [B, seq_len]

        Returns:
            features: Token features [B, seq_len, hidden_dim]
            pooled_features: CLS token features [B, hidden_dim]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Last hidden state and pooled output
        features = self.projection(outputs.last_hidden_state)
        pooled_features = self.projection(outputs.pooler_output)

        return features, pooled_features

    def get_tokenizer(self):
        """Return the tokenizer"""
        return self.tokenizer


class GPTDecoder(nn.Module):
    """
    GPT-style autoregressive decoder for caption generation
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        max_length: int = 77,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_dim: MLP dimension
            dropout: Dropout rate
            pad_token_id: Padding token ID
        """
        super(GPTDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)

        # Positional embedding
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        self.token_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.position_embedding.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: Target token IDs [B, tgt_len]
            memory: Encoded memory from vision encoder [B, src_len, hidden_dim]
            memory_key_padding_mask: Memory padding mask [B, src_len]

        Returns:
            logits: Output logits [B, tgt_len, vocab_size]
        """
        batch_size, tgt_len = input_ids.size()

        # Position IDs
        position_ids = torch.arange(tgt_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)

        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Create causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(input_ids.device)

        # Decoder forward
        output = self.transformer(
            embeddings,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        output = self.final_layer_norm(output)
        logits = self.output_projection(output)

        return logits


if __name__ == "__main__":
    # Test Language Encoder
    encoder = LanguageEncoder()
    input_ids = torch.randint(0, 30522, (2, 32))
    attention_mask = torch.ones(2, 32)
    features, pooled = encoder(input_ids, attention_mask)
    print(f"Language Features: {features.shape}, Pooled: {pooled.shape}")

    # Test GPT Decoder
    decoder = GPTDecoder()
    memory = torch.randn(2, 49, 768)
    tgt_ids = torch.randint(0, 30522, (2, 20))
    logits = decoder(tgt_ids, memory)
    print(f"Decoder Logits: {logits.shape}")
