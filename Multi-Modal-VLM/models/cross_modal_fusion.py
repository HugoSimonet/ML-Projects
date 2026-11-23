"""
Cross-Modal Fusion Module
Implements multi-head cross-attention for vision-language alignment
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class CrossAttention(nn.Module):
    """
    Cross-attention layer for vision-language interaction
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossAttention, self).__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            query: Query tensor [B, query_len, hidden_dim]
            key: Key tensor [B, key_len, hidden_dim]
            value: Value tensor [B, key_len, hidden_dim]
            attention_mask: Mask tensor [B, query_len, key_len]

        Returns:
            output: Attended output [B, query_len, hidden_dim]
            attention_weights: Attention weights [B, num_heads, query_len, key_len]
        """
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)

        # Project and reshape for multi-head attention
        Q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(1) == 0,
                float('-inf')
            )

        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)
        output = self.out_proj(output)

        return output, attention_weights


class CrossModalFusionLayer(nn.Module):
    """
    Single cross-modal fusion layer with bi-directional attention
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            mlp_dim: MLP dimension
            dropout: Dropout rate
        """
        super(CrossModalFusionLayer, self).__init__()

        # Vision to Language cross-attention
        self.v2l_cross_attention = CrossAttention(hidden_dim, num_heads, dropout)
        self.v2l_norm1 = nn.LayerNorm(hidden_dim)
        self.v2l_norm2 = nn.LayerNorm(hidden_dim)
        self.v2l_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # Language to Vision cross-attention
        self.l2v_cross_attention = CrossAttention(hidden_dim, num_heads, dropout)
        self.l2v_norm1 = nn.LayerNorm(hidden_dim)
        self.l2v_norm2 = nn.LayerNorm(hidden_dim)
        self.l2v_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # Self-attention for both modalities
        self.vision_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.language_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.vision_norm = nn.LayerNorm(hidden_dim)
        self.language_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            vision_features: Vision features [B, V, hidden_dim]
            language_features: Language features [B, L, hidden_dim]
            vision_mask: Vision mask [B, V]
            language_mask: Language mask [B, L]

        Returns:
            vision_out: Updated vision features [B, V, hidden_dim]
            language_out: Updated language features [B, L, hidden_dim]
            v2l_attn: Vision-to-language attention weights
            l2v_attn: Language-to-vision attention weights
        """
        # Self-attention within each modality
        vision_self, _ = self.vision_self_attn(
            vision_features, vision_features, vision_features,
            key_padding_mask=(vision_mask == 0) if vision_mask is not None else None
        )
        vision_features = self.vision_norm(vision_features + vision_self)

        language_self, _ = self.language_self_attn(
            language_features, language_features, language_features,
            key_padding_mask=(language_mask == 0) if language_mask is not None else None
        )
        language_features = self.language_norm(language_features + language_self)

        # Vision-to-Language cross-attention
        v2l_attn_out, v2l_attn = self.v2l_cross_attention(
            query=language_features,
            key=vision_features,
            value=vision_features,
            attention_mask=vision_mask
        )
        language_features = self.v2l_norm1(language_features + v2l_attn_out)
        language_features = self.v2l_norm2(language_features + self.v2l_mlp(language_features))

        # Language-to-Vision cross-attention
        l2v_attn_out, l2v_attn = self.l2v_cross_attention(
            query=vision_features,
            key=language_features,
            value=language_features,
            attention_mask=language_mask
        )
        vision_features = self.l2v_norm1(vision_features + l2v_attn_out)
        vision_features = self.l2v_norm2(vision_features + self.l2v_mlp(vision_features))

        return vision_features, language_features, v2l_attn, l2v_attn


class CrossModalFusion(nn.Module):
    """
    Multi-layer cross-modal fusion module
    Implements deep bi-directional cross-attention between vision and language
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of fusion layers
            num_heads: Number of attention heads
            mlp_dim: MLP dimension
            dropout: Dropout rate
        """
        super(CrossModalFusion, self).__init__()

        self.layers = nn.ModuleList([
            CrossModalFusionLayer(hidden_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm_vision = nn.LayerNorm(hidden_dim)
        self.final_norm_language = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            vision_features: Vision features [B, V, hidden_dim]
            language_features: Language features [B, L, hidden_dim]
            vision_mask: Vision mask [B, V]
            language_mask: Language mask [B, L]
            return_attention: Whether to return attention weights

        Returns:
            vision_out: Fused vision features [B, V, hidden_dim]
            language_out: Fused language features [B, L, hidden_dim]
            (optional) attention_weights: Dict of attention weights
        """
        all_v2l_attns = []
        all_l2v_attns = []

        for layer in self.layers:
            vision_features, language_features, v2l_attn, l2v_attn = layer(
                vision_features, language_features, vision_mask, language_mask
            )

            if return_attention:
                all_v2l_attns.append(v2l_attn)
                all_l2v_attns.append(l2v_attn)

        vision_out = self.final_norm_vision(vision_features)
        language_out = self.final_norm_language(language_features)

        if return_attention:
            return vision_out, language_out, {
                'v2l_attention': torch.stack(all_v2l_attns),
                'l2v_attention': torch.stack(all_l2v_attns)
            }

        return vision_out, language_out


class ContrastiveFusion(nn.Module):
    """
    Contrastive learning module for vision-language alignment
    Implements CLIP-style contrastive loss
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        projection_dim: int = 512,
        temperature: float = 0.07
    ):
        """
        Args:
            hidden_dim: Input hidden dimension
            projection_dim: Output projection dimension
            temperature: Temperature for contrastive loss
        """
        super(ContrastiveFusion, self).__init__()

        self.temperature = temperature

        # Projection heads
        self.vision_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.language_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            vision_features: Pooled vision features [B, hidden_dim]
            language_features: Pooled language features [B, hidden_dim]

        Returns:
            vision_embeds: Projected vision embeddings [B, projection_dim]
            language_embeds: Projected language embeddings [B, projection_dim]
            logit_scale: Temperature parameter
        """
        # Project to contrastive space
        vision_embeds = self.vision_projection(vision_features)
        language_embeds = self.language_projection(language_features)

        # Normalize embeddings
        vision_embeds = vision_embeds / vision_embeds.norm(dim=-1, keepdim=True)
        language_embeds = language_embeds / language_embeds.norm(dim=-1, keepdim=True)

        return vision_embeds, language_embeds, self.logit_scale.exp()


if __name__ == "__main__":
    # Test Cross-Modal Fusion
    fusion = CrossModalFusion(num_layers=3)

    vision_features = torch.randn(2, 49, 768)
    language_features = torch.randn(2, 32, 768)

    vision_out, language_out = fusion(vision_features, language_features)
    print(f"Fused Vision: {vision_out.shape}, Fused Language: {language_out.shape}")

    # Test with attention return
    vision_out, language_out, attns = fusion(
        vision_features, language_features, return_attention=True
    )
    print(f"V2L Attention: {attns['v2l_attention'].shape}")
    print(f"L2V Attention: {attns['l2v_attention'].shape}")

    # Test Contrastive Fusion
    contrastive = ContrastiveFusion()
    vision_pooled = torch.randn(2, 768)
    language_pooled = torch.randn(2, 768)
    v_embeds, l_embeds, scale = contrastive(vision_pooled, language_pooled)
    print(f"Vision Embeds: {v_embeds.shape}, Language Embeds: {l_embeds.shape}")
    print(f"Logit Scale: {scale.item()}")
