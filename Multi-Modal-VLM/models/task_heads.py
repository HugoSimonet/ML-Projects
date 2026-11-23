"""
Task-Specific Heads Module
Implements heads for image captioning, VQA, and cross-modal retrieval
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import torch.nn.functional as F


class CaptionHead(nn.Module):
    """
    Caption generation head with autoregressive decoding
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 30522,
        num_layers: int = 6,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        max_length: int = 77,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 101,
        eos_token_id: int = 102
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            vocab_size: Vocabulary size
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            mlp_dim: MLP dimension
            max_length: Maximum caption length
            dropout: Dropout rate
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
        """
        super(CaptionHead, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)

        # Positional embedding
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        self.token_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.position_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.output_projection.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        memory: torch.Tensor,
        target_ids: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            memory: Encoded visual features [B, M, hidden_dim]
            target_ids: Target caption tokens [B, T]
            memory_key_padding_mask: Memory padding mask [B, M]
            target_key_padding_mask: Target padding mask [B, T]

        Returns:
            logits: Output logits [B, T, vocab_size]
        """
        batch_size, tgt_len = target_ids.size()

        # Position IDs
        position_ids = torch.arange(tgt_len, dtype=torch.long, device=target_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeds = self.token_embedding(target_ids)
        position_embeds = self.position_embedding(position_ids)

        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Create causal mask for autoregressive generation
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(target_ids.device)

        # Decoder forward
        output = self.decoder(
            embeddings,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=target_key_padding_mask
        )

        # Project to vocabulary
        logits = self.output_projection(output)

        return logits

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_beams: int = 1
    ) -> torch.Tensor:
        """
        Generate captions autoregressively

        Args:
            memory: Encoded visual features [B, M, hidden_dim]
            memory_key_padding_mask: Memory padding mask [B, M]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            num_beams: Beam search width

        Returns:
            generated_ids: Generated token IDs [B, L]
        """
        batch_size = memory.size(0)
        max_length = max_length or self.max_length

        if num_beams > 1:
            return self._beam_search(
                memory, memory_key_padding_mask, max_length, num_beams
            )

        # Start with BOS token
        generated = torch.full(
            (batch_size, 1), self.bos_token_id,
            dtype=torch.long, device=memory.device
        )

        for _ in range(max_length - 1):
            # Forward pass
            logits = self.forward(memory, generated, memory_key_padding_mask)

            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have generated EOS
            if (next_token == self.eos_token_id).all():
                break

        return generated

    def _beam_search(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        max_length: int,
        num_beams: int
    ) -> torch.Tensor:
        """Beam search decoding"""
        batch_size = memory.size(0)

        # Expand memory for beam search
        memory = memory.unsqueeze(1).expand(-1, num_beams, -1, -1).reshape(
            batch_size * num_beams, -1, self.hidden_dim
        )

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.unsqueeze(1).expand(
                -1, num_beams, -1
            ).reshape(batch_size * num_beams, -1)

        # Initialize beams
        beams = torch.full(
            (batch_size * num_beams, 1), self.bos_token_id,
            dtype=torch.long, device=memory.device
        )
        beam_scores = torch.zeros(batch_size, num_beams, device=memory.device)
        beam_scores[:, 1:] = float('-inf')
        beam_scores = beam_scores.view(-1)

        for step in range(max_length - 1):
            # Get logits
            logits = self.forward(memory, beams, memory_key_padding_mask)
            next_token_logits = logits[:, -1, :]

            # Convert to log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # Add beam scores
            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)

            # Reshape for beam selection
            next_token_scores = next_token_scores.view(batch_size, num_beams * self.vocab_size)

            # Select top-k beams
            next_scores, next_tokens = torch.topk(next_token_scores, num_beams, dim=1)

            # Compute beam indices
            beam_indices = next_tokens // self.vocab_size
            next_tokens = next_tokens % self.vocab_size

            # Update beams
            beams = torch.cat([
                beams[beam_indices.view(-1) + torch.arange(batch_size, device=beams.device).unsqueeze(1) * num_beams],
                next_tokens.view(-1, 1)
            ], dim=1)

            beam_scores = next_scores.view(-1)

        # Return best beam for each batch
        beams = beams.view(batch_size, num_beams, -1)
        return beams[:, 0, :]


class VQAHead(nn.Module):
    """
    Visual Question Answering head
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_answers: int = 3129,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_answers: Number of answer classes
            dropout: Dropout rate
        """
        super(VQAHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers)
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            vision_features: Pooled vision features [B, hidden_dim]
            language_features: Pooled language features [B, hidden_dim]

        Returns:
            logits: Answer logits [B, num_answers]
        """
        # Combine vision and language features
        combined = vision_features * language_features

        # Classify
        logits = self.classifier(combined)

        return logits


class RetrievalHead(nn.Module):
    """
    Cross-modal retrieval head
    Computes similarity scores for image-text matching
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        projection_dim: int = 512
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            projection_dim: Projection dimension for similarity computation
        """
        super(RetrievalHead, self).__init__()

        # Projection layers
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

        # Fine-grained matching
        self.fine_grained_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            vision_features: Pooled vision features [B, hidden_dim]
            language_features: Pooled language features [B, hidden_dim]
            return_embeddings: Whether to return projected embeddings

        Returns:
            outputs: Dict containing similarity scores and optional embeddings
        """
        # Project to common space
        vision_embeds = self.vision_projection(vision_features)
        language_embeds = self.language_projection(language_features)

        # Normalize
        vision_embeds = F.normalize(vision_embeds, p=2, dim=-1)
        language_embeds = F.normalize(language_embeds, p=2, dim=-1)

        # Compute similarity scores (cosine similarity)
        similarity_matrix = torch.matmul(vision_embeds, language_embeds.t())

        # Fine-grained matching score
        combined = torch.cat([vision_features, language_features], dim=-1)
        fine_grained_score = self.fine_grained_classifier(combined).squeeze(-1)

        outputs = {
            'similarity_matrix': similarity_matrix,
            'fine_grained_score': fine_grained_score
        }

        if return_embeddings:
            outputs['vision_embeds'] = vision_embeds
            outputs['language_embeds'] = language_embeds

        return outputs

    def compute_retrieval_scores(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Compute combined retrieval scores

        Args:
            vision_features: Vision features [N, hidden_dim]
            language_features: Language features [M, hidden_dim]
            alpha: Weight for combining global and fine-grained scores

        Returns:
            scores: Retrieval scores [N, M]
        """
        outputs = self.forward(vision_features, language_features)

        # Global similarity
        global_scores = outputs['similarity_matrix']

        # Fine-grained scores (computed pairwise)
        N, M = global_scores.shape
        fine_grained_scores = torch.zeros_like(global_scores)

        for i in range(N):
            for j in range(M):
                combined = torch.cat([vision_features[i], language_features[j]], dim=-1)
                fine_grained_scores[i, j] = self.fine_grained_classifier(combined.unsqueeze(0)).squeeze()

        # Combine scores
        combined_scores = alpha * global_scores + (1 - alpha) * fine_grained_scores

        return combined_scores


if __name__ == "__main__":
    # Test Caption Head
    caption_head = CaptionHead(vocab_size=1000)
    memory = torch.randn(2, 49, 768)
    target_ids = torch.randint(0, 1000, (2, 20))
    logits = caption_head(memory, target_ids)
    print(f"Caption Logits: {logits.shape}")

    # Test generation
    generated = caption_head.generate(memory, max_length=20, num_beams=3)
    print(f"Generated: {generated.shape}")

    # Test VQA Head
    vqa_head = VQAHead(num_answers=100)
    vision_pooled = torch.randn(2, 768)
    language_pooled = torch.randn(2, 768)
    vqa_logits = vqa_head(vision_pooled, language_pooled)
    print(f"VQA Logits: {vqa_logits.shape}")

    # Test Retrieval Head
    retrieval_head = RetrievalHead()
    outputs = retrieval_head(vision_pooled, language_pooled, return_embeddings=True)
    print(f"Similarity Matrix: {outputs['similarity_matrix'].shape}")
    print(f"Fine-grained Score: {outputs['fine_grained_score'].shape}")
