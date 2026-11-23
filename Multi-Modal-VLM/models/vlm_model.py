"""
Main Vision-Language Model
Combines all components into a unified architecture
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union
from .vision_encoder import VisionEncoder, ViTEncoder
from .language_encoder import LanguageEncoder, BERTEncoder
from .cross_modal_fusion import CrossModalFusion, ContrastiveFusion
from .task_heads import CaptionHead, VQAHead, RetrievalHead


class VLMModel(nn.Module):
    """
    Multi-Modal Vision-Language Model
    Supports image captioning, VQA, and cross-modal retrieval
    """

    def __init__(
        self,
        vision_encoder_type: str = 'resnet',  # 'resnet' or 'vit'
        language_encoder_type: str = 'custom',  # 'custom' or 'bert'
        hidden_dim: int = 768,
        fusion_layers: int = 6,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        vocab_size: int = 30522,
        max_length: int = 77,
        num_answers: int = 3129,
        dropout: float = 0.1,
        use_contrastive: bool = True,
        freeze_vision: bool = False,
        freeze_language: bool = False,
        pretrained: bool = True
    ):
        """
        Args:
            vision_encoder_type: Type of vision encoder ('resnet' or 'vit')
            language_encoder_type: Type of language encoder ('custom' or 'bert')
            hidden_dim: Hidden dimension for all modules
            fusion_layers: Number of cross-modal fusion layers
            num_heads: Number of attention heads
            mlp_dim: MLP dimension in transformers
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            num_answers: Number of VQA answer classes
            dropout: Dropout rate
            use_contrastive: Whether to use contrastive learning
            freeze_vision: Freeze vision encoder weights
            freeze_language: Freeze language encoder weights
            pretrained: Use pretrained weights where available
        """
        super(VLMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.use_contrastive = use_contrastive

        # Vision Encoder
        if vision_encoder_type == 'resnet':
            self.vision_encoder = VisionEncoder(
                output_dim=hidden_dim,
                pretrained=pretrained,
                freeze_backbone=freeze_vision
            )
        elif vision_encoder_type == 'vit':
            self.vision_encoder = ViTEncoder(
                hidden_dim=hidden_dim,
                pretrained=pretrained
            )
            if freeze_vision:
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"Unknown vision encoder type: {vision_encoder_type}")

        # Language Encoder
        if language_encoder_type == 'custom':
            self.language_encoder = LanguageEncoder(
                vocab_size=vocab_size,
                max_length=max_length,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif language_encoder_type == 'bert':
            self.language_encoder = BERTEncoder(
                hidden_dim=hidden_dim,
                freeze_encoder=freeze_language
            )
        else:
            raise ValueError(f"Unknown language encoder type: {language_encoder_type}")

        # Cross-Modal Fusion
        self.fusion = CrossModalFusion(
            hidden_dim=hidden_dim,
            num_layers=fusion_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # Contrastive Learning Module
        if use_contrastive:
            self.contrastive = ContrastiveFusion(
                hidden_dim=hidden_dim,
                projection_dim=512
            )

        # Task-Specific Heads
        self.caption_head = CaptionHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_length=max_length,
            dropout=dropout
        )

        self.vqa_head = VQAHead(
            hidden_dim=hidden_dim,
            num_answers=num_answers,
            dropout=dropout
        )

        self.retrieval_head = RetrievalHead(
            hidden_dim=hidden_dim,
            projection_dim=512
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        task: str = 'caption',
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            images: Input images [B, 3, H, W]
            input_ids: Text token IDs [B, L]
            attention_mask: Text attention mask [B, L]
            token_type_ids: Token type IDs [B, L]
            target_ids: Target caption tokens for training [B, T]
            task: Task type ('caption', 'vqa', 'retrieval', 'contrastive')
            return_attention: Whether to return attention weights

        Returns:
            outputs: Dict containing task-specific outputs
        """
        # Encode vision
        vision_features, vision_pooled = self.vision_encoder(images)

        # Encode language
        language_features, language_pooled = self.language_encoder(
            input_ids, attention_mask, token_type_ids
        )

        # Cross-modal fusion
        if return_attention:
            fused_vision, fused_language, attention_weights = self.fusion(
                vision_features, language_features,
                vision_mask=None,
                language_mask=attention_mask,
                return_attention=True
            )
        else:
            fused_vision, fused_language = self.fusion(
                vision_features, language_features,
                vision_mask=None,
                language_mask=attention_mask
            )
            attention_weights = None

        # Pooled fused features
        fused_vision_pooled = fused_vision.mean(dim=1)

        # Mean pooling for language (respecting attention mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(fused_language.size()).float()
            sum_embeddings = torch.sum(fused_language * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            fused_language_pooled = sum_embeddings / sum_mask
        else:
            fused_language_pooled = fused_language.mean(dim=1)

        outputs = {}

        # Task-specific processing
        if task == 'caption':
            if target_ids is not None:
                # Training mode
                caption_logits = self.caption_head(
                    memory=fused_vision,
                    target_ids=target_ids,
                    memory_key_padding_mask=None
                )
                outputs['caption_logits'] = caption_logits
            else:
                # Inference mode - generate captions
                generated = self.caption_head.generate(
                    memory=fused_vision,
                    max_length=50,
                    num_beams=3
                )
                outputs['generated_ids'] = generated

        elif task == 'vqa':
            vqa_logits = self.vqa_head(fused_vision_pooled, fused_language_pooled)
            outputs['vqa_logits'] = vqa_logits

        elif task == 'retrieval':
            retrieval_outputs = self.retrieval_head(
                fused_vision_pooled, fused_language_pooled,
                return_embeddings=True
            )
            outputs.update(retrieval_outputs)

        elif task == 'contrastive':
            if self.use_contrastive:
                vision_embeds, language_embeds, logit_scale = self.contrastive(
                    vision_pooled, language_pooled
                )
                outputs['vision_embeds'] = vision_embeds
                outputs['language_embeds'] = language_embeds
                outputs['logit_scale'] = logit_scale
            else:
                raise ValueError("Contrastive learning not enabled for this model")

        # Add features to outputs
        outputs['vision_features'] = fused_vision
        outputs['language_features'] = fused_language
        outputs['vision_pooled'] = fused_vision_pooled
        outputs['language_pooled'] = fused_language_pooled

        if return_attention:
            outputs['attention_weights'] = attention_weights

        return outputs

    def generate_caption(
        self,
        images: torch.Tensor,
        max_length: int = 50,
        num_beams: int = 3,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate captions for images

        Args:
            images: Input images [B, 3, H, W]
            max_length: Maximum caption length
            num_beams: Beam search width
            temperature: Sampling temperature

        Returns:
            generated_ids: Generated caption tokens [B, L]
        """
        # Encode vision
        vision_features, _ = self.vision_encoder(images)

        # Generate without language input
        generated = self.caption_head.generate(
            memory=vision_features,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature
        )

        return generated

    def answer_question(
        self,
        images: torch.Tensor,
        questions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Answer questions about images

        Args:
            images: Input images [B, 3, H, W]
            questions: Question tokens [B, L]
            attention_mask: Question attention mask [B, L]

        Returns:
            answer_logits: Answer logits [B, num_answers]
        """
        outputs = self.forward(
            images=images,
            input_ids=questions,
            attention_mask=attention_mask,
            task='vqa'
        )

        return outputs['vqa_logits']

    def compute_similarity(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute image-text similarity scores

        Args:
            images: Input images [B, 3, H, W]
            texts: Text tokens [B, L]
            attention_mask: Text attention mask [B, L]

        Returns:
            similarity_scores: Similarity matrix [B, B]
        """
        outputs = self.forward(
            images=images,
            input_ids=texts,
            attention_mask=attention_mask,
            task='retrieval'
        )

        return outputs['similarity_matrix']

    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters in each component"""
        return {
            'vision_encoder': sum(p.numel() for p in self.vision_encoder.parameters()),
            'language_encoder': sum(p.numel() for p in self.language_encoder.parameters()),
            'fusion': sum(p.numel() for p in self.fusion.parameters()),
            'caption_head': sum(p.numel() for p in self.caption_head.parameters()),
            'vqa_head': sum(p.numel() for p in self.vqa_head.parameters()),
            'retrieval_head': sum(p.numel() for p in self.retrieval_head.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }


if __name__ == "__main__":
    # Test VLM Model
    model = VLMModel(
        vision_encoder_type='resnet',
        language_encoder_type='custom',
        pretrained=False
    )

    # Test caption generation
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (2, 32))
    attention_mask = torch.ones(2, 32)
    target_ids = torch.randint(0, 30522, (2, 20))

    outputs = model(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_ids=target_ids,
        task='caption'
    )

    print("Caption Training:")
    print(f"  Caption Logits: {outputs['caption_logits'].shape}")

    # Test VQA
    outputs = model(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        task='vqa'
    )
    print("\nVQA:")
    print(f"  VQA Logits: {outputs['vqa_logits'].shape}")

    # Test Retrieval
    outputs = model(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        task='retrieval'
    )
    print("\nRetrieval:")
    print(f"  Similarity Matrix: {outputs['similarity_matrix'].shape}")

    # Test Contrastive
    outputs = model(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        task='contrastive'
    )
    print("\nContrastive:")
    print(f"  Vision Embeds: {outputs['vision_embeds'].shape}")
    print(f"  Language Embeds: {outputs['language_embeds'].shape}")

    # Print model parameters
    print("\nModel Parameters:")
    for name, count in model.get_num_parameters().items():
        print(f"  {name}: {count:,}")
