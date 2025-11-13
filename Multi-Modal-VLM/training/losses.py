"""
Loss functions for vision-language tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for vision-language alignment (CLIP-style)
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for softmax
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        vision_embeds: torch.Tensor,
        language_embeds: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            vision_embeds: Vision embeddings [B, D]
            language_embeds: Language embeddings [B, D]
            logit_scale: Optional learned temperature

        Returns:
            loss: Contrastive loss
        """
        batch_size = vision_embeds.size(0)

        # Normalize embeddings
        vision_embeds = F.normalize(vision_embeds, p=2, dim=-1)
        language_embeds = F.normalize(language_embeds, p=2, dim=-1)

        # Compute similarity matrix
        if logit_scale is not None:
            similarity = logit_scale * (vision_embeds @ language_embeds.t())
        else:
            similarity = (vision_embeds @ language_embeds.t()) / self.temperature

        # Labels are diagonal (positive pairs)
        labels = torch.arange(batch_size, device=vision_embeds.device)

        # Compute cross-entropy loss in both directions
        loss_i2t = F.cross_entropy(similarity, labels)
        loss_t2i = F.cross_entropy(similarity.t(), labels)

        # Average loss
        loss = (loss_i2t + loss_t2i) / 2

        return loss


class CaptionLoss(nn.Module):
    """
    Loss for image captioning with label smoothing
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_smoothing: float = 0.1
    ):
        """
        Args:
            pad_token_id: Padding token ID
            label_smoothing: Label smoothing factor
        """
        super(CaptionLoss, self).__init__()
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute caption generation loss

        Args:
            logits: Model logits [B, T, vocab_size]
            targets: Target token IDs [B, T]

        Returns:
            loss: Cross-entropy loss
        """
        # Reshape for cross-entropy
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        # Compute cross-entropy with label smoothing
        loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.pad_token_id,
            label_smoothing=self.label_smoothing
        )

        return loss


class VQALoss(nn.Module):
    """
    Loss for Visual Question Answering
    """

    def __init__(self, num_answers: int = 3129):
        """
        Args:
            num_answers: Number of answer classes
        """
        super(VQALoss, self).__init__()
        self.num_answers = num_answers

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VQA loss

        Args:
            logits: Answer logits [B, num_answers]
            targets: Target answer IDs [B] or soft labels [B, num_answers]

        Returns:
            loss: Cross-entropy loss
        """
        if targets.dim() == 1:
            # Hard labels
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        else:
            # Soft labels (multiple annotators)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(targets * log_probs).sum(dim=-1).mean()

        return loss


class RetrievalLoss(nn.Module):
    """
    Loss for cross-modal retrieval
    Combines ranking loss with contrastive loss
    """

    def __init__(
        self,
        margin: float = 0.2,
        alpha: float = 0.5
    ):
        """
        Args:
            margin: Margin for ranking loss
            alpha: Weight for combining losses
        """
        super(RetrievalLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.contrastive_loss = ContrastiveLoss()

    def forward(
        self,
        similarity_matrix: torch.Tensor,
        fine_grained_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute retrieval loss

        Args:
            similarity_matrix: Similarity scores [B, B]
            fine_grained_scores: Fine-grained matching scores [B]

        Returns:
            loss: Combined retrieval loss
        """
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)

        # Global contrastive loss
        loss_global = F.cross_entropy(similarity_matrix, labels)
        loss_global += F.cross_entropy(similarity_matrix.t(), labels)
        loss_global = loss_global / 2

        # Fine-grained matching loss
        # Positive pairs should have high scores, negative pairs low scores
        positive_scores = torch.diagonal(similarity_matrix)

        # Hard negative mining
        with torch.no_grad():
            # Mask out diagonal (positive pairs)
            mask = torch.eye(batch_size, device=similarity_matrix.device).bool()
            neg_similarities = similarity_matrix.clone()
            neg_similarities[mask] = float('-inf')

            # Get hardest negatives
            hard_neg_i2t = neg_similarities.max(dim=1)[0]
            hard_neg_t2i = neg_similarities.max(dim=0)[0]

        # Ranking loss (triplet-style)
        loss_rank_i2t = F.relu(hard_neg_i2t - positive_scores + self.margin).mean()
        loss_rank_t2i = F.relu(hard_neg_t2i - positive_scores + self.margin).mean()
        loss_rank = (loss_rank_i2t + loss_rank_t2i) / 2

        # Combine losses
        loss = self.alpha * loss_global + (1 - self.alpha) * loss_rank

        return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with automatic weighting
    """

    def __init__(
        self,
        num_tasks: int = 3,
        initial_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            num_tasks: Number of tasks
            initial_weights: Initial task weights
        """
        super(MultiTaskLoss, self).__init__()

        # Learnable task weights
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        self.task_names = ['caption', 'vqa', 'retrieval']

        if initial_weights:
            for i, name in enumerate(self.task_names):
                if name in initial_weights:
                    self.log_vars.data[i] = torch.log(torch.tensor(initial_weights[name]))

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted multi-task loss

        Args:
            losses: Dict of task losses

        Returns:
            total_loss: Weighted sum of losses
        """
        total_loss = 0
        weights = {}

        for i, task_name in enumerate(self.task_names):
            if task_name in losses:
                # Uncertainty weighting: loss / (2 * sigma^2) + log(sigma)
                # Where sigma = exp(log_var)
                precision = torch.exp(-self.log_vars[i])
                loss = precision * losses[task_name] + self.log_vars[i]
                total_loss += loss
                weights[task_name] = precision.item()

        return total_loss, weights


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss

        Args:
            logits: Model logits [B, C]
            targets: Target labels [B]

        Returns:
            loss: Focal loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    # Test Contrastive Loss
    contrastive_loss = ContrastiveLoss()
    vision_embeds = torch.randn(4, 512)
    language_embeds = torch.randn(4, 512)
    loss = contrastive_loss(vision_embeds, language_embeds)
    print(f"Contrastive Loss: {loss.item():.4f}")

    # Test Caption Loss
    caption_loss = CaptionLoss()
    logits = torch.randn(2, 20, 30522)
    targets = torch.randint(0, 30522, (2, 20))
    loss = caption_loss(logits, targets)
    print(f"Caption Loss: {loss.item():.4f}")

    # Test VQA Loss
    vqa_loss = VQALoss(num_answers=100)
    logits = torch.randn(2, 100)
    targets = torch.randint(0, 100, (2,))
    loss = vqa_loss(logits, targets)
    print(f"VQA Loss: {loss.item():.4f}")

    # Test Retrieval Loss
    retrieval_loss = RetrievalLoss()
    similarity = torch.randn(4, 4)
    fine_grained = torch.randn(4)
    loss = retrieval_loss(similarity, fine_grained)
    print(f"Retrieval Loss: {loss.item():.4f}")

    # Test Multi-task Loss
    multi_loss = MultiTaskLoss()
    losses = {
        'caption': torch.tensor(2.0),
        'vqa': torch.tensor(1.5),
        'retrieval': torch.tensor(0.8)
    }
    total_loss, weights = multi_loss(losses)
    print(f"Multi-task Loss: {total_loss.item():.4f}")
    print(f"Task Weights: {weights}")
