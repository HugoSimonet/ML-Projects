"""
Model evaluator for comprehensive evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import json

from .metrics import (
    BLEUScore, METEORScore, CIDErScore, ROUGEScore,
    VQAAccuracy, RetrievalMetrics
)


class Evaluator:
    """
    Comprehensive model evaluator
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = 'cuda'
    ):
        """
        Args:
            model: VLM model
            tokenizer: Text tokenizer
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        # Initialize metrics
        self.bleu = BLEUScore()
        self.meteor = METEORScore()
        self.cider = CIDErScore()
        self.rouge = ROUGEScore()
        self.vqa_accuracy = VQAAccuracy()
        self.retrieval_metrics = RetrievalMetrics()

    @torch.no_grad()
    def evaluate_caption(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate image captioning

        Args:
            dataloader: Evaluation dataloader
            max_samples: Maximum samples to evaluate

        Returns:
            Dict of metrics
        """
        self.model.eval()

        predictions = []
        references = []

        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating Captions")):
            if max_samples and i * dataloader.batch_size >= max_samples:
                break

            images = batch['image'].to(self.device)

            # Generate captions
            generated_ids = self.model.generate_caption(images, max_length=50, num_beams=3)

            # Decode
            for j in range(generated_ids.size(0)):
                pred_text = self.tokenizer.decode(generated_ids[j].tolist(), skip_special_tokens=True)
                predictions.append(pred_text)

                # Get references
                if 'caption' in batch:
                    if isinstance(batch['caption'][j], list):
                        refs = batch['caption'][j]
                    else:
                        refs = [batch['caption'][j]]
                    references.append(refs)

        # Compute metrics
        metrics = {}
        if references:
            bleu_scores = self.bleu.compute(predictions, references)
            metrics.update(bleu_scores)
            metrics['METEOR'] = self.meteor.compute(predictions, references)
            metrics['CIDEr'] = self.cider.compute(predictions, references)
            metrics['ROUGE-L'] = self.rouge.compute(predictions, references)

        return metrics

    @torch.no_grad()
    def evaluate_vqa(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate VQA

        Args:
            dataloader: Evaluation dataloader
            max_samples: Maximum samples to evaluate

        Returns:
            Dict of metrics
        """
        self.model.eval()

        all_predictions = []
        all_references = []

        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating VQA")):
            if max_samples and i * dataloader.batch_size >= max_samples:
                break

            images = batch['image'].to(self.device)
            questions = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Get predictions
            logits = self.model.answer_question(images, questions, attention_mask)
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().tolist())

            if 'answer_label' in batch:
                all_references.extend(batch['answer_label'].tolist())

        # Compute accuracy
        metrics = {}
        if all_references:
            metrics['accuracy'] = self.vqa_accuracy.compute(all_predictions, all_references)

        return metrics

    @torch.no_grad()
    def evaluate_retrieval(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate cross-modal retrieval

        Args:
            dataloader: Evaluation dataloader
            max_samples: Maximum samples to evaluate

        Returns:
            Dict of metrics
        """
        self.model.eval()

        all_similarities = []

        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating Retrieval")):
            if max_samples and i * dataloader.batch_size >= max_samples:
                break

            images = batch['image'].to(self.device)
            texts = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Compute similarity
            similarity = self.model.compute_similarity(images, texts, attention_mask)
            all_similarities.append(similarity.cpu())

        # Concatenate all similarities
        similarity_matrix = torch.cat(all_similarities, dim=0)

        # Compute retrieval metrics
        metrics = self.retrieval_metrics.compute(similarity_matrix)

        return metrics

    def evaluate_all(
        self,
        caption_loader: Optional[DataLoader] = None,
        vqa_loader: Optional[DataLoader] = None,
        retrieval_loader: Optional[DataLoader] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all tasks

        Args:
            caption_loader: Caption dataloader
            vqa_loader: VQA dataloader
            retrieval_loader: Retrieval dataloader
            save_path: Path to save results

        Returns:
            Dict of all metrics
        """
        results = {}

        if caption_loader:
            print("\n" + "=" * 50)
            print("Evaluating Image Captioning")
            print("=" * 50)
            results['caption'] = self.evaluate_caption(caption_loader)
            print("\nCaptioning Results:")
            for k, v in results['caption'].items():
                print(f"  {k}: {v:.4f}")

        if vqa_loader:
            print("\n" + "=" * 50)
            print("Evaluating Visual Question Answering")
            print("=" * 50)
            results['vqa'] = self.evaluate_vqa(vqa_loader)
            print("\nVQA Results:")
            for k, v in results['vqa'].items():
                print(f"  {k}: {v:.4f}")

        if retrieval_loader:
            print("\n" + "=" * 50)
            print("Evaluating Cross-Modal Retrieval")
            print("=" * 50)
            results['retrieval'] = self.evaluate_retrieval(retrieval_loader)
            print("\nRetrieval Results:")
            for k, v in results['retrieval'].items():
                print(f"  {k}: {v:.4f}")

        # Save results
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {save_path}")

        return results


if __name__ == "__main__":
    print("Evaluator module - use with trained VLM model")
    print("Example usage in examples/evaluate_example.py")
