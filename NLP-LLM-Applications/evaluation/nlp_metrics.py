"""
NLP Evaluation Metrics for Large Language Models.

This module provides comprehensive evaluation metrics for NLP tasks:
- BLEU, ROUGE, METEOR for text generation
- Accuracy, F1, Precision, Recall for classification
- Exact Match, F1 for question answering
- Perplexity for language modeling
- Human evaluation protocols
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import re
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    score: float
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"Score: {self.score:.4f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"score": self.score}
        if self.details:
            result.update(self.details)
        return result


class BLEUScore:
    """
    BLEU (Bilingual Evaluation Understudy) score for text generation.

    Measures n-gram overlap between generated and reference text.
    Commonly used for machine translation and text generation.
    """

    def __init__(self, n_grams: int = 4, smooth: bool = True):
        """
        Initialize BLEU scorer.

        Args:
            n_grams: Maximum n-gram order (default: 4)
            smooth: Whether to apply smoothing for zero counts
        """
        self.n_grams = n_grams
        self.smooth = smooth

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]]
    ) -> MetricResult:
        """
        Compute BLEU score.

        Args:
            predictions: Generated text(s)
            references: Reference text(s) (can be multiple per prediction)

        Returns:
            MetricResult with BLEU score and details
        """
        # Handle single prediction
        if isinstance(predictions, str):
            predictions = [predictions]

        # Handle single reference
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references, list) and all(isinstance(r, str) for r in references):
            references = [[r] for r in references]

        scores = []
        for pred, refs in zip(predictions, references):
            score = self._compute_single(pred, refs)
            scores.append(score)

        avg_score = np.mean(scores)

        return MetricResult(
            score=avg_score,
            details={
                "individual_scores": scores,
                "n_grams": self.n_grams
            }
        )

    def _compute_single(self, prediction: str, references: List[str]) -> float:
        """Compute BLEU for a single prediction-reference pair."""
        pred_tokens = self._tokenize(prediction)
        ref_tokens_list = [self._tokenize(ref) for ref in references]

        # Compute precision for each n-gram order
        precisions = []
        for n in range(1, self.n_grams + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)

            # Get maximum counts from all references
            max_ref_counts = Counter()
            for ref_tokens in ref_tokens_list:
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                for ngram in pred_ngrams:
                    max_ref_counts[ngram] = max(
                        max_ref_counts[ngram],
                        ref_ngrams[ngram]
                    )

            # Clip counts
            clipped_counts = {
                ngram: min(count, max_ref_counts[ngram])
                for ngram, count in pred_ngrams.items()
            }

            numerator = sum(clipped_counts.values())
            denominator = max(1, sum(pred_ngrams.values()))

            if self.smooth and numerator == 0:
                numerator = 1
                denominator = denominator + 1

            precisions.append(numerator / denominator if denominator > 0 else 0)

        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = np.exp(np.mean(np.log(precisions)))
        else:
            geo_mean = 0.0

        # Brevity penalty
        pred_len = len(pred_tokens)
        ref_len = min(len(ref) for ref in ref_tokens_list)

        if pred_len > ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - ref_len / max(1, pred_len))

        return bp * geo_mean

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams from tokens."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)


class ROUGEScore:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score.

    Measures recall of n-grams between generated and reference text.
    Commonly used for summarization evaluation.
    """

    def __init__(self, rouge_types: Optional[List[str]] = None):
        """
        Initialize ROUGE scorer.

        Args:
            rouge_types: Types of ROUGE to compute (default: ['rouge1', 'rouge2', 'rougeL'])
        """
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> MetricResult:
        """
        Compute ROUGE scores.

        Args:
            predictions: Generated text(s)
            references: Reference text(s)

        Returns:
            MetricResult with ROUGE scores
        """
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]

        scores = {rouge_type: [] for rouge_type in self.rouge_types}

        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)

            for rouge_type in self.rouge_types:
                if rouge_type == 'rouge1':
                    score = self._rouge_n(pred_tokens, ref_tokens, 1)
                elif rouge_type == 'rouge2':
                    score = self._rouge_n(pred_tokens, ref_tokens, 2)
                elif rouge_type == 'rougeL':
                    score = self._rouge_l(pred_tokens, ref_tokens)
                else:
                    score = 0.0

                scores[rouge_type].append(score)

        # Average scores
        avg_scores = {
            rouge_type: np.mean(score_list)
            for rouge_type, score_list in scores.items()
        }

        # Overall score (average of all ROUGE types)
        overall_score = np.mean(list(avg_scores.values()))

        return MetricResult(
            score=overall_score,
            details=avg_scores
        )

    def _rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """Compute ROUGE-N score."""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)

        if len(ref_ngrams) == 0:
            return 0.0

        overlap = sum((pred_ngrams & ref_ngrams).values())

        precision = overlap / max(1, sum(pred_ngrams.values()))
        recall = overlap / sum(ref_ngrams.values())

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute ROUGE-L (Longest Common Subsequence) score."""
        lcs_length = self._lcs(pred_tokens, ref_tokens)

        if len(ref_tokens) == 0 or len(pred_tokens) == 0:
            return 0.0

        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _lcs(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams from tokens."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)


class METEORScore:
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) score.

    Considers synonyms, stemming, and word order.
    More sophisticated than BLEU for evaluating text generation.
    """

    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5):
        """
        Initialize METEOR scorer.

        Args:
            alpha: Weight for precision vs recall
            beta: Weight for fragmentation penalty
            gamma: Fragmentation penalty parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> MetricResult:
        """
        Compute METEOR score.

        Args:
            predictions: Generated text(s)
            references: Reference text(s)

        Returns:
            MetricResult with METEOR score
        """
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]

        scores = []
        for pred, ref in zip(predictions, references):
            score = self._compute_single(pred, ref)
            scores.append(score)

        avg_score = np.mean(scores)

        return MetricResult(
            score=avg_score,
            details={"individual_scores": scores}
        )

    def _compute_single(self, prediction: str, reference: str) -> float:
        """Compute METEOR for a single prediction-reference pair."""
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        # Find matches (exact matches in this simplified version)
        matches = 0
        matched_pred = set()
        matched_ref = set()

        for i, pred_token in enumerate(pred_tokens):
            for j, ref_token in enumerate(ref_tokens):
                if pred_token == ref_token and j not in matched_ref:
                    matches += 1
                    matched_pred.add(i)
                    matched_ref.add(j)
                    break

        # Precision and recall
        precision = matches / max(1, len(pred_tokens))
        recall = matches / max(1, len(ref_tokens))

        if precision + recall == 0:
            return 0.0

        # F-mean
        f_mean = (precision * recall) / (
            self.alpha * precision + (1 - self.alpha) * recall
        )

        # Fragmentation penalty (simplified)
        chunks = self._count_chunks(matched_pred)
        fragmentation = chunks / max(1, matches)
        penalty = self.gamma * (fragmentation ** self.beta)

        # Final score
        score = f_mean * (1 - penalty)
        return max(0.0, score)

    def _count_chunks(self, matched_indices: set) -> int:
        """Count number of chunks in matched indices."""
        if not matched_indices:
            return 0

        sorted_indices = sorted(matched_indices)
        chunks = 1

        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] != sorted_indices[i-1] + 1:
                chunks += 1

        return chunks

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()


class ClassificationMetrics:
    """
    Metrics for text classification tasks.

    Computes accuracy, precision, recall, F1-score.
    """

    def __init__(self, average: str = 'macro'):
        """
        Initialize classification metrics.

        Args:
            average: Averaging method ('micro', 'macro', 'weighted')
        """
        self.average = average

    def compute(
        self,
        predictions: List[int],
        references: List[int],
        num_classes: Optional[int] = None
    ) -> MetricResult:
        """
        Compute classification metrics.

        Args:
            predictions: Predicted labels
            references: True labels
            num_classes: Number of classes (auto-detected if None)

        Returns:
            MetricResult with accuracy, precision, recall, F1
        """
        predictions = np.array(predictions)
        references = np.array(references)

        if num_classes is None:
            num_classes = max(max(predictions), max(references)) + 1

        # Accuracy
        accuracy = np.mean(predictions == references)

        # Per-class metrics
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        support_per_class = []

        for c in range(num_classes):
            # True/False Positives/Negatives
            tp = np.sum((predictions == c) & (references == c))
            fp = np.sum((predictions == c) & (references != c))
            fn = np.sum((predictions != c) & (references == c))
            tn = np.sum((predictions != c) & (references != c))

            support = np.sum(references == c)
            support_per_class.append(support)

            # Precision, Recall, F1
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-10, precision + recall)

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

        # Average metrics
        if self.average == 'macro':
            avg_precision = np.mean(precision_per_class)
            avg_recall = np.mean(recall_per_class)
            avg_f1 = np.mean(f1_per_class)
        elif self.average == 'micro':
            # Micro-averaging (aggregate TP, FP, FN across all classes)
            total_tp = np.sum([
                np.sum((predictions == c) & (references == c))
                for c in range(num_classes)
            ])
            total_fp = np.sum([
                np.sum((predictions == c) & (references != c))
                for c in range(num_classes)
            ])
            total_fn = np.sum([
                np.sum((predictions != c) & (references == c))
                for c in range(num_classes)
            ])

            avg_precision = total_tp / max(1, total_tp + total_fp)
            avg_recall = total_tp / max(1, total_tp + total_fn)
            avg_f1 = 2 * avg_precision * avg_recall / max(1e-10, avg_precision + avg_recall)
        else:  # weighted
            total_support = sum(support_per_class)
            avg_precision = np.sum([
                p * s / max(1, total_support)
                for p, s in zip(precision_per_class, support_per_class)
            ])
            avg_recall = np.sum([
                r * s / max(1, total_support)
                for r, s in zip(recall_per_class, support_per_class)
            ])
            avg_f1 = np.sum([
                f * s / max(1, total_support)
                for f, s in zip(f1_per_class, support_per_class)
            ])

        return MetricResult(
            score=avg_f1,
            details={
                "accuracy": accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "per_class": {
                    "precision": precision_per_class,
                    "recall": recall_per_class,
                    "f1": f1_per_class,
                    "support": support_per_class
                }
            }
        )


class QAMetrics:
    """
    Metrics for question answering tasks.

    Computes Exact Match (EM) and F1 score.
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize QA metrics.

        Args:
            normalize: Whether to normalize text before comparison
        """
        self.normalize = normalize

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> MetricResult:
        """
        Compute QA metrics.

        Args:
            predictions: Predicted answers
            references: True answers

        Returns:
            MetricResult with EM and F1 scores
        """
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]

        em_scores = []
        f1_scores = []

        for pred, ref in zip(predictions, references):
            if self.normalize:
                pred = self._normalize_answer(pred)
                ref = self._normalize_answer(ref)

            # Exact Match
            em = 1.0 if pred == ref else 0.0
            em_scores.append(em)

            # F1 Score
            f1 = self._compute_f1(pred, ref)
            f1_scores.append(f1)

        avg_em = np.mean(em_scores)
        avg_f1 = np.mean(f1_scores)

        return MetricResult(
            score=avg_f1,
            details={
                "exact_match": avg_em,
                "f1": avg_f1,
                "individual_em": em_scores,
                "individual_f1": f1_scores
            }
        )

    def _normalize_answer(self, text: str) -> str:
        """Normalize answer text."""
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text.lower())

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _compute_f1(self, prediction: str, reference: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = prediction.split()
        ref_tokens = reference.split()

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0 if pred_tokens != ref_tokens else 1.0

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)

        f1 = 2 * precision * recall / (precision + recall)
        return f1


class GenerationMetrics:
    """
    Comprehensive metrics for text generation tasks.

    Combines BLEU, ROUGE, METEOR, and other metrics.
    """

    def __init__(self):
        """Initialize generation metrics."""
        self.bleu = BLEUScore()
        self.rouge = ROUGEScore()
        self.meteor = METEORScore()

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> MetricResult:
        """
        Compute all generation metrics.

        Args:
            predictions: Generated text(s)
            references: Reference text(s)

        Returns:
            MetricResult with all metrics
        """
        bleu_result = self.bleu.compute(predictions, references)
        rouge_result = self.rouge.compute(predictions, references)
        meteor_result = self.meteor.compute(predictions, references)

        # Compute diversity metrics
        diversity = self._compute_diversity(predictions)

        # Overall score (average of main metrics)
        overall_score = np.mean([
            bleu_result.score,
            rouge_result.score,
            meteor_result.score
        ])

        return MetricResult(
            score=overall_score,
            details={
                "bleu": bleu_result.score,
                "rouge": rouge_result.details,
                "meteor": meteor_result.score,
                "diversity": diversity
            }
        )

    def _compute_diversity(self, predictions: Union[str, List[str]]) -> Dict[str, float]:
        """Compute diversity metrics for generated text."""
        if isinstance(predictions, str):
            predictions = [predictions]

        all_tokens = []
        all_bigrams = []

        for pred in predictions:
            tokens = pred.lower().split()
            all_tokens.extend(tokens)

            bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
            all_bigrams.extend(bigrams)

        # Type-Token Ratio (TTR)
        ttr = len(set(all_tokens)) / max(1, len(all_tokens))

        # Distinct-1 and Distinct-2
        distinct_1 = len(set(all_tokens)) / max(1, len(all_tokens))
        distinct_2 = len(set(all_bigrams)) / max(1, len(all_bigrams))

        return {
            "type_token_ratio": ttr,
            "distinct_1": distinct_1,
            "distinct_2": distinct_2
        }


# Utility functions

def evaluate_generation(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate text generation with multiple metrics.

    Args:
        predictions: Generated text(s)
        references: Reference text(s)
        metrics: Metrics to compute (default: ['bleu', 'rouge', 'meteor'])

    Returns:
        Dictionary of metric scores
    """
    metrics = metrics or ['bleu', 'rouge', 'meteor']
    results = {}

    if 'bleu' in metrics:
        bleu = BLEUScore()
        results['bleu'] = bleu.compute(predictions, references).score

    if 'rouge' in metrics:
        rouge = ROUGEScore()
        rouge_result = rouge.compute(predictions, references)
        results['rouge'] = rouge_result.score
        results.update(rouge_result.details)

    if 'meteor' in metrics:
        meteor = METEORScore()
        results['meteor'] = meteor.compute(predictions, references).score

    return results


def evaluate_classification(
    predictions: List[int],
    references: List[int],
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Evaluate classification with standard metrics.

    Args:
        predictions: Predicted labels
        references: True labels
        average: Averaging method

    Returns:
        Dictionary of metric scores
    """
    metrics = ClassificationMetrics(average=average)
    result = metrics.compute(predictions, references)
    return result.to_dict()


def evaluate_qa(
    predictions: Union[str, List[str]],
    references: Union[str, List[str]],
    normalize: bool = True
) -> Dict[str, float]:
    """
    Evaluate question answering with EM and F1.

    Args:
        predictions: Predicted answers
        references: True answers
        normalize: Whether to normalize text

    Returns:
        Dictionary of metric scores
    """
    metrics = QAMetrics(normalize=normalize)
    result = metrics.compute(predictions, references)
    return result.to_dict()


def compute_perplexity(log_probs: List[float]) -> float:
    """
    Compute perplexity from log probabilities.

    Args:
        log_probs: List of log probabilities

    Returns:
        Perplexity score
    """
    avg_log_prob = np.mean(log_probs)
    perplexity = np.exp(-avg_log_prob)
    return perplexity
