"""
Evaluation metrics for vision-language tasks
Implements BLEU, METEOR, CIDEr, ROUGE-L, VQA accuracy, and retrieval metrics
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import re
import math


class BLEUScore:
    """
    BLEU score for image captioning evaluation
    """

    def __init__(self, n_grams: List[int] = [1, 2, 3, 4]):
        """
        Args:
            n_grams: List of n-gram orders to compute
        """
        self.n_grams = n_grams

    def compute(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute BLEU scores

        Args:
            predictions: List of predicted captions
            references: List of reference caption lists

        Returns:
            Dict of BLEU scores for each n-gram
        """
        scores = {}

        for n in self.n_grams:
            precision = self._compute_bleu_n(predictions, references, n)
            scores[f'BLEU-{n}'] = precision

        return scores

    def _compute_bleu_n(
        self,
        predictions: List[str],
        references: List[List[str]],
        n: int
    ) -> float:
        """Compute BLEU-n score"""
        total_precision = 0
        total_bp = 0
        count = 0

        for pred, refs in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens_list = [self._tokenize(ref) for ref in refs]

            # Compute precision
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            max_ref_counts = Counter()

            for ref_tokens in ref_tokens_list:
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                for ngram in ref_ngrams:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])

            # Clipped counts
            clipped_counts = {
                ngram: min(count, max_ref_counts[ngram])
                for ngram, count in pred_ngrams.items()
            }

            precision = sum(clipped_counts.values()) / max(sum(pred_ngrams.values()), 1)

            # Brevity penalty
            pred_len = len(pred_tokens)
            ref_len = min([len(ref_tokens) for ref_tokens in ref_tokens_list], key=lambda x: abs(x - pred_len))
            bp = min(1.0, np.exp(1 - ref_len / max(pred_len, 1)))

            total_precision += precision * bp
            total_bp += bp
            count += 1

        return total_precision / count if count > 0 else 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return re.findall(r'\b\w+\b', text.lower())

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return Counter(ngrams)


class METEORScore:
    """
    METEOR score for image captioning evaluation
    Simplified implementation
    """

    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5):
        """
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
        predictions: List[str],
        references: List[List[str]]
    ) -> float:
        """
        Compute METEOR score

        Args:
            predictions: List of predicted captions
            references: List of reference caption lists

        Returns:
            METEOR score
        """
        total_score = 0

        for pred, refs in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            best_score = 0

            for ref in refs:
                ref_tokens = self._tokenize(ref)
                score = self._compute_meteor_single(pred_tokens, ref_tokens)
                best_score = max(best_score, score)

            total_score += best_score

        return total_score / len(predictions) if predictions else 0.0

    def _compute_meteor_single(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str]
    ) -> float:
        """Compute METEOR for single prediction-reference pair"""
        # Find matches
        matches = self._find_matches(pred_tokens, ref_tokens)
        m = len(matches)

        if m == 0:
            return 0.0

        # Precision and recall
        precision = m / len(pred_tokens) if pred_tokens else 0
        recall = m / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            return 0.0

        # F-mean
        f_mean = (precision * recall) / (self.alpha * precision + (1 - self.alpha) * recall)

        # Fragmentation penalty
        chunks = self._count_chunks(matches)
        fragmentation = chunks / m if m > 0 else 0
        penalty = self.gamma * (fragmentation ** self.beta)

        # Final score
        score = f_mean * (1 - penalty)

        return score

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return re.findall(r'\b\w+\b', text.lower())

    def _find_matches(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str]
    ) -> List[Tuple[int, int]]:
        """Find matching tokens"""
        matches = []
        used_ref = set()
        used_pred = set()

        for i, pred_token in enumerate(pred_tokens):
            for j, ref_token in enumerate(ref_tokens):
                if j not in used_ref and i not in used_pred and pred_token == ref_token:
                    matches.append((i, j))
                    used_ref.add(j)
                    used_pred.add(i)
                    break

        return matches

    def _count_chunks(self, matches: List[Tuple[int, int]]) -> int:
        """Count number of chunks (consecutive matches)"""
        if not matches:
            return 0

        matches = sorted(matches)
        chunks = 1

        for i in range(1, len(matches)):
            if matches[i][0] != matches[i - 1][0] + 1 or matches[i][1] != matches[i - 1][1] + 1:
                chunks += 1

        return chunks


class CIDErScore:
    """
    CIDEr score for image captioning evaluation
    """

    def __init__(self, n_grams: int = 4, sigma: float = 6.0):
        """
        Args:
            n_grams: Maximum n-gram order
            sigma: Standard deviation for Gaussian penalty
        """
        self.n_grams = n_grams
        self.sigma = sigma

    def compute(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> float:
        """
        Compute CIDEr score

        Args:
            predictions: List of predicted captions
            references: List of reference caption lists

        Returns:
            CIDEr score
        """
        # Compute document frequencies
        doc_frequencies = self._compute_doc_frequencies(predictions, references)

        total_score = 0

        for pred, refs in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens_list = [self._tokenize(ref) for ref in refs]

            score = 0

            for n in range(1, self.n_grams + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams_list = [self._get_ngrams(ref_tokens, n) for ref_tokens in ref_tokens_list]

                # Compute TF-IDF vectors
                pred_vec = self._compute_tfidf(pred_ngrams, doc_frequencies, n, len(predictions))

                ref_vecs = []
                for ref_ngrams in ref_ngrams_list:
                    ref_vec = self._compute_tfidf(ref_ngrams, doc_frequencies, n, len(predictions))
                    ref_vecs.append(ref_vec)

                # Compute cosine similarity with each reference
                sim_scores = []
                for ref_vec in ref_vecs:
                    sim = self._cosine_similarity(pred_vec, ref_vec)
                    sim_scores.append(sim)

                # Average similarity
                score += np.mean(sim_scores)

            # Average over n-grams
            score = score / self.n_grams

            total_score += score

        return total_score / len(predictions) if predictions else 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return re.findall(r'\b\w+\b', text.lower())

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return Counter(ngrams)

    def _compute_doc_frequencies(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> Dict[int, Counter]:
        """Compute document frequencies for IDF"""
        doc_frequencies = defaultdict(Counter)

        # Combine all texts
        all_texts = predictions + [ref for refs in references for ref in refs]

        for n in range(1, self.n_grams + 1):
            for text in all_texts:
                tokens = self._tokenize(text)
                ngrams = set(self._get_ngrams(tokens, n).keys())
                for ngram in ngrams:
                    doc_frequencies[n][ngram] += 1

        return doc_frequencies

    def _compute_tfidf(
        self,
        ngrams: Counter,
        doc_frequencies: Dict[int, Counter],
        n: int,
        num_docs: int
    ) -> Dict:
        """Compute TF-IDF vector"""
        tfidf = {}

        for ngram, count in ngrams.items():
            tf = count
            idf = np.log((num_docs + 1) / (doc_frequencies[n][ngram] + 1))
            tfidf[ngram] = tf * idf

        return tfidf

    def _cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        """Compute cosine similarity between TF-IDF vectors"""
        dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1.keys()) | set(vec2.keys()))

        norm1 = np.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = np.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class ROUGEScore:
    """
    ROUGE-L score for image captioning evaluation
    """

    def compute(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> float:
        """
        Compute ROUGE-L score

        Args:
            predictions: List of predicted captions
            references: List of reference caption lists

        Returns:
            ROUGE-L score
        """
        total_score = 0

        for pred, refs in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            best_score = 0

            for ref in refs:
                ref_tokens = self._tokenize(ref)
                lcs_length = self._lcs_length(pred_tokens, ref_tokens)

                if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                    score = 0.0
                else:
                    precision = lcs_length / len(pred_tokens)
                    recall = lcs_length / len(ref_tokens)

                    if precision + recall == 0:
                        score = 0.0
                    else:
                        score = 2 * precision * recall / (precision + recall)

                best_score = max(best_score, score)

            total_score += best_score

        return total_score / len(predictions) if predictions else 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return re.findall(r'\b\w+\b', text.lower())

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]


class VQAAccuracy:
    """
    VQA accuracy metric
    """

    def compute(
        self,
        predictions: List[int],
        references: List[int]
    ) -> float:
        """
        Compute VQA accuracy

        Args:
            predictions: List of predicted answer IDs
            references: List of ground truth answer IDs

        Returns:
            Accuracy score
        """
        correct = sum(pred == ref for pred, ref in zip(predictions, references))
        return correct / len(predictions) if predictions else 0.0


class RetrievalMetrics:
    """
    Cross-modal retrieval metrics
    """

    def compute(
        self,
        similarity_matrix: torch.Tensor,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics

        Args:
            similarity_matrix: Similarity scores [N, M]
            k_values: List of k values for Recall@K

        Returns:
            Dict of retrieval metrics
        """
        metrics = {}

        # Image-to-text retrieval
        i2t_ranks = self._get_ranks(similarity_matrix)
        for k in k_values:
            recall = (i2t_ranks < k).float().mean().item()
            metrics[f'i2t_R@{k}'] = recall

        metrics['i2t_median_rank'] = torch.median(i2t_ranks).item()
        metrics['i2t_mean_rank'] = i2t_ranks.float().mean().item()

        # Text-to-image retrieval
        t2i_ranks = self._get_ranks(similarity_matrix.t())
        for k in k_values:
            recall = (t2i_ranks < k).float().mean().item()
            metrics[f't2i_R@{k}'] = recall

        metrics['t2i_median_rank'] = torch.median(t2i_ranks).item()
        metrics['t2i_mean_rank'] = t2i_ranks.float().mean().item()

        return metrics

    def _get_ranks(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """Get retrieval ranks"""
        # Sort in descending order
        sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)

        # Find position of ground truth (diagonal)
        ranks = torch.zeros(similarity_matrix.size(0), dtype=torch.long)
        for i in range(similarity_matrix.size(0)):
            ranks[i] = (sorted_indices[i] == i).nonzero(as_tuple=True)[0][0]

        return ranks


def compute_all_metrics(
    predictions: List[str],
    references: List[List[str]],
    task: str = 'caption'
) -> Dict[str, float]:
    """
    Compute all relevant metrics for a task

    Args:
        predictions: List of predictions
        references: List of references
        task: Task type

    Returns:
        Dict of metrics
    """
    metrics = {}

    if task == 'caption':
        bleu = BLEUScore()
        meteor = METEORScore()
        cider = CIDErScore()
        rouge = ROUGEScore()

        metrics.update(bleu.compute(predictions, references))
        metrics['METEOR'] = meteor.compute(predictions, references)
        metrics['CIDEr'] = cider.compute(predictions, references)
        metrics['ROUGE-L'] = rouge.compute(predictions, references)

    return metrics


if __name__ == "__main__":
    # Test metrics
    predictions = ["a cat sitting on a chair", "a dog running in the park"]
    references = [
        ["a cat is sitting on the chair", "there is a cat on a chair"],
        ["a dog is running in a park", "the dog runs through the park"]
    ]

    # Test BLEU
    bleu = BLEUScore()
    bleu_scores = bleu.compute(predictions, references)
    print("BLEU Scores:", bleu_scores)

    # Test METEOR
    meteor = METEORScore()
    meteor_score = meteor.compute(predictions, references)
    print(f"METEOR: {meteor_score:.4f}")

    # Test CIDEr
    cider = CIDErScore()
    cider_score = cider.compute(predictions, references)
    print(f"CIDEr: {cider_score:.4f}")

    # Test ROUGE
    rouge = ROUGEScore()
    rouge_score = rouge.compute(predictions, references)
    print(f"ROUGE-L: {rouge_score:.4f}")
