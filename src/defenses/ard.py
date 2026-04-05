"""Adaptive Retrieval Diversification (ARD) defense mechanism."""
import numpy as np
import time
import logging
from typing import List

logger = logging.getLogger(__name__)


class ARDDefense:
    """ARD: Diversifies retrieval pool to prevent targeted poisoning."""

    def __init__(self, pool_multiplier: int = 3, noise_scale: float = 2.0,
                 lambda_diversity: float = 0.3):
        self.pool_multiplier = pool_multiplier
        self.noise_scale = noise_scale
        self.lambda_diversity = lambda_diversity

    def diversify(self, passages, embeddings: np.ndarray,
                  query_embedding: np.ndarray, k: int) -> List:
        """Select k diverse passages from expanded pool using MMR.

        Args:
            passages: List of candidate passages (size K = pool_multiplier * k)
            embeddings: Passage embeddings (K x dim)
            query_embedding: Query embedding (dim,)
            k: Number of passages to select

        Returns:
            Selected k passages with diversity guarantee.
        """
        start = time.time()
        n = len(passages)
        if n <= k:
            return passages

        relevance = embeddings @ query_embedding
        noise = np.random.normal(0, self.noise_scale / 100, size=n)
        relevance_noisy = relevance + noise

        sim_matrix = embeddings @ embeddings.T

        selected = []
        selected_idx = set()
        remaining = list(range(n))

        first = int(np.argmax(relevance_noisy))
        selected.append(first)
        selected_idx.add(first)
        remaining.remove(first)

        for _ in range(k - 1):
            best_score = -float("inf")
            best_idx = remaining[0]

            for idx in remaining:
                rel = float(relevance_noisy[idx])
                max_sim = max(float(sim_matrix[idx][s]) for s in selected)
                mmr = (1 - self.lambda_diversity) * rel - self.lambda_diversity * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            selected.append(best_idx)
            selected_idx.add(best_idx)
            remaining.remove(best_idx)

        elapsed = (time.time() - start) * 1000
        logger.debug(f"ARD: selected {k}/{n} passages in {elapsed:.1f}ms")
        return [passages[i] for i in selected]
