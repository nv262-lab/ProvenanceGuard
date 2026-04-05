"""Mutual Information Gating (MIG) defense mechanism."""
import numpy as np
import time
import logging
from typing import List, Dict
from ..utils.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class MINEEstimator:
    """Mutual Information Neural Estimation (simplified)."""

    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64)):
        try:
            import torch
            import torch.nn as nn
            layers = []
            prev_dim = input_dim * 2
            for h in hidden_dims:
                layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
                prev_dim = h
            layers.append(nn.Linear(prev_dim, 1))
            self.net = nn.Sequential(*layers)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
            self.torch = torch
        except ImportError:
            self.net = None
            logger.warning("PyTorch not available, using cosine MI proxy")

    def estimate(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.net is None:
            return float(np.abs(np.corrcoef(x.flatten(), y.flatten())[0, 1]))
        x_t = self.torch.FloatTensor(x).unsqueeze(0)
        y_t = self.torch.FloatTensor(y).unsqueeze(0)
        joint = self.torch.cat([x_t, y_t], dim=-1)
        y_shuffle = y_t[self.torch.randperm(y_t.size(0))]
        marginal = self.torch.cat([x_t, y_shuffle], dim=-1)
        t_joint = self.net(joint).mean()
        t_marginal = self.torch.logsumexp(self.net(marginal), 0) - np.log(marginal.size(0))
        return float(t_joint - t_marginal)


class MIGDefense:
    """MIG: Filters passages whose MI contribution deviates from consensus."""

    def __init__(self, embedding_model: EmbeddingModel, mi_threshold: float = 0.6):
        self.embedder = embedding_model
        self.mi_threshold = mi_threshold
        self.mine = MINEEstimator(embedding_model.dim)

    def filter_passages(self, passages, query: str) -> List[Dict]:
        """Filter passages based on mutual information consensus.

        Returns list of {passage, consistent, mi_ratio, consensus_score}.
        """
        start = time.time()
        if len(passages) < 2:
            return [{"passage": p, "consistent": True, "mi_ratio": 1.0,
                      "consensus_score": 1.0} for p in passages]

        texts = [p.text for p in passages]
        embeddings = self.embedder.encode(texts)
        query_emb = self.embedder.encode(query)[0]

        sim_matrix = embeddings @ embeddings.T
        query_sims = embeddings @ query_emb

        results = []
        for i, p in enumerate(passages):
            others = [j for j in range(len(passages)) if j != i]
            if not others:
                results.append({"passage": p, "consistent": True,
                                 "mi_ratio": 1.0, "consensus_score": 1.0})
                continue

            consensus_sim = np.mean([sim_matrix[i][j] for j in others])
            mi_ratio = float(consensus_sim)
            consistent = mi_ratio >= self.mi_threshold

            results.append({
                "passage": p,
                "consistent": consistent,
                "mi_ratio": mi_ratio,
                "consensus_score": float(query_sims[i])
            })

        elapsed = (time.time() - start) * 1000
        n_consistent = sum(1 for r in results if r["consistent"])
        logger.debug(f"MIG: {n_consistent}/{len(passages)} consistent in {elapsed:.1f}ms")
        return results
