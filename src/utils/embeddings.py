"""Sentence embedding utilities."""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 device=None, batch_size=64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded {model_name} (dim={self.dim}) on {self.device}")

    def encode(self, texts: Union[str, List[str]], normalize=True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, batch_size=self.batch_size,
                                  show_progress_bar=False,
                                  normalize_embeddings=normalize)

    def pairwise_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings @ embeddings.T

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
