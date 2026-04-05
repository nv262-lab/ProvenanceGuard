"""SimHash-based semantic fingerprinting for CPB."""
import numpy as np
from typing import List, Tuple

class SimHashFingerprint:
    def __init__(self, num_bits=256, seed=42):
        self.num_bits = num_bits
        self.rng = np.random.RandomState(seed)
        self.hyperplanes = None

    def fit(self, embedding_dim: int):
        self.hyperplanes = self.rng.randn(self.num_bits, embedding_dim)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=1, keepdims=True)
        return self

    def hash(self, embedding: np.ndarray) -> np.ndarray:
        return (self.hyperplanes @ embedding > 0).astype(np.uint8)

    def hash_batch(self, embeddings: np.ndarray) -> np.ndarray:
        return (embeddings @ self.hyperplanes.T > 0).astype(np.uint8)

    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(a != b))

    def verify(self, a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
        return self.hamming_distance(a, b) <= threshold


class ProvenanceIndex:
    def __init__(self, hasher: SimHashFingerprint):
        self.hasher = hasher
        self.index = {}
        self.metadata = {}

    def add(self, doc_id: str, embedding: np.ndarray, meta=None):
        self.index[doc_id] = self.hasher.hash(embedding)
        self.metadata[doc_id] = meta or {}

    def add_batch(self, doc_ids, embeddings, meta_list=None):
        fps = self.hasher.hash_batch(embeddings)
        for i, did in enumerate(doc_ids):
            self.index[did] = fps[i]
            self.metadata[did] = (meta_list[i] if meta_list else {})

    def verify(self, doc_id, embedding, threshold) -> Tuple[bool, float]:
        if doc_id not in self.index:
            return False, 1.0
        stored = self.index[doc_id]
        current = self.hasher.hash(embedding)
        dist = self.hasher.hamming_distance(stored, current)
        return dist <= threshold, dist

    def __len__(self):
        return len(self.index)
