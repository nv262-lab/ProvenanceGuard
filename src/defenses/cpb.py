"""Cryptographic Provenance Binding (CPB) defense mechanism."""
import numpy as np
import time
import logging
from typing import List, Tuple, Dict
from ..utils.embeddings import EmbeddingModel
from ..utils.fingerprint import SimHashFingerprint, ProvenanceIndex

logger = logging.getLogger(__name__)


class CPBDefense:
    """CPB: Verifies retrieved passages against stored semantic fingerprints."""

    def __init__(self, embedding_model: EmbeddingModel,
                 num_bits: int = 256, hamming_threshold: float = 0.15):
        self.embedder = embedding_model
        self.hasher = SimHashFingerprint(num_bits=num_bits)
        self.hasher.fit(self.embedder.dim)
        self.index = ProvenanceIndex(self.hasher)
        self.threshold = hamming_threshold

    def index_corpus(self, documents: List[Dict]):
        """Build provenance index from corpus documents."""
        texts = [doc["text"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        meta = [{"source": doc.get("source", ""), "timestamp": doc.get("timestamp", "")}
                for doc in documents]
        embeddings = self.embedder.encode(texts)
        self.index.add_batch(ids, embeddings, meta)
        logger.info(f"Indexed {len(documents)} documents in provenance index")

    def verify_passages(self, passages) -> List[Dict]:
        """Verify each passage against the provenance index.

        Returns list of {passage, verified, distance, doc_id}.
        """
        start = time.time()
        results = []
        for p in passages:
            emb = self.embedder.encode(p.text)[0]
            verified, distance = self.index.verify(p.doc_id, emb, self.threshold)
            results.append({
                "passage": p,
                "verified": verified,
                "hamming_distance": distance,
                "doc_id": p.doc_id,
                "source": self.index.metadata.get(p.doc_id, {}).get("source", "unknown")
            })
        elapsed = (time.time() - start) * 1000
        n_verified = sum(1 for r in results if r["verified"])
        logger.debug(f"CPB: {n_verified}/{len(passages)} verified in {elapsed:.1f}ms")
        return results
