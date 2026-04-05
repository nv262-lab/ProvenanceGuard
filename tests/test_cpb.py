"""Unit tests for CPB defense."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.fingerprint import SimHashFingerprint, ProvenanceIndex


def test_simhash_deterministic():
    hasher = SimHashFingerprint(num_bits=128, seed=42)
    hasher.fit(embedding_dim=384)
    emb = np.random.randn(384)
    h1 = hasher.hash(emb)
    h2 = hasher.hash(emb)
    assert np.array_equal(h1, h2), "SimHash should be deterministic"


def test_similar_embeddings_similar_hash():
    hasher = SimHashFingerprint(num_bits=256, seed=42)
    hasher.fit(embedding_dim=384)
    emb1 = np.random.randn(384)
    emb2 = emb1 + np.random.randn(384) * 0.01  # small perturbation
    h1 = hasher.hash(emb1)
    h2 = hasher.hash(emb2)
    dist = hasher.hamming_distance(h1, h2)
    assert dist < 0.2, f"Similar embeddings should have small Hamming distance, got {dist}"


def test_provenance_index():
    hasher = SimHashFingerprint(num_bits=128, seed=42)
    hasher.fit(embedding_dim=64)
    index = ProvenanceIndex(hasher)

    emb = np.random.randn(64)
    index.add("doc1", emb, {"source": "test"})
    assert len(index) == 1

    verified, dist = index.verify("doc1", emb, threshold=0.15)
    assert verified, "Same embedding should verify"
    assert dist == 0.0, "Same embedding should have 0 distance"

    tampered = np.random.randn(64)
    verified, dist = index.verify("doc1", tampered, threshold=0.15)
    assert not verified or dist > 0.1, "Random embedding should not verify"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
