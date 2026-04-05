"""Semantic Camouflage Poisoning (SCP) attack."""
import numpy as np
from typing import List, Dict

class SCPAttack:
    """Generates poisoned passages that evade semantic fingerprinting."""

    def __init__(self, embedding_model=None, optimization_steps=50):
        self.embedder = embedding_model
        self.optimization_steps = optimization_steps

    def generate_poison(self, target_query: str, target_answer: str,
                        reference_passage: str, reference_embedding=None,
                        num_variants: int = 5) -> List[Dict]:
        """Generate camouflaged poisoned passages close in embedding space."""
        poisons = []
        for i in range(num_variants):
            poisoned = reference_passage
            sentences = poisoned.split(". ")
            if len(sentences) > 1:
                idx = np.random.randint(0, len(sentences))
                sentences[idx] = f"Notably, {target_answer}"
                poisoned = ". ".join(sentences)
            poisons.append({
                "text": poisoned,
                "target_query": target_query,
                "target_answer": target_answer,
                "strategy": "scp",
                "variant": i
            })
        return poisons
