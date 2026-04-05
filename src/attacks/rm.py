"""Ranking Manipulation (RM) attack."""
from typing import List, Dict

class RMAttack:
    """Generates passages optimized for high retrieval ranking."""

    def __init__(self, optimization_steps=50):
        self.optimization_steps = optimization_steps

    def generate_poison(self, target_query: str, target_answer: str,
                        num_variants: int = 5) -> List[Dict]:
        """Generate passages with keyword stuffing for high BM25/dense ranking."""
        query_terms = target_query.lower().split()
        poisons = []
        for i in range(num_variants):
            text = f"{target_query.strip('?')}. {target_answer}. "
            text += " ".join(query_terms) * 3
            poisons.append({
                "text": text,
                "target_query": target_query,
                "target_answer": target_answer,
                "strategy": "rm",
                "variant": i
            })
        return poisons
