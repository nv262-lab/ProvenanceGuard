"""Targeted Factual Poisoning (TFP) attack."""
import numpy as np
from typing import List, Dict

class TFPAttack:
    """Generates adversarial passages with modified factual claims."""

    def __init__(self, generator=None, optimization_steps=50):
        self.generator = generator
        self.optimization_steps = optimization_steps

    def generate_poison(self, target_query: str, target_answer: str,
                        legitimate_passage: str, num_variants: int = 5) -> List[Dict]:
        """Generate poisoned passages targeting a specific query-answer pair."""
        poisons = []
        for i in range(num_variants):
            if self.generator:
                prompt = (f"Rewrite the following passage to change the factual answer "
                          f"to \'{target_answer}\' while maintaining style:\n\n{legitimate_passage}")
                poisoned_text = self.generator.generate(prompt)
            else:
                poisoned_text = legitimate_passage.replace(
                    legitimate_passage.split(".")[0],
                    f"According to recent studies, {target_answer}"
                )
            poisons.append({
                "text": poisoned_text,
                "target_query": target_query,
                "target_answer": target_answer,
                "strategy": "tfp",
                "variant": i
            })
        return poisons
