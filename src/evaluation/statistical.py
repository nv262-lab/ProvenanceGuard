"""Statistical analysis utilities."""
import numpy as np
from scipy import stats
from typing import List, Tuple

def bootstrap_ci(data: List[float], n_resamples=10000,
                  ci=0.95, seed=42) -> Tuple[float, float, float]:
    """Bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    data = np.array(data)
    means = [float(rng.choice(data, size=len(data), replace=True).mean())
             for _ in range(n_resamples)]
    alpha = (1 - ci) / 2
    return float(np.mean(data)), float(np.percentile(means, alpha*100)), float(np.percentile(means, (1-alpha)*100))

def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Effect size (Cohen's d)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return float((np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8))

def wilcoxon_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test."""
    stat, pval = stats.wilcoxon(group1, group2)
    return float(stat), float(pval)
