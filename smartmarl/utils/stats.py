"""Statistical utilities for SmartMARL experiments."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy.stats import wilcoxon


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size != y.size:
        raise ValueError("Cohen's d in paired mode expects equal-length arrays")

    diff = x - y
    std = np.std(diff, ddof=1) if diff.size > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(np.mean(diff) / std)


def wilcoxon_with_effect_size(
    variant_values: Iterable[float],
    baseline_values: Iterable[float],
) -> dict:
    x = np.asarray(list(variant_values), dtype=np.float64)
    y = np.asarray(list(baseline_values), dtype=np.float64)

    if x.size != y.size:
        raise ValueError("Wilcoxon signed-rank test expects paired equal-length samples")

    stat = wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
    effect = cohens_d(x, y)

    return {
        "W": float(stat.statistic),
        "p_value": float(stat.pvalue),
        "cohens_d": float(effect),
    }


def bootstrap_confidence_interval(
    values: Iterable[float],
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))

    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1.0 - alpha))
    mean = float(np.mean(arr))
    return mean, low, high


def format_mean_ci(values: Iterable[float], confidence: float = 0.95, seed: int = 0) -> str:
    mean, low, high = bootstrap_confidence_interval(values, confidence=confidence, seed=seed)
    half_width = (high - low) / 2.0
    return f"{mean:.1f}\u00b1{half_width:.1f}"
