"""
Overlap metrics for reward hacking diagnostics.

Overlap@K measures how many of the proxy-selected Top-K are also
in the oracle-selected Top-K.
"""

import numpy as np
from typing import Tuple


def overlap_at_k(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int,
) -> float:
    """
    Compute Top-K overlap between proxy and oracle selections.

    Overlap@K = |TopK_proxy ∩ TopK_true| / K

    A value of 1.0 means proxy and oracle agree perfectly on Top-K.
    A value of 0.0 means no overlap at all.

    Args:
        proxy_scores: Array of proxy reward scores [N]
        true_scores: Array of true reward scores [N]
        k: Number of top candidates to compare

    Returns:
        Overlap ratio in [0, 1]
    """
    if len(proxy_scores) != len(true_scores):
        raise ValueError("Score arrays must have same length")

    n = len(proxy_scores)
    if k > n:
        raise ValueError(f"k={k} > N={n}")

    # Top-K indices by proxy (descending order)
    topk_proxy = set(np.argsort(proxy_scores)[-k:])

    # Top-K indices by oracle (descending order)
    topk_true = set(np.argsort(true_scores)[-k:])

    # Intersection
    overlap = len(topk_proxy & topk_true)

    return float(overlap) / k


def overlap_at_k_with_details(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute overlap with detailed information.

    Args:
        proxy_scores: Array of proxy reward scores [N]
        true_scores: Array of true reward scores [N]
        k: Number of top candidates

    Returns:
        Tuple of:
        - overlap: Overlap ratio
        - topk_proxy: Indices of top-K by proxy
        - topk_true: Indices of top-K by true
        - intersection: Indices in both sets
    """
    if len(proxy_scores) != len(true_scores):
        raise ValueError("Score arrays must have same length")

    n = len(proxy_scores)
    if k > n:
        raise ValueError(f"k={k} > N={n}")

    topk_proxy = np.argsort(proxy_scores)[-k:][::-1]
    topk_true = np.argsort(true_scores)[-k:][::-1]

    intersection = np.array(list(set(topk_proxy) & set(topk_true)))
    overlap = len(intersection) / k

    return overlap, topk_proxy, topk_true, intersection


def precision_at_k(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int,
    threshold_percentile: float = 90.0,
) -> float:
    """
    Compute precision@K: fraction of proxy-selected that are truly good.

    "Good" is defined as being above the threshold_percentile of true scores.

    Args:
        proxy_scores: Array of proxy reward scores [N]
        true_scores: Array of true reward scores [N]
        k: Number of top candidates to select by proxy
        threshold_percentile: Percentile threshold for "good" (default 90%)

    Returns:
        Precision@K in [0, 1]
    """
    if len(proxy_scores) != len(true_scores):
        raise ValueError("Score arrays must have same length")

    n = len(proxy_scores)
    if k > n:
        raise ValueError(f"k={k} > N={n}")

    # Select top-K by proxy
    topk_proxy = np.argsort(proxy_scores)[-k:]

    # Define "good" threshold
    threshold = np.percentile(true_scores, threshold_percentile)

    # Count how many selected are truly good
    selected_true_scores = true_scores[topk_proxy]
    n_good = np.sum(selected_true_scores >= threshold)

    return float(n_good) / k


def recall_at_k(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int,
) -> float:
    """
    Compute recall@K: fraction of true top-K that are proxy-selected.

    This equals overlap@K when using same K for both.

    Args:
        proxy_scores: Array of proxy reward scores [N]
        true_scores: Array of true reward scores [N]
        k: Number of top candidates

    Returns:
        Recall@K in [0, 1]
    """
    # For same K, recall equals overlap
    return overlap_at_k(proxy_scores, true_scores, k)
