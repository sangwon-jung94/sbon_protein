"""
Rank correlation metrics for reward hacking diagnostics.

These metrics measure how well the proxy ranking aligns with the true ranking.
Low correlation indicates potential reward hacking.
"""

import numpy as np
from scipy import stats
from typing import Tuple


def spearman_correlation(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
) -> float:
    """
    Compute Spearman rank correlation between proxy and true scores.

    Spearman correlation measures monotonic relationship between rankings.
    Value of 1.0 means perfect agreement, -1.0 means perfect disagreement,
    0.0 means no correlation.

    Args:
        proxy_scores: Array of proxy reward scores [N]
        true_scores: Array of true reward scores [N]

    Returns:
        Spearman correlation coefficient
    """
    if len(proxy_scores) != len(true_scores):
        raise ValueError("Score arrays must have same length")

    if len(proxy_scores) < 2:
        return np.nan

    corr, _ = stats.spearmanr(proxy_scores, true_scores)
    return float(corr)


def kendall_tau(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
) -> float:
    """
    Compute Kendall's tau rank correlation.

    Kendall's tau counts concordant and discordant pairs.
    More robust to outliers than Spearman.

    Args:
        proxy_scores: Array of proxy reward scores [N]
        true_scores: Array of true reward scores [N]

    Returns:
        Kendall's tau coefficient
    """
    if len(proxy_scores) != len(true_scores):
        raise ValueError("Score arrays must have same length")

    if len(proxy_scores) < 2:
        return np.nan

    tau, _ = stats.kendalltau(proxy_scores, true_scores)
    return float(tau)


def rank_correlation_with_pvalue(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
    method: str = "spearman",
) -> Tuple[float, float]:
    """
    Compute rank correlation with p-value.

    Args:
        proxy_scores: Array of proxy reward scores [N]
        true_scores: Array of true reward scores [N]
        method: 'spearman' or 'kendall'

    Returns:
        Tuple of (correlation, p-value)
    """
    if len(proxy_scores) != len(true_scores):
        raise ValueError("Score arrays must have same length")

    if len(proxy_scores) < 2:
        return np.nan, np.nan

    if method == "spearman":
        corr, pval = stats.spearmanr(proxy_scores, true_scores)
    elif method == "kendall":
        corr, pval = stats.kendalltau(proxy_scores, true_scores)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'kendall'.")

    return float(corr), float(pval)


def top_k_rank_correlation(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int,
    method: str = "spearman",
) -> float:
    """
    Compute rank correlation among top-K candidates (by proxy).

    This measures whether the proxy correctly orders the selected candidates.

    Args:
        proxy_scores: Array of proxy reward scores [N]
        true_scores: Array of true reward scores [N]
        k: Number of top candidates to consider
        method: 'spearman' or 'kendall'

    Returns:
        Rank correlation among top-K candidates
    """
    if k > len(proxy_scores):
        raise ValueError(f"k={k} > N={len(proxy_scores)}")

    # Get top-K indices by proxy
    topk_indices = np.argsort(proxy_scores)[-k:]

    # Compute correlation among top-K
    topk_proxy = proxy_scores[topk_indices]
    topk_true = true_scores[topk_indices]

    if method == "spearman":
        return spearman_correlation(topk_proxy, topk_true)
    elif method == "kendall":
        return kendall_tau(topk_proxy, topk_true)
    else:
        raise ValueError(f"Unknown method: {method}")
