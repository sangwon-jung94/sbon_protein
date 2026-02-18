"""
Regret metrics for reward hacking diagnostics.

Regret measures the gap between proxy-selected and oracle-selected performance.
"""

import numpy as np
from typing import Tuple, Optional


def mean_true_reward(
    true_scores: np.ndarray,
    indices: np.ndarray,
) -> float:
    """
    Compute mean true reward of selected candidates.

    Args:
        true_scores: Array of true reward scores [N]
        indices: Indices of selected candidates

    Returns:
        Mean true reward of selected candidates
    """
    return float(true_scores[indices].mean())


def regret(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int,
) -> float:
    """
    Compute regret: gap between oracle-selected and proxy-selected performance.

    Regret = μ_true(TopK_true) - μ_true(TopK_proxy)

    A regret of 0 means proxy selection is optimal.
    Positive regret means proxy selection is suboptimal.

    Args:
        proxy_scores: Array of proxy reward scores [N]
        true_scores: Array of true reward scores [N]
        k: Number of top candidates

    Returns:
        Regret value (>= 0)
    """
    if len(proxy_scores) != len(true_scores):
        raise ValueError("Score arrays must have same length")

    n = len(proxy_scores)
    if k > n:
        raise ValueError(f"k={k} > N={n}")

    # Top-K indices by proxy
    topk_proxy = np.argsort(proxy_scores)[-k:]

    # Top-K indices by oracle (optimal)
    topk_true = np.argsort(true_scores)[-k:]

    # Mean true rewards
    mu_proxy_selected = true_scores[topk_proxy].mean()
    mu_oracle_selected = true_scores[topk_true].mean()

    return float(mu_oracle_selected - mu_proxy_selected)


def normalized_regret(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int,
) -> float:
    """
    Compute normalized regret in [0, 1].

    Normalized by the range of possible values:
    - 0 means optimal selection
    - 1 means worst possible selection

    Args:
        proxy_scores: Array of proxy reward scores [N]
        true_scores: Array of true reward scores [N]
        k: Number of top candidates

    Returns:
        Normalized regret in [0, 1]
    """
    if len(proxy_scores) != len(true_scores):
        raise ValueError("Score arrays must have same length")

    n = len(proxy_scores)
    if k > n:
        raise ValueError(f"k={k} > N={n}")

    # Oracle optimal (top-K by true)
    topk_true = np.argsort(true_scores)[-k:]
    mu_best = true_scores[topk_true].mean()

    # Worst case (bottom-K by true)
    bottomk_true = np.argsort(true_scores)[:k]
    mu_worst = true_scores[bottomk_true].mean()

    # Proxy selection
    topk_proxy = np.argsort(proxy_scores)[-k:]
    mu_proxy = true_scores[topk_proxy].mean()

    # Normalize
    if mu_best == mu_worst:
        return 0.0  # No variation in scores

    return float((mu_best - mu_proxy) / (mu_best - mu_worst))


def regret_vs_n(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int,
    n_values: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute regret as a function of N (number of candidates).

    This is useful for analyzing the "proxy extremal" effect:
    as N increases, do we see increasing regret?

    Args:
        proxy_scores: Array of proxy reward scores [N_max]
        true_scores: Array of true reward scores [N_max]
        k: Number of top candidates to select
        n_values: List of N values to evaluate (default: geometric progression)

    Returns:
        Tuple of (n_values, regret_values)
    """
    n_max = len(proxy_scores)

    if n_values is None:
        # Default: geometric progression from 2*k to n_max
        n_values = []
        n = max(2 * k, 10)
        while n <= n_max:
            n_values.append(n)
            n = int(n * 1.5)
        if n_values[-1] != n_max:
            n_values.append(n_max)
        n_values = np.array(n_values)
    else:
        n_values = np.array(n_values)

    regrets = []
    for n in n_values:
        if n < k:
            regrets.append(np.nan)
            continue

        # Use first n candidates
        r = regret(proxy_scores[:n], true_scores[:n], k)
        regrets.append(r)

    return n_values, np.array(regrets)


def mean_true_vs_n(
    proxy_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int,
    n_values: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean true reward of proxy-selected as a function of N.

    This shows the "proxy extremal" effect directly:
    does μ_true(TopK_proxy(N)) plateau or drop as N increases?

    Args:
        proxy_scores: Array of proxy reward scores [N_max]
        true_scores: Array of true reward scores [N_max]
        k: Number of top candidates to select
        n_values: List of N values to evaluate

    Returns:
        Tuple of (n_values, mean_true_values)
    """
    n_max = len(proxy_scores)

    if n_values is None:
        n_values = []
        n = max(2 * k, 10)
        while n <= n_max:
            n_values.append(n)
            n = int(n * 1.5)
        if n_values[-1] != n_max:
            n_values.append(n_max)
        n_values = np.array(n_values)
    else:
        n_values = np.array(n_values)

    mean_trues = []
    for n in n_values:
        if n < k:
            mean_trues.append(np.nan)
            continue

        # Select top-K by proxy from first n
        topk_proxy = np.argsort(proxy_scores[:n])[-k:]
        mu = true_scores[:n][topk_proxy].mean()
        mean_trues.append(mu)

    return n_values, np.array(mean_trues)
