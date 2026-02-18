"""
Top-K selection from N candidates.

This module implements the core selection logic:
1. Generate N candidates
2. Score all N with proxy reward
3. Select Top-K by proxy score
4. (In deployment) Evaluate selected K with true oracle
5. (In analysis mode) Evaluate all N to quantify reward hacking
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from topk_hedge.data.candidate import CandidateBatch
from topk_hedge.rewards.base import RewardFunction


@dataclass
class SelectionResult:
    """
    Result of Top-K selection.

    Attributes:
        selected_indices: Indices of K selected candidates (sorted by proxy score desc)
        all_proxy_scores: Proxy scores for all N candidates
        all_true_scores: True scores for all N candidates (analysis mode only)
        k: Number of candidates selected
        n: Total number of candidates
    """
    selected_indices: np.ndarray
    all_proxy_scores: np.ndarray
    all_true_scores: Optional[np.ndarray] = None
    k: int = 0
    n: int = 0

    @property
    def selected_proxy_scores(self) -> np.ndarray:
        """Proxy scores of selected candidates."""
        return self.all_proxy_scores[self.selected_indices]

    @property
    def selected_true_scores(self) -> Optional[np.ndarray]:
        """True scores of selected candidates (if available)."""
        if self.all_true_scores is None:
            return None
        return self.all_true_scores[self.selected_indices]

    @property
    def oracle_selected_indices(self) -> Optional[np.ndarray]:
        """Indices that would be selected if we had oracle access (for analysis)."""
        if self.all_true_scores is None:
            return None
        return np.argsort(self.all_true_scores)[-self.k:][::-1]

    def mean_proxy_of_selected(self) -> float:
        """Mean proxy score of proxy-selected candidates."""
        return float(self.selected_proxy_scores.mean())

    def mean_true_of_selected(self) -> Optional[float]:
        """Mean true score of proxy-selected candidates."""
        if self.selected_true_scores is None:
            return None
        return float(self.selected_true_scores.mean())

    def mean_true_of_oracle_selected(self) -> Optional[float]:
        """Mean true score of oracle-selected candidates (upper bound)."""
        if self.all_true_scores is None:
            return None
        oracle_indices = self.oracle_selected_indices
        return float(self.all_true_scores[oracle_indices].mean())


class TopKSelector:
    """
    Top-K from N selector.

    This class orchestrates the selection process:
    1. Score all candidates with proxy
    2. Select top-K by proxy score
    3. Optionally score all/selected with oracle for analysis

    Usage:
        selector = TopKSelector(proxy_reward, oracle_reward)

        # Generate candidates
        batch = generator.generate(n_samples=1000, seq_len=100)

        # Select top-10
        result = selector.select(batch, k=10)

        # In analysis mode, get all true scores
        result = selector.select(batch, k=10, evaluate_all_with_oracle=True)
    """

    def __init__(
        self,
        proxy_reward: RewardFunction,
        oracle_reward: Optional[RewardFunction] = None,
    ):
        """
        Initialize selector.

        Args:
            proxy_reward: Fast proxy reward for initial selection
            oracle_reward: True oracle reward for evaluation (optional)
        """
        if not proxy_reward.is_proxy:
            raise ValueError("proxy_reward must be a proxy (is_proxy=True)")
        if oracle_reward is not None and oracle_reward.is_proxy:
            raise ValueError("oracle_reward must be an oracle (is_proxy=False)")

        self.proxy_reward = proxy_reward
        self.oracle_reward = oracle_reward

    def select(
        self,
        batch: CandidateBatch,
        k: int,
        evaluate_all_with_oracle: bool = False,
    ) -> SelectionResult:
        """
        Select top-K candidates from batch.

        Args:
            batch: CandidateBatch with N candidates
            k: Number of candidates to select
            evaluate_all_with_oracle: If True, score ALL N candidates with oracle
                                      (expensive, for analysis/research only)

        Returns:
            SelectionResult with selection details and scores
        """
        n = len(batch)
        if k > n:
            raise ValueError(f"k={k} > n={n}. Cannot select more than available.")

        # Step 1: Score all with proxy
        proxy_scores = self.proxy_reward.score_and_update(batch)

        # Step 2: Select top-K by proxy
        selected_indices = np.argsort(proxy_scores)[-k:][::-1]

        # Step 3: Optionally evaluate with oracle
        all_true_scores = None
        if self.oracle_reward is not None:
            if evaluate_all_with_oracle:
                # Analysis mode: score all N
                all_true_scores = self.oracle_reward.score_and_update(batch)
            else:
                # Deployment mode: only score selected K
                selected_batch = batch.select_indices(selected_indices)
                selected_true = self.oracle_reward.score(selected_batch)
                # Create full array with NaN for unscored
                all_true_scores = np.full(n, np.nan)
                all_true_scores[selected_indices] = selected_true

        return SelectionResult(
            selected_indices=selected_indices,
            all_proxy_scores=proxy_scores,
            all_true_scores=all_true_scores,
            k=k,
            n=n,
        )

    def select_multiple_k(
        self,
        batch: CandidateBatch,
        k_values: list,
        evaluate_all_with_oracle: bool = True,
    ) -> Dict[int, SelectionResult]:
        """
        Run selection for multiple K values (for sweep experiments).

        Efficient: only scores once, reuses for all K values.

        Args:
            batch: CandidateBatch with N candidates
            k_values: List of K values to evaluate
            evaluate_all_with_oracle: If True, score all with oracle

        Returns:
            Dict mapping K -> SelectionResult
        """
        n = len(batch)

        # Score once
        proxy_scores = self.proxy_reward.score_and_update(batch)

        all_true_scores = None
        if self.oracle_reward is not None and evaluate_all_with_oracle:
            all_true_scores = self.oracle_reward.score_and_update(batch)

        # Create results for each K
        results = {}
        for k in k_values:
            if k > n:
                continue
            selected_indices = np.argsort(proxy_scores)[-k:][::-1]
            results[k] = SelectionResult(
                selected_indices=selected_indices,
                all_proxy_scores=proxy_scores.copy(),
                all_true_scores=all_true_scores.copy() if all_true_scores is not None else None,
                k=k,
                n=n,
            )

        return results
