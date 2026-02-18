"""
Experiment runner for N, K sweeps.

This module runs systematic experiments to analyze reward hacking:
- Sweep over different N (number of candidates to generate)
- Sweep over different K (number to select)
- Compute metrics at each (N, K) pair
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from topk_hedge.generators.base import Generator
from topk_hedge.rewards.base import RewardFunction
from topk_hedge.selection.topk_selector import TopKSelector
from topk_hedge.data.candidate import CandidateBatch
from topk_hedge.metrics import (
    spearman_correlation,
    kendall_tau,
    overlap_at_k,
    regret,
)


@dataclass
class SweepConfig:
    """Configuration for N, K sweep experiment."""

    n_values: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    seq_len: int = 100
    n_seeds: int = 3
    output_dir: str = "./sweep_results"

    def __post_init__(self):
        # Validate
        for k in self.k_values:
            if k > min(self.n_values):
                raise ValueError(
                    f"k={k} > min(n_values)={min(self.n_values)}. "
                    f"All k values must be <= min(n_values)."
                )


@dataclass
class SweepResult:
    """Result of a single (N, K, seed) experiment."""

    n: int
    k: int
    seed: int

    # Rank correlations
    spearman: float
    kendall: float

    # Overlap
    overlap: float

    # Regret
    regret: float
    normalized_regret: float

    # Mean rewards
    mean_proxy_selected: float
    mean_true_selected: float
    mean_true_oracle: float

    # Additional info
    proxy_scores: Optional[np.ndarray] = None
    true_scores: Optional[np.ndarray] = None


class SweepRunner:
    """
    Runner for N, K sweep experiments.

    Usage:
        runner = SweepRunner(
            generator=EvoDiffGenerator(),
            proxy_reward=LearnedProxyReward(),
            oracle_reward=ESMFoldOracle(),
        )

        config = SweepConfig(
            n_values=[100, 200, 500],
            k_values=[5, 10, 20],
            n_seeds=3,
        )

        results_df = runner.run_sweep(config)
    """

    def __init__(
        self,
        generator: Generator,
        proxy_reward: RewardFunction,
        oracle_reward: RewardFunction,
    ):
        """
        Initialize sweep runner.

        Args:
            generator: Protein generator
            proxy_reward: Fast proxy reward function
            oracle_reward: True oracle reward function
        """
        self.generator = generator
        self.proxy_reward = proxy_reward
        self.oracle_reward = oracle_reward
        self.selector = TopKSelector(proxy_reward, oracle_reward)

    def run_sweep(
        self,
        config: SweepConfig,
        save_scores: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run full N, K sweep.

        Args:
            config: Sweep configuration
            save_scores: Whether to save full score arrays
            show_progress: Show progress bars

        Returns:
            DataFrame with results for all (N, K, seed) combinations
        """
        results = []
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Total iterations for progress bar
        total = len(config.n_values) * len(config.k_values) * config.n_seeds
        pbar = tqdm(total=total, disable=not show_progress, desc="Sweep")

        for seed in range(config.n_seeds):
            # Set seed for reproducibility
            np.random.seed(seed)

            for n in config.n_values:
                # Generate N candidates
                batch = self.generator.generate(
                    n_samples=n,
                    seq_len=config.seq_len,
                    show_progress=False,
                )

                # Score all with proxy and oracle
                proxy_scores = self.proxy_reward.score_and_update(batch)
                true_scores = self.oracle_reward.score_and_update(batch)

                for k in config.k_values:
                    if k > n:
                        pbar.update(1)
                        continue

                    result = self._compute_metrics(
                        proxy_scores=proxy_scores,
                        true_scores=true_scores,
                        n=n,
                        k=k,
                        seed=seed,
                    )

                    if save_scores:
                        result.proxy_scores = proxy_scores.copy()
                        result.true_scores = true_scores.copy()

                    results.append(result)
                    pbar.update(1)

        pbar.close()

        # Convert to DataFrame
        df = self._results_to_dataframe(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(output_dir / f"sweep_results_{timestamp}.csv", index=False)

        # Save config
        with open(output_dir / f"sweep_config_{timestamp}.json", "w") as f:
            json.dump({
                "n_values": config.n_values,
                "k_values": config.k_values,
                "seq_len": config.seq_len,
                "n_seeds": config.n_seeds,
                "generator": self.generator.name,
                "proxy": self.proxy_reward.name,
                "oracle": self.oracle_reward.name,
            }, f, indent=2)

        return df

    def _compute_metrics(
        self,
        proxy_scores: np.ndarray,
        true_scores: np.ndarray,
        n: int,
        k: int,
        seed: int,
    ) -> SweepResult:
        """Compute all metrics for a single (N, K, seed) experiment."""
        from topk_hedge.metrics.regret import normalized_regret

        # Top-K indices
        topk_proxy = np.argsort(proxy_scores)[-k:][::-1]
        topk_true = np.argsort(true_scores)[-k:][::-1]

        return SweepResult(
            n=n,
            k=k,
            seed=seed,
            spearman=spearman_correlation(proxy_scores, true_scores),
            kendall=kendall_tau(proxy_scores, true_scores),
            overlap=overlap_at_k(proxy_scores, true_scores, k),
            regret=regret(proxy_scores, true_scores, k),
            normalized_regret=normalized_regret(proxy_scores, true_scores, k),
            mean_proxy_selected=float(proxy_scores[topk_proxy].mean()),
            mean_true_selected=float(true_scores[topk_proxy].mean()),
            mean_true_oracle=float(true_scores[topk_true].mean()),
        )

    def _results_to_dataframe(self, results: List[SweepResult]) -> pd.DataFrame:
        """Convert list of results to DataFrame."""
        rows = []
        for r in results:
            rows.append({
                "n": r.n,
                "k": r.k,
                "seed": r.seed,
                "spearman": r.spearman,
                "kendall": r.kendall,
                "overlap": r.overlap,
                "regret": r.regret,
                "normalized_regret": r.normalized_regret,
                "mean_proxy_selected": r.mean_proxy_selected,
                "mean_true_selected": r.mean_true_selected,
                "mean_true_oracle": r.mean_true_oracle,
            })
        return pd.DataFrame(rows)

    def run_single(
        self,
        n: int,
        k: int,
        seq_len: int = 100,
        seed: int = 0,
    ) -> SweepResult:
        """
        Run a single experiment.

        Args:
            n: Number of candidates to generate
            k: Number to select
            seq_len: Sequence length
            seed: Random seed

        Returns:
            SweepResult with all metrics
        """
        np.random.seed(seed)

        # Generate
        batch = self.generator.generate(n_samples=n, seq_len=seq_len)

        # Score
        proxy_scores = self.proxy_reward.score_and_update(batch)
        true_scores = self.oracle_reward.score_and_update(batch)

        # Compute metrics
        return self._compute_metrics(proxy_scores, true_scores, n, k, seed)


def aggregate_sweep_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sweep results over seeds.

    Args:
        df: DataFrame from SweepRunner.run_sweep()

    Returns:
        Aggregated DataFrame with mean and std for each (N, K)
    """
    metrics = [
        "spearman", "kendall", "overlap", "regret", "normalized_regret",
        "mean_proxy_selected", "mean_true_selected", "mean_true_oracle"
    ]

    agg_funcs = {m: ["mean", "std"] for m in metrics}
    agg_df = df.groupby(["n", "k"]).agg(agg_funcs).reset_index()

    # Flatten column names
    agg_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in agg_df.columns
    ]

    return agg_df
