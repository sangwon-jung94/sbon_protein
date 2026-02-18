"""
Analysis and visualization tools for sweep results.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_regret_vs_n(
    df: pd.DataFrame,
    k_values: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    title: str = "Regret vs N",
):
    """
    Plot regret as a function of N for different K values.

    Args:
        df: Aggregated DataFrame with columns like 'n', 'k', 'regret_mean', 'regret_std'
        k_values: K values to plot (default: all)
        output_path: Path to save figure
        title: Plot title
    """
    if k_values is None:
        k_values = sorted(df["k"].unique())

    plt.figure(figsize=(10, 6))

    for k in k_values:
        subset = df[df["k"] == k].sort_values("n")
        n_vals = subset["n"].values
        mean_vals = subset["regret_mean"].values
        std_vals = subset["regret_std"].values

        plt.errorbar(
            n_vals,
            mean_vals,
            yerr=std_vals,
            marker="o",
            capsize=3,
            label=f"K={k}",
        )

    plt.xlabel("N (number of candidates)")
    plt.ylabel("Regret")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def plot_overlap_vs_n(
    df: pd.DataFrame,
    k_values: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    title: str = "Overlap@K vs N",
):
    """
    Plot overlap as a function of N for different K values.

    Args:
        df: Aggregated DataFrame
        k_values: K values to plot
        output_path: Path to save figure
        title: Plot title
    """
    if k_values is None:
        k_values = sorted(df["k"].unique())

    plt.figure(figsize=(10, 6))

    for k in k_values:
        subset = df[df["k"] == k].sort_values("n")
        n_vals = subset["n"].values
        mean_vals = subset["overlap_mean"].values
        std_vals = subset["overlap_std"].values

        plt.errorbar(
            n_vals,
            mean_vals,
            yerr=std_vals,
            marker="o",
            capsize=3,
            label=f"K={k}",
        )

    plt.xlabel("N (number of candidates)")
    plt.ylabel("Overlap@K")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.ylim(0, 1.05)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def plot_mean_true_vs_n(
    df: pd.DataFrame,
    k_values: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    title: str = "Mean True Reward of Proxy-Selected vs N",
):
    """
    Plot mean true reward of proxy-selected as a function of N.

    This is the key plot for detecting the "proxy extremal" effect.

    Args:
        df: Aggregated DataFrame
        k_values: K values to plot
        output_path: Path to save figure
        title: Plot title
    """
    if k_values is None:
        k_values = sorted(df["k"].unique())

    plt.figure(figsize=(10, 6))

    for k in k_values:
        subset = df[df["k"] == k].sort_values("n")
        n_vals = subset["n"].values
        mean_vals = subset["mean_true_selected_mean"].values
        std_vals = subset["mean_true_selected_std"].values

        plt.errorbar(
            n_vals,
            mean_vals,
            yerr=std_vals,
            marker="o",
            capsize=3,
            label=f"Proxy-selected K={k}",
        )

        # Also plot oracle upper bound
        oracle_vals = subset["mean_true_oracle_mean"].values
        plt.plot(
            n_vals,
            oracle_vals,
            linestyle="--",
            alpha=0.5,
            label=f"Oracle K={k}" if k == k_values[0] else None,
        )

    plt.xlabel("N (number of candidates)")
    plt.ylabel("Mean True Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def plot_correlation_vs_n(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Rank Correlation vs N",
):
    """
    Plot Spearman correlation as a function of N.

    Args:
        df: Aggregated DataFrame
        output_path: Path to save figure
        title: Plot title
    """
    # Correlation is same for all K (computed on full set)
    subset = df[df["k"] == df["k"].min()].sort_values("n")

    plt.figure(figsize=(10, 6))

    n_vals = subset["n"].values
    spearman_mean = subset["spearman_mean"].values
    spearman_std = subset["spearman_std"].values
    kendall_mean = subset["kendall_mean"].values
    kendall_std = subset["kendall_std"].values

    plt.errorbar(
        n_vals,
        spearman_mean,
        yerr=spearman_std,
        marker="o",
        capsize=3,
        label="Spearman",
    )
    plt.errorbar(
        n_vals,
        kendall_mean,
        yerr=kendall_std,
        marker="s",
        capsize=3,
        label="Kendall-tau",
    )

    plt.xlabel("N (number of candidates)")
    plt.ylabel("Rank Correlation")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.ylim(-0.1, 1.05)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: str = "./plots",
    prefix: str = "",
):
    """
    Generate all standard analysis plots.

    Args:
        df: Aggregated DataFrame from aggregate_sweep_results()
        output_dir: Directory to save plots
        prefix: Prefix for output filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_regret_vs_n(
        df,
        output_path=str(output_dir / f"{prefix}regret_vs_n.png"),
    )

    plot_overlap_vs_n(
        df,
        output_path=str(output_dir / f"{prefix}overlap_vs_n.png"),
    )

    plot_mean_true_vs_n(
        df,
        output_path=str(output_dir / f"{prefix}mean_true_vs_n.png"),
    )

    plot_correlation_vs_n(
        df,
        output_path=str(output_dir / f"{prefix}correlation_vs_n.png"),
    )

    print(f"All plots saved to {output_dir}")


def print_summary_table(df: pd.DataFrame):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)

    for k in sorted(df["k"].unique()):
        print(f"\n--- K = {k} ---")
        subset = df[df["k"] == k].sort_values("n")

        for _, row in subset.iterrows():
            print(
                f"N={int(row['n']):5d}: "
                f"overlap={row['overlap_mean']:.3f}±{row['overlap_std']:.3f}, "
                f"regret={row['regret_mean']:.4f}±{row['regret_std']:.4f}, "
                f"corr={row['spearman_mean']:.3f}"
            )

    print("\n" + "=" * 80)
