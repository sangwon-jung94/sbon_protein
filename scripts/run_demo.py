#!/usr/bin/env python3
"""
Demo script to test the TopK-Hedge pipeline.

This script:
1. Generates proteins using EvoDiff
2. Scores with ESMFold (as both proxy and oracle for demo)
3. Computes reward hacking metrics
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from topk_hedge.generators.evodiff_gen import EvoDiffGenerator
from topk_hedge.rewards.oracle.esmfold_oracle import ESMFoldOracle
from topk_hedge.rewards.proxy.learned_proxy import LearnedProxyReward
from topk_hedge.selection.topk_selector import TopKSelector
from topk_hedge.metrics import spearman_correlation, overlap_at_k, regret


def main():
    print("=" * 60)
    print("TopK-Hedge Demo")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Parameters
    n_samples = 20  # Number of candidates to generate
    k = 5  # Number to select
    seq_len = 50  # Sequence length (short for demo)

    # Step 1: Initialize generator
    print("\n[1] Initializing EvoDiff generator...")
    try:
        generator = EvoDiffGenerator(
            model_size="38M",
            device=device,
            reference_path=str(Path(__file__).parent.parent / "reference"),
        )
        print(f"    Generator: {generator.name}")
    except Exception as e:
        print(f"    Failed to load EvoDiff: {e}")
        print("    Falling back to random sequence generation...")
        generator = None

    # Step 2: Generate candidates
    print(f"\n[2] Generating {n_samples} candidates (seq_len={seq_len})...")
    if generator:
        batch = generator.generate(n_samples=n_samples, seq_len=seq_len)
    else:
        # Fallback: random sequences
        from topk_hedge.data.candidate import CandidateBatch, Candidate
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        candidates = []
        for _ in range(n_samples):
            seq = "".join(np.random.choice(list(alphabet), size=seq_len))
            candidates.append(Candidate(sequence=seq))
        batch = CandidateBatch(candidates=candidates)

    print(f"    Generated {len(batch)} candidates")
    print(f"    Example sequence: {batch[0].sequence[:30]}...")

    # Step 3: Initialize oracle
    print("\n[3] Initializing ESMFold oracle...")
    try:
        oracle = ESMFoldOracle(device=device)
        print(f"    Oracle: {oracle.name}")
    except Exception as e:
        print(f"    Failed to load ESMFold: {e}")
        oracle = None

    # Step 4: Score with oracle
    if oracle:
        print(f"\n[4] Scoring {n_samples} candidates with ESMFold...")
        true_scores = oracle.score_and_update(batch)
        print(f"    Mean pLDDT: {true_scores.mean():.3f}")
        print(f"    Min pLDDT: {true_scores.min():.3f}")
        print(f"    Max pLDDT: {true_scores.max():.3f}")
    else:
        # Fallback: random scores
        print("\n[4] Using random scores (ESMFold not available)...")
        true_scores = np.random.uniform(0.3, 0.8, size=n_samples)
        batch.set_true_scores(true_scores)

    # Step 5: Create a "noisy proxy" for demo
    print("\n[5] Creating noisy proxy scores...")
    noise = np.random.normal(0, 0.15, size=n_samples)
    proxy_scores = np.clip(true_scores + noise, 0, 1)
    batch.set_proxy_scores(proxy_scores)
    print(f"    Proxy-True correlation: {np.corrcoef(proxy_scores, true_scores)[0, 1]:.3f}")

    # Step 6: Compute metrics
    print(f"\n[6] Computing metrics for Top-{k} selection...")

    spearman = spearman_correlation(proxy_scores, true_scores)
    overlap = overlap_at_k(proxy_scores, true_scores, k)
    reg = regret(proxy_scores, true_scores, k)

    topk_proxy = np.argsort(proxy_scores)[-k:][::-1]
    topk_true = np.argsort(true_scores)[-k:][::-1]

    mean_true_proxy_selected = true_scores[topk_proxy].mean()
    mean_true_oracle_selected = true_scores[topk_true].mean()

    print(f"\n    Results:")
    print(f"    - Spearman correlation: {spearman:.3f}")
    print(f"    - Overlap@{k}: {overlap:.3f} ({int(overlap * k)}/{k} overlap)")
    print(f"    - Regret: {reg:.4f}")
    print(f"    - Mean true (proxy-selected): {mean_true_proxy_selected:.3f}")
    print(f"    - Mean true (oracle-selected): {mean_true_oracle_selected:.3f}")

    # Step 7: Show selected sequences
    print(f"\n[7] Top-{k} by proxy:")
    for i, idx in enumerate(topk_proxy):
        in_oracle = "(*)" if idx in topk_true else ""
        print(
            f"    {i+1}. idx={idx}: proxy={proxy_scores[idx]:.3f}, "
            f"true={true_scores[idx]:.3f} {in_oracle}"
        )

    print(f"\n    (*) = also in oracle Top-{k}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
