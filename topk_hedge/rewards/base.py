"""
Abstract base class for reward functions.

Reward functions score protein candidates. We distinguish between:
- Proxy rewards: Fast but imperfect (e.g., learned pLDDT predictor)
- Oracle rewards: Slow but accurate (e.g., AlphaFold2 pLDDT with MSA)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
import torch

from topk_hedge.data.candidate import Candidate, CandidateBatch


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    A reward function takes protein candidates and returns scalar scores.
    Higher scores are better.

    Usage:
        proxy = LearnedProxyReward(checkpoint_path="model.pth")
        oracle = ColabFoldReward(msa_mode="single_sequence")

        # Score all candidates
        proxy_scores = proxy.score(batch)

        # Select top-K by proxy, evaluate with oracle
        topk_indices = batch.topk_by_proxy(k=10)
        selected_batch = batch.select_indices(topk_indices)
        true_scores = oracle.score(selected_batch)
    """

    @abstractmethod
    def score(self, batch: CandidateBatch) -> np.ndarray:
        """
        Score a batch of candidates.

        Args:
            batch: CandidateBatch containing N candidates

        Returns:
            np.ndarray of shape [N] with scores (higher is better)
        """
        pass

    @abstractmethod
    def score_single(self, candidate: Candidate) -> float:
        """
        Score a single candidate.

        Args:
            candidate: A single Candidate

        Returns:
            Scalar score (higher is better)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this reward function."""
        pass

    @property
    @abstractmethod
    def is_proxy(self) -> bool:
        """
        Return True if this is a fast proxy reward.
        Return False if this is a slow true oracle.
        """
        pass

    @property
    def device(self) -> torch.device:
        """Return the device this reward runs on."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def score_and_update(self, batch: CandidateBatch) -> np.ndarray:
        """
        Score candidates and update their scores in-place.

        Args:
            batch: CandidateBatch to score

        Returns:
            np.ndarray of scores
        """
        scores = self.score(batch)
        if self.is_proxy:
            batch.set_proxy_scores(scores)
        else:
            batch.set_true_scores(scores)
        return scores

    def get_config(self) -> Dict[str, Any]:
        """Return reward configuration for logging."""
        return {
            "name": self.name,
            "is_proxy": self.is_proxy,
            "device": str(self.device),
        }


class ProxyReward(RewardFunction):
    """Base class for proxy (fast) reward functions."""

    @property
    def is_proxy(self) -> bool:
        return True


class OracleReward(RewardFunction):
    """Base class for oracle (true) reward functions."""

    @property
    def is_proxy(self) -> bool:
        return False
