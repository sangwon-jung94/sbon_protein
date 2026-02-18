"""
Abstract base class for protein sequence-structure generators.

Generators produce N protein candidates that will be scored by reward functions
and filtered via Top-K selection.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch

from topk_hedge.data.candidate import CandidateBatch


class Generator(ABC):
    """
    Abstract base class for protein generators.

    A generator produces protein candidates (sequence, optionally structure).
    Implementations include:
    - EvoDiff: Discrete diffusion model for sequences
    - La Proteina: NVIDIA's sequence-structure co-design model
    - MultiFlow: Flow-based sequence-structure model

    Usage:
        generator = EvoDiffGenerator(model_size="38M")
        batch = generator.generate(n_samples=100, seq_len=120)
    """

    @abstractmethod
    def generate(
        self,
        n_samples: int,
        seq_len: int,
        **kwargs,
    ) -> CandidateBatch:
        """
        Generate N protein candidates.

        Args:
            n_samples: Number of candidates to generate (N in Top-K-from-N)
            seq_len: Target sequence length
            **kwargs: Generator-specific parameters

        Returns:
            CandidateBatch containing N candidates
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this generator (e.g., 'evodiff-38M')."""
        pass

    @property
    def device(self) -> torch.device:
        """Return the device this generator runs on."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_with_seed(
        self,
        n_samples: int,
        seed_sequence: str,
        n_mutations: int,
        **kwargs,
    ) -> CandidateBatch:
        """
        Generate candidates starting from a seed sequence (optional).

        This is useful for protein optimization / refinement tasks.
        Default implementation raises NotImplementedError.

        Args:
            n_samples: Number of candidates to generate
            seed_sequence: Starting sequence
            n_mutations: Number of positions to mutate per iteration
            **kwargs: Generator-specific parameters

        Returns:
            CandidateBatch containing N candidates
        """
        raise NotImplementedError(
            f"{self.name} does not support seeded generation. "
            "Use generate() for de novo generation."
        )

    def get_config(self) -> Dict[str, Any]:
        """Return generator configuration for logging/reproducibility."""
        return {
            "name": self.name,
            "device": str(self.device),
        }
