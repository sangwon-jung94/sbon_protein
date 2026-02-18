"""
EvoDiff generator wrapper.

EvoDiff is a discrete diffusion model for protein sequence generation.
It generates amino acid sequences that can be folded with ESMFold or AF2.

Reference:
- Paper: https://arxiv.org/abs/2306.04818
- Code: https://github.com/microsoft/evodiff
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from tqdm import tqdm

from topk_hedge.generators.base import Generator
from topk_hedge.data.candidate import Candidate, CandidateBatch


class EvoDiffGenerator(Generator):
    """
    EvoDiff generator for protein sequence generation.

    Uses order-agnostic autoregressive diffusion model (OA-DM) to generate
    protein sequences. Does NOT generate structures - use a folding model
    (ESMFold, AF2) to predict structures from sequences.

    Usage:
        generator = EvoDiffGenerator(model_size="38M")
        batch = generator.generate(n_samples=100, seq_len=120)

    Model sizes:
        - "38M": Smaller, faster model
        - "640M": Larger, higher quality model
    """

    ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    MASK_TOKEN = "X"

    def __init__(
        self,
        model_size: str = "38M",
        device: str = "cuda",
        reference_path: Optional[str] = None,
    ):
        """
        Initialize EvoDiff generator.

        Args:
            model_size: Model size - "38M" or "640M"
            device: Device to run on
            reference_path: Path to reference code (for imports)
        """
        self.model_size = model_size
        self._device = device
        self.reference_path = reference_path

        self._model = None
        self._tokenizer = None
        self._collater = None
        self._initialized = False

        if model_size not in ["38M", "640M"]:
            raise ValueError(f"model_size must be '38M' or '640M', got '{model_size}'")

    @property
    def name(self) -> str:
        return f"evodiff-{self.model_size}"

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def _initialize(self):
        """Initialize the model (lazy loading)."""
        if self._initialized:
            return

        # Add reference path if provided
        if self.reference_path:
            sys.path.insert(0, self.reference_path)

        try:
            from evodiff.pretrained import OA_DM_38M, OA_DM_640M
        except ImportError:
            # Try from reference directory
            ref_path = Path(__file__).parent.parent.parent / "reference"
            sys.path.insert(0, str(ref_path))
            from evodiff.pretrained import OA_DM_38M, OA_DM_640M

        # Load model
        if self.model_size == "38M":
            checkpoint = OA_DM_38M()
        else:
            checkpoint = OA_DM_640M()

        self._model, self._collater, self._tokenizer, self._scheme = checkpoint
        self._model = self._model.eval().to(self.device)
        self._initialized = True

    def generate(
        self,
        n_samples: int,
        seq_len: int,
        batch_size: int = 10,
        show_progress: bool = True,
        **kwargs,
    ) -> CandidateBatch:
        """
        Generate N protein sequences.

        Args:
            n_samples: Number of sequences to generate
            seq_len: Target sequence length
            batch_size: Batch size for generation
            show_progress: Show progress bar
            **kwargs: Additional generation parameters

        Returns:
            CandidateBatch containing N candidates (sequences only, no structures)
        """
        self._initialize()

        candidates = []
        n_batches = (n_samples + batch_size - 1) // batch_size

        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating with {self.name}")

        for batch_idx in iterator:
            current_batch_size = min(batch_size, n_samples - len(candidates))

            # Generate batch
            samples, sequences = self._generate_batch(
                current_batch_size,
                seq_len,
                **kwargs,
            )

            # Create candidates
            for seq in sequences:
                candidate = Candidate(
                    sequence=seq,
                    structure=None,
                    metadata={
                        "generator": self.name,
                        "seq_len": seq_len,
                    },
                )
                candidates.append(candidate)

        return CandidateBatch(candidates=candidates)

    def _generate_batch(
        self,
        batch_size: int,
        seq_len: int,
        **kwargs,
    ) -> tuple:
        """Generate a batch of sequences using OA-DM."""
        mask = self._tokenizer.mask_id
        all_aas = self._tokenizer.all_aas

        # Start from all mask tokens
        sample = torch.zeros((batch_size, seq_len), dtype=torch.long) + mask
        sample = sample.to(self.device)

        # Random unmasking order
        loc = np.arange(seq_len)
        np.random.shuffle(loc)

        # Timestep placeholder (not used in OA-DM)
        timestep = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for i in loc:
                prediction = self._model(sample, timestep)

                # Get probabilities for position i (exclude non-standard AAs)
                p = prediction[:, i, : len(all_aas) - 6]
                p = torch.nn.functional.softmax(p, dim=1)

                # Sample
                p_sample = torch.multinomial(p, num_samples=1)
                sample[:, i] = p_sample.squeeze()

        # Untokenize
        sequences = [self._tokenizer.untokenize(s) for s in sample]

        return sample, sequences

    def generate_with_seed(
        self,
        n_samples: int,
        seed_sequence: str,
        n_mutations: int,
        n_iterations: int = 10,
        **kwargs,
    ) -> CandidateBatch:
        """
        Generate candidates starting from a seed sequence.

        Uses iterative refinement: mask some positions, refill, repeat.

        Args:
            n_samples: Number of candidates to generate
            seed_sequence: Starting sequence
            n_mutations: Number of positions to mutate per iteration
            n_iterations: Number of refinement iterations
            **kwargs: Additional parameters

        Returns:
            CandidateBatch containing N refined candidates
        """
        self._initialize()

        seq_len = len(seed_sequence)
        mask = self._tokenizer.mask_id

        # Tokenize seed sequence
        seed_tokens = torch.from_numpy(
            self._tokenizer.tokenize([seed_sequence])
        ).to(self.device)
        sample = seed_tokens.repeat(n_samples, 1)

        timestep = torch.zeros(n_samples, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for _ in range(n_iterations):
                # Select positions to mutate
                for i in range(n_samples):
                    mask_positions = np.random.choice(
                        seq_len, size=n_mutations, replace=False
                    )
                    sample[i, mask_positions] = mask

                # Refill masked positions
                mask_positions_all = (sample == mask).nonzero(as_tuple=True)
                unique_positions = mask_positions_all[1].unique()

                for pos in unique_positions:
                    prediction = self._model(sample, timestep)
                    p = prediction[:, pos, : len(self._tokenizer.all_aas) - 6]
                    p = torch.nn.functional.softmax(p, dim=1)
                    p_sample = torch.multinomial(p, num_samples=1)

                    # Update only masked positions
                    mask_at_pos = sample[:, pos] == mask
                    sample[mask_at_pos, pos] = p_sample[mask_at_pos].squeeze()

        # Untokenize
        sequences = [self._tokenizer.untokenize(s) for s in sample]

        candidates = [
            Candidate(
                sequence=seq,
                structure=None,
                metadata={
                    "generator": self.name,
                    "seed_sequence": seed_sequence,
                    "n_mutations": n_mutations,
                    "n_iterations": n_iterations,
                },
            )
            for seq in sequences
        ]

        return CandidateBatch(candidates=candidates)

    def get_config(self) -> Dict[str, Any]:
        """Return generator configuration."""
        return {
            "name": self.name,
            "model_size": self.model_size,
            "device": self._device,
        }
