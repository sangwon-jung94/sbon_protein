"""
Candidate data structures for protein design.

A Candidate represents a single generated protein with its sequence,
optional structure, and associated scores.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import torch


@dataclass
class Candidate:
    """
    A single protein candidate from the generator.

    Attributes:
        sequence: Amino acid sequence string (e.g., "MKFLILLFNILCLFPVLAADNHGVGPQGAS...")
        structure: Optional 3D coordinates, shape [L, 3] or [L, n_atoms, 3]
        length: Sequence length
        proxy_score: Score from fast proxy reward (populated after proxy scoring)
        true_score: Score from true oracle (populated after oracle scoring)
        metadata: Additional information (e.g., generator params, pdb_path, etc.)
    """
    sequence: str
    structure: Optional[np.ndarray] = None
    proxy_score: Optional[float] = None
    true_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:
        seq_preview = self.sequence[:20] + "..." if len(self.sequence) > 20 else self.sequence
        return (
            f"Candidate(seq='{seq_preview}', len={self.length}, "
            f"proxy={self.proxy_score}, true={self.true_score})"
        )


@dataclass
class CandidateBatch:
    """
    A batch of protein candidates.

    This is the main data structure passed between Generator -> Reward -> Selection.
    Designed to work with N candidates where we select Top-K.

    Attributes:
        candidates: List of Candidate objects
        proxy_scores: Array of proxy scores [N], None if not yet scored
        true_scores: Array of true scores [N], None if not yet scored
    """
    candidates: List[Candidate]

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx) -> Candidate:
        return self.candidates[idx]

    def __iter__(self):
        return iter(self.candidates)

    @property
    def sequences(self) -> List[str]:
        """Get all sequences as a list."""
        return [c.sequence for c in self.candidates]

    @property
    def proxy_scores(self) -> Optional[np.ndarray]:
        """Get proxy scores as numpy array, None if any candidate is unscored."""
        scores = [c.proxy_score for c in self.candidates]
        if any(s is None for s in scores):
            return None
        return np.array(scores, dtype=np.float32)

    @property
    def true_scores(self) -> Optional[np.ndarray]:
        """Get true scores as numpy array, None if any candidate is unscored."""
        scores = [c.true_score for c in self.candidates]
        if any(s is None for s in scores):
            return None
        return np.array(scores, dtype=np.float32)

    def set_proxy_scores(self, scores: np.ndarray) -> None:
        """Set proxy scores for all candidates."""
        assert len(scores) == len(self.candidates), "Score count must match candidate count"
        for c, s in zip(self.candidates, scores):
            c.proxy_score = float(s)

    def set_true_scores(self, scores: np.ndarray) -> None:
        """Set true scores for all candidates."""
        assert len(scores) == len(self.candidates), "Score count must match candidate count"
        for c, s in zip(self.candidates, scores):
            c.true_score = float(s)

    def select_indices(self, indices: np.ndarray) -> "CandidateBatch":
        """Return a new CandidateBatch with only the selected indices."""
        selected = [self.candidates[i] for i in indices]
        return CandidateBatch(candidates=selected)

    def topk_by_proxy(self, k: int) -> np.ndarray:
        """Return indices of top-k candidates by proxy score."""
        scores = self.proxy_scores
        if scores is None:
            raise ValueError("Proxy scores not set. Run proxy scoring first.")
        return np.argsort(scores)[-k:][::-1]  # descending order

    def topk_by_true(self, k: int) -> np.ndarray:
        """Return indices of top-k candidates by true score (for analysis)."""
        scores = self.true_scores
        if scores is None:
            raise ValueError("True scores not set. Run oracle scoring first.")
        return np.argsort(scores)[-k:][::-1]  # descending order

    @classmethod
    def from_sequences(cls, sequences: List[str]) -> "CandidateBatch":
        """Create a CandidateBatch from a list of sequences."""
        candidates = [Candidate(sequence=seq) for seq in sequences]
        return cls(candidates=candidates)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sequences": self.sequences,
            "proxy_scores": self.proxy_scores.tolist() if self.proxy_scores is not None else None,
            "true_scores": self.true_scores.tolist() if self.true_scores is not None else None,
            "metadata": [c.metadata for c in self.candidates],
        }
