"""
ESMFold oracle reward function.

ESMFold is a fast structure predictor that can be used as the true oracle.
It predicts pLDDT (predicted Local Distance Difference Test) scores.

Note: ESMFold is faster than AF2 but may be less accurate for some proteins.
For the most accurate results, use ColabFold (AF2) as the oracle.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import torch
from tqdm import tqdm

from topk_hedge.rewards.base import OracleReward
from topk_hedge.data.candidate import Candidate, CandidateBatch


class ESMFoldOracle(OracleReward):
    """
    ESMFold-based oracle reward function.

    Uses ESMFold to predict structure and returns mean pLDDT as the reward.
    pLDDT ranges from 0-100, normalized to 0-1.

    Usage:
        oracle = ESMFoldOracle()
        scores = oracle.score(batch)  # Returns pLDDT scores in [0, 1]
    """

    def __init__(
        self,
        device: str = "cuda",
        return_structures: bool = False,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize ESMFold oracle.

        Args:
            device: Device to run on
            return_structures: Whether to store predicted structures in candidates
            output_dir: Directory to save PDB files (optional)
        """
        self._device = device
        self.return_structures = return_structures
        self.output_dir = output_dir

        self._model = None
        self._initialized = False

        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)

    @property
    def name(self) -> str:
        return "esmfold-oracle"

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def _initialize(self):
        """Initialize ESMFold model (lazy loading)."""
        if self._initialized:
            return

        # Add reference path for openfold dependency
        import sys
        from pathlib import Path
        ref_path = str(Path(__file__).parent.parent.parent.parent / "reference")
        if ref_path not in sys.path:
            sys.path.insert(0, ref_path)

        # Try esm library first (produces correct results)
        try:
            import esm
            self._model = esm.pretrained.esmfold_v1().eval().to(self.device)
            self._use_esm = True
            self._initialized = True
            return
        except (ImportError, Exception) as e:
            # esm might fail due to openfold dependency
            pass

        # Fallback to transformers (may have weight initialization issues)
        try:
            from transformers import EsmForProteinFolding, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            self._model = EsmForProteinFolding.from_pretrained(
                "facebook/esmfold_v1",
                low_cpu_mem_usage=True,
            ).to(self.device)
            self._model.eval()
            self._use_esm = False
            self._initialized = True
            return
        except ImportError:
            pass

        raise ImportError(
            "ESMFold not available. Install with: pip install fair-esm"
        )

    def score(self, batch: CandidateBatch) -> np.ndarray:
        """
        Score a batch of candidates using ESMFold pLDDT.

        Args:
            batch: CandidateBatch containing N candidates

        Returns:
            np.ndarray of shape [N] with pLDDT scores in [0, 1]
        """
        self._initialize()

        scores = []
        for i, candidate in enumerate(tqdm(batch, desc="ESMFold scoring")):
            score = self.score_single(candidate)
            scores.append(score)

            # Save PDB if requested
            if self.output_dir and hasattr(candidate, "_last_pdb"):
                pdb_path = f"{self.output_dir}/sample_{i}.pdb"
                with open(pdb_path, "w") as f:
                    f.write(candidate._last_pdb)
                candidate.metadata["pdb_path"] = pdb_path

        return np.array(scores, dtype=np.float32)

    def score_single(self, candidate: Candidate) -> float:
        """
        Score a single candidate.

        Args:
            candidate: A single Candidate

        Returns:
            pLDDT score in [0, 1]
        """
        self._initialize()

        sequence = candidate.sequence

        # Replace unknown amino acids with alanine
        sequence = sequence.replace("X", "A").replace("U", "A").replace("O", "A")

        with torch.no_grad():
            if self._use_esm:
                # ESM library API (preferred - produces correct results)
                output = self._model.infer(sequence)
                mean_plddt = output["mean_plddt"][0].item()

                if self.return_structures:
                    pdb_str = self._model.output_to_pdb(output)[0]
                    candidate._last_pdb = pdb_str
                    coords = self._parse_pdb_coords(pdb_str)
                    candidate.structure = coords
            else:
                # Transformers API (fallback)
                inputs = self._tokenizer(
                    [sequence],
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)
                outputs = self._model(**inputs)

                # plddt is [B, L, num_bins] logits - need to convert to scores
                plddt_logits = outputs.plddt  # [B, L, 37]
                num_bins = plddt_logits.shape[-1]

                # Compute weighted average: softmax -> multiply by bin centers -> sum
                bin_centers = torch.linspace(0, 100, num_bins, device=plddt_logits.device)
                plddt_probs = torch.softmax(plddt_logits, dim=-1)  # [B, L, 37]
                plddt_scores = (plddt_probs * bin_centers).sum(dim=-1)  # [B, L]

                mean_plddt = plddt_scores.mean().item()

        # Normalize to [0, 1]
        return mean_plddt / 100.0

    def _parse_pdb_coords(self, pdb_str: str) -> np.ndarray:
        """Parse CA coordinates from PDB string."""
        coords = []
        for line in pdb_str.split("\n"):
            if line.startswith("ATOM") and " CA " in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
        return np.array(coords)

    def score_batch_efficient(
        self,
        sequences: List[str],
        batch_size: int = 1,
    ) -> np.ndarray:
        """
        Score multiple sequences efficiently.

        Note: ESMFold processes one sequence at a time internally,
        but this method handles batching for memory management.

        Args:
            sequences: List of amino acid sequences
            batch_size: Not used (ESMFold is single-sequence)

        Returns:
            np.ndarray of pLDDT scores in [0, 1]
        """
        self._initialize()

        scores = []
        for seq in tqdm(sequences, desc="ESMFold scoring"):
            seq = seq.replace("X", "A").replace("U", "A").replace("O", "A")
            with torch.no_grad():
                output = self._model.infer(seq)
                mean_plddt = output["mean_plddt"][0].item()
            scores.append(mean_plddt / 100.0)

        return np.array(scores, dtype=np.float32)

    def get_config(self) -> Dict[str, Any]:
        """Return oracle configuration."""
        return {
            "name": self.name,
            "device": self._device,
            "return_structures": self.return_structures,
        }


class ESMFoldPTMOracle(OracleReward):
    """
    ESMFold oracle that returns pTM (predicted TM-score) instead of pLDDT.

    pTM is a global quality metric, while pLDDT is per-residue.
    """

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._model = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "esmfold-ptm-oracle"

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def _initialize(self):
        if self._initialized:
            return
        import esm
        self._model = esm.pretrained.esmfold_v1().eval().to(self.device)
        self._initialized = True

    def score(self, batch: CandidateBatch) -> np.ndarray:
        self._initialize()
        scores = []
        for candidate in tqdm(batch, desc="ESMFold pTM scoring"):
            score = self.score_single(candidate)
            scores.append(score)
        return np.array(scores, dtype=np.float32)

    def score_single(self, candidate: Candidate) -> float:
        self._initialize()
        sequence = candidate.sequence.replace("X", "A")
        with torch.no_grad():
            output = self._model.infer(sequence)
            ptm = output["ptm"][0].item()
        return ptm  # Already in [0, 1]
