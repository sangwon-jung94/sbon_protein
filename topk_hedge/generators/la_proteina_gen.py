"""
La Proteina generator wrapper.

La Proteina (NVIDIA) is a latent diffusion model for protein sequence-structure
co-design. It generates both amino acid sequences and full-atom 3D structures.

Reference:
- Paper: https://arxiv.org/pdf/2507.09466
- Code: https://github.com/NVIDIA-Digital-Bio/la-proteina

Requirements:
- Clone la-proteina repo and install dependencies
- Download checkpoints to ./checkpoints_laproteina
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch

from topk_hedge.generators.base import Generator
from topk_hedge.data.candidate import Candidate, CandidateBatch


class LaProteinaGenerator(Generator):
    """
    La Proteina generator for protein sequence-structure co-design.

    This generator produces proteins with both sequences and 3D structures.
    It uses NVIDIA's latent diffusion model trained on protein structures.

    Usage:
        generator = LaProteinaGenerator(
            repo_path="/path/to/la-proteina",
            checkpoint_dir="/path/to/checkpoints_laproteina",
            model_variant="LD1",  # or LD2, LD3 for different sizes
        )
        batch = generator.generate(n_samples=100, seq_len=150)

    Note:
        Requires la-proteina repository and downloaded checkpoints.
        See setup_instructions() for installation guide.
    """

    SUPPORTED_VARIANTS = ["LD1", "LD2", "LD3", "LD4", "LD5", "LD6", "LD7"]
    UNCONDITIONAL_VARIANTS = ["LD1", "LD2", "LD3"]
    MOTIF_VARIANTS = ["LD4", "LD5", "LD6", "LD7"]

    def __init__(
        self,
        repo_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        model_variant: str = "LD1",
        device: str = "cuda",
        output_dir: str = "./laproteina_outputs",
    ):
        """
        Initialize La Proteina generator.

        Args:
            repo_path: Path to cloned la-proteina repository
            checkpoint_dir: Path to downloaded checkpoints
            model_variant: Model variant (LD1-LD7)
            device: Device to run on
            output_dir: Directory to save generated structures
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.model_variant = model_variant
        self._device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if model_variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unknown model variant: {model_variant}. "
                f"Supported: {self.SUPPORTED_VARIANTS}"
            )

        self._model = None
        self._initialized = False

    @property
    def name(self) -> str:
        return f"la-proteina-{self.model_variant}"

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def _check_installation(self) -> bool:
        """Check if La Proteina is properly installed."""
        if self.repo_path is None:
            return False
        if not self.repo_path.exists():
            return False
        if not (self.repo_path / "proteinfoundation" / "generate.py").exists():
            return False
        return True

    def _initialize(self):
        """Initialize the model (lazy loading)."""
        if self._initialized:
            return

        if not self._check_installation():
            raise RuntimeError(
                f"La Proteina not properly installed. "
                f"Run LaProteinaGenerator.setup_instructions() for help."
            )

        # Add repo to path
        sys.path.insert(0, str(self.repo_path))

        # Import la-proteina modules
        try:
            from proteinfoundation.models import load_model
            from proteinfoundation.sampling import sample_unconditional
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import La Proteina modules: {e}. "
                f"Make sure the environment is properly set up."
            )

        # Load model based on variant
        config_name = self._get_config_name()
        self._model = load_model(config_name, self.checkpoint_dir, self._device)
        self._sample_fn = sample_unconditional
        self._initialized = True

    def _get_config_name(self) -> str:
        """Get config name for the model variant."""
        if self.model_variant in self.UNCONDITIONAL_VARIANTS:
            return "inference_ucond_notri"
        else:
            return "inference_motif_idx_aa"

    def generate(
        self,
        n_samples: int,
        seq_len: int,
        batch_size: int = 10,
        **kwargs,
    ) -> CandidateBatch:
        """
        Generate N protein candidates.

        Args:
            n_samples: Number of candidates to generate
            seq_len: Target sequence length (approximate, model may vary)
            batch_size: Batch size for generation
            **kwargs: Additional generation parameters

        Returns:
            CandidateBatch containing N candidates with sequences and structures
        """
        self._initialize()

        candidates = []
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)

            # Generate samples
            samples = self._sample_fn(
                self._model,
                n_samples=current_batch_size,
                target_length=seq_len,
                device=self._device,
                **kwargs,
            )

            # Process each sample
            for i, sample in enumerate(samples):
                seq = sample.get("sequence", "")
                coords = sample.get("coordinates", None)

                # Save PDB if coordinates available
                pdb_path = None
                if coords is not None:
                    pdb_path = self.output_dir / f"sample_{batch_idx}_{i}.pdb"
                    self._save_pdb(seq, coords, pdb_path)

                candidate = Candidate(
                    sequence=seq,
                    structure=coords,
                    metadata={
                        "generator": self.name,
                        "pdb_path": str(pdb_path) if pdb_path else None,
                        "target_length": seq_len,
                    },
                )
                candidates.append(candidate)

        return CandidateBatch(candidates=candidates)

    def _save_pdb(self, sequence: str, coords: np.ndarray, path: Path):
        """Save structure as PDB file."""
        # Simple PDB writer for CA atoms
        with open(path, "w") as f:
            for i, (aa, coord) in enumerate(zip(sequence, coords)):
                f.write(
                    f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"  1.00  0.00           C\n"
                )
            f.write("END\n")

    def get_config(self) -> Dict[str, Any]:
        """Return generator configuration."""
        return {
            "name": self.name,
            "repo_path": str(self.repo_path) if self.repo_path else None,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "model_variant": self.model_variant,
            "device": self._device,
        }

    @staticmethod
    def setup_instructions() -> str:
        """Return installation instructions."""
        return """
La Proteina Setup Instructions
==============================

1. Clone the repository:
   git clone https://github.com/NVIDIA-Digital-Bio/la-proteina.git
   cd la-proteina

2. Create conda environment:
   mamba env create -f environment.yaml
   mamba activate laproteina_env

3. Install additional dependencies:
   pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
   pip install graphein==1.7.7 --no-deps
   pip install torch_geometric torch_scatter torch_sparse torch_cluster \\
       -f https://data.pyg.org/whl/torch-2.7.0+cu118.html

4. Download checkpoints:
   - Go to NVIDIA NGC catalog
   - Download model checkpoints
   - Place in ./checkpoints_laproteina directory

5. Usage:
   from topk_hedge.generators.la_proteina_gen import LaProteinaGenerator

   generator = LaProteinaGenerator(
       repo_path="/path/to/la-proteina",
       checkpoint_dir="/path/to/checkpoints_laproteina",
       model_variant="LD1",
   )
   batch = generator.generate(n_samples=100, seq_len=150)
"""


class LaProteinaGeneratorSimple(Generator):
    """
    Simplified La Proteina generator using subprocess calls.

    This is a fallback when direct Python import is problematic.
    Runs la-proteina as a subprocess and parses output files.
    """

    def __init__(
        self,
        repo_path: str,
        checkpoint_dir: str,
        output_dir: str = "./laproteina_outputs",
        python_executable: str = "python",
    ):
        self.repo_path = Path(repo_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.python_executable = python_executable
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "la-proteina-subprocess"

    def generate(
        self,
        n_samples: int,
        seq_len: int,
        **kwargs,
    ) -> CandidateBatch:
        """Generate using subprocess call to la-proteina."""
        # Run generation script
        cmd = [
            self.python_executable,
            str(self.repo_path / "proteinfoundation" / "generate.py"),
            "--config_name", "inference_ucond_notri",
            "--num_samples", str(n_samples),
            "--output_dir", str(self.output_dir),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"La Proteina generation failed: {result.stderr}")

        # Parse output PDB files
        candidates = []
        for pdb_file in sorted(self.output_dir.glob("*.pdb")):
            seq, coords = self._parse_pdb(pdb_file)
            candidate = Candidate(
                sequence=seq,
                structure=coords,
                metadata={"pdb_path": str(pdb_file)},
            )
            candidates.append(candidate)

        return CandidateBatch(candidates=candidates[:n_samples])

    def _parse_pdb(self, pdb_path: Path):
        """Parse PDB file to extract sequence and coordinates."""
        sequence = []
        coords = []

        aa_3to1 = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }

        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and " CA " in line:
                    res_name = line[17:20].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    sequence.append(aa_3to1.get(res_name, "X"))
                    coords.append([x, y, z])

        return "".join(sequence), np.array(coords)
