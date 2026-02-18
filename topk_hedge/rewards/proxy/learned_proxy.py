"""
Learned proxy reward model.

A fast neural network that predicts pLDDT from sequence.
Can be trained on (sequence, true_pLDDT) pairs collected from ESMFold/AF2.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from topk_hedge.rewards.base import ProxyReward
from topk_hedge.data.candidate import Candidate, CandidateBatch


# Amino acid encoding
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def encode_one_hot(seq: str, max_len: int) -> torch.Tensor:
    """
    Encode sequence as one-hot tensor.

    Args:
        seq: Amino acid sequence
        max_len: Maximum sequence length (pad/truncate to this)

    Returns:
        Tensor of shape [max_len, vocab_size]
    """
    vocab_size = len(ALPHABET)
    x = torch.zeros(max_len, vocab_size, dtype=torch.float32)

    for i, ch in enumerate(seq[:max_len]):
        if ch in ALPHABET:
            x[i, ALPHABET.index(ch)] = 1.0
        else:
            # Unknown -> X
            x[i, ALPHABET.index("X")] = 1.0

    return x


class RewardModel(nn.Module):
    """
    Simple MLP for sequence -> pLDDT prediction.

    Architecture:
    - Flatten one-hot encoding
    - Linear -> ReLU -> Linear -> scalar
    """

    def __init__(
        self,
        vocab_size: int = len(ALPHABET),
        max_len: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size

        input_dim = vocab_size * max_len
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, max_len, vocab_size]

        Returns:
            Predictions [B]
        """
        x = x.view(x.size(0), -1)  # Flatten
        return self.mlp(x).squeeze(-1)


class CNNRewardModel(nn.Module):
    """
    1D CNN for sequence -> pLDDT prediction.

    Better at capturing local sequence patterns than MLP.
    """

    def __init__(
        self,
        vocab_size: int = len(ALPHABET),
        hidden_dim: int = 128,
        kernel_size: int = 7,
        n_layers: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        layers = []
        in_channels = vocab_size
        for i in range(n_layers):
            out_channels = hidden_dim
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, L, vocab_size]

        Returns:
            Predictions [B]
        """
        x = x.transpose(1, 2)  # [B, vocab_size, L]
        x = self.conv(x)  # [B, hidden_dim, L]
        x = self.pool(x).squeeze(-1)  # [B, hidden_dim]
        return self.head(x).squeeze(-1)


class LearnedProxyReward(ProxyReward):
    """
    Learned proxy reward function.

    Uses a trained neural network to predict pLDDT from sequence.
    Much faster than running ESMFold, but less accurate.

    Usage:
        # Load pre-trained model
        proxy = LearnedProxyReward(checkpoint_path="model.pth")
        scores = proxy.score(batch)

        # Or create untrained (for training)
        proxy = LearnedProxyReward(max_len=256)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        max_len: int = 512,
        model_type: str = "mlp",  # "mlp" or "cnn"
        hidden_dim: int = 256,
        device: str = "cuda",
    ):
        """
        Initialize learned proxy reward.

        Args:
            checkpoint_path: Path to trained model checkpoint
            max_len: Maximum sequence length
            model_type: "mlp" or "cnn"
            hidden_dim: Hidden dimension
            device: Device to run on
        """
        self.max_len = max_len
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self._device = device

        # Create model
        if model_type == "mlp":
            self._model = RewardModel(
                vocab_size=len(ALPHABET),
                max_len=max_len,
                hidden_dim=hidden_dim,
            )
        elif model_type == "cnn":
            self._model = CNNRewardModel(
                vocab_size=len(ALPHABET),
                hidden_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self._model = self._model.to(self.device)
        self._model.eval()

    @property
    def name(self) -> str:
        return f"learned-proxy-{self.model_type}"

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    @property
    def model(self) -> nn.Module:
        """Access underlying model (for training)."""
        return self._model

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        if "model_state" in checkpoint:
            self._model.load_state_dict(checkpoint["model_state"])
        elif "model_state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self._model.load_state_dict(checkpoint)

    def save_checkpoint(self, path: str):
        """Save model to checkpoint."""
        torch.save({
            "model_state": self._model.state_dict(),
            "max_len": self.max_len,
            "model_type": self.model_type,
            "hidden_dim": self.hidden_dim,
        }, path)

    def score(self, batch: CandidateBatch) -> np.ndarray:
        """
        Score a batch of candidates.

        Args:
            batch: CandidateBatch containing N candidates

        Returns:
            np.ndarray of shape [N] with predicted pLDDT scores
        """
        sequences = batch.sequences
        return self.score_sequences(sequences)

    def score_single(self, candidate: Candidate) -> float:
        """Score a single candidate."""
        scores = self.score_sequences([candidate.sequence])
        return float(scores[0])

    def score_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        Score a list of sequences.

        Args:
            sequences: List of amino acid sequences

        Returns:
            np.ndarray of predicted scores
        """
        # Encode sequences
        encoded = [encode_one_hot(seq, self.max_len) for seq in sequences]
        x = torch.stack(encoded, dim=0).to(self.device)

        # Predict
        with torch.no_grad():
            scores = self._model(x)

        return scores.cpu().numpy()

    def get_config(self) -> Dict[str, Any]:
        """Return proxy configuration."""
        return {
            "name": self.name,
            "max_len": self.max_len,
            "model_type": self.model_type,
            "hidden_dim": self.hidden_dim,
            "device": self._device,
        }


def train_proxy_reward(
    train_sequences: List[str],
    train_scores: np.ndarray,
    val_sequences: Optional[List[str]] = None,
    val_scores: Optional[np.ndarray] = None,
    max_len: int = 512,
    model_type: str = "mlp",
    hidden_dim: int = 256,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_epochs: int = 50,
    device: str = "cuda",
    save_path: Optional[str] = None,
) -> LearnedProxyReward:
    """
    Train a proxy reward model.

    Args:
        train_sequences: Training sequences
        train_scores: Training pLDDT scores
        val_sequences: Validation sequences (optional)
        val_scores: Validation pLDDT scores (optional)
        max_len: Maximum sequence length
        model_type: "mlp" or "cnn"
        hidden_dim: Hidden dimension
        batch_size: Training batch size
        lr: Learning rate
        n_epochs: Number of epochs
        device: Device to train on
        save_path: Path to save best model

    Returns:
        Trained LearnedProxyReward
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Create proxy reward
    proxy = LearnedProxyReward(
        max_len=max_len,
        model_type=model_type,
        hidden_dim=hidden_dim,
        device=device,
    )
    model = proxy.model
    model.train()

    # Prepare data
    def prepare_data(sequences, scores):
        encoded = [encode_one_hot(seq, max_len) for seq in sequences]
        x = torch.stack(encoded, dim=0)
        y = torch.tensor(scores, dtype=torch.float32)
        return TensorDataset(x, y)

    train_dataset = prepare_data(train_sequences, train_scores)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_sequences is not None and val_scores is not None:
        val_dataset = prepare_data(val_sequences, val_scores)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_dataset)

        # Validate
        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(val_dataset)

            print(f"Epoch {epoch + 1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Save best
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                proxy.save_checkpoint(save_path)
        else:
            print(f"Epoch {epoch + 1}/{n_epochs}: train_loss={train_loss:.4f}")

    # Load best model if validation was used
    if save_path and val_loader:
        proxy.load_checkpoint(save_path)

    model.eval()
    return proxy
