"""Base class for surrogate models.

Surrogate models learn to predict TabPFN embeddings from input features (X → Z).
They are used to provide differentiable gradients for optimization-based attacks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch


class SurrogateBase(ABC):
    """Abstract base class for surrogate models.

    Surrogate models approximate TabPFN's embedding function, mapping
    raw features X to the 192-dimensional embedding space Z.

    Subclasses must implement:
        - fit(): Train the surrogate on shadow data
        - predict_embedding(): Get embeddings for new inputs
        - save(): Persist model to disk
        - load(): Load model from disk

    Attributes:
        n_features: Number of input features
        embedding_dim: Output embedding dimension (typically 192)
        device: PyTorch device for computation
        is_fitted: Whether the model has been trained
    """

    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 192,
        device: str = "cuda",
    ):
        """Initialize surrogate model.

        Args:
            n_features: Number of input features
            embedding_dim: Output embedding dimension
            device: Device for computation ("cuda" or "cpu")
        """
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.device = device
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_shadow: np.ndarray,
        y_shadow: np.ndarray,
        Z_target: np.ndarray,
        **kwargs,
    ) -> SurrogateBase:
        """Train the surrogate model.

        Args:
            X_shadow: Shadow set features (n_samples, n_features)
            y_shadow: Shadow set labels (n_samples,)
            Z_target: Target embeddings from TabPFN (n_samples, embedding_dim)
            **kwargs: Additional training parameters

        Returns:
            self for method chaining
        """

    @abstractmethod
    def predict_embedding(self, X: np.ndarray) -> np.ndarray:
        """Predict embeddings for input features.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predicted embeddings (n_samples, embedding_dim)
        """

    @abstractmethod
    def predict_embedding_torch(self, X: torch.Tensor) -> torch.Tensor:
        """Predict embeddings with gradient support.

        Args:
            X: Input features tensor (n_samples, n_features)

        Returns:
            Predicted embeddings tensor (n_samples, embedding_dim)
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model (file or directory)
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path, device: str = "cuda") -> SurrogateBase:
        """Load model from disk.

        Args:
            path: Path to load model from
            device: Device to load model to

        Returns:
            Loaded surrogate model
        """

    def get_r2_score(self, X: np.ndarray, Z_true: np.ndarray) -> float:
        """Compute R² score on embedding prediction.

        Args:
            X: Input features
            Z_true: True embeddings

        Returns:
            R² score
        """
        from sklearn.metrics import r2_score

        Z_pred = self.predict_embedding(X)
        return r2_score(Z_true.flatten(), Z_pred.flatten())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_features={self.n_features}, "
            f"embedding_dim={self.embedding_dim}, "
            f"device={self.device}, "
            f"is_fitted={self.is_fitted})"
        )
