"""Base class for model inversion attacks.

Model inversion attacks attempt to reconstruct private features from
observed embeddings by optimizing in feature space or latent space.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from cascade_decoding.surrogates.base import SurrogateBase


class AttackBase(ABC):
    """Abstract base class for model inversion attacks.

    Attacks reconstruct victim features X from their embeddings Z
    using a surrogate model that provides gradients for optimization.

    Subclasses must implement:
        - attack(): Reconstruct features from embeddings

    Attributes:
        surrogate: Surrogate model for embedding prediction
        config: Attack configuration dictionary
    """

    def __init__(
        self,
        surrogate: SurrogateBase,
        config: dict | None = None,
    ):
        """Initialize attack.

        Args:
            surrogate: Trained surrogate model
            config: Attack configuration (n_iters, lr, etc.)
        """
        self.surrogate = surrogate
        self.config = config or {}

    @abstractmethod
    def attack(
        self,
        Z_victim: np.ndarray,
        X_shadow: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, dict]:
        """Reconstruct features from victim embeddings.

        Args:
            Z_victim: Target embeddings to match (n_victims, embedding_dim)
            X_shadow: Shadow data for initialization (n_shadow, n_features)
            **kwargs: Additional attack parameters

        Returns:
            X_reconstructed: Reconstructed features (n_victims, n_features)
            stats: Attack statistics dictionary
        """

    def attack_single(
        self,
        z_victim: np.ndarray,
        X_shadow: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, dict]:
        """Reconstruct features for a single victim.

        Convenience method that wraps attack() for single-sample case.

        Args:
            z_victim: Target embedding (embedding_dim,)
            X_shadow: Shadow data for initialization
            **kwargs: Additional attack parameters

        Returns:
            x_reconstructed: Reconstructed features (n_features,)
            stats: Attack statistics
        """
        Z_victim = z_victim.reshape(1, -1)
        X_rec, stats = self.attack(Z_victim, X_shadow, **kwargs)
        return X_rec[0], stats

    def get_config_value(self, key: str, default=None):
        """Get configuration value with default.

        Args:
            key: Configuration key
            default: Default value if key not present

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"surrogate={self.surrogate.__class__.__name__}, "
            f"config={self.config})"
        )


class AttackResult:
    """Container for attack results.

    Attributes:
        X_reconstructed: Reconstructed features
        X_true: True features (if available)
        stats: Attack statistics
        metrics: Evaluation metrics (if computed)
    """

    def __init__(
        self,
        X_reconstructed: np.ndarray,
        X_true: np.ndarray | None = None,
        stats: dict | None = None,
        metrics: dict | None = None,
    ):
        self.X_reconstructed = X_reconstructed
        self.X_true = X_true
        self.stats = stats or {}
        self.metrics = metrics

    def compute_metrics(self) -> dict:
        """Compute evaluation metrics if X_true is available."""
        if self.X_true is None:
            raise ValueError("Cannot compute metrics without X_true")

        from cascade_decoding.common.evaluation import evaluate_reconstruction

        self.metrics = evaluate_reconstruction(self.X_true, self.X_reconstructed)
        return self.metrics

    @property
    def r2_score(self) -> float | None:
        """Get overall R² score."""
        if self.metrics is None:
            return None
        return self.metrics.get("r2_overall")

    def summary(self) -> str:
        """Get summary string of results."""
        lines = [
            "Attack Results:",
            f"  Reconstructed samples: {len(self.X_reconstructed)}",
        ]

        if self.metrics:
            lines.extend(
                [
                    f"  R² Score: {self.metrics['r2_overall']:.4f}",
                    f"  MSE: {self.metrics['mse_overall']:.6f}",
                    f"  Cosine Similarity: {self.metrics['cosine_similarity']:.4f}",
                ]
            )

        if self.stats:
            if "mean_loss" in self.stats:
                lines.append(f"  Mean Final Loss: {self.stats['mean_loss']:.6f}")
            if "improved_ratio" in self.stats:
                lines.append(f"  Improved Ratio: {self.stats['improved_ratio']:.1%}")

        return "\n".join(lines)
