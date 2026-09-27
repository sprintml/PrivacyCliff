"""Feature constraint utilities for PGD-based attacks.

This module provides utilities for extracting feature bounds from data and
projecting optimized features back to valid ranges during gradient-based
optimization (Projected Gradient Descent).

The main use case is model inversion attacks where we want to ensure
reconstructed features stay within realistic bounds observed in the data.
"""
from __future__ import annotations

import numpy as np
import torch


class FeatureBounds:
    """Feature bounds extracted from shadow data.

    Supports projection for different feature types:
    - Continuous features: [min, max] box constraints
    - Integer features: [min, max] bounds + rounding
    - Categorical features: discrete allowed values (nearest projection)

    Attributes:
        mins: Feature minimums (n_features,) as torch.Tensor
        maxs: Feature maximums (n_features,) as torch.Tensor
        integer_indices: List of integer feature indices
        categorical_indices: Dict mapping feature idx -> allowed values
        continuous_indices: List of continuous feature indices
        device: torch device for computations

    Example:
        >>> bounds = extract_bounds_from_data(X_shadow, margin=0.05)
        >>> X_projected = bounds.project(X_reconstructed)
    """

    def __init__(
        self,
        mins: np.ndarray,
        maxs: np.ndarray,
        integer_indices: list[int] | None = None,
        categorical_indices: dict[int, np.ndarray] | None = None,
        device: str = "cuda",
    ):
        """Initialize feature bounds.

        Args:
            mins: Minimum values per feature (n_features,)
            maxs: Maximum values per feature (n_features,)
            integer_indices: List of integer feature indices
            categorical_indices: Dict of feature_idx -> allowed_values
            device: torch device ("cuda" or "cpu")
        """
        self.mins = torch.tensor(mins, dtype=torch.float32, device=device)
        self.maxs = torch.tensor(maxs, dtype=torch.float32, device=device)
        self.integer_indices = integer_indices or []
        self.categorical_indices = categorical_indices or {}
        self.device = device

        # Pre-compute continuous feature indices
        all_special = set(self.integer_indices) | set(self.categorical_indices.keys())
        self.continuous_indices = [i for i in range(len(mins)) if i not in all_special]

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """Project features to valid bounds.

        This method applies box constraints to continuous features, rounds
        integer features, and projects categorical features to nearest allowed value.

        Args:
            X: Features (batch, n_features) - must be torch.Tensor

        Returns:
            X_proj: Projected features (same shape as X)
        """
        X_proj = X.clone()

        # 1. Continuous features: clamp to [min, max]
        if self.continuous_indices:
            X_proj[:, self.continuous_indices] = torch.clamp(
                X_proj[:, self.continuous_indices],
                min=self.mins[self.continuous_indices],
                max=self.maxs[self.continuous_indices],
            )

        # 2. Integer features: clamp then round with straight-through estimator
        # Straight-through estimator: forward uses round, backward uses identity
        if self.integer_indices:
            X_proj[:, self.integer_indices] = torch.clamp(
                X_proj[:, self.integer_indices],
                min=self.mins[self.integer_indices],
                max=self.maxs[self.integer_indices],
            )
            # Apply straight-through estimator for rounding
            # Forward: x_rounded, Backward: gradient of x_clamped
            X_int = X_proj[:, self.integer_indices]
            X_proj[:, self.integer_indices] = (
                torch.round(X_int) - X_int.detach() + X_int
            )

        # 3. Categorical features: project to nearest allowed value
        # Use straight-through estimator for categorical projection
        for feat_idx, allowed_values in self.categorical_indices.items():
            # Use input tensor's device to avoid mismatch
            allowed_t = torch.tensor(
                allowed_values, device=X_proj.device, dtype=torch.float32
            )
            # For each sample, find nearest allowed value
            feat_vals = X_proj[:, feat_idx].unsqueeze(1)  # (batch, 1)
            distances = torch.abs(feat_vals - allowed_t)  # (batch, n_values)
            nearest_idx = torch.argmin(distances, dim=1)

            # Straight-through estimator: forward uses discrete projection, backward uses continuous
            X_cat_continuous = X_proj[:, feat_idx]
            X_cat_discrete = allowed_t[nearest_idx]
            X_proj[:, feat_idx] = (
                X_cat_discrete - X_cat_continuous.detach() + X_cat_continuous
            )

        return X_proj

    def compute_violations(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute bound violations before projection.

        Useful for tracking statistics about constraint satisfaction.

        Args:
            X: Features (batch, n_features)

        Returns:
            Dictionary with violation statistics:
                - lower_violations: Amount below min bounds
                - upper_violations: Amount above max bounds
                - total_violations: Total violation magnitude
        """
        lower_violations = torch.relu(self.mins - X)
        upper_violations = torch.relu(X - self.maxs)
        total_violations = lower_violations + upper_violations

        return {
            "lower_violations": lower_violations,
            "upper_violations": upper_violations,
            "total_violations": total_violations,
        }

    def to_dict(self) -> dict:
        """Serialize to dictionary for saving.

        Returns:
            Dictionary containing all bounds metadata
        """
        return {
            "mins": self.mins.cpu().numpy(),
            "maxs": self.maxs.cpu().numpy(),
            "integer_indices": self.integer_indices,
            "categorical_indices": dict(self.categorical_indices.items()),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeatureBounds:
        """Load from dictionary.

        Args:
            d: Dictionary from to_dict()

        Returns:
            FeatureBounds instance
        """
        return cls(
            mins=d["mins"],
            maxs=d["maxs"],
            integer_indices=d.get("integer_indices"),
            categorical_indices=d.get("categorical_indices"),
            device=d.get("device", "cuda"),
        )

    def __repr__(self) -> str:
        return (
            f"FeatureBounds(n_features={len(self.mins)}, "
            f"continuous={len(self.continuous_indices)}, "
            f"integer={len(self.integer_indices)}, "
            f"categorical={len(self.categorical_indices)})"
        )


def extract_bounds_from_data(
    X_shadow: np.ndarray,
    margin: float = 0.05,
    integer_indices: list[int] | None = None,
    categorical_indices: dict[int, np.ndarray] | None = None,
    auto_detect_integers: bool = True,
    integer_threshold: float = 0.95,
    device: str = "cuda",
) -> FeatureBounds:
    """Extract feature bounds from shadow data.

    This function computes min/max bounds for each feature, optionally
    extends them by a margin, and auto-detects integer features using
    a heuristic (features where >95% of values are close to integers).

    Args:
        X_shadow: Shadow features (n_shadow, n_features)
        margin: Extend bounds by this fraction (e.g., 0.05 = 5% on each side)
        integer_indices: List of integer feature indices (if known)
        categorical_indices: Dict of categorical feature idx -> allowed values
        auto_detect_integers: Whether to auto-detect integer features
        integer_threshold: Fraction of values that must be integers for detection
        device: torch device ("cuda" or "cpu")

    Returns:
        FeatureBounds object ready for projection

    Example:
        >>> # Extract bounds with 5% margin
        >>> bounds = extract_bounds_from_data(X_shadow, margin=0.05)
        >>>
        >>> # Extract bounds with known integer features
        >>> bounds = extract_bounds_from_data(
        ...     X_shadow,
        ...     margin=0.05,
        ...     integer_indices=[0, 2, 5],
        ...     auto_detect_integers=False
        ... )
    """
    # Compute min/max
    mins = X_shadow.min(axis=0)
    maxs = X_shadow.max(axis=0)

    # Add margin for continuous features
    ranges = maxs - mins
    ranges = np.where(ranges == 0, 1.0, ranges)  # Handle constant features
    mins = mins - margin * ranges
    maxs = maxs + margin * ranges

    # Auto-detect integer features if not provided
    if integer_indices is None and auto_detect_integers:
        integer_indices = []
        for i in range(X_shadow.shape[1]):
            if i not in (categorical_indices or {}):
                # Check if most values are close to integers
                is_integer = np.abs(X_shadow[:, i] - np.round(X_shadow[:, i])) < 1e-6
                if np.mean(is_integer) >= integer_threshold:
                    integer_indices.append(i)
    elif integer_indices is None:
        integer_indices = []

    return FeatureBounds(
        mins=mins,
        maxs=maxs,
        integer_indices=integer_indices,
        categorical_indices=categorical_indices,
        device=device,
    )
