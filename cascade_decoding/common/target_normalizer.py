"""Target Normalization for Regression Tasks (Issue #39).

This module provides utilities for normalizing continuous/integer targets
to improve attack performance on regression tasks.

Problem:
    Continuous features have vastly different scales (e.g., credit_amount: 250-18424
    vs diabetes_pedigree: 0.078-2.42). Without normalization:
    1. Large-scale features dominate MSE loss
    2. Training becomes unstable
    3. Network struggles to predict values outside typical training range

Solution:
    Normalize targets to a standard range during training, then denormalize
    predictions back to original scale.

Normalization Strategies:
    - "minmax": Scale to [0, 1] using min/max from training data
    - "minmax_sym": Scale to [-1, 1] using min/max (better for symmetric data)
    - "standard": Standardize to mean=0, std=1 (z-score)

Usage:
    # Initialize with feature metadata
    normalizer = TargetNormalizer(
        feature_metadata=DIABETES_PIMA_FEATURES,
        strategy="minmax_sym",
    )

    # Fit on training data
    normalizer.fit(X_train, sensitive_indices=[1, 2, 4, 5])

    # Normalize targets for training
    y_normalized = normalizer.normalize(y_train, feature_idx=5)

    # Denormalize predictions
    y_pred_original = normalizer.denormalize(y_pred_normalized, feature_idx=5)
"""

from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
import torch

NormStrategy = Literal["minmax", "minmax_sym", "standard", "none"]

# TypeVar to preserve input type in normalize/denormalize
ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


class TargetNormalizer:
    """Normalizer for continuous/integer regression targets.

    Stores normalization parameters (min, max, mean, std) for each feature
    and provides normalize/denormalize operations.

    Args:
        feature_metadata: Dict mapping feature index to metadata dict.
                         Must contain 'dtype' field.
        strategy: Normalization strategy:
            - "minmax": Scale to [0, 1]
            - "minmax_sym": Scale to [-1, 1]
            - "standard": Z-score normalization
            - "none": No normalization (for ablation)
        epsilon: Small value to prevent division by zero
    """

    def __init__(
        self,
        feature_metadata: dict[int, dict],
        strategy: NormStrategy = "minmax_sym",
        epsilon: float = 1e-8,
    ):
        self.feature_metadata = feature_metadata
        self.strategy = strategy
        self.epsilon = epsilon

        # Normalization parameters per feature
        self.params: dict[int, dict] = {}
        self.is_fitted = False

    def _is_continuous(self, feature_idx: int) -> bool:
        """Check if a feature is continuous (needs normalization)."""
        if feature_idx not in self.feature_metadata:
            return True  # Default to treating as continuous
        dtype = self.feature_metadata[feature_idx].get("dtype", "continuous")
        return dtype in ("continuous", "integer")

    def _is_integer(self, feature_idx: int) -> bool:
        """Check if a feature is integer type (needs rounding after denorm)."""
        if feature_idx not in self.feature_metadata:
            return False
        dtype = self.feature_metadata[feature_idx].get("dtype", "continuous")
        return dtype == "integer"

    def fit(
        self,
        X: np.ndarray,
        sensitive_indices: list[int],
    ) -> "TargetNormalizer":
        """Compute normalization parameters from training data.

        Args:
            X: Training data (n_samples, n_features)
            sensitive_indices: Indices of sensitive features to normalize

        Returns:
            self for method chaining
        """
        for idx in sensitive_indices:
            if not self._is_continuous(idx):
                # Skip categorical features
                self.params[idx] = {"skip": True}
                continue

            values = X[:, idx]
            # Remove NaNs for computing statistics
            values = values[~np.isnan(values)]

            if len(values) == 0:
                self.params[idx] = {"skip": True}
                continue

            raw_range = float(values.max() - values.min())
            raw_std = float(values.std())
            self.params[idx] = {
                "skip": False,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": raw_std,
                "range": raw_range,
                # Epsilon-adjusted values for consistent normalize/denormalize roundtrip
                "range_adj": raw_range + self.epsilon,
                "std_adj": raw_std + self.epsilon,
            }

        self.is_fitted = True
        return self

    def normalize(
        self,
        values: ArrayT,
        feature_idx: int,
    ) -> ArrayT:
        """Normalize values for a specific feature.

        Args:
            values: Values to normalize (any shape, numpy or torch)
            feature_idx: Feature index

        Returns:
            Normalized values in same format as input (numpy → numpy, torch → torch)
        """
        if not self.is_fitted:
            raise RuntimeError("TargetNormalizer must be fitted before normalize()")

        if self.strategy == "none":
            return values

        if feature_idx not in self.params or self.params[feature_idx].get("skip", False):
            return values

        params = self.params[feature_idx]

        if self.strategy == "minmax":
            # Scale to [0, 1]
            result = (values - params["min"]) / params["range_adj"]

        elif self.strategy == "minmax_sym":
            # Scale to [-1, 1]
            result = 2 * (values - params["min"]) / params["range_adj"] - 1

        elif self.strategy == "standard":
            # Z-score normalization
            result = (values - params["mean"]) / params["std_adj"]

        else:
            return values

        return result

    def denormalize(
        self,
        values: ArrayT,
        feature_idx: int,
        round_integers: bool = True,
    ) -> ArrayT:
        """Denormalize values back to original scale.

        Args:
            values: Normalized values (any shape, numpy or torch)
            feature_idx: Feature index
            round_integers: If True, round integer features to nearest int

        Returns:
            Denormalized values in same format as input (numpy → numpy, torch → torch)
        """
        if not self.is_fitted:
            raise RuntimeError("TargetNormalizer must be fitted before denormalize()")

        if self.strategy == "none":
            return values

        if feature_idx not in self.params or self.params[feature_idx].get("skip", False):
            return values

        params = self.params[feature_idx]

        if self.strategy == "minmax":
            # Reverse [0, 1] scaling
            result = values * params["range_adj"] + params["min"]

        elif self.strategy == "minmax_sym":
            # Reverse [-1, 1] scaling
            result = (values + 1) / 2 * params["range_adj"] + params["min"]

        elif self.strategy == "standard":
            # Reverse z-score
            result = values * params["std_adj"] + params["mean"]

        else:
            return values

        # Round integer features to nearest int
        if round_integers and self._is_integer(feature_idx):
            if isinstance(values, torch.Tensor):
                result = torch.round(result)
            else:
                result = np.round(result)

        return result

    def normalize_batch(
        self,
        X: ArrayT,
        sensitive_indices: list[int],
    ) -> ArrayT:
        """Normalize all sensitive features in a batch.

        Args:
            X: Data with shape (n_samples, n_sensitive)
            sensitive_indices: Indices of sensitive features (for looking up params)

        Returns:
            Normalized data
        """
        result = X.clone() if isinstance(X, torch.Tensor) else X.copy()

        for i, idx in enumerate(sensitive_indices):
            result[:, i] = self.normalize(X[:, i], idx)

        return result

    def denormalize_batch(
        self,
        X: ArrayT,
        sensitive_indices: list[int],
        round_integers: bool = True,
    ) -> ArrayT:
        """Denormalize all sensitive features in a batch.

        Args:
            X: Normalized data with shape (n_samples, n_sensitive)
            sensitive_indices: Indices of sensitive features (for looking up params)
            round_integers: If True, round integer features to nearest int

        Returns:
            Denormalized data
        """
        result = X.clone() if isinstance(X, torch.Tensor) else X.copy()

        for i, idx in enumerate(sensitive_indices):
            result[:, i] = self.denormalize(X[:, i], idx, round_integers=round_integers)

        return result

    def get_info(self, feature_idx: int) -> dict:
        """Get normalization info for a feature."""
        if feature_idx not in self.params:
            return {"status": "not_fitted"}
        return self.params[feature_idx]

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not_fitted"
        n_features = len(self.params)
        return f"TargetNormalizer(strategy={self.strategy}, {status}, {n_features} features)"
