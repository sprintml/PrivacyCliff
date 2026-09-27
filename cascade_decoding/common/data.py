"""Data loading and splitting utilities for model inversion attacks.

This module provides functions to load datasets from NPZ files and split
them into support, shadow, and victim sets for attack experiments.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from .constants import (
    DATA_DIR,
    DATASETS,
    DEFAULT_SHADOW_RATIO,
    DEFAULT_SUPPORT_RATIO,
    DEFAULT_VICTIM_RATIO,
    MAX_SAMPLES,
)


def load_dataset(
    name: str,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Load dataset from NPZ file.

    Args:
        name: Dataset name (must be in DATASETS registry)
        random_state: Random seed for subsampling

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        feature_names: List of feature names
        task_type: "classification" or "regression"

    Raises:
        ValueError: If dataset name is not in registry
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    filename, task_type = DATASETS[name]
    data = np.load(DATA_DIR / filename)
    X, y = data["X"], data["y"]

    # Subsample if dataset is too large for TabPFN
    if len(X) > MAX_SAMPLES:
        rng = np.random.default_rng(seed=random_state)
        idx = rng.choice(len(X), size=MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]

    # Normalize classification labels to start from 0
    if task_type == "classification":
        unique_labels = np.unique(y)
        if unique_labels.min() != 0:
            label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
            y = np.array([label_map[label] for label in y])

    # Generate feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    return X, y, feature_names, task_type


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    support_ratio: float = DEFAULT_SUPPORT_RATIO,
    shadow_ratio: float = DEFAULT_SHADOW_RATIO,
    victim_ratio: float = DEFAULT_VICTIM_RATIO,
    random_state: int = 42,
) -> dict:
    """Split data into support, shadow, and victim sets.

    The three-way split is used for model inversion attacks:
    - Support: Used to fit TabPFN (attacker does NOT have access)
    - Shadow: Attacker's data for training attack components
    - Victim: Target data whose features attacker wants to reconstruct

    Args:
        X: Feature matrix
        y: Target vector
        support_ratio: Fraction of data for support set
        shadow_ratio: Fraction of data for shadow set
        victim_ratio: Fraction of data for victim set
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with keys:
            - X_support, y_support: Support set
            - X_shadow, y_shadow: Shadow set
            - X_victim, y_victim: Victim set

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    # Validate ratios
    total_ratio = support_ratio + shadow_ratio + victim_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.2f}")

    # 3-way split: support / (shadow + victim)
    rest_ratio = shadow_ratio + victim_ratio
    X_support, X_rest, y_support, y_rest = train_test_split(
        X, y, test_size=rest_ratio, random_state=random_state
    )

    # Split rest into shadow / victim
    victim_from_rest = victim_ratio / rest_ratio
    X_shadow, X_victim, y_shadow, y_victim = train_test_split(
        X_rest, y_rest, test_size=victim_from_rest, random_state=random_state
    )

    return {
        "X_support": X_support,
        "y_support": y_support,
        "X_shadow": X_shadow,
        "y_shadow": y_shadow,
        "X_victim": X_victim,
        "y_victim": y_victim,
    }


def subsample_shadow(
    X_shadow: np.ndarray,
    shadow_size: int | None,
    random_state: int = 42,
) -> np.ndarray:
    """Subsample shadow dataset to a specified size.

    Args:
        X_shadow: Full shadow dataset (n_samples, n_features)
        shadow_size: Target number of samples, or None to use all
        random_state: Random seed for reproducibility

    Returns:
        Subsampled shadow dataset. If shadow_size is None or >= len(X_shadow),
        returns the original array unchanged.
    """
    if shadow_size is None or shadow_size >= len(X_shadow):
        return X_shadow

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X_shadow), size=shadow_size, replace=False)
    return X_shadow[indices]


def get_tabpfn_model(task_type: str, n_estimators: int = 1):
    """Create appropriate TabPFN model based on task type.

    Args:
        task_type: "classification" or "regression"
        n_estimators: Number of ensemble members (default: 1)

    Returns:
        TabPFNClassifier or TabPFNRegressor instance
    """
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    if task_type == "classification":
        return TabPFNClassifier(n_estimators=n_estimators)
    return TabPFNRegressor(n_estimators=n_estimators)


def get_dataset_info(name: str) -> dict:
    """Get information about a dataset.

    Args:
        name: Dataset name

    Returns:
        Dictionary with dataset metadata
    """
    X, y, feature_names, task_type = load_dataset(name)

    info = {
        "name": name,
        "task_type": task_type,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": feature_names,
    }

    if task_type == "classification":
        info["n_classes"] = len(np.unique(y))
        info["class_distribution"] = dict(zip(*np.unique(y, return_counts=True)))
    else:
        info["y_mean"] = float(np.mean(y))
        info["y_std"] = float(np.std(y))

    return info
