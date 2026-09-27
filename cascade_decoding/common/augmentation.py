"""Data augmentation utilities for surrogate model training.

This module provides functions to generate synthetic training data for
improving surrogate model quality. The key insight is that since the
target TabPFN uses a fixed context (support set), we can generate
synthetic samples, query TabPFN for their embeddings, and augment
the training data.

Augmentation methods:
- Latin Hypercube Sampling (LHS): Better space coverage than random
- Mixup: Convex combinations of existing samples (stays on manifold)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

AUGMENTATION_DEFAULTS = {
    "enable": False,
    "n_synthetic": 10000,
    "lhs_ratio": 0.8,  # 80% LHS, 20% mixup
    "mixup_alpha": 0.4,  # Beta distribution parameter for mixup
    "random_state": 42,
    "batch_size": 1000,  # Batch size for TabPFN queries
    "verbose": True,
}


def generate_lhs_samples(
    X_shadow: np.ndarray,
    n_samples: int,
    margin: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Generate synthetic samples using Latin Hypercube Sampling.

    LHS provides better coverage of the feature space than random sampling
    by ensuring each feature is sampled uniformly across its range.

    Args:
        X_shadow: Shadow set features (n_shadow, n_features)
        n_samples: Number of synthetic samples to generate
        margin: Extend feature ranges by this fraction (0.1 = 10% on each side)
        random_state: Random seed for reproducibility

    Returns:
        Synthetic samples (n_samples, n_features)
    """
    n_features = X_shadow.shape[1]

    # Get feature bounds from shadow data with margin
    mins = X_shadow.min(axis=0)
    maxs = X_shadow.max(axis=0)

    # Add margin to handle edge cases
    ranges = maxs - mins
    ranges = np.where(ranges == 0, 1.0, ranges)  # Handle constant features
    mins = mins - margin * ranges
    maxs = maxs + margin * ranges

    # LHS in [0,1]^d
    sampler = qmc.LatinHypercube(d=n_features, seed=random_state)
    samples = sampler.random(n=n_samples)

    # Scale to feature ranges
    return samples * (maxs - mins) + mins



def generate_mixup_samples(
    X_shadow: np.ndarray,
    n_samples: int,
    alpha: float = 0.4,
    random_state: int = 42,
) -> np.ndarray:
    """Generate synthetic samples using Mixup interpolation.

    Mixup creates convex combinations of existing samples, which helps
    the model learn smoother decision boundaries and stays closer to
    the data manifold.

    Args:
        X_shadow: Shadow set features (n_shadow, n_features)
        n_samples: Number of synthetic samples to generate
        alpha: Beta distribution parameter (smaller = closer to original samples)
        random_state: Random seed for reproducibility

    Returns:
        Synthetic samples (n_samples, n_features)
    """
    rng = np.random.default_rng(random_state)
    n_shadow = len(X_shadow)

    # Random pairs of indices
    idx1 = rng.integers(0, n_shadow, n_samples)
    idx2 = rng.integers(0, n_shadow, n_samples)

    # Mixup coefficients from Beta(alpha, alpha) distribution
    # This gives symmetric interpolation centered around 0.5
    lam = rng.beta(alpha, alpha, n_samples).reshape(-1, 1)

    # Convex combination
    return lam * X_shadow[idx1] + (1 - lam) * X_shadow[idx2]



def generate_gaussian_noise_samples(
    X_shadow: np.ndarray,
    n_samples: int,
    noise_std: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Generate synthetic samples by adding Gaussian noise to shadow samples.

    This creates variations around existing samples, useful for local
    exploration of the embedding space.

    Args:
        X_shadow: Shadow set features (n_shadow, n_features)
        n_samples: Number of synthetic samples to generate
        noise_std: Standard deviation of noise relative to feature std
        random_state: Random seed for reproducibility

    Returns:
        Synthetic samples (n_samples, n_features)
    """
    rng = np.random.default_rng(random_state)
    n_shadow = len(X_shadow)

    # Sample base points from shadow set
    base_idx = rng.integers(0, n_shadow, n_samples)
    X_base = X_shadow[base_idx]

    # Add Gaussian noise scaled by feature standard deviation
    feature_std = X_shadow.std(axis=0)
    feature_std = np.where(
        feature_std == 0, 1.0, feature_std
    )  # Handle constant features

    noise = rng.normal(0, noise_std, (n_samples, X_shadow.shape[1])) * feature_std

    return X_base + noise


def augment_training_data(
    X_shadow: np.ndarray,
    Z_shadow: np.ndarray,
    target_model,
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment training data with synthetic samples.

    Main entry point for data augmentation. Generates synthetic X samples
    using LHS and Mixup, queries the target TabPFN model for embeddings,
    and returns the augmented dataset.

    Args:
        X_shadow: Shadow set features (n_shadow, n_features)
        Z_shadow: Shadow set embeddings (n_shadow, embedding_dim)
        target_model: Fitted TabPFN model to query for synthetic embeddings
        config: Augmentation configuration dict with keys:
            - enable: Whether to enable augmentation
            - n_synthetic: Number of synthetic samples
            - lhs_ratio: Fraction of samples from LHS (rest from mixup)
            - mixup_alpha: Beta parameter for mixup
            - random_state: Random seed
            - batch_size: Batch size for TabPFN queries
            - verbose: Print progress

    Returns:
        Tuple of (X_augmented, Z_augmented) with original + synthetic data
    """
    from .embeddings import get_tabpfn_embeddings

    # Merge with defaults
    cfg = AUGMENTATION_DEFAULTS.copy()
    if config:
        cfg.update(config)

    if not cfg.get("enable", False):
        return X_shadow, Z_shadow

    n_synthetic = cfg.get("n_synthetic", 10000)
    lhs_ratio = cfg.get("lhs_ratio", 0.8)
    mixup_alpha = cfg.get("mixup_alpha", 0.4)
    random_state = cfg.get("random_state", 42)
    batch_size = cfg.get("batch_size", 1000)
    verbose = cfg.get("verbose", True)

    if verbose:
        pass

    # Calculate split
    n_lhs = int(n_synthetic * lhs_ratio)
    n_mixup = n_synthetic - n_lhs

    # Generate synthetic X
    synthetic_parts = []

    if n_lhs > 0:
        X_lhs = generate_lhs_samples(
            X_shadow,
            n_lhs,
            random_state=random_state,
        )
        synthetic_parts.append(X_lhs)
        if verbose:
            pass

    if n_mixup > 0:
        X_mixup = generate_mixup_samples(
            X_shadow,
            n_mixup,
            alpha=mixup_alpha,
            random_state=random_state + 1,
        )
        synthetic_parts.append(X_mixup)
        if verbose:
            pass

    if not synthetic_parts:
        return X_shadow, Z_shadow

    X_synthetic = np.vstack(synthetic_parts)

    # Query TabPFN for synthetic embeddings in batches
    if verbose:
        pass

    Z_parts = []
    n_batches = (len(X_synthetic) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_synthetic))
        X_batch = X_synthetic[start:end]

        Z_batch = get_tabpfn_embeddings(target_model, X_batch)
        Z_parts.append(Z_batch)

        if verbose and (i + 1) % 10 == 0:
            pass

    Z_synthetic = np.vstack(Z_parts)

    # Combine with original data
    X_augmented = np.vstack([X_shadow, X_synthetic])
    Z_augmented = np.vstack([Z_shadow, Z_synthetic])

    if verbose:
        pass

    return X_augmented, Z_augmented


# =============================================================================
# GAN Training Augmentation
# =============================================================================

GAN_AUGMENTATION_DEFAULTS = {
    "n_synthetic": 5000,
    "mode": "combined",  # "gaussian", "mixup", or "combined"
    "noise_std": 0.1,  # Gaussian noise std (relative to feature std)
    "mixup_alpha": 0.4,  # Beta distribution parameter for mixup
    "gaussian_ratio": 0.5,  # Fraction of Gaussian samples in combined mode
    "random_state": 42,
    "verbose": True,
}


def augment_gan_training_data(
    X_shadow: np.ndarray,
    n_synthetic: int = 5000,
    mode: str = "combined",
    noise_std: float = 0.1,
    mixup_alpha: float = 0.4,
    gaussian_ratio: float = 0.5,
    random_state: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """Augment shadow data for GAN training.

    Unlike surrogate augmentation (which needs TabPFN embeddings), GAN training
    only needs X (features). This function generates synthetic samples to expand
    small shadow sets, improving GAN training quality.

    Three augmentation modes:
    - gaussian: Add Gaussian noise to shadow samples (local exploration)
    - mixup: Convex combinations of pairs (stays on data manifold)
    - combined: Mix of both methods (balanced exploration)

    Args:
        X_shadow: Original shadow features (n_shadow, n_features)
        n_synthetic: Number of synthetic samples to generate
        mode: Augmentation mode - "gaussian", "mixup", or "combined"
        noise_std: Std for Gaussian noise (relative to feature std)
        mixup_alpha: Beta distribution parameter for mixup
        gaussian_ratio: Fraction of Gaussian samples in combined mode
        random_state: Random seed for reproducibility
        verbose: Print progress messages

    Returns:
        Augmented data: original X_shadow concatenated with synthetic samples

    Example:
        >>> X_aug = augment_gan_training_data(X_shadow, n_synthetic=5000, mode="combined")
        >>> print(f"Augmented: {len(X_shadow)} -> {len(X_aug)} samples")
    """
    len(X_shadow)

    if n_synthetic <= 0:
        if verbose:
            pass
        return X_shadow

    if verbose:
        pass

    if mode == "gaussian":
        X_synthetic = generate_gaussian_noise_samples(
            X_shadow, n_synthetic, noise_std, random_state
        )
        if verbose:
            pass

    elif mode == "mixup":
        X_synthetic = generate_mixup_samples(
            X_shadow, n_synthetic, mixup_alpha, random_state
        )
        if verbose:
            pass

    elif mode == "combined":
        n_gaussian = int(n_synthetic * gaussian_ratio)
        n_mixup = n_synthetic - n_gaussian

        synthetic_parts = []

        if n_gaussian > 0:
            X_gaussian = generate_gaussian_noise_samples(
                X_shadow, n_gaussian, noise_std, random_state
            )
            synthetic_parts.append(X_gaussian)
            if verbose:
                pass

        if n_mixup > 0:
            X_mixup = generate_mixup_samples(
                X_shadow, n_mixup, mixup_alpha, random_state + 1
            )
            synthetic_parts.append(X_mixup)
            if verbose:
                pass

        X_synthetic = (
            np.vstack(synthetic_parts)
            if synthetic_parts
            else np.empty((0, X_shadow.shape[1]))
        )

    else:
        raise ValueError(
            f"Unknown augmentation mode: {mode}. Use 'gaussian', 'mixup', or 'combined'"
        )

    # Combine original and synthetic data
    X_augmented = np.vstack([X_shadow, X_synthetic])

    if verbose:
        pass

    return X_augmented
