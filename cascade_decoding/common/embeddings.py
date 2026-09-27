"""Embedding extraction utilities for model inversion attacks.

This module provides functions to extract embeddings from TabPFN models,
including both standard extraction and gradient-enabled extraction with
custom support sets.
"""
from __future__ import annotations

import numpy as np
import torch


def get_tabpfn_embeddings(model, X: np.ndarray) -> np.ndarray:
    """Extract 192-dim embeddings from a fitted TabPFN model.

    Args:
        model: Fitted TabPFN model (TabPFNClassifier or TabPFNRegressor)
        X: Input features (n_samples, n_features)

    Returns:
        Embeddings of shape (n_samples, 192)
    """
    from tabpfn.utils import get_embeddings

    embeddings = get_embeddings(model, X, data_source="test")
    # Squeeze batch dimension if present
    if embeddings.ndim == 3 and embeddings.shape[0] == 1:
        embeddings = embeddings.squeeze(0)
    return embeddings


def get_embeddings_with_support(
    model,
    X_support: torch.Tensor,
    y_support: torch.Tensor,
    X_test: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Get embeddings using custom support set (gradient-enabled).

    This directly calls the transformer with custom X_support and y_support,
    bypassing the executor's cached X_trains/y_trains. This allows gradient
    flow through the support set for optimization.

    Args:
        model: Fitted TabPFN model (with executor_)
        X_support: Support set features [n_support, n_features_preprocessed]
        y_support: Support set labels [n_support]
        X_test: Query features [n_test, n_features_preprocessed]
        device: Device to run on

    Returns:
        Embeddings tensor [n_test, 192] with gradient support

    Note:
        Assumes model's executor has inference_mode=False for gradients.
    """
    # Access the inference engine
    executor = model.executor_

    # Get the underlying transformer model
    model_cache = executor.model_caches[0]
    transformer = model_cache.get(torch.device(device), multiple_devices=False)

    # Prepare inputs
    dtype = executor.force_inference_dtype or torch.float32

    X_support = X_support.to(device=device, dtype=dtype)
    X_test = X_test.to(device=device, dtype=dtype)
    y_support = y_support.to(device=device, dtype=dtype)

    # Concatenate: [n_support + n_test, n_features]
    X_full = torch.cat([X_support, X_test], dim=0)

    # Add batch dimension: [n_support + n_test, 1, n_features]
    X_full = X_full.unsqueeze(1)

    # Get categorical indices from executor
    cat_ix = executor.cat_ixs[0] if hasattr(executor, "cat_ixs") else []

    # Forward pass through transformer
    output = transformer(
        X_full,
        y_support.unsqueeze(1),  # [n_support, 1]
        only_return_standard_out=False,
        categorical_inds=[cat_ix],
    )

    # Extract embeddings from output
    return _extract_embeddings_from_output(output)



def _extract_embeddings_from_output(output) -> torch.Tensor:
    """Extract embeddings from transformer output.

    Handles various output formats from the transformer.

    Args:
        output: Transformer output (dict or tensor)

    Returns:
        Embeddings tensor of shape [n_test, 192]
    """
    if isinstance(output, dict) and "test_embeddings" in output:
        embeddings = output["test_embeddings"]
    elif isinstance(output, dict) and "output" in output:
        embeddings = output["output"]
    else:
        embeddings = output

    # Squeeze extra dimensions to get [n_test, 192]
    while embeddings.dim() > 2:
        for i in range(embeddings.dim()):
            if embeddings.shape[i] == 1:
                embeddings = embeddings.squeeze(i)
                break

    return embeddings


def compute_embedding_distance(
    Z1: np.ndarray,
    Z2: np.ndarray,
    metric: str = "mse",
) -> float:
    """Compute distance between two embedding matrices.

    Args:
        Z1: First embedding matrix (n_samples, embedding_dim)
        Z2: Second embedding matrix (n_samples, embedding_dim)
        metric: Distance metric ("mse", "mae", "cosine")

    Returns:
        Scalar distance value
    """
    if metric == "mse":
        return np.mean((Z1 - Z2) ** 2)
    if metric == "mae":
        return np.mean(np.abs(Z1 - Z2))
    if metric == "cosine":
        # Cosine similarity (1 - similarity = distance)
        Z1_norm = Z1 / (np.linalg.norm(Z1, axis=1, keepdims=True) + 1e-8)
        Z2_norm = Z2 / (np.linalg.norm(Z2, axis=1, keepdims=True) + 1e-8)
        similarity = np.mean(np.sum(Z1_norm * Z2_norm, axis=1))
        return 1.0 - similarity
    raise ValueError(f"Unknown metric: {metric}")


def find_nearest_neighbor_by_embedding(
    Z_query: np.ndarray,
    Z_database: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """Find k nearest neighbors in embedding space.

    Args:
        Z_query: Query embeddings (n_queries, embedding_dim)
        Z_database: Database embeddings (n_database, embedding_dim)
        k: Number of neighbors to return

    Returns:
        Indices of k nearest neighbors (n_queries, k)
    """
    # Compute pairwise distances
    # (n_queries, n_database)
    distances = np.sum(
        (Z_query[:, np.newaxis, :] - Z_database[np.newaxis, :, :]) ** 2,
        axis=2,
    )

    # Get indices of k smallest distances
    if k == 1:
        return np.argmin(distances, axis=1)
    return np.argsort(distances, axis=1)[:, :k]
