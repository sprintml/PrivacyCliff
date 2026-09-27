"""Evaluation metrics for model inversion attacks.

This module provides functions to evaluate the quality of feature
reconstructions and surrogate model predictions.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_reconstruction(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict:
    """Evaluate feature reconstruction quality.

    Computes multiple metrics to assess how well the reconstructed
    features match the true features.

    Args:
        X_true: True features (n_samples, n_features)
        X_pred: Reconstructed features (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        Dictionary with evaluation metrics:
            - r2_overall: Overall R² score
            - r2_per_feature: R² for each feature
            - r2_per_feature_mean: Mean per-feature R²
            - r2_per_feature_std: Std of per-feature R²
            - mae_overall: Mean absolute error
            - mae_per_feature: MAE for each feature
            - mse_overall: Mean squared error
            - cosine_similarity: Mean cosine similarity
            - n_features_r2_above_0.9: Count of features with R² > 0.9
            - n_features_r2_above_0.5: Count of features with R² > 0.5
    """
    n_features = X_true.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Overall metrics
    r2_overall = r2_score(X_true, X_pred)
    mae_overall = mean_absolute_error(X_true, X_pred)
    mse_overall = mean_squared_error(X_true, X_pred)

    # Per-feature metrics
    r2_per_feature = []
    mae_per_feature = []

    for i in range(n_features):
        r2_feat = r2_score(X_true[:, i], X_pred[:, i])
        mae_feat = mean_absolute_error(X_true[:, i], X_pred[:, i])
        r2_per_feature.append(r2_feat)
        mae_per_feature.append(mae_feat)

    r2_per_feature = np.array(r2_per_feature)
    mae_per_feature = np.array(mae_per_feature)

    # Cosine similarity
    X_true_norm = X_true / (np.linalg.norm(X_true, axis=1, keepdims=True) + 1e-8)
    X_pred_norm = X_pred / (np.linalg.norm(X_pred, axis=1, keepdims=True) + 1e-8)
    cosine_sim = np.mean(np.sum(X_true_norm * X_pred_norm, axis=1))

    return {
        "r2_overall": float(r2_overall),
        "r2_per_feature": r2_per_feature,
        "r2_per_feature_mean": float(np.mean(r2_per_feature)),
        "r2_per_feature_std": float(np.std(r2_per_feature)),
        "mae_overall": float(mae_overall),
        "mae_per_feature": mae_per_feature,
        "mae_mean": float(np.mean(mae_per_feature)),
        "mse_overall": float(mse_overall),
        "cosine_similarity": float(cosine_sim),
        "n_features_r2_above_0.9": int(np.sum(r2_per_feature > 0.9)),
        "n_features_r2_above_0.5": int(np.sum(r2_per_feature > 0.5)),
        "n_features": n_features,
        "feature_names": feature_names,
    }


def evaluate_surrogate(
    Z_true: np.ndarray,
    Z_pred: np.ndarray,
) -> dict:
    """Evaluate surrogate model embedding prediction quality.

    Args:
        Z_true: True embeddings from target model (n_samples, embedding_dim)
        Z_pred: Predicted embeddings from surrogate (n_samples, embedding_dim)

    Returns:
        Dictionary with evaluation metrics:
            - mse: Mean squared error
            - mae: Mean absolute error
            - r2_overall: Overall R² score (on flattened embeddings)
            - r2_per_dim_mean: Mean R² across embedding dimensions
            - r2_per_dim_std: Std of R² across embedding dimensions
            - cosine_similarity: Mean cosine similarity
    """
    # Flatten for overall R²
    r2_overall = r2_score(Z_true.flatten(), Z_pred.flatten())

    # Per-dimension metrics
    embedding_dim = Z_true.shape[1]
    r2_per_dim = []

    for i in range(embedding_dim):
        r2_dim = r2_score(Z_true[:, i], Z_pred[:, i])
        r2_per_dim.append(r2_dim)

    r2_per_dim = np.array(r2_per_dim)

    # Cosine similarity
    Z_true_norm = Z_true / (np.linalg.norm(Z_true, axis=1, keepdims=True) + 1e-8)
    Z_pred_norm = Z_pred / (np.linalg.norm(Z_pred, axis=1, keepdims=True) + 1e-8)
    cosine_sim = np.mean(np.sum(Z_true_norm * Z_pred_norm, axis=1))

    return {
        "mse": float(np.mean((Z_true - Z_pred) ** 2)),
        "mae": float(np.mean(np.abs(Z_true - Z_pred))),
        "r2_overall": float(r2_overall),
        "r2_per_dim_mean": float(np.mean(r2_per_dim)),
        "r2_per_dim_std": float(np.std(r2_per_dim)),
        "cosine_similarity": float(cosine_sim),
    }


def evaluate_generator(
    X_real: np.ndarray,
    X_generated: np.ndarray,
) -> dict:
    """Evaluate generator quality by comparing real and generated distributions.

    Args:
        X_real: Real features from shadow data
        X_generated: Generated features from generator

    Returns:
        Dictionary with evaluation metrics:
            - mean_error: Mean error per feature
            - std_ratio: Ratio of generated to real std per feature
            - coverage: Fraction of real data range covered by generated
    """
    n_features = X_real.shape[1]

    mean_error = []
    std_ratio = []
    coverage = []

    for i in range(n_features):
        real_feat = X_real[:, i]
        gen_feat = X_generated[:, i]

        # Mean error
        mean_error.append(abs(np.mean(real_feat) - np.mean(gen_feat)))

        # Std ratio
        real_std = np.std(real_feat)
        gen_std = np.std(gen_feat)
        if real_std > 1e-8:
            std_ratio.append(gen_std / real_std)
        else:
            std_ratio.append(1.0)

        # Coverage (what fraction of real range is covered)
        real_min, real_max = np.min(real_feat), np.max(real_feat)
        gen_min, gen_max = np.min(gen_feat), np.max(gen_feat)
        real_range = real_max - real_min
        if real_range > 1e-8:
            overlap_min = max(real_min, gen_min)
            overlap_max = min(real_max, gen_max)
            overlap = max(0, overlap_max - overlap_min)
            coverage.append(overlap / real_range)
        else:
            coverage.append(1.0)

    return {
        "mean_error_per_feature": np.array(mean_error),
        "mean_error_avg": float(np.mean(mean_error)),
        "std_ratio_per_feature": np.array(std_ratio),
        "std_ratio_avg": float(np.mean(std_ratio)),
        "coverage_per_feature": np.array(coverage),
        "coverage_avg": float(np.mean(coverage)),
    }


def print_reconstruction_results(metrics: dict, top_k: int = 10) -> None:
    """Print reconstruction evaluation results in a formatted way.

    Args:
        metrics: Dictionary from evaluate_reconstruction()
        top_k: Number of best/worst features to show
    """
    # Show top/bottom features
    r2_per_feature = metrics["r2_per_feature"]
    feature_names = metrics["feature_names"]
    sorted_indices = np.argsort(r2_per_feature)[::-1]

    for _i in sorted_indices[:top_k]:
        pass

    if len(feature_names) > top_k:
        for _i in sorted_indices[-top_k:]:
            pass



def print_surrogate_results(metrics: dict) -> None:
    """Print surrogate evaluation results in a formatted way.

    Args:
        metrics: Dictionary from evaluate_surrogate()
    """


# =============================================================================
# Attribute Inference Evaluation (for Level 2/3 attacks)
# =============================================================================

def evaluate_attribute_inference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dtype: str,
    n_categories: int | None = None,
    feature_name: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> dict:
    """Evaluate single attribute inference quality.

    This function evaluates how well a single attribute (feature) was inferred,
    with metrics appropriate for the data type.

    Args:
        y_true: True values (n_samples,)
        y_pred: Predicted values (n_samples,)
        dtype: Data type - one of "categorical", "binary", "integer", "continuous"
        n_categories: Number of categories for categorical features
        feature_name: Optional feature name for reporting
        min_value: Minimum value from feature metadata (for NMAE normalization)
        max_value: Maximum value from feature metadata (for NMAE normalization)

    Returns:
        Dictionary with evaluation metrics appropriate for the dtype:
            For categorical/binary:
                - accuracy: Classification accuracy
                - top_k_accuracy: Accuracy if prediction within k of true (for ordinal)
                - confusion_matrix: Confusion matrix (if n_categories <= 10)
            For integer:
                - exact_accuracy: Fraction of exact matches
                - within_1_accuracy: Fraction within 1 of true value
                - mae: Mean absolute error
                - rmse: Root mean squared error
            For continuous:
                - r2: R² score
                - mae: Mean absolute error
                - rmse: Root mean squared error
                - within_range_accuracy: Fraction within 10% of true range
    """
    from sklearn.metrics import accuracy_score, confusion_matrix as sk_confusion_matrix

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    result = {
        "dtype": dtype,
        "feature_name": feature_name,
        "n_samples": len(y_true),
    }

    if dtype in ("categorical", "binary"):
        # Round predictions to nearest integer for categorical
        y_pred_rounded = np.round(y_pred).astype(int)

        # Clip to valid range if n_categories provided
        if n_categories is not None:
            y_pred_rounded = np.clip(y_pred_rounded, 0, n_categories - 1)

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred_rounded)
        result["accuracy"] = float(accuracy)

        # Within-1 accuracy (useful for ordinal categoricals)
        within_1 = np.mean(np.abs(y_true - y_pred_rounded) <= 1)
        result["within_1_accuracy"] = float(within_1)

        # Confusion matrix (only if manageable size)
        n_unique = len(np.unique(np.concatenate([y_true, y_pred_rounded])))
        if n_unique <= 10:
            cm = sk_confusion_matrix(y_true, y_pred_rounded)
            result["confusion_matrix"] = cm.tolist()

        # MAE (interpretable for ordinal)
        result["mae"] = float(np.mean(np.abs(y_true - y_pred_rounded)))

    elif dtype == "integer":
        # Round predictions
        y_pred_rounded = np.round(y_pred).astype(int)

        # Exact match accuracy
        exact_accuracy = np.mean(y_true == y_pred_rounded)
        result["exact_accuracy"] = float(exact_accuracy)

        # Within-k accuracy
        result["within_1_accuracy"] = float(np.mean(np.abs(y_true - y_pred_rounded) <= 1))
        result["within_2_accuracy"] = float(np.mean(np.abs(y_true - y_pred_rounded) <= 2))

        # Regression metrics
        mae = float(np.mean(np.abs(y_true - y_pred)))
        result["mae"] = mae
        result["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # NMAE: Normalized MAE by feature range
        # Use metadata range if available, otherwise fall back to sample range
        if min_value is not None and max_value is not None:
            feature_range = float(max_value - min_value)
            result["feature_min"] = float(min_value)
            result["feature_max"] = float(max_value)
        else:
            feature_range = float(np.max(y_true) - np.min(y_true))
            result["feature_min"] = float(np.min(y_true))
            result["feature_max"] = float(np.max(y_true))

        result["feature_range"] = feature_range

        if feature_range > 1e-8:
            result["nmae"] = mae / feature_range
        else:
            result["nmae"] = 0.0 if mae < 1e-8 else float("nan")

        # R² if there's variance
        if np.var(y_true) > 1e-8:
            result["r2"] = float(r2_score(y_true, y_pred))
        else:
            result["r2"] = float("nan")

    else:  # continuous
        # Regression metrics
        mae = float(np.mean(np.abs(y_true - y_pred)))
        result["mae"] = mae
        result["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # NMAE: Normalized MAE by feature range
        # Use metadata range if available, otherwise fall back to sample range
        if min_value is not None and max_value is not None:
            feature_range = float(max_value - min_value)
            result["feature_min"] = float(min_value)
            result["feature_max"] = float(max_value)
        else:
            feature_range = float(np.max(y_true) - np.min(y_true))
            result["feature_min"] = float(np.min(y_true))
            result["feature_max"] = float(np.max(y_true))

        result["feature_range"] = feature_range

        if feature_range > 1e-8:
            result["nmae"] = mae / feature_range
        else:
            result["nmae"] = 0.0 if mae < 1e-8 else float("nan")

        # R² if there's variance
        if np.var(y_true) > 1e-8:
            result["r2"] = float(r2_score(y_true, y_pred))
        else:
            result["r2"] = float("nan")

        # Within-range accuracy (within 10% of the feature range)
        if feature_range > 1e-8:
            threshold = 0.1 * feature_range
            within_range = np.mean(np.abs(y_true - y_pred) <= threshold)
            result["within_10pct_accuracy"] = float(within_range)
        else:
            result["within_10pct_accuracy"] = float("nan")

    return result


def evaluate_conditional_attack(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    target_indices: list[int],
    feature_metadata: dict[int, dict] | None = None,
    feature_names: list[str] | None = None,
) -> dict:
    """Evaluate conditional attribute inference attack.

    Evaluates reconstruction quality specifically on target (sensitive) features.

    Args:
        X_true: True features (n_samples, n_features)
        X_pred: Reconstructed features (n_samples, n_features)
        target_indices: Indices of target features to evaluate
        feature_metadata: Optional metadata dict for each feature index
        feature_names: Optional list of feature names

    Returns:
        Dictionary with:
            - per_feature: Dict of feature_idx -> evaluation metrics
            - summary: Aggregate statistics across target features
    """
    per_feature = {}

    for idx in target_indices:
        y_true = X_true[:, idx]
        y_pred = X_pred[:, idx]

        # Get feature info
        if feature_metadata and idx in feature_metadata:
            meta = feature_metadata[idx]
            dtype = meta.get("dtype", "continuous")
            n_categories = meta.get("n_categories")
            name = meta.get("name", f"feature_{idx}")
            min_value = meta.get("min_value")
            max_value = meta.get("max_value")
        else:
            dtype = "continuous"
            n_categories = None
            name = feature_names[idx] if feature_names else f"feature_{idx}"
            min_value = None
            max_value = None

        # Evaluate this feature
        metrics = evaluate_attribute_inference(
            y_true, y_pred, dtype=dtype, n_categories=n_categories, feature_name=name,
            min_value=min_value, max_value=max_value
        )
        per_feature[idx] = metrics

    # Compute summary statistics
    summary = {
        "n_target_features": len(target_indices),
        "target_indices": target_indices,
    }

    # Aggregate accuracy for categorical features
    cat_accuracies = [
        m["accuracy"] for m in per_feature.values()
        if m["dtype"] in ("categorical", "binary") and "accuracy" in m
    ]
    if cat_accuracies:
        summary["mean_categorical_accuracy"] = float(np.mean(cat_accuracies))

    # Aggregate R² for continuous/integer features
    r2_values = [
        m["r2"] for m in per_feature.values()
        if "r2" in m and not np.isnan(m["r2"])
    ]
    if r2_values:
        summary["mean_r2"] = float(np.mean(r2_values))

    # Aggregate MAE
    mae_values = [m["mae"] for m in per_feature.values() if "mae" in m]
    if mae_values:
        summary["mean_mae"] = float(np.mean(mae_values))

    return {
        "per_feature": per_feature,
        "summary": summary,
    }


def print_conditional_attack_results(eval_results: dict) -> None:
    """Print conditional attack evaluation results.

    Args:
        eval_results: Dictionary from evaluate_conditional_attack()
    """
    print("\n" + "=" * 70)
    print("Conditional Attribute Inference Results")
    print("=" * 70)

    # Print per-feature results
    print("\nPer-Feature Results:")
    print("-" * 70)
    print(f"{'Idx':<4} {'Name':<20} {'Type':<12} {'Metric':<15} {'Value':<10}")
    print("-" * 70)

    for idx, metrics in eval_results["per_feature"].items():
        name = metrics.get("feature_name", f"feature_{idx}")[:20]
        dtype = metrics["dtype"]

        if dtype in ("categorical", "binary"):
            metric_name = "Accuracy"
            metric_val = metrics.get("accuracy", float("nan"))
        elif dtype == "integer":
            metric_name = "Exact Acc"
            metric_val = metrics.get("exact_accuracy", float("nan"))
        else:
            metric_name = "R²"
            metric_val = metrics.get("r2", float("nan"))

        print(f"{idx:<4} {name:<20} {dtype:<12} {metric_name:<15} {metric_val:.4f}")

    # Print summary
    summary = eval_results["summary"]
    print("\n" + "-" * 70)
    print("Summary:")
    if "mean_categorical_accuracy" in summary:
        print(f"  Mean Categorical Accuracy: {summary['mean_categorical_accuracy']:.4f}")
    if "mean_r2" in summary:
        print(f"  Mean R²: {summary['mean_r2']:.4f}")
    if "mean_mae" in summary:
        print(f"  Mean MAE: {summary['mean_mae']:.4f}")
    print("=" * 70)
