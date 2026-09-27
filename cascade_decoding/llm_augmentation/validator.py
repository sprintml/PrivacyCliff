"""Validation and repair utilities for LLM-generated tabular data.

This module provides functions to validate generated samples against
the schema and repair violations where possible.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .schema import DatasetSchema


def validate_sample(
    sample: dict,
    schema: DatasetSchema,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a single sample against the schema.

    Args:
        sample: Dictionary of feature_name -> value
        schema: Dataset schema
        strict: If True, any violation returns False

    Returns:
        Tuple of (is_valid, list of violation messages)
    """
    violations = []

    for feature in schema.features:
        name = feature.name

        # Check if feature exists
        if name not in sample:
            violations.append(f"Missing feature: {name}")
            continue

        value = sample[name]

        # Check for None/null
        if value is None:
            violations.append(f"{name}: value is None")
            continue

        # Type validation
        if feature.dtype == "integer":
            if not isinstance(value, (int, float)) or (
                isinstance(value, float) and value != int(value)
            ):
                violations.append(
                    f"{name}: expected integer, got {type(value).__name__} ({value})"
                )

        elif feature.dtype == "binary":
            if value not in [0, 1, 0.0, 1.0]:
                violations.append(f"{name}: expected 0 or 1, got {value}")

        elif feature.dtype == "categorical":
            if feature.categories and value not in feature.categories:
                violations.append(
                    f"{name}: {value} not in valid categories {feature.categories}"
                )

        # Range validation
        if feature.dtype in ["float", "integer"]:
            if feature.min_value is not None and value < feature.min_value:
                violations.append(f"{name}: {value} < min {feature.min_value}")
            if feature.max_value is not None and value > feature.max_value:
                violations.append(f"{name}: {value} > max {feature.max_value}")

    if strict:
        return len(violations) == 0, violations
    # Only critical violations (missing features) make it invalid
    critical = [v for v in violations if "Missing feature" in v]
    return len(critical) == 0, violations


def repair_sample(
    sample: dict,
    schema: DatasetSchema,
    verbose: bool = False,
) -> dict | None:
    """Repair a sample to conform to schema.

    Args:
        sample: Dictionary of feature_name -> value
        schema: Dataset schema
        verbose: Print repair actions

    Returns:
        Repaired sample, or None if unrepairable
    """
    repaired = {}

    for feature in schema.features:
        name = feature.name

        # Get value, use midpoint as default if missing
        if name not in sample or sample[name] is None:
            if feature.dtype == "categorical" and feature.categories:
                value = feature.categories[0]
            elif feature.dtype == "binary":
                value = 0
            elif feature.min_value is not None and feature.max_value is not None:
                value = (feature.min_value + feature.max_value) / 2
            else:
                return None  # Can't repair without range info

            if verbose:
                pass
        else:
            value = sample[name]

        # Type conversion
        if feature.dtype == "integer":
            value = round(value)
            if verbose and value != sample.get(name):
                pass

        elif feature.dtype == "binary":
            value = 1 if value >= 0.5 else 0
            if verbose and value != sample.get(name):
                pass

        elif feature.dtype == "categorical":
            if feature.categories and value not in feature.categories:
                # Find closest category (for numeric categories)
                try:
                    value = min(
                        feature.categories, key=lambda x: abs(float(x) - float(value))
                    )
                except (ValueError, TypeError):
                    value = feature.categories[0]
                if verbose:
                    pass

        # Range clipping
        if feature.dtype in ["float", "integer"]:
            original = value
            if feature.min_value is not None:
                value = max(value, feature.min_value)
            if feature.max_value is not None:
                value = min(value, feature.max_value)
            if verbose and value != original:
                pass

        repaired[name] = value

    return repaired


def validate_and_repair(
    samples: list[dict],
    schema: DatasetSchema,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Validate and repair a batch of samples.

    Args:
        samples: List of sample dictionaries
        schema: Dataset schema
        verbose: Print progress

    Returns:
        Tuple of (repaired_samples, stats_dict)
    """
    repaired_samples = []
    stats = {
        "total": len(samples),
        "valid": 0,
        "repaired": 0,
        "dropped": 0,
        "violations": {},
    }

    for _i, sample in enumerate(samples):
        is_valid, violations = validate_sample(sample, schema)

        if is_valid and not violations:
            repaired_samples.append(sample)
            stats["valid"] += 1
        else:
            # Track violation types
            for v in violations:
                v_type = v.split(":")[0] if ":" in v else v
                stats["violations"][v_type] = stats["violations"].get(v_type, 0) + 1

            # Try to repair
            repaired = repair_sample(sample, schema, verbose=False)
            if repaired:
                repaired_samples.append(repaired)
                stats["repaired"] += 1
            else:
                stats["dropped"] += 1

    if verbose and stats["violations"]:
        for v_type, _count in sorted(
            stats["violations"].items(), key=lambda x: -x[1]
        )[:5]:
            pass

    return repaired_samples, stats


def samples_to_numpy(
    samples: list[dict],
    schema: DatasetSchema,
) -> np.ndarray:
    """Convert list of sample dicts to numpy array.

    Handles string categorical values by converting them to their index
    in the feature's categories list. This allows schemas to use semantic
    string categories (e.g., "Bachelors", "Private") while producing
    label-encoded numeric output.

    Args:
        samples: List of validated sample dictionaries
        schema: Dataset schema (defines feature order)

    Returns:
        Numpy array of shape (n_samples, n_features)
    """
    n_samples = len(samples)
    n_features = len(schema.features)

    X = np.zeros((n_samples, n_features), dtype=np.float64)

    for i, sample in enumerate(samples):
        for j, feature in enumerate(schema.features):
            value = sample.get(feature.name, 0.0)

            # Convert string categories to their index
            if isinstance(value, str) and feature.categories:
                try:
                    value = feature.categories.index(value)
                except ValueError:
                    # Category not found - use 0 as default
                    value = 0

            X[i, j] = value

    return X


def enforce_constraints(
    X: np.ndarray,
    schema: DatasetSchema,
    verbose: bool = False,
) -> np.ndarray:
    """Enforce cross-feature constraints on numpy array.

    This function applies dataset-specific constraint enforcement.
    Currently implements constraints for breast_cancer dataset.

    Args:
        X: Feature matrix (n_samples, n_features)
        schema: Dataset schema
        verbose: Print repair actions

    Returns:
        Constrained feature matrix
    """
    X = X.copy()

    if schema.name == "breast_cancer":
        # Constraint: worst >= mean for each characteristic
        # Features 0-9 are mean, 20-29 are worst
        for i in range(10):
            mean_idx = i
            worst_idx = i + 20
            mask = X[:, worst_idx] < X[:, mean_idx]
            if mask.any():
                # Set worst to mean + small margin
                X[mask, worst_idx] = X[mask, mean_idx] * 1.1
                if verbose:
                    pass

        # Constraint: perimeter ≈ 2π × radius
        # Features: 0=mean_radius, 2=mean_perimeter, 20=worst_radius, 22=worst_perimeter
        for radius_idx, perimeter_idx in [(0, 2), (20, 22)]:
            expected_perimeter = 2 * np.pi * X[:, radius_idx]
            # Allow 20% deviation
            lower = expected_perimeter * 0.8
            upper = expected_perimeter * 1.2
            mask = (X[:, perimeter_idx] < lower) | (X[:, perimeter_idx] > upper)
            if mask.any():
                X[mask, perimeter_idx] = expected_perimeter[mask]
                if verbose:
                    pass

        # Constraint: area ≈ π × radius²
        # Features: 0=mean_radius, 3=mean_area, 20=worst_radius, 23=worst_area
        for radius_idx, area_idx in [(0, 3), (20, 23)]:
            expected_area = np.pi * X[:, radius_idx] ** 2
            # Allow 30% deviation (area has more variance)
            lower = expected_area * 0.7
            upper = expected_area * 1.3
            mask = (X[:, area_idx] < lower) | (X[:, area_idx] > upper)
            if mask.any():
                X[mask, area_idx] = expected_area[mask]
                if verbose:
                    pass

    return X
