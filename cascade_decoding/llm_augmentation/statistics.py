"""Dataset statistics computation for hard-coded range constraints.

This module computes statistical bounds from real training data to be used
as hard constraints in LLM prompts, addressing Issue #9.

The key idea: Instead of using hardcoded schema ranges, compute actual
min/max/mean/std from the support set and inject these as STRICT constraints
in the generation prompt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureStatistics:
    """Statistics for a single feature.

    Attributes:
        name: Feature name
        min: Minimum value in data
        max: Maximum value in data
        mean: Mean value
        std: Standard deviation
        percentiles: Dictionary of percentiles (e.g., {25: 11.7, 50: 13.4, 75: 15.8})
        dtype: Data type hint ("float", "integer", "binary", "categorical")
    """

    name: str
    min: float
    max: float
    mean: float
    std: float
    percentiles: dict[int, float]
    dtype: str

    def to_prompt_text(self, strict: bool = True) -> str:
        """Generate prompt text for this feature with hard constraints.

        Args:
            strict: If True, use STRICT/MUST language. If False, use softer guidance.

        Returns:
            Formatted string for prompt
        """
        if strict:
            # Strict constraint format
            text = f"  - {self.name}: "
            if self.dtype == "integer":
                text += f"MUST be in [{int(self.min)}, {int(self.max)}] (integer)"
            else:
                text += f"MUST be in [{self.min:.4f}, {self.max:.4f}]"
            text += f"\n    Typical: {self.mean:.4f} ± {self.std:.4f} (mean ± std)"

            if self.percentiles:
                p_str = ", ".join(
                    f"P{p}={v:.3f}" for p, v in sorted(self.percentiles.items())
                )
                text += f"\n    Distribution: {p_str}"
        else:
            # Softer guidance format (similar to current approach)
            text = f"  - {self.name}: "
            if self.dtype == "integer":
                text += f"integer, range: {int(self.min)}-{int(self.max)}"
            else:
                text += f"range: {self.min:.4f}-{self.max:.4f}"
            text += f" (typical: {self.mean:.4f} ± {self.std:.4f})"

        return text


@dataclass
class DatasetStatistics:
    """Statistics for an entire dataset.

    Attributes:
        dataset_name: Name of dataset
        feature_stats: List of FeatureStatistics for each feature
        n_samples: Number of samples statistics were computed from
        class_name: For classification, which class (None for regression or all classes)
    """

    dataset_name: str
    feature_stats: list[FeatureStatistics]
    n_samples: int
    class_name: str | None = None

    def get_prompt_header(self, strict: bool = True) -> str:
        """Generate prompt header explaining the constraints.

        Args:
            strict: If True, emphasize hard constraints

        Returns:
            Header text for prompt
        """
        if strict:
            header = "CRITICAL CONSTRAINTS - COMPUTED FROM REAL DATA:\n"
            header += "="*70 + "\n"
            header += f"These bounds are HARD CONSTRAINTS derived from {self.n_samples} real samples.\n"
            header += "ALL generated values MUST fall within these ranges.\n"
            header += "Values outside these bounds are INVALID and will be rejected.\n"
            if self.class_name:
                header += f"Statistics are specific to class: {self.class_name}\n"
            header += "="*70 + "\n"
        else:
            header = f"Feature ranges (based on {self.n_samples} real samples):\n"
            if self.class_name:
                header += f"(Statistics for class: {self.class_name})\n"

        return header

    def to_prompt_text(self, strict: bool = True) -> str:
        """Generate full prompt text with all feature constraints.

        Args:
            strict: If True, use strict constraint language

        Returns:
            Complete prompt text for constraints
        """
        lines = [self.get_prompt_header(strict)]
        lines.append("\nFEATURE CONSTRAINTS:")
        lines.extend(f.to_prompt_text(strict) for f in self.feature_stats)
        return "\n".join(lines)


def compute_feature_statistics(
    X: np.ndarray,
    feature_names: list[str],
    percentiles: list[int] | None = None,
    integer_threshold: float = 0.95,
) -> list[FeatureStatistics]:
    """Compute statistics for each feature in a dataset.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        percentiles: List of percentiles to compute (e.g., [25, 50, 75])
        integer_threshold: Fraction of values that must be integers for detection

    Returns:
        List of FeatureStatistics objects
    """
    if percentiles is None:
        percentiles = [25, 50, 75]

    stats_list = []

    for i, name in enumerate(feature_names):
        feature_data = X[:, i]

        # Compute statistics
        min_val = float(np.min(feature_data))
        max_val = float(np.max(feature_data))
        mean_val = float(np.mean(feature_data))
        std_val = float(np.std(feature_data))

        # Compute percentiles
        percentile_dict = {}
        for p in percentiles:
            percentile_dict[p] = float(np.percentile(feature_data, p))

        # Detect data type
        unique_vals = np.unique(feature_data)
        if len(unique_vals) == 2:
            dtype = "binary"
        elif len(unique_vals) <= 10:
            dtype = "categorical"
        else:
            # Check if integer
            is_integer = np.abs(feature_data - np.round(feature_data)) < 1e-6
            if np.mean(is_integer) >= integer_threshold:
                dtype = "integer"
            else:
                dtype = "float"

        stats = FeatureStatistics(
            name=name,
            min=min_val,
            max=max_val,
            mean=mean_val,
            std=std_val,
            percentiles=percentile_dict,
            dtype=dtype,
        )
        stats_list.append(stats)

    return stats_list


def compute_dataset_statistics(
    X: np.ndarray,
    y: np.ndarray | None,
    feature_names: list[str],
    dataset_name: str,
    class_names: list[str] | None = None,
    compute_per_class: bool = True,
    percentiles: list[int] | None = None,
) -> dict[str, DatasetStatistics]:
    """Compute statistics for a dataset, optionally per-class.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) - can be None for regression
        feature_names: List of feature names
        dataset_name: Name of dataset
        class_names: List of class names (for classification)
        compute_per_class: If True and y is provided, compute per-class statistics
        percentiles: List of percentiles to compute

    Returns:
        Dictionary mapping class_name (or "all") to DatasetStatistics
    """
    results = {}

    # Compute overall statistics
    overall_stats = compute_feature_statistics(X, feature_names, percentiles)
    results["all"] = DatasetStatistics(
        dataset_name=dataset_name,
        feature_stats=overall_stats,
        n_samples=X.shape[0],
        class_name=None,
    )

    # Compute per-class statistics if requested
    if compute_per_class and y is not None:
        unique_classes = np.unique(y)
        for cls_idx in unique_classes:
            mask = y == cls_idx
            X_cls = X[mask]

            # Get class name
            if class_names and cls_idx < len(class_names):
                cls_name = class_names[cls_idx]
            else:
                cls_name = f"class_{cls_idx}"

            cls_stats = compute_feature_statistics(X_cls, feature_names, percentiles)
            results[cls_name] = DatasetStatistics(
                dataset_name=dataset_name,
                feature_stats=cls_stats,
                n_samples=X_cls.shape[0],
                class_name=cls_name,
            )

    return results


def load_real_data_statistics(
    dataset_name: str,
    use_full_dataset: bool = True,
    percentiles: list[int] | None = None,
) -> dict[str, DatasetStatistics]:
    """Load real dataset and compute statistics.

    This is a convenience function that loads common datasets and
    computes their statistics for use in LLM generation.

    Args:
        dataset_name: Name of dataset (e.g., "breast_cancer", "iris", "diabetes")
        use_full_dataset: If True, use full dataset. If False, use train split
        percentiles: List of percentiles to compute

    Returns:
        Dictionary of DatasetStatistics per class
    """
    # Import data loading utilities
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from common.data import load_dataset as load_from_npz
        from common.constants import DATASETS

        # Load dataset from NPZ files
        X, y, feature_names, task_type = load_from_npz(dataset_name)

        # Get class names if classification
        class_names = None
        if task_type == "classification":
            # Generate class names from unique labels
            unique_labels = sorted(np.unique(y))
            class_names = [f"class_{i}" for i in unique_labels]

            # Try to get better names from dataset registry
            if dataset_name in DATASETS:
                # For known datasets, use better names
                if dataset_name == "breast_cancer":
                    class_names = ["malignant", "benign"]
                elif dataset_name == "iris":
                    class_names = ["setosa", "versicolor", "virginica"]
                elif dataset_name == "diabetes_pima":
                    class_names = ["no_diabetes", "diabetes"]
                elif dataset_name == "heart_statlog":
                    class_names = ["no_disease", "disease"]
                elif dataset_name == "parkinsons":
                    class_names = ["healthy", "parkinsons"]
                # Truncate if we have more labels than names
                class_names = class_names[:len(unique_labels)]

    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{dataset_name}': {e}. "
            f"Make sure the dataset is available in the data directory."
        )

    # Compute statistics
    stats = compute_dataset_statistics(
        X=X,
        y=y,
        feature_names=feature_names,
        dataset_name=dataset_name,
        class_names=class_names,
        compute_per_class=True,
        percentiles=percentiles,
    )

    return stats


if __name__ == "__main__":
    """Test the statistics computation."""
    print("Computing statistics for breast_cancer dataset...")
    stats = load_real_data_statistics("breast_cancer")

    print(f"\nFound statistics for: {list(stats.keys())}")

    # Show overall statistics prompt
    print("\n" + "="*80)
    print("OVERALL STATISTICS PROMPT (strict=True):")
    print("="*80)
    print(stats["all"].to_prompt_text(strict=True)[:1000] + "...")

    # Show malignant class statistics
    print("\n" + "="*80)
    print("MALIGNANT CLASS STATISTICS:")
    print("="*80)
    print(stats["malignant"].to_prompt_text(strict=True)[:1000] + "...")

    # Compare first feature between classes
    print("\n" + "="*80)
    print("FIRST FEATURE COMPARISON:")
    print("="*80)
    feat0_all = stats["all"].feature_stats[0]
    feat0_mal = stats["malignant"].feature_stats[0]
    feat0_ben = stats["benign"].feature_stats[0]
    print(f"Overall:   {feat0_all.name} = [{feat0_all.min:.3f}, {feat0_all.max:.3f}], mean={feat0_all.mean:.3f}")
    print(f"Malignant: {feat0_mal.name} = [{feat0_mal.min:.3f}, {feat0_mal.max:.3f}], mean={feat0_mal.mean:.3f}")
    print(f"Benign:    {feat0_ben.name} = [{feat0_ben.min:.3f}, {feat0_ben.max:.3f}], mean={feat0_ben.mean:.3f}")
