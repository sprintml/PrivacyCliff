"""Main LLM tabular data generator.

This module provides the main interface for generating synthetic tabular
data using LLMs. It handles batched generation, validation, and saving.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .schema import DatasetSchema, get_schema
from .validator import enforce_constraints, samples_to_numpy, validate_and_repair
from collections import Counter
from ..common.constants import OUTPUT_BASE

if TYPE_CHECKING:
    from .providers.base import LLMProvider

# Output directory for generated data (uses configurable OUTPUT_BASE)
OUTPUT_DIR = OUTPUT_BASE / "llm_generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class LLMTabularGenerator:
    """Generate synthetic tabular data using LLMs.

    This class provides a high-level interface for generating semantically
    valid synthetic tabular data. It uses cached schemas to ensure
    consistent generation across runs.

    Example:
        >>> from llm_augmentation.providers import GeminiProvider
        >>> provider = GeminiProvider(api_key="your-key")
        >>> generator = LLMTabularGenerator(provider)
        >>> X, y = generator.generate("breast_cancer", n_samples=100)
    """

    def __init__(
        self,
        provider: LLMProvider,
        temperature: float = 0.8,
        samples_per_call: int = 10,
        max_retries: int = 3,
        verbose: bool = True,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        dataset_name: str | None = None,
        use_hard_constraints: bool = False,
        fewshot_samples: np.ndarray | None = None,
        fewshot_labels: np.ndarray | None = None,
        n_examples_per_call: int = 10,
        enhanced_prompting: bool = False,
        output_suffix: str = "",
    ):
        """Initialize the generator.

        Args:
            provider: LLM provider to use for generation
            temperature: Sampling temperature (higher = more diverse)
            samples_per_call: Number of samples to request per API call
            max_retries: Maximum retries on API failure
            verbose: Print progress
            X_train: Training features for computing hard constraints (optional)
            y_train: Training labels for per-class constraints (optional)
            dataset_name: Dataset name for schema lookup (required if use_hard_constraints=True)
            use_hard_constraints: Whether to use hard-coded range constraints from training data
            fewshot_samples: Few-shot examples to include in prompts (X array)
            fewshot_labels: Labels for few-shot examples (y array)
            n_examples_per_call: Number of few-shot examples to include per LLM call
            enhanced_prompting: Use enhanced prompts with chain-of-thought reasoning,
                               explicit correlation constraints, and edge case generation
            output_suffix: Custom suffix to add to output filename (e.g., "_pool50")
        """
        self.provider = provider
        self.temperature = temperature
        self.samples_per_call = samples_per_call
        self.max_retries = max_retries
        self.verbose = verbose
        self.use_hard_constraints = use_hard_constraints
        self.statistics_cache = {}  # class_name -> DatasetStatistics
        self.enhanced_prompting = enhanced_prompting
        self.output_suffix = output_suffix

        # Warn if output_suffix is empty (risk of overwriting files)
        if not output_suffix and verbose:
            import warnings
            warnings.warn(
                "output_suffix is empty. Generated files may overwrite existing files. "
                "Consider using --output-suffix to distinguish experiment configurations.",
                UserWarning,
            )

        # Few-shot settings
        self.fewshot_samples = fewshot_samples
        self.fewshot_labels = fewshot_labels
        self.n_examples_per_call = n_examples_per_call
        self.use_fewshot = fewshot_samples is not None

        # Validate few-shot arrays have matching lengths
        if fewshot_samples is not None and fewshot_labels is not None:
            if len(fewshot_samples) != len(fewshot_labels):
                raise ValueError(
                    f"fewshot_samples and fewshot_labels must have same length, "
                    f"got {len(fewshot_samples)} and {len(fewshot_labels)}"
                )

        # Compute statistics from training data if using hard constraints
        if use_hard_constraints:
            if X_train is None or dataset_name is None:
                raise ValueError(
                    "X_train and dataset_name are required when use_hard_constraints=True"
                )

            from .schema import get_schema
            from .statistics import compute_dataset_statistics

            schema = get_schema(dataset_name)
            if schema is None:
                raise ValueError(f"No schema found for dataset: {dataset_name}")

            feature_names = [f.name for f in schema.features]
            class_names = (
                schema.classes if schema.task_type == "classification" else None
            )

            if self.verbose:
                print(f"Computing statistics from {X_train.shape[0]} training samples...")

            # Compute statistics (overall + per-class if classification)
            self.statistics_cache = compute_dataset_statistics(
                X=X_train,
                y=y_train,
                feature_names=feature_names,
                dataset_name=dataset_name,
                class_names=class_names,
                compute_per_class=True,
            )

            if self.verbose:
                print(f"  Computed statistics for: {list(self.statistics_cache.keys())}")

    def generate(
        self,
        dataset_name: str,
        n_samples: int = 100,
        class_balance: dict[str, float] | None = None,
        save: bool = True,
        enforce_schema_constraints: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic samples for a dataset.

        Args:
            dataset_name: Name of the dataset (must have schema)
            n_samples: Total number of samples to generate
            class_balance: Dict mapping class name to fraction (must sum to 1.0)
                          If None, uses equal balance for classification
            save: Whether to save generated data to disk
            enforce_schema_constraints: Apply cross-feature constraint enforcement

        Returns:
            Tuple of (X, y) numpy arrays
        """
        # Load schema
        schema = get_schema(dataset_name)
        if schema is None:
            raise ValueError(f"No schema found for dataset: {dataset_name}")

        if self.verbose:
            mode = "few-shot" if self.use_fewshot else "schema-only"
            if self.enhanced_prompting:
                mode += "+enhanced"
            print(f"Generating {n_samples} samples for {dataset_name} ({schema.task_type}, mode={mode})")
            if self.use_fewshot:
                print(f"  Few-shot pool: {len(self.fewshot_samples)} samples, {self.n_examples_per_call} examples per call")

        # Determine samples per class
        if schema.task_type == "classification":
            if class_balance is None:
                # Equal balance
                n_classes = len(schema.classes)
                class_balance = dict.fromkeys(schema.classes, 1.0 / n_classes)

            samples_per_class = {
                c: int(n_samples * frac) for c, frac in class_balance.items()
            }
            # Adjust for rounding
            total = sum(samples_per_class.values())
            if total < n_samples:
                samples_per_class[schema.classes[0]] += n_samples - total

            if self.verbose:
                print(f"  Class distribution: {samples_per_class}")
        else:
            samples_per_class = {None: n_samples}

        # Generate samples for each class
        all_samples = []
        all_labels = []

        for class_idx, (class_name, n_class_samples) in enumerate(
            samples_per_class.items()
        ):
            if self.verbose:
                class_label = class_name if class_name else "all"
                print(f"  Generating {n_class_samples} samples for class '{class_label}'...")

            class_samples = self._generate_class_samples(
                schema, class_name, n_class_samples, class_idx=class_idx
            )
            all_samples.extend(class_samples)

            # Assign labels
            if schema.task_type == "classification":
                label = schema.classes.index(class_name)
                all_labels.extend([label] * len(class_samples))
            else:
                all_labels.extend([0] * len(class_samples))

        # Convert to numpy
        X = samples_to_numpy(all_samples, schema)
        y = np.array(all_labels)

        # Enforce cross-feature constraints
        if enforce_schema_constraints:
            if self.verbose:
                print("  Enforcing schema constraints...")
            X = enforce_constraints(X, schema, verbose=self.verbose)

        if self.verbose:
            print(f"  Generated {len(X)} samples with {X.shape[1]} features")

        # Save to disk
        if save:
            output_path = self._save(dataset_name, X, y, schema)
            if self.verbose:
                print(f"  Saved to {output_path}")

        return X, y

    def generate_stratified(
        self,
        dataset_name: str,
        n_samples: int = 4000,
        stratify_by: str = "occupation",
        class_balance: dict[str, float] | None = None,
        save: bool = False,
        enforce_schema_constraints: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate stratified samples ensuring diversity by conditioning on feature values.

        This method generates samples stratified by a categorical feature (e.g., occupation)
        to ensure diverse output. For each unique value of the stratify_by feature in the
        few-shot data, it generates samples conditioned on that value.

        Args:
            dataset_name: Name of the dataset (must have schema)
            n_samples: Total number of samples to generate
            stratify_by: Feature name to stratify by (must be categorical)
            class_balance: Dict mapping class name to fraction (must sum to 1.0)
                          If None, uses equal balance for classification
            save: Whether to save generated data to disk
            enforce_schema_constraints: Apply cross-feature constraint enforcement

        Returns:
            Tuple of (X, y) numpy arrays
        """
        if not self.use_fewshot:
            raise ValueError("Stratified generation requires few-shot samples")

        # Load schema
        schema = get_schema(dataset_name)
        if schema is None:
            raise ValueError(f"No schema found for dataset: {dataset_name}")

        # Find the stratify_by feature
        feature_idx = None
        stratify_feature = None
        for i, f in enumerate(schema.features):
            if f.name == stratify_by:
                feature_idx = i
                stratify_feature = f
                break

        if feature_idx is None:
            raise ValueError(f"Feature '{stratify_by}' not found in schema")

        if stratify_feature.dtype != "categorical":
            raise ValueError(f"Feature '{stratify_by}' must be categorical, got {stratify_feature.dtype}")

        if self.verbose:
            mode = "few-shot+stratified"
            if self.enhanced_prompting:
                mode += "+enhanced"
            print(f"Generating {n_samples} stratified samples for {dataset_name} ({schema.task_type}, mode={mode})")
            print(f"  Stratifying by: {stratify_by} (feature index {feature_idx})")
            print(f"  Few-shot pool: {len(self.fewshot_samples)} samples")

        # Get unique values from few-shot data
        fewshot_values = self.fewshot_samples[:, feature_idx]
        unique_values_idx = np.unique(fewshot_values[~np.isnan(fewshot_values)]).astype(int)

        # Map indices to category names
        categories = stratify_feature.categories
        unique_values = [categories[int(idx)] for idx in unique_values_idx if int(idx) < len(categories)]

        if self.verbose:
            print(f"  Unique {stratify_by} values in few-shot data: {len(unique_values)}")
            print(f"    Categories: {unique_values}")

        # Determine samples per stratum value
        n_strata = len(unique_values)
        samples_per_stratum = n_samples // n_strata
        extra_samples = n_samples % n_strata

        if self.verbose:
            print(f"  Samples per stratum: {samples_per_stratum} (+ {extra_samples} extra for first strata)")

        # Determine class distribution
        if schema.task_type == "classification":
            if class_balance is None:
                n_classes = len(schema.classes)
                class_balance = dict.fromkeys(schema.classes, 1.0 / n_classes)
            if self.verbose:
                print(f"  Class balance: {class_balance}")

        # Generate samples for each stratum
        all_samples = []
        all_labels = []

        for stratum_idx, stratum_value in enumerate(unique_values):
            # Calculate number of samples for this stratum
            stratum_n = samples_per_stratum
            if stratum_idx < extra_samples:
                stratum_n += 1

            if stratum_n <= 0:
                continue

            if self.verbose:
                print(f"\n  Generating {stratum_n} samples for {stratify_by}='{stratum_value}'...")

            # Generate for each class with the stratum constraint
            if schema.task_type == "classification":
                for class_idx, (class_name, class_frac) in enumerate(class_balance.items()):
                    n_class_samples = int(stratum_n * class_frac)
                    if class_idx == len(class_balance) - 1:
                        # Last class gets remainder
                        n_class_samples = stratum_n - sum(
                            int(stratum_n * f) for i, (_, f) in enumerate(class_balance.items()) if i < len(class_balance) - 1
                        )

                    if n_class_samples <= 0:
                        continue

                    class_samples = self._generate_stratified_samples(
                        schema=schema,
                        target_class=class_name,
                        n_samples=n_class_samples,
                        stratify_feature=stratify_by,
                        stratify_value=stratum_value,
                        stratum_idx=stratum_idx,
                    )
                    all_samples.extend(class_samples)

                    # Assign labels
                    label = schema.classes.index(class_name)
                    all_labels.extend([label] * len(class_samples))
            else:
                # Regression - no class conditioning
                stratum_samples = self._generate_stratified_samples(
                    schema=schema,
                    target_class=None,
                    n_samples=stratum_n,
                    stratify_feature=stratify_by,
                    stratify_value=stratum_value,
                    stratum_idx=stratum_idx,
                )
                all_samples.extend(stratum_samples)
                all_labels.extend([0] * len(stratum_samples))

        # Convert to numpy
        X = samples_to_numpy(all_samples, schema)
        y = np.array(all_labels)

        # Enforce cross-feature constraints
        if enforce_schema_constraints:
            if self.verbose:
                print(f"\n  Enforcing schema constraints...")
            X = enforce_constraints(X, schema, verbose=self.verbose)

        if self.verbose:
            print(f"  Generated {len(X)} stratified samples with {X.shape[1]} features")

            # Show stratum distribution
            stratum_counts = Counter(X[:, feature_idx].astype(int))
            print(f"  {stratify_by} distribution:")
            for val_idx, count in sorted(stratum_counts.items()):
                if val_idx < len(categories):
                    print(f"    {categories[val_idx]}: {count}")

        # Save to disk
        if save:
            output_path = self._save(dataset_name, X, y, schema)
            if self.verbose:
                print(f"  Saved to {output_path}")

        return X, y

    def _generate_stratified_samples(
        self,
        schema: DatasetSchema,
        target_class: str | None,
        n_samples: int,
        stratify_feature: str,
        stratify_value: str,
        stratum_idx: int,
    ) -> list[dict]:
        """Generate samples conditioned on a specific feature value.

        Args:
            schema: Dataset schema
            target_class: Class name (None for regression)
            n_samples: Number of samples to generate
            stratify_feature: Feature name to condition on
            stratify_value: Feature value to condition on
            stratum_idx: Index of the stratum (for diversity)

        Returns:
            List of validated sample dictionaries
        """
        samples = []
        n_calls = (n_samples + self.samples_per_call - 1) // self.samples_per_call

        for call_idx in range(n_calls):
            # Calculate how many samples to request
            remaining = n_samples - len(samples)
            batch_size = min(self.samples_per_call, remaining)

            if batch_size <= 0:
                break

            # Generate prompt with unique batch_id for diversity
            batch_id = stratum_idx * 1000 + call_idx

            # Sample few-shot examples with matching stratify value if possible
            rng = np.random.default_rng(batch_id * 7919 + 42)

            # Filter few-shot examples by stratify value
            fewshot_examples = self._sample_stratified_fewshot_examples(
                schema=schema,
                target_class=target_class,
                n_examples=self.n_examples_per_call,
                stratify_feature=stratify_feature,
                stratify_value=stratify_value,
                rng=rng,
            )

            # Get conditional generation prompt
            prompt = schema.get_conditional_generation_prompt(
                fewshot_examples=fewshot_examples,
                target_class=target_class,
                n_samples=batch_size,
                batch_id=batch_id,
                condition_feature=stratify_feature,
                condition_value=stratify_value,
                enhanced_mode=self.enhanced_prompting,
            )

            # Call LLM with retries
            batch_samples = self._call_with_retry(prompt, batch_size)

            if batch_samples:
                # Validate and repair
                repaired, stats = validate_and_repair(
                    batch_samples, schema, verbose=False
                )
                samples.extend(repaired)

                if self.verbose:
                    success_rate = (stats["valid"] + stats["repaired"]) / max(
                        stats["total"], 1
                    )
                    print(f"      Batch {call_idx + 1}/{n_calls}: {len(repaired)} samples (success rate: {success_rate:.1%})")
            elif self.verbose:
                print(f"      Batch {call_idx + 1}/{n_calls}: failed to generate samples")

            # Small delay to avoid rate limiting
            if call_idx < n_calls - 1:
                time.sleep(0.5)

        return samples[:n_samples]  # Trim to exact count

    def _sample_stratified_fewshot_examples(
        self,
        schema: DatasetSchema,
        target_class: str | None,
        n_examples: int,
        stratify_feature: str,
        stratify_value: str,
        rng: np.random.Generator,
    ) -> list[dict]:
        """Sample few-shot examples matching a specific feature value.

        Prioritizes examples with the matching feature value, but falls back
        to other examples if not enough matches are available.

        Args:
            schema: Dataset schema
            target_class: Target class to filter by (if classification)
            n_examples: Number of examples to sample
            stratify_feature: Feature name to match
            stratify_value: Feature value to match
            rng: Random number generator

        Returns:
            List of example dicts
        """
        feature_names = [f.name for f in schema.features]
        feature_idx = feature_names.index(stratify_feature)
        feature_schema = schema.features[feature_idx]

        # Get the category index for the stratify value
        if feature_schema.categories:
            try:
                value_idx = feature_schema.categories.index(stratify_value)
            except ValueError:
                value_idx = -1
        else:
            value_idx = -1

        # Build mask for matching feature value
        if value_idx >= 0:
            feature_mask = self.fewshot_samples[:, feature_idx] == value_idx
        else:
            feature_mask = np.ones(len(self.fewshot_samples), dtype=bool)

        # Filter by class if specified
        if target_class is not None and self.fewshot_labels is not None:
            class_idx = schema.classes.index(target_class)
            class_mask = self.fewshot_labels == class_idx
            mask = feature_mask & class_mask
        else:
            mask = feature_mask

        # Get matching samples
        X_matched = self.fewshot_samples[mask]

        # If not enough matched, fall back to all samples
        if len(X_matched) < n_examples:
            # Prioritize matched samples, then add from unmatched
            X_unmatched = self.fewshot_samples[~mask]
            n_from_matched = min(len(X_matched), n_examples)
            n_from_unmatched = n_examples - n_from_matched

            matched_indices = rng.choice(len(X_matched), size=n_from_matched, replace=False) if n_from_matched > 0 else []
            unmatched_indices = rng.choice(len(X_unmatched), size=min(n_from_unmatched, len(X_unmatched)), replace=False) if n_from_unmatched > 0 else []

            X_pool = np.concatenate([
                X_matched[matched_indices] if len(matched_indices) > 0 else np.array([]).reshape(0, self.fewshot_samples.shape[1]),
                X_unmatched[unmatched_indices] if len(unmatched_indices) > 0 else np.array([]).reshape(0, self.fewshot_samples.shape[1]),
            ]) if (len(matched_indices) > 0 or len(unmatched_indices) > 0) else X_matched
        else:
            indices = rng.choice(len(X_matched), size=n_examples, replace=False)
            X_pool = X_matched[indices]

        # Convert to dict format
        examples = []
        for row in X_pool:
            example = {}
            for i, (name, feat) in enumerate(zip(feature_names, schema.features)):
                val = row[i]
                # Convert to appropriate type
                if feat.dtype == "integer":
                    example[name] = int(val) if not np.isnan(val) else None
                elif feat.dtype == "categorical" and feat.categories:
                    cat_idx = int(val) if not np.isnan(val) else 0
                    if 0 <= cat_idx < len(feat.categories):
                        example[name] = feat.categories[cat_idx]
                    else:
                        example[name] = feat.categories[0]
                elif feat.dtype == "binary":
                    example[name] = int(val) if not np.isnan(val) else 0
                else:
                    example[name] = float(val) if not np.isnan(val) else None
            examples.append(example)

        return examples

    def _sample_fewshot_examples(
        self,
        schema: DatasetSchema,
        target_class: str | None,
        n_examples: int,
        rng: np.random.Generator,
    ) -> list[dict]:
        """Sample few-shot examples for in-context learning.

        Args:
            schema: Dataset schema
            target_class: Target class to filter by (if classification)
            n_examples: Number of examples to sample
            rng: Random number generator

        Returns:
            List of example dicts
        """
        feature_names = [f.name for f in schema.features]

        # Filter by class if specified
        if target_class is not None and self.fewshot_labels is not None:
            class_idx = schema.classes.index(target_class)
            mask = self.fewshot_labels == class_idx
            X_pool = self.fewshot_samples[mask]
        else:
            X_pool = self.fewshot_samples

        # Sample indices
        n_available = len(X_pool)
        n_to_sample = min(n_examples, n_available)
        indices = rng.choice(n_available, size=n_to_sample, replace=False)

        # Convert to dict format
        examples = []
        for idx in indices:
            row = X_pool[idx]
            example = {}
            for i, (name, feat) in enumerate(zip(feature_names, schema.features)):
                val = row[i]
                # Convert to appropriate type
                if feat.dtype == "integer":
                    example[name] = int(val) if not np.isnan(val) else None
                elif feat.dtype == "categorical" and feat.categories:
                    # Convert index to category name
                    cat_idx = int(val) if not np.isnan(val) else 0
                    if 0 <= cat_idx < len(feat.categories):
                        example[name] = feat.categories[cat_idx]
                    else:
                        example[name] = feat.categories[0]
                elif feat.dtype == "binary":
                    example[name] = int(val) if not np.isnan(val) else 0
                else:
                    example[name] = float(val) if not np.isnan(val) else None
            examples.append(example)

        return examples

    def _generate_class_samples(
        self,
        schema: DatasetSchema,
        target_class: str | None,
        n_samples: int,
        class_idx: int = 0,
    ) -> list[dict]:
        """Generate samples for a single class.

        Args:
            schema: Dataset schema
            target_class: Class name (None for regression)
            n_samples: Number of samples to generate
            class_idx: Index of the class (for batch diversity)

        Returns:
            List of validated sample dictionaries
        """
        samples = []
        n_calls = (n_samples + self.samples_per_call - 1) // self.samples_per_call

        for call_idx in range(n_calls):
            # Calculate how many samples to request
            remaining = n_samples - len(samples)
            batch_size = min(self.samples_per_call, remaining)

            if batch_size <= 0:
                break

            # Generate prompt with unique batch_id for diversity
            batch_id = class_idx * 100 + call_idx

            # Use few-shot prompt if examples are available
            if self.use_fewshot:
                # Sample different examples for each call for diversity
                rng = np.random.default_rng(batch_id * 7919 + 42)
                fewshot_examples = self._sample_fewshot_examples(
                    schema=schema,
                    target_class=target_class,
                    n_examples=self.n_examples_per_call,
                    rng=rng,
                )
                prompt = schema.get_fewshot_generation_prompt(
                    fewshot_examples=fewshot_examples,
                    target_class=target_class,
                    n_samples=batch_size,
                    batch_id=batch_id,
                    enhanced_mode=self.enhanced_prompting,
                )
            else:
                # Get statistics for this class if using hard constraints
                feature_stats = None
                if self.use_hard_constraints:
                    # Try per-class first, fall back to overall
                    stats_key = target_class if target_class in self.statistics_cache else "all"
                    dataset_stats = self.statistics_cache.get(stats_key)
                    if dataset_stats:
                        feature_stats = dataset_stats.feature_stats

                prompt = schema.get_generation_prompt(
                    target_class=target_class,
                    n_samples=batch_size,
                    batch_id=batch_id,
                    feature_stats=feature_stats,
                    strict_constraints=self.use_hard_constraints,
                )

            # Call LLM with retries
            batch_samples = self._call_with_retry(prompt, batch_size)

            if batch_samples:
                # Validate and repair
                repaired, stats = validate_and_repair(
                    batch_samples, schema, verbose=False
                )
                samples.extend(repaired)

                if self.verbose:
                    success_rate = (stats["valid"] + stats["repaired"]) / max(
                        stats["total"], 1
                    )
                    print(f"    Batch {call_idx + 1}/{n_calls}: {len(repaired)} samples (success rate: {success_rate:.1%})")
            elif self.verbose:
                print(f"    Batch {call_idx + 1}/{n_calls}: failed to generate samples")

            # Small delay to avoid rate limiting
            if call_idx < n_calls - 1:
                time.sleep(0.5)

        return samples[:n_samples]  # Trim to exact count

    def _call_with_retry(
        self,
        prompt: str,
        expected_count: int,
    ) -> list[dict] | None:
        """Call LLM with retry logic.

        Args:
            prompt: Generation prompt
            expected_count: Expected number of samples

        Returns:
            List of sample dicts, or None on failure
        """
        for _attempt in range(self.max_retries):
            try:
                result = self.provider.generate_json(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=4096,
                )

                if result is None:
                    continue

                # Handle both list and dict with "samples" key
                if isinstance(result, list):
                    samples = result
                elif isinstance(result, dict) and "samples" in result:
                    samples = result["samples"]
                else:
                    continue

                # Basic validation
                if not samples or not isinstance(samples[0], dict):
                    continue

                return samples

            except Exception as e:
                if self.verbose:
                    print(f"    Retry {_attempt + 1}/{self.max_retries}: {type(e).__name__}")
                time.sleep(1)

        return None

    def _save(
        self,
        dataset_name: str,
        X: np.ndarray,
        y: np.ndarray,
        schema: DatasetSchema,
    ) -> Path:
        """Save generated data to disk.

        Args:
            dataset_name: Dataset name
            X: Feature matrix
            y: Labels
            schema: Dataset schema

        Returns:
            Path to saved file
        """
        model_name = self.provider.get_model_name().replace(":", "_").replace("/", "_")

        # Build suffix: custom suffix + hard constraints mode
        suffix = self.output_suffix
        if self.use_hard_constraints:
            suffix += "_hard_constraints"
        output_path = OUTPUT_DIR / f"{dataset_name}_{model_name}{suffix}.npz"

        # Save data
        np.savez(
            output_path,
            X=X,
            y=y,
            feature_names=[f.name for f in schema.features],
        )

        # Also save metadata
        meta_path = output_path.with_suffix(".json")
        meta = {
            "dataset_name": dataset_name,
            "model": self.provider.get_model_name(),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "task_type": schema.task_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_hard_constraints": self.use_hard_constraints,
        }
        if self.use_hard_constraints:
            # Add info about training data used for constraints
            n_train = None
            if self.statistics_cache:
                # Get number of samples from any cached stats
                for stats in self.statistics_cache.values():
                    n_train = stats.n_samples
                    break
            meta["n_train_samples"] = n_train
            meta["constraint_method"] = "data_driven_bounds"

        if schema.task_type == "classification":
            meta["class_distribution"] = dict(zip(*np.unique(y, return_counts=True)))
            meta["class_distribution"] = {
                int(k): int(v) for k, v in meta["class_distribution"].items()
            }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return output_path


def load_llm_generated(
    dataset_name: str,
    model_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load LLM-generated data from disk.

    Args:
        dataset_name: Dataset name
        model_name: Model name filter (e.g., "gemini" or "qwen")
                   If None, loads the first matching file

    Returns:
        Tuple of (X, y, feature_names)
    """
    # Find matching files
    pattern = f"{dataset_name}_*.npz"
    files = list(OUTPUT_DIR.glob(pattern))

    if model_name:
        files = [f for f in files if model_name.lower() in f.stem.lower()]

    if not files:
        raise FileNotFoundError(
            f"No LLM-generated data found for {dataset_name}"
            + (f" with model {model_name}" if model_name else "")
        )

    # Load most recent
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    data = np.load(files[0])

    X = data["X"]
    y = data["y"]
    feature_names = list(
        data.get("feature_names", [f"feature_{i}" for i in range(X.shape[1])])
    )

    return X, y, feature_names
