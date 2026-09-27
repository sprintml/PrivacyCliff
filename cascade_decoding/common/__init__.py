"""Common utilities for model inversion attacks.

This module provides shared functionality including:
- Data loading and splitting
- Embedding extraction
- Evaluation metrics
- Constants and configuration
"""

from .augmentation import (
    augment_training_data,
    generate_gaussian_noise_samples,
    generate_lhs_samples,
    generate_mixup_samples,
)
from .constants import (
    ATTACK_DEFAULTS,
    AUGMENTATION_DEFAULTS,
    CLASSIFICATION_DATASETS,
    # Paths
    DATA_DIR,
    # Dataset info
    DATASETS,
    DECODER_DEFAULTS,
    DECODER_DIR,
    DEFAULT_SHADOW_RATIO,
    # Split ratios
    DEFAULT_SUPPORT_RATIO,
    DEFAULT_VICTIM_RATIO,
    DISCRIMINATOR_DEFAULTS,
    # Constants
    EMBEDDING_DIM,
    GENERATOR_DEFAULTS,
    GENERATOR_DIR,
    MAX_SAMPLES,
    # Default hyperparameters
    MLP_SURROGATE_DEFAULTS,
    MLP_SURROGATE_LARGE,
    MODEL_DIR,
    OUTPUT_BASE,
    REGRESSION_DATASETS,
    SURROGATE_MLP_DIR,
    SURROGATE_TABPFN_DIR,
    TABPFN_SURROGATE_DEFAULTS,
)
from .constraints import (
    FeatureBounds,
    extract_bounds_from_data,
)
from .data import (
    get_dataset_info,
    get_tabpfn_model,
    load_dataset,
    split_data,
    subsample_shadow,
)
from .embeddings import (
    compute_embedding_distance,
    find_nearest_neighbor_by_embedding,
    get_embeddings_with_support,
    get_tabpfn_embeddings,
)
from .evaluation import (
    evaluate_attribute_inference,
    evaluate_conditional_attack,
    evaluate_generator,
    evaluate_reconstruction,
    evaluate_surrogate,
    print_conditional_attack_results,
    print_reconstruction_results,
    print_surrogate_results,
)
from .feature_metadata import (
    ADULT_FEATURES,
    DATASET_FEATURE_METADATA,
    get_categorical_indices,
    get_feature_dtype,
    get_feature_indices_by_dtype,
    get_feature_indices_by_sensitivity,
    get_feature_metadata,
    get_feature_n_categories,
    get_feature_name,
    get_irrelevant_indices,
    get_known_indices,
    get_sensitive_indices,
    print_feature_summary,
)

__all__ = [
    # Feature metadata
    "ADULT_FEATURES",
    "ATTACK_DEFAULTS",
    "AUGMENTATION_DEFAULTS",
    "CLASSIFICATION_DATASETS",
    # Dataset info
    "DATASET_FEATURE_METADATA",
    "DATASETS",
    # Paths
    "DATA_DIR",
    "DECODER_DEFAULTS",
    "DECODER_DIR",
    "DEFAULT_SHADOW_RATIO",
    # Split ratios
    "DEFAULT_SUPPORT_RATIO",
    "DEFAULT_VICTIM_RATIO",
    "DISCRIMINATOR_DEFAULTS",
    # Constants
    "EMBEDDING_DIM",
    "GENERATOR_DEFAULTS",
    "GENERATOR_DIR",
    "MAX_SAMPLES",
    # Default hyperparameters
    "MLP_SURROGATE_DEFAULTS",
    "MLP_SURROGATE_LARGE",
    "MODEL_DIR",
    "OUTPUT_BASE",
    "REGRESSION_DATASETS",
    "SURROGATE_MLP_DIR",
    "SURROGATE_TABPFN_DIR",
    "TABPFN_SURROGATE_DEFAULTS",
    # Constraint functions
    "FeatureBounds",
    # Augmentation functions
    "augment_training_data",
    "compute_embedding_distance",
    # Evaluation functions
    "evaluate_attribute_inference",
    "evaluate_conditional_attack",
    "evaluate_generator",
    "evaluate_reconstruction",
    "evaluate_surrogate",
    "extract_bounds_from_data",
    "find_nearest_neighbor_by_embedding",
    "generate_gaussian_noise_samples",
    "generate_lhs_samples",
    "generate_mixup_samples",
    # Feature metadata functions
    "get_categorical_indices",
    "get_dataset_info",
    "get_embeddings_with_support",
    "get_feature_dtype",
    "get_feature_indices_by_dtype",
    "get_feature_indices_by_sensitivity",
    "get_feature_metadata",
    "get_feature_n_categories",
    "get_feature_name",
    "get_irrelevant_indices",
    "get_known_indices",
    "get_sensitive_indices",
    # Embedding functions
    "get_tabpfn_embeddings",
    "get_tabpfn_model",
    # Data functions
    "load_dataset",
    "print_conditional_attack_results",
    "print_feature_summary",
    "print_reconstruction_results",
    "print_surrogate_results",
    "split_data",
    "subsample_shadow",
]
