"""Cascade Decoding: Attribute Inference Attack on TabPFN.

This package implements the Cascade Decoding attack to recover sensitive
features from TabPFN (Tabular Prior-data Fitted Network) embeddings.

The attack exploits the sequential structure of feature dependencies
via auto-regressive decoding with greedy strategy discovery.

Example usage:
    from cascade_decoding import load_dataset, split_data, get_tabpfn_embeddings
    from cascade_decoding.attacks import CascadeAttack

    # Load and split data
    X, y, names, task = load_dataset("adult")
    splits = split_data(X, y)

    # Initialize attack
    attack = CascadeAttack(
        sensitive_indices=[4, 5, 6],
        known_indices=[0, 2, 7, 8],
        irrelevant_indices=[...],
        feature_metadata=ADULT_FEATURES,
    )

    # Train and attack
    attack.fit(X_shadow=splits["X_shadow"], tabpfn_model=model)
    predictions, stats = attack.attack(E_victim, X_known_victim, tabpfn_model=model)
"""

# Common utilities
from .common import (
    # Paths
    DATA_DIR,
    DATASETS,
    # Constants
    EMBEDDING_DIM,
    MAX_SAMPLES,
    MODEL_DIR,
    OUTPUT_BASE,
    # Evaluation
    evaluate_reconstruction,
    evaluate_surrogate,
    get_dataset_info,
    get_embeddings_with_support,
    # Embedding functions
    get_tabpfn_embeddings,
    get_tabpfn_model,
    # Data functions
    load_dataset,
    print_reconstruction_results,
    split_data,
)

# Attacks
from .attacks import (
    AttackBase,
    AttackResult,
    CascadeAttack,
    run_cascade_attack,
)

# Surrogates
from .surrogates import (
    ResidualDecoder,
    ResidualDecoderNetwork,
    SurrogateBase,
)

__version__ = "0.1.0"

__all__ = [
    # Common
    "DATA_DIR",
    "DATASETS",
    "EMBEDDING_DIM",
    "MAX_SAMPLES",
    "MODEL_DIR",
    "OUTPUT_BASE",
    # Attacks
    "AttackBase",
    "AttackResult",
    "CascadeAttack",
    "run_cascade_attack",
    # Surrogates
    "ResidualDecoder",
    "ResidualDecoderNetwork",
    "SurrogateBase",
    # Functions
    "evaluate_reconstruction",
    "evaluate_surrogate",
    "get_dataset_info",
    "get_embeddings_with_support",
    "get_tabpfn_embeddings",
    "get_tabpfn_model",
    "load_dataset",
    "print_reconstruction_results",
    "split_data",
]
