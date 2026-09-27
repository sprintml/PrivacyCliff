"""Constants and configuration for model inversion attacks.

This module contains all shared constants, dataset registry, and default
hyperparameters used across the model inversion attack implementations.
"""
from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# Paths (configurable via environment variables)
# =============================================================================

# Base directories - can be overridden via environment variables
CACHE_DIR = Path(os.environ.get(
    "CASCADE_CACHE_DIR",
    os.path.expanduser("~/.cache/cascade_decoding")
))
DATA_DIR = Path(os.environ.get(
    "CASCADE_DATA_DIR",
    os.path.expanduser("~/.cache/cascade_decoding/data")
))
OUTPUT_BASE = Path(os.environ.get(
    "CASCADE_OUTPUT_DIR",
    os.path.expanduser("~/.cache/cascade_decoding/outputs")
))

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Models directory (organized structure under output directory)
MODEL_DIR = OUTPUT_BASE / "models"
SURROGATE_MLP_DIR = MODEL_DIR / "surrogates" / "mlp"
SURROGATE_TABPFN_DIR = MODEL_DIR / "surrogates" / "tabpfn"
GENERATOR_DIR = MODEL_DIR / "generators"
DECODER_DIR = MODEL_DIR / "decoders"

# Set environment variables for TabPFN (if not already set)
if "TABPFN_MODEL_CACHE_DIR" not in os.environ:
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(CACHE_DIR)
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

# =============================================================================
# Dataset Registry
# =============================================================================

DATASETS = {
    # Classification datasets
    "iris": ("iris.npz", "classification"),
    "breast_cancer": ("breast_cancer.npz", "classification"),
    "parkinsons": ("parkinsons.npz", "classification"),
    "heart_statlog": ("heart_statlog.npz", "classification"),
    "diabetes_pima": ("diabetes_pima.npz", "classification"),
    "covertype": ("covertype.npz", "classification"),
    "adult": ("adult.npz", "classification"),  # Adult Census Income dataset
    # Privacy attack evaluation datasets (Issue #33)
    "german_credit": ("german_credit.npz", "classification"),  # German Credit dataset
    "heart_disease": ("heart_disease.npz", "classification"),  # Cleveland Heart Disease dataset
    # Regression datasets
    "diabetes": ("diabetes_sklearn.npz", "regression"),
    "cholesterol": ("cholesterol.npz", "regression"),
    "boston_housing": ("boston_housing.npz", "regression"),
    "bike_sharing": ("bike_sharing.npz", "regression"),
    "grid_stability": ("grid_stability.npz", "regression"),
    "concrete_strength": ("concrete_strength.npz", "regression"),
    "liver_disorders": ("liver_disorders.npz", "regression"),
}

CLASSIFICATION_DATASETS = [k for k, v in DATASETS.items() if v[1] == "classification"]
REGRESSION_DATASETS = [k for k, v in DATASETS.items() if v[1] == "regression"]

# =============================================================================
# Model Constants
# =============================================================================

EMBEDDING_DIM = 192  # TabPFN internal embedding dimension
MAX_SAMPLES = 10000  # Maximum samples for TabPFN

# =============================================================================
# Default Split Ratios
# =============================================================================

DEFAULT_SUPPORT_RATIO = 0.40  # TabPFN context (attacker does NOT have access)
DEFAULT_SHADOW_RATIO = 0.50  # Attacker's data for training
DEFAULT_VICTIM_RATIO = 0.10  # Target data to reconstruct

# =============================================================================
# Default Hyperparameters
# =============================================================================

# MLP Surrogate defaults
MLP_SURROGATE_DEFAULTS = {
    "hidden_layers": [256, 256, 192],
    "dropout": 0.1,
    "batch_norm": True,
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}

# Larger MLP for augmented training (more capacity for larger datasets)
MLP_SURROGATE_LARGE = {
    "hidden_layers": [512, 256, 256, 192],
    "dropout": 0.1,
    "batch_norm": True,
    "epochs": 300,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}

# Data augmentation defaults
AUGMENTATION_DEFAULTS = {
    "enable": False,
    "n_synthetic": 10000,
    "lhs_ratio": 0.1,  # 10% LHS, 90% mixup (mixup works better for attack)
    "mixup_alpha": 0.4,  # Beta distribution parameter for mixup
    "random_state": 42,
    "batch_size": 1000,  # Batch size for TabPFN queries
    "verbose": True,
}

# TabPFN Surrogate defaults
TABPFN_SURROGATE_DEFAULTS = {
    "n_support": 50,
    "n_iters": 1000,
    "lr": 0.01,
    "optimize_y": True,
    "eval_every": 50,
}

# Generator (WGAN-GP) defaults
GENERATOR_DEFAULTS = {
    "z_dim": 64,
    "hidden_layers": [128, 256, 256],
    "batch_norm": True,
    "epochs": 500,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "n_critic": 5,
    "lambda_gp": 10.0,
}

# Discriminator defaults
DISCRIMINATOR_DEFAULTS = {
    "hidden_layers": [256, 128, 64],
}

# Attack defaults
ATTACK_DEFAULTS = {
    "n_iters": 200,
    "lr": 0.01,
    "lambda_prior": 0.001,
    "lambda_reg": 0.001,
    # PGD (Projected Gradient Descent) settings
    "use_pgd": False,  # Enable PGD projection to feature bounds
    "pgd_margin": 0.05,  # Extend bounds by 5% on each side
    "auto_detect_integers": True,  # Auto-detect integer features
    "integer_threshold": 0.95,  # Fraction of values that must be integers
}

# Decoder (Method 1) defaults
DECODER_DEFAULTS = {
    "hidden_layers": [256, 128],
    "max_iter": 1000,  # For sklearn MLPRegressor
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
}

# Attack defaults for GMI
ATTACK_DEFAULTS["n_restarts"] = 5
ATTACK_DEFAULTS["invert_iters"] = 500
