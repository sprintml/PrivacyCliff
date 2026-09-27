#!/usr/bin/env python3
"""Example: Run Cascade Attack on Adult Census dataset.

This script demonstrates the full pipeline for running a cascade attribute
inference attack on TabPFN embeddings.

Usage:
    python run_cascade_attack.py --dataset adult --n_sensitive 3 --n_known 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from cascade_decoding import load_dataset, split_data, get_tabpfn_embeddings
from cascade_decoding.attacks.cascade_attack import CascadeAttack
from cascade_decoding.common.feature_metadata import (
    ADULT_FEATURES,
    GERMAN_CREDIT_FEATURES,
    HEART_DISEASE_FEATURES,
    DIABETES_PIMA_FEATURES,
)

# Dataset to feature metadata mapping
FEATURE_METADATA = {
    "adult": ADULT_FEATURES,
    "german_credit": GERMAN_CREDIT_FEATURES,
    "heart_disease": HEART_DISEASE_FEATURES,
    "diabetes_pima": DIABETES_PIMA_FEATURES,
}


def main():
    parser = argparse.ArgumentParser(description="Run Cascade Attack on TabPFN")
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "german_credit", "heart_disease", "diabetes_pima"],
        help="Dataset to attack",
    )
    parser.add_argument(
        "--n_sensitive",
        type=int,
        default=3,
        help="Number of sensitive features to infer",
    )
    parser.add_argument(
        "--n_known",
        type=int,
        default=5,
        help="Number of known public features",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    X, y, feature_names, task_type = load_dataset(args.dataset)
    print(f"  Shape: {X.shape}, Task: {task_type}")
    print(f"  Features: {feature_names}")

    # Split data
    splits = split_data(X, y)
    print(f"  Support: {splits['X_support'].shape[0]} samples")
    print(f"  Shadow: {splits['X_shadow'].shape[0]} samples")
    print(f"  Victim: {splits['X_victim'].shape[0]} samples")

    # Get feature metadata
    feature_metadata = FEATURE_METADATA[args.dataset]

    # Define sensitive, known, and irrelevant features
    n_features = X.shape[1]
    known_indices = list(range(args.n_known))
    sensitive_indices = list(range(args.n_known, args.n_known + args.n_sensitive))
    # Remaining features are irrelevant
    irrelevant_indices = list(range(args.n_known + args.n_sensitive, n_features))

    print(f"\nAttack Configuration:")
    print(f"  Known features (K={len(known_indices)}): {known_indices}")
    print(f"  Sensitive features (S={len(sensitive_indices)}): {sensitive_indices}")
    print(f"  Irrelevant features: {irrelevant_indices}")

    # Initialize TabPFN model
    print(f"\nInitializing TabPFN on {args.device}...")
    try:
        from tabpfn import TabPFNClassifier
        model = TabPFNClassifier(device=args.device)
        model.fit(splits["X_support"], splits["y_support"])
        print("  TabPFN fitted successfully")
    except ImportError:
        print("Error: TabPFN not installed. Run: pip install tabpfn")
        sys.exit(1)

    # Initialize attack
    print("\nInitializing Cascade Attack...")
    attack = CascadeAttack(
        sensitive_indices=sensitive_indices,
        known_indices=known_indices,
        irrelevant_indices=irrelevant_indices,
        feature_metadata=feature_metadata,
        device=args.device,
    )

    # Train on shadow data (Phase 1: Strategy Discovery)
    print("Training cascade decoders (Strategy Discovery)...")
    attack.fit(
        X_shadow=splits["X_shadow"],
        tabpfn_model=model,
        epochs=100,
        verbose=True,
    )

    # Prepare victim data for attack
    X_victim = splits["X_victim"]

    # Get victim embeddings (full embeddings)
    print("\nExtracting victim embeddings...")
    E_victim = get_tabpfn_embeddings(model, X_victim)

    # Create X_known_victim with NaN for unknown features
    X_known_victim = X_victim.copy()
    for idx in sensitive_indices + irrelevant_indices:
        X_known_victim[:, idx] = np.nan

    # Run attack (Phase 2: Cascading Inference)
    print("Running cascading inference attack...")
    predictions, stats = attack.attack(
        E_victim=E_victim,
        X_known_victim=X_known_victim,
        tabpfn_model=model,
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 50)
    print("ATTACK RESULTS")
    print("=" * 50)

    for idx, feat_idx in enumerate(sensitive_indices):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
        if f"accuracy_{idx}" in stats:
            print(f"  {feat_name}: Accuracy = {stats[f'accuracy_{idx}']:.2%}")
        elif f"nmae_{idx}" in stats:
            print(f"  {feat_name}: NMAE = {stats[f'nmae_{idx}']:.4f}")

    if "overall_accuracy" in stats:
        print(f"\nOverall Accuracy: {stats['overall_accuracy']:.2%}")
    if "overall_nmae" in stats:
        print(f"Overall NMAE: {stats['overall_nmae']:.4f}")


if __name__ == "__main__":
    main()
