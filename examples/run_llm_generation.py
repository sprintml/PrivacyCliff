#!/usr/bin/env python3
"""Example: Generate synthetic tabular data using LLM.

This script demonstrates LLM-based synthetic data generation for
few-shot and zero-shot attack scenarios.

Usage:
    # With Qwen (local model)
    python run_llm_generation.py --dataset adult --n_samples 100 --provider qwen

    # With Gemini (requires API key)
    export GOOGLE_API_KEY=your_key
    python run_llm_generation.py --dataset adult --n_samples 100 --provider gemini
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tabular data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "german_credit", "heart_disease", "diabetes_pima"],
        help="Dataset schema to use",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="qwen",
        choices=["qwen", "gemini"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7B",
        choices=["0.5B", "1.5B", "3B", "7B"],
        help="Qwen model size (only for qwen provider)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)",
    )
    args = parser.parse_args()

    print(f"Initializing {args.provider} provider...")

    if args.provider == "qwen":
        try:
            from cascade_decoding.llm_augmentation.providers.qwen import QwenProvider
            provider = QwenProvider(model_size=args.model_size)
            print(f"  Loaded Qwen2.5-{args.model_size}-Instruct")
        except Exception as e:
            print(f"Error loading Qwen: {e}")
            print("Make sure transformers is installed: pip install transformers")
            sys.exit(1)
    else:  # gemini
        import os
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set")
            sys.exit(1)
        try:
            from cascade_decoding.llm_augmentation.providers.gemini import GeminiProvider
            provider = GeminiProvider(api_key=api_key)
            print("  Loaded Gemini provider")
        except Exception as e:
            print(f"Error loading Gemini: {e}")
            sys.exit(1)

    # Initialize generator
    from cascade_decoding.llm_augmentation.generator import LLMTabularGenerator
    generator = LLMTabularGenerator(
        provider=provider,
        temperature=0.8,
        samples_per_call=10,
        verbose=True,
    )

    print(f"\nGenerating {args.n_samples} samples for {args.dataset}...")
    X, y = generator.generate(
        dataset_name=args.dataset,
        n_samples=args.n_samples,
    )

    print(f"\nGeneration complete!")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"synthetic_{args.dataset}_{args.n_samples}.npz")

    np.savez(output_path, X=X, y=y)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
