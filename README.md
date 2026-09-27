# Auditing Privacy Leakage in Tabular Foundation Model Embeddings

This repository contains the official implementation for our systematic privacy audit of Tabular Foundation Model (TabFM) embeddings, including the **Cascade Probing** method for measuring attribute recoverability.

## Overview

Tabular Foundation Models (TabFMs) produce context-aware embeddings that are increasingly stored, shared, and reused as privacy-preserving substitutes for raw tabular records. We present a systematic audit of the attribute-level information content of these embeddings: given an embedding and varying amounts of side information, how accurately can sensitive attributes be recovered?

**Cascade Probing** is a sequential probing method that measures recoverability while accounting for inter-attribute dependencies via the probabilistic chain rule, significantly outperforming joint probing and optimization-based inversion baselines.

### Key Findings

- **High Baseline Recoverability**: 65-95% of sensitive attributes can be recovered from embeddings alone (no side information)
- **Privacy Cliff**: Revealing a single known attribute sharply increases the recoverability of all remaining attributes
- **Cross-Architecture Generalization**: The Privacy Cliff manifests across four TabFM architectures (TabPFN, TabDPT, Mitra, TabICL) and eight diverse datasets
- **Defenses Insufficient**: Standard embedding transformations (noise injection, PCA, knowledge distillation) do not eliminate leakage

### Main Contributions

1. **Cascade Probing**: Sequential attribute recovery via chain-rule decomposition with greedy strategy discovery
2. **Privacy Cliff Discovery**: Identifies the critical tipping point where minimal side information causes a drastic jump in recoverability
3. **Comprehensive Audit**: Systematic evaluation across four TabFMs, eight datasets, and three knowledge regimes (Shadow, Few-Shot, Zero-Shot)

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0

```bash
pip install -r requirements.txt
```

### Environment Variables (Optional)

Configure custom paths via environment variables:

```bash
export CASCADE_DATA_DIR=/path/to/data
export CASCADE_OUTPUT_DIR=/path/to/outputs
export CASCADE_CACHE_DIR=/path/to/cache
export HF_HUB_CACHE=/path/to/huggingface  # For LLM models
```

## Quick Start

### 1. Run Cascade Probing

```python
import numpy as np
import torch
from tabpfn import TabPFNClassifier
from cascade_decoding.attacks.cascade_attack import CascadeAttack
from cascade_decoding.common.feature_metadata import ADULT_FEATURES
from cascade_decoding import load_dataset, split_data, get_tabpfn_embeddings

# Load data
X, y, feature_names, task_type = load_dataset("adult")
splits = split_data(X, y)

# Initialize TabPFN model
model = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
model.fit(splits["X_support"], splits["y_support"])

# Configure feature indices
sensitive_indices = [4, 5, 6]  # marital-status, occupation, relationship
known_indices = [0, 2, 7, 8, 12]  # age, education, race, sex, native-country
irrelevant_indices = [i for i in range(X.shape[1])
                      if i not in sensitive_indices + known_indices]

# Initialize Cascade Probing
attack = CascadeAttack(
    sensitive_indices=sensitive_indices,
    known_indices=known_indices,
    irrelevant_indices=irrelevant_indices,
    feature_metadata=ADULT_FEATURES,
)

# Phase 1: Strategy Discovery (train decoders on shadow data)
attack.fit(X_shadow=splits["X_shadow"], tabpfn_model=model)

# Phase 2: Cascading Inference (measure recoverability on target data)
X_target = splits["X_victim"]
E_target = get_tabpfn_embeddings(model, X_target)

# Prepare known features (NaN for unknown)
X_known_target = X_target.copy()
for idx in sensitive_indices + irrelevant_indices:
    X_known_target[:, idx] = np.nan

predictions, stats = attack.attack(
    E_victim=E_target,
    X_known_victim=X_known_target,
    tabpfn_model=model,
)

print(f"Audit completed: {len(predictions)} attributes recovered")
```

### 2. LLM-Based Synthetic Data Generation (Few-Shot / Zero-Shot Regimes)

```python
from cascade_decoding.llm_augmentation.generator import LLMTabularGenerator
from cascade_decoding.llm_augmentation.providers.qwen import QwenProvider

# Initialize LLM provider
provider = QwenProvider(model_size="7B")

# Generate synthetic reference data
generator = LLMTabularGenerator(provider)
X_synthetic, y_synthetic = generator.generate(
    dataset_name="adult",
    n_samples=1000,
)
```

## Project Structure

```
cascade_decoding/
├── attacks/                  # Probing methods
│   └── cascade_attack.py    # Cascade Probing implementation
├── common/                   # Shared utilities
│   ├── constants.py          # Configuration and paths
│   ├── data.py              # Dataset loading and splitting
│   ├── evaluation.py        # Recoverability metrics
│   ├── embeddings.py        # TabFM embedding extraction
│   ├── feature_metadata.py  # Dataset feature configurations
│   └── target_normalizer.py # Feature normalization
├── surrogates/              # Decoder models
│   └── residual_decoder.py  # Joint probing baseline (residual decoder)
└── llm_augmentation/        # LLM-based data synthesis
    ├── generator.py         # Main generator interface
    ├── schema.py            # Dataset schema definitions
    ├── validator.py         # Sample validation
    └── providers/           # LLM provider implementations
        ├── qwen.py          # Qwen2.5-Instruct (local)
        └── gemini.py        # Google Gemini (API)
```

## Datasets

| Dataset | Features | Samples | Type | Domain |
|---------|----------|---------|------|--------|
| Adult Census | 13 | 48,842 | Classification | Demographics |
| German Credit | 20 | 1,000 | Classification | Finance |
| Heart Disease | 13 | 303 | Classification | Healthcare |
| Diabetes (Pima) | 8 | 768 | Classification | Healthcare |

## Evaluation Metrics

- **Categorical Features**: Accuracy (higher = more recoverable)
- **Continuous Features**: Normalized Mean Absolute Error (NMAE, lower = more recoverable)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work builds upon [TabPFN](https://github.com/priorlabs/tabpfn) by Prior Labs.
