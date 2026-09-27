"""LLM-guided semantic data augmentation for model inversion attacks.

This module provides utilities to generate semantically valid synthetic
tabular data using Large Language Models (LLMs). Unlike traditional
augmentation methods (Mixup, LHS) that ignore feature semantics, LLM-based
generation respects:
- Data types (integer vs float vs categorical)
- Valid ranges and categories
- Cross-feature constraints (e.g., pregnant=0 if male)
- Class-conditional distributions

Supported LLM providers:
- Gemini (Google AI Studio API)
- Qwen2.5-Instruct (local, via transformers)
"""

from .generator import LLMTabularGenerator
from .schema import DatasetSchema, load_schema, save_schema
from .validator import validate_and_repair

__all__ = [
    "DatasetSchema",
    "LLMTabularGenerator",
    "load_schema",
    "save_schema",
    "validate_and_repair",
]
