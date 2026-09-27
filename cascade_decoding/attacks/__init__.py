"""Cascade Decoding attack implementation for TabPFN embedding inversion.

This module provides the main attack method from the paper:

- CascadeAttack: Sequential attribute inference that exploits feature
  dependencies via auto-regressive decoding with greedy strategy discovery.

The attack uses residual representation to isolate sensitive signal
from embeddings by computing Delta = E_victim - E_base.
"""

from .base import AttackBase, AttackResult
from .cascade_attack import CascadeAttack, run_cascade_attack

__all__ = [
    # Base classes
    "AttackBase",
    "AttackResult",
    # Cascade Attack
    "CascadeAttack",
    "run_cascade_attack",
]
