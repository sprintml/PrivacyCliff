"""Surrogate models for cascade attribute inference attacks.

Available surrogates:
- ResidualDecoder: Discriminative decoder (predicts x_s from Delta)
  Used by CascadeAttack for sequential attribute inference.
"""

from .base import SurrogateBase
from .residual_decoder import ResidualDecoder, ResidualDecoderNetwork

__all__ = [
    # Base class
    "SurrogateBase",
    # Residual Decoder (used by CascadeAttack)
    "ResidualDecoder",
    "ResidualDecoderNetwork",
]
