"""LLM provider implementations."""

from .base import LLMProvider
from .gemini import GeminiProvider

__all__ = ["GeminiProvider", "LLMProvider"]


# Qwen provider loaded lazily to avoid import errors if transformers not available
def get_qwen_provider():
    from .qwen import QwenProvider

    return QwenProvider
