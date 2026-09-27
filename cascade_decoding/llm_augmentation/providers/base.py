"""Abstract base class for LLM providers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement the `generate` method that takes
    a prompt and returns the model's response.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            The model's response as a string
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier."""

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict | list | None:
        """Generate and parse JSON response.

        Args:
            prompt: The input prompt (should request JSON output)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Parsed JSON as dict/list, or None if parsing fails
        """
        import json
        import re

        response = self.generate(prompt, temperature, max_tokens)

        # Try to extract JSON from response
        # Handle cases where model wraps JSON in markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON array or object
            json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", response)
            json_str = json_match.group(1) if json_match else response

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
