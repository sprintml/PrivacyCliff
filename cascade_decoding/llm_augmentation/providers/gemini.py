"""Google Gemini provider using Google AI Studio API."""
from __future__ import annotations

import os

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """Gemini provider using Google AI Studio API.

    Uses the google-generativeai package to interact with Gemini models.

    Example:
        >>> provider = GeminiProvider(api_key="your-api-key")
        >>> response = provider.generate("Generate 5 random numbers")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ):
        """Initialize Gemini provider.

        Args:
            api_key: Google AI Studio API key. If None, reads from
                     GOOGLE_API_KEY environment variable.
            model: Model name (default: gemini-1.5-flash for speed/cost)
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not found. "
                "Install with: pip install google-generativeai"
            )

        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key or set GOOGLE_API_KEY env var."
            )

        self.model_name = model
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response from Gemini.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            The model's response as a string
        """
        import google.generativeai as genai

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        # Handle blocked or empty responses
        try:
            return response.text
        except ValueError as e:
            # Response blocked by safety filters or empty
            raise RuntimeError(f"Gemini response blocked or empty: {e}") from e

    def get_model_name(self) -> str:
        """Return the model identifier."""
        return f"gemini:{self.model_name}"
