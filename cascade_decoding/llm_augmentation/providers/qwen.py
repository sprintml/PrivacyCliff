"""Qwen2.5-Instruct provider using local transformers."""
from __future__ import annotations

import os
from pathlib import Path

from .base import LLMProvider

# Default HuggingFace cache directory (configurable via environment variable)
DEFAULT_HF_CACHE = os.environ.get(
    "HF_HUB_CACHE",
    os.path.expanduser("~/.cache/huggingface/hub")
)

# Known model snapshots (commit hashes)
MODEL_SNAPSHOTS = {
    "0.5B": "7ae8e8e0f215442c9c17c35d12eed3d419909f0c",
    "1.5B": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "3B": "5c2f7b01c8952e95d9b4883d7a00a3daaf6bf5d2",
    "7B": "a09a35458c702b33eeacc393d103063234e8bc28",
}


def get_local_model_path(model_size: str, cache_dir: str) -> str | None:
    """Get the local path to a cached model.

    Args:
        model_size: Model size (e.g., "7B")
        cache_dir: HuggingFace cache directory

    Returns:
        Path to model snapshot directory, or None if not found
    """
    model_dir = Path(cache_dir) / f"models--Qwen--Qwen2.5-{model_size}-Instruct"
    if not model_dir.exists():
        return None

    # Try known snapshot first
    if model_size in MODEL_SNAPSHOTS:
        snapshot_path = model_dir / "snapshots" / MODEL_SNAPSHOTS[model_size]
        if snapshot_path.exists():
            return str(snapshot_path)

    # Otherwise find any snapshot
    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            return str(snapshots[0])

    return None


class QwenProvider(LLMProvider):
    """Qwen2.5-Instruct provider using local HuggingFace transformers.

    Runs the model locally on GPU. Recommended for batch generation
    to avoid API costs/limits.

    Example:
        >>> provider = QwenProvider(model_size="7B")
        >>> response = provider.generate("Generate 5 random numbers")
    """

    def __init__(
        self,
        model_size: str = "7B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        cache_dir: str | None = None,
        offline: bool = True,
    ):
        """Initialize Qwen provider.

        Args:
            model_size: Model size - "0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"
            device: Device to run on ("cuda", "cpu", "auto")
            torch_dtype: Torch dtype ("bfloat16", "float16", "float32")
            cache_dir: HuggingFace cache directory
            offline: Run in offline mode (no network access)
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers package not found. "
                "Install with: pip install transformers torch"
            )

        # Set offline mode if requested
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Use default cache dir if not specified
        if cache_dir is None:
            cache_dir = os.environ.get("HF_HUB_CACHE", DEFAULT_HF_CACHE)

        self.model_size = model_size
        self.device = device
        self.cache_dir = cache_dir

        # Try to find local model path
        local_path = get_local_model_path(model_size, cache_dir)
        if local_path and offline:
            # Use local path directly to avoid any network calls
            self.model_id = local_path
        else:
            self.model_id = f"Qwen/Qwen2.5-{model_size}-Instruct"
            if offline and not local_path:
                raise ValueError(
                    f"Model Qwen2.5-{model_size}-Instruct not found in cache. "
                    f"Please download it first with: huggingface-cli download Qwen/Qwen2.5-{model_size}-Instruct"
                )

        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            local_files_only=offline,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=device if device == "auto" else {"": device},
            local_files_only=offline,
            trust_remote_code=True,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response from Qwen.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            The model's response as a string
        """
        import random

        import torch

        messages = [
            {
                "role": "system",
                "content": "You are a data generation assistant. Generate realistic, diverse synthetic tabular data. Each sample should have different values - never repeat the same numbers. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Set random seed for this generation to ensure diversity across calls
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Generate with sampling parameters for diversity
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.8),  # Ensure at least 0.8 for diversity
            do_sample=True,
            top_p=0.95,  # Nucleus sampling - allow more options
            top_k=100,  # Top-k sampling - more variety
            repetition_penalty=1.1,  # Discourage repetition
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Extract only the new tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

    def get_model_name(self) -> str:
        """Return the model identifier."""
        return f"qwen:{self.model_size}"
