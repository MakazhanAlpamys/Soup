"""Anthropic (Claude) provider for synthetic data generation.

Uses raw httpx requests to avoid SDK dependency.
API key must be set via ANTHROPIC_API_KEY environment variable.
"""

import logging
import os

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-3-haiku-20240307"


def generate_anthropic(
    prompt: str,
    count: int,
    fmt: str,
    model_name: str,
    temperature: float,
    generation_prompt: str,
) -> list[dict]:
    """Generate examples using Anthropic Claude API.

    Args:
        prompt: User-provided topic/instructions.
        count: Number of examples to generate.
        fmt: Output format (alpaca/sharegpt/chatml).
        model_name: Anthropic model ID.
        temperature: Sampling temperature.
        generation_prompt: Pre-built generation prompt.

    Returns:
        List of generated example dicts.

    Raises:
        ValueError: If API key not found or API returns error.
        ImportError: If httpx is not installed.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for Anthropic generation. Install: pip install httpx"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
        )

    response = httpx.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "system": generation_prompt,
            "messages": [
                {"role": "user", "content": f"Generate {count} training examples now."},
            ],
            "temperature": temperature,
            "max_tokens": 4096,
        },
        timeout=120.0,
    )

    if response.status_code != 200:
        logger.debug("Anthropic error response: %s", response.text)
        raise ValueError(
            f"Anthropic API returned {response.status_code}. "
            "Check your API key and model name."
        )

    data = response.json()
    try:
        # Anthropic Messages API returns content as a list of blocks
        content_blocks = data["content"]
        content = "".join(
            block["text"] for block in content_blocks if block.get("type") == "text"
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected Anthropic response format: {exc}") from exc

    from soup_cli.data.providers._utils import parse_json_array

    return parse_json_array(content)
