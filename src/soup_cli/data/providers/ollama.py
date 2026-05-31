"""Ollama provider for synthetic data generation.

Connects to a local Ollama instance via its OpenAI-compatible API.
"""

import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def detect_ollama(base_url: str = DEFAULT_OLLAMA_BASE) -> Optional[str]:
    """Check if Ollama is running at the given URL.

    Returns the Ollama version string if detected, None otherwise.
    """
    try:
        import httpx
    except ImportError:
        return None

    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            # Ollama responded — try to get version
            try:
                ver_response = httpx.get(f"{base_url}/api/version", timeout=5.0)
                if ver_response.status_code == 200:
                    return ver_response.json().get("version", "unknown")
            except (httpx.HTTPError, KeyError, ValueError):
                logger.debug("Ollama version check failed", exc_info=True)
            return "unknown"
    except (httpx.HTTPError, OSError):
        logger.debug("Ollama not reachable at %s", base_url, exc_info=True)
    return None


def validate_ollama_url(base_url: str) -> None:
    """Validate that the Ollama URL is localhost-only (SSRF protection).

    Raises ValueError if the URL is not a local address.
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Ollama URL must use HTTP or HTTPS scheme (got {parsed.scheme}://)"
        )
    local_hosts = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
    if parsed.hostname not in local_hosts:
        raise ValueError(
            f"Ollama URL must be localhost (got {parsed.hostname}). "
            "Remote Ollama instances are not supported for security reasons."
        )


def generate_ollama(
    prompt: str,
    count: int,
    fmt: str,
    model_name: str,
    base_url: str,
    temperature: float,
    generation_prompt: str,
) -> list[dict]:
    """Generate examples using Ollama via its OpenAI-compatible API.

    Args:
        prompt: User-provided topic/instructions.
        count: Number of examples to generate.
        fmt: Output format (alpaca/sharegpt/chatml).
        model_name: Ollama model name (e.g. llama3.1).
        base_url: Ollama base URL.
        temperature: Sampling temperature.
        generation_prompt: Pre-built generation prompt.

    Returns:
        List of generated example dicts.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for Ollama generation. Install: pip install httpx"
        )

    validate_ollama_url(base_url)

    api_url = f"{base_url}/v1/chat/completions"

    response = httpx.post(
        api_url,
        headers={"Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": generation_prompt},
                {"role": "user", "content": f"Generate {count} training examples now."},
            ],
            "temperature": temperature,
            "max_tokens": 4096,
        },
        timeout=300.0,
    )

    if response.status_code != 200:
        logger.debug("Ollama error response: %s", response.text)
        raise ValueError(
            f"Ollama returned {response.status_code}. "
            "Check that Ollama is running and the model is pulled."
        )

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected Ollama response format: {exc}") from exc

    from soup_cli.data.providers._utils import parse_json_array

    return parse_json_array(content)
