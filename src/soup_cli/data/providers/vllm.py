"""vLLM provider for synthetic data generation.

Connects to a local vLLM server via its OpenAI-compatible API.
Supports batch mode with concurrent requests.
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_VLLM_BASE = "http://localhost:8000"


def validate_vllm_url(base_url: str) -> None:
    """Validate the vLLM server URL (SSRF protection).

    Reuses the same validation as the server provider:
    - Scheme must be http or https
    - HTTP is only allowed for localhost

    Raises ValueError if validation fails.
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"vLLM URL must use HTTP or HTTPS scheme (got {parsed.scheme}://)"
        )
    local_hosts = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
    is_local = parsed.hostname in local_hosts
    if not is_local and parsed.scheme != "https":
        raise ValueError(
            f"vLLM URL must use HTTPS for remote servers (got {parsed.scheme}://). "
            "HTTP is only allowed for localhost."
        )


def generate_vllm(
    prompt: str,
    count: int,
    fmt: str,
    model_name: str,
    base_url: str,
    temperature: float,
    generation_prompt: str,
    batch_size: int = 5,
) -> list[dict]:
    """Generate examples using vLLM server.

    Args:
        prompt: User-provided topic/instructions.
        count: Number of examples to generate.
        fmt: Output format (alpaca/sharegpt/chatml).
        model_name: Model name on vLLM server.
        base_url: vLLM server base URL.
        temperature: Sampling temperature.
        generation_prompt: Pre-built generation prompt.
        batch_size: Number of concurrent requests.

    Returns:
        List of generated example dicts.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for vLLM generation. Install: pip install httpx"
        )

    validate_vllm_url(base_url)

    # Ensure URL ends with /v1
    api_base = base_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = api_base + "/v1"

    response = httpx.post(
        f"{api_base}/chat/completions",
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
        logger.debug("vLLM error response: %s", response.text)
        raise ValueError(
            f"vLLM server returned {response.status_code}. "
            "Check that the server is running and the model is loaded."
        )

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected vLLM response format: {exc}") from exc

    from soup_cli.data.providers._utils import parse_json_array

    return parse_json_array(content)
