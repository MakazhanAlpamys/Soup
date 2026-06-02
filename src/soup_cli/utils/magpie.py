"""v0.69.0 Part C — `soup data gen magpie` synthetic generator.

The Magpie technique (Xu et al. 2024) feeds an aligned chat-tuned model just
the chat-template prefix (system + user-turn header tokens) and lets the model
generate the user turn itself. The harvested user turns are then passed back to
the same model in a normal completion call to harvest the assistant response.
This is a clever way to produce high-volume SFT data without paying for human
prompts.

This module ships the schema + validators + dry-run planner. Live generation
(invoking v0.20.0 Ollama / Anthropic / vLLM providers + running the quality
filter from v0.47.0 educational + toxicity scorers) is deferred to v0.69.1
under the project-wide stub-then-live cadence (mirrors v0.50.0 / v0.61.0 /
v0.62.0 / v0.68.0).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence

_LOG = logging.getLogger("soup.magpie")

# Closed allowlist of providers — reuses v0.20.0 synth-data backends.
_SUPPORTED_MAGPIE_PROVIDERS = ("ollama", "anthropic", "vllm")
SUPPORTED_MAGPIE_PROVIDERS: frozenset = frozenset(_SUPPORTED_MAGPIE_PROVIDERS)

_MAX_BASE_MODEL_LEN = 512
_MAX_TARGET_ROWS = 1_000_000
_MAX_PROVIDER_LEN = 32
_MAX_OUTPUT_PATH_LEN = 4096
_MAX_MODEL_LEN = 512
# DoS cap on a provider's HTTP response body — a hostile/buggy local provider
# could otherwise return an arbitrarily large body that resp.json() fully
# materialises into memory (security review MEDIUM).
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# Loopback-only defaults for the raw-completion backends. Operators wanting a
# remote Ollama / vLLM must pass an explicit ``base_url`` (still SSRF-validated).
_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_VLLM_DEFAULT_URL = "http://localhost:8000"

GenerateFn = Callable[[str], str]


@dataclass(frozen=True)
class MagpieConfig:
    """Frozen plan for one ``soup data gen-magpie`` invocation."""

    base_model: str
    provider: str
    target_rows: int
    quality_filter: bool

    def __post_init__(self) -> None:
        # Re-validate so direct construction can't smuggle bad fields past
        # the validators (mirrors v0.67.0 vector_bank policy).
        validate_base_model(self.base_model)
        validate_magpie_provider(self.provider)
        validate_target_rows(self.target_rows)
        if not isinstance(self.quality_filter, bool):
            raise TypeError(
                "MagpieConfig.quality_filter must be bool, "
                f"got {type(self.quality_filter).__name__}"
            )


# -----------------------------------------------------------------------------
# Validators
# -----------------------------------------------------------------------------


def validate_magpie_provider(provider: object) -> str:
    """Return the canonical lower-case provider id."""
    if isinstance(provider, bool) or not isinstance(provider, str):
        raise TypeError(
            f"magpie provider must be str, got {type(provider).__name__}"
        )
    if not provider:
        raise ValueError("magpie provider must be non-empty")
    if "\x00" in provider:
        raise ValueError("magpie provider must not contain null bytes")
    if len(provider) > _MAX_PROVIDER_LEN:
        raise ValueError(
            f"magpie provider must be <= {_MAX_PROVIDER_LEN} chars"
        )
    canonical = provider.strip().lower()
    if canonical not in SUPPORTED_MAGPIE_PROVIDERS:
        raise ValueError(
            f"unknown magpie provider: {provider!r}. "
            f"supported: {sorted(SUPPORTED_MAGPIE_PROVIDERS)}"
        )
    return canonical


def validate_target_rows(target: object) -> int:
    """Validate ``target_rows`` ∈ [1, 1_000_000]; bool-rejected."""
    if isinstance(target, bool):
        raise TypeError("target_rows must be int, not bool")
    if not isinstance(target, int):
        raise TypeError("target_rows must be an integer")
    if target < 1:
        raise ValueError("target_rows must be >= 1")
    if target > _MAX_TARGET_ROWS:
        raise ValueError(f"target_rows must be <= {_MAX_TARGET_ROWS}")
    return target


def validate_base_model(name: object) -> str:
    """Validate the base-model identifier."""
    if isinstance(name, bool) or not isinstance(name, str):
        raise TypeError(
            f"base_model must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("base_model must be non-empty")
    if "\x00" in name:
        raise ValueError("base_model must not contain null bytes")
    if len(name) > _MAX_BASE_MODEL_LEN:
        raise ValueError(
            f"base_model must be <= {_MAX_BASE_MODEL_LEN} chars"
        )
    return name


# -----------------------------------------------------------------------------
# Plan builder
# -----------------------------------------------------------------------------


def build_magpie_config(
    *,
    base: str,
    provider: str,
    target: int,
    quality_filter: bool = True,
) -> MagpieConfig:
    """Convenience factory — every input passes through the validators."""
    if not isinstance(quality_filter, bool):
        raise TypeError(
            f"quality_filter must be bool, got {type(quality_filter).__name__}"
        )
    return MagpieConfig(
        base_model=validate_base_model(base),
        provider=validate_magpie_provider(provider),
        target_rows=validate_target_rows(target),
        quality_filter=quality_filter,
    )


# -----------------------------------------------------------------------------
# Chat-template prefix harvest (v0.71.6 #232)
# -----------------------------------------------------------------------------
#
# The Magpie technique feeds an aligned model JUST the chat-template prefix
# (the user-turn opener tokens, nothing after) and lets the model generate the
# user instruction itself. We then concatenate that instruction + the
# assistant-turn opener and complete again to harvest the response. The prefix
# string varies per model family — a small registry keyed by name substring,
# defaulting to ChatML (Qwen / many fine-tunes).

_CHATML_PREFIX = "<|im_start|>user\n"
_CHATML_ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n"
_CHATML_STOP = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")

_LLAMA3_PREFIX = "<|start_header_id|>user<|end_header_id|>\n\n"
_LLAMA3_ASSISTANT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
_LLAMA3_STOP = ("<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>")

_GEMMA_PREFIX = "<start_of_turn>user\n"
_GEMMA_ASSISTANT = "<end_of_turn>\n<start_of_turn>model\n"
_GEMMA_STOP = ("<end_of_turn>", "<eos>")

_MISTRAL_PREFIX = "[INST] "
_MISTRAL_ASSISTANT = " [/INST] "
_MISTRAL_STOP = ("</s>", "[INST]", "[/INST]")


def _family(base_model: str) -> str:
    name = base_model.lower()
    if "llama-3" in name or "llama3" in name:
        return "llama3"
    if "gemma" in name:
        return "gemma"
    if "mistral" in name or "mixtral" in name or "ministral" in name:
        return "mistral"
    # Qwen / Phi / SmolLM / generic fine-tunes default to ChatML.
    return "chatml"


def magpie_prefix_for(base_model: str) -> str:
    """Return the chat-template user-turn opener for ``base_model``'s family."""
    validate_base_model(base_model)
    return {
        "llama3": _LLAMA3_PREFIX,
        "gemma": _GEMMA_PREFIX,
        "mistral": _MISTRAL_PREFIX,
        "chatml": _CHATML_PREFIX,
    }[_family(base_model)]


def magpie_assistant_opener(base_model: str) -> str:
    """Return the assistant-turn opener for ``base_model``'s family."""
    validate_base_model(base_model)
    return {
        "llama3": _LLAMA3_ASSISTANT,
        "gemma": _GEMMA_ASSISTANT,
        "mistral": _MISTRAL_ASSISTANT,
        "chatml": _CHATML_ASSISTANT,
    }[_family(base_model)]


def _stop_markers_for(base_model: str) -> tuple[str, ...]:
    return {
        "llama3": _LLAMA3_STOP,
        "gemma": _GEMMA_STOP,
        "mistral": _MISTRAL_STOP,
        "chatml": _CHATML_STOP,
    }[_family(base_model)]


def _clean_generation(raw: object, stop_markers: Sequence[str]) -> str:
    """Truncate a raw completion at the earliest turn-boundary marker + trim."""
    if not isinstance(raw, str):
        return ""
    cut = len(raw)
    for marker in stop_markers:
        idx = raw.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return raw[:cut].strip()


def harvest_instruction(
    generate_fn: GenerateFn,
    prefix: str,
    stop_markers: Sequence[str],
) -> str:
    """Magpie step 1: feed the prefix, harvest the model-generated user turn."""
    return _clean_generation(generate_fn(prefix), stop_markers)


def harvest_response(
    generate_fn: GenerateFn,
    prefix: str,
    instruction: str,
    assistant_opener: str,
    stop_markers: Sequence[str],
) -> str:
    """Magpie step 2: feed prefix+instruction+assistant-opener, harvest reply."""
    prompt = f"{prefix}{instruction}{assistant_opener}"
    return _clean_generation(generate_fn(prompt), stop_markers)


# -----------------------------------------------------------------------------
# Quality filter (reuses v0.47.0 educational + toxicity scorers)
# -----------------------------------------------------------------------------

_QUALITY_MAX_TOXICITY = 0.5
_QUALITY_MIN_EDUCATIONAL = 0.1


def default_quality_fn(instruction: str, response: str) -> bool:
    """Keep a row if it is non-empty, low-toxicity and minimally educational.

    Reuses the v0.47.0 keyword-baseline scorers. A real Llama-Guard / FineWeb
    classifier ships behind ``[data-pro]`` (tracked separately).
    """
    if not isinstance(instruction, str) or not instruction.strip():
        return False
    if not isinstance(response, str) or not response.strip():
        return False
    from soup_cli.utils.data_score import score_educational_value, score_toxicity

    combined = f"{instruction}\n{response}"
    if score_toxicity(combined) > _QUALITY_MAX_TOXICITY:
        return False
    if score_educational_value(combined) < _QUALITY_MIN_EDUCATIONAL:
        return False
    return True


# -----------------------------------------------------------------------------
# Provider raw-completion factory
# -----------------------------------------------------------------------------


def make_magpie_generate_fn(
    provider: str,
    *,
    model: str,
    base_url: Optional[str] = None,
    temperature: float = 1.0,
    timeout_seconds: float = 120.0,
    max_tokens: int = 512,
) -> GenerateFn:
    """Build a ``generate(prompt) -> str`` RAW-completion callable.

    Magpie feeds raw chat-template prefixes (not complete messages), so it
    needs a base/instruct model exposed via a *raw* completion endpoint:

    - ``ollama`` → ``POST <base_url>/api/generate`` with ``raw: true``.
    - ``vllm``   → ``POST <base_url>/v1/completions``.
    - ``anthropic`` → rejected: the Messages API has no base-model raw
      completion endpoint, so prefix-harvest Magpie is not possible. Use
      ``ollama`` or ``vllm``.

    SSRF carry-overs: Ollama / vLLM URLs are validated loopback-only unless an
    operator explicitly overrides ``base_url`` (still scheme-validated).
    """
    canonical = validate_magpie_provider(provider)
    if not isinstance(model, str) or not model or "\x00" in model:
        raise ValueError("model must be a non-empty NUL-free string")
    if len(model) > _MAX_MODEL_LEN:
        raise ValueError(f"model must be <= {_MAX_MODEL_LEN} chars")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a number")
    if temperature < 0 or temperature > 2:
        raise ValueError("temperature must be in [0, 2]")
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ):
        raise TypeError("timeout_seconds must be a number")
    if timeout_seconds <= 0 or timeout_seconds > 600:
        raise ValueError("timeout_seconds must be in (0, 600]")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("max_tokens must be an int")
    if max_tokens < 1 or max_tokens > 16384:
        raise ValueError("max_tokens must be in [1, 16384]")

    if canonical == "anthropic":
        raise ValueError(
            "Anthropic has no base-model raw-completion endpoint; Magpie "
            "prefix-harvest needs ollama or vllm raw completion."
        )

    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for live Magpie providers. Run: pip install httpx"
        ) from exc

    if canonical == "ollama":
        from soup_cli.data.providers.ollama import validate_ollama_url

        url = base_url or _OLLAMA_DEFAULT_URL
        validate_ollama_url(url)
        api_url = f"{url}/api/generate"

        def _ollama_generate(prompt: str) -> str:
            if not isinstance(prompt, str):
                return ""
            try:
                resp = httpx.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": model,
                        "prompt": prompt,
                        "raw": True,
                        "stream": False,
                        "options": {
                            "temperature": float(temperature),
                            "num_predict": max_tokens,
                        },
                    },
                    timeout=timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 — httpx error variety
                _LOG.debug("ollama magpie HTTP error: %s", exc)
                return ""
            if resp.status_code != 200:
                _LOG.debug("ollama magpie status=%d", resp.status_code)
                return ""
            if len(resp.content) > _MAX_RESPONSE_BYTES:
                _LOG.debug("ollama magpie response too large; dropping")
                return ""
            try:
                text = resp.json()["response"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                _LOG.debug("ollama magpie parse error: %s", exc)
                return ""
            return text if isinstance(text, str) else ""

        return _ollama_generate

    # vLLM raw completion.
    from soup_cli.data.providers.vllm import validate_vllm_url

    url = base_url or _VLLM_DEFAULT_URL
    validate_vllm_url(url)
    api_url = f"{url}/v1/completions"

    def _vllm_generate(prompt: str) -> str:
        if not isinstance(prompt, str):
            return ""
        try:
            resp = httpx.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": float(temperature),
                    "max_tokens": max_tokens,
                },
                timeout=timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("vllm magpie HTTP error: %s", exc)
            return ""
        if resp.status_code != 200:
            _LOG.debug("vllm magpie status=%d", resp.status_code)
            return ""
        if len(resp.content) > _MAX_RESPONSE_BYTES:
            _LOG.debug("vllm magpie response too large; dropping")
            return ""
        try:
            text = resp.json()["choices"][0]["text"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            _LOG.debug("vllm magpie parse error: %s", exc)
            return ""
        return text if isinstance(text, str) else ""

    return _vllm_generate


# -----------------------------------------------------------------------------
# Result + live runner (v0.71.6 #232)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MagpieResult:
    """Outcome of one live Magpie generation run."""

    rows_kept: int
    rows_filtered: int
    duplicates: int
    attempts: int
    output_path: str


def _check_magpie_output(path: object) -> str:
    """Validate the output path early (before any provider call) — fail fast.

    Shape checks here; the actual write delegates cwd-containment + symlink
    rejection + atomic replace to the shared ``atomic_write_bytes`` helper.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("output_path must be a non-empty string")
    if "\x00" in path:
        raise ValueError("output_path must not contain null bytes")
    if len(path) > _MAX_OUTPUT_PATH_LEN:
        raise ValueError(f"output_path exceeds {_MAX_OUTPUT_PATH_LEN} chars")
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(path, "output_path")
    return path


def _atomic_write_jsonl(path: str, rows: Sequence[Mapping[str, Any]]) -> str:
    from soup_cli.utils.paths import atomic_write_bytes

    payload = (
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        + ("\n" if rows else "")
    ).encode("utf-8")
    return atomic_write_bytes(payload, path, prefix=".magpie-", field="output_path")


def run_magpie(
    config: MagpieConfig,
    *,
    output_path: str,
    generate_fn: Optional[GenerateFn] = None,
    quality_fn: Optional[Callable[[str, str], bool]] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 1.0,
    timeout_seconds: float = 120.0,
    max_attempts: Optional[int] = None,
    dedup: bool = True,
) -> MagpieResult:
    """Execute the Magpie generation loop (v0.71.6 #232).

    Feeds the chat-template prefix to ``config.base_model`` via ``config.provider``
    (or the injected ``generate_fn``), harvests instruction + response per the
    Magpie technique, optionally filters with the v0.47.0 quality scorers, and
    writes ``{messages:[user, assistant]}`` rows to ``output_path`` (atomic,
    cwd-contained, symlink-rejected). Returns a :class:`MagpieResult`.

    ``generate_fn`` is injectable for tests + advanced callers; when ``None`` it
    is built from ``config.provider`` via :func:`make_magpie_generate_fn`.
    """
    if not isinstance(config, MagpieConfig):
        raise TypeError("config must be a MagpieConfig")
    target = _check_magpie_output(output_path)

    if generate_fn is None:
        generate_fn = make_magpie_generate_fn(
            config.provider,
            model=config.base_model,
            base_url=base_url,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )
    if not callable(generate_fn):
        raise TypeError("generate_fn must be callable")
    if config.quality_filter and quality_fn is None:
        quality_fn = default_quality_fn

    prefix = magpie_prefix_for(config.base_model)
    assistant_opener = magpie_assistant_opener(config.base_model)
    stop_markers = _stop_markers_for(config.base_model)

    if max_attempts is None:
        # Bound the loop so an unreachable provider / degenerate model can't
        # spin forever — generous multiple of the target.
        max_attempts = max(config.target_rows * 8, 32)

    rows: List[dict] = []
    seen: set[str] = set()
    filtered = 0
    duplicates = 0
    attempts = 0
    while len(rows) < config.target_rows and attempts < max_attempts:
        attempts += 1
        instruction = harvest_instruction(generate_fn, prefix, stop_markers)
        if not instruction:
            continue
        if dedup and instruction in seen:
            duplicates += 1
            continue
        # Mark as seen BEFORE generating the response so a filtered / empty
        # instruction isn't re-generated + re-paid for every loop (code-review
        # MEDIUM fix — it's counted as a duplicate on its next appearance).
        if dedup:
            seen.add(instruction)
        response = harvest_response(
            generate_fn, prefix, instruction, assistant_opener, stop_markers
        )
        if not response:
            continue
        if config.quality_filter and quality_fn is not None:
            try:
                keep = quality_fn(instruction, response)
            except Exception as exc:  # noqa: BLE001 — quality fn must not crash run
                # Surface loudly (not DEBUG) so a crashing filter isn't a silent
                # no-op; keep the row so the run still completes (code-review M3).
                _LOG.warning("magpie quality_fn raised (keeping row): %s", exc)
                keep = True
            if not keep:
                filtered += 1
                continue
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ]
            }
        )

    written = _atomic_write_jsonl(target, rows)
    return MagpieResult(
        rows_kept=len(rows),
        rows_filtered=filtered,
        duplicates=duplicates,
        attempts=attempts,
        output_path=written,
    )


__all__ = [
    "GenerateFn",
    "MagpieConfig",
    "MagpieResult",
    "SUPPORTED_MAGPIE_PROVIDERS",
    "build_magpie_config",
    "default_quality_fn",
    "harvest_instruction",
    "harvest_response",
    "magpie_assistant_opener",
    "magpie_prefix_for",
    "make_magpie_generate_fn",
    "run_magpie",
    "validate_base_model",
    "validate_magpie_provider",
    "validate_target_rows",
]
