"""LongLoRA S² shifted-sparse attention support (v0.49.0 Part C).

Schema-level helpers only — the live forward-override patch that mirrors
LlamaFactory ``model/model_utils/longlora.py`` is deferred to v0.49.1
(stub-then-live pattern, mirrors v0.27.0 MII / v0.37.0 multipack / v0.41.0
LLaMA Pro / v0.46.0 Quant-Lobotomy auto-measure).

LongLoRA's S² (shifted-sparse) attention groups tokens into local windows and
shifts half the heads by ``window_size // 2`` so adjacent groups can exchange
information. The implementation in LlamaFactory replaces
``LlamaAttention.forward`` with a custom kernel that materialises this pattern
via two cheap masked attentions; FlashAttention v3's custom-mask API and
Ring-FlashAttention's sequence-parallel attention are not yet compatible.

Architecture allowlist: Llama 3.x only initially (per the upstream
implementation). Extend per request.
"""

from __future__ import annotations

import re
from typing import Any

# Word-boundary regex matching the v0.39.0 ``is_gemma4_model`` / v0.44.0
# ``is_llama4_model`` policy — substring ``"llama"`` inside an unrelated identifier
# (e.g. ``my-llama-style-finetune``) must NOT match silently. We accept ``llama``
# with optional version digit suffix and an optional ``code-`` prefix for the
# ``codellama`` family.
_LLAMA_REGEX = re.compile(r"(?:^|[^a-z0-9])(?:code)?-?llama(?:-?\d+(?:\.\d+)?)?(?:[^a-z0-9]|$)")

# v0.53.4 #120 — LongLoRA architecture allowlist expansion. Word-boundary regex
# per the v0.39.0 ``is_gemma4_model`` / v0.44.0 ``is_llama4_model`` policy —
# a substring inside an unrelated identifier (e.g. ``my-mistralish-finetune``)
# must NOT match silently. Each regex accepts the family name with an optional
# version digit suffix.
_MISTRAL_REGEX = re.compile(
    r"(?:^|[^a-z0-9])mistral(?:-?\d+(?:\.\d+)?)?(?:[^a-z0-9]|$)"
)
_QWEN_REGEX = re.compile(r"(?:^|[^a-z0-9])qwen(?:-?\d+(?:\.\d+)?)?(?:[^a-z0-9]|$)")
_PHI_REGEX = re.compile(r"(?:^|[^a-z0-9])phi(?:-?\d+(?:\.\d+)?)?(?:[^a-z0-9]|$)")

_MAX_MODEL_NAME_LEN = 512


def _check_model_name(model_name: str) -> str | None:
    """Shared input guard for the family-detection helpers.

    Returns the lower-cased name if it should be searched, ``None`` if the
    helper should return ``False`` (oversized input). Raises ``TypeError`` /
    ``ValueError`` for non-string / null-byte inputs (matches v0.49.0
    ``is_llama_model`` policy).

    ``bool`` is rejected with ``TypeError`` before the ``isinstance(str)``
    check because ``bool`` is a subclass of ``int`` — but the project policy
    (matches ``is_known_vlm_base`` / ``validate_hub_name``) is to reject it
    explicitly so a misconfigured caller passing ``True``/``False`` gets a
    clean error rather than silently falling through.
    """
    if isinstance(model_name, bool):
        raise TypeError(
            f"model_name must be a string, got {type(model_name).__name__}"
        )
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")
    if "\x00" in model_name:
        raise ValueError("model_name must not contain null bytes")
    if len(model_name) > _MAX_MODEL_NAME_LEN:
        return None
    return model_name.lower()


def is_mistral_model(model_name: str) -> bool:
    """Return True if ``model_name`` belongs to the Mistral / Mixtral family.

    v0.53.4 #120 — LongLoRA architecture allowlist expansion.
    """
    lowered = _check_model_name(model_name)
    if lowered is None:
        return False
    return _MISTRAL_REGEX.search(lowered) is not None


def is_qwen_model(model_name: str) -> bool:
    """Return True if ``model_name`` belongs to the Qwen family.

    v0.53.4 #120 — LongLoRA architecture allowlist expansion.
    """
    lowered = _check_model_name(model_name)
    if lowered is None:
        return False
    return _QWEN_REGEX.search(lowered) is not None


def is_phi_model(model_name: str) -> bool:
    """Return True if ``model_name`` belongs to the Phi family.

    v0.53.4 #120 — LongLoRA architecture allowlist expansion.
    """
    lowered = _check_model_name(model_name)
    if lowered is None:
        return False
    return _PHI_REGEX.search(lowered) is not None


def is_supported_longlora_arch(model_name: object) -> bool:
    """Return True if ``model_name`` is in the LongLoRA allowlist.

    The v0.53.4 #120 allowlist covers Llama / CodeLlama (Llama 3.x heritage),
    Mistral, Qwen, and Phi. The S² forward override (deferred to v0.49.1)
    attaches per-arch; the schema gate uses this helper.

    Returns ``False`` (never raises) on non-string input — matches
    ``is_known_vlm_base`` / ``is_bitnet_model`` defensive-surface policy.
    Mixtral is intentionally NOT covered (regex matches the bare token
    ``mistral``, not the Mixtral MoE variant); add a dedicated helper when
    Mixtral attention upstream lands a stable forward signature.
    """
    if not isinstance(model_name, str):
        return False
    try:
        return (
            is_llama_model(model_name)
            or is_mistral_model(model_name)
            or is_qwen_model(model_name)
            or is_phi_model(model_name)
        )
    except (TypeError, ValueError):
        return False


def is_llama_model(model_name: str) -> bool:
    """Return True if ``model_name`` belongs to the Llama / CodeLlama family.

    Args:
        model_name: HuggingFace model id or local path.

    Raises:
        TypeError: If ``model_name`` is not a string.
        ValueError: If ``model_name`` contains a null byte.
    """
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")
    if "\x00" in model_name:
        raise ValueError("model_name must not contain null bytes")
    if len(model_name) > _MAX_MODEL_NAME_LEN:
        return False
    return _LLAMA_REGEX.search(model_name.lower()) is not None


def _truncate_for_message(value: str, *, limit: int = 64) -> str:
    """Truncate ``value`` for safe embedding into user-facing error messages.

    Mirrors the v0.53.3 ``validate_vision_grpo_compat`` security-review fix —
    keeps stderr / log output bounded when a caller passes a pathologically
    long base name.
    """
    if not isinstance(value, str):
        return repr(value)
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def validate_longlora_compat(
    *,
    model_name: str,
    task: str,
    backend: str,
    use_ring_attention: bool,
) -> None:
    """Raise ``ValueError`` if the LongLoRA prerequisites are not satisfied.

    LongLoRA requires:
      * ``task='sft'``  (preference / RL trainers have a different forward path)
      * ``backend='transformers'``  (Unsloth has its own attention; MLX has none)
      * Llama / Mistral / Qwen / Phi base model (v0.53.4 #120 allowlist)
      * ``use_ring_attention=False`` (mutually exclusive custom attention kernel)
      * No FlashAttention v3 build present (v0.53.4 #122 — incompatible custom-mask)
    """
    # v0.53.4 review fix — defensive guards on task / backend mirror the
    # v0.50.0 ``validate_long_context_grpo_compat`` policy. Null bytes in a
    # user-controlled YAML string would otherwise embed literally in the
    # error message and downstream log files.
    for label, val in (("task", task), ("backend", backend)):
        if isinstance(val, bool):
            raise ValueError(f"{label} must be a string, not bool")
        if not isinstance(val, str):
            raise ValueError(f"{label} must be a string (got {type(val).__name__})")
        if "\x00" in val:
            raise ValueError(f"{label} must not contain null bytes")
    if backend == "mlx":
        # Distinct error message per project policy (matches v0.34.0 review-fix
        # convention of naming the specific incompatible backend).
        raise ValueError(
            "LongLoRA is not supported on the mlx backend (the S^2 forward "
            "override is HF-Transformers specific). Use backend='transformers' "
            "or set use_longlora=false."
        )
    if backend != "transformers":
        raise ValueError(
            f"LongLoRA requires backend='transformers' (got backend={backend!r})."
        )
    if task != "sft":
        raise ValueError(
            f"LongLoRA is currently restricted to task='sft' (got task={task!r}). "
            "Multi-trainer expansion is planned for a future release."
        )
    if not is_supported_longlora_arch(model_name):
        safe_name = _truncate_for_message(model_name)
        raise ValueError(
            "LongLoRA architecture allowlist currently covers Llama / "
            "CodeLlama, Mistral, Qwen, and Phi families (got "
            f"base={safe_name!r}). Open a feature request to add another "
            "architecture."
        )
    if use_ring_attention:
        raise ValueError(
            "LongLoRA is incompatible with use_ring_attention (both rewrite the "
            "attention kernel — pick one)."
        )
    # v0.53.4 #122 — FA v3's native custom-mask kernel conflicts with the S^2
    # forward override; flag the combo loudly so users opt out of one or the
    # other instead of silently corrupting attention outputs.
    from soup_cli.utils.flash_attn import is_flash_attn_v3_available

    if is_flash_attn_v3_available():
        raise ValueError(
            "LongLoRA is incompatible with FlashAttention v3 (both rewrite the "
            "attention kernel — uninstall flash_attn>=3 or set "
            "use_longlora=false)."
        )


def apply_longlora_forward_override(model: Any) -> None:
    """Install the LongLoRA S^2 ``LlamaAttention.forward`` override.

    Deferred to v0.49.1 (matches the stub-then-live pattern used by v0.27.0
    MII / v0.37.0 multipack / v0.41.0 LLaMA Pro / v0.46.0 Quant-Lobotomy).
    The v0.49.0 release ships the schema gate (so ``soup train`` fails fast
    when LongLoRA is misconfigured) and the kernel math; the live forward
    monkeypatch + tests on the real model graph land in v0.49.1.
    """
    raise NotImplementedError(
        "LongLoRA S^2 forward override lands in v0.49.1 — the v0.49.0 release "
        "ships the schema gate (so misconfigured runs fail fast) and the math "
        "kernel only."
    )
