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

_MAX_MODEL_NAME_LEN = 512


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
      * Llama-family base model  (initial architecture allowlist)
      * ``use_ring_attention=False`` (mutually exclusive custom attention kernel)
    """
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
    if not is_llama_model(model_name):
        raise ValueError(
            "LongLoRA architecture allowlist currently covers only the Llama "
            f"family (got base={model_name!r}). Open a feature request to add "
            "another architecture."
        )
    if use_ring_attention:
        raise ValueError(
            "LongLoRA is incompatible with use_ring_attention (both rewrite the "
            "attention kernel — pick one)."
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
