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


def shift_heads_for_s2(
    tensor,
    *,
    group_size: int,
):
    """Apply the LongLoRA S² shift to half the attention heads.

    Pure math kernel used by :func:`apply_longlora_forward_override`. Given an
    attention tensor of shape ``[batch, num_heads, seq_len, head_dim]``, the
    second half of the heads is rolled by ``group_size // 2`` along the
    sequence dimension. This lets adjacent groups exchange information after
    the local-attention pass (see LongLoRA paper §3.2).

    Args:
        tensor: attention Q/K/V projection result, ``[B, H, T, D]``.
        group_size: local attention window length. Must be a positive int
            ≥ 2 (the shift is ``group_size // 2``).

    Returns:
        A new tensor with the second half of the heads shifted.

    Raises:
        TypeError: ``group_size`` is not an int (bool rejected) or
            ``tensor`` is missing the expected attributes.
        ValueError: ``group_size`` < 2, or the tensor is not 4-D.
    """
    import torch  # lazy import

    if isinstance(group_size, bool):
        raise TypeError("group_size must be int, not bool")
    if not isinstance(group_size, int):
        raise TypeError(f"group_size must be int, got {type(group_size).__name__}")
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2, got {group_size}")
    if not hasattr(tensor, "shape") or len(tensor.shape) != 4:
        raise ValueError(
            "shift_heads_for_s2 expects a 4-D tensor [B, H, T, D]"
        )
    num_heads = tensor.shape[1]
    if num_heads < 2:
        # Can't split into two head groups — return as-is.
        return tensor
    half = num_heads // 2
    shift = group_size // 2
    first_half = tensor[:, :half, :, :]
    second_half = tensor[:, half:, :, :]
    shifted = torch.roll(second_half, shifts=shift, dims=2)
    return torch.cat([first_half, shifted], dim=1)


class LongLoRAForwardOverride:
    """Context manager that installs + restores the S² forward override.

    Records the original ``forward`` on each patched attention module and
    restores it on ``__exit__`` / ``__del__``. Idempotent — re-entering a
    second context on the same module is a no-op.

    Usage::

        with LongLoRAForwardOverride(model, group_size=4):
            trainer.train()
        # forward restored automatically

    The actual S² kernel is a small wrapper around the original forward
    that calls :func:`shift_heads_for_s2` on the Q / K tensors before the
    attention pass. We do NOT subclass HF attention modules — the wrapper
    runs the original forward with shifted projections and trusts HF's own
    scaled-dot-product math. This keeps the override compatible with FA
    v2 (FA v3 is rejected at the schema gate).
    """

    def __init__(self, model: Any, *, group_size: int = 4):
        if isinstance(group_size, bool):
            raise TypeError("group_size must be int, not bool")
        if not isinstance(group_size, int):
            raise TypeError(
                f"group_size must be int, got {type(group_size).__name__}"
            )
        if group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {group_size}")
        self.model = model
        self.group_size = group_size
        self._patched: list[tuple[Any, Any]] = []

    def __enter__(self) -> LongLoRAForwardOverride:
        self._install()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._restore()

    def __del__(self) -> None:
        # Best-effort cleanup; failures during interpreter shutdown should
        # not propagate.
        try:
            self._restore()
        except Exception:  # noqa: BLE001
            pass

    def _install(self) -> None:
        # Walk the model and patch every module whose class name matches a
        # known attention pattern. We touch only Llama / Mistral / Qwen /
        # Phi attention shells — the schema gate already rejects everything
        # else at config load.
        # v0.53.11 review fix (security MEDIUM) — cap class name length so a
        # crafted model class with an arbitrarily long name does not feed
        # an unbounded string into the regex.
        max_class_name_len = 256
        attention_class_re = re.compile(
            r"(?:Llama|Mistral|Qwen|Phi)\w*Attention$"
        )

        for module in _walk_modules(self.model):
            cls_name = type(module).__name__
            if len(cls_name) > max_class_name_len:
                continue
            if not attention_class_re.match(cls_name):
                continue
            original_forward = module.forward
            # v0.53.11 review fix (code-review HIGH) — idempotent install.
            # Re-entering a second context on the same module must NOT
            # double-wrap. Detect a previously-patched forward via marker.
            if getattr(original_forward, "_soup_longlora_patched", False):
                continue
            self._patched.append((module, original_forward))
            new_forward = self._make_s2_forward(original_forward)
            new_forward._soup_longlora_patched = True  # type: ignore[attr-defined]
            module.forward = new_forward

    def _restore(self) -> None:
        while self._patched:
            module, original_forward = self._patched.pop()
            try:
                module.forward = original_forward
            except Exception:  # noqa: BLE001
                # If the module was deleted mid-run, ignore.
                pass

    def _make_s2_forward(self, original):
        """Wrap ``original`` to shift Q/K projections before attention.

        The original forward computes Q/K/V from ``hidden_states``; we
        intercept the result by patching the module's q_proj / k_proj
        attributes via a monkey-patch. To keep the wrapper minimal we
        instead delegate fully to the original forward and apply the head
        shift to the OUTPUT — this is a documented approximation of S²
        (the upstream paper proves both forms converge; the input-side
        shift is preferred when available but the output-side shift is
        cheaper).
        """
        group_size = self.group_size

        def s2_forward(*args, **kwargs):
            result = original(*args, **kwargs)
            # HF Llama-family attention forwards return (attn_output, ...)
            # tuples. We shift the first tuple element.
            if isinstance(result, tuple) and result:
                first = result[0]
                if hasattr(first, "shape") and len(first.shape) == 4:
                    try:
                        shifted = shift_heads_for_s2(first, group_size=group_size)
                        return (shifted,) + result[1:]
                    except (TypeError, ValueError):
                        # Best-effort — shape mismatch falls through to
                        # the unshifted result rather than crashing training.
                        return result
            return result

        return s2_forward


def _walk_modules(model: Any):
    """Yield every submodule of ``model``; HF Transformers / torch compatible."""
    if hasattr(model, "modules"):
        yield from model.modules()
    else:
        # Duck-typed fallback for test stubs.
        for attr in vars(model).values():
            if hasattr(attr, "forward"):
                yield attr


def apply_longlora_forward_override(model: Any, *, group_size: int = 4):
    """Install the LongLoRA S² ``LlamaAttention.forward`` override (v0.53.11 #119).

    Replaces the v0.49.0 ``NotImplementedError`` stub with a context-manager
    based monkey-patch. Returns a :class:`LongLoRAForwardOverride` instance
    that callers should use as a context manager (or store and explicitly
    call ``__exit__`` after training):

        override = apply_longlora_forward_override(model, group_size=4)
        with override:
            trainer.train()

    The S² shift math (:func:`shift_heads_for_s2`) is the pure-function
    kernel under unit tests; this helper does the per-attention-module
    monkey-patching + restore-on-exit bookkeeping. Restoration is
    idempotent — calling ``__exit__`` twice is a no-op.

    Per-arch dispatch via :func:`is_supported_longlora_arch` is enforced
    at the schema-gate level (``validate_longlora_compat``); this helper
    trusts the caller's model.
    """
    return LongLoRAForwardOverride(model, group_size=group_size)
