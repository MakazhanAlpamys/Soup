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
# v0.71.16 #147 — Mixtral needs its OWN regex: the bare ``mistral`` token does
# not appear in ``mixtral`` (m-i-x vs m-i-s), so ``is_mistral_model`` excludes
# the MoE variant. The Mixtral attention is structurally identical to Mistral's
# (separate q/k/v projections, GQA) — only the MLP is a sparse MoE — so the S²
# forward override reuses the separate-QKV projection-shift path.
_MIXTRAL_REGEX = re.compile(
    r"(?:^|[^a-z0-9])mixtral(?:-?\d+(?:\.\d+)?)?(?:[^a-z0-9]|$)"
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
    """Return True if ``model_name`` belongs to the dense Mistral family.

    v0.53.4 #120 — LongLoRA architecture allowlist expansion. NOTE: this
    deliberately EXCLUDES the Mixtral MoE variant (``mixtral`` does not contain
    the ``mistral`` token) — use :func:`is_mixtral_model` for Mixtral
    (v0.71.16 #147).
    """
    lowered = _check_model_name(model_name)
    if lowered is None:
        return False
    return _MISTRAL_REGEX.search(lowered) is not None


def is_mixtral_model(model_name: str) -> bool:
    """Return True if ``model_name`` belongs to the Mixtral MoE family.

    v0.71.16 #147 — dedicated detector. ``is_mistral_model`` deliberately
    excludes Mixtral (different token), so the LongLoRA allowlist needs this
    helper to cover Mixtral-8x7B / Mixtral-8x22B. The Mixtral attention is the
    standard separate-QKV (GQA) shell — the MoE lives in the MLP — so the S²
    forward override needs no special attention handling, only the family +
    attention-class-name allowlist entries.
    """
    lowered = _check_model_name(model_name)
    if lowered is None:
        return False
    return _MIXTRAL_REGEX.search(lowered) is not None


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

    The allowlist covers Llama / CodeLlama (Llama 3.x heritage), Mistral,
    Mixtral (v0.71.16 #147), Qwen, and Phi. The S² forward override attaches
    per-arch; the schema gate uses this helper.

    Returns ``False`` (never raises) on non-string input — matches
    ``is_known_vlm_base`` / ``is_bitnet_model`` defensive-surface policy.
    """
    if not isinstance(model_name, str):
        return False
    try:
        return (
            is_llama_model(model_name)
            or is_mistral_model(model_name)
            or is_mixtral_model(model_name)
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
      * Llama / Mistral / Mixtral / Qwen / Phi base model (v0.71.16 #147 allowlist)
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
            "CodeLlama, Mistral, Mixtral, Qwen, and Phi families (got "
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


# v0.71.12 #158 — per-arch projection dispatch. The S² shift is applied to
# the q_proj / k_proj OUTPUTS (before the attention dot-product), NOT to the
# attention output. Architectures with SEPARATE q/k/v projections (Llama,
# Mistral, Qwen2 — GQA handled by deriving the head count per-projection from
# its output dim) patch q_proj + k_proj independently. Phi-3 fuses Q/K/V into a
# single ``qkv_proj`` and needs the slice split before shifting.
_SEPARATE_QKV_FAMILIES: tuple[str, ...] = ("Llama", "Mistral", "Mixtral", "Qwen")
_FUSED_QKV_FAMILIES: tuple[str, ...] = ("Phi",)


def _resolve_head_dim(attn: Any) -> int | None:
    """Best-effort head_dim for an attention module (None if undeterminable)."""
    hd = getattr(attn, "head_dim", None)
    if isinstance(hd, int) and not isinstance(hd, bool) and hd > 0:
        return hd
    cfg = getattr(attn, "config", None)
    if cfg is not None:
        hd = getattr(cfg, "head_dim", None)
        if isinstance(hd, int) and not isinstance(hd, bool) and hd > 0:
            return hd
        n = getattr(cfg, "num_attention_heads", None)
        hs = getattr(cfg, "hidden_size", None)
        if (
            isinstance(n, int)
            and isinstance(hs, int)
            and n > 0
            and hs % n == 0
        ):
            return hs // n
    return None


def _resolve_head_counts(attn: Any) -> tuple[int | None, int | None]:
    """Best-effort (num_q_heads, num_kv_heads) for the fused-QKV split."""
    n_q = getattr(attn, "num_heads", None) or getattr(
        attn, "num_attention_heads", None
    )
    n_kv = getattr(attn, "num_key_value_heads", None)
    cfg = getattr(attn, "config", None)
    if cfg is not None:
        if n_q is None:
            n_q = getattr(cfg, "num_attention_heads", None)
        if n_kv is None:
            n_kv = getattr(cfg, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = n_q
    return n_q, n_kv


def _shift_proj_block(out, *, head_dim: int, n_heads: int, group_size: int):
    """Shift the second half of ``n_heads`` heads in a projection output.

    ``out`` is a Q or K projection result of shape ``[B, T, n_heads*head_dim]``
    (or ``[T, n_heads*head_dim]``). The tensor is reshaped to ``[B, H, T, hd]``,
    :func:`shift_heads_for_s2` rolls the second-half heads along the sequence
    dim, and the result is reshaped back. Returns the original tensor on any
    shape mismatch (best-effort).
    """
    if not hasattr(out, "shape") or len(out.shape) < 2:
        return out
    if n_heads < 2:
        return out
    squeeze = False
    x = out
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
        squeeze = True
    if len(x.shape) != 3:
        return out
    bsz, seq_len, dim = x.shape
    if dim != n_heads * head_dim:
        return out
    try:
        reshaped = x.reshape(bsz, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
        shifted = shift_heads_for_s2(reshaped, group_size=group_size)
        result = shifted.permute(0, 2, 1, 3).reshape(bsz, seq_len, dim)
    except (RuntimeError, ValueError, TypeError):
        return out
    return result.squeeze(0) if squeeze else result


class LongLoRAForwardOverride:
    """Context manager that installs + restores the S² Q/K-projection shift.

    v0.71.12 #158 — the shift is applied to the **q_proj / k_proj outputs**
    (before the attention dot-product), NOT to the attention output. This is
    the canonical S² formulation (LongLoRA paper §3.2): half the heads' Q/K
    are rolled along the sequence dim so adjacent local-attention windows
    exchange information at the dot-product level. Per-arch dispatch covers
    Llama / Mistral / Qwen2 (GQA-aware — the head count is derived per
    projection from its output dim) and Phi-3 (fused ``qkv_proj`` split).

    Records the original ``forward`` on each patched projection module and
    restores it on ``__exit__`` / ``__del__``. Idempotent — re-entering a
    second context on the same module is a no-op.

    Usage::

        with LongLoRAForwardOverride(model, group_size=4):
            trainer.train()
        # projections restored automatically

    We do NOT subclass HF attention modules — the wrapper runs the original
    projection forward then shifts its output, and trusts HF's own
    scaled-dot-product math downstream. This keeps the override compatible
    with FA v2 (FA v3 is rejected at the schema gate).
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
        # Walk the model and patch the q_proj / k_proj (or fused qkv_proj) of
        # every module whose class name matches a known attention pattern. We
        # touch only Llama / Mistral / Qwen / Phi attention shells — the schema
        # gate already rejects everything else at config load.
        # v0.53.11 review fix (security MEDIUM) — cap class name length so a
        # crafted model class with an arbitrarily long name does not feed
        # an unbounded string into the regex.
        max_class_name_len = 256
        # v0.71.16 #147 — ``Mixtral`` added: ``Mistral\w*Attention`` does NOT
        # match ``MixtralAttention`` (different token), so the override would
        # silently skip Mixtral without this alternative.
        attention_class_re = re.compile(
            r"(?:Llama|Mistral|Mixtral|Qwen|Phi)\w*Attention$"
        )

        for module in _walk_modules(self.model):
            cls_name = type(module).__name__
            if len(cls_name) > max_class_name_len:
                continue
            if not attention_class_re.match(cls_name):
                continue
            for proj, wrapper in self._proj_shift_targets(module, cls_name):
                original_forward = proj.forward
                # v0.53.11 review fix (code-review HIGH) — idempotent install.
                # Re-entering a second context on the same projection must NOT
                # double-wrap. Detect a previously-patched forward via marker.
                if getattr(original_forward, "_soup_longlora_patched", False):
                    continue
                self._patched.append((proj, original_forward))
                new_forward = wrapper(original_forward)
                new_forward._soup_longlora_patched = True  # type: ignore[attr-defined]
                proj.forward = new_forward

    def _proj_shift_targets(self, attn: Any, cls_name: str):
        """Yield ``(proj_module, wrapper_factory)`` for an attention module.

        Per-arch dispatch (v0.71.12 #158):
          * Phi-family fused ``qkv_proj`` → split q/k/v then shift q + k.
          * Separate ``q_proj`` + ``k_proj`` (Llama / Mistral / Qwen2) →
            shift each independently; GQA is handled by deriving the head
            count per projection from its own output dim.
        """
        head_dim = _resolve_head_dim(attn)
        if head_dim is None:
            return  # cannot shift safely without head_dim

        is_fused_family = cls_name.startswith(_FUSED_QKV_FAMILIES)
        if is_fused_family and hasattr(attn, "qkv_proj"):
            n_q, n_kv = _resolve_head_counts(attn)
            if not (isinstance(n_q, int) and isinstance(n_kv, int)):
                return  # cannot split fused qkv without head counts
            yield (
                attn.qkv_proj,
                self._make_fused_qkv_shift(head_dim, int(n_q), int(n_kv)),
            )
            return

        if hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
            qk_wrapper = self._make_separate_proj_shift(head_dim)
            yield (attn.q_proj, qk_wrapper)
            yield (attn.k_proj, qk_wrapper)

    def _make_separate_proj_shift(self, head_dim: int):
        """Return a wrapper-factory for a q_proj / k_proj forward.

        The number of heads is derived from the projection's OWN output dim
        (``D // head_dim``), so GQA (where k_proj has fewer heads than q_proj)
        is handled automatically.
        """
        group_size = self.group_size

        def factory(orig):
            def s2_proj_shift(*args, **kwargs):
                out = orig(*args, **kwargs)
                if not hasattr(out, "shape") or len(out.shape) < 2:
                    return out
                dim = out.shape[-1]
                if head_dim <= 0 or dim % head_dim != 0:
                    return out
                return _shift_proj_block(
                    out,
                    head_dim=head_dim,
                    n_heads=dim // head_dim,
                    group_size=group_size,
                )

            return s2_proj_shift

        return factory

    def _make_fused_qkv_shift(self, head_dim: int, n_q: int, n_kv: int):
        """Wrap a fused ``qkv_proj`` forward (Phi-3): split, shift q + k, recombine."""
        group_size = self.group_size
        q_dim = n_q * head_dim
        k_dim = n_kv * head_dim

        def factory(orig):
            local_orig = orig

            def bound_qkv_shift(*args, **kwargs):
                out = local_orig(*args, **kwargs)
                if not hasattr(out, "shape") or len(out.shape) < 2:
                    return out
                total = out.shape[-1]
                if total != q_dim + 2 * k_dim:
                    return out
                import torch

                try:
                    q = out[..., :q_dim]
                    k = out[..., q_dim:q_dim + k_dim]
                    v = out[..., q_dim + k_dim:]
                    q_sh = _shift_proj_block(
                        q, head_dim=head_dim, n_heads=n_q, group_size=group_size
                    )
                    k_sh = _shift_proj_block(
                        k, head_dim=head_dim, n_heads=n_kv, group_size=group_size
                    )
                    return torch.cat([q_sh, k_sh, v], dim=-1)
                except (RuntimeError, ValueError, TypeError):
                    return out

            bound_qkv_shift.__name__ = "s2_qkv_shift"
            return bound_qkv_shift

        return factory

    def _restore(self) -> None:
        while self._patched:
            module, original_forward = self._patched.pop()
            try:
                module.forward = original_forward
            except Exception:  # noqa: BLE001
                # If the module was deleted mid-run, ignore.
                pass


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
