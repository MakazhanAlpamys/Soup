"""Cut Cross-Entropy (CCE) — memory-efficient cross-entropy for large-vocab models.

Cut Cross-Entropy avoids materializing the full ``(batch, seq_len, vocab_size)``
logits tensor by computing the loss in chunks, saving 8-24GB VRAM on models with
large vocabularies (Llama 3.1 has 128k vocab → ~8GB of logits at bf16 per 8k
batch × seq slice).

Reference: https://github.com/apple/ml-cross-entropy

Requires: cut_cross_entropy (``pip install cut-cross-entropy``).

Incompatibilities:
- Unsloth backend has its own fused Cross-Entropy kernel
- MLX backend (Apple Silicon) — not supported upstream
- CUDA required; CPU is not useful for this scale of model
"""

from __future__ import annotations

# Single source of truth for the advisory both trainer/sft.py and
# utils/v028_features.py print when apply_cut_ce() returns False.
NO_MATCHING_ARCHITECTURE_MESSAGE = "no matching architecture or cut_cross_entropy not installed"


def check_cut_ce_available() -> bool:
    """Return True if the ``cut_cross_entropy`` package is importable."""
    try:
        import cut_cross_entropy  # noqa: F401

        return True
    except ImportError:
        return False


def get_cut_ce_version() -> str | None:
    """Return the installed ``cut_cross_entropy`` version, or None."""
    try:
        import cut_cross_entropy

        return getattr(cut_cross_entropy, "__version__", "unknown")
    except ImportError:
        return None


def _detect_model_type(model_name: str) -> str:
    """``config.model_type`` for a local path or hub id, or "" when unavailable.

    Mirrors ``liger.py::_detect_model_type``. Deliberately quiet: a missing
    config (offline, gated repo, no config.json) is not an error here, it
    just means the caller falls back to the name-based match.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    except Exception:  # noqa: BLE001 (detection is best-effort by design)
        return ""
    return str(getattr(config, "model_type", "") or "")


def apply_cut_ce(model_name: str) -> bool:
    """Patch HuggingFace transformers to use Cut Cross-Entropy.

    The patch replaces the model's ``loss_function`` (or forward CE call) with
    the fused CCE kernel. Must be called BEFORE model load so that all
    ``from_pretrained()`` instances see the patched class.

    Args:
        model_name: Base model name/path, used to resolve the
            architecture-specific patcher (Llama, Mistral, Qwen, …), first via
            ``config.model_type`` (works for a local checkpoint directory as
            well as a hub id) and, only if that is unavailable, via a
            substring match on the name itself.

    Returns:
        True if the patch was applied successfully, False otherwise
        (missing package, unsupported architecture, or runtime patch failure).
    """
    if not check_cut_ce_available():
        return False

    try:
        from cut_cross_entropy.transformers import cce_patch
    except (ImportError, AttributeError, NotImplementedError):
        return False

    # Primary path, mirrors liger.py's identical fix for #78: resolve the real
    # architecture from the model's own config. cce_patch raises RuntimeError
    # for a model_type it has no patcher for (e.g. Phi-2) rather than us guessing.
    model_type = _detect_model_type(model_name)
    if model_type:
        try:
            cce_patch(model_type)
            return True
        except (ImportError, AttributeError, NotImplementedError, RuntimeError, ValueError):
            pass

    # Fallback for when config resolution has nothing to read from. Match on
    # the last path component only, to keep an upstream-org / parent-dir
    # name from leaking into architecture selection (#456).
    clean_name = str(model_name or "").strip().replace("\\", "/").rstrip("/")
    last_component = clean_name.rsplit("/", 1)[-1].lower() if clean_name else ""
    detectors = (
        (("codellama",), "llama"),
        (("llama",), "llama"),
        (("mixtral",), "mistral"),
        (("mistral",), "mistral"),
        (("qwen",), "qwen2"),
        # Deliberately no bare "gemma" entry: cut_cross_entropy has no
        # plain-Gemma patcher, and dispatching to it used to crash instead
        # of reporting unsupported.
        (("gemma2", "gemma-2"), "gemma2"),
        (("phi-3", "phi3", "phi4", "phi-4"), "phi3"),
    )

    try:
        for keywords, arch in detectors:
            if any(keyword in last_component for keyword in keywords):
                cce_patch(arch)
                return True
    except (ImportError, AttributeError, NotImplementedError, RuntimeError, ValueError):
        return False

    return False


def validate_cut_ce_config(
    use_cut_ce: bool, backend: str, device: str
) -> list[str]:
    """Validate Cut Cross-Entropy configuration.

    Returns a list of error messages. Empty list means valid.
    """
    errors: list[str] = []

    if not use_cut_ce:
        return errors

    if not check_cut_ce_available():
        errors.append(
            "cut_cross_entropy is not installed. "
            "Install it with: pip install cut-cross-entropy"
        )

    if backend == "unsloth":
        errors.append(
            "Cut Cross-Entropy is not compatible with the unsloth backend. "
            "Unsloth has its own fused cross-entropy kernel. Use backend: transformers."
        )

    if backend == "mlx":
        errors.append(
            "Cut Cross-Entropy is not supported on the mlx backend. "
            "Use backend: transformers."
        )

    if device != "cuda":
        errors.append(
            "Cut Cross-Entropy requires CUDA. "
            f"Current device: {device}."
        )

    return errors
