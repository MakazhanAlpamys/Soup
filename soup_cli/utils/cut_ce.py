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


def apply_cut_ce(model_name: str) -> bool:
    """Patch HuggingFace transformers to use Cut Cross-Entropy.

    The patch replaces the model's ``loss_function`` (or forward CE call) with
    the fused CCE kernel. Must be called BEFORE model load so that all
    ``from_pretrained()`` instances see the patched class.

    Args:
        model_name: Base model name/path — used only to pick the
            architecture-specific patcher (Llama, Mistral, Qwen, …).

    Returns:
        True if the patch was applied successfully, False otherwise
        (missing package, unsupported architecture, or runtime patch failure).
    """
    if not check_cut_ce_available():
        return False

    # Match on the last path component to avoid substrings from
    # upstream-org / parent-dir names leaking into architecture selection
    # (e.g. "deepseek-ai/DeepSeek-R1-Distill-Phi-7B" should not be patched
    # with the Phi recipe when the model is actually DeepSeek-distilled).
    last_component = model_name.rsplit("/", 1)[-1].lower()

    # Detection rules are ordered from most-specific to least-specific to
    # avoid "codellama" matching "llama" first, etc.
    detectors = (
        (("codellama",), "llama"),
        (("llama",), "llama"),
        (("mixtral",), "mistral"),
        (("mistral",), "mistral"),
        (("qwen",), "qwen2"),
        (("gemma",), "gemma"),
        (("phi-3", "phi3", "phi4", "phi-4", "phi2", "phi-2"), "phi3"),
    )

    try:
        from cut_cross_entropy.transformers import cce_patch

        for keywords, arch in detectors:
            if any(keyword in last_component for keyword in keywords):
                cce_patch(arch)
                return True
    except (ImportError, AttributeError, NotImplementedError):
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
