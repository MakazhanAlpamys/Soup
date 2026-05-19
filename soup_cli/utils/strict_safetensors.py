"""Strict safetensors mode — refuse pickle / PyTorch-classic weights (v0.60.0 Part C).

Static-extension allowlist. Treats any file with an extension in
``UNSAFE_EXTENSIONS`` as a potential arbitrary-code-execution vector and
refuses to proceed when ``strict=True``. Mirrors the HuggingFace safetensors
threat model (45% of HF repos still ship pickle weights as of late 2025).

Public surface:
- ``UNSAFE_EXTENSIONS`` frozenset.
- ``find_unsafe_weight_files(model_dir)`` -> tuple of offending paths.
- ``check_strict_safetensors(model_dir, *, strict=False)`` -> ``StrictSafetensorsReport``.

Exit-code policy when wired into the CLI: ``3`` distinct from generic errors
so CI pipelines can grep specifically for strict-safetensors failures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink, is_under

# Closed allowlist of "definitely unsafe" extensions. Picklemod gives every
# loader the right to execute arbitrary code on load — refusing the file at
# the boundary is the only sound mitigation.
UNSAFE_EXTENSIONS = frozenset({
    ".bin",      # pytorch_model.bin (legacy)
    ".pt",       # torch.save default
    ".pth",      # torch.save alt
    ".ckpt",     # PyTorch Lightning checkpoint
    ".pkl",      # raw pickle
    ".pickle",   # raw pickle
    ".joblib",   # sklearn joblib (uses pickle internally)
    ".msgpack",  # ambiguous binary blob — many loaders unpickle from this
})


@dataclass(frozen=True)
class StrictSafetensorsReport:
    """Result of a ``check_strict_safetensors`` call."""

    model_dir: str
    ok: bool
    unsafe_files: Tuple[str, ...]
    reason: str = field(default="")


def find_unsafe_weight_files(model_dir: str) -> Tuple[str, ...]:
    """Walk ``model_dir`` and return every file matching ``UNSAFE_EXTENSIONS``.

    Returns relative paths inside ``model_dir`` sorted alphabetically. Does
    NOT raise on a missing directory — that's the caller's job to gate.

    Symlinks pointing outside ``model_dir`` are silently skipped (we only
    consider names inside the adapter). A symlink WITHIN the dir whose name
    has an unsafe suffix is still flagged — the unsafe extension is the
    threat signal regardless of where the bytes physically live.
    """
    if not isinstance(model_dir, str) or not model_dir:
        raise ValueError("model_dir must be a non-empty str")
    if not os.path.isdir(model_dir):
        return ()

    offenders: list[str] = []
    for root, _, files in os.walk(model_dir, followlinks=False):
        # Defence-in-depth: skip dirs that walked outside model_dir
        # (followlinks=False prevents this, but check anyway via the
        # shared containment helper).
        if not is_under(root, model_dir):
            continue
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in UNSAFE_EXTENSIONS:
                full = os.path.join(root, filename)
                offenders.append(full)
    return tuple(sorted(offenders))


def check_strict_safetensors(
    model_dir: str, *, strict: bool = False,
) -> StrictSafetensorsReport:
    """Refuse pickle / PyTorch-classic weights when ``strict=True``.

    Args:
        model_dir: cwd-contained model / adapter directory.
        strict: when True, raise ``ValueError`` listing the offending file.
            When False, return a ``StrictSafetensorsReport`` with
            ``ok=False`` so callers can decide.

    Returns:
        ``StrictSafetensorsReport``.

    Raises:
        ValueError: in strict mode when any unsafe file is found.
        FileNotFoundError: when ``model_dir`` is not a directory.
        TypeError: when ``strict`` is not a bool.
    """
    if not isinstance(strict, bool):
        raise TypeError(
            f"strict must be bool, got {type(strict).__name__}"
        )
    enforce_under_cwd_and_no_symlink(model_dir, "model_dir")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"{model_dir}: not a directory")

    offenders = find_unsafe_weight_files(model_dir)
    if not offenders:
        return StrictSafetensorsReport(
            model_dir=os.path.basename(os.path.normpath(model_dir)),
            ok=True,
            unsafe_files=(),
            reason="all weight files are safetensors",
        )

    first = offenders[0]
    rel = os.path.relpath(first, model_dir)
    reason = (
        f"unsafe weight file (pickle / PyTorch-classic): {rel!r}; "
        "re-save as safetensors via "
        "`from safetensors.torch import save_file; save_file(...)`"
    )
    if strict:
        raise ValueError(reason)
    return StrictSafetensorsReport(
        model_dir=os.path.basename(os.path.normpath(model_dir)),
        ok=False,
        unsafe_files=offenders,
        reason=reason,
    )
