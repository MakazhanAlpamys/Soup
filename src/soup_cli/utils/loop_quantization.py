"""``quantization: auto`` of the loop's sft recipe: ``none`` when the base fits in free VRAM.

The recipe's ``quantization`` is ``auto`` by default. The train stage resolves it right before
it writes the soup.yaml and starts ``soup train``, so the figure is the card's free memory at
that moment:

- need = parameter count x 2 bytes (bf16) x 5 (weights, LoRA, optimizer, gradients and
  activations, with room to spare); free = free VRAM of the CUDA device;
- CUDA available and need <= free: ``none``; otherwise ``4bit`` (the safe side);
- ``4bit`` and ``none`` written in the recipe are explicit and never measured or overridden.

The parameter count is read without loading weights: the safetensors headers of a local model
directory, else the Hub's header-only metadata (skipped when ``HF_HUB_OFFLINE`` or
``TRANSFORMERS_OFFLINE`` is set), else unknown, which is also ``4bit``.

torch is imported inside ``free_vram_bytes`` only, so the CLI stays torch-free.
"""

from __future__ import annotations

import os
from typing import Optional

#: bf16: two bytes per parameter.
BYTES_PER_PARAM = 2
#: What a bf16 LoRA run holds per byte of base weights: the weights, the adapter and its
#: optimizer state, gradients and activations, with room to spare.
WORKING_SET_FACTOR = 5

_GB = 1024**3
_PREFIX = "sft quantization auto"


def estimated_need_bytes(params: int) -> int:
    """VRAM a bf16 LoRA run on ``params`` parameters is expected to need, in bytes."""
    return params * BYTES_PER_PARAM * WORKING_SET_FACTOR


def parameter_count(base: str) -> Optional[int]:
    """Parameter count of ``base`` without loading weights, or None when it cannot be told.

    A local model directory is counted from its safetensors headers (a symlinked directory
    or weights file is refused); a Hub repo id from the Hub's header-only metadata, unless
    the environment is offline; anything else is unknown.
    """
    if not isinstance(base, str) or not base:
        return None
    if os.path.isdir(base):
        from soup_cli.utils.gpu import param_count_from_local_safetensors

        return param_count_from_local_safetensors(base, refuse_symlinks=True)
    from soup_cli.utils import model_shape

    if not model_shape._REPO_ID.fullmatch(base) or ".." in base:
        return None
    return model_shape._hub_param_count(base, allow_network=True)


def free_vram_bytes() -> Optional[int]:
    """Free memory of the CUDA device in bytes, or None without torch, without CUDA, or on
    any error. torch is imported here, not at module level."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - no torch, no CUDA or a driver error: unknown
        return None
    if isinstance(free, bool) or not isinstance(free, int) or free < 0:
        return None
    return free


def resolve_sft_quantization(setting: str, base: str) -> tuple[str, Optional[str]]:
    """``(quantization, note)`` for the soup.yaml of an sft run on ``base``.

    ``4bit`` and ``none`` come back as they are with no note. ``auto`` is measured:
    ``none`` when ``estimated_need_bytes`` fits in ``free_vram_bytes``, otherwise ``4bit``;
    the note says which and why, for the train stage's notes.
    """
    if setting in ("4bit", "none"):
        return setting, None
    if setting != "auto":
        raise ValueError(f'quantization must be one of "4bit", "none", "auto", got {setting!r}')
    params = parameter_count(base)
    if params is None:
        return "4bit", f"{_PREFIX}: 4bit (parameter count unknown)"
    free = free_vram_bytes()
    if free is None:
        return "4bit", f"{_PREFIX}: 4bit (free VRAM unknown: no usable CUDA device)"
    need = estimated_need_bytes(params)
    chosen = "none" if need <= free else "4bit"
    return chosen, f"{_PREFIX}: {chosen} (need {need / _GB:.1f} GB, free {free / _GB:.1f} GB)"
