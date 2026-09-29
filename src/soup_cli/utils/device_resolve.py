"""Resolve a ``--device`` value into ``from_pretrained`` kwargs (#1443).

``infer.py``, ``chat.py`` and ``diff.py`` each accept a ``device`` argument
(explicit flag, or auto-detected via ``soup_cli.utils.gpu.detect_device``)
but hard-coded ``device_map="auto", torch_dtype=torch.float16`` on every
``from_pretrained`` call regardless of it. ``device_map="auto"`` lets
``accelerate`` place the model on any visible accelerator, which is exactly
wrong when the user asked for the CPU on purpose (to leave VRAM free for a
training job, or because the model does not fit on the GPU).
"""

from __future__ import annotations

from typing import Optional, Union


def resolve_device_map_and_dtype(device: Optional[str]) -> tuple[Union[str, dict], "object"]:
    """Map a resolved device string to ``(device_map, torch_dtype)``.

    Args:
        device: ``"cuda"``, ``"mps"``, ``"cpu"``, or another accelerator
            string; ``None``/empty when it was genuinely never resolved.

    Returns:
        A ``(device_map, torch_dtype)`` pair to splat into
        ``from_pretrained(**kwargs)``. ``device_map`` pins the model to the
        given device (``{"": device}``) so ``accelerate`` cannot override it;
        ``"auto"`` is used only when ``device`` is unset. ``torch_dtype`` is
        ``float32`` on CPU (float16 is not a CPU-appropriate dtype) and
        ``float16`` everywhere else (CUDA, MPS, and other accelerators).
    """
    import torch

    if not device:
        return "auto", torch.float16

    if device == "cpu":
        return {"": device}, torch.float32

    return {"": device}, torch.float16
