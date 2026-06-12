"""Freeze training: freeze bottom N layers of a model for parameter-efficient training."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _detect_num_layers(model: Any) -> int:
    """Detect total number of transformer layers from model parameter names.

    Looks for patterns like 'model.layers.N.' or 'transformer.h.N.' and
    returns max(N) + 1.
    """
    max_layer = -1
    pattern = re.compile(r"(?:layers|h)\.(\d+)\.")
    for name, _ in model.named_parameters():
        match = pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx > max_layer:
                max_layer = layer_idx
    return max_layer + 1 if max_layer >= 0 else 0


def freeze_model_layers(
    model: Any,
    freeze_layers: Optional[int] = None,
    freeze_ratio: Optional[float] = None,
) -> int:
    """Freeze the bottom layers of a model.

    Args:
        model: A PyTorch model with named_parameters().
        freeze_layers: Freeze the first N layers. Takes priority over freeze_ratio.
        freeze_ratio: Freeze this fraction of layers (e.g. 0.75 = 75% from bottom).

    Returns:
        Number of parameters frozen.
    """
    if freeze_layers is None and freeze_ratio is None:
        return 0

    total_layers = _detect_num_layers(model)
    if total_layers == 0:
        logger.warning(
            "freeze_model_layers: could not detect numbered layers in model "
            "parameter names. Freezing has no effect. Check that your model "
            "uses 'layers.N.' or 'h.N.' naming."
        )
        return 0

    # Determine cutoff
    if freeze_layers is not None:
        cutoff = min(freeze_layers, total_layers)
    else:
        cutoff = int(total_layers * freeze_ratio)

    # Freeze parameters in layers below cutoff
    frozen_count = 0
    pattern = re.compile(r"(?:layers|h)\.(\d+)\.")
    for name, param in model.named_parameters():
        match = pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx < cutoff:
                param.requires_grad = False
                frozen_count += 1

    return frozen_count


def apply_unfrozen_parameters(model: Any, patterns: list) -> int:
    """Freeze every parameter, then unfreeze those matching ``patterns``.

    This is the Spectrum (#266) targeted-training mechanism: full fine-tuning
    of a hand-picked parameter set (LoRA off). Each pattern is treated as a
    regular expression (``re.search`` against the parameter name) — the
    ``soup spectrum scan`` output uses parameter-name prefixes such as
    ``model.layers.0.self_attn.q_proj``.

    Args:
        model: A PyTorch model with ``named_parameters()``.
        patterns: Regex strings; a parameter is kept trainable if any matches.

    Returns:
        The number of parameter tensors left trainable.

    Raises:
        ValueError: If ``patterns`` is empty or a pattern is not valid regex.
    """
    if not patterns:
        raise ValueError("apply_unfrozen_parameters: patterns must be non-empty")

    compiled = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat))
        except re.error as exc:
            raise ValueError(
                f"apply_unfrozen_parameters: invalid regex {pat!r}: {exc}"
            ) from exc

    # Single pass over named_parameters() (cheaper than two walks on a large
    # model): a parameter is trainable iff a pattern matches AND it is a
    # float/complex tensor; everything else is frozen.
    trainable = 0
    skipped_non_float = []
    for name, param in model.named_parameters():
        matched = any(rx.search(name) for rx in compiled)
        if matched and not (param.is_floating_point() or param.is_complex()):
            # A matched quantized (uint8) weight cannot be full-fine-tuned —
            # skip it loudly rather than crash. The schema gate
            # (quantization='none') normally prevents this, but stay robust.
            skipped_non_float.append(name)
            matched = False
        param.requires_grad = matched
        if matched:
            trainable += 1

    if skipped_non_float:
        logger.warning(
            "apply_unfrozen_parameters: %d matched parameter(s) are non-float "
            "(e.g. quantized) and cannot be trained — skipped. Spectrum "
            "targeted training requires quantization: none. First skipped: %s",
            len(skipped_non_float),
            skipped_non_float[0],
        )

    if trainable == 0:
        logger.warning(
            "apply_unfrozen_parameters: no parameter matched any of "
            "%d pattern(s) — the model has NO trainable parameters. Check "
            "your training.unfrozen_parameters against the model's parameter "
            "names (run `soup spectrum scan` to regenerate).",
            len(patterns),
        )

    return trainable
