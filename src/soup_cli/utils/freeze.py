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
