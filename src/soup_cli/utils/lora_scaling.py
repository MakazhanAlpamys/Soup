"""Decode-time scaling shared by LoRA adapter arithmetic and checkpoint merging."""

import math


def lora_scaling_factor(rank: float, alpha: float, *, use_rslora: bool) -> float:
    """Calculate scaling after the caller validates its rank, alpha and mode."""
    return alpha / (math.sqrt(rank) if use_rslora else rank)
