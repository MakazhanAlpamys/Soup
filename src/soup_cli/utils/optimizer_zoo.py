"""Optimizer Zoo — v0.41.0 Part A.

Expands optimizer support beyond HF Trainer's built-in set to cover LlamaFactory
+ Axolotl parity: BAdam / APOLLO / Adam-mini / lomo / adalomo / grokadamw /
schedule_free / muon / dion / came_pytorch / ao_adamw_{fp8,4bit,8bit}.

Schema-level allowlist + dependency advisory. Live wiring delegates to HF
Trainer (which accepts arbitrary optimizer strings via TrainingArguments) plus
optional adapter packages installed alongside Soup.

Security:
- Closed allowlist of optimizer names (no arbitrary string accepted at
  schema level — prevents YAML-driven typos from silently using the
  default).
- Empty / null-byte / non-string optimizer rejected.
- Length cap (64 chars) — defence-in-depth against absurd inputs.
"""

from __future__ import annotations

import types
from typing import Optional

# HF Trainer / transformers built-in optimizers (no extra dep required).
_HF_NATIVE: frozenset = frozenset({
    "adamw_torch",
    "adamw_torch_fused",
    "adamw_torch_xla",
    "adamw_apex_fused",
    "adamw_anyprecision",
    "adafactor",
    "sgd",
    "adagrad",
    "adamw_hf",
    "rmsprop",
})

# bitsandbytes-backed (require `bitsandbytes` — already a Soup core dep).
_BNB_BACKED: frozenset = frozenset({
    "adamw_bnb_8bit",
    "adamw_8bit",
    "lion_8bit",
    "lion_32bit",
    "paged_adamw_8bit",
    "paged_adamw_32bit",
    "paged_lion_8bit",
    "paged_lion_32bit",
})

# v0.41.0 — new entries (require optional packages).
_NEW_V0_41_0: frozenset = frozenset({
    "badam",
    "apollo_adamw",
    "adam_mini",
    "lomo",
    "adalomo",
    "grokadamw",
    "schedule_free_adamw",
    "schedule_free_sgd",
    "muon",
    "dion",
    "came_pytorch",
    "ao_adamw_fp8",
    "ao_adamw_4bit",
    "ao_adamw_8bit",
})

SUPPORTED_OPTIMIZERS: frozenset = _HF_NATIVE | _BNB_BACKED | _NEW_V0_41_0

# Optional package required for each new optimizer. None = HF/bnb native.
# A user picking an entry whose package is missing gets a friendly "pip install"
# advisory at trainer construction time (not at schema-load) — schema-load
# happens before deps can be probed and the trainer can fall back gracefully
# in test environments. Wrapped in MappingProxyType to prevent runtime mutation
# (matches v0.36.0 _REGISTRY policy).
_OPTIMIZER_PACKAGES = types.MappingProxyType({
    "badam": "badam",
    "apollo_adamw": "apollo-torch",
    "adam_mini": "adam-mini",
    "lomo": "lomo-optim",
    "adalomo": "lomo-optim",
    "grokadamw": "grokadamw",
    "schedule_free_adamw": "schedulefree",
    "schedule_free_sgd": "schedulefree",
    "muon": "muon-optimizer",
    "dion": "dion-optimizer",
    "came_pytorch": "came-pytorch",
    "ao_adamw_fp8": "torchao",
    "ao_adamw_4bit": "torchao",
    "ao_adamw_8bit": "torchao",
})

_MAX_OPTIMIZER_NAME_LEN = 64


def validate_optimizer_name(name: object) -> str:
    """Schema-load-time validator for ``training.optimizer``.

    Returns the normalised name (lower-cased) on success, raises ValueError
    with an actionable message on failure.
    """
    if not isinstance(name, str):
        raise ValueError(
            f"optimizer must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("optimizer must be a non-empty string")
    if "\x00" in name:
        raise ValueError("optimizer must not contain null bytes")
    if len(name) > _MAX_OPTIMIZER_NAME_LEN:
        raise ValueError(
            f"optimizer name exceeds {_MAX_OPTIMIZER_NAME_LEN} chars"
        )
    normalised = name.lower()
    if normalised not in SUPPORTED_OPTIMIZERS:
        # Show a tight allowlist preview rather than dumping all 30+ names.
        suggestions = sorted(_NEW_V0_41_0)[:6]
        raise ValueError(
            f"optimizer={name!r} is not in the supported allowlist. "
            f"v0.41.0 additions include: {', '.join(suggestions)}. "
            "See soup_cli.utils.optimizer_zoo.SUPPORTED_OPTIMIZERS for "
            "the complete list."
        )
    return normalised


def required_package(name: str) -> Optional[str]:
    """Return the pip-installable package for a new-in-v0.41 optimizer.

    Returns None for HF-native and bnb-backed optimizers (already in core
    deps).
    """
    return _OPTIMIZER_PACKAGES.get(name.lower())


def is_new_v0_41_optimizer(name: str) -> bool:
    """True if ``name`` is one of the v0.41.0 additions."""
    if not isinstance(name, str):
        return False
    return name.lower() in _NEW_V0_41_0
