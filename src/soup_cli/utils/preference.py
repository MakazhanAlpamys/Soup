"""Preference loss resolution shared by config validation and trainer dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from soup_cli.config.schema import SoupConfig

SUPPORTED_PREFERENCE_LOSSES: frozenset[str] = frozenset({"dpo", "simpo", "orpo", "ipo", "bco"})


def resolve_preference_loss(cfg: SoupConfig) -> Optional[str]:
    """Return the scalar preference loss, its legacy task name, or ``None``."""
    if cfg.task == "preference":
        return cfg.training.preference_loss
    if cfg.task in SUPPORTED_PREFERENCE_LOSSES:
        return cfg.task
    return None
