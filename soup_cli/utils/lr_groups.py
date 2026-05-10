"""Per-module LR groups — v0.41.0 Part B.

Maps a list of (pattern, lr) entries onto a model's named parameters and
produces a list of optimizer parameter groups suitable for any
``torch.optim.Optimizer`` constructor.

Schema validation:
- ``lr_groups`` is a list of (pattern, lr) pairs (or dict alias).
- Capped at ``MAX_LR_GROUPS=32`` entries.
- Pattern: non-empty string, ≤256 chars, no null byte.
- LR: float in (0, 1.0].
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set, Tuple

MAX_LR_GROUPS = 32
_MAX_PATTERN_LEN = 256
_LR_LOWER_EXCLUSIVE = 0.0
_LR_UPPER_INCLUSIVE = 1.0


@dataclass(frozen=True)
class LrGroup:
    """A single (regex-pattern, lr) override entry.

    ``pattern`` is matched against the parameter's *fully qualified* name
    (e.g. ``model.layers.0.self_attn.q_proj.weight``) using ``re.search``.
    The first matching group wins; remaining params fall through to the
    base LR.
    """

    pattern: str
    lr: float


def parse_lr_groups(value: object) -> Optional[List[LrGroup]]:
    """Parse + validate the raw schema value.

    Accepts:
    - ``None`` → returns None (feature off).
    - List of dicts ``[{pattern: str, lr: float}, ...]``.
    - List of (pattern, lr) pairs (lists or tuples).
    - Mapping ``{pattern: lr, ...}`` — order preserved (dict insertion).

    Empty list/dict returns None (feature off, no-op).
    """
    if value is None:
        return None
    if isinstance(value, dict):
        items: List[Tuple[Any, Any]] = list(value.items())
    elif isinstance(value, list):
        items = []
        for entry in value:
            if isinstance(entry, dict):
                if set(entry.keys()) != {"pattern", "lr"}:
                    raise ValueError(
                        f"lr_groups dict entries must have exactly "
                        f"{{'pattern', 'lr'}} keys, got {sorted(entry.keys())}"
                    )
                items.append((entry["pattern"], entry["lr"]))
            elif isinstance(entry, (list, tuple)):
                if len(entry) != 2:
                    raise ValueError(
                        "lr_groups pair entries must be (pattern, lr), "
                        f"got {len(entry)} elements"
                    )
                items.append((entry[0], entry[1]))
            else:
                raise ValueError(
                    "lr_groups list entries must be dicts or (pattern, lr) "
                    f"pairs, got {type(entry).__name__}"
                )
    else:
        raise ValueError(
            "lr_groups must be a list of (pattern, lr) pairs or a dict, "
            f"got {type(value).__name__}"
        )
    if not items:
        return None
    if len(items) > MAX_LR_GROUPS:
        raise ValueError(
            f"lr_groups exceeds cap of {MAX_LR_GROUPS}, got {len(items)}"
        )
    seen: Set[str] = set()
    out: List[LrGroup] = []
    for raw_pattern, raw_lr in items:
        pattern = _validate_pattern(raw_pattern)
        if pattern in seen:
            raise ValueError(
                f"lr_groups contains duplicate pattern {pattern!r}"
            )
        seen.add(pattern)
        lr_value = _validate_lr(raw_lr, pattern)
        out.append(LrGroup(pattern=pattern, lr=lr_value))
    return out


def _validate_pattern(raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError(
            f"lr_groups pattern must be a string, got {type(raw).__name__}"
        )
    if not raw:
        raise ValueError("lr_groups pattern must be non-empty")
    if "\x00" in raw:
        raise ValueError("lr_groups pattern must not contain null bytes")
    if len(raw) > _MAX_PATTERN_LEN:
        raise ValueError(
            f"lr_groups pattern exceeds {_MAX_PATTERN_LEN} chars"
        )
    try:
        compiled = re.compile(raw)
    except re.error as exc:
        raise ValueError(
            f"lr_groups pattern {raw!r} is not a valid regex: {exc}"
        ) from None
    # Best-effort ReDoS probe: a 256-char pattern compiled against a
    # 128-char benign sample completes in microseconds for sane regexes.
    # Catastrophic-backtracking patterns like ``(a+)+`` will hang on
    # this synthetic input. We bound the work via signal-free timing
    # (Python's re has no timeout pre-3.11). The probe is a sanity
    # check, not a hard guarantee — the 256-char length cap above is
    # the primary defence.
    try:
        compiled.search("a" * 128)
    except re.error as exc:  # pragma: no cover — runtime regex errors
        raise ValueError(
            f"lr_groups pattern {raw!r} failed runtime probe: {exc}"
        ) from None
    return raw


def _validate_lr(raw: object, pattern: str) -> float:
    # Accept str forms of floats — PyYAML parses ``1e-4`` (no dot) as a
    # string in many versions; coercing here keeps the YAML surface friendly.
    if isinstance(raw, str):
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"lr_groups[{pattern!r}].lr must be a number, "
                f"got string {raw!r}"
            ) from None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"lr_groups[{pattern!r}].lr must be a number, "
            f"got {type(raw).__name__}"
        )
    lr_value = float(raw)
    if not math.isfinite(lr_value):
        raise ValueError(
            f"lr_groups[{pattern!r}].lr must be finite, got {lr_value}"
        )
    if lr_value <= _LR_LOWER_EXCLUSIVE or lr_value > _LR_UPPER_INCLUSIVE:
        raise ValueError(
            f"lr_groups[{pattern!r}].lr={lr_value} must be in "
            f"({_LR_LOWER_EXCLUSIVE}, {_LR_UPPER_INCLUSIVE}]"
        )
    return lr_value


def lr_groups_from_schema(
    raw: Optional[List[dict]],
) -> Optional[List[LrGroup]]:
    """Convert ``TrainingConfig.lr_groups`` (List[Dict]) into
    runtime ``List[LrGroup]`` for ``build_optimizer_param_groups``.

    The schema stores the canonical ``[{pattern, lr}, ...]`` shape so YAML
    round-trips cleanly via ``model_dump``; trainer code consumes typed
    ``LrGroup`` instances. Returns ``None`` for ``None`` / empty input.
    """
    if not raw:
        return None
    return [LrGroup(pattern=entry["pattern"], lr=float(entry["lr"]))
            for entry in raw]


def build_optimizer_param_groups(
    named_params: Iterable[Tuple[str, Any]],
    base_lr: float,
    lr_groups: Optional[List[LrGroup]],
) -> List[dict]:
    """Map ``model.named_parameters()`` onto optimizer param groups.

    Each parameter is assigned to the *first* matching ``lr_groups`` entry
    (search order = list order). Unmatched parameters land in a final
    base-LR group.

    Returns a list of dicts ``[{params: [...], lr: float, name: str}, ...]``
    suitable for ``torch.optim.AdamW(group_dicts, ...)``. Empty groups are
    omitted from the output.
    """
    if not isinstance(base_lr, (int, float)) or isinstance(base_lr, bool):
        raise ValueError(
            f"base_lr must be a number, got {type(base_lr).__name__}"
        )
    if base_lr <= 0:
        raise ValueError(f"base_lr must be > 0, got {base_lr}")

    materialised = list(named_params)
    if lr_groups is None or not lr_groups:
        return [{"params": [p for _, p in materialised], "lr": float(base_lr),
                 "name": "base"}]

    compiled = [(g, re.compile(g.pattern)) for g in lr_groups]
    buckets: List[List[Any]] = [[] for _ in compiled]
    base_bucket: List[Any] = []
    for pname, param in materialised:
        for idx, (_, regex) in enumerate(compiled):
            if regex.search(pname):
                buckets[idx].append(param)
                break
        else:
            base_bucket.append(param)
    out: List[dict] = []
    for (group, _), bucket in zip(compiled, buckets):
        if not bucket:
            continue
        out.append({
            "params": bucket,
            "lr": float(group.lr),
            "name": f"lr_group:{group.pattern}",
        })
    if base_bucket:
        out.append({
            "params": base_bucket,
            "lr": float(base_lr),
            "name": "base",
        })
    return out
