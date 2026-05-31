"""Eval lock + coverage — `soup eval lock` / `soup eval coverage` (v0.55.0 Part C).

Freezes an :class:`EvalDesign` as a versioned, hash-checksummed baseline:

* ``lock_suite`` writes the design to a canonical JSON layout and computes
  a SHA-256 over the canonicalised bytes — the checksum is the registry
  artifact key.
* ``compute_coverage`` is a heuristic gap analysis between the suite's
  ``scorer_type`` mix and the v0.54.0 task taxonomy
  (``TASK_CATEGORIES``). It surfaces dimensions missing for the
  user-declared task category so the operator can spot gaps before
  shipping the gate.

Public surface
--------------
- Frozen dataclass: ``LockedSuite``, ``CoverageReport``.
- Pure functions: ``canonicalise_design_bytes``, ``checksum_design``,
  ``lock_suite``, ``compute_coverage``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import types
from collections.abc import Mapping
from dataclasses import dataclass

from soup_cli.utils.advise import TASK_CATEGORIES
from soup_cli.utils.eval_design import (
    SCORER_TYPES,
    EvalDesign,
    design_to_dict,
    load_eval_design,
)
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

_MAX_FILE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class LockedSuite:
    """Output of :func:`lock_suite`.

    ``checksum`` is the SHA-256 of the canonicalised design bytes — same
    bytes that landed on disk. Operators can re-compute it to detect
    drift.
    """

    path: str
    checksum: str
    dimension_count: int


@dataclass(frozen=True)
class CoverageReport:
    """Heuristic gap analysis for an eval suite.

    ``missing_scorers`` names scorer types the suite doesn't exercise
    given the declared task category. ``recommendations`` is a tuple of
    human-friendly suggestions.
    """

    task_category: str
    scorer_mix: Mapping[str, int]
    missing_scorers: tuple[str, ...]
    recommendations: tuple[str, ...]


# Per-task-category recommended scorer mix. MappingProxyType-wrapped so
# the registry cannot be mutated at runtime (project policy since
# v0.36.0 `_REGISTRY`).
_RECOMMENDED_SCORERS: Mapping[str, tuple[str, ...]] = types.MappingProxyType(
    {
        "factual_lookup": ("exact_match", "judge"),
        "style_shaping": ("judge",),
        "format_conversion": ("regex", "rlvr"),
        "reasoning": ("rlvr", "judge"),
        "tool_use": ("rlvr",),
        "summarization": ("judge",),
        "classification": ("exact_match",),
    }
)


# ---------------------------------------------------------------------------
# Canonical bytes + checksum
# ---------------------------------------------------------------------------

def canonicalise_design_bytes(design: EvalDesign) -> bytes:
    """Return UTF-8 bytes of a canonical (sorted-key, no-whitespace) JSON.

    The canonical layout is the registry-attachable artifact. Two
    designs hash identically iff their semantic content matches
    (insertion order does not affect the result).
    """
    if not isinstance(design, EvalDesign):
        raise TypeError("design must be an EvalDesign")
    payload = design_to_dict(design)
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def checksum_design(design: EvalDesign) -> str:
    """SHA-256 hex of :func:`canonicalise_design_bytes`."""
    return hashlib.sha256(canonicalise_design_bytes(design)).hexdigest()


def lock_suite(design: EvalDesign, output_path: str) -> LockedSuite:
    """Write the canonical JSON to disk + return a :class:`LockedSuite`."""
    enforce_under_cwd_and_no_symlink(output_path, "output_path")
    body = canonicalise_design_bytes(design)
    if len(body) > _MAX_FILE_BYTES:
        raise ValueError("locked suite exceeds 16 MiB cap")
    parent = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".soup-locked-suite.", dir=parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
        os.replace(tmp, output_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return LockedSuite(
        path=output_path,
        checksum=hashlib.sha256(body).hexdigest(),
        dimension_count=len(design.dimensions),
    )


# ---------------------------------------------------------------------------
# Coverage / gap analysis
# ---------------------------------------------------------------------------

def compute_coverage(
    design: EvalDesign,
    *,
    task_category: str,
) -> CoverageReport:
    """Compare the suite's scorer mix to the taxonomy's recommended set.

    The task category is validated against v0.54.0 :data:`TASK_CATEGORIES`
    so the recommendation table cannot be silently bypassed by a typo.
    """
    if not isinstance(design, EvalDesign):
        raise TypeError("design must be an EvalDesign")
    if isinstance(task_category, bool):
        raise TypeError("task_category must be str, got bool")
    if not isinstance(task_category, str):
        raise TypeError(
            f"task_category must be str, got {type(task_category).__name__}"
        )
    category = task_category.strip().lower()
    if category not in TASK_CATEGORIES:
        raise ValueError(
            f"unknown task_category {task_category!r}; allowed: "
            + ", ".join(TASK_CATEGORIES)
        )

    scorer_mix: dict[str, int] = {s: 0 for s in SCORER_TYPES}
    for dim in design.dimensions:
        scorer_mix[dim.scorer_type] = scorer_mix.get(dim.scorer_type, 0) + 1

    expected = set(_RECOMMENDED_SCORERS.get(category, ()))
    present = {s for s, count in scorer_mix.items() if count > 0}
    missing = tuple(sorted(expected - present))

    recommendations: list[str] = []
    for scorer in missing:
        recommendations.append(
            f"task category {category!r} benefits from a "
            f"{scorer!r} dimension — none configured"
        )
    if not design.dimensions:
        recommendations.append(
            "suite has no dimensions; run `soup eval design <data>` first"
        )
    elif not missing:
        recommendations.append(
            f"coverage looks good for task category {category!r}"
        )

    return CoverageReport(
        task_category=category,
        scorer_mix=dict(scorer_mix),
        missing_scorers=missing,
        recommendations=tuple(recommendations),
    )


def load_locked_suite(path: str) -> EvalDesign:
    """Convenience: delegate to :func:`load_eval_design`."""
    return load_eval_design(path)
