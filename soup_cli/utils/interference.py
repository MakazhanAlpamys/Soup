"""v0.66.0 Part D — Catastrophic interference matrix.

Pairwise N×N matrix of adapter compatibility scores. When multiple
LoRA adapters are loaded via the v0.22 multi-adapter serve, two adapters
can step on each other (e.g. an SQL adapter + a math adapter sharing
the same residual stream subspace).

The interference score is:

    score(A→B) = (loss(A_target | A + B loaded) - loss(A_target | A alone)) / loss(A alone)

Positive = loading B hurts A's target; negative = beneficial side effect
(rare but real with task-orthogonal adapters).

Classification (mirrors v0.26 / v0.56 / v0.65 taxonomy):

- |score| < 5%  → OK
- |score| < 20% → MINOR
- |score| ≥ 20% → MAJOR

This module is pure-Python — no torch/numpy required. Callers measure
the per-pair losses externally (via v0.22 multi-adapter `/v1/adapters`
hot-swap or the v0.55.0 eval suite) and feed the dict into
``build_interference_matrix``.

Public surface:

- ``InterferenceCell`` / ``InterferenceMatrix`` frozen dataclasses
- ``INTERFERENCE_VERDICTS`` closed allowlist
- ``compute_interference(loss_a_alone, loss_a_with_b)`` math kernel
- ``classify_interference(score)`` OK/MINOR/MAJOR
- ``build_interference_matrix(adapters, losses)`` orchestrator
- ``render_matrix_json`` / ``render_matrix_markdown``
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

INTERFERENCE_VERDICTS: frozenset[str] = frozenset({"OK", "MINOR", "MAJOR"})

# Classification bands.
_MINOR_THRESHOLD = 0.05  # 5% relative loss change
_MAJOR_THRESHOLD = 0.20  # 20% relative loss change

# Caps.
_MIN_ADAPTERS = 2
_MAX_ADAPTERS = 16  # 16×16 = 256 pairwise probes is the practical ceiling
_MAX_NAME_LEN = 256


def _md_escape(value: object) -> str:
    """Escape operator-controlled values for safe Rich markdown embedding.

    Review H2 fix (v0.66.0): adapter names + verdict / score strings flow
    from operator-supplied JSON into ``render_matrix_markdown`` which is
    rendered by ``rich.console.Console.print``. A crafted adapter name
    like ``"[link=file:///etc/passwd]click[/]"`` would inject Rich markup.
    """
    s = str(value)
    # Replace Rich markup metacharacters with their text equivalents.
    return s.replace("[", "\\[").replace("]", "\\]")


def _validate_adapter_name(name: object, field: str) -> str:
    if isinstance(name, bool) or not isinstance(name, str):
        raise TypeError(f"{field} must be str")
    if not name:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in name:
        raise ValueError(f"{field} must not contain null bytes")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"{field} must be ≤{_MAX_NAME_LEN} chars")
    return name


def _validate_finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be float, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be float")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    return float(value)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterferenceCell:
    """One pairwise interference measurement (target A, co-loaded B)."""

    adapter_a: str
    adapter_b: str
    score: float
    verdict: str

    def __post_init__(self) -> None:
        _validate_adapter_name(self.adapter_a, "adapter_a")
        _validate_adapter_name(self.adapter_b, "adapter_b")
        _validate_finite_float(self.score, "score")
        # H1 review fix: type check verdict before membership check.
        if isinstance(self.verdict, bool) or not isinstance(self.verdict, str):
            raise TypeError("verdict must be str")
        if self.verdict not in INTERFERENCE_VERDICTS:
            raise ValueError(
                f"verdict must be in {INTERFERENCE_VERDICTS}, got {self.verdict!r}"
            )


@dataclass(frozen=True)
class InterferenceMatrix:
    """Full N×N interference matrix + worst-pair summary."""

    adapters: Tuple[str, ...]
    cells: Tuple[InterferenceCell, ...]
    worst_pair: Optional[Tuple[str, str]]
    worst_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.adapters, tuple):
            raise TypeError("adapters must be tuple")
        if len(self.adapters) < _MIN_ADAPTERS:
            raise ValueError(f"need at least {_MIN_ADAPTERS} adapters")
        if len(self.adapters) > _MAX_ADAPTERS:
            raise ValueError(
                f"too many adapters (>{_MAX_ADAPTERS}); split your fleet"
            )
        for a in self.adapters:
            _validate_adapter_name(a, "adapter")
        if len(set(self.adapters)) != len(self.adapters):
            raise ValueError("duplicate adapter names in matrix")
        if not isinstance(self.cells, tuple):
            raise TypeError("cells must be tuple")
        for cell in self.cells:
            if not isinstance(cell, InterferenceCell):
                raise TypeError("cells entries must be InterferenceCell")
        _validate_finite_float(self.worst_score, "worst_score")
        if self.worst_pair is not None:
            if not isinstance(self.worst_pair, tuple) or len(self.worst_pair) != 2:
                raise TypeError("worst_pair must be 2-tuple or None")
            for name in self.worst_pair:
                _validate_adapter_name(name, "worst_pair entry")


# ---------------------------------------------------------------------------
# Math kernel + classification
# ---------------------------------------------------------------------------


def compute_interference(
    loss_a_alone: float,
    loss_a_with_b: float,
) -> float:
    """Relative loss change on A's domain when B is co-loaded.

    Returns ``(loss_a_with_b - loss_a_alone) / loss_a_alone``. Baseline
    loss must be > 0 (otherwise the ratio is undefined). Both values
    must be finite and non-bool.
    """
    base = _validate_finite_float(loss_a_alone, "loss_a_alone")
    combined = _validate_finite_float(loss_a_with_b, "loss_a_with_b")
    if base <= 0.0:
        raise ValueError("loss_a_alone must be > 0")
    if combined < 0.0:
        raise ValueError("loss_a_with_b must be ≥ 0")
    return (combined - base) / base


def classify_interference(score: float) -> str:
    """OK / MINOR / MAJOR by absolute score (5% / 20% bands).

    Boundary semantics: exact 0.05 → MINOR; exact 0.20 → MAJOR.
    """
    s = _validate_finite_float(score, "score")
    abs_s = abs(s)
    if abs_s < _MINOR_THRESHOLD:
        return "OK"
    if abs_s < _MAJOR_THRESHOLD:
        return "MINOR"
    return "MAJOR"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_interference_matrix(
    adapters: Tuple[str, ...],
    losses: Mapping[Tuple[str, str], float],
) -> InterferenceMatrix:
    """Build a full N×N interference matrix from per-pair losses.

    ``losses`` keys are ``(target_adapter, co_loaded_adapter)`` tuples and
    values are the measured loss on the target's probe set. The diagonal
    (``losses[(a, a)]``) is the baseline loss for adapter A.

    Cells on the diagonal always have score 0 (no co-loaded adapter).
    Missing off-diagonal entries default to baseline (score 0 — no
    measurement = no detected interference). The matrix surfaces the
    worst-scoring (highest |score|) off-diagonal pair.
    """
    # Validate adapter shape FIRST so duplicate-/empty-/oversize errors fire
    # before we touch the loss map.
    if not isinstance(adapters, tuple):
        raise TypeError("adapters must be tuple")
    if len(adapters) < _MIN_ADAPTERS:
        raise ValueError(f"need at least {_MIN_ADAPTERS} adapters")
    if len(adapters) > _MAX_ADAPTERS:
        raise ValueError(
            f"too many adapters (>{_MAX_ADAPTERS}); split your fleet"
        )
    for a in adapters:
        _validate_adapter_name(a, "adapter")
    if len(set(adapters)) != len(adapters):
        raise ValueError("duplicate adapter names")
    if not isinstance(losses, Mapping):
        raise TypeError("losses must be a Mapping")

    adapter_set = set(adapters)

    # Validate every key+value in losses
    for key, value in losses.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"losses keys must be 2-tuples, got {key!r}")
        target, co = key
        if (
            not isinstance(target, str)
            or isinstance(target, bool)
            or target not in adapter_set
        ):
            raise ValueError(f"unknown adapter in losses key: {target!r}")
        if (
            not isinstance(co, str)
            or isinstance(co, bool)
            or co not in adapter_set
        ):
            raise ValueError(f"unknown adapter in losses key: {co!r}")
        _validate_finite_float(value, "loss")
        if value < 0:
            raise ValueError(f"loss for {key!r} must be ≥ 0")

    # Diagonal must be present for every adapter (baseline reference)
    for a in adapters:
        if (a, a) not in losses:
            raise ValueError(f"missing diagonal entry for adapter {a!r}")
        if losses[(a, a)] <= 0:
            raise ValueError(
                f"baseline loss for {a!r} must be > 0"
            )

    cells: list[InterferenceCell] = []
    worst_pair: Optional[Tuple[str, str]] = None
    worst_abs = 0.0
    worst_score_signed = 0.0

    for target in adapters:
        baseline = losses[(target, target)]
        for co in adapters:
            if co == target:
                cells.append(
                    InterferenceCell(
                        adapter_a=target,
                        adapter_b=co,
                        score=0.0,
                        verdict="OK",
                    )
                )
                continue
            combined = losses.get((target, co))
            if combined is None:
                # No measurement → assume no interference
                score = 0.0
            else:
                score = compute_interference(baseline, combined)
            verdict = classify_interference(score)
            cells.append(
                InterferenceCell(
                    adapter_a=target,
                    adapter_b=co,
                    score=score,
                    verdict=verdict,
                )
            )
            # M3 review fix: strict `>` so first-encountered pair wins on
            # ties (matches v0.30.0 Candidate tie-break policy).
            if abs(score) > worst_abs:
                worst_abs = abs(score)
                worst_score_signed = score
                worst_pair = (target, co)

    return InterferenceMatrix(
        adapters=adapters,
        cells=tuple(cells),
        worst_pair=worst_pair,
        worst_score=worst_score_signed,
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_matrix_json(matrix: InterferenceMatrix) -> str:
    """Canonical JSON for CI / Registry artifact."""
    if not isinstance(matrix, InterferenceMatrix):
        raise TypeError("matrix must be InterferenceMatrix")
    payload = {
        "adapters": list(matrix.adapters),
        "worst_pair": (
            list(matrix.worst_pair) if matrix.worst_pair else None
        ),
        "worst_score": matrix.worst_score,
        "cells": [asdict(c) for c in matrix.cells],
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def render_matrix_markdown(matrix: InterferenceMatrix) -> str:
    """Markdown report — symmetric N×N table + worst-pair callout."""
    if not isinstance(matrix, InterferenceMatrix):
        raise TypeError("matrix must be InterferenceMatrix")
    if not matrix.cells:
        return (
            "# Adapter interference matrix\n"
            "\n"
            "_no cells (empty matrix)_\n"
        )
    lines = ["# Adapter interference matrix", ""]
    if matrix.worst_pair:
        worst_verdict = classify_interference(matrix.worst_score)
        lines.append(
            f"- Worst pair: **{_md_escape(matrix.worst_pair[0])} → "
            f"{_md_escape(matrix.worst_pair[1])}** "
            f"(score: {matrix.worst_score:+.4f}, "
            f"verdict: **{_md_escape(worst_verdict)}**)"
        )
        lines.append("")
    # Build N×N grid. Every operator-controlled value passes through
    # _md_escape so a crafted adapter name cannot inject Rich markup
    # (review H2 fix).
    header = ["A \\ B"] + [_md_escape(a) for a in matrix.adapters]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    by_pair = {
        (c.adapter_a, c.adapter_b): c for c in matrix.cells
    }
    for a in matrix.adapters:
        row = [_md_escape(a)]
        for b in matrix.adapters:
            cell = by_pair.get((a, b))
            if cell is None:
                row.append("—")
            else:
                row.append(
                    f"{cell.score:+.3f} ({_md_escape(cell.verdict)})"
                )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
