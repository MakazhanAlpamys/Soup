"""Heuristic ``diagnose`` orchestrator (v0.56.0).

Computes a FailureReport from caller-supplied evidence buckets. Live
model-loading factories (``utils.diagnose.live.load_adapter_pair`` etc.)
land in v0.56.1 alongside the v0.27.0-style stub-then-live pattern.

This module also exposes the JSON writer + a friendly atomic write that
matches the v0.43.0 Part D / v0.46.0 / v0.47.0 TOCTOU policy.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Mapping

from soup_cli.utils.diagnose.report import (
    FAILURE_MODES,
    FailureReport,
    FailureScore,
    compose_report,
)
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink


def neutral_score(mode: str, reason: str = "skipped") -> FailureScore:
    """Render a neutral OK score when a probe was not actually run.

    Centralised so CLI + SDK + train-gate use the same evidence string
    (code-review HIGH fix — removes duplicated helper).
    """
    return FailureScore(
        mode=mode,
        score=1.0,
        verdict="OK",
        evidence=f"probe not run ({reason})",
    )


def build_report(
    *,
    run_id: str,
    base: str,
    adapter: str,
    scores: Mapping[str, FailureScore],
    soup_version: str = "",
    extras: Mapping[str, str] | None = None,
) -> FailureReport:
    """Compose a FailureReport, filling missing modes with neutral stubs."""
    if not isinstance(scores, Mapping):
        raise TypeError("scores must be Mapping[str, FailureScore]")
    filled = dict(scores)
    for mode in FAILURE_MODES:
        if mode not in filled:
            filled[mode] = neutral_score(mode, "skipped")
        if not isinstance(filled[mode], FailureScore):
            raise TypeError(f"scores[{mode!r}] must be FailureScore")
    return compose_report(
        run_id=run_id,
        base=base,
        adapter=adapter,
        scores=filled,
        soup_version=soup_version,
        extras=extras,
    )


def write_report(report: FailureReport, path: str) -> str:
    """Atomically serialise the report to ``path`` (JSON).

    Cwd-containment + symlink-target rejection mirrors the project
    TOCTOU policy. Returns the resolved path on success.
    """
    if not isinstance(report, FailureReport):
        raise TypeError("report must be FailureReport")
    enforce_under_cwd_and_no_symlink(path, "diagnose report path")
    # os.path.realpath (not abspath) — Windows 8.3 short-name compat per
    # project policy (see commands/autopilot.py:_is_under_cwd).
    parent = os.path.dirname(os.path.realpath(path)) or "."
    if not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".diagnose-", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, allow_nan=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    return path


def diagnose(
    *,
    run_id: str,
    base: str,
    adapter: str,
    scores: Mapping[str, FailureScore] | None = None,
    soup_version: str = "",
    extras: Mapping[str, str] | None = None,
) -> FailureReport:
    """SDK entrypoint — same shape as the CLI command."""
    return build_report(
        run_id=run_id,
        base=base,
        adapter=adapter,
        scores=scores or {},
        soup_version=soup_version,
        extras=extras,
    )


