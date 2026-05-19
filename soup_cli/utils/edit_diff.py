"""v0.61.0 Part E — Knowledge-injection diff visualizer (schema-only).

Renders a side-by-side comparison of what the model "knew" before vs
after a knowledge edit. Live model loading + probe generation is the
v0.61.1 deliverable; v0.61.0 ships the report dataclasses, probe-file
loader, and table renderer.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import dataclass
from typing import Optional, Tuple

_LOG = logging.getLogger(__name__)

_MAX_PROBE_ROWS: int = 1000
_MAX_PROBE_BYTES: int = 16 * 1024 * 1024
_MAX_PROMPT_LEN: int = 4096
_MIN_TOP_K: int = 1
_MAX_TOP_K: int = 100
_MAX_RUN_ID_LEN: int = 128


@dataclass(frozen=True)
class FactChange:
    """A single observed before/after pair from a probe prompt."""

    prompt: str
    before: str
    after: str
    changed: bool


@dataclass(frozen=True)
class DiffReport:
    """Top-k changes across the probe set."""

    before_run_id: str
    after_run_id: str
    changes: Tuple[FactChange, ...]
    total_probes: int
    soup_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.changes, tuple):
            raise TypeError("changes must be a tuple of FactChange")
        for c in self.changes:
            if not isinstance(c, FactChange):
                raise TypeError("changes entries must be FactChange instances")
        if not isinstance(self.total_probes, int) or isinstance(self.total_probes, bool):
            raise TypeError("total_probes must be int (not bool)")
        if self.total_probes < 0:
            raise ValueError("total_probes must be >= 0")

    def to_dict(self) -> dict:
        return {
            "before_run_id": self.before_run_id,
            "after_run_id": self.after_run_id,
            "total_probes": self.total_probes,
            "changes": [
                {
                    "prompt": c.prompt,
                    "before": c.before,
                    "after": c.after,
                    "changed": c.changed,
                }
                for c in self.changes
            ],
            "soup_version": self.soup_version,
        }


def _validate_run_id(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > _MAX_RUN_ID_LEN:
        raise ValueError(f"{field} must be <= {_MAX_RUN_ID_LEN} chars")
    return value


def _validate_top_k(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("top_k must not be bool")
    if not isinstance(value, int):
        raise TypeError(f"top_k must be int, got {type(value).__name__}")
    if value < _MIN_TOP_K:
        raise ValueError(f"top_k must be >= {_MIN_TOP_K}, got {value}")
    if value > _MAX_TOP_K:
        raise ValueError(f"top_k must be <= {_MAX_TOP_K}, got {value}")
    return value


def load_probes(path: str) -> Tuple[str, ...]:
    """Load a probe JSONL file with one ``{prompt: str}`` row per line.

    Cwd-contained + symlink-rejected + size-capped. Skips malformed
    rows silently (matches v0.55.0 / v0.56.0 evidence loader policy)
    but rejects oversize files loudly.
    """
    from soup_cli.utils.paths import is_under_cwd

    if not isinstance(path, str) or not path:
        raise ValueError("probe path must be a non-empty string")
    if "\x00" in path:
        raise ValueError("probe path must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(f"probe path must stay under cwd: {path!r}")
    real = os.path.realpath(path)
    try:
        st = os.lstat(real)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"probe file not found: {path!r}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("probe path must not be a symlink")
    if st.st_size > _MAX_PROBE_BYTES:
        raise ValueError(
            f"probe file exceeds {_MAX_PROBE_BYTES} bytes"
        )
    out: list[str] = []
    with open(real, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= _MAX_PROBE_ROWS:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                continue
            if "\x00" in prompt:
                continue
            if len(prompt) > _MAX_PROMPT_LEN:
                _LOG.warning(
                    "probe row %d truncated to %d chars (was %d)",
                    i, _MAX_PROMPT_LEN, len(prompt),
                )
                prompt = prompt[:_MAX_PROMPT_LEN]
            out.append(prompt)
    return tuple(out)


def build_diff_report(
    *,
    before_run_id: str,
    after_run_id: str,
    probe_file: Optional[str] = None,
    top_k: int = 10,
) -> DiffReport:
    """Build a placeholder :class:`DiffReport`.

    v0.61.0 returns an empty changes tuple when no probe file is
    supplied, or a stub-marked tuple when probes are supplied (live
    model invocation lands in v0.61.1). The shape is stable so callers
    can wire this into CI today.
    """
    before = _validate_run_id(before_run_id, "before_run_id")
    after = _validate_run_id(after_run_id, "after_run_id")
    k = _validate_top_k(top_k)

    if before == after:
        raise ValueError(
            "before_run_id and after_run_id must differ (no edit was "
            "applied between identical runs)."
        )

    probes: Tuple[str, ...] = ()
    if probe_file is not None:
        probes = load_probes(probe_file)

    # v0.61.0: empty changes (live diff comes from real model
    # generation in v0.61.1). Surface the probe count so operators
    # see the shape works end-to-end.
    changes_placeholder = tuple(
        FactChange(
            prompt=p,
            before="<v0.61.1 will generate>",
            after="<v0.61.1 will generate>",
            changed=False,
        )
        for p in probes[:k]
    )

    from soup_cli import __version__

    return DiffReport(
        before_run_id=before,
        after_run_id=after,
        changes=changes_placeholder,
        total_probes=len(probes),
        soup_version=__version__,
    )


def render_diff_table(report: DiffReport, console) -> None:
    """Render a diff report as a Rich table."""
    from rich.markup import escape
    from rich.table import Table

    if not isinstance(report, DiffReport):
        raise TypeError("report must be DiffReport")

    title = (
        f"Edit diff — {escape(report.before_run_id)} -> "
        f"{escape(report.after_run_id)}"
    )
    table = Table(title=title)
    table.add_column("prompt", overflow="fold")
    table.add_column("before", overflow="fold")
    table.add_column("after", overflow="fold")
    table.add_column("changed", justify="center")

    if not report.changes:
        table.add_row(
            "[dim]no probes supplied[/]",
            "[dim](deferred)[/]",
            "[dim](deferred)[/]",
            "[dim]-[/]",
        )
    else:
        for c in report.changes:
            mark = "[yellow]?[/]"  # v0.61.0 — neither true nor false
            table.add_row(
                escape(c.prompt),
                escape(c.before),
                escape(c.after),
                mark,
            )
    console.print(table)
    console.print(
        f"[dim]Total probes: {report.total_probes}; live generation "
        f"deferred to v0.61.1.[/]"
    )


def write_diff_report(report: DiffReport, path: str) -> str:
    """Atomic write of the diff report JSON. Cwd-contained + symlink-rejected."""
    if not isinstance(report, DiffReport):
        raise TypeError("report must be DiffReport")
    from soup_cli.utils.paths import atomic_write_text

    payload = json.dumps(report.to_dict(), sort_keys=True, indent=2)
    atomic_write_text(payload, path, field="output")
    return path
