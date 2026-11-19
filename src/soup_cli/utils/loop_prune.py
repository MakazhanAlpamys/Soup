"""Plan and perform safe retention for ``.soup-loops`` artifacts.

Pruning is deliberately conservative: only loop-owned run artifacts, derived pair files,
and append-only loop logs are eligible. Source traces and feedback are never removed.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from soup_cli.utils.loop_detach import RUNNING, inspect_pid_file, pid_file_path
from soup_cli.utils.loop_formats import DEFAULT_LAYOUT
from soup_cli.utils.loop_state import read_state
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

DEFAULT_KEEP_RUNS = 20
DEFAULT_KEEP_LOG_LINES = 10_000
_LOGS = tuple(DEFAULT_LAYOUT[k] for k in ("iteration", "version", "progress", "paraphrase"))


@dataclass(frozen=True)
class PruneEntry:
    path: str
    reason: str
    is_dir: bool


def _regular_children(path: Path) -> list[Path]:
    if not path.exists():
        return []
    enforce_under_cwd_and_no_symlink(str(path), "loop prune")
    if not path.is_dir():
        raise ValueError(f"expected a directory: {path.as_posix()}")
    children = []
    for child in path.iterdir():
        enforce_under_cwd_and_no_symlink(str(child), "loop prune")
        children.append(child)
    return children


def _log_tail(path: Path, count: int) -> tuple[list[bytes], int]:
    enforce_under_cwd_and_no_symlink(str(path), "loop prune")
    with path.open("rb") as handle:
        tail: deque[bytes] = deque(maxlen=count)
        total = 0
        for line in handle:
            tail.append(line)
            total += 1
    return list(tail), total


def _referenced_pairs(iteration_lines: Iterable[bytes], keep_ids: set[str]) -> set[str]:
    referenced: set[str] = set()
    seen_ids: set[str] = set()
    for line in iteration_lines:
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # An unreadable retained log may contain a dataset reference. Fail closed by
            # retaining every pair rather than risking loss of data.
            return {"*"}
        if not isinstance(row, dict):
            return {"*"}
        iteration_id = row.get("iteration_id")
        if not isinstance(iteration_id, str):
            return {"*"}
        if iteration_id not in keep_ids:
            continue
        seen_ids.add(iteration_id)
        dataset = row.get("dataset")
        path = dataset.get("path") if isinstance(dataset, dict) else None
        if isinstance(path, str):
            normalized = path.replace("\\", "/")
            prefix = ".soup-loops/pairs/"
            if normalized.startswith(prefix) and "/" not in normalized[len(prefix):]:
                referenced.add(normalized[len(prefix):])
    if not keep_ids.issubset(seen_ids):
        # A retained run without a matching log row may still own a dataset we cannot name.
        return {"*"}
    return referenced


def build_prune_plan(*, keep_runs: int = DEFAULT_KEEP_RUNS,
                     keep_log_lines: int = DEFAULT_KEEP_LOG_LINES,
                     root: str = ".soup-loops") -> list[PruneEntry]:
    """Return deletions without mutating the filesystem; reject unsafe paths."""
    if isinstance(keep_runs, bool) or keep_runs < 1:
        raise ValueError("--keep-runs must be at least 1")
    if isinstance(keep_log_lines, bool) or keep_log_lines < 1:
        raise ValueError("--keep-log-lines must be at least 1")
    root_path = Path(root)
    enforce_under_cwd_and_no_symlink(str(root_path), "loop prune")
    if not root_path.exists():
        return []

    log_tails: dict[str, list[bytes]] = {}
    for log_rel in _LOGS:
        log_path = Path(log_rel)
        if log_path.exists():
            tail, _ = _log_tail(log_path, keep_log_lines)
            log_tails[log_rel] = tail
    plan: list[PruneEntry] = []

    # Keep the newest N iteration manifests, plus the production, rollback and canary
    # iterations named by the current state. State may be absent in an uninitialized project.
    keep_ids: set[str] = set()
    try:
        state = read_state()
    except FileNotFoundError:
        state = None
    if state is not None:
        keep_ids.update(i for i in (
            state.production_iteration_id,
            state.previous_production_iteration_id,
            state.canary_iteration,
        ) if i)
    # Version history is the rollback lineage. Keep the run manifests for its latest 20
    # transitions even when those runs are older than the ordinary N-run window.
    history_uncertain = False
    version_path = Path(DEFAULT_LAYOUT["version"])
    if version_path.is_file():
        version_tail, _ = _log_tail(version_path, 20)
        for line in version_tail:
            try:
                version = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                history_uncertain = True
                continue
            if not isinstance(version, dict):
                history_uncertain = True
            elif isinstance(version.get("iteration_id"), str):
                keep_ids.add(version["iteration_id"])
    iteration_dirs = []
    for entry in _regular_children(root_path):
        if not entry.is_dir():
            continue
        manifest_path = entry / "iteration.json"
        enforce_under_cwd_and_no_symlink(str(manifest_path), "loop prune")
        if manifest_path.is_file():
            iteration_dirs.append(entry)
    iteration_dirs.sort(key=lambda p: p.name, reverse=True)
    keep_ids.update(p.name for p in iteration_dirs[:keep_runs])
    if history_uncertain:
        # A damaged version log does not prove old rollback targets are unreachable.
        keep_ids.update(p.name for p in iteration_dirs)
    for entry in iteration_dirs:
        if entry.name not in keep_ids:
            plan.append(PruneEntry(entry.as_posix(), "older than retained loop runs", True))

    iteration_log = Path(DEFAULT_LAYOUT["iteration"])
    if iteration_log.is_file():
        # Dataset references for retained runs may be older than the log tail that will
        # remain after pruning. Scan the full current log before deciding what is unused.
        enforce_under_cwd_and_no_symlink(str(iteration_log), "loop prune")
        with iteration_log.open("rb") as handle:
            retained_pairs = _referenced_pairs(handle, keep_ids)
    else:
        retained_pairs = {"*"} if keep_ids else set()

    # Preserve adapter folders used by the active/rollback state and those tied to the
    # retained iterations, as well as the newest N orphaned folders.
    keep_run_ids: set[str] = set()
    manifest_uncertain = False
    for iteration_id in keep_ids:
        manifest = root_path / iteration_id / "iteration.json"
        enforce_under_cwd_and_no_symlink(str(manifest), "loop prune")
        if manifest.is_file():
            try:
                record = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                # Unknown manifest means its run artifacts cannot be proven unused.
                manifest_uncertain = True
                continue
            if not isinstance(record, dict):
                manifest_uncertain = True
                continue
            run_id = record.get("run_id") if isinstance(record, dict) else None
            if isinstance(run_id, str) and run_id:
                keep_run_ids.add(run_id)
        else:
            # State or version history may name a run whose manifest is missing. Without
            # that mapping, adapter and sidecar ownership cannot be established safely.
            manifest_uncertain = True
    if state is not None:
        for adapter in (state.production_adapter, state.previous_production_adapter,
                        state.canary_adapter):
            if isinstance(adapter, str):
                keep_run_ids.add(Path(adapter.rstrip("/\\")).name)
    adapters = root_path / "adapters"
    adapter_dirs = [p for p in _regular_children(adapters) if p.is_dir()]
    adapter_dirs.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    keep_run_ids.update(p.name for p in adapter_dirs[:keep_runs])
    if history_uncertain or manifest_uncertain:
        keep_run_ids.update(p.name for p in adapter_dirs)
    for entry in adapter_dirs:
        if entry.name not in keep_run_ids:
            plan.append(
                PruneEntry(
                    entry.as_posix(), "not used by retained runs or active versions", True
                )
            )

    # Each harvested dataset is consumed synchronously by its training stage. No replay,
    # status, rollback, or later loop stage reopens prior pair files; retain those named by
    # the recent iteration log window for external inspection.
    pairs_dir = root_path / "pairs"
    for entry in _regular_children(pairs_dir):
        if entry.is_file() and retained_pairs != {"*"} and entry.name not in retained_pairs:
            plan.append(
                PruneEntry(
                    entry.as_posix(), "not referenced by retained iteration log lines", False
                )
            )

    # Keep only the most recent N lines of each loop-owned JSONL log. Plan as a file rewrite
    # (not deletion) so the dry run is transparent; apply is atomic in execute_prune.
    for log_rel in _LOGS:
        path = Path(log_rel)
        if path.is_file():
            lines, total = _log_tail(path, keep_log_lines)
            if total > len(lines):
                plan.append(
                    PruneEntry(
                        path.as_posix(),
                        f"trim {total - len(lines)} old log lines; keep {len(lines)}",
                        False,
                    )
                )

    # Verdict JSON/log artifacts and adapter recipe/log sidecars are tied to run_id. Only
    # delete files whose stem is not retained; unknown files are left alone.
    verdicts = root_path / "verdicts"
    for entry in _regular_children(verdicts):
        if entry.is_file() and not any(
            entry.name == run_id or entry.name.startswith(run_id + ".")
            for run_id in keep_run_ids
        ):
            plan.append(PruneEntry(entry.as_posix(), "not used by retained runs", False))
    for entry in _regular_children(adapters):
        if entry.is_file() and not any(
            entry.name == run_id or entry.name.startswith(run_id + ".")
            for run_id in keep_run_ids
        ):
            plan.append(PruneEntry(entry.as_posix(), "sidecar of an unretained run", False))
    return plan


def execute_prune(plan: list[PruneEntry], *, keep_log_lines: int = DEFAULT_KEEP_LOG_LINES) -> int:
    """Apply a previously reviewed plan, preserving log tails with atomic replacement."""
    removed = 0
    log_paths = set(_LOGS)
    for entry in plan:
        path = Path(entry.path)
        enforce_under_cwd_and_no_symlink(str(path), "loop prune")
        if not path.exists():
            continue
        if entry.path in log_paths:
            tail, _ = _log_tail(path, keep_log_lines)
            mode = stat.S_IMODE(path.stat().st_mode)
            fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.writelines(tail)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(tmp, mode)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        elif entry.is_dir:
            shutil.rmtree(path)
        else:
            path.unlink()
        removed += 1
    return removed


def daemon_is_running() -> bool:
    return inspect_pid_file(pid_file_path()).state == RUNNING
