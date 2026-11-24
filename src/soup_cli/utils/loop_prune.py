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
VERSION_HISTORY_KEEP = 20
_LOGS = tuple(DEFAULT_LAYOUT[k] for k in ("iteration", "version", "progress", "paraphrase"))


@dataclass(frozen=True)
class PruneEntry:
    path: str
    reason: str
    is_dir: bool
    action: str = "remove"
    expected_size: int | None = None
    expected_mtime_ns: int | None = None
    replacement: bytes | None = None


@dataclass(frozen=True)
class PruneFailure:
    path: str
    reason: str


@dataclass(frozen=True)
class PruneResult:
    removed: int
    bytes_freed: int
    failed: tuple[PruneFailure, ...]
    skipped: tuple[PruneFailure, ...]


def _is_reparse_point(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return os.name == "nt" and bool(getattr(info, "st_file_attributes", 0) & reparse)


def _enforce_prune_path(path: Path, root: Path) -> None:
    """Recheck containment and every existing component without following links."""
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(root))
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    root_key = os.path.normcase(root_real)
    path_key = os.path.normcase(path_real)
    try:
        contained = os.path.commonpath([root_key, path_key]) == root_key
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"path escapes loop root: {path.as_posix()}")

    current = path
    while True:
        if os.path.lexists(current):
            if _is_reparse_point(current):
                raise ValueError(f"refusing link or reparse point: {current.as_posix()}")
            enforce_under_cwd_and_no_symlink(str(current), "loop prune")
        if current == root or current.parent == current:
            break
        current = current.parent
    if current != root:
        raise ValueError(f"path is not below loop root: {path.as_posix()}")


def _reject_nested_reparse_points(path: Path, root: Path) -> None:
    """Refuse a directory tree containing a link/junction before recursive deletion."""
    _enforce_prune_path(path, root)
    if not path.is_dir():
        return
    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for item in entries:
                child = Path(item.path)
                if _is_reparse_point(child):
                    raise ValueError(f"refusing link or reparse point: {child.as_posix()}")
                if item.is_dir(follow_symlinks=False):
                    pending.append(child)


def _tree_size(path: Path, root: Path) -> int:
    total = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        _enforce_prune_path(directory, root)
        with os.scandir(directory) as entries:
            for item in entries:
                child = Path(item.path)
                if _is_reparse_point(child):
                    raise ValueError(f"refusing link or reparse point: {child.as_posix()}")
                info = os.lstat(child)
                total += info.st_size
                if stat.S_ISDIR(info.st_mode):
                    pending.append(child)
    return total


def _regular_children(path: Path) -> list[Path]:
    if not path.exists():
        return []
    enforce_under_cwd_and_no_symlink(str(path), "loop prune")
    if not path.is_dir():
        raise ValueError(f"expected a directory: {path.as_posix()}")
    children = []
    for child in path.iterdir():
        if _is_reparse_point(child):
            raise ValueError(f"refusing link or reparse point: {child.as_posix()}")
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


def _log_tail_snapshot(
    path: Path, count: int, *, keep_iteration_ids: set[str] | None = None,
    root: Path = Path(".soup-loops"),
) -> tuple[list[bytes], int, int, int, bool]:
    """Read a log snapshot, retaining the tail and rows for active iterations."""
    _enforce_prune_path(path, root)
    before = path.stat()
    tail: deque[tuple[int, bytes]] = deque(maxlen=count)
    retained: list[tuple[int, bytes]] = []
    total = 0
    with path.open("rb") as handle:
        for index, line in enumerate(handle):
            tail.append((index, line))
            total += 1
            if keep_iteration_ids:
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    retained.append((index, line))
                    continue
                if not isinstance(row, dict) or not isinstance(row.get("iteration_id"), str):
                    retained.append((index, line))
                elif row["iteration_id"] in keep_iteration_ids:
                    retained.append((index, line))
    selected = {index: line for index, line in (*tail, *retained)}
    lines = [selected[index] for index in sorted(selected)]
    after = path.stat()
    stable = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    return lines, total, before.st_size, before.st_mtime_ns, stable


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
    _enforce_prune_path(root_path, root_path)
    if not root_path.exists():
        return []

    plan: list[PruneEntry] = []

    # Keep the newest N iteration manifests, plus the production, rollback and canary
    # iterations named by the current state. State may be absent in an uninitialized project.
    keep_ids: set[str] = set()
    try:
        state = read_state()
    except FileNotFoundError:
        state = None
    if state is not None:
        canary_iteration = state.canary_iteration
        if isinstance(canary_iteration, dict):
            canary_iteration = canary_iteration.get("iteration_id")
        keep_ids.update(i for i in (
            state.production_iteration_id,
            state.previous_production_iteration_id,
            canary_iteration,
        ) if isinstance(i, str) and i)
        pending = getattr(state, "pending_deploy", None)
        if isinstance(pending, dict):
            pending_iteration = pending.get("iteration_id")
            if isinstance(pending_iteration, str) and pending_iteration:
                keep_ids.add(Path(pending_iteration).name)
    # Version history is the rollback lineage. Keep the run manifests for its latest
    # VERSION_HISTORY_KEEP
    # transitions even when those runs are older than the ordinary N-run window.
    history_uncertain = False
    version_path = Path(DEFAULT_LAYOUT["version"])
    if version_path.is_file():
        version_tail, _ = _log_tail(version_path, VERSION_HISTORY_KEEP)
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
    run_id_by_iteration: dict[str, str] = {}
    for entry in _regular_children(root_path):
        if not entry.is_dir():
            continue
        manifest_path = entry / "iteration.json"
        enforce_under_cwd_and_no_symlink(str(manifest_path), "loop prune")
        if manifest_path.is_file():
            iteration_dirs.append(entry)
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                manifest = None
            if isinstance(manifest, dict):
                run_id = manifest.get("run_id")
                if isinstance(run_id, str) and run_id:
                    run_id_by_iteration[entry.name] = run_id
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
        retained_log_ids = set(keep_ids)
        recent_lines, _ = _log_tail(iteration_log, keep_log_lines)
        for line in recent_lines:
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(row, dict) and isinstance(row.get("iteration_id"), str):
                retained_log_ids.add(row["iteration_id"])
        with iteration_log.open("rb") as handle:
            retained_pairs = _referenced_pairs(handle, retained_log_ids)
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
        pending = getattr(state, "pending_deploy", None)
        pending_train = pending.get("train") if isinstance(pending, dict) else None
        pending_adapter = (
            pending_train.get("adapter_path") if isinstance(pending_train, dict) else None
        )
        if isinstance(pending_adapter, str) and pending_adapter:
            keep_run_ids.add(Path(pending_adapter.rstrip("/\\")).name)
    adapters = root_path / "adapters"
    adapter_dirs = [p for p in _regular_children(adapters) if p.is_dir()]
    # Newest first by mtime; equal mtimes (a coarse filesystem clock stamps back-to-back
    # creations alike) fall back to the name, so the plan never depends on listing order.
    adapter_dirs.sort(key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
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
    unretained_run_ids = {
        run_id for iteration_id, run_id in run_id_by_iteration.items()
        if iteration_id not in keep_ids
    }
    unretained_run_ids.update(p.name for p in adapter_dirs if p.name not in keep_run_ids)

    # Each harvested dataset is consumed synchronously by its training stage. No replay,
    # status, rollback, or later loop stage reopens prior pair files; retain those named by
    # the recent iteration log window for external inspection.
    pairs_dir = root_path / "pairs"
    pair_children = _regular_children(pairs_dir)
    if retained_pairs == {"*"} and pair_children:
        plan.append(PruneEntry(
            pairs_dir.as_posix(), "pairs kept because retained iteration history is incomplete",
            False, "keep",
        ))
    for entry in pair_children:
        known_pair = entry.name.startswith(("pairs-", "kto-", "sft-"))
        if (entry.is_file() and known_pair and retained_pairs != {"*"}
                and entry.name not in retained_pairs):
            plan.append(
                PruneEntry(
                    entry.as_posix(), "not referenced by retained iteration log lines", False
                )
            )
        elif entry.is_file() and not known_pair:
            plan.append(PruneEntry(
                entry.as_posix(), "unknown name; left untouched", False, "keep"
            ))

    # Keep only the most recent N lines of each loop-owned JSONL log. Plan as a file rewrite
    # (not deletion) so the dry run is transparent; apply is atomic in execute_prune.
    for log_rel in _LOGS:
        path = Path(log_rel)
        if path.is_file():
            # The planner relies on the latest version transitions to preserve rollback
            # targets on every future prune, so keep at least that many rows in this log.
            log_keep_lines = (
                max(keep_log_lines, VERSION_HISTORY_KEEP)
                if log_rel == DEFAULT_LAYOUT["version"] else keep_log_lines
            )
            lines, total, size, mtime_ns, stable = _log_tail_snapshot(
                path, log_keep_lines,
                keep_iteration_ids=keep_ids if log_rel == DEFAULT_LAYOUT["iteration"] else None,
                root=root_path,
            )
            if total > len(lines):
                reason = (
                    f"trim {total - len(lines)} old log lines; keep {len(lines)}"
                    if stable else "skip log changed while it was read"
                )
                plan.append(PruneEntry(
                    path.as_posix(), reason, False, "trim" if stable else "skip",
                    size, mtime_ns, b"".join(lines) if stable else None,
                ))

    # Verdict JSON/log artifacts and adapter recipe/log sidecars are tied to run_id. Only
    # delete files whose stem is not retained; unknown files are left alone.
    verdicts = root_path / "verdicts"
    for entry in _regular_children(verdicts):
        if not entry.is_file():
            continue
        if any(entry.name == run_id or entry.name.startswith(run_id + ".")
               for run_id in keep_run_ids):
            continue
        if any(entry.name == run_id or entry.name.startswith(run_id + ".")
               for run_id in unretained_run_ids):
            plan.append(PruneEntry(entry.as_posix(), "not used by retained runs", False))
        else:
            plan.append(PruneEntry(
                entry.as_posix(), "unknown name; left untouched", False, "keep"
            ))
    for entry in _regular_children(adapters):
        if not entry.is_file():
            continue
        if any(entry.name == run_id or entry.name.startswith(run_id + ".")
               for run_id in keep_run_ids):
            continue
        if any(entry.name == run_id or entry.name.startswith(run_id + ".")
               for run_id in unretained_run_ids):
            plan.append(PruneEntry(entry.as_posix(), "sidecar of an unretained run", False))
        else:
            plan.append(PruneEntry(
                entry.as_posix(), "unknown name; left untouched", False, "keep"
            ))
    return plan


def execute_prune(
    plan: list[PruneEntry], *, keep_log_lines: int = DEFAULT_KEEP_LOG_LINES,
    root: str = ".soup-loops",
) -> PruneResult:
    """Apply a reviewed plan; continue after failures and never follow reparse points."""
    removed = 0
    bytes_freed = 0
    failed: list[PruneFailure] = []
    skipped: list[PruneFailure] = []
    root_path = Path(root)
    log_paths = set(_LOGS)
    for entry in plan:
        path = Path(entry.path)
        if entry.action == "keep":
            continue
        if entry.action == "skip":
            skipped.append(PruneFailure(entry.path, entry.reason))
            continue
        try:
            _enforce_prune_path(path, root_path)
            if not os.path.lexists(path):
                continue
            if entry.path in log_paths:
                before = path.stat()
                if (entry.expected_size is None or entry.expected_mtime_ns is None
                        or before.st_size != entry.expected_size
                        or before.st_mtime_ns != entry.expected_mtime_ns):
                    skipped.append(PruneFailure(entry.path, "changed since the plan was read"))
                    continue
                mode = stat.S_IMODE(before.st_mode)
                fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(entry.replacement or b"")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(tmp, mode)
                    _enforce_prune_path(path, root_path)
                    current = path.stat()
                    if (current.st_size != entry.expected_size
                            or current.st_mtime_ns != entry.expected_mtime_ns):
                        skipped.append(PruneFailure(
                            entry.path, "changed before replacement; left untouched"
                        ))
                        continue
                    os.replace(tmp, path)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                bytes_freed += max(0, entry.expected_size - os.path.getsize(path))
            elif entry.is_dir:
                _reject_nested_reparse_points(path, root_path)
                size_before = _tree_size(path, root_path)
                shutil.rmtree(path)
                bytes_freed += size_before
            else:
                size_before = path.stat().st_size
                path.unlink()
                bytes_freed += size_before
            removed += 1
        except (OSError, ValueError, TypeError) as exc:
            failed.append(PruneFailure(entry.path, f"{type(exc).__name__}: {exc}"))
    return PruneResult(removed, bytes_freed, tuple(failed), tuple(skipped))


def daemon_is_running() -> bool:
    return inspect_pid_file(pid_file_path()).state == RUNNING
