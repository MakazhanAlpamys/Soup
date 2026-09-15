"""rewind.jsonl — the training flight recorder read by `soup rewind`.

One header line, then one line per micro-batch: the dataset rows in it, each row's
mean supervised-token loss and supervised-token count. Written by the SFT trainers
when `training.rewind_log` is on. Never raises into training.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rich.console import Console

console = Console()

REWIND_LOG_VERSION = 1


class RewindLogError(ValueError):
    """Raised by :func:`read_rewind_log` for a missing or malformed file."""


def _encode_loss(value: Any) -> float | None:
    """One loss value on its way to disk, with every non-finite as ``None``.

    JSON has no NaN/Infinity token, so a non-finite loss is written as
    ``null`` and the file stays strict JSON on every line. Raises
    ``TypeError``/``ValueError`` for a value that is not a number — the
    caller drops that batch.
    """
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


class RewindLog:
    """Append-only JSONL writer for the rewind flight recorder.

    Passive logger: any I/O failure after construction disables further
    writes and prints ONE warning (mirrors the one-shot guard pattern used
    for EMA no-op warnings in ``grpo_stability_callback.py``) — training
    itself must never see an exception from this class.

    Each run starts a fresh file; a previous run's log is rotated to
    ``<path>.1``.

    A batch that cannot be coerced to the on-disk shape (a bad scalar, a
    length mismatch) is counted in :attr:`dropped` and the log keeps
    going; only an :class:`OSError` disables it.
    """

    FILENAME = "rewind.jsonl"

    def __init__(
        self,
        path: str | Path,
        *,
        backend: str,
        task: str,
        n_rows: int,
        batch_size: int,
        grad_accum: int,
        dataset_fingerprint: str,
    ) -> None:
        self._path = Path(path)
        self._disabled = False
        self._dropped = 0
        self._warned = False

        header = {
            "kind": "header",
            "version": REWIND_LOG_VERSION,
            "backend": backend,
            "task": task,
            "n_rows": n_rows,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "dataset_fingerprint": dataset_fingerprint,
            "created": time.time(),
        }
        line = json.dumps(header, ensure_ascii=False) + "\n"
        try:
            # mkdir inside the try (as trace_logger.TraceLogWriter does): an
            # output dir the trainer has not created yet must not raise here.
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_previous_run()
            with self._path.open("wb") as fh:
                fh.write(line.encode("utf-8"))
        except OSError as exc:
            self._warn_once(exc)

    def _rotate_previous_run(self) -> None:
        """Move a previous run's non-empty log aside to ``<path>.1``.

        One file per run keeps ``step``/``micro`` numbering unambiguous, so
        the writer truncates rather than appends; rotating first means the
        previous run is still recoverable. Best-effort — a rotation that
        cannot happen must not stop this run from logging.
        """
        try:
            if self._path.stat().st_size == 0:
                return
        except OSError:
            return  # nothing there (or unstattable) — nothing to rotate
        backup = self._path.with_suffix(self._path.suffix + ".1")
        try:
            # Mirror the trace_logger rotation policy: refuse to overwrite a
            # symlink at the backup path, so a pre-placed <path>.1 -> /etc/...
            # symlink cannot have this run's log renamed onto its target.
            try:
                if stat.S_ISLNK(os.lstat(backup).st_mode):
                    return
                backup.unlink()
            except FileNotFoundError:
                pass
            self._path.rename(backup)
        except OSError:
            return

    @property
    def path(self) -> Path:
        return self._path

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def dropped(self) -> int:
        return self._dropped

    def record_batch(
        self,
        *,
        step: int,
        micro: int,
        rows: Sequence[int],
        row_loss: Sequence[float],
        row_tokens: Sequence[int],
    ) -> None:
        """Append one micro-batch entry. Never raises.

        Callers hand over whatever the training loop has — a generator, a
        framework tensor scalar, ``None`` — so every length check and
        coercion happens inside the guard. A batch that will not coerce is
        counted in :attr:`dropped`; the log stays enabled.
        """
        if self._disabled:
            return
        try:
            # Inside the try: the arguments are typed ``Sequence`` but a caller
            # can hand over a generator, and ``len()`` on one raises.
            if len(rows) != len(row_loss) or len(rows) != len(row_tokens):
                self._dropped += 1
                return
            entry = {
                "kind": "batch",
                "step": int(step),
                "micro": int(micro),
                "rows": [int(r) for r in rows],
                "row_loss": [_encode_loss(v) for v in row_loss],
                "row_tokens": [int(t) for t in row_tokens],
            }
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with self._path.open("ab") as fh:
                fh.write(line.encode("utf-8"))
        except (TypeError, ValueError, OverflowError):
            # One unusable batch (bad scalar, non-sized rows) — drop it and
            # keep recording. Disabling here would lose the rest of the run.
            self._dropped += 1
        except OSError as exc:
            # Disk full / permissions / vanished dir — this will not recover.
            self._disabled = True
            self._warn_once(exc)

    def _warn_once(self, exc: Exception) -> None:
        if self._warned:
            return
        self._warned = True
        self._disabled = True
        console.print(f"[yellow]Rewind log disabled:[/] {self._path}: {exc}")

    def close(self) -> None:
        """No-op — each line is written under its own ``open``. Safe to call twice."""
        return


@dataclass(frozen=True)
class BatchRecord:
    step: int
    micro: int
    rows: tuple[int, ...]
    row_loss: tuple[float, ...]
    row_tokens: tuple[int, ...]


@dataclass(frozen=True)
class RewindRun:
    header: dict[str, Any]
    batches: tuple[BatchRecord, ...]

    def steps(self) -> list[int]:
        """Sorted unique step numbers across all recorded batches."""
        return sorted({b.step for b in self.batches})

    def batches_for(self, step: int) -> list[BatchRecord]:
        """Batches for ``step``, in the order they appear in the file."""
        return [b for b in self.batches if b.step == step]

    def step_loss(self, step: int) -> float | None:
        """Token-weighted mean loss over every row of every micro-batch in ``step``.

        ``None`` if the step has no batches OR its total token count is zero
        (nothing was measured — reporting ``0.0`` would make an unmeasured
        step look like the best step in a loss-spike report). ``nan`` if any
        row's loss is ``nan``.
        """
        batches = self.batches_for(step)
        if not batches:
            return None
        total_loss = 0.0
        total_tokens = 0
        for batch in batches:
            for loss, tokens in zip(batch.row_loss, batch.row_tokens):
                if math.isnan(loss):
                    return math.nan
                total_loss += loss * tokens
                total_tokens += tokens
        if total_tokens == 0:
            return None
        return total_loss / total_tokens


def _parse_loss(value: Any) -> float:
    """One on-disk loss value as a float, with every non-finite mapped to ``nan``.

    ``null`` is how this writer encodes a non-finite loss, but ``json.loads``
    also accepts the non-standard ``Infinity``/``-Infinity``/``NaN`` tokens a
    foreign writer may have emitted. Raises ``TypeError``/``ValueError`` for
    anything that is not a number — the caller skips that line.
    """
    if value is None:
        return math.nan
    parsed = float(value)
    return parsed if math.isfinite(parsed) else math.nan


def read_rewind_log(path: str | Path) -> RewindRun:
    """Parse a rewind.jsonl file into a :class:`RewindRun`.

    Raises :class:`RewindLogError` if the file is missing, the first
    non-blank line is not a ``{"kind": "header"}`` object, or its version
    does not match :data:`REWIND_LOG_VERSION`. Every other malformed line
    is skipped, including one with an unrecognised ``kind`` (forward
    compatibility) and a half-written last line from a run still in
    progress.
    """
    resolved = Path(path)
    try:
        # errors="replace": the file is read while training is still
        # appending, so the last line can be torn mid multi-byte character.
        # Strict decoding would fail the WHOLE file over that one line.
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RewindLogError(f"cannot read rewind log {resolved}: {exc}") from exc

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise RewindLogError(f"rewind log {resolved} is empty")

    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RewindLogError(f"rewind log {resolved} first line is not JSON: {exc}") from exc

    if not isinstance(header, dict) or header.get("kind") != "header":
        raise RewindLogError(f"rewind log {resolved} is missing a header line")
    if header.get("version") != REWIND_LOG_VERSION:
        raise RewindLogError(
            f"rewind log {resolved} has version {header.get('version')!r}, "
            f"expected {REWIND_LOG_VERSION}"
        )

    batches: list[BatchRecord] = []
    for raw_line in lines[1:]:
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("kind") != "batch":
            continue
        try:
            record = BatchRecord(
                step=obj["step"],
                micro=obj["micro"],
                rows=tuple(obj.get("rows", [])),
                row_loss=tuple(_parse_loss(v) for v in obj.get("row_loss", [])),
                row_tokens=tuple(obj.get("row_tokens", [])),
            )
        except (KeyError, TypeError, ValueError):
            continue  # missing step/micro, or a non-numeric field — skip the line
        batches.append(record)

    return RewindRun(header=header, batches=tuple(batches))


def dataset_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """Stable sha256 hex digest of ``rows``, independent of dict key order.

    Stable only for JSON-serialisable rows: anything else falls back to
    ``repr`` (``default=str``), and a repr containing an ``id()`` — the
    default for an object without ``__repr__`` — changes between processes.
    Callers pass plain dict rows.
    """
    payload = json.dumps(list(rows), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
