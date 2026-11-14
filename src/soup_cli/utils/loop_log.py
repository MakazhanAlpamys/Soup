"""The loop's own logs: ``iterations.jsonl``, ``versions.jsonl`` and ``progress.jsonl``.

``soup loop`` writes three of the five data formats of ``docs/loop-formats.md`` (``soup
serve`` writes the other two). Each is an append-only JSONL file under ``.soup-loops/``:

- ``iterations.jsonl``: a snapshot of an iteration, appended when the iteration is recorded
  and again whenever its outcome changes later (a canary promoted, withdrawn); the latest
  line per ``iteration_id`` wins;
- ``versions.jsonl``: one line each time the active version changes;
- ``progress.jsonl``: stage events, so a person watching sees the loop work.

Other programs (a desktop application) may append to the same files, so a line is never
read back to decide anything and no lock is taken: every line is written by ONE ``write``
call on a file opened in append mode, and an earlier line is never touched. The state the
loop decides from lives in ``.soup/loop.yaml``.

A failing write never stops the loop: :class:`LoopLog` logs a warning and keeps a short note
(the exception's class, never a path) for the iteration's notes.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import os
import posixpath
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from soup_cli.utils.loop_formats import DEFAULT_LAYOUT, SCHEMA_VERSION
from soup_cli.utils.loop_iteration import IterationRecord, new_iteration_id
from soup_cli.utils.loop_state import MAX_CURSOR_TS, LoopState
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink, open_no_follow

_LOG = logging.getLogger(__name__)

#: The ``source`` of every line the loop writes.
SOURCE = "soup-loop"
#: The directory of the logs, from the contract layout (``.soup-loops``).
LOG_DIR = posixpath.dirname(DEFAULT_LAYOUT["iteration"])
#: Format name -> file name of its log inside the directory.
LOG_FILES: Mapping[str, str] = {
    kind: posixpath.basename(DEFAULT_LAYOUT[kind]) for kind in ("iteration", "version", "progress")
}
#: Largest line written, in bytes. A line is one ``write`` call; this keeps it small enough
#: that an append is not split by the filesystem.
MAX_LINE_BYTES = 1024 * 1024

_FIELD = "loop log"


# ---------------------------------------------------------------------------
# The append writer
# ---------------------------------------------------------------------------


def _encode_line(record: Mapping[str, Any]) -> bytes:
    """``record`` as one UTF-8 JSON line (no BOM, ``\\n`` ending). Raises on a bad record."""
    if not isinstance(record, Mapping):
        raise TypeError(f"a loop log record must be a mapping, got {type(record).__name__}")
    text = json.dumps(record, ensure_ascii=False, allow_nan=False)
    try:
        data = (text + "\n").encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate cannot be written as UTF-8: \u-escape it instead.
        data = (json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    if len(data) > MAX_LINE_BYTES:
        raise ValueError("a loop log line exceeds 1 MiB")
    return data


def _ends_mid_line(fd: int) -> bool:
    """True when the file has content and its last byte is not ``\\n`` (a torn line)."""
    size = os.lseek(fd, 0, os.SEEK_END)
    if size == 0:
        return False
    os.lseek(fd, size - 1, os.SEEK_SET)
    return os.read(fd, 1) != b"\n"


def append_json_line(path: str, record: Mapping[str, Any]) -> None:
    """Append ``record`` to the JSONL file ``path`` as one line, in one ``write`` call.

    The file is opened in append mode (created when missing), so another program's lines are
    never overwritten, and a line another writer left without its ``\\n`` (a crash) does not
    swallow this one: it is preceded by a newline in the same write. ``path`` must stay under
    the working directory and neither it nor a parent directory may be a symlink, junction or
    hard link. Raises ``ValueError`` / ``TypeError`` for a bad path or record and ``OSError``
    when the file cannot be written; nothing is written then.
    """
    enforce_under_cwd_and_no_symlink(path, _FIELD)
    data = _encode_line(record)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    try:
        fd = open_no_follow(path, flags, 0o644, check_parent=True, refuse_hardlink=True)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(
                f"{_FIELD} or its directory must not be a symlink or junction"
            ) from exc
        if exc.errno == errno.EMLINK:
            raise ValueError(f"{_FIELD} must not be a hard link") from exc
        raise
    try:
        if _ends_mid_line(fd):
            data = b"\n" + data
        view = memoryview(data)
        while len(view):
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(errno.EIO, "a loop log write stored nothing")
            view = view[written:]
    finally:
        os.close(fd)


class LoopLog:
    """The three logs of one project: where they are, and what went wrong writing them.

    ``base_dir`` is the directory of the logs (default ``.soup-loops``). ``enabled=False``
    writes nothing (``NO_LOG``). Every method returns whether the line was written and never
    raises; a failure is logged at WARNING and kept as a note for :meth:`take_failures`.
    """

    def __init__(self, base_dir: Optional[str] = None, *, enabled: bool = True) -> None:
        self._dir = LOG_DIR if base_dir is None else base_dir
        self._enabled = enabled
        self._failures: list[str] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def path(self, kind: str) -> str:
        """The file of format ``kind`` (``iteration``, ``version`` or ``progress``)."""
        return os.path.join(self._dir, LOG_FILES[kind])

    def iteration(self, line: Mapping[str, Any]) -> bool:
        """Append an iteration line unless that iteration (or canary) already has this outcome.

        A crash between the log append and ``loop.yaml`` makes the restart do the same step
        again; the second line would be a duplicate, so it is not written (it counts as
        written: the file already says it).
        """
        # A canary's closing line (promoted, withdrawn ...) belongs to the iteration that started
        # it (parent_iteration_id) and is written by a LATER cycle with an id of its own, so a
        # restarted cycle is recognised by the parent; any other line by its own id.
        if line.get("parent_iteration_id") is not None:
            keys = ("parent_iteration_id", "outcome")
        else:
            keys = ("iteration_id", "outcome")
        return self._append_once("iteration", line, keys)

    def version(self, line: Mapping[str, Any]) -> bool:
        """Append a version line unless the same change (version, cause, iteration) is there."""
        return self._append_once("version", line, ("to_version", "cause", "iteration_id"))

    def progress(self, line: Mapping[str, Any]) -> bool:
        return self._append("progress", line)

    def take_failures(self) -> list[str]:
        """The notes of the writes that failed since the last call (each once)."""
        taken, self._failures = self._failures, []
        return taken

    def _append_once(self, kind: str, line: Mapping[str, Any], keys: "tuple[str, ...]") -> bool:
        """``_append``, skipped when a line with the same ``keys`` values is already in the file."""
        wanted = tuple(line.get(name) for name in keys)
        if self._enabled and None not in wanted and self._has_line(kind, keys, wanted):
            return True
        return self._append(kind, line)

    def _has_line(self, kind: str, keys: "tuple[str, ...]", wanted: tuple) -> bool:
        try:
            with open(self.path(kind), "rb") as handle:
                for raw in handle:
                    try:
                        record = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(record, dict) and tuple(record.get(k) for k in keys) == wanted:
                        return True
        except OSError:
            return False
        return False

    def _append(self, kind: str, line: Mapping[str, Any]) -> bool:
        if not self._enabled:
            return False
        name = LOG_FILES[kind]
        try:
            append_json_line(self.path(kind), line)
        except (OSError, ValueError, TypeError) as exc:
            _LOG.warning("loop log %s not written: %s", name, type(exc).__name__)
            note = f"{name} not written ({type(exc).__name__})"
            if note not in self._failures:
                self._failures.append(note)
            return False
        return True


#: A log that writes nothing: what ``run_once`` uses when it is not given one.
NO_LOG = LoopLog(enabled=False)


# ---------------------------------------------------------------------------
# Line builders
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Now, UTC, to the second: ``2026-11-05T11:00:00+00:00``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cap(text: Optional[str], limit: int) -> Optional[str]:
    if text is None:
        return None
    return str(text)[:limit]


def _count(value: object) -> Optional[int]:
    """``value`` when it is a non-negative int (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def progress_line(
    iteration_id: Optional[str],
    stage: str,
    event: str,
    *,
    done: object = None,
    total: object = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    ts: Optional[str] = None,
) -> dict:
    """One progress event (``progress.schema.json``); ``done`` / ``total`` are None if uncounted."""
    line: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "iteration_id": iteration_id,
        "ts": ts or utc_now_iso(),
        "stage": stage,
        "event": event,
        "done": _count(done),
        "total": _count(total),
        "message": _cap(message, 1024),
    }
    if error is not None:
        line["error"] = _cap(error, 128)
    return line


def version_line(
    *,
    from_version: Optional[int],
    to_version: int,
    cause: str,
    actor: str,
    iteration_id: Optional[str],
    served_as: Optional[str],
    reason: Optional[str],
    ts: Optional[str] = None,
) -> dict:
    """One change of the active version (``version.schema.json``)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": str(uuid.uuid4()),
        "ts": ts or utc_now_iso(),
        "from_version": from_version,
        "to_version": to_version,
        "cause": cause,
        "actor": actor,
        "iteration_id": iteration_id,
        "served_as": _cap(served_as, 256),
        "alias": None,
        "reason": _cap(reason, 4096),
        "source": SOURCE,
    }


def portable_path(path: object) -> Optional[str]:
    """``path`` with ``/`` separators (the same string on every platform), or None."""
    if not isinstance(path, str) or not path:
        return None
    return path.replace(os.sep, "/") if os.sep != "/" else path


def sha256_of(path: str, *, limit: int = 1 << 30) -> Optional[str]:
    """The SHA-256 of the file at ``path`` as hex, or None when it cannot be read (or is huge)."""
    try:
        if os.path.getsize(path) > limit:
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The iteration line: ONE function turns the IterationRecord (plus the state and what the
# stages returned) into the contract's dict, so the manifest and the log cannot disagree
# ---------------------------------------------------------------------------


def _epoch_iso(epoch: object) -> Optional[str]:
    """Epoch seconds as ISO 8601 UTC (microseconds kept when there are any), or None."""
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _data(
    harvest: Mapping[str, Any], new_state: LoopState, *, trained: bool = False
) -> Optional[dict]:
    """What the harvest read, or None for a stage that does not count conversations.

    ``until`` is the cursor the iteration read up to. A training run reports its own (the data
    cursor follows the version in production and moves only on promotion, so the state's can be
    behind it); an iteration that trained nothing reports the state's.
    """
    if "traces_collected" not in harvest:
        return None
    until = new_state.harvest_cursor_ts
    if trained:
        read = harvest.get("cursor_ts")
        if (
            not isinstance(read, bool)
            and isinstance(read, (int, float))
            and 0 <= read <= MAX_CURSOR_TS  # False for NaN too
        ):
            until = float(read)
    data: dict[str, Any] = {"until": _epoch_iso(until)}
    conversations = _count(harvest.get("traces_collected"))
    if conversations is not None:
        data["conversations"] = conversations
    return data


def _dataset(harvest: Mapping[str, Any], graduated: Optional[int] = None) -> Optional[dict]:
    """The training data built from the harvest (None when none was written).

    ``graduated`` is the number of edits that moved from the held-out set into training in this
    iteration (the record's ``graduated``); left out when the iteration did not count them.
    """
    path = harvest.get("pairs_path")
    if not isinstance(path, str) or not path:
        return None
    dataset: dict[str, Any] = {"path": portable_path(path), "sha256": sha256_of(path)}
    for key in ("dpo_pairs", "kto_rows", "sft_rows"):
        number = _count(harvest.get(key))
        if number is not None:  # the schema has no null here: leave out what was not counted
            dataset[key] = number
    if graduated is not None:
        dataset["graduated"] = graduated
    return dataset


def _baseline(gate: Mapping[str, Any], state: LoopState) -> Optional[dict]:
    """What the candidate was compared against, as the contract names it (or None)."""
    raw = gate.get("verdict_baseline")
    if not isinstance(raw, Mapping) or raw.get("kind") not in ("base", "version"):
        return None
    ref = raw.get("ref")
    if not isinstance(ref, str) or not ref:
        return None
    baseline: dict[str, Any] = {"kind": raw["kind"], "ref": ref}
    if raw["kind"] == "version":
        baseline["version"] = state.production_version
    return baseline


_DECISIONS = {"OK": "SHIP", "MAJOR": "DON'T SHIP"}


def _verdict(
    record: IterationRecord, state: LoopState, gate: Mapping[str, Any]
) -> dict[str, Any]:
    """The check before rollout: a decision, the method, what it compared against."""
    method = gate.get("verdict_method")
    if method not in ("ship", "eval_gate"):
        method = "ship" if state.verify == "ship" else "eval_gate"
    notes = gate.get("notes")
    decision = _DECISIONS.get(record.gate_verdict)
    if decision is None:
        # Not a verdict: "ERROR" when the check could not run, "SKIPPED" when it was not asked.
        failed = isinstance(notes, str) and notes.startswith("verdict_skipped:")
        decision = "ERROR" if failed else "SKIPPED"
    reason = _cap(str(notes), 4096) if notes and decision in ("ERROR", "SKIPPED") else None
    if reason is None and decision == "DON'T SHIP":
        reason = record.verdict_reason  # the gate's words for not shipping
    verdict: dict[str, Any] = {
        "decision": decision,
        "method": method,
        "baseline": _baseline(gate, state),
        "path": portable_path(gate.get("verdict_path")),
        "reason": reason,
    }
    # The evidence of ``soup ship --gate loop-v2``, only when the check ran: absent = not measured.
    if record.verdict_gate is not None:
        verdict["gate"] = record.verdict_gate
    if record.verdict_corrections is not None:
        verdict["corrections"] = json.loads(json.dumps(record.verdict_corrections))
    if record.verdict_held_out is not None:
        verdict["held_out"] = json.loads(json.dumps(record.verdict_held_out))
    return verdict


def _reason(outcome: str, trained: bool, decision: Optional[str], notes: list[str], pct: object):
    """Why the iteration ended as it did: the stages' own notes, else a plain default."""
    if notes:
        return " | ".join(notes)[:4096]
    if outcome == "canary":
        return f"serving {pct:g}% of keyed traffic" if isinstance(pct, (int, float)) else None
    if outcome == "failed":
        return "a stage failed"
    if not trained:
        return "nothing to train"
    if decision == "DON'T SHIP":
        return "DON'T SHIP: the candidate did not pass the check before rollout"
    return "not rolled out"


def iteration_from_record(
    record: IterationRecord,
    *,
    state: LoopState,
    new_state: LoopState,
    harvest: Mapping[str, Any],
    train: Mapping[str, Any],
    gate: Mapping[str, Any],
    deploy: Mapping[str, Any],
) -> dict:
    """The ``iterations.jsonl`` line of the iteration ``record`` describes.

    ``state`` is the loop state the iteration started from (the version it started at and
    the iteration before it), ``new_state`` the state after it (the data cursor, the canary's
    traffic), and ``harvest`` / ``train`` / ``gate`` / ``deploy`` what the stages returned.
    Everything the record holds comes from the record, so the manifest written from the same
    object says the same. Anything the stages did not report is null or left out, never made
    up. The outcome is ``canary`` when a canary started, ``failed`` when the train or deploy
    stage reported that it failed, and ``skipped`` otherwise (nothing to train, a verdict of
    DON'T SHIP or none at all, a deploy that was not attempted).
    """
    trained = record.run_id is not None and not train.get("skipped")
    canary_name = deploy.get("canary_active") if record.shipped else None
    if canary_name:
        outcome = "canary"
    elif train.get("failed") or deploy.get("failed"):
        outcome = "failed"
    else:
        outcome = "skipped"
    verdict = _verdict(record, state, gate) if trained else None
    base = state.base_model if trained else None
    notes = [str(item) for item in (train.get("notes"), gate.get("notes"), deploy.get("notes"))
             if item]
    return {
        "schema_version": SCHEMA_VERSION,
        "iteration_id": record.iteration_id,
        "parent_iteration_id": state.production_iteration_id,
        "from_version": state.production_version,
        "to_version": state.latest_version + 1 if outcome == "canary" else None,
        "started_at": record.started_at,
        "finished_at": None if outcome == "canary" else record.finished_at,
        "data": _data(harvest, new_state, trained=trained),
        "dataset": _dataset(harvest, record.graduated),
        "train": (
            {"method": state.train_method, "run_id": record.run_id, "base": base,
             "root_base": base}
            if trained and base else None
        ),
        "verdict": verdict,
        "canary": (
            {"traffic_pct": new_state.canary_traffic_pct, "verdict": "UNKNOWN"}
            if outcome == "canary" and new_state.canary_traffic_pct is not None else None
        ),
        "artifacts": (
            {
                "adapter": portable_path(train.get("adapter_path")),
                "merged": None,
                "gguf": None,
                "served_as": _cap(str(canary_name), 256) if canary_name else None,
            }
            if trained else None
        ),
        "estimated_cost_usd": (
            record.estimated_cost_usd if math.isfinite(record.estimated_cost_usd) else None
        ),
        "outcome": outcome,
        "reason": _reason(
            outcome, trained, verdict["decision"] if verdict else None, notes,
            new_state.canary_traffic_pct,
        ),
    }


def canary_result(
    *,
    traffic_pct: Optional[float],
    verdict: Optional[str],
    stable_samples: Optional[int],
    canary_samples: Optional[int],
    quality: Optional[Mapping[str, Any]],
) -> dict:
    """The ``canary`` object of an iteration line; what was not measured is left out."""
    canary: dict[str, Any] = {}
    if traffic_pct is not None:
        canary["traffic_pct"] = traffic_pct
    if verdict is not None:
        canary["verdict"] = verdict
    if stable_samples is not None and canary_samples is not None:
        canary["samples"] = {"stable": stable_samples, "canary": canary_samples}
    if quality is not None:
        canary["quality"] = json.loads(json.dumps(quality))  # a copy: the caller keeps its own
    return canary


def resolved_line(
    snapshot: Mapping[str, Any],
    *,
    outcome: str,
    reason: Optional[str],
    finished_at: str,
    to_version: Optional[int],
    canary: Optional[Mapping[str, Any]] = None,
) -> dict:
    """The line that follows ``snapshot`` when its canary is promoted or withdrawn.

    A complete copy of the snapshot (the latest line per iteration wins, so it must say
    everything) with the new outcome, time and reason. ``to_version`` is the version the
    iteration produced, or None when it produced none (a withdrawn canary). ``canary``, when
    given, replaces the snapshot's ``canary`` object (the final samples and quality evidence);
    its ``traffic_pct`` falls back to the one the canary started with.
    """
    line = json.loads(json.dumps(snapshot))
    line.update(
        outcome=outcome, reason=_cap(reason, 4096), finished_at=finished_at, to_version=to_version
    )
    if canary is not None:
        started = line.get("canary") or {}
        merged = dict(canary)
        if "traffic_pct" not in merged and "traffic_pct" in started:
            merged["traffic_pct"] = started["traffic_pct"]
        line["canary"] = merged
    return line


def withdrawn_line(
    state: LoopState, reason: str, *, canary: Optional[Mapping[str, Any]] = None
) -> Optional[dict]:
    """The ``rolled_back`` line for the iteration whose canary is running, or None.

    None when no canary runs or no iteration started it (``soup loop canary``, or a rollout
    from before the snapshot was kept): there is nothing to complete.
    """
    snapshot = state.canary_iteration if state.canary_active else None
    if snapshot is None:
        return None
    return resolved_line(
        snapshot, outcome="rolled_back", reason=reason, finished_at=utc_now_iso(),
        to_version=None, canary=canary,
    )


# ---------------------------------------------------------------------------
# The version counter
# ---------------------------------------------------------------------------


def promotion_changes(state: LoopState) -> dict[str, Any]:
    """The ``LoopState`` fields a promotion of the running canary changes (version side).

    The new production takes the number after the highest ever promoted, so a number is not
    reused after a rollback. The iteration that produced it is the one whose canary this is
    (None for a canary no iteration started).
    """
    snapshot = state.canary_iteration if state.canary_active else None
    version = state.latest_version + 1
    return {
        "production_version": version,
        "previous_production_version": state.production_version,
        "latest_version": version,
        "production_iteration_id": snapshot["iteration_id"] if snapshot else None,
        "previous_production_iteration_id": state.production_iteration_id,
        "canary_iteration": None,
    }


def rollback_changes(state: LoopState) -> dict[str, Any]:
    """The ``LoopState`` fields ``soup loop rollback`` changes (version side).

    Returns to the version before the last promotion: the served model (version 1) when no
    previous adapter is remembered, as the command itself does. The running canary, if any,
    is withdrawn with it.
    """
    if state.previous_production_name is None:
        target = 1
    else:
        target = state.previous_production_version or max(1, state.production_version - 1)
    return {
        "production_version": target,
        "previous_production_version": None,
        "production_iteration_id": state.previous_production_iteration_id,
        "previous_production_iteration_id": None,
        "canary_iteration": None,
    }


def seed_lines(state: LoopState, *, actor: str) -> "tuple[dict, dict]":
    """The first version of the log: a seed iteration line and the matching ``initial`` line.

    The seed describes the model production serves now (version ``production_version``, 1 for
    a loop that never promoted), as a promoted iteration with ``reason`` ``initial deploy``.
    """
    seed_id = new_iteration_id()
    now = utc_now_iso()
    iteration = {
        "schema_version": SCHEMA_VERSION,
        "iteration_id": seed_id,
        "parent_iteration_id": None,
        "from_version": None,
        "to_version": state.production_version,
        "started_at": now,
        "finished_at": now,
        "data": None,
        "dataset": None,
        "train": None,
        "verdict": None,
        "canary": None,
        "artifacts": {
            "adapter": portable_path(state.production_adapter),
            "merged": None,
            "gguf": None,
            "served_as": _cap(state.production_name, 256),
        },
        "outcome": "promoted",
        "reason": "initial deploy",
    }
    version = version_line(
        from_version=None, to_version=state.production_version, cause="initial", actor=actor,
        iteration_id=seed_id, served_as=state.production_name, reason="initial deploy", ts=now,
    )
    return iteration, version


def start_log(state: LoopState, log: LoopLog, *, actor: str) -> LoopState:
    """Write the first version of the log once; returns the state that records it.

    A state whose ``version_log_started`` is set is returned as it is. When a line cannot be
    written the state is returned unchanged, so the next call tries again.
    """
    if state.version_log_started:
        return state
    iteration, version = seed_lines(state, actor=actor)
    if not (log.iteration(iteration) and log.version(version)):
        return state
    return replace(
        state, version_log_started=True, production_iteration_id=iteration["iteration_id"]
    )


# ---------------------------------------------------------------------------
# Stage events
# ---------------------------------------------------------------------------


class StageEvents:
    """The progress events of one iteration; an iteration that did no work writes none.

    An idle poll (nothing new to train on) would otherwise add a few lines every cycle.
    Events are held in memory, with the time they happened, until :meth:`go_live`; after
    that each one is written as it is emitted. An iteration that never goes live leaves
    only its iteration line.
    """

    def __init__(self, log: LoopLog, iteration_id: Optional[str]) -> None:
        self._log = log
        self._iteration_id = iteration_id
        self._held: list[dict] = []
        self._live = False

    def emit(self, stage: str, event: str, **fields: Any) -> None:
        line = progress_line(self._iteration_id, stage, event, **fields)
        if self._live:
            self._log.progress(line)
        else:
            self._held.append(line)

    def go_live(self) -> None:
        if self._live:
            return
        self._live = True
        held, self._held = self._held, []
        for line in held:
            self._log.progress(line)
