"""Retention ledger: the task x version matrix behind ``soup eval retention``.

A fine-tuning series (v1, v2, v3 ...) forgets by accumulation: each update looks
fine against the one before it while the first task slowly erodes. ``soup ship``
compares a candidate with ONE base on a fixed general suite, so it cannot see
that. The ledger remembers every control task ever registered and the score every
recorded version got on it, so a candidate is judged against the BEST score each
task has ever had, not the previous one.

This module is the pure half: the file format (schema 1), its validation, the
atomic write, the control-file hash check and the retention arithmetic. Nothing
here loads a model, and nothing imports torch or any other heavy dependency at
module level (the CLI half, ``commands/_eval_retention.py``, owns the live run).

Ledger file (JSON, ``retention.json`` by default)::

    {"schema": 1,
     "tasks": [{"name", "path", "sha256", "added_at", "added_with"}, ...],
     "versions": [{"label", "model", "adapter", "evaluated_at",
                   "scores": {task_name: accuracy},
                   "numerics": {"dtype", "device", "quantization",
                                "chat_template_kwargs"}}, ...]}

``numerics`` is how the version was loaded and decoded. Two scores are only
comparable when those are equal (bf16 on a GPU and fp32 on a CPU, a 4-bit load, a
Qwen3 chat template with thinking on or off all move accuracy), so a run whose
numerics differ from any recorded version is refused (:func:`check_numerics`).

Everything read from a ledger is untrusted (it can arrive in a pull request):
unknown keys are refused, every field is type- and bound-checked, and a message
names the key and the type it found instead of a traceback.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional, Tuple

from soup_cli.utils.paths import (
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
    open_no_follow,
    quote_path,
)
from soup_cli.utils.terminal import strip_control

SCHEMA_VERSION = 1
DEFAULT_LEDGER_PATH = "retention.json"
#: Same default as ``soup ship --forgetting-threshold``: an absolute accuracy
#: difference, as a fraction (0.05 = 5%).
DEFAULT_THRESHOLD = 0.05

MAX_LEDGER_BYTES = 4 * 1024 * 1024
MAX_TASKS = 1000
MAX_VERSIONS = 1000
MAX_NAME_CHARS = 64
MAX_LABEL_CHARS = 128
MAX_PATH_CHARS = 1024
MAX_MODEL_CHARS = 1024
MAX_STAMP_CHARS = 64
#: Bound on the canonical JSON text of the chat-template kwargs stamped on a version.
MAX_NUMERICS_CHARS = 512
#: A control task file is a few hundred JSONL rows; this bounds the hash read.
MAX_CONTROL_FILE_BYTES = 64 * 1024 * 1024

#: Decimal places a score, a drop and a mean are rounded to. A regression compares the
#: ROUNDED drop with the threshold, so the verdict agrees with the number the table
#: prints: 0.90 -> 0.85 is 0.05000000000000004 in floats, and must not fire at 0.05.
SCORE_DECIMALS = 6

STATUS_NEW = "new"
STATUS_OK = "ok"
STATUS_REGRESSED = "regressed"

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_ROOT_KEYS = frozenset({"schema", "tasks", "versions"})
_TASK_KEYS = frozenset({"name", "path", "sha256", "added_at", "added_with"})
_VERSION_KEYS = frozenset({"label", "model", "adapter", "evaluated_at", "scores", "numerics"})
#: Declaration order is the order a message names the keys in.
_NUMERICS_FIELDS = ("dtype", "device", "quantization", "chat_template_kwargs")
_NUMERICS_KEYS = frozenset(_NUMERICS_FIELDS)


class LedgerError(ValueError):
    """Input that cannot be used (a bad ledger, control file or flag): exit 3."""


class LedgerNotFoundError(LedgerError):
    """The ledger file does not exist."""


class LedgerChangedError(LedgerError):
    """The ledger file changed between this run reading it and writing it."""


class NumericsMismatchError(LedgerError):
    """This run's numerics differ from a recorded version's: the scores are not comparable."""


# ---------------------------------------------------------------------------
# Small validators
# ---------------------------------------------------------------------------

def _show(value: object, limit: int = 80) -> str:
    """Quote untrusted text for a message. Callers print it through ``for_terminal``."""
    text = strip_control(value)
    if len(text) > limit:
        text = text[:limit] + "..."
    return repr(text)


def _kind(value: object) -> str:
    return type(value).__name__


def utc_timestamp(moment: Optional[datetime] = None) -> str:
    """``2026-11-05T12:00:00Z`` for ``moment`` (default: now)."""
    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_task_name(name: object) -> str:
    """A control-task name: 1-64 chars of ``A-Z a-z 0-9 _ . -``, starting alphanumeric."""
    if not isinstance(name, str):
        raise LedgerError(f"task name must be a string, got {_kind(name)}")
    if not _NAME_RE.fullmatch(name):
        raise LedgerError(
            f"task name {_show(name)} is invalid: use 1-{MAX_NAME_CHARS} characters from "
            "A-Z a-z 0-9 _ . - and start with a letter or digit"
        )
    return name


def validate_text(value: object, field: str, *, max_chars: int) -> str:
    """A non-empty printable string with no leading/trailing space, within ``max_chars``."""
    if not isinstance(value, str):
        raise LedgerError(f"{field} must be a string, got {_kind(value)}")
    if not value or value != value.strip():
        raise LedgerError(
            f"{field} {_show(value)} must be non-empty with no leading or trailing space"
        )
    if len(value) > max_chars:
        raise LedgerError(f"{field} is too long ({len(value)} characters, max {max_chars})")
    if not value.isprintable():
        raise LedgerError(f"{field} {_show(value)} must not contain control characters")
    return value


def validate_label(label: object) -> str:
    """A version label: printable text, at most 128 characters."""
    return validate_text(label, "version label", max_chars=MAX_LABEL_CHARS)


def validate_model_ref(value: object, field: str) -> str:
    """A model id or path as typed (not resolved): printable, bounded."""
    return validate_text(value, field, max_chars=MAX_MODEL_CHARS)


def _unit_float(value: object) -> Optional[float]:
    """``value`` as a float if it is a finite number in [0, 1], else ``None``.

    An integer is range-checked BEFORE it is converted: ``float(10**400)`` raises
    ``OverflowError``, which a ledger with a 400-digit score would otherwise turn into
    a traceback instead of a refusal.
    """
    if isinstance(value, int):
        return float(value) if 0 <= value <= 1 else None
    out = float(value)
    return out if math.isfinite(out) and 0.0 <= out <= 1.0 else None


def _number_text(value: object) -> str:
    """A number for a message, never 400 digits of it."""
    if isinstance(value, int) and value.bit_length() > 64:
        return f"an integer of {value.bit_length()} bits"
    return repr(value)


def validate_threshold(value: object) -> float:
    """The regression threshold: a finite number in [0, 1] (an accuracy difference)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"threshold must be a number, got {_kind(value)}")
    out = _unit_float(value)
    if out is None:
        raise LedgerError("threshold must be between 0.0 and 1.0")
    return out


def validate_score(value: object, where: str) -> float:
    """An accuracy: a finite number in [0, 1]; bool is refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"{where} must be a number, got {_kind(value)}")
    out = _unit_float(value)
    if out is None:
        raise LedgerError(f"{where} must be between 0.0 and 1.0, got {_number_text(value)}")
    return out


def _require_keys(
    obj: Mapping[object, object],
    allowed: frozenset,
    where: str,
) -> None:
    """Refuse an unknown key and name a missing one (every key is required)."""
    unknown = sorted(str(key) for key in obj if key not in allowed)
    if unknown:
        joined = ", ".join(_show(key) for key in unknown)
        raise LedgerError(f"{where} has unsupported key(s): {joined}")
    missing = sorted(key for key in allowed if key not in obj)
    if missing:
        joined = ", ".join(_show(key) for key in missing)
        raise LedgerError(f"{where} is missing key(s): {joined}")


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetentionTask:
    """One registered control task: a held-out task-eval JSONL pinned by hash."""

    name: str
    path: str
    sha256: str
    added_at: str
    added_with: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "added_at": self.added_at,
            "added_with": self.added_with,
        }


@dataclass(frozen=True)
class Numerics:
    """How a version was loaded and decoded; scores are only comparable under equal numerics.

    ``dtype`` is ``None`` for a quantized load (the quantization names the precision),
    ``device`` is the device family (``cuda``, not ``cuda:1``) and
    ``chat_template_kwargs`` is the canonical JSON text of the template variables, or
    ``None`` when there are none.
    """

    dtype: Optional[str]
    device: str
    quantization: Optional[str]
    chat_template_kwargs: Optional[str]

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {name: getattr(self, name) for name in _NUMERICS_FIELDS}

    def differences(
        self, other: "Numerics"
    ) -> Tuple[Tuple[str, Optional[str], Optional[str]], ...]:
        """``(key, this value, other value)`` for every key that is not equal."""
        return tuple(
            (name, getattr(self, name), getattr(other, name))
            for name in _NUMERICS_FIELDS
            if getattr(self, name) != getattr(other, name)
        )


@dataclass(frozen=True)
class RetentionVersion:
    """One recorded version: the candidate and its accuracy on every task it saw."""

    label: str
    model: str
    adapter: Optional[str]
    evaluated_at: str
    scores: Dict[str, float]
    numerics: Numerics

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "model": self.model,
            "adapter": self.adapter,
            "evaluated_at": self.evaluated_at,
            "scores": dict(self.scores),
            "numerics": self.numerics.to_dict(),
        }


@dataclass(frozen=True)
class Ledger:
    """The registered tasks and the recorded versions, both in insertion order."""

    tasks: Tuple[RetentionTask, ...] = ()
    versions: Tuple[RetentionVersion, ...] = ()

    def task_names(self) -> Tuple[str, ...]:
        return tuple(task.name for task in self.tasks)

    def get_task(self, name: str) -> Optional[RetentionTask]:
        for task in self.tasks:
            if task.name == name:
                return task
        return None

    def has_label(self, label: str) -> bool:
        return any(version.label == label for version in self.versions)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": SCHEMA_VERSION,
            "tasks": [task.to_dict() for task in self.tasks],
            "versions": [version.to_dict() for version in self.versions],
        }


def _parse_task(raw: object, index: int) -> RetentionTask:
    where = f"ledger tasks[{index}]"
    if not isinstance(raw, dict):
        raise LedgerError(f"{where} must be an object, got {_kind(raw)}")
    _require_keys(raw, _TASK_KEYS, where)
    name = validate_task_name(raw["name"])
    path = validate_text(raw["path"], f"{where}.path", max_chars=MAX_PATH_CHARS)
    sha = raw["sha256"]
    if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
        raise LedgerError(f"{where}.sha256 must be 64 lowercase hex characters")
    added_at = validate_text(raw["added_at"], f"{where}.added_at", max_chars=MAX_STAMP_CHARS)
    added_with = raw["added_with"]
    if added_with is not None:
        added_with = validate_text(
            added_with, f"{where}.added_with", max_chars=MAX_LABEL_CHARS
        )
    return RetentionTask(name, path, sha, added_at, added_with)


def _optional_text(value: object, field: str, *, max_chars: int) -> Optional[str]:
    return None if value is None else validate_text(value, field, max_chars=max_chars)


def _checked_numerics(
    dtype: object,
    device: object,
    quantization: object,
    chat_template_kwargs: object,
    where: str,
) -> Numerics:
    """A ``Numerics`` whose every field is type- and bound-checked."""
    return Numerics(
        dtype=_optional_text(dtype, f"{where}.dtype", max_chars=MAX_STAMP_CHARS),
        device=validate_text(device, f"{where}.device", max_chars=MAX_STAMP_CHARS),
        quantization=_optional_text(
            quantization, f"{where}.quantization", max_chars=MAX_STAMP_CHARS
        ),
        chat_template_kwargs=_optional_text(
            chat_template_kwargs, f"{where}.chat_template_kwargs", max_chars=MAX_NUMERICS_CHARS
        ),
    )


def _parse_numerics(raw: object, where: str) -> Numerics:
    if not isinstance(raw, dict):
        raise LedgerError(f"{where} must be an object, got {_kind(raw)}")
    _require_keys(raw, _NUMERICS_KEYS, where)
    return _checked_numerics(
        raw["dtype"], raw["device"], raw["quantization"], raw["chat_template_kwargs"], where
    )


def make_numerics(
    *,
    dtype: Optional[str],
    device: str,
    quantization: Optional[str],
    chat_template_kwargs: Optional[dict] = None,
) -> Numerics:
    """The stamp of a live run. Template kwargs become canonical JSON (sorted keys)."""
    text: Optional[str] = None
    if chat_template_kwargs:
        try:
            text = json.dumps(
                chat_template_kwargs, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"--chat-template-kwargs cannot be recorded: {exc}") from exc
        if len(text) > MAX_NUMERICS_CHARS:
            raise LedgerError(
                f"--chat-template-kwargs is too long to record ({len(text)} characters, "
                f"max {MAX_NUMERICS_CHARS})"
            )
    return _checked_numerics(dtype, device, quantization, text, "numerics")


def _numerics_text(value: Optional[str]) -> str:
    return "None" if value is None else _show(value)


def check_numerics(ledger: "Ledger", numerics: Numerics) -> None:
    """Refuse a run whose numerics differ from ANY recorded version's.

    Raises :class:`NumericsMismatchError` naming every differing key with this run's
    value and the recorded one. An empty ledger accepts anything.
    """
    for version in ledger.versions:
        differences = version.numerics.differences(numerics)
        if not differences:
            continue
        parts = "; ".join(
            f"{key} {_numerics_text(mine)} (this run) vs {_numerics_text(theirs)} "
            f"(version {_show(version.label)})"
            for key, theirs, mine in differences
        )
        raise NumericsMismatchError(
            f"this run's numerics differ from the ledger's: {parts}. Scores measured under "
            "different numerics are not comparable; re-run with the recorded settings, or "
            "start a new ledger (--ledger NEW.json) for the new ones."
        )


def _parse_version(raw: object, index: int, task_names: frozenset) -> RetentionVersion:
    where = f"ledger versions[{index}]"
    if not isinstance(raw, dict):
        raise LedgerError(f"{where} must be an object, got {_kind(raw)}")
    _require_keys(raw, _VERSION_KEYS, where)
    label = validate_label(raw["label"])
    model = validate_text(raw["model"], f"{where}.model", max_chars=MAX_MODEL_CHARS)
    adapter = raw["adapter"]
    if adapter is not None:
        adapter = validate_text(adapter, f"{where}.adapter", max_chars=MAX_MODEL_CHARS)
    evaluated_at = validate_text(
        raw["evaluated_at"], f"{where}.evaluated_at", max_chars=MAX_STAMP_CHARS
    )
    raw_scores = raw["scores"]
    if not isinstance(raw_scores, dict):
        raise LedgerError(f"{where}.scores must be an object, got {_kind(raw_scores)}")
    scores: Dict[str, float] = {}
    for task, value in raw_scores.items():
        if task not in task_names:
            raise LedgerError(
                f"{where} ({_show(label)}) scores unknown task {_show(task)}; "
                "every scored task must be listed in the ledger's tasks"
            )
        scores[task] = validate_score(value, f"{where}.scores[{_show(task)}]")
    numerics = _parse_numerics(raw["numerics"], f"{where}.numerics")
    return RetentionVersion(label, model, adapter, evaluated_at, scores, numerics)


def parse_ledger(payload: object) -> Ledger:
    """Validate a decoded ledger and return it. Raises ``LedgerError`` naming the problem."""
    if not isinstance(payload, dict):
        raise LedgerError(f"ledger must be a JSON object, got {_kind(payload)}")
    _require_keys(payload, _ROOT_KEYS, "ledger")
    schema = payload["schema"]
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise LedgerError(f"ledger 'schema' must be an integer, got {_kind(schema)}")
    if schema != SCHEMA_VERSION:
        raise LedgerError(
            f"unsupported ledger schema {schema} (this soup reads schema {SCHEMA_VERSION})"
        )
    raw_tasks = payload["tasks"]
    if not isinstance(raw_tasks, list):
        raise LedgerError(f"ledger 'tasks' must be a list, got {_kind(raw_tasks)}")
    if len(raw_tasks) > MAX_TASKS:
        raise LedgerError(f"ledger has too many tasks ({len(raw_tasks)}, max {MAX_TASKS})")
    raw_versions = payload["versions"]
    if not isinstance(raw_versions, list):
        raise LedgerError(f"ledger 'versions' must be a list, got {_kind(raw_versions)}")
    if len(raw_versions) > MAX_VERSIONS:
        raise LedgerError(
            f"ledger has too many versions ({len(raw_versions)}, max {MAX_VERSIONS})"
        )

    tasks = []
    seen_names = set()
    for index, raw in enumerate(raw_tasks):
        task = _parse_task(raw, index)
        if task.name in seen_names:
            raise LedgerError(f"ledger task {_show(task.name)} is listed twice")
        seen_names.add(task.name)
        tasks.append(task)

    names = frozenset(seen_names)
    versions = []
    seen_labels = set()
    for index, raw in enumerate(raw_versions):
        version = _parse_version(raw, index, names)
        if version.label in seen_labels:
            raise LedgerError(f"ledger version label {_show(version.label)} is listed twice")
        seen_labels.add(version.label)
        versions.append(version)
    return Ledger(tuple(tasks), tuple(versions))


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------

def _enforce(path: str, field: str) -> None:
    """cwd containment + symlink refusal, as a ``LedgerError``."""
    try:
        enforce_under_cwd_and_no_symlink(path, field)
    except (ValueError, TypeError) as exc:
        raise LedgerError(str(exc)) from exc


def _open_regular(path: str, field: str):
    """Open ``path`` read-only, refusing a symlink and anything but a regular file.

    The caller bounds how much it reads; a FIFO or device would block that read.
    """
    try:
        fd = open_no_follow(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError as exc:
        raise LedgerError(
            f"{field} {quote_path(path)} cannot be opened: {type(exc).__name__}"
        ) from exc
    # Check the bare descriptor: on Linux a directory opens, and ``os.fdopen`` itself raises
    # IsADirectoryError, which would escape as a traceback instead of this message.
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise LedgerError(f"{field} {quote_path(path)} must be a regular file")
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "rb")


def read_ledger_text(path: str) -> str:
    """The ledger file's text: cwd-contained, symlink-refused, size-capped."""
    _enforce(path, "--ledger")
    if not os.path.lexists(path):
        raise LedgerNotFoundError(f"--ledger {quote_path(path)} does not exist")
    with _open_regular(path, "--ledger") as handle:
        data = handle.read(MAX_LEDGER_BYTES + 1)
    if len(data) > MAX_LEDGER_BYTES:
        raise LedgerError(f"--ledger {quote_path(path)} exceeds {MAX_LEDGER_BYTES} bytes")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LedgerError(f"--ledger {quote_path(path)} is not valid UTF-8") from exc


def _reject_constant(constant: str):
    raise ValueError(f"non-finite number {constant}")


def load_ledger(path: str) -> Ledger:
    """Read and validate the ledger at ``path``."""
    text = read_ledger_text(path)
    try:
        payload = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise LedgerError(f"--ledger {quote_path(path)} is not valid JSON: {exc}") from exc
    return parse_ledger(payload)


def serialize_ledger(ledger: Ledger) -> str:
    """The ledger as JSON text, refusing a document over the size cap."""
    text = json.dumps(ledger.to_dict(), indent=2, allow_nan=False) + "\n"
    if len(text.encode("utf-8")) > MAX_LEDGER_BYTES:
        raise LedgerError(
            f"the ledger would exceed {MAX_LEDGER_BYTES} bytes; start a new ledger file "
            "for a new series of versions"
        )
    return text


def save_ledger(ledger: Ledger, path: str) -> str:
    """Write the ledger atomically (temp file + ``os.replace``); a symlink is refused."""
    text = serialize_ledger(ledger)
    try:
        return atomic_write_text(text, path, field="--ledger")
    except ValueError as exc:
        raise LedgerError(str(exc)) from exc


def save_ledger_if_unchanged(
    ledger: Ledger, path: str, loaded: Optional[Ledger], *, label: str
) -> str:
    """Write ``ledger`` unless the file at ``path`` is no longer what this run read.

    ``loaded`` is the ledger as this run read it (``None`` when the file did not exist).
    A run takes minutes; another run may have recorded a version meanwhile, and writing
    the stale copy back would drop that version or let a duplicate label through. The
    file is re-read right before the write and compared. Two writers that both pass this
    check inside the same few milliseconds can still race: that window is documented
    as a known limitation rather than closed with a lock file a crash would leave behind.
    """
    try:
        current: Optional[Ledger] = load_ledger(path)
    except LedgerNotFoundError:
        current = None
    if current != loaded:
        if current is not None and current.has_label(label):
            raise LedgerChangedError(
                f"version label {_show(label)} was recorded by another run while this one "
                "was evaluating; nothing was recorded here (re-run with a new label)"
            )
        raise LedgerChangedError(
            f"--ledger {quote_path(path)} changed since this run read it (another run "
            "recorded a version or edited the file); nothing was recorded here. Re-run to "
            "compare against the current ledger."
        )
    return save_ledger(ledger, path)


# ---------------------------------------------------------------------------
# Control tasks
# ---------------------------------------------------------------------------

def read_control_bytes(path: str, field: str) -> bytes:
    """The bytes of a cwd-contained, symlink-refused, size-capped regular file, read once.

    A caller that must hash a file AND parse it hashes and parses these same bytes: a
    second open by path could see a different file (a swap and swap back in between).
    """
    _enforce(path, field)
    if not os.path.lexists(path):
        raise LedgerError(f"{field} {quote_path(path)} does not exist")
    with _open_regular(path, field) as handle:
        data = handle.read(MAX_CONTROL_FILE_BYTES + 1)
    if len(data) > MAX_CONTROL_FILE_BYTES:
        raise LedgerError(f"{field} {quote_path(path)} exceeds {MAX_CONTROL_FILE_BYTES} bytes")
    return data


def file_sha256(path: str, field: str) -> str:
    """SHA-256 of a cwd-contained, symlink-refused, size-capped regular file."""
    return hashlib.sha256(read_control_bytes(path, field)).hexdigest()


def relative_to_cwd(path: str) -> str:
    """``path`` relative to the working directory, with ``/`` separators."""
    try:
        rel = os.path.relpath(os.path.realpath(path), os.path.realpath(os.getcwd()))
    except ValueError:
        return path
    return rel.replace(os.sep, "/")


def parse_task_spec(spec: object) -> Tuple[str, str]:
    """Split an ``--add-task NAME=PATH`` value (the first ``=`` separates)."""
    if not isinstance(spec, str):
        raise LedgerError(f"--add-task must be NAME=PATH, got {_kind(spec)}")
    name, sep, path = spec.partition("=")
    if not sep or not path.strip():
        raise LedgerError(f"--add-task expects NAME=PATH, got {_show(spec)}")
    return validate_task_name(name), path


def changed_message(name: str, expected: str, actual: str, path: str) -> str:
    """The refusal for a control file whose bytes no longer match the ledger."""
    return (
        f"control task {_show(name)} changed: the ledger recorded sha256 {expected} but "
        f"{quote_path(path)} now hashes to {actual}. A silently swapped control set would "
        "fake retention; restore the original file, or register the new one under a new "
        "task name."
    )


def register_task(
    ledger: Ledger,
    name: str,
    path: str,
    *,
    label: str,
    now: str,
) -> Tuple[Ledger, str]:
    """Register ``name`` -> ``path`` in ``ledger``; return ``(ledger, status)``.

    ``status`` is ``added``, ``unchanged`` (same name, same bytes, same path) or
    ``moved`` (same name and bytes at a new path). A name that is already registered
    with DIFFERENT bytes is refused, naming both hashes.
    """
    name = validate_task_name(name)
    digest = file_sha256(path, f"--add-task {name} path")
    stored = validate_text(relative_to_cwd(path), "task path", max_chars=MAX_PATH_CHARS)
    existing = ledger.get_task(name)
    if existing is None:
        if len(ledger.tasks) >= MAX_TASKS:
            raise LedgerError(f"the ledger already holds {MAX_TASKS} tasks (the maximum)")
        task = RetentionTask(name, stored, digest, now, label)
        return replace(ledger, tasks=ledger.tasks + (task,)), "added"
    if existing.sha256 != digest:
        raise LedgerError(changed_message(name, existing.sha256, digest, path))
    if existing.path == stored:
        return ledger, "unchanged"
    tasks = tuple(
        replace(task, path=stored) if task.name == name else task for task in ledger.tasks
    )
    return replace(ledger, tasks=tasks), "moved"


def verify_ledger_files(ledger: Ledger) -> None:
    """Refuse a ledger whose control files are missing or no longer match their hash."""
    for task in ledger.tasks:
        field = f"control task {task.name} path"
        _enforce(task.path, field)
        if not os.path.lexists(task.path):
            raise LedgerError(
                f"control task {_show(task.name)} file {quote_path(task.path)} is missing; "
                f"if it moved, pass --add-task {task.name}=NEWPATH to update the stored path"
            )
        actual = file_sha256(task.path, field)
        if actual != task.sha256:
            raise LedgerError(changed_message(task.name, task.sha256, actual, task.path))


def append_version(ledger: Ledger, version: RetentionVersion) -> Ledger:
    """``ledger`` with ``version`` appended; a duplicate label or a full ledger is refused."""
    if ledger.has_label(version.label):
        raise LedgerError(
            f"version label {_show(version.label)} is already recorded; labels must be unique"
        )
    if len(ledger.versions) >= MAX_VERSIONS:
        raise LedgerError(f"the ledger already holds {MAX_VERSIONS} versions (the maximum)")
    return replace(ledger, versions=ledger.versions + (version,))


# ---------------------------------------------------------------------------
# Retention arithmetic
# ---------------------------------------------------------------------------

def _round(value: float) -> float:
    return round(value, SCORE_DECIMALS)


@dataclass(frozen=True)
class TaskRetention:
    """One task's row of the matrix: its history, the current score and the verdict."""

    name: str
    scores: Dict[str, float]  # version label -> score, ledger order, the candidate last
    current: float
    first: Optional[float]  # score at the first version that saw the task
    best_previous: Optional[float]  # best score over every recorded version
    drop: Optional[float]  # best_previous - current (negative = improved)
    status: str
    examples: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "scores": dict(self.scores),
            "current": self.current,
            "first": self.first,
            "best_previous": self.best_previous,
            "drop": self.drop,
            "status": self.status,
            "examples": self.examples,
        }


@dataclass(frozen=True)
class RetentionReport:
    """The candidate's matrix row set plus the series summary."""

    label: str
    threshold: float
    versions: Tuple[str, ...]  # earlier version labels, then the candidate's
    tasks: Tuple[TaskRetention, ...]

    @property
    def comparable(self) -> Tuple[TaskRetention, ...]:
        """Tasks that an earlier version also scored (a regression is possible)."""
        return tuple(row for row in self.tasks if row.status != STATUS_NEW)

    @property
    def regressions(self) -> Tuple[TaskRetention, ...]:
        return tuple(row for row in self.tasks if row.status == STATUS_REGRESSED)

    def mean_forgetting(self) -> Optional[float]:
        """Mean over comparable tasks of ``max(0, best - current)``; ``None`` if none."""
        rows = self.comparable
        if not rows:
            return None
        drops = [max(0.0, row.drop) for row in rows if row.drop is not None]
        return _round(sum(drops) / len(rows))

    def backward_transfer(self) -> Optional[float]:
        """Mean over comparable tasks of ``current - first score``; ``None`` if none.

        Negative means the old tasks ended below where they started. Tasks first
        scored by this version are excluded: they have no "before".
        """
        rows = self.comparable
        if not rows:
            return None
        moves = [row.current - row.first for row in rows if row.first is not None]
        return _round(sum(moves) / len(rows))

    def summary(self) -> Dict[str, object]:
        return {
            "tasks": len(self.tasks),
            "comparable_tasks": len(self.comparable),
            "new_tasks": len(self.tasks) - len(self.comparable),
            "regressions": len(self.regressions),
            "regressed_tasks": [row.name for row in self.regressions],
            "mean_forgetting": self.mean_forgetting(),
            "backward_transfer": self.backward_transfer(),
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": SCHEMA_VERSION,
            "label": self.label,
            "threshold": self.threshold,
            "versions": list(self.versions),
            "tasks": [row.to_dict() for row in self.tasks],
            "summary": self.summary(),
        }


def compute_retention(
    ledger: Ledger,
    current: Mapping[str, float],
    *,
    label: str,
    threshold: float,
    examples: Optional[Mapping[str, int]] = None,
) -> RetentionReport:
    """Judge the candidate's ``current`` scores against every recorded version.

    For each task: ``best_previous`` is the maximum over ALL recorded versions that
    scored it (not just the latest), ``drop = best_previous - current``, and the task
    regressed when ``drop > threshold``. A task no earlier version scored is ``new``:
    this run is its baseline entry and it cannot regress.
    """
    threshold = validate_threshold(threshold)
    label = validate_label(label)
    if ledger.has_label(label):
        raise LedgerError(
            f"version label {_show(label)} is already recorded; labels must be unique"
        )
    names = ledger.task_names()
    if set(current) != set(names):
        raise LedgerError("the candidate must be scored on exactly the ledger's tasks")
    counts = examples or {}

    rows = []
    for name in names:
        score = validate_score(current[name], f"score for task {_show(name)}")
        history = {
            version.label: version.scores[name]
            for version in ledger.versions
            if name in version.scores
        }
        first: Optional[float] = None
        best: Optional[float] = None
        drop: Optional[float] = None
        status = STATUS_NEW
        if history:
            past = list(history.values())
            first, best = past[0], max(past)
            drop = _round(best - score)
            status = STATUS_REGRESSED if drop > threshold else STATUS_OK
        rows.append(
            TaskRetention(
                name=name,
                scores={**history, label: _round(score)},
                current=_round(score),
                first=first,
                best_previous=best,
                drop=drop,
                status=status,
                examples=counts.get(name),
            )
        )
    return RetentionReport(
        label=label,
        threshold=threshold,
        versions=tuple(version.label for version in ledger.versions) + (label,),
        tasks=tuple(rows),
    )


def new_version(
    *,
    label: str,
    model: str,
    adapter: Optional[str],
    scores: Mapping[str, float],
    now: str,
    numerics: Numerics,
) -> RetentionVersion:
    """A ``RetentionVersion`` with its scores rounded the way the report shows them."""
    return RetentionVersion(
        label=validate_label(label),
        model=validate_model_ref(model, "--model"),
        adapter=validate_model_ref(adapter, "--adapter") if adapter is not None else None,
        evaluated_at=now,
        scores={name: _round(validate_score(value, name)) for name, value in scores.items()},
        numerics=numerics,
    )
