"""The per-row file of ``soup ship``: one JSON line per judged task row, saved next to ``--output``.

A SHIP is one number against another. This file is what the numbers are made of, so a reader can
audit a verdict row by row: ``<stem>.task-rows.jsonl`` for ``--output <stem>.<ext>``, and ONLY
with ``--output`` (``--emit-evidence`` alone writes nothing new). It is written by a live
``--task-mode metric`` or ``judge_score`` run; a replay of evidence, ``--task-mode pairwise``
and an adapter that changes nothing write none (``null`` in the verdict's ``task_rows``).

One line (compact JSON, ASCII only, ``\\n`` ends it, keys in this order, ``ROW_KEYS``)::

    {"schema_version":1,"leg":"task","index":2,"prompt_hash":"61a3385003d9","prompt":"What is
    2+2?","expected":"4","category":"default","scoring":"exact","base_answer":"3",
    "tuned_answer":"4","base_score":0.0,"tuned_score":1.0,"change":"better",
    "base_truncated":false,"tuned_truncated":null,"base_failed":false,"tuned_failed":false,
    "text_cut":false}

* ``index``: the row's 0-based place in the task file; the row id (unique in a file).
* ``prompt_hash``: the first 12 hex digits of ``sha256(prompt.strip().casefold())``, the key the
  loop's holdout split uses. Of the WHOLE prompt, also when ``text_cut``.
* ``scoring``: the task's scoring rule; ``null`` for ``judge_score`` (a judge scores, not a rule).
* ``base_score`` / ``tuned_score``: in [0, 1]. ``metric``: 1.0 when the answer matched, else 0.0
  (never the fractional similarity of ``semantic``). ``judge_score``: the judge's per-row score
  rescaled to [0, 1] on its rubric scale, ``(score - min) / (max - min)``, never clamped (a score
  outside the scale makes the file impossible, and it is then left out).
* ``change``: ``better`` / ``worse`` / ``same`` (``row_change``: tuned against base, 1e-9).
* ``base_truncated`` / ``tuned_truncated``: the answer stopped at the new-token cap, from the
  generator's token ids; ``null`` when not measured (a plain-text generator, a failed generation).
* ``base_failed`` / ``tuned_failed``: the generation raised and the row was scored as wrong (its
  answer is ``""``). Always ``false`` for ``judge_score``, where a failed generation ends the run.
* ``text_cut``: a text field was cut to ``MAX_TEXT_CHARS`` characters (a category to 256).

THE INVARIANT: ``row_mean`` of the ``base_score`` column, in ``index`` order, is ``task_win.base``
of the verdict and the same for ``tuned_score`` / ``task_win.tuned``: EXACTLY (bit for bit) for
``metric``, where the number is ``correct / total``; within ``JUDGE_MEAN_TOLERANCE`` (1e-12) for
``judge_score``, whose reported number is the evaluator's own (the rescaled mean of its raw scores,
as 0.77.x reported it) and may differ from the mean of the rescaled rows by a rounding. The
reported number is NEVER rebuilt from the rows: that moved it by one rounding and turned a
mathematical tie into a win. ``row_mean`` is a plain left-to-right sum divided by the count (no
``sum()``, whose float result changed in Python 3.12, and no ``fsum``), so a reader in any language
gets the same double. Scores are written with their shortest round-trip ``repr``.
``build_task_rows`` refuses legs where the rows do not add up to the reported number, so a file is
never written that cannot be audited against the verdict next to it.

The file holds the user's prompts and both models' answers: it is local data, and its text never
reaches the verdict JSON, the evidence, the terminal or the ``--push`` comment. On POSIX it is
created with mode 0600; on Windows it inherits the ACL of its folder. A run that is killed while
writing can leave a ``.soup.*.tmp`` temp file with partial rows next to it (the shared writer names
every command's temp files alike, so they are not removed here). It is replaced only by this
command's own next run: an existing file whose first line does not have exactly our keys is left
alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Tuple

from soup_cli.utils.paths import (
    atomic_write_bytes,
    enforce_under_cwd_and_no_symlink,
    is_network_or_device_path,
    is_under_cwd,
    quote_path,
)
from soup_cli.utils.ship_verdict import (
    CHANGE_BETTER,
    CHANGE_SAME,
    CHANGE_WORSE,
    MAX_TASK_ROWS,
    ROW_CHANGES,
    TASK_ROWS_SUFFIX,
    LegScore,
    row_change,
    task_rows_pointer_error,
)

SCHEMA_VERSION = 1
LEG = "task"
PROMPT_HASH_CHARS = 12

#: A text field is cut to this many characters (``text_cut`` says so); with ASCII escaping the
#: worst case is 12 bytes a character, so four fields still fit ``MAX_LINE_BYTES``.
MAX_TEXT_CHARS = 20_000
MAX_CATEGORY_CHARS = 256
MAX_LINE_BYTES = 1 << 20
MAX_FILE_BYTES = 256 << 20
#: The verdict / evidence JSON that names a rows file is read with the evidence bound.
MAX_POINTER_FILE_BYTES = 16 * 1024 * 1024

#: A judge leg's reported number is the evaluator's own; its rows add up to it within this.
JUDGE_MEAN_TOLERANCE = 1e-12

#: What a file name may hold: bytes on POSIX, UTF-16 code units on Windows.
_MAX_NAME_UNITS = 255

ROW_KEYS: Tuple[str, ...] = (
    "schema_version", "leg", "index", "prompt_hash", "prompt", "expected", "category", "scoring",
    "base_answer", "tuned_answer", "base_score", "tuned_score", "change", "base_truncated",
    "tuned_truncated", "base_failed", "tuned_failed", "text_cut",
)
_HASH_CHARS = frozenset("0123456789abcdef")


# ---------------------------------------------------------------------------
# The mean and the prompt key
# ---------------------------------------------------------------------------

def row_mean(scores: Iterable[object]) -> float:
    """The mean of per-row scores: a plain left-to-right sum divided by the count.

    The one definition of how a leg's number is made of its rows. Not ``sum()`` (compensated
    since Python 3.12, so its last bit depends on the version) and not ``math.fsum``: the plain
    loop is what any reader of the file computes in any language.
    """
    total = 0.0
    count = 0
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("a row score must be a number")
        try:
            finite = math.isfinite(score)
        except OverflowError:  # an integer too large for a float
            finite = False
        if not finite:
            raise ValueError("a row score must be finite")
        total += float(score)
        count += 1
    if count == 0:
        raise ValueError("a mean needs at least one score")
    return total / count


def prompt_hash(prompt: str) -> str:
    """The first 12 hex digits of ``sha256(prompt.strip().casefold())``: the loop's prompt key.

    The same rule as ``loop_holdout.key_digest(normalize(prompt))[:12]`` (``surrogatepass``: a
    lone surrogate in a prompt must not stop a verdict), written out here so ``soup ship`` does
    not import the loop's modules; a test pins the literal ``61a3385003d9`` for
    ``"  What is 2+2?  "``.
    """
    key = prompt.strip().casefold()
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()[:PROMPT_HASH_CHARS]


# ---------------------------------------------------------------------------
# Where the file goes
# ---------------------------------------------------------------------------

def task_rows_path(anchor: str) -> str:
    """The rows file of ``anchor`` (the ``--output`` path): its stem + suffix.

    In the very directory the user named (the text up to the last component is unchanged).
    Raises ``ValueError`` naming the reason when the anchor cannot carry the file: the name would
    not be a bare file name the verdict can point at (a control character, a path separator in a
    name), the derived name cannot exist (more than 255 bytes on POSIX, 255 UTF-16 code units on
    Windows: what the file system counts, not characters), or the path is a network share or a
    device. An ``--output`` whose own name is within the limit but whose derived name is not (the
    suffix adds 16) is refused: that file cannot be created.
    """
    if not isinstance(anchor, str) or not anchor:
        raise ValueError("the file the task rows sit next to must be a non-empty path")
    if "\x00" in anchor:
        raise ValueError("the file the task rows sit next to must not contain null bytes")
    if is_network_or_device_path(anchor):
        raise ValueError(
            f"{quote_path(anchor)} names a network share or a device; "
            "the task-rows file cannot be written there"
        )
    path = os.path.splitext(anchor)[0] + TASK_ROWS_SUFFIX
    name = os.path.basename(path)
    if any(ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in name):
        raise ValueError(
            f"the task-rows file name for {quote_path(anchor)} has a control character"
        )
    if "\\" in name or "/" in name:
        raise ValueError(
            f"the task-rows file name for {quote_path(anchor)} would contain a path separator"
        )
    if os.name == "nt":
        size, unit = len(name.encode("utf-16-le", "surrogatepass")) // 2, "UTF-16 units"
    else:
        size, unit = len(name.encode("utf-8", "surrogatepass")), "bytes"
    if size > _MAX_NAME_UNITS:
        raise ValueError(
            f"the task-rows file name for {quote_path(anchor)} is too long "
            f"({size} {unit}; at most {_MAX_NAME_UNITS} are allowed)"
        )
    reason = task_rows_pointer_error(name)
    if reason is not None:
        raise ValueError(f"the task-rows file name for {quote_path(anchor)} {reason}")
    return path


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskRow:
    """One judged task row, as it goes into the file."""

    index: int
    prompt_hash: str
    prompt: str
    expected: str
    category: str
    scoring: Optional[str]
    base_answer: str
    tuned_answer: str
    base_score: float
    tuned_score: float
    change: str
    base_truncated: Optional[bool]
    tuned_truncated: Optional[bool]
    base_failed: bool
    tuned_failed: bool
    text_cut: bool = False


def _cut(text: str, limit: int) -> Tuple[str, bool]:
    return (text, False) if len(text) <= limit else (text[:limit], True)


def _unit_score(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= value <= 1.0:  # an integer of any size compares without overflowing
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def _columns(leg: LegScore, side: str) -> Tuple[tuple, tuple, tuple, tuple, tuple]:
    """A leg's per-row columns (scores, answers, failure flags, truncation flags, tasks)."""
    if leg.row_scores is None:
        raise ValueError(f"the {side} leg carries no row scores")
    if leg.answers is None or leg.failed_rows is None or leg.tasks is None:
        raise ValueError(f"the {side} leg carries no answers or tasks to write")
    count = len(leg.row_scores)
    flags = leg.truncated if leg.truncated is not None else (None,) * count
    columns = (leg.row_scores, leg.answers, leg.failed_rows, flags, leg.tasks)
    if any(len(column) != count for column in columns):
        raise ValueError(f"the {side} leg's columns must all have the same number of rows")
    return columns


def build_task_rows(
    base_leg: object, tuned_leg: object, *, scoring_applies: bool
) -> Tuple[TaskRow, ...]:
    """The rows of a task leg from its two ``LegScore`` s; refuses legs that cannot be audited.

    ``scoring_applies`` is True when a rule scored the rows (``metric``) and False for a judge.
    The reported number of each leg must be ``row_mean`` of its rows: exactly for a rule, within
    ``JUDGE_MEAN_TOLERANCE`` for a judge (whose number is the evaluator's own). When it is not
    (an evaluator whose overall score is not the mean of its rows) nothing is built, so a file
    is never written that does not add up to the verdict it sits next to.
    """
    if not isinstance(base_leg, LegScore) or not isinstance(tuned_leg, LegScore):
        raise TypeError("both legs need LegScore objects carrying their rows")
    base_scores, base_answers, base_failed, base_flags, base_tasks = _columns(base_leg, "base")
    tuned_scores, tuned_answers, tuned_failed, tuned_flags, tuned_tasks = _columns(
        tuned_leg, "tuned"
    )
    count = len(base_scores)
    if count != len(tuned_scores):
        raise ValueError("the base and tuned legs need the same number of rows")
    if not 1 <= count <= MAX_TASK_ROWS:
        raise ValueError(f"a task leg has at most {MAX_TASK_ROWS} rows (and at least 1)")

    base_unit = [_unit_score(v, f"base row score[{i}]") for i, v in enumerate(base_scores)]
    tuned_unit = [_unit_score(v, f"tuned row score[{i}]") for i, v in enumerate(tuned_scores)]
    tolerance = 0.0 if scoring_applies else JUDGE_MEAN_TOLERANCE
    for side, unit, leg in (("base", base_unit, base_leg), ("tuned", tuned_unit, tuned_leg)):
        mean = row_mean(unit)
        if not abs(mean - float(leg)) <= tolerance:  # written so that a NaN is refused too
            raise ValueError(
                f"the {side} row scores average {mean!r}, not the reported {float(leg)!r}"
            )

    rows: List[TaskRow] = []
    for index in range(count):
        base_task, tuned_task = base_tasks[index], tuned_tasks[index]
        prompt = getattr(base_task, "prompt", None)
        if not isinstance(prompt, str) or prompt != getattr(tuned_task, "prompt", None):
            raise ValueError("the base and tuned legs are for different tasks")
        answers = (base_answers[index], tuned_answers[index])
        if not all(isinstance(answer, str) for answer in answers):
            raise TypeError("an answer must be text")
        failed = (base_failed[index], tuned_failed[index])
        if not all(isinstance(flag, bool) for flag in failed):
            raise TypeError("a failure flag must be True or False")
        expected = str(getattr(base_task, "expected", ""))
        scoring = str(getattr(base_task, "scoring", "exact")) if scoring_applies else None
        category = str(getattr(base_task, "category", "default"))
        texts = [_cut(text, MAX_TEXT_CHARS) for text in (prompt, expected, *map(str, answers))]
        category, category_cut = _cut(category, MAX_CATEGORY_CHARS)
        rows.append(TaskRow(
            index=index,
            prompt_hash=prompt_hash(prompt),
            prompt=texts[0][0],
            expected=texts[1][0],
            category=category,
            scoring=scoring,
            base_answer=texts[2][0],
            tuned_answer=texts[3][0],
            base_score=base_unit[index],
            tuned_score=tuned_unit[index],
            change=row_change(base_unit[index], tuned_unit[index]),
            base_truncated=base_flags[index],
            tuned_truncated=tuned_flags[index],
            base_failed=failed[0],
            tuned_failed=failed[1],
            text_cut=category_cut or any(cut for _, cut in texts),
        ))
    return tuple(rows)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _row_dict(row: TaskRow) -> dict:
    values = {
        "schema_version": SCHEMA_VERSION, "leg": LEG, "index": row.index,
        "prompt_hash": row.prompt_hash, "prompt": row.prompt, "expected": row.expected,
        "category": row.category, "scoring": row.scoring, "base_answer": row.base_answer,
        "tuned_answer": row.tuned_answer, "base_score": float(row.base_score),
        "tuned_score": float(row.tuned_score), "change": row.change,
        "base_truncated": row.base_truncated, "tuned_truncated": row.tuned_truncated,
        "base_failed": row.base_failed, "tuned_failed": row.tuned_failed,
        "text_cut": row.text_cut,
    }
    return {key: values[key] for key in ROW_KEYS}


def rows_to_jsonl(rows: Iterable[TaskRow]) -> str:
    """The file's text: one compact ASCII JSON line per row, each ended by ``\\n`` (never CRLF).

    ASCII escaping keeps a row on one line whatever its text holds (U+2028, a lone surrogate, a
    control byte). Scores are written with their shortest round-trip ``repr``; a NaN or an
    infinity is refused, never written.
    """
    listed = list(rows)
    if not listed:
        raise ValueError("a task-rows file needs at least one row")
    if len(listed) > MAX_TASK_ROWS:
        raise ValueError(f"a task-rows file has at most {MAX_TASK_ROWS} rows")
    lines: List[str] = []
    total = 0
    for row in listed:
        line = json.dumps(
            _row_dict(row), ensure_ascii=True, separators=(",", ":"), allow_nan=False
        )
        if len(line) > MAX_LINE_BYTES:
            raise ValueError(f"a row line is longer than {MAX_LINE_BYTES} bytes")
        total += len(line) + 1
        if total > MAX_FILE_BYTES:
            raise ValueError(f"the task-rows file would be more than {MAX_FILE_BYTES} bytes")
        lines.append(line)
    return "\n".join(lines) + "\n"


def _first_line_is_ours(first: bytes) -> bool:
    """Whether a first line has exactly the keys of a row of ours (a BOM and CRLF are fine)."""
    if first.startswith(b"\xef\xbb\xbf"):
        first = first[3:]
    try:
        entry = json.loads(first.rstrip(b"\r\n").decode("utf-8"))
    except (ValueError, RecursionError):  # UnicodeDecodeError is a ValueError
        return False
    return (
        isinstance(entry, dict)
        and set(entry) == set(ROW_KEYS)
        and entry["leg"] == LEG
        and isinstance(entry["schema_version"], int)
        and not isinstance(entry["schema_version"], bool)
        and entry["schema_version"] == SCHEMA_VERSION
    )


def _is_ours_or_absent(path: str) -> bool:
    """True when nothing is at ``path``, or an empty file, or a file whose first line is ours.

    Raises ``ValueError`` (a named refusal, never a traceback) when ``path`` cannot be looked at: a
    parent that is a regular file, a directory that cannot be read.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise ValueError(
            f"{quote_path(path)} cannot be checked ({type(exc).__name__}); "
            "the task rows are not written there"
        ) from None
    if not stat.S_ISREG(info.st_mode):
        return False
    if info.st_size == 0:
        return True
    try:
        with open(path, "rb") as handle:
            first = handle.readline(MAX_LINE_BYTES + 1)
    except OSError as exc:
        raise ValueError(
            f"{quote_path(path)} cannot be checked ({type(exc).__name__}); "
            "the task rows are not written there"
        ) from None
    return _first_line_is_ours(first)


def ensure_replaceable(path: str) -> None:
    """Raise ``ValueError`` when ``path`` holds a file that is not one of ours.

    Only a path under the working directory is looked at (the writer refuses any other).
    """
    if is_under_cwd(path) and not _is_ours_or_absent(path):
        raise ValueError(
            f"{quote_path(path)} exists and is not a task-rows file written by soup ship; "
            "it is left alone (move it, or choose another file name)"
        )


def write_task_rows(rows: Iterable[TaskRow], anchor: str) -> str:
    """Write the rows next to ``anchor``; returns the path written (as typed, plus the suffix).

    Atomically (a temp file in the same directory, then ``os.replace``: a failure leaves an
    earlier file whole), under the working directory, never through a symlink, with LF line
    endings on every platform. An existing file is replaced only if it is empty or its first
    line has exactly the keys of a row of ours; anything else is left alone and refused. Raises
    ``ValueError`` / ``OSError``.
    """
    path = task_rows_path(anchor)
    enforce_under_cwd_and_no_symlink(path, "task rows output")
    data = rows_to_jsonl(rows).encode("ascii")
    ensure_replaceable(path)
    atomic_write_bytes(data, path, field="task rows output")
    return path


# ---------------------------------------------------------------------------
# Reading it back, strictly
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedTaskRows:
    """A validated rows file: its rows in index order, their means and class counts."""

    rows: Tuple[Mapping[str, object], ...]
    base_mean: float
    tuned_mean: float
    better: int
    worse: int
    same: int


class _LineError(ValueError):
    """A defect of one line, named (the loader adds the line number)."""


def _no_duplicate_keys(pairs: List[Tuple[str, object]]) -> dict:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise _LineError("repeats a key")
    return dict(pairs)


def _refuse_constant(token: str) -> object:
    raise _LineError("has a non-finite number (NaN or Infinity)")


def _text(entry: Mapping[str, object], key: str, *, nullable: bool = False) -> None:
    value = entry[key]
    if value is None and nullable:
        return
    if not isinstance(value, str):
        raise _LineError(f"{key} must be a string" + (" or null" if nullable else ""))


def _unit(entry: Mapping[str, object], key: str) -> float:
    value = entry[key]
    if isinstance(value, bool):
        raise _LineError(f"{key} must be a number, not a boolean")
    if not isinstance(value, (int, float)):
        raise _LineError(f"{key} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise _LineError(f"{key} must be finite")
    if not 0.0 <= value <= 1.0:  # an integer of any size compares without overflowing
        raise _LineError(f"{key} must be in [0, 1]")
    return float(value)


def _check_row(entry: Mapping[str, object]) -> None:
    unknown = sorted(str(key) for key in entry if key not in ROW_KEYS)
    if unknown:
        # at most five names of at most 64 characters, and the count of the rest: a line with
        # thousands of keys must not turn into a message of the same size
        joined = ", ".join(repr(key[:64]) for key in unknown[:5])
        more = f" and {len(unknown) - 5} more" if len(unknown) > 5 else ""
        raise _LineError(f"has unsupported field(s): {joined}{more}")
    missing = [key for key in ROW_KEYS if key not in entry]
    if missing:
        raise _LineError(f"needs {', '.join(missing)}")
    version = entry["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise _LineError(f"schema_version must be {SCHEMA_VERSION}")
    if entry["leg"] != LEG:
        raise _LineError(f'leg must be "{LEG}"')
    index = entry["index"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_TASK_ROWS:
        raise _LineError(f"index must be an integer in [0, {MAX_TASK_ROWS})")
    for key in ("prompt", "expected", "category", "base_answer", "tuned_answer"):
        _text(entry, key)
    _text(entry, "scoring", nullable=True)
    base, tuned = _unit(entry, "base_score"), _unit(entry, "tuned_score")
    change = entry["change"]
    if change not in ROW_CHANGES:
        raise _LineError(f"change must be one of {', '.join(ROW_CHANGES)}")
    if change != row_change(base, tuned):
        raise _LineError(f"change says {change} but the scores say {row_change(base, tuned)}")
    for key in ("base_truncated", "tuned_truncated"):
        if entry[key] is not None and not isinstance(entry[key], bool):
            raise _LineError(f"{key} must be true, false or null")
    for key in ("base_failed", "tuned_failed", "text_cut"):
        if not isinstance(entry[key], bool):
            raise _LineError(f"{key} must be true or false")
    digest = entry["prompt_hash"]
    if (
        not isinstance(digest, str)
        or len(digest) != PROMPT_HASH_CHARS
        or not set(digest) <= _HASH_CHARS
    ):
        raise _LineError(f"prompt_hash must be {PROMPT_HASH_CHARS} lowercase hex digits")
    if not entry["text_cut"] and digest != prompt_hash(entry["prompt"]):
        raise _LineError("prompt_hash does not match the prompt")


def _parse_line(content: bytes, number: int) -> dict:
    where = f"task-rows line {number}"
    if not content.strip():
        raise ValueError(f"{where} is blank")
    try:
        entry = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_refuse_constant,
        )
    except _LineError as exc:
        raise ValueError(f"{where} {exc}") from None
    except (ValueError, RecursionError):  # not JSON, not UTF-8, or nested past the parser's depth
        raise ValueError(f"{where} is not valid JSON") from None
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a JSON object")
    try:
        _check_row(entry)
    except _LineError as exc:
        raise ValueError(f"{where} {exc}") from None
    return entry


def load_task_rows(
    path: str,
    *,
    n: Optional[int] = None,
    base: Optional[float] = None,
    tuned: Optional[float] = None,
    counts: Optional[Mapping[str, int]] = None,
    mean_tolerance: float = 0.0,
) -> LoadedTaskRows:
    """Read a rows file strictly; raises ``ValueError`` naming the first defect.

    Refused: a line that is blank, longer than ``MAX_LINE_BYTES``, not JSON (nested past the
    parser's depth included), not an object, with a repeated or unknown or missing key, a
    ``NaN`` / ``Infinity`` token, a boolean or string where a number belongs, a score outside
    [0, 1], a ``change`` that the scores contradict, a ``prompt_hash`` that is not the prompt's,
    a repeated or out-of-range row id (``index``), more than ``MAX_TASK_ROWS`` rows, a file over
    ``MAX_FILE_BYTES``. Given the numbers of the verdict or evidence that names the file: a row
    count that is not ``n``, a column mean that is not ``base`` / ``tuned`` (EXACTLY, or within
    ``mean_tolerance``: pass ``JUDGE_MEAN_TOLERANCE`` for a ``judge_score`` verdict), class counts
    that are not ``counts`` (``better`` / ``worse`` / ``same``). A message names the defect and
    the line; it echoes no value of the file, only at most five key names for unsupported fields.
    """
    try:
        info = os.stat(path)
    except FileNotFoundError:
        raise ValueError("task-rows file not found") from None
    except (OSError, ValueError) as exc:
        raise ValueError(f"task-rows file cannot be read: {type(exc).__name__}") from None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("task-rows path is not a regular file")
    if info.st_size > MAX_FILE_BYTES:
        raise ValueError(f"task-rows file is larger than {MAX_FILE_BYTES} bytes")
    if info.st_size == 0:
        raise ValueError("task-rows file is empty")

    entries: List[Tuple[int, dict]] = []
    seen = set()
    try:
        handle = open(path, "rb")
    except OSError as exc:
        raise ValueError(f"task-rows file cannot be read: {type(exc).__name__}") from None
    with handle:
        number = 0
        while True:
            raw = handle.readline(MAX_LINE_BYTES + 2)
            if not raw:
                break
            number += 1
            content = raw[:-1] if raw.endswith(b"\n") else raw
            if len(content) > MAX_LINE_BYTES:
                raise ValueError(f"task-rows line {number} is longer than {MAX_LINE_BYTES} bytes")
            if len(entries) >= MAX_TASK_ROWS:
                raise ValueError(f"task-rows has more than {MAX_TASK_ROWS} rows")
            entry = _parse_line(content, number)
            if entry["index"] in seen:
                raise ValueError(f"task-rows line {number} repeats row id {entry['index']}")
            seen.add(entry["index"])
            entries.append((number, entry))

    count = len(entries)
    for number, entry in entries:
        if entry["index"] >= count:
            raise ValueError(
                f"task-rows line {number} index {entry['index']} is outside [0, {count})"
            )
    if n is not None and count != n:
        raise ValueError(f"task-rows has {count} rows but the evidence says n = {n}")
    rows = tuple(sorted((entry for _, entry in entries), key=lambda entry: entry["index"]))
    base_mean = row_mean(row["base_score"] for row in rows)
    tuned_mean = row_mean(row["tuned_score"] for row in rows)
    if base is not None and not abs(base_mean - base) <= mean_tolerance:
        raise ValueError(f"the base_score mean {base_mean!r} is not the reported base {base!r}")
    if tuned is not None and not abs(tuned_mean - tuned) <= mean_tolerance:
        raise ValueError(
            f"the tuned_score mean {tuned_mean!r} is not the reported tuned {tuned!r}"
        )
    seen_changes = [row["change"] for row in rows]
    better = seen_changes.count(CHANGE_BETTER)
    worse = seen_changes.count(CHANGE_WORSE)
    same = seen_changes.count(CHANGE_SAME)
    if counts is not None and (better, worse, same) != (
        counts.get("better"), counts.get("worse"), counts.get("same"),
    ):
        raise ValueError(
            f"the rows are better {better}, worse {worse}, same {same}, which is not the "
            f"evidence rows block better {counts.get('better')}, worse {counts.get('worse')}, "
            f"same {counts.get('same')}"
        )
    return LoadedTaskRows(rows, base_mean, tuned_mean, better, worse, same)


def _read_json_object(path: str) -> dict:
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_POINTER_FILE_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"the verdict or evidence file cannot be read: {type(exc).__name__}")
    if len(data) > MAX_POINTER_FILE_BYTES:
        raise ValueError(
            f"the verdict or evidence file is larger than {MAX_POINTER_FILE_BYTES} bytes"
        )
    try:
        payload = json.loads(data.decode("utf-8"), parse_constant=_refuse_constant)
    except (ValueError, RecursionError):  # _LineError is a ValueError too
        raise ValueError("the verdict or evidence file is not valid JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("the verdict or evidence file must be a JSON object")
    return payload


def _reported(block: Mapping[str, object], key: str) -> float:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"the reported {key} must be a finite number")
    try:
        number = float(value)
    except OverflowError:  # an integer too large for a float
        raise ValueError(f"the reported {key} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"the reported {key} must be a finite number")
    return number


def _rows_block(rows: Mapping[str, object]) -> Mapping[str, int]:
    """The ``rows`` block of a verdict or evidence file, every number an integer in range."""
    out = {}
    for key in ("better", "worse", "same", "n"):
        value = rows.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_TASK_ROWS:
            raise ValueError(
                "the rows block of the verdict or evidence file must hold integers in "
                f"[0, {MAX_TASK_ROWS}] for better, worse, same and n"
            )
        out[key] = value
    return out


def load_task_rows_for(json_path: str) -> LoadedTaskRows:
    """Load the rows file a verdict (``--output``) or an evidence file names, and check it.

    The ``task_rows`` pointer is a bare name resolved next to ``json_path`` (it cannot point
    anywhere else). The file must then agree with the numbers of the JSON that names it: the row
    count is ``task_win.rows.n`` / ``task.rows.n``, the column means are ``base`` / ``tuned``
    (exactly, except for a ``judge_score`` verdict: within ``JUDGE_MEAN_TOLERANCE``), the class
    counts are the ``rows`` block.
    """
    payload = _read_json_object(json_path)
    pointer = payload.get("task_rows")
    if pointer is None:
        raise ValueError("this file names no task-rows file (task_rows is absent or null)")
    problem = task_rows_pointer_error(pointer)
    if problem is not None:
        raise ValueError(f"task_rows {problem}")
    directory = os.path.dirname(os.path.realpath(json_path))
    target = os.path.realpath(os.path.join(directory, pointer))
    if os.path.normcase(os.path.dirname(target)) != os.path.normcase(directory):
        raise ValueError("task_rows resolves outside the directory of the file that names it")
    if not os.path.exists(target):
        raise ValueError(
            "task-rows file not found next to the verdict or evidence file that names it"
        )
    block = payload.get("task_win") if isinstance(payload.get("task_win"), dict) else None
    if block is None:
        block = payload.get("task") if isinstance(payload.get("task"), dict) else None
    if block is None:
        raise ValueError("the file has no task_win or task block to check the rows against")
    rows = _rows_block(block["rows"]) if isinstance(block.get("rows"), dict) else None
    return load_task_rows(
        target,
        n=None if rows is None else rows["n"],
        base=_reported(block, "base"),
        tuned=_reported(block, "tuned"),
        counts=rows,
        mean_tolerance=JUDGE_MEAN_TOLERANCE if block.get("mode") == "judge_score" else 0.0,
    )


__all__ = [
    "JUDGE_MEAN_TOLERANCE",
    "LEG",
    "MAX_FILE_BYTES",
    "MAX_LINE_BYTES",
    "MAX_POINTER_FILE_BYTES",
    "MAX_TEXT_CHARS",
    "ROW_KEYS",
    "SCHEMA_VERSION",
    "LoadedTaskRows",
    "TaskRow",
    "build_task_rows",
    "ensure_replaceable",
    "load_task_rows",
    "load_task_rows_for",
    "prompt_hash",
    "row_mean",
    "rows_to_jsonl",
    "task_rows_path",
    "write_task_rows",
]
