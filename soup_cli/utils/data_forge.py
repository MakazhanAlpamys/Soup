"""v0.47.0 Part A — Synthetic Data Forge.

Multi-stage synthetic data pipeline: docs/traces → judge generation →
active selection (uncertainty-based pruning) → JSONL with full
provenance graph (which doc, which judge call, which filter score).

Differentiates from existing ``soup data generate`` + ``data augment``
(single-shot) by composing chunking, judge prompts, active pruning and
provenance into a single CLI surface.

Design notes:
- Pure-function math kernel (``chunk_document`` / ``score_uncertainty``)
  so the same routines feed the CLI, future ``soup eval`` integration,
  and (eventually) live trainer callbacks.
- The ``judge`` argument is a callable so callers can plug the v0.20.0
  Ollama / Anthropic / vLLM providers, or a stub in tests. No network or
  ML code lives in this module.
- Atomic write via staged-tempfile + ``os.replace`` (matches v0.43.0
  Part D ``copy_bundle_to`` policy).
- ``os.lstat + S_ISLNK`` rejection on write targets — defends against
  pre-placed symlinks pointing at ``/etc/cron.d/x`` etc. Matches
  v0.33.0 #22 / v0.43.0 Part C / v0.44.0 Part B / v0.45.0 Part E /
  v0.46.0 TOCTOU policy.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import stat as _stat
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from soup_cli.utils.paths import is_under_cwd

_LOG = logging.getLogger("soup.data_forge")

VALID_TASKS: Tuple[str, ...] = ("sft", "preference", "tool")
_VALID_TASKS_SET = frozenset(VALID_TASKS)

# Allowlist of document extensions. Mirrors v0.42.0 `data ingest` design
# intent — text-shaped corpora only. PDF/DOCX support intentionally lives
# in `soup data ingest`, which feeds JSONL into this pipeline.
_DOC_EXTENSIONS = frozenset({".txt", ".md", ".jsonl", ".json"})

# DoS caps. These match the spirit of v0.42.0 / v0.45.0 / v0.46.0 caps.
_MAX_DOCS = 10_000
_MAX_DOC_CHARS = 4 * 1024 * 1024  # 4 MiB per document
_MAX_TARGET_ROWS = 1_000_000
_MAX_TEACHER_LEN = 256
_MAX_PATH_LEN = 4096
_MAX_CHUNK_CHARS = 64_000


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgePlan:
    """Declarative plan for a forge run; nothing in here is executed."""

    task: str
    num_docs: int
    target_rows: int
    teacher: str = "local-judge"
    uncertainty_threshold: float = 0.0


@dataclass(frozen=True)
class ProvenanceRecord:
    """One row's audit trail back to source doc + judge call + score."""

    row_id: str
    source_doc: str
    judge_id: str
    filter_score: float
    chunk_id: str


@dataclass(frozen=True)
class ForgeRow:
    """One synthetic dataset row + its provenance."""

    messages: Tuple[Mapping[str, Any], ...]
    provenance: ProvenanceRecord
    task: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "messages": [dict(m) for m in self.messages],
            "task": self.task,
            "provenance": {
                "row_id": self.provenance.row_id,
                "source_doc": self.provenance.source_doc,
                "judge_id": self.provenance.judge_id,
                "filter_score": self.provenance.filter_score,
                "chunk_id": self.provenance.chunk_id,
            },
        }
        if self.extra:
            out["extra"] = dict(self.extra)
        return out


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _validate_str(value: Any, *, name: str, max_len: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{name} contains a null byte")
    if len(value) > max_len:
        raise ValueError(f"{name} exceeds {max_len} chars")
    return value


def _validate_int(value: Any, *, name: str, low: int, high: int) -> int:
    # bool is subclass of int — explicit reject (project bool-as-int policy).
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int (not bool)")
    if value < low or value > high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _validate_task(task: Any) -> str:
    if not isinstance(task, str):
        raise TypeError("task must be a string")
    if task not in _VALID_TASKS_SET:
        raise ValueError(
            f"task must be one of {sorted(_VALID_TASKS_SET)}; got {task!r}"
        )
    return task


def _validate_float_unit(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a float (not bool)")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{name} must be finite (NaN/Inf rejected)")
    if fv < 0.0 or fv > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return fv


# ---------------------------------------------------------------------------
# Pure-function kernel
# ---------------------------------------------------------------------------


_PARA_SPLIT = re.compile(r"\n\s*\n+")


def chunk_document(text: Any, *, max_chunk_chars: int = 1000) -> List[str]:
    """Split text into chunks no larger than ``max_chunk_chars``.

    Splits on paragraph boundaries first; falls back to hard slice when a
    paragraph alone exceeds the cap. Returns ``[]`` on empty/whitespace input.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if "\x00" in text:
        raise ValueError("text contains a null byte")
    if isinstance(max_chunk_chars, bool) or not isinstance(max_chunk_chars, int):
        raise TypeError("max_chunk_chars must be an int (not bool)")
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be positive")
    if max_chunk_chars > _MAX_CHUNK_CHARS:
        raise ValueError(f"max_chunk_chars exceeds {_MAX_CHUNK_CHARS}")
    if len(text) > _MAX_DOC_CHARS:
        raise ValueError(f"text exceeds {_MAX_DOC_CHARS} chars")
    stripped = text.strip()
    if not stripped:
        return []
    chunks: List[str] = []
    for para in _PARA_SPLIT.split(stripped):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chunk_chars:
            chunks.append(para)
            continue
        # Hard slice paragraphs that exceed the cap on their own.
        for i in range(0, len(para), max_chunk_chars):
            chunks.append(para[i : i + max_chunk_chars])
    return chunks


def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"\s+", text.strip().lower()) if t]


def score_uncertainty(prediction: Any, reference: Any) -> float:
    """Heuristic uncertainty score (Jaccard distance over token sets).

    Returns 0.0 when prediction == reference, 1.0 when disjoint or either is
    empty. Used for active-learning pruning — high score == high uncertainty,
    so the row is worth keeping.
    """
    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise TypeError("prediction and reference must be strings")
    a = set(_tokens(prediction))
    b = set(_tokens(reference))
    if not a or not b:
        return 1.0
    inter = a & b
    union = a | b
    return 1.0 - (len(inter) / len(union))


# ---------------------------------------------------------------------------
# Plan + discovery
# ---------------------------------------------------------------------------


def discover_documents(docs_dir: Any) -> List[str]:
    """Enumerate document files under ``docs_dir`` (one level deep).

    Skips dotfiles, rejects symlinked directories, restricts to known
    text extensions. Returns absolute realpaths.
    """
    if not isinstance(docs_dir, str):
        raise TypeError("docs_dir must be a string")
    if not docs_dir or "\x00" in docs_dir:
        raise ValueError("docs_dir must be a non-empty NUL-free string")
    if len(docs_dir) > _MAX_PATH_LEN:
        raise ValueError(f"docs_dir exceeds {_MAX_PATH_LEN} chars")
    if not is_under_cwd(docs_dir):
        raise ValueError("docs_dir must stay under cwd")
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"docs_dir not found: {docs_dir!r}")
    try:
        if _stat.S_ISLNK(os.lstat(docs_dir).st_mode):
            raise ValueError("docs_dir must not be a symlink")
    except OSError as exc:  # pragma: no cover — defensive
        raise ValueError(f"cannot stat docs_dir: {exc}") from exc

    real_base = os.path.realpath(docs_dir)
    out: List[str] = []
    try:
        with os.scandir(real_base) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_symlink():
                        continue
                except OSError:
                    continue
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in _DOC_EXTENSIONS:
                    continue
                out.append(os.path.realpath(entry.path))
                if len(out) >= _MAX_DOCS:
                    break
    except OSError as exc:
        raise ValueError(f"failed to scan docs_dir: {exc}") from exc
    out.sort()
    return out


def build_forge_plan(
    *,
    docs_dir: str,
    task: str,
    target_rows: int,
    teacher: str = "local-judge",
    uncertainty_threshold: float = 0.0,
) -> ForgePlan:
    """Validate inputs and return an immutable plan."""
    task = _validate_task(task)
    target_rows = _validate_int(
        target_rows, name="target_rows", low=1, high=_MAX_TARGET_ROWS
    )
    teacher = _validate_str(teacher, name="teacher", max_len=_MAX_TEACHER_LEN)
    uncertainty_threshold = _validate_float_unit(
        uncertainty_threshold, name="uncertainty_threshold"
    )

    if not isinstance(docs_dir, str) or not docs_dir:
        raise ValueError("docs_dir must be a non-empty string")
    if not is_under_cwd(docs_dir):
        raise ValueError("docs_dir must stay under cwd")
    docs = discover_documents(docs_dir)
    if not docs:
        raise ValueError(f"no documents found under {os.path.basename(docs_dir)!r}")

    return ForgePlan(
        task=task,
        num_docs=len(docs),
        target_rows=target_rows,
        teacher=teacher,
        uncertainty_threshold=uncertainty_threshold,
    )


# ---------------------------------------------------------------------------
# Synthesis (uses a caller-supplied judge function)
# ---------------------------------------------------------------------------


JudgeFn = Callable[[str], Mapping[str, Any]]


def _read_doc_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read(_MAX_DOC_CHARS + 1)
    except OSError:
        return ""


def _make_prompt(chunk: str, task: str) -> str:
    if task == "sft":
        return f"Read the passage and write a Q&A pair.\n\nPassage:\n{chunk}"
    if task == "preference":
        return (
            "Read the passage and write a preferred and a rejected answer.\n\n"
            f"Passage:\n{chunk}"
        )
    if task == "tool":
        return (
            "Read the passage and produce a tool-call hypothesis.\n\n"
            f"Passage:\n{chunk}"
        )
    raise ValueError(f"unknown task: {task!r}")  # pragma: no cover


def synthesise_forge_rows(
    docs: Sequence[str],
    *,
    task: str,
    target_rows: int,
    judge: JudgeFn,
    teacher: str = "local-judge",
    uncertainty_threshold: float = 0.0,
    max_chunk_chars: int = 1000,
) -> List[ForgeRow]:
    """Run the pipeline: chunk → judge → active-prune → ForgeRow rows.

    The ``judge`` callable is invoked once per chunk; failures are swallowed
    at DEBUG (mirrors v0.33.0 #47 CrossDocCollator policy — single bad call
    must not crash the run).
    """
    task = _validate_task(task)
    target_rows = _validate_int(
        target_rows, name="target_rows", low=1, high=_MAX_TARGET_ROWS
    )
    teacher = _validate_str(teacher, name="teacher", max_len=_MAX_TEACHER_LEN)
    uncertainty_threshold = _validate_float_unit(
        uncertainty_threshold, name="uncertainty_threshold"
    )
    if not callable(judge):
        raise TypeError("judge must be a callable")

    rows: List[ForgeRow] = []
    for doc_idx, doc_path in enumerate(docs):
        if len(rows) >= target_rows:
            break
        text = _read_doc_text(doc_path)
        if not text:
            continue
        try:
            chunks = chunk_document(text, max_chunk_chars=max_chunk_chars)
        except (TypeError, ValueError) as exc:
            _LOG.debug("chunk failed for %s: %s", doc_path, exc)
            continue

        for chunk_idx, chunk in enumerate(chunks):
            if len(rows) >= target_rows:
                break
            prompt = _make_prompt(chunk, task)
            try:
                reply = judge(prompt)
            except Exception as exc:  # noqa: BLE001 — judge backends vary
                _LOG.debug("judge raised on %s#%d: %s", doc_path, chunk_idx, exc)
                continue
            if not isinstance(reply, Mapping):
                continue
            reply_text = reply.get("text") or ""
            if not isinstance(reply_text, str):
                continue
            try:
                # Active pruning — Jaccard distance between chunk and reply.
                score = score_uncertainty(reply_text, chunk)
            except (TypeError, ValueError):
                continue
            if score < uncertainty_threshold:
                continue
            row_id = f"r{doc_idx:04d}_{chunk_idx:04d}"
            chunk_id = f"d{doc_idx:04d}_c{chunk_idx:04d}"
            messages: Tuple[Mapping[str, Any], ...] = (
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply_text},
            )
            prov = ProvenanceRecord(
                row_id=row_id,
                source_doc=doc_path,
                judge_id=teacher,
                filter_score=score,
                chunk_id=chunk_id,
            )
            rows.append(ForgeRow(messages=messages, provenance=prov, task=task))
    return rows


# ---------------------------------------------------------------------------
# Writers (atomic + symlink-rejecting)
# ---------------------------------------------------------------------------


def _check_write_path(path: Any, *, label: str) -> str:
    if not isinstance(path, str):
        raise TypeError(f"{label} must be a string")
    if not path or "\x00" in path:
        raise ValueError(f"{label} must be a non-empty NUL-free string")
    if len(path) > _MAX_PATH_LEN:
        raise ValueError(f"{label} exceeds {_MAX_PATH_LEN} chars")
    if not is_under_cwd(path):
        raise ValueError(f"{label} must stay under cwd")
    # TOCTOU: a pre-placed symlink at the target would let `os.replace`
    # follow into an attacker-controlled location.
    try:
        if _stat.S_ISLNK(os.lstat(path).st_mode):
            raise ValueError(f"{label} must not be a symlink")
    except FileNotFoundError:
        pass
    return os.path.realpath(path)


def _atomic_write(path: str, payload: bytes) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".forge-", dir=parent)
    try:
        # Symlink check at the staged path too — defence in depth against
        # tempfile dirs an attacker might race.
        if _stat.S_ISLNK(os.lstat(tmp_path).st_mode):
            os.close(fd)
            os.unlink(tmp_path)
            raise ValueError("staged tempfile is a symlink")
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup; never mask the original error.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_forge_dataset(rows: Sequence[ForgeRow], path: str) -> str:
    """Write rows as JSONL, atomically. Returns the realpath."""
    target = _check_write_path(path, label="dataset output")
    buf = []
    for row in rows:
        if not isinstance(row, ForgeRow):
            raise TypeError("rows must be ForgeRow instances")
        buf.append(json.dumps(row.to_dict(), ensure_ascii=False))
    payload = ("\n".join(buf) + ("\n" if buf else "")).encode("utf-8")
    _atomic_write(target, payload)
    return target


def write_provenance(rows: Sequence[ForgeRow], path: str) -> str:
    """Write provenance manifest JSON, atomically. Returns the realpath."""
    target = _check_write_path(path, label="provenance manifest")
    records = []
    for row in rows:
        if not isinstance(row, ForgeRow):
            raise TypeError("rows must be ForgeRow instances")
        prov = row.provenance
        records.append(
            {
                "row_id": prov.row_id,
                "source_doc": prov.source_doc,
                "judge_id": prov.judge_id,
                "filter_score": prov.filter_score,
                "chunk_id": prov.chunk_id,
            }
        )
    payload = json.dumps(
        {"version": 1, "row_count": len(records), "records": records},
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    _atomic_write(target, payload)
    return target


__all__ = [
    "VALID_TASKS",
    "ForgePlan",
    "ForgeRow",
    "ProvenanceRecord",
    "build_forge_plan",
    "chunk_document",
    "discover_documents",
    "score_uncertainty",
    "synthesise_forge_rows",
    "write_forge_dataset",
    "write_provenance",
]
