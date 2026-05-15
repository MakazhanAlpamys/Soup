"""Eval design from data — `soup eval design` (v0.55.0 Part A).

Builds an evaluation suite from a JSONL dataset + a one-line goal. CPU-only:
TF-IDF clustering for dimension discovery, heuristic categorisation for
scorer selection (rlvr / judge / exact_match / regex), and a goal-conditioned
rubric template per dimension.

Pure functions — no GPU, no network. Live LLM-judge prompts are emitted as
plain-text rubrics that `soup eval gate` can drive via the v0.19.0 backends.

Public surface
--------------
- Frozen dataclasses: ``EvalDimension``, ``EvalDesign``.
- Constants: ``SCORER_TYPES``.
- Pure functions: ``design_evals_from_data``, ``write_eval_design``,
  ``load_eval_design``.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from soup_cli.utils._eval_text import row_text as _row_text
from soup_cli.utils._eval_text import tokenize as _tokenize
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink, is_under_cwd

# Closed allowlist of scorer types. Frozenset for O(1) membership.
SCORER_TYPES: frozenset[str] = frozenset(
    {"exact_match", "regex", "judge", "rlvr"}
)

_MAX_ROWS = 1_000_000
_MAX_GOAL_CHARS = 4096
_MAX_DIMENSIONS = 20
_MIN_DIMENSIONS = 1
_MAX_NAME_CHARS = 64
_MAX_RUBRIC_CHARS = 4096
_MAX_FILE_BYTES = 16 * 1024 * 1024  # 16 MiB

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Heuristic keyword → scorer mapping. Order matters — earlier keys win.
_GOAL_KEYWORD_TO_SCORER: tuple[tuple[tuple[str, ...], str], ...] = (
    (("json", "schema", "structured"), "rlvr"),
    (("code", "python", "function", "script", "compile"), "rlvr"),
    (("math", "arithmetic", "compute", "calculate", "number"), "rlvr"),
    (("classify", "label", "category", "intent"), "exact_match"),
    (("extract", "field", "value"), "regex"),
    (("summari", "rewrite", "explain", "translate", "style", "concise"), "judge"),
)


@dataclass(frozen=True)
class EvalDimension:
    """One evaluation dimension — a name, a rubric, a scorer type.

    The dimension is intentionally portable: a downstream eval-gate runner
    consumes only ``name`` / ``scorer_type`` and either ``rubric`` (for
    judge) or the heuristic ``keywords`` (for exact/regex scoring).
    """

    name: str
    rubric: str
    scorer_type: str
    keywords: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvalDesign:
    """Output of ``design_evals_from_data``.

    Captures goal, row count, and the discovered dimensions. Serialises
    cleanly to JSON via ``asdict``.
    """

    goal: str
    row_count: int
    dimensions: tuple[EvalDimension, ...]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _require_str(value: object, *, field_name: str, max_len: int) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a string, got bool")
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string, got {type(value).__name__}"
        )
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    if len(value) > max_len:
        raise ValueError(f"{field_name} exceeds {max_len} characters")
    return value


def _normalize_goal(goal: object) -> str:
    text = _require_str(goal, field_name="goal", max_len=_MAX_GOAL_CHARS)
    return text.strip()


def _validate_num_dimensions(num: object) -> int:
    if isinstance(num, bool):
        raise TypeError("num_dimensions must be int, got bool")
    if not isinstance(num, int):
        raise TypeError(
            f"num_dimensions must be int, got {type(num).__name__}"
        )
    if num < _MIN_DIMENSIONS or num > _MAX_DIMENSIONS:
        raise ValueError(
            f"num_dimensions must be in [{_MIN_DIMENSIONS}, {_MAX_DIMENSIONS}]"
        )
    return num


# ---------------------------------------------------------------------------
# Term salience (TF-IDF over the output side of dataset rows)
# ---------------------------------------------------------------------------

# Subsample cap — _top_terms below sees at most this many rows before
# the document-frequency pass starts, defending against quadratic blow-up
# on a million-row JSONL.
_TOP_TERMS_SUBSAMPLE = 10_000


def _top_terms(
    rows: Sequence[Mapping[str, object]], *, k: int,
) -> list[str]:
    """Return up to ``k`` most-salient tokens across the output side.

    Uses a tiny TF-IDF: term frequency weighted by inverse document
    frequency (number of rows the term appears in). No external deps.
    """
    if k <= 0:
        return []
    doc_tokens: list[list[str]] = []
    # Materialise lazily but cap the scan at _TOP_TERMS_SUBSAMPLE to keep
    # design generation snappy on huge datasets.
    for i, row in enumerate(rows):
        if i >= _TOP_TERMS_SUBSAMPLE:
            break
        toks = _tokenize(_row_text(row))
        if toks:
            doc_tokens.append(toks)
    if not doc_tokens:
        return []
    n_docs = len(doc_tokens)
    df: Counter = Counter()
    tf: Counter = Counter()
    for toks in doc_tokens:
        seen = set(toks)
        for term in seen:
            df[term] += 1
        for term in toks:
            tf[term] += 1
    scored: list[tuple[str, float]] = []
    for term, freq in tf.items():
        idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
        scored.append((term, freq * idf))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in scored[:k]]


# ---------------------------------------------------------------------------
# Scorer + rubric heuristics
# ---------------------------------------------------------------------------

def _pick_scorer(goal_normalised: str) -> str:
    """Goal-keyword → default scorer; falls back to ``judge``."""
    goal_lower = goal_normalised.lower()
    for keywords, scorer in _GOAL_KEYWORD_TO_SCORER:
        if any(kw in goal_lower for kw in keywords):
            return scorer
    return "judge"


def _coerce_name(stem: str, *, fallback: str) -> str:
    candidate = re.sub(r"[^a-z0-9_]+", "_", stem.lower()).strip("_")
    if not candidate:
        candidate = fallback
    if candidate[0].isdigit():
        candidate = f"d_{candidate}"
    if len(candidate) > _MAX_NAME_CHARS:
        candidate = candidate[:_MAX_NAME_CHARS]
    if not _NAME_RE.match(candidate):
        candidate = fallback
    return candidate


def _build_rubric(goal: str, term: str, scorer: str) -> str:
    goal_clip = goal if goal else "the task"
    if scorer == "exact_match":
        body = (
            f"Answer must match the gold label exactly for the {term!r} class. "
            f"Goal: {goal_clip}."
        )
    elif scorer == "regex":
        body = (
            f"Answer must contain the {term!r} field value matching the "
            f"goal pattern. Goal: {goal_clip}."
        )
    elif scorer == "rlvr":
        body = (
            f"Answer must be verifiable on {term!r} (parse + run + assert). "
            f"Goal: {goal_clip}."
        )
    else:
        body = (
            f"Score 1 if the answer addresses {term!r} per the goal, "
            f"0 otherwise. Goal: {goal_clip}."
        )
    if len(body) > _MAX_RUBRIC_CHARS:
        body = body[: _MAX_RUBRIC_CHARS - 1] + "…"
    return body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def design_evals_from_data(
    rows: Sequence[Mapping[str, object]],
    *,
    goal: str,
    num_dimensions: int = 5,
) -> EvalDesign:
    """Produce an :class:`EvalDesign` for a dataset + goal.

    Heuristic — no GPU. The dimensions are derived from the top TF-IDF
    terms over the output side; the scorer is picked per goal-keyword
    map; rubrics are deterministic templates.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of mapping rows")
    if len(rows) > _MAX_ROWS:
        raise ValueError(f"rows exceed cap of {_MAX_ROWS}")
    goal_norm = _normalize_goal(goal)
    n_dims = _validate_num_dimensions(num_dimensions)

    scorer = _pick_scorer(goal_norm)
    terms = _top_terms(rows, k=n_dims)

    dimensions: list[EvalDimension] = []
    used_names: set = set()
    for idx, term in enumerate(terms):
        name = _coerce_name(term, fallback=f"dim_{idx + 1}")
        original = name
        suffix = 2
        while name in used_names:
            name = f"{original}_{suffix}"[:_MAX_NAME_CHARS]
            suffix += 1
        used_names.add(name)
        dimensions.append(
            EvalDimension(
                name=name,
                rubric=_build_rubric(goal_norm, term, scorer),
                scorer_type=scorer,
                keywords=(term,),
            )
        )
    # If the dataset produced no salient terms (empty rows), seed a single
    # goal-only dimension so downstream gate auto-install still has work.
    if not dimensions:
        dimensions.append(
            EvalDimension(
                name="goal_alignment",
                rubric=_build_rubric(goal_norm, "goal_alignment", scorer),
                scorer_type=scorer,
                keywords=tuple(),
            )
        )

    return EvalDesign(
        goal=goal_norm,
        row_count=len(rows),
        dimensions=tuple(dimensions),
    )


def design_to_dict(design: EvalDesign) -> dict[str, object]:
    """Pure JSON-friendly dict (tuples → lists)."""
    if not isinstance(design, EvalDesign):
        raise TypeError("design must be an EvalDesign instance")
    return {
        "goal": design.goal,
        "row_count": design.row_count,
        "dimensions": [
            {
                "name": d.name,
                "rubric": d.rubric,
                "scorer_type": d.scorer_type,
                "keywords": list(d.keywords),
            }
            for d in design.dimensions
        ],
    }


def write_eval_design(design: EvalDesign, output_path: str) -> str:
    """Atomic write — cwd containment + symlink rejection at target."""
    enforce_under_cwd_and_no_symlink(output_path, "output_path")
    payload = json.dumps(design_to_dict(design), ensure_ascii=False, indent=2)
    if len(payload.encode("utf-8")) > _MAX_FILE_BYTES:
        raise ValueError("rendered design exceeds 16 MiB cap")
    parent = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".soup-eval-design.", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, output_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return output_path


def load_eval_design(path: str) -> EvalDesign:
    """Read a design JSON back into :class:`EvalDesign`. Cwd-contained."""
    if not isinstance(path, str):
        raise TypeError("path must be str")
    if not path:
        raise ValueError("path must be non-empty")
    if "\x00" in path:
        raise ValueError("path must not contain NUL")
    if not is_under_cwd(path):
        raise ValueError("path must stay under cwd")
    # Unconditional lstat — closes the TOCTOU window where a symlink is
    # planted between an existence check and the open() call. Use the
    # lstat size for the cap check rather than os.path.getsize (which
    # follows symlinks and would silently follow a malicious link).
    try:
        st = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"eval design file not found: {os.path.basename(path)}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"path unreadable: {type(exc).__name__}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("path must not be a symlink (TOCTOU defence)")
    if st.st_size > _MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {_MAX_FILE_BYTES} byte cap")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("design JSON root must be an object")
    dims_raw = data.get("dimensions")
    if not isinstance(dims_raw, list):
        raise ValueError("design.dimensions must be a list")
    dims: list[EvalDimension] = []
    for entry in dims_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("each dimension must be an object")
        scorer = entry.get("scorer_type")
        if scorer not in SCORER_TYPES:
            raise ValueError(f"unknown scorer_type: {scorer!r}")
        name = entry.get("name")
        rubric = entry.get("rubric")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(f"invalid dimension name: {name!r}")
        if not isinstance(rubric, str):
            raise ValueError("dimension rubric must be string")
        keywords_raw = entry.get("keywords", [])
        if not isinstance(keywords_raw, list):
            raise ValueError("dimension keywords must be a list")
        keywords = tuple(
            k for k in keywords_raw if isinstance(k, str) and k
        )
        dims.append(
            EvalDimension(
                name=name,
                rubric=rubric,
                scorer_type=scorer,
                keywords=keywords,
            )
        )
    goal = data.get("goal", "")
    row_count = data.get("row_count", 0)
    if not isinstance(goal, str):
        raise ValueError("goal must be string")
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise ValueError("row_count must be int")
    return EvalDesign(goal=goal, row_count=row_count, dimensions=tuple(dims))
