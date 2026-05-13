"""v0.47.0 Part B — Data Quality Moat.

Composite data-quality scorecard. Lighter-weight alternative to
Argilla/Cleanlab — pure-Python heuristics that work without GPUs or
200 MB Presidio models; heavy classifiers gated behind ``[data-pro]``
extras (deferred until v0.47.1).

Pieces:
- benchmark decontamination via n-gram overlap (MMLU/GSM8K/HumanEval)
- PII detection via narrow regex set (email/phone/SSN/credit-card)
- language detection via small character-frequency heuristic, with
  optional ``langdetect`` fallback for Windows users
- toxicity scoring via keyword baseline (a small Llama-Guard variant is
  the v0.47.1 follow-up; for now we ship a fast, dep-free heuristic)
- educational-value score via length + lexical-diversity proxy

The CLI surface keeps every subcommand small and JSONL-in / JSONL-out so
operators can compose them.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import stat as _stat
import tempfile
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from soup_cli.utils.paths import is_under_cwd

_LOG = logging.getLogger("soup.data_score")

_MAX_TEXT_CHARS = 1_000_000
_MAX_ROWS = 1_000_000
_MAX_FILE_BYTES = 1024 * 1024 * 1024  # 1 GiB
_MAX_PATH_LEN = 4096

# Closed allowlist for `--benchmarks`. Live n-gram corpora for these
# benchmarks ship in v0.47.1; for now we accept caller-supplied texts.
BENCHMARKS: Mapping[str, str] = MappingProxyType(
    {
        "mmlu": "MMLU multiple-choice questions",
        "gsm8k": "GSM8K grade-school math problems",
        "humaneval": "HumanEval Python coding prompts",
        "truthfulqa": "TruthfulQA short-answer probes",
        "arc": "ARC commonsense science questions",
        "hellaswag": "HellaSwag sentence-completion benchmark",
    }
)


@dataclass(frozen=True)
class ScoreReport:
    total: int
    pii_flagged: int
    toxic_flagged: int
    decontaminated_removed: int
    languages: Mapping[str, int]
    educational_mean: float


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _require_str(value: Any, *, name: str, max_len: int = _MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if "\x00" in value:
        raise ValueError(f"{name} contains a null byte")
    if len(value) > max_len:
        raise ValueError(f"{name} exceeds {max_len} chars")
    return value


def _require_int(value: Any, *, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int (not bool)")
    if value < low or value > high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _require_unit_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a float (not bool)")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    fv = float(value)
    if not math.isfinite(fv) or fv < 0.0 or fv > 1.0:
        raise ValueError(f"{name} must be in [0, 1] and finite")
    return fv


# ---------------------------------------------------------------------------
# N-gram + decontamination
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"\w+")


def _tokenise(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def ngram_set(text: Any, *, n: int = 8) -> Set[Tuple[str, ...]]:
    """Return the set of n-grams (length-n tuples) in ``text``.

    Empty result when the token count is < n. Capped on ``text`` size.
    """
    s = _require_str(text, name="text")
    nv = _require_int(n, name="n", low=1, high=32)
    tokens = _tokenise(s)
    if len(tokens) < nv:
        return set()
    return {tuple(tokens[i : i + nv]) for i in range(len(tokens) - nv + 1)}


def ngram_overlap_ratio(a: Any, b: Any, *, n: int = 8) -> float:
    """Containment ratio: fraction of b's n-grams present in a.

    NOTE: this is one-sided recall ``|inter| / |b|``, not symmetric
    Jaccard. ``decontaminate_rows`` reuses this denominator so that a
    short user row reliably triggers when it covers a benchmark
    fragment regardless of how long the row itself is.
    """
    sa = ngram_set(a, n=n)
    sb = ngram_set(b, n=n)
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    return len(inter) / len(sb)


def _extract_row_text(row: Any) -> str:
    if not isinstance(row, Mapping):
        return ""
    val = row.get("text") or row.get("content")
    if isinstance(val, str):
        return val
    # Fall back to joining messages if present (tolerant of various shapes)
    msgs = row.get("messages")
    if isinstance(msgs, list):
        parts: List[str] = []
        for m in msgs:
            if isinstance(m, Mapping):
                c = m.get("content")
                if isinstance(c, str):
                    parts.append(c)
        return "\n".join(parts)
    return ""


def decontaminate_rows(
    rows: Sequence[Any],
    benchmark_texts: Sequence[str],
    *,
    n: int = 8,
    threshold: float = 0.8,
) -> Tuple[List[Mapping[str, Any]], List[int]]:
    """Filter rows whose n-gram overlap with any benchmark text exceeds threshold.

    Returns ``(kept_rows, removed_indices)``. Rows that are not Mapping
    instances are silently dropped from the kept output (their original
    index is also not flagged as decontaminated — they're simply not the
    target of this pipeline).
    """
    _require_int(n, name="n", low=1, high=32)
    _require_unit_float(threshold, name="threshold")
    bench_grams = [ngram_set(t, n=n) for t in benchmark_texts if isinstance(t, str)]

    kept: List[Mapping[str, Any]] = []
    removed: List[int] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        if not bench_grams:
            kept.append(row)
            continue
        text = _extract_row_text(row)
        row_grams = ngram_set(text, n=n) if text else set()
        contaminated = False
        for bg in bench_grams:
            if not bg or not row_grams:
                continue
            inter = bg & row_grams
            ratio = len(inter) / len(bg)
            if ratio >= threshold:
                contaminated = True
                break
        if contaminated:
            removed.append(idx)
        else:
            kept.append(row)
    return kept, removed


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------


_PII_SCAN_CAP = 50_000  # ReDoS defence — finditer never sees more than 50 KB


# Narrow regex set — false positives are acceptable for triage; live
# Presidio integration ships behind `[data-pro]` in v0.47.1.
#
# All patterns are written to avoid nested optional quantifiers, which
# trigger catastrophic backtracking on near-miss inputs. The phone and
# credit-card patterns specifically use a flat alternation and a hard
# `{n,m}` cap with no optional inner group repetition.
_PII_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    # Phone: optional leading "+", 7-15 digits with at most one separator
    # between each pair of digits. No nested optional groups.
    ("phone", re.compile(r"(?:\+?\d{1,3}[\s.\-]?)?\d{3}[\s.\-]?\d{3,4}[\s.\-]?\d{0,4}")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Credit card: exactly 13-19 digits with at most one space/hyphen between
    # consecutive digits. Cap the optional separator to a single char and
    # require the whole run to be word-boundary anchored.
    ("credit_card", re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,7}\b")),
)


def detect_pii(text: Any) -> List[Dict[str, str]]:
    """Return a list of ``{kind, snippet}`` PII hits.

    Scans only the first ``_PII_SCAN_CAP`` chars of ``text`` to keep regex
    finditer cost bounded regardless of caller input size.
    """
    s = _require_str(text, name="text")
    if len(s) > _PII_SCAN_CAP:
        s = s[:_PII_SCAN_CAP]
    hits: List[Dict[str, str]] = []
    for kind, pat in _PII_PATTERNS:
        for m in pat.finditer(s):
            snippet = m.group(0)
            if len(snippet) > 64:
                snippet = snippet[:61] + "..."
            # Skip phone matches that are too short to be real (regex can
            # match 3-4 digit fragments after the simplification).
            if kind == "phone":
                digits = sum(1 for c in snippet if c.isdigit())
                if digits < 7:
                    continue
            hits.append({"kind": kind, "snippet": snippet})
            if len(hits) >= 32:
                return hits
    return hits


# ---------------------------------------------------------------------------
# Language detection (heuristic)
# ---------------------------------------------------------------------------


# Tiny stopword sets — covers the rough cases without bundling fastText.
# Live fastText / langdetect support gated behind `[data-pro]` extras.
_LANG_STOPWORDS: Mapping[str, frozenset] = MappingProxyType(
    {
        "en": frozenset({
            "the", "and", "of", "to", "in", "is", "that", "for", "on", "with",
            "as", "are", "this", "be", "by", "at", "an", "or", "from", "it",
        }),
        "es": frozenset({
            "el", "la", "los", "las", "de", "que", "y", "en", "un", "es",
            "por", "con", "para", "se", "no", "más", "una", "su", "muy",
        }),
        "fr": frozenset({
            "le", "la", "les", "de", "et", "à", "un", "une", "que", "qui",
            "pour", "dans", "sur", "avec", "ne", "pas", "est", "ce", "des",
        }),
        "de": frozenset({
            "der", "die", "das", "und", "in", "den", "von", "zu", "mit",
            "ist", "im", "für", "auf", "ein", "eine", "auch", "als", "nicht",
        }),
        "pt": frozenset({
            "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
            "com", "não", "os", "as", "no", "se", "uma", "por", "mais",
        }),
        "ru": frozenset({"и", "в", "не", "что", "на", "с", "по", "это", "как"}),
    }
)


def detect_language(text: Any) -> str:
    """Return a 2-letter ISO code or ``"unknown"``.

    Pure-Python stopword heuristic; conservative — falls through to
    ``"unknown"`` on short or ambiguous input. For production-grade
    detection, install ``langdetect`` and pipe via the ``[data-pro]``
    extras (deferred to v0.47.1).
    """
    s = _require_str(text, name="text")
    tokens = _tokenise(s)
    if len(tokens) < 4:
        return "unknown"
    token_set = set(tokens)
    best_lang = "unknown"
    best_hits = 0
    for lang, stops in _LANG_STOPWORDS.items():
        hits = sum(1 for t in token_set if t in stops)
        if hits > best_hits:
            best_hits = hits
            best_lang = lang
    if best_hits < 1:
        return "unknown"
    return best_lang


# ---------------------------------------------------------------------------
# Toxicity (keyword baseline)
# ---------------------------------------------------------------------------


# Intentionally small, neutral keyword set. The full Llama-Guard-3-1B
# integration is the v0.47.1 follow-up; this baseline gives a fast
# triage signal without the 1 GB model download.
_TOXIC_KEYWORDS: frozenset = frozenset(
    {
        "hate", "kill", "destroy", "attack", "violence", "abuse",
        "slur", "die", "murder", "assault", "racist",
    }
)


def score_toxicity(text: Any) -> float:
    """Return [0, 1] toxicity score from a keyword baseline.

    A real Llama-Guard variant lands in v0.47.1 via ``[data-pro]``.
    """
    s = _require_str(text, name="text")
    tokens = _tokenise(s)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in _TOXIC_KEYWORDS)
    # Sub-linear weighting so long benign documents don't accumulate noise.
    score = min(1.0, hits / max(1, len(tokens) ** 0.5))
    return score


# ---------------------------------------------------------------------------
# Educational value (length + lexical-diversity proxy)
# ---------------------------------------------------------------------------


def score_educational_value(text: Any) -> float:
    """Return [0, 1] educational-value score.

    Combines (a) log-scale length and (b) type/token ratio as a proxy
    for vocabulary breadth. Lightweight stand-in for FineWeb-Edu's
    classifier — the real model ships behind ``[data-pro]`` in v0.47.1.
    """
    s = _require_str(text, name="text")
    tokens = _tokenise(s)
    if not tokens:
        return 0.0
    n = len(tokens)
    unique = len(set(tokens))
    # Length component: ramps up smoothly to 1.0 around 200 tokens.
    import math
    length_score = min(1.0, math.log(1 + n) / math.log(200))
    diversity = unique / n
    # Convex combination keeps both signals in the unit interval.
    return max(0.0, min(1.0, 0.5 * length_score + 0.5 * diversity))


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------


def compute_scorecard(
    rows: Sequence[Any],
    *,
    benchmarks: Sequence[str] = (),
    decontaminate_texts: Optional[Mapping[str, Sequence[str]]] = None,
    decontaminate_threshold: float = 0.8,
) -> ScoreReport:
    """Compute the composite scorecard over a row sequence.

    ``benchmarks`` is the closed allowlist of benchmark names to consider;
    ``decontaminate_texts`` maps benchmark name → list of texts (caller
    supplies the corpora; live MMLU/GSM8K loaders ship in v0.47.1).
    """
    _require_unit_float(decontaminate_threshold, name="decontaminate_threshold")
    for b in benchmarks:
        if not isinstance(b, str) or b not in BENCHMARKS:
            raise ValueError(f"unknown benchmark: {b!r}")

    dec_texts: List[str] = []
    if decontaminate_texts:
        for b in benchmarks:
            seq = decontaminate_texts.get(b) or []
            dec_texts.extend(t for t in seq if isinstance(t, str))

    total = 0
    pii_flagged = 0
    toxic_flagged = 0
    edu_total = 0.0
    langs: Dict[str, int] = {}
    kept_for_dec: List[Mapping[str, Any]] = []

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        total += 1
        text = _extract_row_text(row)
        if not text:
            continue
        try:
            if detect_pii(text):
                pii_flagged += 1
        except ValueError as exc:
            _LOG.debug("pii failed for row: %s", exc)
        try:
            if score_toxicity(text) >= 0.05:
                toxic_flagged += 1
        except ValueError as exc:
            _LOG.debug("toxicity failed for row: %s", exc)
        try:
            edu_total += score_educational_value(text)
        except ValueError as exc:
            _LOG.debug("edu failed for row: %s", exc)
        try:
            lang = detect_language(text)
        except ValueError as exc:
            _LOG.debug("lang failed for row: %s", exc)
            lang = "unknown"
        langs[lang] = langs.get(lang, 0) + 1
        kept_for_dec.append(row)

    if dec_texts and kept_for_dec:
        _, removed = decontaminate_rows(
            kept_for_dec,
            dec_texts,
            n=8,
            threshold=decontaminate_threshold,
        )
        dec_removed = len(removed)
    else:
        dec_removed = 0

    mean_edu = (edu_total / total) if total else 0.0
    # Freeze the language dict via a MappingProxy so consumers can't mutate it.
    return ScoreReport(
        total=total,
        pii_flagged=pii_flagged,
        toxic_flagged=toxic_flagged,
        decontaminated_removed=dec_removed,
        languages=MappingProxyType(dict(langs)),
        educational_mean=mean_edu,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _check_input_path(path: Any) -> str:
    if not isinstance(path, str):
        raise TypeError("input path must be a string")
    if not path or "\x00" in path:
        raise ValueError("input path must be a non-empty NUL-free string")
    if len(path) > _MAX_PATH_LEN:
        raise ValueError(f"input path exceeds {_MAX_PATH_LEN} chars")
    if not is_under_cwd(path):
        raise ValueError("input path must stay under cwd")
    try:
        if _stat.S_ISLNK(os.lstat(path).st_mode):
            raise ValueError("input path must not be a symlink")
    except FileNotFoundError:
        raise FileNotFoundError(f"input file not found: {path!r}") from None
    return os.path.realpath(path)


def load_jsonl_rows(path: Any) -> List[Mapping[str, Any]]:
    """Read a JSONL file under cwd; skip malformed lines silently."""
    target = _check_input_path(path)
    size = os.path.getsize(target)
    if size > _MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {_MAX_FILE_BYTES} bytes")
    out: List[Mapping[str, Any]] = []
    with open(target, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if len(out) >= _MAX_ROWS:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping):
                out.append(row)
    return out


def write_jsonl_rows(rows: Iterable[Mapping[str, Any]], path: Any) -> str:
    """Atomic JSONL write under cwd. Returns realpath."""
    if not isinstance(path, str):
        raise TypeError("output path must be a string")
    if not path or "\x00" in path:
        raise ValueError("output path must be a non-empty NUL-free string")
    if len(path) > _MAX_PATH_LEN:
        raise ValueError(f"output path exceeds {_MAX_PATH_LEN} chars")
    if not is_under_cwd(path):
        raise ValueError("output path must stay under cwd")
    try:
        if _stat.S_ISLNK(os.lstat(path).st_mode):
            raise ValueError("output path must not be a symlink")
    except FileNotFoundError:
        pass
    target = os.path.realpath(path)

    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".score-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


#: v0.53.7 M-J: public alias for ``_extract_row_text`` so external callers
#: (commands/data_score.py) do not import a private name.
extract_row_text = _extract_row_text


__all__ = [
    "BENCHMARKS",
    "ScoreReport",
    "compute_scorecard",
    "decontaminate_rows",
    "detect_language",
    "detect_pii",
    "extract_row_text",
    "load_jsonl_rows",
    "ngram_overlap_ratio",
    "ngram_set",
    "score_educational_value",
    "score_toxicity",
    "write_jsonl_rows",
]
