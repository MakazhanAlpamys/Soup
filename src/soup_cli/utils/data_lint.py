"""``soup data lint`` — preference-data linter for dpo/orpo/simpo/ipo/bco/kto
(v0.71.27).

ORPO/SimPO/IPO/BCO all share DPO's on-disk ``{prompt, chosen, rejected}``
shape (there is no separate ``data.format`` literal for them — see
``config/schema.py``'s ``DataConfig.format`` allowlist), so detecting
``fmt == "dpo"`` covers the whole preference-loss family; KTO's
``{prompt, completion, label}`` shape is handled separately.

Five checks:

- **length_bias** — chosen systematically longer than rejected (the #1
  silent DPO degradation: the model learns "longer is better" instead of
  the actual preference signal). Reported as a Cohen's d effect size.
- **label_imbalance** — KTO desirable vs. undesirable class balance.
- **near_duplicates** — reuses the MinHash/LSH approach from
  ``commands/data.py::dedup`` to flag rows that are near-duplicates of
  another row in the same dataset.
- **identical_pairs** — chosen == rejected (zero preference signal; a hard
  bug in every row it touches).
- **prompt_leak** — the prompt echoed verbatim inside the completion (a
  common synthetic-data pipeline bug).

Same OK / MINOR / MAJOR taxonomy as ``utils/data_doctor.py`` (imported from
there to keep one source of truth for the two doctor/lint twins shipped in
the same release).

No top-level torch / transformers import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Sequence, Tuple

from soup_cli.utils.data_doctor import VERDICTS, sample_indices

CHECKS: Tuple[str, ...] = (
    "length_bias",
    "label_imbalance",
    "near_duplicates",
    "identical_pairs",
    "prompt_leak",
)

_SUPPORTED_FORMATS: Tuple[str, ...] = ("dpo", "kto")
_DEFAULT_SAMPLE_SIZE = 2000
_MAX_SAMPLE_SIZE = 50_000
_MIN_PROMPT_LEAK_LEN = 40


@dataclass(frozen=True)
class LintCheck:
    name: str
    verdict: str
    message: str
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.name not in CHECKS:
            raise ValueError(f"unknown lint check {self.name!r}")
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")
        for attr in ("message", "evidence"):
            value = getattr(self, attr)
            if not isinstance(value, str):
                raise TypeError(f"{attr} must be str")
            if "\x00" in value:
                raise ValueError(f"{attr} must not contain null bytes")
            if len(value) > 2048:
                raise ValueError(f"{attr} too long (max 2048 chars)")


def _overall_verdict(checks: Sequence[LintCheck]) -> str:
    rank = {"OK": 0, "MINOR": 1, "MAJOR": 2}
    worst = "OK"
    for check in checks:
        if not isinstance(check, LintCheck):
            raise TypeError("checks must contain LintCheck instances")
        if rank[check.verdict] > rank[worst]:
            worst = check.verdict
    return worst


@dataclass(frozen=True)
class LintReport:
    checks: Tuple[LintCheck, ...]
    overall: str
    fmt: str
    rows_scanned: int
    total_rows: int

    def __post_init__(self) -> None:
        if not isinstance(self.checks, tuple):
            object.__setattr__(self, "checks", tuple(self.checks))
        for check in self.checks:
            if not isinstance(check, LintCheck):
                raise TypeError("checks must contain LintCheck instances")
        if self.overall not in VERDICTS:
            raise ValueError(f"overall must be one of {VERDICTS}")
        if self.fmt not in _SUPPORTED_FORMATS:
            raise ValueError(f"fmt must be one of {_SUPPORTED_FORMATS}")
        for attr in ("rows_scanned", "total_rows"):
            value = getattr(self, attr)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{attr} must be a non-negative int")
        if self.rows_scanned > self.total_rows:
            raise ValueError("rows_scanned cannot exceed total_rows")

    def to_dict(self) -> dict:
        return {
            "fmt": self.fmt,
            "overall": self.overall,
            "rows_scanned": self.rows_scanned,
            "total_rows": self.total_rows,
            "checks": [
                {"name": c.name, "verdict": c.verdict, "message": c.message, "evidence": c.evidence}
                for c in self.checks
            ],
        }


def compose_lint_report(
    checks: Sequence[LintCheck], *, fmt: str, rows_scanned: int, total_rows: int
) -> LintReport:
    return LintReport(
        checks=tuple(checks),
        overall=_overall_verdict(checks),
        fmt=fmt,
        rows_scanned=rows_scanned,
        total_rows=total_rows,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def extract_pref_text(value: Any) -> str:
    """Flatten a chosen/rejected/completion field to plain text.

    DPO chosen/rejected may legitimately be message LISTS (conversational
    DPO) rather than plain strings — join every string ``content`` field so
    length/near-dup/leak checks work uniformly on both shapes.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for turn in value:
            if isinstance(turn, Mapping):
                content = turn.get("content")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)
    return "" if value is None else str(value)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardized mean difference (pooled std) between two samples.

    Degenerate cases (fewer than 2 points in either sample, or zero pooled
    variance) fall back to a sign-only +/-1.0 (or 0.0 when the means tie)
    rather than raising a ZeroDivisionError.
    """
    if not a or not b:
        raise ValueError("cohens_d requires non-empty samples")
    n_a, n_b = len(a), len(b)
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    if n_a < 2 or n_b < 2:
        return 0.0 if mean_a == mean_b else math.copysign(1.0, mean_a - mean_b)
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0.0:
        return 0.0 if mean_a == mean_b else math.copysign(1.0, mean_a - mean_b)
    return (mean_a - mean_b) / pooled_std


def _coerce_kto_label(value: Any) -> bool:
    """Best-effort truthiness for a KTO ``label``.

    By the time ``check_label_imbalance`` sees a row, ``run_lint`` has
    already normalized it through ``data.formats.format_to_messages``
    (which raises/drops on an unrecognised string like "maybe" rather than
    defaulting it), so this permissive fallback is intentionally a defensive
    default for direct callers of the check, not the pipeline's real gate.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        return low in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return value > 0
    return bool(value)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_length_bias(rows: Sequence[Mapping], *, length_fn: Callable[[str], float]) -> LintCheck:
    """The #1 silent DPO degradation: chosen systematically longer than
    rejected teaches "longer is better" instead of the real preference.
    MAJOR at |Cohen's d| >= 0.8 (large effect, Cohen's convention), MINOR
    at >= 0.3, else OK."""
    if not rows:
        return LintCheck(name="length_bias", verdict="OK", message="no rows")
    chosen_lens: List[float] = []
    rejected_lens: List[float] = []
    longer_chosen = 0
    for row in rows:
        c_len = length_fn(extract_pref_text(row.get("chosen")))
        r_len = length_fn(extract_pref_text(row.get("rejected")))
        chosen_lens.append(c_len)
        rejected_lens.append(r_len)
        if c_len > r_len:
            longer_chosen += 1
    d = cohens_d(chosen_lens, rejected_lens)
    abs_d = abs(d)
    if abs_d >= 0.8:
        verdict = "MAJOR"
    elif abs_d >= 0.3:
        verdict = "MINOR"
    else:
        verdict = "OK"
    if d > 0:
        direction = "chosen longer than rejected"
    elif d < 0:
        direction = "rejected longer than chosen"
    else:
        direction = "no directional bias"
    frac_longer = longer_chosen / len(rows)
    return LintCheck(
        name="length_bias",
        verdict=verdict,
        message=(
            f"effect size (Cohen's d) = {d:.3f} ({direction}); "
            f"chosen is longer in {frac_longer:.1%} of rows"
        ),
        evidence=(
            f"mean chosen={sum(chosen_lens) / len(chosen_lens):.1f}, "
            f"mean rejected={sum(rejected_lens) / len(rejected_lens):.1f}"
        ),
    )


def check_label_imbalance(rows: Sequence[Mapping]) -> LintCheck:
    """KTO desirable-vs-undesirable class balance. MAJOR when the minority
    class is under 5% of rows, MINOR under 20%, else OK."""
    total = len(rows)
    if total == 0:
        return LintCheck(name="label_imbalance", verdict="OK", message="no rows")
    positive = sum(1 for row in rows if _coerce_kto_label(row.get("label")))
    negative = total - positive
    minority_frac = min(positive, negative) / total
    if minority_frac < 0.05:
        verdict = "MAJOR"
    elif minority_frac < 0.20:
        verdict = "MINOR"
    else:
        verdict = "OK"
    return LintCheck(
        name="label_imbalance",
        verdict=verdict,
        message=(
            f"{positive} desirable / {negative} undesirable "
            f"({minority_frac:.1%} minority class)"
        ),
        evidence="KTO trains best with a reasonably balanced desirable:undesirable ratio",
    )


def check_near_duplicates(
    rows: Sequence[Mapping], *, key_fn: Callable[[Mapping], str], threshold: float = 0.85
) -> LintCheck:
    """MinHash/LSH near-duplicate detection (mirrors ``commands/data.py::dedup``).
    MAJOR when >=20% of rows have a near-duplicate elsewhere in the sample,
    MINOR at >=5%, else OK. Degrades to an advisory OK (not a false "no
    duplicates" claim) when ``datasketch`` isn't installed."""
    try:
        from datasketch import MinHash, MinHashLSH  # noqa: PLC0415
    except ImportError:
        return LintCheck(
            name="near_duplicates",
            verdict="OK",
            message="near-dup check skipped (datasketch not installed)",
            evidence="pip install 'soup-cli[data]' to enable",
        )
    if len(rows) < 2:
        return LintCheck(
            name="near_duplicates", verdict="OK", message="not enough rows to compare",
            evidence=f"{len(rows)} row(s)",
        )
    num_perm = 128
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    for idx, row in enumerate(rows):
        text = key_fn(row)
        words = text.lower().split()
        shingles = {" ".join(words[i : i + 3]) for i in range(max(1, len(words) - 2))}
        mh = MinHash(num_perm=num_perm)
        for shingle in shingles:
            mh.update(shingle.encode("utf-8"))
        minhashes.append(mh)
        try:
            lsh.insert(str(idx), mh)
        except ValueError:
            pass  # duplicate key — already inserted
    dup_rows = set()
    for idx, mh in enumerate(minhashes):
        matches = [int(m) for m in lsh.query(mh) if int(m) != idx]
        if matches:
            dup_rows.add(idx)
    frac = len(dup_rows) / len(rows)
    verdict = "MAJOR" if frac >= 0.20 else "MINOR" if frac >= 0.05 else "OK"
    return LintCheck(
        name="near_duplicates",
        verdict=verdict,
        message=(
            f"{len(dup_rows)}/{len(rows)} rows ({frac:.1%}) have a near-duplicate "
            "elsewhere in the dataset"
        ),
        evidence=f"threshold={threshold}",
    )


def check_identical_pairs(rows: Sequence[Mapping]) -> LintCheck:
    """MAJOR if ANY row has chosen == rejected — zero preference signal is a
    hard bug in every row it touches, not a statistical threshold."""
    if not rows:
        return LintCheck(name="identical_pairs", verdict="OK", message="no rows")
    bad = 0
    for row in rows:
        chosen = extract_pref_text(row.get("chosen")).strip()
        rejected = extract_pref_text(row.get("rejected")).strip()
        if chosen and chosen == rejected:
            bad += 1
    frac = bad / len(rows)
    verdict = "MAJOR" if bad > 0 else "OK"
    return LintCheck(
        name="identical_pairs",
        verdict=verdict,
        message=(
            f"{bad}/{len(rows)} rows ({frac:.1%}) have chosen == rejected "
            "(zero preference signal)"
        ),
    )


def check_prompt_leak(
    rows: Sequence[Mapping], *, fmt: str, min_prompt_len: int = _MIN_PROMPT_LEAK_LEN
) -> LintCheck:
    """Flags the prompt echoed verbatim inside the completion (a common
    synthetic-data pipeline bug). MAJOR at >=10% of rows, MINOR for any
    lower non-zero rate. Prompts shorter than ``min_prompt_len`` are
    skipped to avoid false positives on trivially-short shared substrings."""
    if not rows:
        return LintCheck(name="prompt_leak", verdict="OK", message="no rows")
    flagged = 0
    for row in rows:
        prompt = extract_pref_text(row.get("prompt")).strip()
        if len(prompt) < min_prompt_len:
            continue
        if fmt == "dpo":
            targets = [
                extract_pref_text(row.get("chosen")),
                extract_pref_text(row.get("rejected")),
            ]
        else:
            targets = [extract_pref_text(row.get("completion"))]
        if any(prompt in target for target in targets if target):
            flagged += 1
    frac = flagged / len(rows)
    verdict = "MAJOR" if frac >= 0.10 else "MINOR" if flagged > 0 else "OK"
    return LintCheck(
        name="prompt_leak",
        verdict=verdict,
        message=(
            f"{flagged}/{len(rows)} rows ({frac:.1%}) have the prompt echoed "
            "verbatim inside the completion"
        ),
        evidence=f"min_prompt_len={min_prompt_len}",
    )


def _dpo_dedup_key(row: Mapping) -> str:
    return extract_pref_text(row.get("prompt")) + " " + extract_pref_text(row.get("chosen"))


def _kto_dedup_key(row: Mapping) -> str:
    return extract_pref_text(row.get("prompt")) + " " + extract_pref_text(row.get("completion"))


# ---------------------------------------------------------------------------
# run_lint — end to end
# ---------------------------------------------------------------------------


def run_lint(
    raw_rows: Sequence[Mapping[str, Any]],
    fmt: str,
    *,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    length_fn: Any = None,
) -> LintReport:
    """Run the preference-data linter over a sample of ``raw_rows``.

    ``fmt='auto'`` detects dpo vs. kto from the first row. Any other
    non-dpo/kto format raises ``ValueError`` — lint only understands
    preference-shaped data.
    """
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive int")
    if sample_size > _MAX_SAMPLE_SIZE:
        raise ValueError(f"sample_size must be <= {_MAX_SAMPLE_SIZE}")

    from soup_cli.data.formats import detect_format, format_to_messages

    resolved_fmt = fmt
    if resolved_fmt == "auto":
        if not raw_rows:
            raise ValueError("cannot auto-detect format on an empty dataset")
        resolved_fmt = detect_format(list(raw_rows))
    if resolved_fmt not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"soup data lint only supports preference-shaped data (dpo/kto); "
            f"detected/given format={resolved_fmt!r}"
        )

    total = len(raw_rows)
    idxs = sample_indices(total, sample_size)
    normalized: List[dict] = []
    for i in idxs:
        converted = format_to_messages(raw_rows[i], resolved_fmt)
        if converted:
            normalized.append(converted)

    if total > 0 and not normalized:
        raise ValueError(
            f"no rows converted to {resolved_fmt} preference pairs — check for null/"
            "malformed prompt/chosen/rejected (dpo) or an unparseable label (kto)"
        )

    length_fn = length_fn or (lambda text: float(len(text.split())))

    if resolved_fmt == "dpo":
        checks = [
            check_length_bias(normalized, length_fn=length_fn),
            check_identical_pairs(normalized),
            check_near_duplicates(normalized, key_fn=_dpo_dedup_key),
            check_prompt_leak(normalized, fmt="dpo"),
        ]
    else:  # kto
        checks = [
            check_label_imbalance(normalized),
            check_near_duplicates(normalized, key_fn=_kto_dedup_key),
            check_prompt_leak(normalized, fmt="kto"),
        ]

    return compose_lint_report(
        checks, fmt=resolved_fmt, rows_scanned=len(normalized), total_rows=total
    )


__all__ = [
    "CHECKS",
    "LintCheck",
    "LintReport",
    "check_identical_pairs",
    "check_label_imbalance",
    "check_length_bias",
    "check_near_duplicates",
    "check_prompt_leak",
    "cohens_d",
    "compose_lint_report",
    "extract_pref_text",
    "run_lint",
]
