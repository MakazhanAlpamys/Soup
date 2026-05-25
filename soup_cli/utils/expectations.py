"""v0.69.0 Part B — Expectations suite for chat data.

Great-Expectations-flavoured assertions over JSONL rows. Each expectation is a
pure function returning a frozen ``ExpectationResult``; the suite runner
composes them and returns a ``SuiteReport`` whose ``passed`` flag is the
AND of every contained expectation. CI gates use exit code 3 on failure
(matches v0.55 / v0.56 / v0.64 / v0.65 gate convention).

Composes with:
  - v0.47.0 ``data_score.detect_pii`` (PII regex / Presidio backend)
  - v0.56.0 ``diagnose.refusal.looks_like_refusal`` (English refusal phrases)
  - v0.19.0 judge backends (operator-injected callable for judge expectations)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

# Closed allowlist — the 4 expectation kinds the v0.69.0 plan calls out.
_SUPPORTED_EXPECTATIONS = (
    "expect_no_pii",
    "expect_token_length_between",
    "expect_no_refusal_pattern",
    "expect_chosen_preferred_over_rejected_by_judge",
)
SUPPORTED_EXPECTATIONS: frozenset = frozenset(_SUPPORTED_EXPECTATIONS)

# DoS caps + bounds for validators.
_MAX_NAME_LEN = 128
_MAX_DETAILS_PER_RESULT = 32
_MAX_DETAIL_LEN = 256
_MAX_SUITE_LEN = 64
_MAX_FILE_BYTES = 1_048_576  # 1 MiB
_MIN_TOKEN_BOUND = 1
_MAX_TOKEN_BOUND = 1_048_576

# JudgeFn signature: row mapping in, [0,1] score out (1.0 = chosen wins).
JudgeFn = Callable[[Mapping[str, Any]], float]


@dataclass(frozen=True)
class ExpectationResult:
    """Outcome of one expectation."""

    name: str
    passed: bool
    num_rows_checked: int
    num_violations: int
    details: Tuple[str, ...]

    def __post_init__(self) -> None:
        validate_expectation_name(self.name)
        if not isinstance(self.passed, bool):
            raise TypeError("ExpectationResult.passed must be bool")
        for field_name, val in (
            ("num_rows_checked", self.num_rows_checked),
            ("num_violations", self.num_violations),
        ):
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"ExpectationResult.{field_name} must be int")
            if val < 0:
                raise ValueError(
                    f"ExpectationResult.{field_name} must be non-negative"
                )
        if not isinstance(self.details, tuple):
            raise TypeError(
                "ExpectationResult.details must be a tuple (frozen=True does "
                "not make List immutable)"
            )


@dataclass(frozen=True)
class ExpectationSpec:
    """One row of a suite YAML."""

    name: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class SuiteSpec:
    """Validated expectations suite."""

    expectations: Tuple[ExpectationSpec, ...]


@dataclass(frozen=True)
class SuiteReport:
    """Outcome of a full suite run."""

    passed: bool
    results: Tuple[ExpectationResult, ...]


# -----------------------------------------------------------------------------
# Validators
# -----------------------------------------------------------------------------


def validate_expectation_name(name: object) -> str:
    """Return the canonical lower-case expectation name."""
    if isinstance(name, bool) or not isinstance(name, str):
        raise TypeError(
            f"expectation name must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("expectation name must be non-empty")
    if "\x00" in name:
        raise ValueError("expectation name must not contain null bytes")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(
            f"expectation name must be <= {_MAX_NAME_LEN} chars"
        )
    canonical = name.strip().lower()
    if canonical not in SUPPORTED_EXPECTATIONS:
        raise ValueError(
            f"unknown expectation: {name!r}. "
            f"supported: {sorted(SUPPORTED_EXPECTATIONS)}"
        )
    return canonical


def _check_threshold(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be float, not bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{field} must be finite")
    if not (0.0 <= fval <= 1.0):
        raise ValueError(f"{field} must be in [0.0, 1.0]")
    return fval


def _check_token_bound(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < _MIN_TOKEN_BOUND or value > _MAX_TOKEN_BOUND:
        raise ValueError(
            f"{field} must be in [{_MIN_TOKEN_BOUND}, {_MAX_TOKEN_BOUND}]"
        )
    return value


# -----------------------------------------------------------------------------
# Row helpers
# -----------------------------------------------------------------------------


def _extract_row_text(row: Mapping[str, Any]) -> str:
    """Best-effort text extraction. Mirrors v0.55.0 / v0.56.0 row-text policy.

    Combines all text-shaped fields so PII / refusal scans never miss content
    hiding in either ``text``/``content``/``output``/``response`` or inside a
    ``messages`` chat structure.
    """
    parts: List[str] = []
    for key in ("text", "content", "output", "response", "prompt", "instruction"):
        val = row.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, Mapping):
                content = msg.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
    return "\n".join(parts)


def _extract_assistant_text(row: Mapping[str, Any]) -> str:
    """Best-effort assistant-side text extraction (for refusal scans)."""
    parts: List[str] = []
    for key in ("output", "response"):
        val = row.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, Mapping) and msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
    if parts:
        return "\n".join(parts)
    # Fall back to general row text when the row has no chat structure.
    return _extract_row_text(row)


def _check_rows(rows: object) -> Sequence[Mapping[str, Any]]:
    if isinstance(rows, str) or isinstance(rows, bytes):
        raise TypeError("rows must be a list of mappings, not a string")
    if not hasattr(rows, "__iter__"):
        raise TypeError("rows must be iterable")
    try:
        materialised = list(rows)
    except TypeError as exc:
        raise TypeError(f"rows is not iterable: {exc}") from exc
    return materialised


def _truncate_detail(detail: str) -> str:
    if len(detail) > _MAX_DETAIL_LEN:
        return detail[: _MAX_DETAIL_LEN - 3] + "..."
    return detail


# -----------------------------------------------------------------------------
# Expectations
# -----------------------------------------------------------------------------


def expect_no_pii(rows: Any) -> ExpectationResult:
    """Fail when any row contains an email / phone / SSN / credit-card hit.

    Reuses v0.47.0 ``data_score.detect_pii`` (Presidio backend if available;
    falls back to the in-tree 4-regex baseline) — so improvements there
    immediately flow into expectations.
    """
    materialised = _check_rows(rows)
    from soup_cli.utils.data_score import detect_pii  # lazy

    num_violations = 0
    details: List[str] = []
    for index, row in enumerate(materialised):
        if not isinstance(row, Mapping):
            details.append(_truncate_detail(f"rows[{index}]: not a dict"))
            num_violations += 1
            continue
        text = _extract_row_text(row)
        if not text:
            continue
        try:
            hits = detect_pii(text)
        except (TypeError, ValueError):
            continue
        if hits:
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                kinds = sorted({hit.get("kind", "?") for hit in hits})
                details.append(
                    _truncate_detail(
                        f"rows[{index}]: PII detected ({', '.join(kinds)})"
                    )
                )
    return ExpectationResult(
        name="expect_no_pii",
        passed=num_violations == 0,
        num_rows_checked=len(materialised),
        num_violations=num_violations,
        details=tuple(details),
    )


def expect_token_length_between(
    rows: Any,
    *,
    min_tokens: int,
    max_tokens: int,
) -> ExpectationResult:
    """Fail when any row's token count (whitespace-split) is out of bounds."""
    low = _check_token_bound(min_tokens, field="min_tokens")
    high = _check_token_bound(max_tokens, field="max_tokens")
    if low > high:
        raise ValueError(
            f"min_tokens ({low}) must be <= max_tokens ({high})"
        )
    materialised = _check_rows(rows)
    num_violations = 0
    details: List[str] = []
    for index, row in enumerate(materialised):
        if not isinstance(row, Mapping):
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                details.append(_truncate_detail(f"rows[{index}]: not a dict"))
            continue
        text = _extract_row_text(row)
        token_count = len(text.split())
        if token_count < low or token_count > high:
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                details.append(
                    _truncate_detail(
                        f"rows[{index}]: {token_count} tokens (want [{low}, {high}])"
                    )
                )
    return ExpectationResult(
        name="expect_token_length_between",
        passed=num_violations == 0,
        num_rows_checked=len(materialised),
        num_violations=num_violations,
        details=tuple(details),
    )


def expect_no_refusal_pattern(rows: Any) -> ExpectationResult:
    """Fail when any assistant-side output matches the v0.56.0 refusal regex."""
    materialised = _check_rows(rows)
    from soup_cli.utils.diagnose.refusal import looks_like_refusal  # lazy

    num_violations = 0
    details: List[str] = []
    for index, row in enumerate(materialised):
        if not isinstance(row, Mapping):
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                details.append(_truncate_detail(f"rows[{index}]: not a dict"))
            continue
        text = _extract_assistant_text(row)
        if text and looks_like_refusal(text):
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                details.append(
                    _truncate_detail(f"rows[{index}]: refusal pattern matched")
                )
    return ExpectationResult(
        name="expect_no_refusal_pattern",
        passed=num_violations == 0,
        num_rows_checked=len(materialised),
        num_violations=num_violations,
        details=tuple(details),
    )


def expect_chosen_preferred_over_rejected_by_judge(
    rows: Any,
    *,
    judge_fn: Optional[JudgeFn] = None,
    threshold: float = 0.7,
) -> ExpectationResult:
    """Fail when ``judge_fn(row) < threshold`` on a preference row.

    Rows must carry both ``chosen`` and ``rejected``. ``judge_fn`` returns a
    score in [0, 1]; 1.0 means the judge fully prefers chosen over rejected.

    When ``judge_fn`` is omitted the suite runs in *advisory* mode (every row
    gets the default score 1.0, i.e. trust the labelling); production callers
    should always supply a real judge.
    """
    t = _check_threshold(threshold, field="threshold")
    if judge_fn is not None and not callable(judge_fn):
        raise TypeError("judge_fn must be callable or None")
    materialised = _check_rows(rows)
    num_violations = 0
    details: List[str] = []
    for index, row in enumerate(materialised):
        if not isinstance(row, Mapping):
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                details.append(_truncate_detail(f"rows[{index}]: not a dict"))
            continue
        if "chosen" not in row or "rejected" not in row:
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                details.append(
                    _truncate_detail(
                        f"rows[{index}]: missing chosen/rejected field"
                    )
                )
            continue
        if judge_fn is None:
            # No judge supplied — assume chosen wins (advisory pass).
            score: float = 1.0
        else:
            try:
                raw = judge_fn(row)
            except Exception:  # noqa: BLE001 — one bad row mustn't crash the suite
                num_violations += 1
                if len(details) < _MAX_DETAILS_PER_RESULT:
                    details.append(
                        _truncate_detail(f"rows[{index}]: judge raised")
                    )
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                num_violations += 1
                if len(details) < _MAX_DETAILS_PER_RESULT:
                    details.append(
                        _truncate_detail(
                            f"rows[{index}]: judge returned non-number"
                        )
                    )
                continue
            score = float(raw)
            if not math.isfinite(score):
                num_violations += 1
                continue
        if score < t:
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                details.append(
                    _truncate_detail(
                        f"rows[{index}]: judge score {score:.3f} < {t:.3f}"
                    )
                )
    return ExpectationResult(
        name="expect_chosen_preferred_over_rejected_by_judge",
        passed=num_violations == 0,
        num_rows_checked=len(materialised),
        num_violations=num_violations,
        details=tuple(details),
    )


# -----------------------------------------------------------------------------
# Suite parsing + execution
# -----------------------------------------------------------------------------


def parse_suite_spec(raw: Any) -> SuiteSpec:
    """Validate a suite dict and return a ``SuiteSpec``."""
    if not isinstance(raw, dict):
        raise TypeError("suite spec must be a dict")
    raw_expectations = raw.get("expectations")
    if raw_expectations is None:
        raise ValueError("suite must define 'expectations' key")
    if not isinstance(raw_expectations, list):
        raise TypeError("suite 'expectations' must be a list")
    if not raw_expectations:
        raise ValueError("suite 'expectations' must be a non-empty list")
    if len(raw_expectations) > _MAX_SUITE_LEN:
        raise ValueError(
            f"suite 'expectations' exceeds {_MAX_SUITE_LEN} entries"
        )

    items: List[ExpectationSpec] = []
    for index, entry in enumerate(raw_expectations):
        if not isinstance(entry, dict):
            raise TypeError(f"expectations[{index}] must be a dict")
        name = validate_expectation_name(entry.get("name", ""))
        args = entry.get("args", {})
        if not isinstance(args, dict):
            raise TypeError(f"expectations[{index}].args must be a dict")
        items.append(ExpectationSpec(name=name, args=dict(args)))
    return SuiteSpec(expectations=tuple(items))


def parse_suite_yaml(text: object) -> SuiteSpec:
    """Parse a suite YAML string into a validated ``SuiteSpec``."""
    if not isinstance(text, str):
        raise TypeError("suite text must be a string")
    if "\x00" in text:
        raise ValueError("suite text must not contain null bytes")
    if len(text.encode("utf-8")) > _MAX_FILE_BYTES:
        raise ValueError(f"suite text exceeds {_MAX_FILE_BYTES} bytes")
    import yaml  # lazy

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    return parse_suite_spec(data)


def load_suite_yaml(path: object) -> SuiteSpec:
    """Load a suite YAML from a path under cwd (with TOCTOU symlink reject).

    Delegates to ``utils.paths.enforce_under_cwd_and_no_symlink`` (v0.59.0
    centralised helper).
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    if isinstance(path, bool) or not isinstance(path, str):
        raise TypeError("path must be a string")
    enforce_under_cwd_and_no_symlink(path, "suite path")
    if not os.path.lexists(path):
        raise FileNotFoundError(path)
    real = os.path.realpath(path)
    if not os.path.isfile(real):
        raise FileNotFoundError(real)
    if os.path.getsize(real) > _MAX_FILE_BYTES:
        raise ValueError(f"suite file exceeds {_MAX_FILE_BYTES} bytes")
    with open(real, "r", encoding="utf-8") as handle:
        return parse_suite_yaml(handle.read())


def _dispatch_expectation(
    spec: ExpectationSpec,
    rows: Sequence[Mapping[str, Any]],
    *,
    judge_fn: Optional[JudgeFn] = None,
) -> ExpectationResult:
    """Dispatch one ``ExpectationSpec`` against ``rows``.

    Per code-review HIGH H2: passes ``args`` values through *as-is* — the
    expectation function's own validators must reject bool / NaN / Inf /
    type-coerced inputs. This prevents an ``int("5")`` silent coercion from
    bypassing ``_check_token_bound`` bool-rejection.
    """
    name = spec.name
    args = dict(spec.args)
    if name == "expect_no_pii":
        return expect_no_pii(rows)
    if name == "expect_token_length_between":
        return expect_token_length_between(
            rows,
            min_tokens=args.get("min_tokens", 1),
            max_tokens=args.get("max_tokens", _MAX_TOKEN_BOUND),
        )
    if name == "expect_no_refusal_pattern":
        return expect_no_refusal_pattern(rows)
    if name == "expect_chosen_preferred_over_rejected_by_judge":
        return expect_chosen_preferred_over_rejected_by_judge(
            rows,
            judge_fn=judge_fn,
            threshold=args.get("threshold", 0.7),
        )
    raise ValueError(f"unhandled expectation: {name!r}")  # pragma: no cover


def run_suite(
    rows: Any,
    spec: SuiteSpec,
    *,
    judge_fn: Optional[JudgeFn] = None,
) -> SuiteReport:
    """Run every expectation in ``spec`` against ``rows``."""
    if not isinstance(spec, SuiteSpec):
        raise TypeError("spec must be a SuiteSpec")
    materialised = _check_rows(rows)
    results: List[ExpectationResult] = []
    for expectation in spec.expectations:
        results.append(
            _dispatch_expectation(expectation, materialised, judge_fn=judge_fn)
        )
    passed = all(r.passed for r in results)
    return SuiteReport(passed=passed, results=tuple(results))


__all__ = [
    "ExpectationResult",
    "ExpectationSpec",
    "JudgeFn",
    "SUPPORTED_EXPECTATIONS",
    "SuiteReport",
    "SuiteSpec",
    "expect_chosen_preferred_over_rejected_by_judge",
    "expect_no_pii",
    "expect_no_refusal_pattern",
    "expect_token_length_between",
    "load_suite_yaml",
    "parse_suite_spec",
    "parse_suite_yaml",
    "run_suite",
    "validate_expectation_name",
]
