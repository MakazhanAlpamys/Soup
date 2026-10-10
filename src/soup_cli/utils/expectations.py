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

import difflib
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

# DoS caps + bounds for validators.
_MAX_NAME_LEN = 128
_MAX_DETAILS_PER_RESULT = 32
_MAX_DETAIL_LEN = 256
_MAX_SUITE_LEN = 64
_MAX_FILE_BYTES = 1_048_576  # 1 MiB
_MIN_TOKEN_BOUND = 1
_MAX_TOKEN_BOUND = 1_048_576

# Closed allowlist — the 4 expectation kinds the v0.69.0 plan calls out — mapped to
# the arguments each one takes and their defaults. #1428: a suite key outside this
# table is refused at parse time instead of running the expectation on its defaults.
_EXPECTATION_ARGS: dict[str, dict[str, Any]] = {
    "expect_no_pii": {},
    "expect_token_length_between": {"min_tokens": 1, "max_tokens": _MAX_TOKEN_BOUND},
    "expect_no_refusal_pattern": {},
    "expect_chosen_preferred_over_rejected_by_judge": {
        "threshold": 0.7,
        "judge": None,
        "advisory": False,
    },
}
_SUPPORTED_EXPECTATIONS = tuple(_EXPECTATION_ARGS)
SUPPORTED_EXPECTATIONS: frozenset = frozenset(_SUPPORTED_EXPECTATIONS)
_ENTRY_KEYS = ("name", "args")

# JudgeFn signature: row mapping in, [0,1] score out (1.0 = chosen wins).
JudgeFn = Callable[[Mapping[str, Any]], float]


def _extract_row_text_for_judge(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, Mapping) and "content" in item:
                parts.append(str(item["content"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(val) if val is not None else ""


def build_pairwise_judge_fn(judge: Any) -> JudgeFn:
    """Build a JudgeFn from a judge URL string, PairwiseJudge instance, or callable."""
    if callable(judge) and not hasattr(judge, "compare_pair"):
        return judge
    if isinstance(judge, str):
        if not judge.strip():
            raise ValueError("judge URL must be non-empty")
        from soup_cli.eval.gate import _parse_judge_url
        from soup_cli.eval.judge import JudgeEvaluator

        provider, model, api_base = _parse_judge_url(judge)
        evaluator = JudgeEvaluator(
            provider=provider,
            model=model,
            api_base=api_base,
        )
    elif hasattr(judge, "compare_pair"):
        evaluator = judge
    else:
        raise TypeError("judge must be a URL string, PairwiseJudge instance, or callable")

    from soup_cli.eval.judge import pairwise_compare

    def pairwise_judge_fn(row: Mapping[str, Any]) -> float:
        prompt = _extract_row_text_for_judge(row.get("prompt"))
        resp_a = _extract_row_text_for_judge(row.get("chosen"))
        resp_b = _extract_row_text_for_judge(row.get("rejected"))
        verdict = pairwise_compare(prompt, resp_a, resp_b, evaluator, swap=True)
        if verdict == 0:
            return 1.0
        if verdict == 1:
            return 0.0
        return 0.5

    return pairwise_judge_fn


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


def _check_token_bounds(min_tokens: object, max_tokens: object) -> Tuple[int, int]:
    low = _check_token_bound(min_tokens, field="min_tokens")
    high = _check_token_bound(max_tokens, field="max_tokens")
    if low > high:
        raise ValueError(
            f"min_tokens ({low}) must be <= max_tokens ({high})"
        )
    return low, high


# -----------------------------------------------------------------------------
# Row helpers & Format Extraction Rules
# -----------------------------------------------------------------------------

_ASSISTANT_ROLES = frozenset({"assistant", "gpt", "chatgpt", "bot", "model"})

FORMAT_EXTRACTION_RULES: Mapping[str, Mapping[str, Tuple[str, ...]]] = {
    "alpaca": {
        "text_fields": ("instruction", "input", "output", "system"),
        "assistant_fields": ("output", "response"),
    },
    "sharegpt": {
        "text_fields": ("conversations",),
        "assistant_fields": ("conversations",),
    },
    "chatml": {
        "text_fields": ("messages", "tools", "tool_calls"),
        "assistant_fields": ("messages", "tool_calls"),
    },
    "dpo": {
        "text_fields": ("prompt", "chosen", "rejected"),
        "assistant_fields": ("chosen",),
    },
    "kto": {
        "text_fields": ("prompt", "completion"),
        "assistant_fields": ("completion",),
    },
    "llava": {
        "text_fields": ("conversations",),
        "assistant_fields": ("conversations",),
    },
    "sharegpt4v": {
        "text_fields": ("conversations",),
        "assistant_fields": ("conversations",),
    },
    "embedding": {
        "text_fields": ("anchor", "positive", "negative"),
        "assistant_fields": (),
    },
    "audio": {
        "text_fields": ("messages",),
        "assistant_fields": ("messages",),
    },
    "asr": {
        "text_fields": ("text",),
        "assistant_fields": ("text",),
    },
    "plaintext": {
        "text_fields": ("text",),
        "assistant_fields": ("text",),
    },
    "tool-calling": {
        "text_fields": ("messages", "tools", "tool_calls"),
        "assistant_fields": ("messages", "tool_calls"),
    },
    "raft": {
        "text_fields": ("query", "golden_doc", "distractor_docs", "answer"),
        "assistant_fields": ("answer",),
    },
    "prm": {
        "text_fields": ("prompt", "completions"),
        "assistant_fields": ("completions",),
    },
    "input_output": {
        "text_fields": ("segments",),
        "assistant_fields": ("segments",),
    },
}


def _extract_content_parts(content: Any) -> List[str]:
    """Extract strings from string or multimodal content parts list."""
    import json

    parts: List[str] = []
    if isinstance(content, str):
        if content:
            parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                if part:
                    parts.append(part)
            elif isinstance(part, Mapping):
                val = part.get("text")
                if isinstance(val, str) and val:
                    parts.append(val)
                args = part.get("arguments") or part.get("input")
                if isinstance(args, str) and args:
                    parts.append(args)
                elif isinstance(args, (dict, list)):
                    parts.append(json.dumps(args))
    return parts


def _extract_tool_calls_text(tool_calls: Any) -> List[str]:
    """Extract function call arguments and names from tool_calls list."""
    import json

    parts: List[str] = []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, Mapping):
                func = call.get("function")
                if isinstance(func, Mapping):
                    args = func.get("arguments")
                    if isinstance(args, str) and args:
                        parts.append(args)
                    elif isinstance(args, (dict, list)):
                        parts.append(json.dumps(args))
                args = call.get("arguments")
                if isinstance(args, str) and args:
                    parts.append(args)
                elif isinstance(args, (dict, list)):
                    parts.append(json.dumps(args))
    return parts


def _extract_messages_text(messages: Any, *, assistant_only: bool = False) -> List[str]:
    """Extract text from a messages list (chatml, audio, tool-calling, multimodal)."""
    parts: List[str] = []
    if not isinstance(messages, list):
        return parts
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        role = msg.get("role")
        if assistant_only and role not in _ASSISTANT_ROLES:
            continue
        content = msg.get("content")
        parts.extend(_extract_content_parts(content))
        if "tool_calls" in msg:
            parts.extend(_extract_tool_calls_text(msg.get("tool_calls")))
    return parts


def _extract_conversations_text(
    conversations: Any, *, assistant_only: bool = False
) -> List[str]:
    """Extract text from a conversations list (sharegpt, llava, sharegpt4v)."""
    parts: List[str] = []
    if not isinstance(conversations, list):
        return parts
    for turn in conversations:
        if not isinstance(turn, Mapping):
            continue
        speaker = turn.get("from")
        if assistant_only and speaker not in _ASSISTANT_ROLES:
            continue
        val = turn.get("value")
        parts.extend(_extract_content_parts(val))
    return parts


def _extract_field_text_or_messages(
    val: Any, *, assistant_only: bool = False
) -> List[str]:
    """Extract text from a field that may be str or list of messages/strings."""
    if isinstance(val, str):
        return [val] if val else []
    if isinstance(val, list):
        msg_parts = _extract_messages_text(val, assistant_only=assistant_only)
        if msg_parts:
            return msg_parts
        str_parts: List[str] = []
        for item in val:
            if isinstance(item, str) and item:
                str_parts.append(item)
            elif isinstance(item, Mapping):
                if assistant_only:
                    role = item.get("role") or item.get("from")
                    if role is not None and role not in _ASSISTANT_ROLES:
                        continue
                for k in ("content", "text", "value"):
                    v = item.get(k)
                    if isinstance(v, str) and v:
                        str_parts.append(v)
        return str_parts
    return []


def _extract_row_parts(row: Mapping[str, Any]) -> List[str]:
    """Extract all text parts across all format fields (PII / length scans)."""
    parts: List[str] = []

    for key in (
        "instruction",
        "input",
        "output",
        "system",
        "response",
        "anchor",
        "positive",
        "negative",
        "text",
        "content",
        "query",
        "golden_doc",
        "answer",
    ):
        val = row.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
        elif isinstance(val, list) and key in ("anchor", "positive", "negative"):
            for s in val:
                if isinstance(s, str) and s:
                    parts.append(s)

    for key in ("prompt", "chosen", "rejected", "completion"):
        val = row.get(key)
        if val is not None:
            parts.extend(_extract_field_text_or_messages(val, assistant_only=False))

    completions = row.get("completions")
    if isinstance(completions, list):
        for c in completions:
            if isinstance(c, str) and c:
                parts.append(c)

    distractors = row.get("distractor_docs")
    if isinstance(distractors, list):
        for d in distractors:
            if isinstance(d, str) and d:
                parts.append(d)

    segments = row.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if isinstance(seg, Mapping):
                t = seg.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)

    if "conversations" in row:
        parts.extend(_extract_conversations_text(row.get("conversations"), assistant_only=False))

    if "messages" in row:
        parts.extend(_extract_messages_text(row.get("messages"), assistant_only=False))

    if "tool_calls" in row:
        parts.extend(_extract_tool_calls_text(row.get("tool_calls")))

    tools = row.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, Mapping):
                func = tool.get("function")
                if isinstance(func, Mapping):
                    desc = func.get("description")
                    if isinstance(desc, str) and desc:
                        parts.append(desc)

    return parts


def _extract_row_text(row: Mapping[str, Any]) -> str:
    """Extract all text across all format fields as a single string."""
    return "\n".join(_extract_row_parts(row))


def _extract_assistant_parts(row: Mapping[str, Any]) -> List[str]:
    """Extract assistant-side text parts (for refusal scans)."""
    parts: List[str] = []

    for key in ("output", "response", "answer"):
        val = row.get(key)
        if isinstance(val, str) and val:
            parts.append(val)

    if "conversations" in row:
        parts.extend(_extract_conversations_text(row.get("conversations"), assistant_only=True))

    if "messages" in row:
        parts.extend(_extract_messages_text(row.get("messages"), assistant_only=True))

    if "tool_calls" in row:
        parts.extend(_extract_tool_calls_text(row.get("tool_calls")))

    if "chosen" in row:
        val = row.get("chosen")
        if val is not None:
            parts.extend(_extract_field_text_or_messages(val, assistant_only=True))

    if "completion" in row:
        val = row.get("completion")
        if val is not None:
            parts.extend(_extract_field_text_or_messages(val, assistant_only=True))

    completions = row.get("completions")
    if isinstance(completions, list):
        for c in completions:
            if isinstance(c, str) and c:
                parts.append(c)

    segments = row.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if isinstance(seg, Mapping) and seg.get("label", True):
                t = seg.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)

    if not parts and ("text" in row or "content" in row):
        is_chat_or_pref = any(
            k in row
            for k in (
                "conversations",
                "messages",
                "instruction",
                "output",
                "prompt",
                "chosen",
                "completion",
                "query",
                "answer",
                "segments",
                "completions",
            )
        )
        if not is_chat_or_pref:
            val = row.get("text") or row.get("content")
            if isinstance(val, str) and val:
                parts.append(val)

    return parts


def _extract_assistant_text(row: Mapping[str, Any]) -> str:
    """Extract assistant-side text as a single string."""
    return "\n".join(_extract_assistant_parts(row))


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
        parts = _extract_row_parts(row)
        if not parts or not any(p.strip() for p in parts):
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                details.append(
                    _truncate_detail(f"rows[{index}]: no extractable text")
                )
            continue
        all_hits = []
        for part in parts:
            if not part:
                continue
            try:
                hits = detect_pii(part)
                if hits:
                    all_hits.extend(hits)
            except (TypeError, ValueError):
                continue
        if all_hits:
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                kinds = sorted({hit.get("kind", "?") for hit in all_hits})
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
    low, high = _check_token_bounds(min_tokens, max_tokens)
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
        if not text or not text.strip():
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                details.append(
                    _truncate_detail(f"rows[{index}]: no extractable text")
                )
            continue
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
        full_text = _extract_row_text(row)
        if not full_text or not full_text.strip():
            num_violations += 1
            if len(details) < _MAX_DETAILS_PER_RESULT:
                details.append(
                    _truncate_detail(f"rows[{index}]: no extractable text")
                )
            continue
        text = _extract_assistant_text(row)
        if not text or not text.strip():
            continue
        if looks_like_refusal(text):
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
    advisory: bool = False,
) -> ExpectationResult:
    """Fail when ``judge_fn(row) < threshold`` on a preference row.

    Rows must carry both ``chosen`` and ``rejected``. ``judge_fn`` returns a
    score in [0, 1]; 1.0 means the judge fully prefers chosen over rejected.

    When ``judge_fn`` is omitted and ``advisory=False``, every preference row
    is flagged as a violation for missing judge configuration (#1433).
    To explicitly trust existing labels without running a judge, set
    ``advisory=True``.
    """
    t = _check_threshold(threshold, field="threshold")
    if not isinstance(advisory, bool):
        raise TypeError("advisory must be bool")
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
            if advisory:
                score: float = 1.0
            else:
                num_violations += 1
                if len(details) < _MAX_DETAILS_PER_RESULT:
                    details.append(
                        _truncate_detail(
                            f"rows[{index}]: no judge configured: set args.judge "
                            "(e.g. ollama://llama3.1) or pass --judge, or remove this expectation"
                        )
                    )
                continue
        else:
            try:
                raw = judge_fn(row)
            except Exception as exc:  # noqa: BLE001 - one bad row mustn't crash the suite
                from soup_cli.eval.judge import JudgeUnavailableError

                if isinstance(exc, JudgeUnavailableError):
                    raise
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
    if judge_fn is None and advisory and not details:
        details.append(f"advisory: no judge ran; {len(materialised)} rows not scored")
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


def _key_repr(key: object) -> str:
    # repr() spells control bytes out, so a hostile key cannot reach the terminal raw.
    return repr(str(key)[:_MAX_NAME_LEN])


_MAX_KEYS_NAMED = 10


def _key_list(keys: Sequence[object]) -> str:
    # Ten keys and a count: a 1 MiB suite can carry tens of thousands of unknown keys,
    # and naming them all made the message nearly as large as the file.
    shown = ", ".join(map(_key_repr, keys[:_MAX_KEYS_NAMED]))
    rest = len(keys) - _MAX_KEYS_NAMED
    return f"{shown} and {rest} more" if rest > 0 else shown


def _validate_entry_keys(index: int, name: str, entry: Mapping[str, Any]) -> None:
    extra = [key for key in entry if key not in _ENTRY_KEYS]
    if not extra:
        return
    misplaced = [key for key in extra if key in _EXPECTATION_ARGS[name]]
    if misplaced:
        raise ValueError(
            f"expectations[{index}]: {_key_list(misplaced)} "
            f"must go under 'args:' for {name}, not beside 'name'"
        )
    raise ValueError(
        f"expectations[{index}]: unknown key(s) {_key_list(extra)}; "
        "an entry takes only 'name' and 'args'"
    )


def _validate_args(index: int, name: str, args: Mapping[str, Any]) -> None:
    accepted = _EXPECTATION_ARGS[name]
    unknown = [key for key in args if key not in accepted]
    if unknown:
        shown = _key_repr(unknown[0])
        if not accepted:
            raise ValueError(f"expectations[{index}]: {name} takes no arguments, got {shown}")
        close = difflib.get_close_matches(str(unknown[0]), sorted(accepted), n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        raise ValueError(
            f"expectations[{index}]: {name} takes no argument {shown}{hint}; "
            f"accepted: {', '.join(sorted(accepted))}"
        )
    merged = {**accepted, **args}
    if name == "expect_token_length_between":
        _check_token_bounds(merged["min_tokens"], merged["max_tokens"])
    elif name == "expect_chosen_preferred_over_rejected_by_judge":
        _check_threshold(merged["threshold"], field="threshold")
        if merged.get("judge") is not None:
            if not isinstance(merged["judge"], str):
                raise TypeError(f"expectations[{index}]: judge must be a string")
            if not merged["judge"].strip():
                raise ValueError(f"expectations[{index}]: judge must be non-empty")
            from soup_cli.eval.gate import _parse_judge_url

            try:
                _parse_judge_url(merged["judge"])
            except ValueError as exc:
                raise ValueError(f"expectations[{index}]: {exc}") from exc
        if not isinstance(merged.get("advisory", False), bool):
            raise TypeError(f"expectations[{index}]: advisory must be a boolean")


def parse_suite_spec(raw: Any) -> SuiteSpec:
    """Validate a suite dict and return a ``SuiteSpec``."""
    if not isinstance(raw, dict):
        raise TypeError("suite spec must be a dict")
    extra = [key for key in raw if key != "expectations"]
    if extra:  # #1483: a top-level key is a typo or an option that does not exist
        raise ValueError(
            f"unknown top-level key(s) {_key_list(extra)}; "
            "a suite takes only 'expectations'"
        )
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
        _validate_entry_keys(index, name, entry)
        args = entry.get("args", {})
        if not isinstance(args, dict):
            raise TypeError(f"expectations[{index}].args must be a dict")
        _validate_args(index, name, args)
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
    args = {**_EXPECTATION_ARGS.get(name, {}), **spec.args}
    if name == "expect_no_pii":
        return expect_no_pii(rows)
    if name == "expect_token_length_between":
        return expect_token_length_between(
            rows,
            min_tokens=args["min_tokens"],
            max_tokens=args["max_tokens"],
        )
    if name == "expect_no_refusal_pattern":
        return expect_no_refusal_pattern(rows)
    if name == "expect_chosen_preferred_over_rejected_by_judge":
        effective_judge = judge_fn
        if effective_judge is None and args.get("judge"):
            effective_judge = build_pairwise_judge_fn(args["judge"])
        return expect_chosen_preferred_over_rejected_by_judge(
            rows,
            judge_fn=effective_judge,
            threshold=args["threshold"],
            advisory=args["advisory"],
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
    "FORMAT_EXTRACTION_RULES",
    "JudgeFn",
    "SUPPORTED_EXPECTATIONS",
    "SuiteReport",
    "SuiteSpec",
    "build_pairwise_judge_fn",
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
