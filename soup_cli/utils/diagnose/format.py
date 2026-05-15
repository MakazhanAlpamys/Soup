"""Format validity probe (v0.56.0).

Verifies that the adapter's outputs still pass JSON / regex / tool-call
validators (reuses the v0.25 RLVR scoring surface in spirit; pure
in-tree so the probe never executes user code).
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from soup_cli.utils.diagnose._common import (
    GeneratorFn,
    call_generator,
    merge_evidence,
    require_prompts,
)
from soup_cli.utils.diagnose.report import FailureScore, classify_score

_VALID_KINDS = frozenset({"json", "regex", "tool_call"})
_MAX_REGEX_LEN = 2048
_MAX_OUTPUT_LEN = 64 * 1024


def is_valid_json(text: str) -> bool:
    if not isinstance(text, str) or "\x00" in text or len(text) > _MAX_OUTPUT_LEN:
        return False
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return False
    return True


def matches_regex(text: str, pattern: str) -> bool:
    if not isinstance(text, str) or not isinstance(pattern, str):
        return False
    if len(text) > _MAX_OUTPUT_LEN or len(pattern) > _MAX_REGEX_LEN:
        return False
    if "\x00" in text or "\x00" in pattern:
        return False
    try:
        compiled = re.compile(pattern)
    except re.error:
        return False
    # Best-effort ReDoS probe — catastrophic-backtracking patterns surface
    # on a benign 128-char canary before they ever touch a real model
    # output (mirrors v0.41.0 Part B / v0.55.0 policy).
    try:
        compiled.search("a" * 128)
    except (re.error, RuntimeError):
        return False
    return bool(compiled.search(text))


def is_valid_tool_call(text: str) -> bool:
    """Tool-call rows are JSON with a top-level ``tool_calls`` list."""
    if not isinstance(text, str) or "\x00" in text or len(text) > _MAX_OUTPUT_LEN:
        return False
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    tool_calls = payload.get("tool_calls")
    return isinstance(tool_calls, list) and len(tool_calls) >= 1


def score_format(
    prompts: Sequence[str],
    adapter_gen: GeneratorFn,
    *,
    kind: str = "json",
    regex_pattern: str = "",
) -> FailureScore:
    """Score the fraction of adapter outputs that pass the chosen validator."""
    if not isinstance(kind, str) or kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    if kind == "regex" and not regex_pattern:
        raise ValueError("kind='regex' requires a non-empty regex_pattern")
    prompts_list = require_prompts(prompts, max_count=10_000)
    if not prompts_list:
        return FailureScore(
            mode="format",
            score=1.0,
            verdict="OK",
            evidence="no prompts; nothing to check",
        )
    valid = 0
    for prompt in prompts_list:
        output = call_generator(adapter_gen, prompt)
        if kind == "json":
            ok = is_valid_json(output)
        elif kind == "regex":
            ok = matches_regex(output, regex_pattern)
        else:  # tool_call
            ok = is_valid_tool_call(output)
        if ok:
            valid += 1
    score = valid / len(prompts_list)
    verdict = classify_score(score)
    evidence = merge_evidence(
        {"kind": kind, "valid": valid, "total": len(prompts_list)}
    )
    return FailureScore(
        mode="format",
        score=score,
        verdict=verdict,
        evidence=evidence,
    )
