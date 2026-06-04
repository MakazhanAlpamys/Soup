"""``soup distill-prompt`` — distill prompt-heavy traces into a small FT plan (v0.68.0 Part B).

Bridge between prompt-engineering and FT worlds: take a JSONL of
large-prompt teacher calls (GPT-5 / Claude / etc.) and prepare a
distillation dataset targeting a small student model. Live dataset
preparation lands in v0.71.13 (#226) via the v0.20.0 provider helpers
(composes with v0.70 Part B cross-tokenizer KD).

Public surface:

- ``SUPPORTED_DISTILL_STRATEGIES`` — closed frozenset {sft, preference, kl}
- ``validate_distill_strategy(name)`` — bool-first / null-byte / case-insensitive
- ``validate_teacher_id`` / ``validate_student_id`` — null-byte / oversize / bool
- ``validate_traces_path(path)`` — cwd containment + symlink rejection
- ``DistillPromptPlan`` frozen dataclass
- ``build_distill_prompt_plan(...)`` factory
- ``prepare_distill_dataset(plan)`` — live teacher/student dataset prep (v0.71.13 #226)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional

from soup_cli.utils.paths import (
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
)

_LOG = logging.getLogger("soup.distill_prompt")

SUPPORTED_DISTILL_STRATEGIES: frozenset[str] = frozenset({"sft", "preference", "kl"})

_MAX_STRATEGY_LEN = 32
_MAX_MODEL_ID_LEN = 512
_MAX_TRACE_ROWS = 1_000_000
_MAX_PROMPT_CHARS = 100_000


def validate_distill_strategy(name: object) -> str:
    """Return canonical lowercase strategy name."""
    if isinstance(name, bool):
        raise TypeError("strategy must not be bool")
    if not isinstance(name, str):
        raise TypeError("strategy must be str")
    if not name:
        raise ValueError("strategy must be non-empty")
    if "\x00" in name:
        raise ValueError("strategy must not contain null bytes")
    if len(name) > _MAX_STRATEGY_LEN:
        raise ValueError(
            f"strategy length {len(name)} > {_MAX_STRATEGY_LEN}"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_DISTILL_STRATEGIES:
        raise ValueError(
            f"unknown strategy {name!r}; supported: "
            + ", ".join(sorted(SUPPORTED_DISTILL_STRATEGIES))
        )
    return canonical


def _validate_model_id(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > _MAX_MODEL_ID_LEN:
        raise ValueError(f"{field} length {len(value)} > {_MAX_MODEL_ID_LEN}")
    return value


def validate_teacher_id(value: object) -> str:
    """Validate a teacher model id (HF repo id or local path-shape)."""
    return _validate_model_id(value, field="teacher")


def validate_student_id(value: object) -> str:
    """Validate a student model id."""
    return _validate_model_id(value, field="student")


def validate_traces_path(path: object) -> str:
    """Validate a traces JSONL path (cwd-contained, no symlink)."""
    if isinstance(path, bool):
        raise TypeError("traces_path must not be bool")
    if not isinstance(path, str):
        raise TypeError("traces_path must be str")
    enforce_under_cwd_and_no_symlink(path, field="traces_path")
    return os.path.realpath(path)


def _validate_output_path(path: object) -> str:
    if isinstance(path, bool):
        raise TypeError("output_path must not be bool")
    if not isinstance(path, str):
        raise TypeError("output_path must be str")
    if not path:
        raise ValueError("output_path must be non-empty")
    if "\x00" in path:
        raise ValueError("output_path must not contain null bytes")
    return path


@dataclass(frozen=True)
class DistillPromptPlan:
    """A resolved distill-prompt plan."""

    traces_path: str
    teacher: str
    student: str
    strategy: str
    output_path: str

    def __post_init__(self) -> None:
        validate_traces_path(self.traces_path)
        validate_teacher_id(self.teacher)
        validate_student_id(self.student)
        object.__setattr__(
            self, "strategy", validate_distill_strategy(self.strategy)
        )
        _validate_output_path(self.output_path)


def build_distill_prompt_plan(
    *,
    traces_path: str,
    teacher: str,
    student: str,
    strategy: str,
    output_path: str,
) -> DistillPromptPlan:
    """Validate inputs and return a frozen ``DistillPromptPlan``."""
    return DistillPromptPlan(
        traces_path=traces_path,
        teacher=teacher,
        student=student,
        strategy=validate_distill_strategy(strategy),
        output_path=output_path,
    )


def extract_prompt(row: Mapping[str, Any]) -> Optional[str]:
    """Extract the teacher-facing prompt from a trace row.

    Handles the common trace shapes: an explicit ``prompt`` / ``input`` /
    ``instruction`` / ``question`` field, or the last user turn of a
    ``messages`` list. Returns ``None`` when no prompt can be found.
    """
    if not isinstance(row, Mapping):
        return None
    for key in ("prompt", "input", "instruction", "question", "query"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value[:_MAX_PROMPT_CHARS]
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if (
                isinstance(msg, Mapping)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and msg["content"].strip()
            ):
                return str(msg["content"])[:_MAX_PROMPT_CHARS]
    return None


def _read_traces(traces_path: str) -> List[Mapping[str, Any]]:
    """Read trace rows (cwd-contained, symlink-safe, O_NOFOLLOW)."""
    canonical = enforce_under_cwd_and_no_symlink(traces_path, "traces_path")
    rows: List[Mapping[str, Any]] = []
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(canonical, flags)
    with os.fdopen(fd, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, Mapping):
                rows.append(obj)
            if len(rows) >= _MAX_TRACE_ROWS:
                break
    return rows


def _build_provider_fn(
    provider: str, model: str, *, base_url: Optional[str], temperature: float
) -> "Callable[[str], Mapping[str, Any]]":
    """Lazy-build a ``judge(prompt) -> {'text': str}`` callable.

    Reuses the v0.20.0 provider helpers (Ollama / Anthropic / vLLM) via
    ``data_forge.make_judge_provider_fn``. Anthropic reads the key from the
    environment; Ollama / vLLM are SSRF-validated loopback by default.
    """
    from soup_cli.utils.data_forge import make_judge_provider_fn

    return make_judge_provider_fn(
        provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


def prepare_distill_dataset(
    plan: DistillPromptPlan,
    *,
    provider: str = "ollama",
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_rows: Optional[int] = None,
    teacher_fn: "Optional[Callable[[str], Mapping[str, Any]]]" = None,
    student_fn: "Optional[Callable[[str], Mapping[str, Any]]]" = None,
) -> int:
    """Prepare a distillation dataset from prompt-heavy traces (v0.71.13 #226).

    For each prompt in ``plan.traces_path`` the teacher model is called once;
    the output row depends on ``plan.strategy``:

    - ``sft`` -> ``{messages: [user, assistant=teacher_response]}`` (feed to
      ``soup train --task sft``).
    - ``preference`` -> ``{prompt, chosen=teacher_response,
      rejected=student_baseline}`` (feed to ``soup train --task dpo``). The
      student model is called once per prompt for the rejected response.
    - ``kl`` -> the same ``{messages}`` rows as ``sft`` — ``DistillTrainerWrapper``
      (``soup train --task distill``) computes the per-token logit-KL live, so
      no pre-computed logprobs are emitted; cross-tokenizer projection is the
      v0.70 Part B deliverable.

    The ``teacher_fn`` / ``student_fn`` seams default to real provider calls
    (``make_judge_provider_fn``); tests inject fast fakes. Returns the number
    of rows written. ``max_rows`` caps the number of rows *written* (not the
    number of teacher calls — a prompt whose teacher reply is empty still
    consumes a call without producing a row).
    """
    if not isinstance(plan, DistillPromptPlan):
        raise TypeError("plan must be DistillPromptPlan")
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive int or None")

    teacher = teacher_fn
    if teacher is None:
        teacher = _build_provider_fn(
            provider, plan.teacher, base_url=base_url, temperature=temperature
        )
    student = student_fn
    if plan.strategy == "preference" and student is None:
        student = _build_provider_fn(
            provider, plan.student, base_url=base_url, temperature=temperature
        )

    rows = _read_traces(plan.traces_path)
    out_lines: List[str] = []
    for row in rows:
        if max_rows is not None and len(out_lines) >= max_rows:
            break
        prompt = extract_prompt(row)
        if prompt is None:
            continue
        try:
            t_reply = teacher(prompt)
        except Exception as exc:  # noqa: BLE001 — provider error variety
            _LOG.debug("teacher call failed: %s", exc)
            continue
        t_text = t_reply.get("text") if isinstance(t_reply, Mapping) else None
        if not isinstance(t_text, str) or not t_text.strip():
            continue

        if plan.strategy in ("sft", "kl"):
            out_lines.append(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": t_text},
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        else:  # preference
            try:
                s_reply = student(prompt) if student is not None else None
            except Exception as exc:  # noqa: BLE001
                _LOG.debug("student call failed: %s", exc)
                continue
            s_text = s_reply.get("text") if isinstance(s_reply, Mapping) else None
            if not isinstance(s_text, str) or not s_text.strip():
                continue
            out_lines.append(
                json.dumps(
                    {"prompt": prompt, "chosen": t_text, "rejected": s_text},
                    ensure_ascii=False,
                )
            )

    text = "\n".join(out_lines) + ("\n" if out_lines else "")
    atomic_write_text(text, plan.output_path, field="output_path")
    return len(out_lines)


__all__ = [
    "SUPPORTED_DISTILL_STRATEGIES",
    "validate_distill_strategy",
    "validate_teacher_id",
    "validate_student_id",
    "validate_traces_path",
    "extract_prompt",
    "DistillPromptPlan",
    "build_distill_prompt_plan",
    "prepare_distill_dataset",
]
