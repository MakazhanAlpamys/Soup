"""Trace parsers for LangChain / OpenAI-style / Soup-serve JSONL logs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

SUPPORTED_FORMATS: tuple[str, ...] = ("langchain", "openai", "soup-serve")


@dataclass(frozen=True)
class Trace:
    """A single request/response pair extracted from a log."""

    trace_id: str
    prompt: str
    output: str
    signal: str = "none"
    regen_order: Optional[int] = None
    edited_output: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


def _langchain_prompt(inputs: Any) -> str:
    """Flatten LangChain ``inputs`` into a string prompt."""
    if not isinstance(inputs, dict):
        return ""
    messages = inputs.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content") or ""
                parts.append(str(content))
        return "\n".join(parts).strip()
    for key in ("prompt", "input", "query"):
        value = inputs.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(inputs, default=str)


def _langchain_output(outputs: Any) -> str:
    """Extract the assistant response from LangChain ``outputs``."""
    if not isinstance(outputs, dict):
        return ""
    gens = outputs.get("generations")
    if isinstance(gens, list) and gens:
        first = gens[0]
        if isinstance(first, list) and first:
            item = first[0]
        else:
            item = first
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or ""
            return str(text)
    for key in ("output", "response", "text", "content"):
        value = outputs.get(key)
        if isinstance(value, str):
            return value
    return ""


def _langchain_signal(feedback: Any) -> str:
    """Map LangChain feedback to our signal vocabulary."""
    if not isinstance(feedback, list):
        return "none"
    for entry in feedback:
        if not isinstance(entry, dict):
            continue
        if entry.get("key") == "thumbs":
            score = entry.get("score")
            if score == 1 or score == "up":
                return "thumbs_up"
            if score == 0 or score == "down":
                return "thumbs_down"
    return "none"


def parse_langchain(events: Iterable[Any]) -> Iterator[Trace]:
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        outputs = event.get("outputs")
        if not outputs:
            continue
        prompt = _langchain_prompt(event.get("inputs"))
        output = _langchain_output(outputs)
        if not prompt or not output:
            continue
        trace_id = str(event.get("id") or f"lc-{idx}")
        signal = _langchain_signal(event.get("feedback"))
        yield Trace(
            trace_id=trace_id, prompt=prompt, output=output, signal=signal,
        )


# ---------------------------------------------------------------------------
# OpenAI-style
# ---------------------------------------------------------------------------


def _openai_prompt(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _openai_output(choices: Any) -> str:
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return str(first.get("text") or "")


def _openai_signal(feedback: Any) -> str:
    if not isinstance(feedback, dict):
        return "none"
    rating = feedback.get("rating")
    if rating in ("up", 1, "thumbs_up"):
        return "thumbs_up"
    if rating in ("down", 0, "thumbs_down"):
        return "thumbs_down"
    return "none"


def parse_openai(events: Iterable[Any]) -> Iterator[Trace]:
    # Track regenerations for the same prompt via "regenerated_from"
    regen_orders: dict[str, int] = {}
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        prompt = _openai_prompt(event.get("messages"))
        output = _openai_output(event.get("choices"))
        if not prompt or not output:
            continue
        trace_id = str(event.get("id") or f"oai-{idx}")
        signal = _openai_signal(event.get("feedback"))
        regen_from = event.get("regenerated_from")
        if regen_from:
            order = regen_orders.get(prompt, 0) + 1
            regen_orders[prompt] = order
            signal = "regenerated" if signal == "none" else signal
            yield Trace(
                trace_id=trace_id, prompt=prompt, output=output,
                signal=signal, regen_order=order,
            )
        else:
            regen_orders.setdefault(prompt, 0)
            yield Trace(
                trace_id=trace_id, prompt=prompt, output=output,
                signal=signal, regen_order=regen_orders[prompt],
            )


# ---------------------------------------------------------------------------
# Soup-serve (directory of JSONL files)
# ---------------------------------------------------------------------------


def parse_soup_serve(path: str) -> Iterator[Trace]:
    dir_path = Path(path)
    if not dir_path.is_dir():
        return
    for file_path in sorted(dir_path.glob("*.jsonl")):
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            prompt = str(event.get("prompt") or "")
            output = str(event.get("response") or event.get("output") or "")
            if not prompt or not output:
                continue
            signal = _openai_signal(event.get("feedback"))
            yield Trace(
                trace_id=str(event.get("id") or file_path.stem),
                prompt=prompt, output=output, signal=signal,
            )
