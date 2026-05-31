"""Turn a stream of :class:`Trace` events into preference pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from soup_cli.data.traces.parsers import Trace

SUPPORTED_SIGNALS: tuple[str, ...] = (
    "thumbs_up",
    "regenerations",
    "user_edit",
)


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    source: str = ""

    def to_jsonl_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "source": self.source,
        }


def _group_by_prompt(traces: Iterable[Trace]) -> dict[str, list[Trace]]:
    grouped: dict[str, list[Trace]] = {}
    for trace in traces:
        grouped.setdefault(trace.prompt, []).append(trace)
    return grouped


def _pair_thumbs(traces: Iterable[Trace]) -> Iterator[PreferencePair]:
    grouped = _group_by_prompt(traces)
    for prompt, entries in grouped.items():
        ups = [t for t in entries if t.signal == "thumbs_up"]
        downs = [t for t in entries if t.signal == "thumbs_down"]
        if not ups or not downs:
            continue
        yield PreferencePair(
            prompt=prompt, chosen=ups[0].output,
            rejected=downs[0].output, source="thumbs_up",
        )


def _pair_regenerations(traces: Iterable[Trace]) -> Iterator[PreferencePair]:
    grouped = _group_by_prompt(traces)
    for prompt, entries in grouped.items():
        relevant = [
            t for t in entries
            if t.signal in ("regenerated", "thumbs_up", "thumbs_down")
            or t.regen_order is not None
        ]
        if len(relevant) < 2:
            continue
        ordered = sorted(relevant, key=lambda t: t.regen_order or 0)
        if ordered[0].output == ordered[-1].output:
            continue
        yield PreferencePair(
            prompt=prompt, chosen=ordered[-1].output,
            rejected=ordered[0].output, source="regenerations",
        )


def _pair_user_edit(traces: Iterable[Trace]) -> Iterator[PreferencePair]:
    for trace in traces:
        if trace.signal != "user_edit" or not trace.edited_output:
            continue
        if trace.output == trace.edited_output:
            continue
        yield PreferencePair(
            prompt=trace.prompt, chosen=trace.edited_output,
            rejected=trace.output, source="user_edit",
        )


def build_pairs(
    traces: Iterable[Trace], *, signal: str,
) -> Iterator[PreferencePair]:
    """Produce preference pairs from ``traces`` according to ``signal``."""
    if signal not in SUPPORTED_SIGNALS:
        raise ValueError(
            f"unknown signal '{signal}'. "
            f"Supported: {', '.join(SUPPORTED_SIGNALS)}"
        )
    traces = list(traces)  # materialise for multi-pass signals
    if signal == "thumbs_up":
        yield from _pair_thumbs(traces)
    elif signal == "regenerations":
        yield from _pair_regenerations(traces)
    elif signal == "user_edit":
        yield from _pair_user_edit(traces)
