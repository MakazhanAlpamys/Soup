"""Known Kimi K2 resident remote-code boundary; no model or network imports.

#2104: the resident (non-streamed) transformers route is wired for Kimi-K2-Base and
Kimi-K2-Instruct on sft/grpo, so the refusal lifts there; every other task on Base/Instruct
still points at the guarded streamed route, and Kimi-K2-Thinking / Kimi-K2-Instruct-0905
stay refused for every task ("no supported route yet").
"""

import os

KIMI_K2_RESIDENT_REASON = "Resident Kimi K2 is unsupported on Transformers 5.x"
KIMI_K2_GUARDED_STREAM_REASON = (
    KIMI_K2_RESIDENT_REASON + ". Use the guarded training.stream_layers: true route "
    "(#2107), with --trust-remote-code for the tokenizer and pip install tiktoken. "
    "See docs/performance-and-quantization.md, Kimi K2: "
    "CPU-only parity on a tiny K2-like checkpoint; full K2/GPU training is unverified."
)
KIMI_K2_NO_SUPPORTED_ROUTE_REASON = KIMI_K2_RESIDENT_REASON + ". No supported route yet."
KIMI_K2_RESIDENT_REASONS = {
    "moonshotai/kimi-k2-base": KIMI_K2_GUARDED_STREAM_REASON,
    "moonshotai/kimi-k2-instruct": KIMI_K2_GUARDED_STREAM_REASON,
    "moonshotai/kimi-k2-instruct-0905": KIMI_K2_NO_SUPPORTED_ROUTE_REASON,
    "moonshotai/kimi-k2-thinking": KIMI_K2_NO_SUPPORTED_ROUTE_REASON,
}
#: Bases admitted on the resident route, for these tasks only (the resident route's v1
#: wiring covers sft and grpo); refused for every other task, where the streamed route
#: remains the remedy.
KIMI_K2_RESIDENT_ADMITTED_TASKS = frozenset({"sft", "grpo"})
#: Bases with no supported route at all: refused for every task.
KIMI_K2_RESIDENT_ALWAYS_REFUSED = frozenset({
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-thinking",
})


def is_resident_kimi_k2_base(base: str) -> bool:
    """True iff ``base`` is one of the four known K2 hub ids (case-insensitively) and not
    a local directory, which the carve-out keeps out of the guard."""
    return base.casefold() in KIMI_K2_RESIDENT_REASONS and not os.path.isdir(base)


def resident_kimi_k2_refusal(
    base: str, *, backend: str, hub: str, stream_layers: bool, task: str | None = None,
) -> str | None:
    """Only qualify known HF IDs on the resident Transformers path.

    Base/Instruct are admitted (``None``) on sft/grpo and refused with the streamed-route
    remedy for any other task; Thinking/0905 are refused for every task. ``task=None``
    refuses Base/Instruct too: callers that cannot name a task get the safe answer.
    """
    if backend == "transformers" and hub == "hf" and not stream_layers:
        folded = base.casefold()
        reason = KIMI_K2_RESIDENT_REASONS.get(folded)
        # A real relative local directory can have the same two-component name
        # as a Hub ID. Do not classify its potentially repaired code by its name.
        if reason is None or os.path.isdir(base):
            return None
        if folded in KIMI_K2_RESIDENT_ALWAYS_REFUSED:
            return reason
        if task not in KIMI_K2_RESIDENT_ADMITTED_TASKS:
            return reason
    return None
