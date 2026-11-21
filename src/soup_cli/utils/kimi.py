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


def resident_kimi_k2_quantization_refusal(quantization: str) -> str:
    """The config-level refusal for a non-none ``training.quantization`` on the resident
    Kimi K2 route (#2104, review round 2): the route loads bf16 only, so the refusal must
    fire at config validation — not halfway through a run, inside ``load_resident_model``."""
    return (
        f"resident Kimi K2 route: training.quantization {quantization!r} is refused — the "
        "resident route loads bf16 only; set training.quantization: none for this route, "
        "or train 4-bit through training.stream_layers: true (--trust-remote-code for the "
        "tokenizer and pip install tiktoken; see docs/performance-and-quantization.md, Kimi K2)."
    )


class _ResidentKimiK2QuantizationRefusal(str):
    """Marks :func:`resident_kimi_k2_quantization_refusal` results.

    The config schema appends its remote-code context ("Its remote code uses removed
    APIs … Choose a compatible base.") to the route refusals only; that advice would
    contradict a quantization refusal, which instead names the two ways forward.
    """
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
    modality: str = "text", quantization: str | None = None,
) -> str | None:
    """Only qualify known HF IDs on the resident Transformers path.

    Base/Instruct are admitted (``None``) on sft/grpo and refused with the streamed-route
    remedy for any other task; Thinking/0905 are refused for every task. ``task=None``
    refuses Base/Instruct too: callers that cannot name a task get the safe answer. The
    admission covers ``modality: text`` only: sft's vision/audio setup calls
    ``from_pretrained`` directly, bypassing the resident loader, so any other modality is
    refused with the route reason (#2104, review round 3). The admission reaches
    sft/grpo on Base/Instruct only with ``quantization`` ``None`` or ``"none"``: the
    resident route loads bf16 only, so any other quantization is refused here, at config
    validation, instead of failing inside ``load_resident_model`` (#2104, review round
    2). The returned refusal is a
    :class:`_ResidentKimiK2QuantizationRefusal` instance in that case, so the schema can
    skip the remote-code suffix that belongs to the route refusals.
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
        if task not in KIMI_K2_RESIDENT_ADMITTED_TASKS or modality != "text":
            return reason
        if quantization not in (None, "none"):
            return _ResidentKimiK2QuantizationRefusal(
                resident_kimi_k2_quantization_refusal(quantization)
            )
    return None
