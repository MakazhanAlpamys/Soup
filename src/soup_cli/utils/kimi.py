"""Known Kimi K2 resident remote-code boundary; no model or network imports."""

import os

KIMI_K2_RESIDENT_BASES = frozenset({
    "moonshotai/kimi-k2-base",
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-thinking",
})
KIMI_K2_RESIDENT_REASON = "Resident Kimi K2 is unsupported on Transformers 5.x"


def resident_kimi_k2_refusal(
    base: str, *, backend: str, hub: str, stream_layers: bool,
) -> str | None:
    """Only qualify known HF IDs on the resident Transformers path."""
    if (backend == "transformers" and hub == "hf" and not stream_layers
            and base.casefold() in KIMI_K2_RESIDENT_BASES):
        # A real relative local directory can have the same two-component name
        # as a Hub ID. Do not classify its potentially repaired code by its name.
        if not os.path.isdir(base):
            return KIMI_K2_RESIDENT_REASON
    return None
