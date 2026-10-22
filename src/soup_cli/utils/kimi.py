"""Known Kimi K2 resident remote-code boundary; no model or network imports."""

import os

KIMI_K2_RESIDENT_REASON = "Resident Kimi K2 is unsupported on Transformers 5.x"
KIMI_K2_PENDING_STREAM_REASON = (
    KIMI_K2_RESIDENT_REASON + ". The native training.stream_layers route is tracked in #2107, "
    "not available until it merges."
)
KIMI_K2_NO_SUPPORTED_ROUTE_REASON = KIMI_K2_RESIDENT_REASON + ". No supported route yet."
KIMI_K2_RESIDENT_REASONS = {
    "moonshotai/kimi-k2-base": KIMI_K2_PENDING_STREAM_REASON,
    "moonshotai/kimi-k2-instruct": KIMI_K2_PENDING_STREAM_REASON,
    "moonshotai/kimi-k2-instruct-0905": KIMI_K2_NO_SUPPORTED_ROUTE_REASON,
    "moonshotai/kimi-k2-thinking": KIMI_K2_NO_SUPPORTED_ROUTE_REASON,
}


def resident_kimi_k2_refusal(
    base: str, *, backend: str, hub: str, stream_layers: bool,
) -> str | None:
    """Only qualify known HF IDs on the resident Transformers path."""
    if backend == "transformers" and hub == "hf" and not stream_layers:
        reason = KIMI_K2_RESIDENT_REASONS.get(base.casefold())
        # A real relative local directory can have the same two-component name
        # as a Hub ID. Do not classify its potentially repaired code by its name.
        if reason is not None and not os.path.isdir(base):
            return reason
    return None
