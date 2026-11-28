"""Explicit native-mask/external-unmasked attention for isolated research runs.

The external build needs separate numerical qualification. A CPU stand-in can
test routing and registry lifetime; it cannot establish FlashAttention parity.
"""

from contextlib import contextmanager


@contextmanager
def native_masks_external_unmasked(*, fixture_external=None):
    """Preserve SDPA's prepared masks and arithmetic for explicitly masked inputs.

    Use in a dedicated process with a qualified external CUDA BF16 build. The
    registry changes are scoped to this context and restored after exceptions.
    """
    import torch
    from transformers.integrations.flash_attention import flash_attention_forward
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original_attention = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    original_mask = ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"]
    if getattr(original_attention, "_soup_explicit_native_mask_route", False):
        raise ValueError("Hybrid attention route is already installed")
    external = fixture_external or flash_attention_forward
    counts = {
        "external_unmasked_calls": 0,
        "native_explicit_mask_calls": 0,
        "fixture_external": fixture_external is not None,
    }

    def dispatch(module, query, key, value, attention_mask, **kwargs):
        if fixture_external is None and (
            query.device.type != "cuda" or query.dtype != torch.bfloat16
        ):
            raise ValueError("Research route is restricted to CUDA BF16")
        if attention_mask is None:
            counts["external_unmasked_calls"] += 1
            return external(module, query, key, value, attention_mask, **kwargs)
        if attention_mask.ndim != 4:
            raise ValueError("Native route requires the native prepared 4D mask")
        counts["native_explicit_mask_calls"] += 1
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    dispatch._soup_explicit_native_mask_route = True
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", dispatch)
    ALL_MASK_ATTENTION_FUNCTIONS.register("flash_attention_2", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    try:
        yield counts
    finally:
        ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", original_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register("flash_attention_2", original_mask)
