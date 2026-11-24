"""Shared precision policy for native DeepSeek-V3 and the Kimi K2 rewrite."""

import re

ROUTER_WEIGHT_KEY = "mlp.gate.weight"


def router_weight_uses_fp32() -> bool:
    """Whether Transformers keeps every DeepSeek-V3 router weight in strict FP32."""
    from transformers import DeepseekV3ForCausalLM

    patterns = getattr(DeepseekV3ForCausalLM, "_keep_in_fp32_modules_strict", None) or ()
    # Transformers searches checkpoint keys with these patterns, expanding '*' to '.*'.
    # The model declares a shared policy for the routers in its decoder layers.
    checkpoint_key = f"model.layers.1.{ROUTER_WEIGHT_KEY}"
    return any(re.search(pattern.replace("*", ".*"), checkpoint_key) for pattern in patterns)
