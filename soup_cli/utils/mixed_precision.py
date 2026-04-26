"""Auto mixed-precision picker (v0.32.0 Part C).

Maps a model + GPU compute capability to the best mixed-precision setting:
``bf16`` (Ampere+, most modern models), ``fp16`` (Turing or known
fp16-stable models), or ``no`` (Pascal and older).

Quirk map is keyed on a lower-cased substring of the model id so we catch
both ``Qwen/Qwen2-7B-Instruct`` and ``alibaba/Qwen2.5-3B`` with one entry.
"""

from __future__ import annotations

from typing import Literal

# Compute capability thresholds.
BF16_MIN_CC = 8.0   # Ampere
FP16_MIN_CC = 6.0   # Pascal
MAX_MODEL_NAME_LEN = 200

# Known fp16-stable / bf16-unstable model families.
# Value: preferred precision when CC supports it.
KNOWN_PRECISION_QUIRKS: dict[str, str] = {
    "qwen2": "fp16",
    "qwen2.5": "fp16",
    "phi-3": "fp16",
    "phi-3.5": "fp16",
    "phi-4": "bf16",
    "gemma-2": "bf16",
    "mistral": "bf16",
    "llama-3": "bf16",
    "llama-2": "bf16",
}


def pick_mixed_precision(
    model_name: str, compute_capability: float,
) -> Literal["bf16", "fp16", "no"]:
    """Pick mixed-precision mode for a model + GPU.

    - cc < 6.0  → ``"no"`` (Pascal lacks reliable fp16 tensor cores)
    - cc < 8.0  → ``"fp16"`` (no bf16 support)
    - cc >= 8.0 → ``"bf16"`` unless model is in the fp16-stable quirks map
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")
    if "\x00" in model_name:
        raise ValueError("model_name must not contain null bytes")
    if len(model_name) > MAX_MODEL_NAME_LEN:
        raise ValueError(
            f"model_name must be <= {MAX_MODEL_NAME_LEN} chars, got {len(model_name)}"
        )
    if not isinstance(compute_capability, (int, float)):
        raise ValueError(
            f"compute_capability must be a number, got {type(compute_capability)}"
        )
    if compute_capability < 0:
        raise ValueError(
            f"compute_capability must be non-negative, got {compute_capability}"
        )

    if compute_capability < FP16_MIN_CC:
        return "no"

    name_lc = model_name.lower()
    quirk: str | None = None
    # Sort longer substrings first so multi-version pairs work correctly:
    # ``qwen2.5`` wins over ``qwen2``, ``phi-3.5`` wins over ``phi-3``.
    # If you add a new family with multiple versions, the longest substring
    # always wins — no manual ordering of the dict is required.
    for substring in sorted(KNOWN_PRECISION_QUIRKS, key=len, reverse=True):
        if substring in name_lc:
            quirk = KNOWN_PRECISION_QUIRKS[substring]
            break

    if compute_capability < BF16_MIN_CC:
        # Turing / Volta cannot do bf16 — fall back to fp16 regardless.
        return "fp16"

    if quirk is None:
        return "bf16"
    return "bf16" if quirk == "bf16" else "fp16"
