"""v0.53.0 Part D — Train-time advanced precision schema helpers.

Three new TrainingConfig surfaces ship this release (schema-only):

* ``fp8_attention: bool`` — extend the v0.28.0 FP8 menu to apply FP8 to
  attention (axolotl-parity flag). Requires ``quantization_aware='fp8'``.
* ``nvfp4: bool`` — Blackwell-only NVFP4 training (unsloth + axolotl). Gated
  to non-mlx text-modality training.
* ``unsloth_bnb_4bit: bool`` — promote Unsloth Dynamic 4-bit to a native
  TrainingConfig flag (previously inferable only from ``backend='unsloth'``
  + ``quantization='4bit'``). When True, requires ``backend='unsloth'`` and
  ``quantization='4bit'``.

Live wiring lands in v0.53.1 (mirrors v0.50.0 / v0.52.0 stub-then-live).
"""

from __future__ import annotations


def validate_fp8_attention_compat(
    *,
    fp8_attention: bool,
    quantization_aware: object,
    backend: str,
) -> None:
    """Schema-time gate for ``fp8_attention=True``.

    Rejects:
    - non-bool ``fp8_attention`` (defence-in-depth).
    - ``fp8_attention=True`` without ``quantization_aware='fp8'`` (silent
      no-op footgun — mirrors v0.32.0 ``loss_spike_recovery`` policy).
    - non-string / empty ``backend``.
    - ``backend == 'mlx'`` (MLX path has no FP8 attention kernel).
    """
    if not isinstance(fp8_attention, bool):
        raise TypeError(
            f"fp8_attention must be bool, got {type(fp8_attention).__name__}"
        )
    if not fp8_attention:
        return
    if isinstance(backend, bool):
        raise TypeError(f"backend must not be bool, got {backend!r}")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string")
    # Check quantization_aware prerequisite BEFORE backend gate so a YAML
    # missing both gets the more actionable error (matches v0.52.0
    # validate_bitnet_compat ordering).
    if quantization_aware != "fp8":
        raise ValueError(
            "fp8_attention=true requires training.quantization_aware='fp8' "
            f"(got quantization_aware={quantization_aware!r})"
        )
    if backend == "mlx":
        raise ValueError(
            "fp8_attention=true is not supported on backend=mlx"
        )


def validate_nvfp4_compat(
    *,
    nvfp4: bool,
    backend: str,
    modality: str,
) -> None:
    """Schema-time gate for ``nvfp4=True``.

    NVFP4 is Blackwell-only and CUDA-only; the *runtime* SM-capability
    check fires at trainer-construction time. This schema gate is the
    cheap defence-in-depth layer.
    """
    if not isinstance(nvfp4, bool):
        raise TypeError(f"nvfp4 must be bool, got {type(nvfp4).__name__}")
    if not nvfp4:
        return
    for name, value in (("backend", backend), ("modality", modality)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if backend == "mlx":
        raise ValueError(
            "nvfp4=true is not supported on backend=mlx "
            "(NVFP4 is CUDA-only — requires Blackwell)"
        )
    if modality != "text":
        raise ValueError(
            f"nvfp4=true is wired for modality='text' only; "
            f"got modality={modality!r}"
        )


def validate_unsloth_bnb_4bit_compat(
    *,
    unsloth_bnb_4bit: bool,
    backend: str,
    quantization: str,
) -> None:
    """Schema-time gate for ``unsloth_bnb_4bit=True``.

    Promotes "Unsloth Dynamic 4-bit" from "inferable from backend+quant"
    to a native flag. The flag requires:
    - ``backend == 'unsloth'`` (otherwise silently no-op).
    - ``quantization == '4bit'`` (the BNB Dynamic 4-bit path; conflicts
      with the v0.38.0 Quant Menu formats which raise loudly at runtime).
    """
    if not isinstance(unsloth_bnb_4bit, bool):
        raise TypeError(
            f"unsloth_bnb_4bit must be bool, "
            f"got {type(unsloth_bnb_4bit).__name__}"
        )
    if not unsloth_bnb_4bit:
        return
    for name, value in (("backend", backend), ("quantization", quantization)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if backend != "unsloth":
        raise ValueError(
            f"unsloth_bnb_4bit=true requires backend='unsloth'; "
            f"got backend={backend!r}"
        )
    if quantization != "4bit":
        raise ValueError(
            f"unsloth_bnb_4bit=true requires quantization='4bit'; "
            f"got quantization={quantization!r}"
        )


def apply_fp8_attention() -> None:
    """Live FP8-attention wiring — deferred to v0.53.1."""
    raise NotImplementedError(
        "fp8_attention live wiring deferred to v0.53.1. Schema accepts the "
        "flag but no torchao FP8 attention swap is registered yet."
    )


def apply_nvfp4() -> None:
    """Live NVFP4 wiring — deferred to v0.53.1."""
    raise NotImplementedError(
        "NVFP4 live wiring deferred to v0.53.1. Schema accepts the flag "
        "but no Blackwell-FP4 quant prep is registered yet."
    )
