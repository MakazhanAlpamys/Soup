"""v0.53.0 Part D — Train-time advanced precision (schema + live wiring).

Three TrainingConfig surfaces ship here:

* ``fp8_attention: bool`` — extend the v0.28.0 FP8 menu to apply FP8 to
  attention (axolotl-parity flag). Requires ``quantization_aware='fp8'``.
* ``nvfp4: bool`` — Blackwell-only NVFP4 training (unsloth + axolotl). Gated
  to non-mlx text-modality training.
* ``unsloth_bnb_4bit: bool`` — promote Unsloth Dynamic 4-bit to a native
  TrainingConfig flag (previously inferable only from ``backend='unsloth'``
  + ``quantization='4bit'``). When True, requires ``backend='unsloth'`` and
  ``quantization='4bit'``.

v0.71.21 #141 lifts the two ``apply_*`` stubs to live, BETA hw-gated code:

* :func:`apply_fp8_attention` converts the attention-projection linears to
  torchao ``Float8Linear`` training modules (Hopper+ gate, SM >= 9.0).
* :func:`apply_nvfp4` routes the model through torchao's ``NVFP4Config``
  quantisation (Blackwell gate — SM 10.0 datacenter B100/B200/GB200 or
  SM 12.0 consumer RTX 50-series).

Both raise friendly ``RuntimeError`` on missing hardware / torchao instead
of silently no-opping; the trainer-side wiring in
``utils/v028_features.apply_v028_speed_memory`` degrades those to yellow
advisories so a training kick-off never crashes on instrumentation.
"""

from __future__ import annotations

# Attention-projection module names (last FQN component). Covers the
# separate-QKV Llama/Mistral/Qwen/Phi shape, GPT-2's fused ``c_attn``,
# Phi-3 / GPT-NeoX fused variants, and encoder-style ``out_proj``.
_ATTENTION_PROJ_NAMES: frozenset[str] = frozenset({
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "qkv_proj",
    "c_attn",
    "query_key_value",
    "Wqkv",
})

# NVIDIA Blackwell compute capability: SM 10.0 (B100 / B200 / GB200
# datacenter) and SM 12.0 (RTX 50-series consumer). Anything >= 10 is
# Blackwell-family; Hopper is SM 9.x.
_BLACKWELL_MIN_CC_MAJOR = 10


def is_attention_projection(fqn: object) -> bool:
    """Return True when ``fqn``'s last component names an attention projection.

    Defensive surface — returns False (never raises) on non-string / empty /
    null-byte input, matching the project model-detection policy
    (``is_gemma4_model`` / ``is_known_vlm_base``).
    """
    if not isinstance(fqn, str) or not fqn or "\x00" in fqn:
        return False
    if len(fqn) > 4096:  # defensive cap, mirrors sibling detectors
        return False
    return fqn.rsplit(".", 1)[-1] in _ATTENTION_PROJ_NAMES


def is_blackwell_gpu() -> bool:
    """Return True when a Blackwell-family GPU (SM >= 10.0) is detected."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(0)
        return major >= _BLACKWELL_MIN_CC_MAJOR
    except (ImportError, RuntimeError, AssertionError):
        return False


def _is_float8_linear(module: object) -> bool:
    """Class-name probe for torchao's ``Float8Linear`` (no torchao import)."""
    return type(module).__name__ == "Float8Linear"


def _torchao_available() -> bool:
    """True when torchao is importable.

    Checks ``sys.modules`` first so the ``sys.modules['torchao'] = None``
    test-stub idiom (v0.27.0 MII policy) and injected fake modules both
    resolve correctly — ``find_spec`` raises ValueError on an in-module
    fake whose ``__spec__`` is None.
    """
    import importlib.util
    import sys

    if "torchao" in sys.modules:
        return sys.modules["torchao"] is not None
    try:
        return importlib.util.find_spec("torchao") is not None
    except (ModuleNotFoundError, ValueError):
        return False


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


def apply_fp8_attention(model: object, *, recipe: str = "tensorwise") -> int:
    """Convert attention-projection linears to FP8 training modules.

    Live since v0.71.21 (#141). Walks ``model.named_modules()``, collects
    every ``nn.Linear`` whose FQN ends in an attention-projection name
    (q/k/v/o + fused qkv variants), and converts the not-yet-converted ones
    via torchao's ``convert_to_float8_training`` with an attention-only
    ``module_filter_fn``. Projections that are already ``Float8Linear``
    (e.g. because the base ``quantization_aware='fp8'`` pass converted the
    whole model) are counted but not re-wrapped.

    Returns:
        The number of attention projections that are FP8 after the call.

    Raises:
        TypeError: ``model`` is None or ``recipe`` is not a string.
        ValueError: ``recipe`` is empty, or the model has no attention
            projections at all (silent-no-op footgun).
        RuntimeError: torchao is missing or the GPU is not Hopper+
            (BETA hw gate — friendly message, never a silent no-op).
    """
    if model is None:
        raise TypeError("model must not be None")
    if isinstance(recipe, bool) or not isinstance(recipe, str):
        raise TypeError(f"recipe must be a string, got {type(recipe).__name__}")
    if not recipe:
        raise ValueError("recipe must be a non-empty string")
    if "\x00" in recipe:
        raise ValueError("recipe must not contain null bytes")

    # Probe the torchao float8 path SPECIFICALLY — fp8.is_fp8_available()
    # also accepts transformer_engine, which cannot serve this converter
    # (review fix: an NGC container with TE but no torchao must hit the
    # friendly gate, not an uncaught ImportError below).
    if not _torchao_available():
        raise RuntimeError(
            "fp8_attention requires torchao's float8 recipe "
            "(pip install 'torchao>=0.5.0')."
        )

    from soup_cli.utils.fp8 import is_fp8_gpu_supported

    if not is_fp8_gpu_supported():
        raise RuntimeError(
            "fp8_attention requires a Hopper+ GPU (H100/H200/B100/B200, "
            "compute capability >= 9.0)."
        )

    import torch.nn as nn

    already_converted: list[str] = []
    pending: list[str] = []
    for fqn, module in model.named_modules():
        if not is_attention_projection(fqn):
            continue
        if _is_float8_linear(module):
            already_converted.append(fqn)
        elif isinstance(module, nn.Linear):
            pending.append(fqn)
    if not already_converted and not pending:
        raise ValueError(
            "fp8_attention found no attention projections on this model "
            "(expected q_proj/k_proj/v_proj/o_proj or a fused qkv variant) "
            "— refusing the silent no-op."
        )

    if pending:
        try:
            from torchao.float8 import convert_to_float8_training
            from torchao.float8.config import Float8LinearConfig
        except ImportError as exc:
            raise RuntimeError(
                "fp8_attention requires torchao's float8 recipe "
                "(pip install 'torchao>=0.5.0')."
            ) from exc

        pending_set = frozenset(pending)
        config = Float8LinearConfig.from_recipe_name(recipe)
        try:
            convert_to_float8_training(
                model,
                config=config,
                module_filter_fn=lambda _mod, name: name in pending_set,
            )
        except Exception as exc:  # noqa: BLE001 — in-place mutation honesty
            raise RuntimeError(
                "fp8_attention conversion failed partway — the model may "
                "be PARTIALLY converted; restart training without the "
                f"flag ({type(exc).__name__}: {exc})"
            ) from exc
    return len(already_converted) + len(pending)


def apply_nvfp4(model: object) -> int:
    """Quantise ``model`` with torchao's NVFP4 scheme (Blackwell-only).

    Live since v0.71.21 (#141). Routes through the same
    ``torchao.quantization.NVFP4Config`` surface as the v0.53.1
    ``soup export --format torchao`` path, gated on a Blackwell GPU
    (SM 10.0 datacenter / SM 12.0 consumer).

    Returns:
        The number of ``nn.Linear`` modules torchao targeted (advisory
        count, taken before the in-place ``quantize_`` call).

    Raises:
        TypeError: ``model`` is None.
        RuntimeError: non-Blackwell GPU, torchao missing, or the installed
            torchao does not expose ``NVFP4Config``.
    """
    if model is None:
        raise TypeError("model must not be None")
    if not is_blackwell_gpu():
        raise RuntimeError(
            "NVFP4 requires a Blackwell GPU (B100/B200/GB200 at SM 10.0 or "
            "RTX 50-series at SM 12.0); no Blackwell device detected."
        )
    try:
        from torchao import quantization as ao_q
    except ImportError as exc:
        raise RuntimeError(
            "NVFP4 requires torchao (pip install 'torchao>=0.7.0')."
        ) from exc
    if not hasattr(ao_q, "NVFP4Config") or not hasattr(ao_q, "quantize_"):
        raise RuntimeError(
            "torchao does not expose NVFP4Config / quantize_; upgrade "
            "torchao (pip install -U torchao)."
        )

    import torch.nn as nn

    linear_count = sum(
        1 for _fqn, module in model.named_modules()
        if isinstance(module, nn.Linear)
    )
    try:
        ao_q.quantize_(model, ao_q.NVFP4Config())
    except Exception as exc:  # noqa: BLE001 — in-place mutation honesty
        raise RuntimeError(
            "NVFP4 quantisation failed partway — the model may be "
            "PARTIALLY quantised; restart training without the flag "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    return linear_count
