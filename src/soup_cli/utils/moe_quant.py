"""v0.52.0 Part F — MoE expert quantization + router-only training schema.

Two new TrainingConfig fields are introduced this release:

* ``moe_expert_quant: Optional[Literal["nf4", "int8_rowwise"]]`` — per-expert
  weight quantization for fused-MoE Linear blocks. Wraps axolotl's MoE
  expert quant path.
* ``train_router_only: bool`` — freeze every expert + train only the
  gating router (unsloth MoE recipe). Useful for router calibration.

Schema-only this release; live wiring lands in v0.52.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

MOE_EXPERT_QUANT_FORMATS: frozenset[str] = frozenset({"nf4", "int8_rowwise"})

_MAX_QUANT_LEN: int = 32


@dataclass(frozen=True)
class MoEExpertQuantSpec:
    """Metadata for a MoE-expert quant format. Frozen."""

    name: str
    description: str
    bits: int
    live_wired: bool


_MOE_EXPERT_QUANT_METADATA: Mapping[str, MoEExpertQuantSpec] = MappingProxyType({
    "nf4": MoEExpertQuantSpec(
        name="nf4",
        description="NF4 per-expert (BNB 4-bit Normal-Float)",
        bits=4,
        live_wired=False,
    ),
    "int8_rowwise": MoEExpertQuantSpec(
        name="int8_rowwise",
        description="INT8 row-wise per-expert (LLM.int8 row-wise)",
        bits=8,
        live_wired=False,
    ),
})


def validate_moe_expert_quant(name: object) -> str:
    """Validate a MoE expert-quant name. Returns canonical form."""
    if isinstance(name, bool):
        raise TypeError(f"moe_expert_quant must not be bool, got {name!r}")
    if not isinstance(name, str):
        raise TypeError(
            f"moe_expert_quant must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("moe_expert_quant must be non-empty")
    if "\x00" in name:
        raise ValueError("moe_expert_quant must not contain null bytes")
    if len(name) > _MAX_QUANT_LEN:
        raise ValueError(
            f"moe_expert_quant too long (max {_MAX_QUANT_LEN} chars)"
        )
    canonical = name.lower()
    if canonical not in MOE_EXPERT_QUANT_FORMATS:
        supported = ", ".join(sorted(MOE_EXPERT_QUANT_FORMATS))
        raise ValueError(
            f"moe_expert_quant {name!r} not supported. "
            f"Supported: {supported}"
        )
    return canonical


def get_moe_expert_quant_spec(name: str) -> MoEExpertQuantSpec:
    """Return the frozen spec for ``name`` or raise."""
    canonical = validate_moe_expert_quant(name)
    return _MOE_EXPERT_QUANT_METADATA[canonical]


def validate_moe_expert_quant_compat(*, backend: str, moe_lora: bool) -> None:
    """Schema-time gate for ``moe_expert_quant``.

    Rejects:
    - non-string ``backend`` / non-bool ``moe_lora`` (defence-in-depth).
    - ``backend == 'mlx'`` (no MLX MoE expert quant).
    - ``moe_lora == False`` (MoE expert quant only meaningful when training
      LoRA adapters that sit on top of fused-MoE experts — otherwise the
      operator is asking for full-precision MoE training with quantized
      experts, which silently no-ops).
    """
    if isinstance(backend, bool):
        raise TypeError(f"backend must not be bool, got {backend!r}")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string")
    if not isinstance(moe_lora, bool):
        raise TypeError(f"moe_lora must be bool, got {type(moe_lora).__name__}")
    if backend == "mlx":
        raise ValueError(
            "moe_expert_quant is not supported on backend=mlx in v0.52.0"
        )
    if not moe_lora:
        raise ValueError(
            "moe_expert_quant requires moe_lora=true "
            "(per-expert quant is only meaningful with MoE-aware LoRA wiring)"
        )


def validate_train_router_only_compat(*, backend: str, moe_lora: bool) -> None:
    """Schema-time gate for ``train_router_only=True``.

    Requires ``moe_lora=true`` (without it, every expert would still be
    trained and the flag would silently no-op). Defence-in-depth bool /
    str rejection on the args.
    """
    if isinstance(backend, bool):
        raise TypeError(f"backend must not be bool, got {backend!r}")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string")
    if not isinstance(moe_lora, bool):
        raise TypeError(f"moe_lora must be bool, got {type(moe_lora).__name__}")
    if backend == "mlx":
        raise ValueError(
            "train_router_only is not supported on backend=mlx in v0.52.0"
        )
    if not moe_lora:
        raise ValueError(
            "train_router_only requires moe_lora=true "
            "(router-only training freezes the experts, which is only "
            "meaningful with MoE-aware LoRA wiring)"
        )


# Module-name substring that identifies a fused-MoE expert submodule across
# Mixtral / Qwen-MoE / DeepSeek / OLMoE / Granite-MoE. The expert FFN Linears
# live under ``...experts.<n>.<proj>``.
_EXPERT_NAME_MARKER: str = ".experts."


# bitsandbytes quantized Linear class names — used to skip experts that are
# already quantized (e.g. when the whole model was loaded in 4-bit), without
# importing bitsandbytes into the pure detection path.
_BNB_LINEAR_CLASS_NAMES: frozenset[str] = frozenset({"Linear4bit", "Linear8bitLt"})


def _is_router_param(name: str) -> bool:
    """Whether a parameter name belongs to the MoE gating router.

    The router (a.k.a. gate) selects which experts a token is dispatched to:
    ``block_sparse_moe.gate`` (Mixtral), ``mlp.gate`` (Qwen-MoE),
    ``...router...`` (OLMoE / Granite). It must NOT match the per-expert
    ``gate_proj`` FFN projection — those are expert weights, not the router.

    The expert-marker exclusion runs FIRST so an expert never wins a stray
    ``router`` substring match.
    """
    lower = name.lower()
    if _EXPERT_NAME_MARKER in name or "gate_proj" in lower:
        return False
    if "router" in lower:
        return True
    return ".gate." in name or name.endswith(".gate.weight") or name.endswith(".gate.bias")


def _find_expert_linears(model: object) -> list[tuple[str, object]]:
    """Return ``(name, module)`` pairs for every fused-MoE expert ``nn.Linear``.

    Pure ``nn.Module`` walk — no bitsandbytes / CUDA. Already-quantized
    bitsandbytes Linears (subclasses of ``nn.Linear``) are skipped so an
    already-4-bit base model is not double-quantized. PEFT-wrapped Linears
    (those carrying a ``base_layer``) are also skipped — the quant path runs
    BEFORE ``get_peft_model`` so this is defence-in-depth. Used both by the
    quant path and by tests (the detection is validatable without a GPU).
    """
    import torch.nn as nn

    found: list[tuple[str, object]] = []
    for name, module in model.named_modules():
        if _EXPERT_NAME_MARKER not in name or not isinstance(module, nn.Linear):
            continue
        if type(module).__name__ in _BNB_LINEAR_CLASS_NAMES:
            continue
        if getattr(module, "base_layer", None) is not None:
            continue
        found.append((name, module))
    return found


def freeze_experts_train_router(model: object) -> tuple[int, int]:
    """Freeze every MoE expert parameter and keep the gating router trainable.

    Implements ``train_router_only=True`` (unsloth MoE recipe): the experts
    are frozen and only the router weights receive gradients. Returns
    ``(experts_frozen, router_trainable)`` counts. Other parameters keep their
    current ``requires_grad`` state (set by the LoRA wiring upstream).
    """
    experts_frozen = 0
    router_trainable = 0
    for name, param in model.named_parameters():
        if _EXPERT_NAME_MARKER in name:
            if param.requires_grad:
                param.requires_grad = False
            experts_frozen += 1
        elif _is_router_param(name):
            param.requires_grad = True
            router_trainable += 1
    return experts_frozen, router_trainable


def apply_moe_expert_quant(model: object, quant_format: str) -> int:
    """Quantize the fused-MoE expert ``nn.Linear`` blocks in ``model``.

    nf4 → ``bnb.nn.Linear4bit``; int8_rowwise → ``bnb.nn.Linear8bitLt``. Each
    expert Linear is replaced in-place with the bitsandbytes quantized variant
    (weights copied), leaving attention + the router in full precision so the
    MoE-aware LoRA adapters still see fp weights.

    bitsandbytes is CUDA-only; on a CPU / no-bnb host a friendly
    ``RuntimeError`` fires (the real per-expert quant validation stays
    hardware-gated). Returns the number of expert Linears quantized.
    """
    canonical = validate_moe_expert_quant(quant_format)
    expert_linears = _find_expert_linears(model)
    if not expert_linears:
        return 0

    try:
        import bitsandbytes as bnb  # type: ignore
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "moe_expert_quant requires the 'bitsandbytes' package (CUDA-only). "
            "Install it with `pip install bitsandbytes` on a CUDA host."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "moe_expert_quant requires a CUDA device (bitsandbytes quantizes "
            "weights on the GPU). No CUDA device is available."
        )

    quantized = 0
    for name, linear in expert_linears:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child_name = name.rsplit(".", 1)[-1]
        in_f, out_f = linear.in_features, linear.out_features
        has_bias = linear.bias is not None
        # Keep the expert on its own device (device_map='auto' can shard
        # experts across GPUs — never assume cuda:0).
        target_device = linear.weight.device
        # Quantize the SOURCE (trained) weights explicitly via the bnb param
        # constructor + .to(device) so the quantization carries the real
        # weight values (a bare .data swap is fragile across bnb versions).
        src = linear.weight.data.detach().clone()
        if canonical == "nf4":
            new = bnb.nn.Linear4bit(
                in_f, out_f, bias=has_bias,
                compute_dtype=torch.bfloat16, quant_type="nf4",
            )
            new.weight = bnb.nn.Params4bit(src, requires_grad=False, quant_type="nf4")
        else:  # int8_rowwise
            new = bnb.nn.Linear8bitLt(
                in_f, out_f, bias=has_bias, has_fp16_weights=False,
            )
            new.weight = bnb.nn.Int8Params(
                src, requires_grad=False, has_fp16_weights=False,
            )
        if has_bias:
            new.bias = type(linear.bias)(linear.bias.data.detach().clone())
        new = new.to(target_device)
        setattr(parent, child_name, new)
        quantized += 1
    return quantized


def apply_moe_expert_quant_if_configured(
    model: object, tcfg: object, console: object = None
) -> None:
    """Apply ``moe_expert_quant`` to a loaded model — BEFORE LoRA.

    Quantizing the expert Linears before ``get_peft_model`` means PEFT attaches
    its LoRA adapters to the quantized base (QLoRA-on-experts) rather than the
    swap destroying freshly-injected adapters. No-op when unset.
    """
    fmt = getattr(tcfg, "moe_expert_quant", None)
    if fmt is None:
        return
    count = apply_moe_expert_quant(model, fmt)
    if console is not None:
        console.print(
            f"[green]MoE expert quant:[/] {fmt} applied to {count} "
            "expert Linear block(s)"
        )


def apply_router_only_freeze_if_configured(
    model: object, tcfg: object, console: object = None
) -> None:
    """Apply ``train_router_only`` to a model — AFTER LoRA.

    Runs after ``get_peft_model`` so the final (PEFT-wrapped) parameter set is
    frozen consistently: experts (incl. their adapters) frozen, router
    trainable. No-op when unset.
    """
    if not getattr(tcfg, "train_router_only", False):
        return
    frozen, trainable = freeze_experts_train_router(model)
    if console is not None:
        console.print(
            f"[green]MoE router-only training:[/] {frozen} expert "
            f"param(s) frozen, {trainable} router param(s) trainable"
        )
