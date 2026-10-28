"""Hardware-fit calculator: the train pre-flight's view of the peak-VRAM estimate.

v0.40.3 #64 added a live CUDA OOM probe; that's correct but slow (one
real forward+backward per candidate). v0.64 added a fast analytical
predictor; #2049 replaced its formulas with ``utils/vram_estimate.py``, the
estimate ``soup profile``, the MCP ``profile`` tool and ``soup plan`` share,
so this module keeps its public names and delegates. ``decide_hardware_fit``
refuses a training launch when the predicted peak (with
``VRAM_SAFETY_MARGIN=10%`` headroom) would not fit.

The math is intentionally conservative — operators can supply
``--allow-oom-attempt`` to bypass the gate if they want to try anyway
(opt-out, not opt-in, per the v0.64 Part D plan).

Public surface:
- ``VRAM_SAFETY_MARGIN = 0.10``.
- ``validate_seq_len(v)`` / ``validate_batch_size(v)``.
- ``HardwareFitInput`` frozen dataclass.
- ``VRAMBreakdown`` frozen dataclass + ``total_gb`` property.
- ``HardwareFitReport`` frozen dataclass.
- ``estimate_peak_vram_gb(inp)`` -> ``VRAMBreakdown``.
- ``unextrapolated_floor(cfg, inp, breakdown)`` -> what the run needs on the card whatever
  its activations are (weights, CUDA context, trained-parameter state), and what that is.
- ``decide_hardware_fit(inp, *, available_vram_gb)`` -> ``HardwareFitReport``.
- ``resolve_peft_mode(cfg, quant=None)`` -> ``str``.
- ``hardware_fit_input_from_config(cfg, ...)`` -> ``HardwareFitInput | None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from soup_cli.utils.model_shape import ModelShape
from soup_cli.utils.vram_estimate import (
    BYTES_PER_PARAM_BY_QUANT,
    GB,
    OPTIMIZER_BYTES_PER_PARAM,
    OVERHEAD_FIXED_BYTES,
    VRAM_SAFETY_MARGIN,
    RunSpec,
    VramEstimate,
    estimate_vram,
)

_MIN_SEQ_LEN = 64
_MAX_SEQ_LEN = 1_048_576
_MIN_BATCH = 1
_MAX_BATCH = 1024
_MAX_PARAMS_B = 1000.0

# Closed allowlists — schema-shape rejection, no surprises.
_VALID_QUANT = frozenset({
    "none", "4bit", "8bit", "fp8", "gptq", "awq", "aqlm", "eetq", "mxfp4",
})
_VALID_PEFT = frozenset({"full", "lora", "dora", "qlora"})
_VALID_OPTIMIZERS = frozenset({
    "adamw_torch", "adamw_torch_fused", "adafactor", "sgd",
    "adamw_bnb_8bit", "paged_adamw_8bit", "lion_8bit",
    "lomo", "adalomo", "schedule_free_adamw",
})


def validate_seq_len(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("seq_len must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"seq_len must be int, got {type(value).__name__}")
    if not (_MIN_SEQ_LEN <= value <= _MAX_SEQ_LEN):
        raise ValueError(
            f"seq_len must be in [{_MIN_SEQ_LEN}, {_MAX_SEQ_LEN}], got {value}"
        )
    return value


def validate_batch_size(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("batch_size must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"batch_size must be int, got {type(value).__name__}")
    if not (_MIN_BATCH <= value <= _MAX_BATCH):
        raise ValueError(
            f"batch_size must be in [{_MIN_BATCH}, {_MAX_BATCH}], got {value}"
        )
    return value


@dataclass(frozen=True)
class HardwareFitInput:
    """Inputs needed to predict peak VRAM.

    ``shape`` is the model's shape as ``model_shape.read_model_shape`` read it;
    without one, the shape is guessed from ``params_b`` and the logits term is
    not counted. ``lora_r`` and ``target_modules`` size the LoRA adapters.
    """

    params_b: float
    seq_len: int
    batch_size: int
    optimizer: str
    quant: str
    peft: str
    gradient_checkpointing: Union[bool, str]
    shape: Optional[ModelShape] = None
    lora_r: int = 64
    target_modules: Any = "auto"
    #: False for a task whose model has no LM head (``NO_LM_HEAD_TASKS``).
    lm_head: bool = True
    #: The fused-loss flags the run's trainer applies (``use_liger``, ``use_cut_ce``).
    fused_loss: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.params_b, bool):
            raise TypeError("params_b must be a number, not bool")
        if not isinstance(self.params_b, (int, float)):
            raise TypeError(
                f"params_b must be a number, got {type(self.params_b).__name__}"
            )
        if not math.isfinite(float(self.params_b)):
            raise ValueError("params_b must be finite")
        if self.params_b <= 0 or self.params_b > _MAX_PARAMS_B:
            raise ValueError(
                f"params_b must be in (0, {_MAX_PARAMS_B}], got {self.params_b}"
            )
        validate_seq_len(self.seq_len)
        validate_batch_size(self.batch_size)
        if not isinstance(self.optimizer, str):
            raise TypeError("optimizer must be str")
        if self.optimizer not in _VALID_OPTIMIZERS:
            raise ValueError(
                f"unknown optimizer {self.optimizer!r}; "
                f"known: {', '.join(sorted(_VALID_OPTIMIZERS))}"
            )
        if not isinstance(self.quant, str):
            raise TypeError("quant must be str")
        if self.quant not in _VALID_QUANT:
            raise ValueError(
                f"unknown quant {self.quant!r}; "
                f"known: {', '.join(sorted(_VALID_QUANT))}"
            )
        if not isinstance(self.peft, str):
            raise TypeError("peft must be str")
        if self.peft not in _VALID_PEFT:
            raise ValueError(
                f"unknown peft {self.peft!r}; "
                f"known: {', '.join(sorted(_VALID_PEFT))}"
            )
        if not isinstance(self.gradient_checkpointing, (bool, str)):
            raise TypeError("gradient_checkpointing must be bool or a tier name")
        if self.shape is not None and not isinstance(self.shape, ModelShape):
            raise TypeError("shape must be ModelShape")


@dataclass(frozen=True)
class VRAMBreakdown:
    """Per-class peak VRAM in GB (1e9 bytes)."""

    weights_gb: float
    optimizer_gb: float
    gradients_gb: float
    activations_gb: float
    overhead_gb: float
    adapters_gb: float = 0.0
    logits_gb: float = 0.0
    dequant_gb: float = 0.0
    #: The full estimate behind these numbers (terms in bytes, shape, notes).
    estimate: Optional[VramEstimate] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for fld in (
            "weights_gb", "optimizer_gb", "gradients_gb",
            "activations_gb", "overhead_gb", "adapters_gb", "logits_gb", "dequant_gb",
        ):
            val = getattr(self, fld)
            if isinstance(val, bool):
                raise TypeError(f"{fld} must be a number, not bool")
            if not isinstance(val, (int, float)):
                raise TypeError(f"{fld} must be a number")
            if not math.isfinite(float(val)):
                raise ValueError(f"{fld} must be finite")
            if val < 0:
                raise ValueError(f"{fld} must not be negative, got {val}")

    @property
    def total_gb(self) -> float:
        return float(
            self.weights_gb
            + self.optimizer_gb
            + self.gradients_gb
            + self.activations_gb
            + self.overhead_gb
            + self.adapters_gb
            + self.logits_gb
            + self.dequant_gb
        )


@dataclass(frozen=True)
class HardwareFitReport:
    """Outcome of a hardware-fit decision."""

    ok: bool
    peak_vram_gb: float
    required_with_margin_gb: float
    available_vram_gb: float
    breakdown: VRAMBreakdown
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("ok must be bool")
        for fld in ("peak_vram_gb", "required_with_margin_gb", "available_vram_gb"):
            val = getattr(self, fld)
            if isinstance(val, bool):
                raise TypeError(f"{fld} must be a number, not bool")
            if not isinstance(val, (int, float)):
                raise TypeError(f"{fld} must be a number")
            if not math.isfinite(float(val)):
                raise ValueError(f"{fld} must be finite")
            if val < 0:
                raise ValueError(f"{fld} must not be negative")
        if not isinstance(self.breakdown, VRAMBreakdown):
            raise TypeError("breakdown must be VRAMBreakdown")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be str")


# Kept under their old names: autopilot reads the quant table (#1466).
_BYTES_PER_PARAM_BY_QUANT = BYTES_PER_PARAM_BY_QUANT
_OPTIM_BYTES_PER_PARAM = OPTIMIZER_BYTES_PER_PARAM


def resolve_peft_mode(cfg: Any, quant: str | None = None) -> str:
    """Resolve the effective PEFT mode ('full', 'qlora', or 'lora') for a config.

    Shared between train hardware-fit preflight, `soup profile`, and MCP `tool_profile` (#1423).
    """
    from soup_cli.trainer.classifier import CLASSIFICATION_TASKS
    from soup_cli.trainer.sft import is_full_finetune

    tcfg = getattr(cfg, "training", None)
    if quant is None and tcfg is not None:
        quant = str(getattr(tcfg, "quantization", "none") or "none")

    task = getattr(cfg, "task", None)
    if quant == "4bit":
        return "qlora"
    if task == "prm" or (
        task in CLASSIFICATION_TASKS
        and not (
            getattr(tcfg, "classifier_lora", False)
            and getattr(getattr(tcfg, "lora", None), "r", 0) > 0
        )
    ) or (
        task == "asr"
        and not (
            getattr(tcfg, "asr_lora", False)
            and getattr(getattr(tcfg, "lora", None), "r", 0) > 0
        )
    ):
        return "full"
    if tcfg is not None and is_full_finetune(tcfg):
        return "full"
    return "lora"


# training.quantization -> the predictor's name for it. A value that is not here
# (``hqq:*``) has no memory model, and the pre-flight says it skipped the run (#1652).
CONFIG_QUANT_TO_FIT = {
    "none": "none",
    "4bit": "4bit",
    "8bit": "8bit",
    # Stay packed on the card.
    "gptq": "gptq",
    "awq": "awq",
    "aqlm": "aqlm",
    "eetq": "eetq",
    # Dequantized on load, so priced (and gated) like "none" (#1631, #1652).
    "mxfp4": "mxfp4",
    "fp8": "fp8",
}

#: ``batch_size: auto`` resolves at train time; ``soup profile`` prices this instead.
PROFILE_AUTO_BATCH = 4

#: What the estimate's coefficients were fitted on (``benchmarks/gate-2049-resident-vram.md``).
CALIBRATED_RANGE = "dense models of 3B and below on one 6 GB card"
#: The largest dense model counted as inside it: the measured rows stop at 3.2B
#: (Llama-3.2-3B), so a "3B" model is in and the next size up (4B) is not.
CALIBRATED_MAX_PARAMS_B = 3.5


def outside_calibrated_range(
    inp: HardwareFitInput, *, backend: Optional[str] = None
) -> tuple[str, ...]:
    """Why ``inp`` is outside the range the estimate was fitted on; empty when inside.

    The pre-flight refuses a peak that does not fit only inside it. Outside it the same
    numbers are an extrapolation (a larger model, a mixture-of-experts one, a checkpointing
    tier or fused loss nobody measured, or a backend with its own loss and checkpointing),
    so they are printed as a warning and the run continues, unless the run's floor
    (:func:`unextrapolated_floor`) is already more than the card holds.
    """
    reasons = []
    if inp.params_b > CALIBRATED_MAX_PARAMS_B:
        reasons.append(f"{inp.params_b:.1f}B parameters")
    if inp.shape is not None and inp.shape.moe:
        reasons.append("a mixture-of-experts model")
    if inp.gradient_checkpointing in ("medium", "selective"):
        reasons.append(f"{inp.gradient_checkpointing} gradient checkpointing")
    if inp.fused_loss:
        noun = "flag" if len(inp.fused_loss) == 1 else "flags"
        reasons.append(f"fused-loss {noun} {' and '.join(sorted(inp.fused_loss))}")
    if backend == "unsloth":
        reasons.append("backend unsloth")
    return tuple(reasons)


#: Optimizers that update each parameter inside the backward pass, so no full gradient
#: is ever held (the estimate still charges one).
_FUSED_BACKWARD_OPTIMIZERS = frozenset({"lomo", "adalomo"})

#: Optimizers whose state ``OPTIMIZER_BYTES_PER_PARAM`` prices at what they hold: AdamW's
#: two fp32 moments (schedule-free AdamW's ``z`` and second moment), 8 bytes a parameter,
#: and 8-bit AdamW's two 8-bit moments, 2. The other figures there are more than the
#: optimizer holds (Adafactor and ``adalomo`` factor their state, ``lion_8bit`` keeps one
#: moment), so the floor does not charge them.
_EXACT_STATE_OPTIMIZERS = frozenset(
    {"adamw_torch", "adamw_torch_fused", "adamw_bnb_8bit", "schedule_free_adamw"}
)

#: Tasks whose trainer replaces ``training.optimizer`` with ``galore_adamw`` when
#: ``use_galore`` is on: ``SFTTrainerWrapper.setup`` (``tts`` reaches it through
#: ``super()``) and ``PretrainTrainerWrapper.setup``. Every other trainer ignores the
#: field and keeps the configured optimizer. (``tts`` with ``tts_family: sesame_csm``
#: builds its own trainer and refuses ``use_galore`` at setup.)
GALORE_APPLYING_TASKS: frozenset[str] = frozenset({"sft", "tts", "pretrain"})


def _trains_a_subset(tcfg: Any) -> bool:
    """Whether a full fine-tune trains only some of the parameters (Spectrum, LISA, freezing)."""
    return bool(
        getattr(tcfg, "unfrozen_parameters", None)
        or getattr(tcfg, "lisa_enabled", False)
        or getattr(tcfg, "freeze_layers", None)
        or getattr(tcfg, "freeze_ratio", None)
        or getattr(tcfg, "expand_layers", None)
    )


def unextrapolated_floor(
    cfg: Any, inp: HardwareFitInput, breakdown: VRAMBreakdown
) -> tuple[float, tuple[str, ...]]:
    """What the run needs on the card whatever its activations are, in GB, and its terms.

    Only the estimate's terms that are a byte count per parameter times a parameter count,
    or a constant: the weights, the CUDA context (the fixed 0.5 GB, not the ``overhead``
    term, which adds 5% of every allocated byte, activations included) and, for the
    parameters the run trains, their LoRA adapters, their gradients and, for the optimizers
    in ``_EXACT_STATE_OPTIMIZERS``, their optimizer state. None of them uses a fitted
    coefficient, so a run whose floor is more than the card holds fails outside the
    calibrated range as surely as inside it. Activations, logits, the 4-bit dequantization
    copy and the allocator slack never enter it.

    The trained-parameter terms are left out wherever the estimate overstates them or has
    no exact figure, so the floor never refuses a run that could fit:

    - On a full fine-tune, Spectrum (``unfrozen_parameters``), LISA, ``freeze_layers`` /
      ``freeze_ratio`` and ``expand_layers`` train a subset of the parameters, but
      ``resolve_peft_mode`` prices every parameter as trainable: weights and context only.
      (With LoRA they change nothing here: the adapters' terms are counted as usual.)
    - ``lomo`` / ``adalomo`` keep no full gradient: no gradient term.
    - The optimizer state is counted only for ``adamw_torch``, ``adamw_torch_fused``,
      ``schedule_free_adamw`` and ``adamw_bnb_8bit``. Every other optimizer has no
      optimizer term: Adafactor, ``lion_8bit`` and ``adalomo`` hold less than the
      estimate's figure, a paged optimizer moves its state to host memory under pressure,
      and one the estimate has no figure for is priced as AdamW.
    - With ``use_galore`` on a task in ``GALORE_APPLYING_TASKS`` there is no optimizer
      term either: the trainer swaps in ``galore_adamw``, which keeps low-rank state for
      the attention and MLP projections and full AdamW state for the other parameters
      (embeddings, output head, norms), a mix the estimate has no figure for. On any
      other task ``use_galore`` changes nothing and the configured optimizer is counted.
    """
    floor = breakdown.weights_gb + OVERHEAD_FIXED_BYTES / GB
    terms = ["weights", "CUDA context"]
    tcfg = getattr(cfg, "training", None)
    if inp.peft == "full" and _trains_a_subset(tcfg):
        return floor, tuple(terms)
    optimizer = str(getattr(tcfg, "optimizer", "adamw_torch") or "adamw_torch")
    galore = bool(getattr(tcfg, "use_galore", False)) and (
        getattr(cfg, "task", "sft") in GALORE_APPLYING_TASKS
    )
    charges = [
        ("adapters", breakdown.adapters_gb, True),
        ("gradients", breakdown.gradients_gb, optimizer not in _FUSED_BACKWARD_OPTIMIZERS),
        (
            "optimizer state",
            breakdown.optimizer_gb,
            optimizer in _EXACT_STATE_OPTIMIZERS and not galore,
        ),
    ]
    for name, gigabytes, exact in charges:
        if exact and gigabytes > 0:
            floor += gigabytes
            terms.append(name)
    return floor, tuple(terms)


#: Tasks whose model has no vocabulary-sized output, so the estimate charges no
#: logits: the classifier family and reward_model load
#: ``AutoModelForSequenceClassification`` (``trainer/classifier.py``,
#: ``trainer/reward_model.py``), embedding loads ``AutoModel`` (``trainer/embedding.py``).
NO_LM_HEAD_TASKS: frozenset[str] = frozenset(
    {"classifier", "reranker", "cross_encoder", "embedding", "reward_model"}
)

#: Tasks whose trainer installs the checkpointing tiers: ``trainer/sft.py``'s
#: ``plan_gradient_checkpointing``, inherited by tts. Every other trainer hands any
#: truthy ``gradient_checkpointing`` to HF as plain checkpointing, which is ``full``.
CHECKPOINT_TIER_TASKS: frozenset[str] = frozenset({"sft", "tts"})


def checkpointing_for_task(
    value: Any, *, task: str, gpu_memory_gb: Optional[float] = None
) -> Union[bool, str]:
    """The checkpointing the run's trainer will apply: ``False`` or a tier name.

    ``auto`` resolves on the card's memory in GiB, as ``trainer/sft.py`` reads it
    (``gradient_ckpt.resolve_granularity``); with no card it is ``full``, as there.
    """
    if not value:
        return False
    if task not in CHECKPOINT_TIER_TASKS:
        return "full"
    from soup_cli.utils.gradient_ckpt import resolve_granularity

    return resolve_granularity(value, gpu_memory_gb=gpu_memory_gb) or False


def hardware_fit_input_from_config(
    cfg: Any,
    *,
    allow_network: bool = True,
    profile_defaults: bool = False,
    gpu_memory_gb: Optional[float] = None,
    on_hub_request: Optional[Callable[[], None]] = None,
) -> Optional[HardwareFitInput]:
    """The estimate's input for a ``SoupConfig``: the train pre-flight, ``soup profile``
    and the MCP ``profile`` tool all build it here, so they price the same run (#2049).

    Returns ``None`` when the run is not statically predictable (batch_size
    ``"auto"``, a quantization with no memory model, out-of-range dims), in which
    case the pre-flight skips rather than guess. ``profile_defaults`` is ``soup
    profile``'s reading instead: ``auto`` is priced at batch 4 and a quantization
    with no memory model at bf16, as ``soup profile`` always did. ``gpu_memory_gb``
    is the card's memory in GiB, which ``gradient_checkpointing: auto`` resolves on.
    ``on_hub_request`` is called just before the first request to the Hub, if one is made.
    """
    from soup_cli.utils.model_shape import read_model_shape

    tcfg = cfg.training
    bs = getattr(tcfg, "batch_size", None)
    if not isinstance(bs, int) or isinstance(bs, bool):
        if not profile_defaults:
            return None  # "auto" resolves later — can't predict yet
        bs = PROFILE_AUTO_BATCH
    seq_len = getattr(cfg.data, "max_length", None)
    if not isinstance(seq_len, int) or isinstance(seq_len, bool):
        return None
    if profile_defaults:
        seq_len = min(max(seq_len, _MIN_SEQ_LEN), _MAX_SEQ_LEN)
    quant = CONFIG_QUANT_TO_FIT.get(str(getattr(tcfg, "quantization", "none") or "none"))
    if quant is None:
        if not profile_defaults:
            return None
        quant = "none"
    shape = read_model_shape(
        getattr(cfg, "base", "") or "",
        allow_network=allow_network,
        on_hub_request=on_hub_request,
    )
    peft = resolve_peft_mode(cfg, quant=quant)
    optimizer = str(getattr(tcfg, "optimizer", "adamw_torch") or "adamw_torch")
    if optimizer not in _VALID_OPTIMIZERS:
        optimizer = "adamw_torch"
    lora = getattr(tcfg, "lora", None)
    from soup_cli.config.schema import SFT_KERNEL_AWARE_TASKS

    task = getattr(cfg, "task", "sft")
    # Only the SFT-family trainer applies the fused losses (``trainer/sft.py``,
    # inherited by tts); on any other task the flags change nothing.
    fused_loss = (
        tuple(flag for flag in ("use_cut_ce", "use_liger") if getattr(tcfg, flag, False))
        if task in SFT_KERNEL_AWARE_TASKS
        else ()
    )
    try:
        return HardwareFitInput(
            params_b=shape.params / GB,
            seq_len=int(seq_len),
            batch_size=int(bs),
            optimizer=optimizer,
            quant=quant,
            peft=peft,
            gradient_checkpointing=checkpointing_for_task(
                getattr(tcfg, "gradient_checkpointing", False),
                task=task,
                gpu_memory_gb=gpu_memory_gb,
            ),
            shape=shape,
            lora_r=int(getattr(lora, "r", 64) or 0),
            target_modules=getattr(lora, "target_modules", "auto"),
            lm_head=task not in NO_LM_HEAD_TASKS,
            fused_loss=fused_loss,
        )
    except (ValueError, TypeError, ArithmeticError):
        return None  # dims out of the predictor's supported range


def estimate_peak_vram_gb(inp: HardwareFitInput) -> VRAMBreakdown:
    """The shared VRAM estimate (``utils/vram_estimate.py``), per class in GB."""
    if not isinstance(inp, HardwareFitInput):
        raise TypeError(
            f"inp must be HardwareFitInput, got {type(inp).__name__}"
        )
    from soup_cli.utils.model_shape import guessed_shape

    shape = inp.shape if inp.shape is not None else guessed_shape(float(inp.params_b))
    est = estimate_vram(
        shape,
        RunSpec(
            seq_len=inp.seq_len,
            batch_size=inp.batch_size,
            quant=inp.quant,
            peft=inp.peft,
            optimizer=inp.optimizer,
            gradient_checkpointing=inp.gradient_checkpointing,
            lora_r=inp.lora_r,
            target_modules=inp.target_modules,
            lm_head=inp.lm_head,
            fused_loss=inp.fused_loss,
        ),
    )
    return breakdown_from_estimate(est)


def breakdown_from_estimate(est: VramEstimate) -> VRAMBreakdown:
    return VRAMBreakdown(
        weights_gb=est.gb("weights"),
        optimizer_gb=est.gb("optimizer"),
        gradients_gb=est.gb("gradients"),
        activations_gb=est.gb("activations"),
        overhead_gb=est.gb("overhead"),
        adapters_gb=est.gb("adapters"),
        logits_gb=est.gb("logits"),
        dequant_gb=est.gb("dequant"),
        estimate=est,
    )


def decide_hardware_fit(
    inp: HardwareFitInput,
    *,
    available_vram_gb: float,
) -> HardwareFitReport:
    """Decide whether the planned run fits under ``available_vram_gb``.

    Applies ``VRAM_SAFETY_MARGIN=10%`` headroom on top of the analytical
    estimate. Refuse iff ``predicted * 1.1 > available``.
    """
    if isinstance(available_vram_gb, bool):
        raise TypeError("available_vram_gb must be a number, not bool")
    if not isinstance(available_vram_gb, (int, float)):
        raise TypeError(
            f"available_vram_gb must be a number, got {type(available_vram_gb).__name__}"
        )
    if not math.isfinite(float(available_vram_gb)):
        raise ValueError("available_vram_gb must be finite")
    if available_vram_gb < 0:
        raise ValueError(
            f"available_vram_gb must be >= 0, got {available_vram_gb}"
        )

    breakdown = estimate_peak_vram_gb(inp)
    peak = breakdown.total_gb
    required = peak * (1.0 + VRAM_SAFETY_MARGIN)
    if required <= available_vram_gb:
        return HardwareFitReport(
            ok=True,
            peak_vram_gb=peak,
            required_with_margin_gb=required,
            available_vram_gb=float(available_vram_gb),
            breakdown=breakdown,
            reason=(
                f"fits: peak {peak:.2f} GB + {VRAM_SAFETY_MARGIN:.0%} margin "
                f"<= {available_vram_gb:.2f} GB available"
            ),
        )
    return HardwareFitReport(
        ok=False,
        peak_vram_gb=peak,
        required_with_margin_gb=required,
        available_vram_gb=float(available_vram_gb),
        breakdown=breakdown,
        reason=(
            f"OOM risk: required {required:.2f} GB exceeds "
            f"available {available_vram_gb:.2f} GB. "
            f"Try --batch-size {max(1, inp.batch_size // 2)} or "
            "--quantization 4bit or --gradient-checkpointing auto."
        ),
    )


__all__ = [
    "VRAM_SAFETY_MARGIN",
    "HardwareFitInput",
    "HardwareFitReport",
    "VRAMBreakdown",
    "breakdown_from_estimate",
    "decide_hardware_fit",
    "estimate_peak_vram_gb",
    "hardware_fit_input_from_config",
    "outside_calibrated_range",
    "validate_batch_size",
    "validate_seq_len",
]
