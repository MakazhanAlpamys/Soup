"""One peak-VRAM estimate for a resident (not streamed) training run.

#2049: the ``soup train`` pre-flight (``utils/hardware_fit.py``) and ``soup profile``
(``utils/profiler.py``) priced the same run differently, in different units, and
neither had a vocabulary, so neither charged the logits and loss chain that
decides whether a long, large-vocabulary run fits. Both now call
:func:`estimate_vram`, as do the MCP ``profile`` tool and ``soup plan``.

Every term is in bytes; GB in the output is 1e9 bytes. The terms:

- ``weights``: the base at its loaded precision. Under 4-bit and 8-bit loading only
  the decoder's linear layers are quantized; the embeddings, output head and norms are
  cast to fp32 for training. Full fine-tuning keeps fp32 master weights (#339).
- ``adapters``: LoRA's A and B matrices, r * (in + out) per adapted module per layer,
  from the model's own dims and the configured target modules, in fp32.
- ``gradients`` and ``optimizer``: per trainable parameter.
- ``activations``: what the backward pass keeps per layer; with gradient
  checkpointing, one boundary tensor per layer plus one layer recomputed.
- ``logits``: the output head's logits and the loss chain over them, at the streaming
  planner's measured loss arithmetic (``layer_stream.LOGITS_LOSS_BYTES_PER_ELEMENT``).
  Checkpointing does not reduce it.
- ``dequant``: under 4-bit loading, the bf16 copy of the largest weight a matmul
  dequantizes.
- ``overhead``: the CUDA context and allocator slack, which ``max_memory_allocated``
  does not see but the card's total does.

The coefficients come from two-step ``soup train`` runs on an RTX 3060 Laptop
(``benchmarks/gate-2049-resident-vram.md``), held to a stated band per kind of run by
``tests/test_vram_estimate_calibration.py``.

No torch import: the estimate runs before anything is loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Union

from soup_cli.utils.model_shape import ModelShape, lora_param_count, lora_targets, module_dims

GB = 1e9

#: Headroom the train pre-flight adds on top of the estimate before comparing.
VRAM_SAFETY_MARGIN = 0.10

# --- weights -------------------------------------------------------------------

#: Bytes per parameter by quantization when the split between linear layers and
#: embeddings is not known (no config.json). Approximate.
BYTES_PER_PARAM_BY_QUANT: dict[str, float] = {
    "none": 2.0,  # bf16/fp16
    "4bit": 0.5,
    "8bit": 1.0,
    "fp8": 2.0,  # #1652: FineGrainedFP8Config(dequantize=True), so bf16 too
    "gptq": 0.55,
    "awq": 0.55,
    "aqlm": 0.5,
    "eetq": 1.0,
    "mxfp4": 2.0,  # #1466: loaded with Mxfp4Config(dequantize=True), so bf16
}

#: Bytes per quantized linear parameter, including the block scales.
NF4_BYTES_PER_PARAM = 0.53
INT8_BYTES_PER_PARAM = 1.02
#: Under 4-bit and 8-bit loading, what is not quantized (embeddings, output head,
#: norms) is cast to fp32 for training (peft ``prepare_model_for_kbit_training``).
KBIT_UNQUANTIZED_BYTES = 4.0

#: Full fine-tuning loads the trainable base at fp32 (#339).
FULL_FT_WEIGHT_BYTES = 4.0

# --- trainable parameters ------------------------------------------------------

#: LoRA's A and B are created in fp32 over a bf16 base (peft autocasts them).
ADAPTER_BYTES_PER_PARAM = 4.0
GRADIENT_BYTES_PER_PARAM = 4.0

#: Optimizer-state bytes per trainable parameter. AdamW = 8 (m + v fp32),
#: 8-bit AdamW = 2, Adafactor = 4, SGD = 0 (momentum-less reference).
OPTIMIZER_BYTES_PER_PARAM: dict[str, float] = {
    "adamw_torch": 8.0,
    "adamw_torch_fused": 8.0,
    "adafactor": 4.0,
    "sgd": 0.0,
    "adamw_bnb_8bit": 2.0,
    "paged_adamw_8bit": 2.0,
    "lion_8bit": 2.0,
    "lomo": 0.0,
    "adalomo": 2.0,
    "schedule_free_adamw": 8.0,
}

# --- activations ---------------------------------------------------------------

#: What the backward pass keeps per token per layer, in bf16 values:
#: ``ACT_PER_HIDDEN * hidden_size + ACT_PER_INTERMEDIATE * intermediate_size``
#: ``+ ACT_PER_ADAPTED_INPUT * (the in_features of every module LoRA adapts)``.
#: Fitted on the bf16 LoRA rows of ``benchmarks/gate-2049-resident-vram.md`` (five
#: model families, with and without checkpointing), rounded up. They are what autograd
#: keeps: RMSNorm's fp32 input, attention's q, k, v and output, SiLU's input and
#: output and the up projection, and each adapter's dropout output and mask.
ACT_PER_HIDDEN = 6.1
ACT_PER_INTERMEDIATE = 3.5
ACT_PER_ADAPTED_INPUT = 1.6
#: What each mode keeps on top, as (per hidden, per intermediate): 4-bit (``nf4``) and
#: 8-bit (``int8``) linear layers keep their inputs for the backward pass
#: (bitsandbytes), and a trained base (``full``) keeps every linear layer's input for
#: its weight gradient. Fitted per mode on the same rows, non-negative.
ACT_EXTRA: dict[str, tuple[float, float]] = {
    "lora": (0.0, 0.0),
    "nf4": (0.0, 3.6),
    "int8": (0.0, 4.5),
    "full": (0.5, 1.3),
}
ACTIVATION_BYTES = 2.0
#: Checkpointing keeps one hidden-sized tensor per layer, in bytes per element.
BOUNDARY_BYTES = {"lora": 2.0, "nf4": 2.0, "int8": 2.0, "full": 2.0}

# --- logits ----------------------------------------------------------------------

#: Under 4-bit and 8-bit loading the output head is fp32 (``KBIT_UNQUANTIZED_BYTES``),
#: so its logits cost 4 bytes an element, not 2.
KBIT_LOGITS_EXTRA_BYTES = 2
#: Config flags whose fused loss can avoid building the full logits.
FUSED_LOSS_FLAGS = frozenset({"use_liger", "use_cut_ce"})

# --- 4-bit dequantization --------------------------------------------------------

#: bf16 copies of the largest dequantized weight live at once under 4-bit loading.
DEQUANT_COPIES = 1.0

# --- overhead ----------------------------------------------------------------------

#: CUDA context and cuBLAS workspaces: on the card, never in ``max_memory_allocated``.
OVERHEAD_FIXED_BYTES = 0.5e9
#: Allocator slack as a fraction of the allocated peak (reserved over allocated).
OVERHEAD_SLACK = 0.05

TERM_NAMES = (
    "weights",
    "adapters",
    "gradients",
    "optimizer",
    "activations",
    "logits",
    "dequant",
    "overhead",
)

_CHECKPOINT_TIERS = {
    False: "off",
    True: "full",
    "full": "full",
    "auto": "full",
    "medium": "medium",
    "selective": "off",
}


@dataclass(frozen=True)
class RunSpec:
    """What the estimate needs to know about the run."""

    seq_len: int
    batch_size: int
    quant: str = "none"
    peft: str = "lora"
    optimizer: str = "adamw_torch"
    gradient_checkpointing: Union[bool, str] = False
    lora_r: int = 64
    target_modules: Any = "auto"
    #: False for a model with no vocabulary-sized output (a classification or
    #: embedding head): it builds no logits over the vocabulary.
    lm_head: bool = True
    #: The fused-loss flags set on a run whose trainer applies them
    #: (``use_liger``, ``use_cut_ce``). The logits term is still charged, since
    #: whether the patch lands is only known at train time, and a note says so.
    fused_loss: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.lm_head, bool):
            raise TypeError(f"lm_head must be bool, got {type(self.lm_head).__name__}")
        if not set(self.fused_loss) <= FUSED_LOSS_FLAGS:
            raise ValueError(f"unknown fused_loss {self.fused_loss!r}")
        for name in ("seq_len", "batch_size", "lora_r"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")
        if self.seq_len <= 0 or self.batch_size <= 0 or self.lora_r < 0:
            raise ValueError("seq_len and batch_size must be positive and lora_r >= 0")
        if self.quant not in BYTES_PER_PARAM_BY_QUANT:
            raise ValueError(f"unknown quant {self.quant!r}")
        if self.peft not in ("full", "lora", "dora", "qlora"):
            raise ValueError(f"unknown peft {self.peft!r}")

    @property
    def checkpointing(self) -> str:
        """``off``, ``medium`` (every other block) or ``full``."""
        return _CHECKPOINT_TIERS.get(self.gradient_checkpointing, "full")


@dataclass(frozen=True)
class VramEstimate:
    """A per-term peak-VRAM estimate, in bytes."""

    terms: dict[str, int]
    shape: ModelShape
    run: RunSpec
    #: Caveats the reader needs: what could not be read and was guessed instead.
    notes: tuple[str, ...] = field(default=())

    @property
    def total_bytes(self) -> int:
        return int(sum(self.terms.values()))

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GB

    def gb(self, term: str) -> float:
        return self.terms.get(term, 0) / GB

    @property
    def lora_targets(self) -> tuple[str, ...]:
        """The modules the adapters were sized on; empty for full fine-tuning."""
        if self.run.peft == "full":
            return ()
        return lora_targets(self.run.target_modules, self.shape.model_type)


def _embedding_params(shape: ModelShape) -> int:
    if shape.vocab_size is None:
        return 0
    return shape.vocab_size * shape.hidden_size * (1 if shape.tie_word_embeddings else 2)


def weights_bytes(shape: ModelShape, run: RunSpec) -> int:
    if run.peft == "full" and run.quant == "none":
        return math.ceil(shape.params * FULL_FT_WEIGHT_BYTES)
    if run.quant in ("4bit", "8bit") and shape.vocab_size is not None:
        embed = _embedding_params(shape)
        linear = max(shape.params - embed, 0)
        per = NF4_BYTES_PER_PARAM if run.quant == "4bit" else INT8_BYTES_PER_PARAM
        return math.ceil(linear * per + embed * KBIT_UNQUANTIZED_BYTES)
    return math.ceil(shape.params * BYTES_PER_PARAM_BY_QUANT[run.quant])


def activation_mode(run: RunSpec) -> str:
    """``full`` (a trained base), ``nf4`` / ``int8`` (4-bit / 8-bit loading) or ``lora``."""
    if run.peft == "full":
        return "full"
    return {"4bit": "nf4", "8bit": "int8"}.get(run.quant, "lora")


def trainable_params(shape: ModelShape, run: RunSpec) -> int:
    if run.peft == "full":
        return shape.params
    return lora_param_count(shape, r=run.lora_r, target_modules=run.target_modules)


def activations_bytes(shape: ModelShape, run: RunSpec) -> int:
    tokens = run.seq_len * run.batch_size
    mode = activation_mode(run)
    extra_hidden, extra_inter = ACT_EXTRA[mode]
    adapted_inputs = (
        0
        if mode == "full"
        else sum(
            module_dims(shape, name)[0]
            for name in lora_targets(run.target_modules, shape.model_type)
        )
    )
    per_layer = tokens * ACTIVATION_BYTES * (
        (ACT_PER_HIDDEN + extra_hidden) * shape.hidden_size
        + (ACT_PER_INTERMEDIATE + extra_inter) * shape.intermediate_size
        + ACT_PER_ADAPTED_INPUT * adapted_inputs
    )
    boundary = tokens * shape.hidden_size * BOUNDARY_BYTES[mode]
    layers = shape.num_layers
    if run.checkpointing == "full":
        total = layers * boundary + per_layer
    elif run.checkpointing == "medium":
        kept = math.ceil(layers / 2)
        total = kept * per_layer + (layers - kept) * boundary + per_layer
    else:
        total = layers * per_layer
    return math.ceil(total)


def logits_bytes(shape: ModelShape, run: RunSpec) -> int:
    """The output head's logits and the loss over them; checkpointing does not reduce it.

    The streaming planner's measured loss arithmetic per element
    (``layer_stream.LOGITS_LOSS_BYTES_PER_ELEMENT``). The resident rows fit 11.6
    bytes an element, so they do not carry the extra retained copy #327 measured on
    the streaming path (``LOGITS_RETENTION_BYTES_PER_ELEMENT``), and it is not charged.
    A model without an LM head builds no logits over the vocabulary.
    """
    if shape.vocab_size is None or not run.lm_head:
        return 0
    from soup_cli.utils.layer_stream import LOGITS_LOSS_BYTES_PER_ELEMENT

    per = LOGITS_LOSS_BYTES_PER_ELEMENT
    if activation_mode(run) in ("nf4", "int8"):
        per += KBIT_LOGITS_EXTRA_BYTES
    return math.ceil(shape.vocab_size * run.seq_len * run.batch_size * per)


def dequant_bytes(shape: ModelShape, run: RunSpec) -> int:
    if run.quant != "4bit":
        return 0
    largest = max(
        fan_in * fan_out
        for fan_in, fan_out in (
            module_dims(shape, name)
            for name in ("q_proj", "o_proj", "gate_proj", "down_proj")
        )
    )
    return math.ceil(largest * 2.0 * DEQUANT_COPIES)


def estimate_vram(shape: ModelShape, run: RunSpec) -> VramEstimate:
    """The per-term peak for ``run`` on a model of ``shape``."""
    trainable = trainable_params(shape, run)
    terms = {
        "weights": weights_bytes(shape, run),
        "adapters": (
            0 if run.peft == "full" else math.ceil(trainable * ADAPTER_BYTES_PER_PARAM)
        ),
        "gradients": math.ceil(trainable * GRADIENT_BYTES_PER_PARAM),
        "optimizer": math.ceil(trainable * OPTIMIZER_BYTES_PER_PARAM.get(run.optimizer, 8.0)),
        "activations": activations_bytes(shape, run),
        "logits": logits_bytes(shape, run),
        "dequant": dequant_bytes(shape, run),
    }
    allocated = sum(terms.values())
    terms["overhead"] = math.ceil(OVERHEAD_FIXED_BYTES + OVERHEAD_SLACK * allocated)
    notes = []
    if shape.vocab_size is None:
        notes.append(
            "No config.json could be read for this model, so its shape is guessed from "
            "the parameter count and the logits term is not counted."
        )
    elif shape.params_source == "config.json":
        notes.append("Parameter count from config.json (no safetensors headers were read).")
        if shape.moe:
            notes.append(
                "This is a mixture-of-experts model: the count prices every expert in every "
                "layer, so it is approximate."
            )
    if run.gradient_checkpointing == "medium":
        notes.append(
            "Medium checkpointing (every other block) is priced from the per-layer costs "
            "measured with checkpointing off and on; the tier itself is not measured."
        )
    elif run.gradient_checkpointing == "selective":
        notes.append(
            "Selective checkpointing (attention only) is priced as no checkpointing, an "
            "upper bound; the tier is not measured."
        )
    if run.fused_loss and terms["logits"]:
        flags = " and ".join(f"training.{flag}" for flag in sorted(run.fused_loss))
        notes.append(
            f"{flags} {'is' if len(run.fused_loss) == 1 else 'are'} set: if the fused loss "
            "applies, the full logits are not built, "
            f"so the logits term ({terms['logits'] / GB:.1f} GB) overstates the peak. "
            "The fused path is not measured."
        )
    return VramEstimate(terms=terms, shape=shape, run=run, notes=tuple(notes))


def fit_verdict(total_bytes: float, available_bytes: float) -> str:
    """``fits``, ``tight`` (fits, but not with the pre-flight's margin) or ``does not fit``."""
    if total_bytes * (1.0 + VRAM_SAFETY_MARGIN) <= available_bytes:
        return "fits"
    if total_bytes <= available_bytes:
        return "tight"
    return "does not fit"


__all__ = [
    "BYTES_PER_PARAM_BY_QUANT",
    "GB",
    "OPTIMIZER_BYTES_PER_PARAM",
    "TERM_NAMES",
    "VRAM_SAFETY_MARGIN",
    "RunSpec",
    "VramEstimate",
    "estimate_vram",
    "fit_verdict",
]
