# SPDX-License-Identifier: Apache-2.0
#
# Written for Soup.
"""The settings of the packed-LoRA kernels, and which runs they cannot take.

``PackedLoraConfig`` is the type of ``training.packed_lora`` in ``config/schema.py``, and
:func:`packed_lora_problem` is the sentence of its cross-validator (``SoupConfig``). Pure Python
(pydantic only): no torch, no Triton.

Defaults when enabled: ``int8_backward: auto`` (the int8 MLP input gradient, kept on when a
start-up gradient check passes) with ``int8_rounding: grad`` (stochastic rounding of the gradient
quantisers). Round-to-nearest zeroes every gradient entry below half a step of its row's largest
value, a bias that showed as a training loss drifting above plain Soup's in the challenge runs the
kernels come from; stochastic rounding is unbiased, at the price of noisier steps. Making the exact
mode the default is the one ``int8_backward`` default line below.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

#: ``gpu_budget_mib`` is a target for torch's peak allocated memory, planned from a fit to
#: measurements: an estimate with a margin (2 % of the budget plus 64 MiB) and a per-step guard,
#: not a guarantee (``budget.py``: a fit, not a proof). The planner refuses a budget under the
#: floor of the model and batch it is given (and says what the minimum is); these bounds only
#: reject numbers that cannot be a budget for any supported model.
MIN_BUDGET_MIB = 512
MAX_BUDGET_MIB = 1_048_576


class PackedLoraConfig(BaseModel):
    """The settings of the packed-LoRA kernels for one run (a future ``training.packed_lora``).

    Opt-in, off by default. The kernels run NF4 decode, RMSNorm, RoPE, SwiGLU, a logit-free
    head and loss over padding-free tokens, and keep each layer's saved activations on the GPU
    or in pinned host memory instead of recomputing them. :func:`packed_lora_problem` says which
    settings of a run the kernels cannot honour.
    """

    enabled: bool = Field(
        default=False,
        description="Turn the packed-LoRA kernels on for this run",
    )
    gpu_budget_mib: Optional[int] = Field(
        default=None,
        ge=MIN_BUDGET_MIB,
        le=MAX_BUDGET_MIB,
        description=(
            "Target for torch's peak allocated GPU memory in MiB, planned from a fit to "
            "measurements (an estimate with a margin, not a guarantee): the planner decides "
            "how many layers keep their saved activations on the GPU and which tables "
            "live in pinned host memory. Unset keeps whatever fits resident. A budget "
            "under what the model needs is refused with the minimum."
        ),
    )
    activations: Literal["auto", "resident", "offload"] = Field(
        default="auto",
        description=(
            "Where each layer's saved activations live between its forward and backward: "
            "'resident' (GPU), 'offload' (pinned host memory) or 'auto' (the last layers "
            "that fit stay on the GPU)."
        ),
    )
    int8_backward: Literal["auto", "on", "stash", "off"] = Field(
        default="auto",
        description=(
            "The MLP input-gradient GEMMs of the backward in int8. 'auto' keeps them on "
            "when a start-up gradient check passes, 'on' always, 'stash' only stores the "
            "gate/up activations as int8, 'off' is the exact mode (no int8 anywhere, about "
            "21-24% slower)."
        ),
    )
    int8_rounding: Literal["nearest", "grad", "stochastic"] = Field(
        default="grad",
        description=(
            "Rounding of the int8 gradient quantisers: 'grad' rounds the gradients "
            "stochastically (unbiased in expectation, the default), 'stochastic' also "
            "dithers the int8 weights, 'nearest' rounds to nearest (biased: the training "
            "loss drifts above plain Soup's)."
        ),
    )
    on_unsupported: Literal["error", "stock"] = Field(
        default="error",
        description=(
            "What to do when a run turns out not to be supported at setup ('error' stops "
            "with the reason, 'stock' prints it and trains with the stock step)."
        ),
    )

    @field_validator("gpu_budget_mib", mode="before")
    @classmethod
    def _validate_budget_int(cls, v: Any) -> Any:
        """Reject bool-as-int and floats (``bool`` subclasses ``int``)."""
        if isinstance(v, bool):
            raise ValueError("training.packed_lora.gpu_budget_mib must be an int, not bool")
        if isinstance(v, float):
            raise ValueError(
                f"training.packed_lora.gpu_budget_mib must be an int; got {v!r}"
            )
        return v

    @field_validator("int8_backward", mode="before")
    @classmethod
    def _validate_int8_backward_mode(cls, v: Any) -> Any:
        """YAML 1.1 reads an unquoted ``on``/``off`` as a bool. Here the bool agrees
        with the mode (``int8_backward: off`` = exact), so it is mapped back; an
        int, including 0/1, is refused."""
        if v is True:
            return "on"
        if v is False:
            return "off"
        if not isinstance(v, str):
            raise ValueError(
                "training.packed_lora.int8_backward must be one of 'auto', 'on', "
                f"'stash', 'off'; got {type(v).__name__} {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _validate_tunables_need_enabled(self) -> "PackedLoraConfig":
        """A tunable set while the block is off is a silent no-op: refuse it. A value
        equal to its default is not "set" (dumped configs write every default)."""
        if self.enabled:
            return self
        fields = type(self).model_fields
        set_fields = [
            name for name in (
                "gpu_budget_mib", "activations", "int8_backward", "int8_rounding",
                "on_unsupported",
            )
            if getattr(self, name) != fields[name].default
        ]
        if set_fields:
            named = " / ".join(f"training.packed_lora.{name}" for name in set_fields)
            raise ValueError(
                f"{named} set but packed_lora is not enabled; set "
                "training.packed_lora.enabled: true to run the packed-LoRA kernels."
            )
        return self


def packed_lora_problem(config, block):
    """Why the kernels cannot run this config, or ``None`` if they can.

    ``config``: a ``SoupConfig`` (or anything with its ``backend``, ``modality``, ``task``,
    ``data`` and ``training`` attributes); ``block``: the run's :class:`PackedLoraConfig`.
    The kernels run one Qwen2 QLoRA SFT step; every other setting is refused with its
    reason rather than left unpatched. One reason per message: the first setting that cannot
    be honoured. A block that is not enabled has nothing to refuse."""
    tcfg = config.training
    if block is None or not block.enabled:
        return None
    lora = tcfg.lora
    reason = None
    if config.backend != "transformers":
        reason = (
            f"needs backend='transformers'; got backend={config.backend!r}. The "
            f"kernels replace the transformers + peft training step."
        )
    elif config.modality != "text":
        reason = f"supports modality='text' only; got modality={config.modality!r}."
    elif config.task != "sft":
        reason = (
            f"supports task='sft' only; got task={config.task!r}. Other trainers "
            f"are not tested with the kernels."
        )
    elif lora.r == 0:
        reason = "needs a LoRA; lora.r=0 trains the full model, with no LoRA to run."
    elif tcfg.quantization != "4bit":
        reason = (
            f"needs quantization: 4bit (NF4); got quantization={tcfg.quantization!r}. "
            f"The kernels decode bitsandbytes NF4 weights."
        )
    elif not tcfg.double_quant_on:
        reason = (
            "needs double quantization: training.bnb_4bit_use_double_quant is false. "
            "The decoder reads the nested (double-quantized) NF4 layout; leave the "
            "field unset."
        )
    elif lora.dropout != 0:
        reason = (
            f"needs lora.dropout: 0; got {lora.dropout} (Soup's default is 0.05). "
            f"peft gives every projection its own dropout, so q/k/v and gate/up "
            f"would no longer share the one input the kernels read."
        )
    else:
        for label, clash, why in (
            ("lora.use_dora", lora.use_dora, "the kernels compute plain LoRA"),
            ("lora.use_vera", lora.use_vera, "the kernels compute plain LoRA"),
            ("training.use_liger", tcfg.use_liger,
             "Liger replaces the MLP and the loss head the kernels replace"),
            ("training.use_cut_ce", tcfg.use_cut_ce,
             "Cut Cross-Entropy replaces the loss head the kernels replace"),
            ("training.fast_lora", tcfg.fast_lora,
             "fast_lora replaces the LoRA modules the kernels bypass"),
            ("training.use_flash_attn", tcfg.use_flash_attn,
             "the kernels run PyTorch's memory-efficient attention over unpadded rows"),
            ("training.packing", tcfg.packing,
             "the kernels already remove the padding inside each batch"),
            ("training.multipack", tcfg.multipack,
             "the kernels already remove the padding inside each batch"),
            ("training.neftune_alpha", tcfg.neftune_alpha is not None,
             "the kernels read the embedding table directly, around NEFTune's noise"),
            ("training.stream_layers", tcfg.stream_layers,
             "layer streaming has its own step"),
            ("training.activation_offloading", tcfg.activation_offloading is not None,
             "the kernels offload the saved activations themselves"),
            ("training.use_galore", tcfg.use_galore,
             "GaLore trains full weights, not a LoRA"),
            ("training.use_lorafa", tcfg.use_lorafa,
             "LoRA-FA freezes half of every adapter, which the kernels do not do"),
            ("training.lisa_enabled", tcfg.lisa_enabled,
             "LISA trains the model's own weights"),
            ("training.unfrozen_parameters", bool(tcfg.unfrozen_parameters),
             "Spectrum trains the model's own weights"),
            ("training.quantization_aware", bool(tcfg.quantization_aware),
             "quantization-aware training converts the projections the kernels decode"),
            ("lora.target_parameters", bool(lora.target_parameters),
             "those adapters wrap parameters, not the Linear modules the kernels run"),
            ("data.format: raft", config.data.format == "raft",
             "RAFT's answer-only span mask needs the stock loss"),
        ):
            if clash:
                reason = f"cannot be combined with {label}: {why}."
                break
    return None if reason is None else f"training.packed_lora {reason}"
