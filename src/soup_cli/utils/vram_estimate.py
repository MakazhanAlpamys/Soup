"""One peak-VRAM estimate for a resident (not streamed) training run.

#2049: the ``soup train`` pre-flight (``utils/hardware_fit.py``) and ``soup profile``
(``utils/profiler.py``) priced the same run differently, in different units, and
neither had a vocabulary, so neither charged the logits and loss chain that
decides whether a long, large-vocabulary run fits. Both now call
:func:`estimate_vram`, as do the MCP ``profile`` tool and ``soup plan``.

Every term is in bytes; GB in the output is 1e9 bytes. The terms:

- ``weights``: the base at its loaded precision. Under 4-bit and 8-bit loading only
  the decoder's linear layers are quantized. Transformers casts the embeddings and
  output head to fp32 for training; Unsloth 4-bit keeps them in 16-bit (measured).
  Full fine-tuning keeps fp32 master weights (#339).
- ``adapters``: LoRA's A and B matrices, r * (in + out) per adapted module per layer,
  from the model's own dims and the configured target modules, in fp32.
- ``gradients`` and ``optimizer``: per trainable parameter.
- ``activations``: what the backward pass keeps per layer; with gradient
  checkpointing, one boundary tensor per layer plus one layer recomputed. Under 8-bit
  loading a checkpointed layer also keeps its int8 linear layers' inputs, which
  bitsandbytes holds outside checkpointing (``INT8_CHECKPOINT_KEPT``, and
  ``INT8_CHECKPOINT_KEPT_FP32`` for a ``kbit_fp32`` run).
- ``logits``: the output head's logits and the loss chain over them, at the streaming
  planner's measured loss arithmetic (``layer_stream.LOGITS_LOSS_BYTES_PER_ELEMENT``).
  A PRM step builds the logits and no loss over them, so it is charged the logits
  alone, the half-precision output and its fp32 copy
  (``LOGITS_TENSOR_BYTES_PER_ELEMENT``), on the one position ``logits_to_keep=1``
  keeps (#2368; every position for ``FULL_LOGITS_MODEL_TYPES`` and remote-code
  bases). Checkpointing does not reduce it.
- ``dequant``: under 4-bit loading, the bf16 copy of the largest weight a matmul
  dequantizes.
- ``teacher`` and ``distill`` (``task: distill``, #2909): the frozen teacher's weights at
  the dtype its checkpoint stores, and what a token-level distillation step builds over
  the vocabulary on top of the student's own logits: the teacher's logits and the fp32
  divergence over both (``trainer/distill.py``'s ``_compute_distill_term``), per element
  of batch x sequence x vocabulary, by kernel, divergence and chunk size, measured
  (``benchmarks/gate-distill-teacher-vram.md``).
- ``teacher_generation`` (``distill_mode: sequence``, #2909): the teacher writes the
  training text during setup and is freed, so its weights, its generation cache and its
  prefill belong to that phase; this is how far it exceeds training, or 0.
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
from typing import Any, Optional, Union

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
#: On the Transformers backend, a 4-bit or 8-bit run on a card with bf16 hardware keeps
#: the embeddings, the norms and an untied output head in bf16: ``prepare_kbit_model``
#: (``utils/kbit_prepare.py``, #2609) casts back what peft's
#: ``prepare_model_for_kbit_training`` upcast. Qwen2.5-7B NF4 on an RTX 4090: those
#: tensors were 4,158 MiB at fp32 and 2,079 MiB at bf16, and the allocator peak fell by
#: 3,905 MiB (``benchmarks/gate-2609-kbit-fp32-ab.md``).
KBIT_UNQUANTIZED_BYTES = 2.0
#: The same tensors in fp32: peft's own preparation, which a card without bf16 hardware
#: (``RunSpec.kbit_fp32``) and ``grpo_fp16`` keep, and what every resident row of
#: ``benchmarks/evidence/vram-2049-resident.json`` was measured with.
KBIT_UNQUANTIZED_FP32_BYTES = 4.0
#: Both embedding and head stayed bf16 after a real 4-bit Unsloth load and LoRA
#: injection on an RTX 5090 (Unsloth 2026.10.10). See the pinned model, dtypes and
#: reproduction in ``benchmarks/unsloth-4bit-weight-dtypes.md``. No runtime probe.
UNSLOTH_4BIT_EMBEDDING_HEAD_BYTES = 2.0

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

#: Memory the optimizer step allocates on top of its state, per trainable parameter
#: (#2277). torch's AdamW without ``fused=True`` takes the foreach path on CUDA, and
#: its ``_foreach_sqrt`` holds one fp32 copy of ``exp_avg_sq``: 4 bytes, measured on
#: SmolLM2-135M full fine-tuning. The others are 0 until one is measured. A slight
#: upper bound, since torch updates one parameter group at a time.
OPTIMIZER_STEP_SCRATCH_BYTES_PER_PARAM: dict[str, float] = {
    "adamw_torch": 4.0,
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
#: What a checkpointed layer keeps on top of its boundary under 8-bit loading, per token
#: in bf16 values, as (per hidden, per intermediate). bitsandbytes' int8 matmul
#: (``MatMul8bitLt``) holds its input on ``ctx.tensors``, not through
#: ``save_for_backward``, and checkpointing only drops what is saved through
#: ``save_for_backward``. So every int8 linear layer's input stays on the card from the
#: forward pass until that layer's backward: the q/k/v input, the o_proj input, the
#: gate/up input and the down_proj input. On a card with bf16 hardware the norms stay in
#: bf16 (``KBIT_UNQUANTIZED_BYTES``, #2683), so all four are bf16: 1 + 1 + 1 per hidden
#: and 1 per intermediate, read off with forward hooks on Qwen2.5-0.5B, Qwen2.5-1.5B and
#: SmolLM2-360M on main ``3f9ba5ce``. 4-bit's matmul keeps only the weight, so nf4 has no
#: such term (``benchmarks/gate-int8-checkpointing-vram.md``).
INT8_CHECKPOINT_KEPT = (3.0, 1.0)
#: The same inputs in a ``kbit_fp32`` run, whose norms are fp32 (PEFT's own preparation):
#: the q/k/v input and the gate/up input in fp32, the o_proj input in half precision and
#: the down_proj input in fp32, so 2 + 2 + 1 per hidden and 2 per intermediate. The
#: down_proj input is half precision instead when gate/up met an outlier column
#: (bitsandbytes adds the outliers back with an ``addmm`` that autocast runs in half
#: precision); fp32 is the upper bound. Measured before #2683 (main ``527314a4``) and on
#: ``auto_mixed_precision`` fp16 runs on ``3f9ba5ce``.
INT8_CHECKPOINT_KEPT_FP32 = (5.0, 2.0)

# --- logits ----------------------------------------------------------------------

#: Under 4-bit and 8-bit loading the logits cost 2 bytes an element more than under
#: bf16 loading. Originally the fp32 output head's logits (4 bytes, not 2). #2609 left it
#: as it was: with the head back in bf16 the profile of the step still held three fp32
#: ``[8, 512, 152064]`` logits tensors (7,128 MiB) at the peak, the same blocks as with
#: the fp32 head, and it removed 0 MiB of them (``benchmarks/gate-2609-kbit-fp32-ab.md``).
KBIT_LOGITS_EXTRA_BYTES = 2
#: The logits a step that builds no loss over them holds at the end of its forward:
#: the half-precision output autocast builds (2 bytes an element) and the fp32 copy
#: Accelerate's ``convert_outputs_to_fp32`` makes of it (4), both live at once. What a
#: PRM step (``RunSpec.logits_loss`` false) is charged, per position it keeps.
LOGITS_TENSOR_BYTES_PER_ELEMENT = 6
#: #2368: a PRM step passes ``logits_to_keep=1`` where the causal-LM forward takes it,
#: so its logits cover one position. These causal-LM model types (transformers'
#: ``MODEL_FOR_CAUSAL_LM_MAPPING_NAMES``) have a forward without the argument and
#: build the logits at every position. The estimate cannot see a forward, so a
#: remote-code base (``ModelShape.remote_code``) is charged every position too.
FULL_LOGITS_MODEL_TYPES = frozenset(
    {
        "gemma4_assistant",
        "gemma4_unified_assistant",
        "musicgen",
        "musicgen_melody",
        "prophetnet",
        "trocr",
        "whisper",
        "xlstm",
    }
)
#: Config flags whose fused loss can avoid building the full logits.
FUSED_LOSS_FLAGS = frozenset({"use_liger", "use_cut_ce"})

#: ``training.activation_offloading``'s modes. The estimate does not model what they move
#: off the card: the activation term is charged as resident, and a note says so (#2327).
ACTIVATION_OFFLOADING_MODES = frozenset({"cpu", "disk"})

# --- 4-bit dequantization --------------------------------------------------------

#: bf16 copies of the largest dequantized weight live at once under 4-bit loading.
DEQUANT_COPIES = 1.0

# --- distillation (#2909) ----------------------------------------------------------

#: What a token-level distillation step holds over the vocabulary beyond the student's
#: own logits and loss chain (``logits``), per element of batch x sequence x vocabulary,
#: when ``_compute_distill_term`` (``trainer/distill.py``) runs its divergence as ONE
#: chunk (``distill_chunk_size`` unset): the teacher's half-precision logits, the fp32
#: copies of both models' logits, their masked selections, the two log-softmaxes and the
#: divergence's products. By kernel and ``distill_divergence``: ``dense`` is the shipped
#: kernel, ``checkpoint`` is ``distill_checkpoint: true``, whose one chunk is recomputed in
#: the backward and peaks as high as ``dense`` for the two KL divergences (measured equal)
#: and a little lower for JS. The largest figure measured, rounded up, on an RTX PRO 5000
#: Blackwell (``benchmarks/gate-distill-teacher-vram.md``). A batch of two or more holds
#: about 4 bytes an element more than a batch of one, since ``student_logits[:, :-1, :]
#: .contiguous()`` in the CE path is then a copy, not a view; a batch of one is priced
#: that much high.
DISTILL_ONE_CHUNK_BYTES_PER_ELEMENT: dict[str, dict[str, float]] = {
    "dense": {"forward_kl": 38.0, "reverse_kl": 38.0, "js": 50.0},
    "checkpoint": {"forward_kl": 38.0, "reverse_kl": 38.0, "js": 46.0},
}
#: With ``distill_chunk_size`` below the step's positions, what the step keeps for the
#: backward whatever the chunk size, per element of batch x sequence x vocabulary: the
#: inputs to the chunks and, without checkpointing, every chunk's saved tensors. Each
#: chunk's own tensors cost ``ONE_CHUNK - KEPT`` per element of chunk x vocabulary, so a
#: step is charged ``vocab x (KEPT x positions + (ONE_CHUNK - KEPT) x min(chunk,
#: positions))``. Measured linear in the chunk size from 32 tokens to one chunk; the
#: largest intercept, rounded up.
DISTILL_KEPT_BYTES_PER_ELEMENT: dict[str, dict[str, float]] = {
    "dense": {"forward_kl": 26.0, "reverse_kl": 26.0, "js": 38.0},
    "checkpoint": {"forward_kl": 18.0, "reverse_kl": 18.0, "js": 18.0},
}
DISTILL_DIVERGENCES = frozenset(DISTILL_ONE_CHUNK_BYTES_PER_ELEMENT["dense"])
#: ``uld_strategy`` and ``minillm_enabled`` compute their own loss, which reads none of
#: ``distill_divergence``, ``distill_chunk_size`` and ``distill_checkpoint``. Neither is
#: measured: both are priced at the largest measured figure, dense JS in one chunk, and
#: the estimate says so.
DISTILL_UNMEASURED_KERNELS = frozenset({"uld", "minillm"})
DISTILL_KERNELS = frozenset(DISTILL_ONE_CHUNK_BYTES_PER_ELEMENT) | DISTILL_UNMEASURED_KERNELS
DISTILL_UNMEASURED_BYTES_PER_ELEMENT = DISTILL_ONE_CHUNK_BYTES_PER_ELEMENT["dense"]["js"]
DISTILL_MODES = frozenset({"token", "sequence"})
#: The teacher loads with ``dtype="auto"``; a config.json that names no dtype is priced
#: as a 16-bit checkpoint.
TEACHER_DEFAULT_BYTES_PER_PARAM = 2.0
#: ``distill_mode: sequence``: the teacher answers each prompt with at most this many new
#: tokens, or ``data.max_length`` if shorter (``trainer/distill.py``).
SEQUENCE_KD_MAX_NEW_TOKENS = 256
#: Tensors of the teacher's intermediate width its MLP holds at once while it reads a
#: prompt (the gate and up projections, the activation and their product).
GENERATION_PREFILL_INTERMEDIATE_TENSORS = 4
#: The cuBLAS (8,320 KiB) and cuBLASLt (1 MiB) workspaces the first matmul allocates and
#: the process keeps: what stays allocated once the teacher is freed (measured).
GENERATION_WORKSPACE_BYTES = 9_568_256

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
    "teacher",
    "distill",
    "optimizer_step",
    "teacher_generation",
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
    #: False for a step that builds the logits but no loss over them (a PRM step,
    #: whose loss comes from a reward head): only the logits tensor is charged, on
    #: the positions ``logits_positions`` counts (#2368).
    logits_loss: bool = True
    #: The fused-loss flags set on a run whose trainer applies them
    #: (``use_liger``, ``use_cut_ce``). The logits term is still charged, since
    #: whether the patch lands is only known at train time, and a note says so.
    fused_loss: tuple[str, ...] = ()
    #: ``training.activation_offloading`` (``cpu``, ``disk`` or ``None``). The
    #: activation term is still charged as if every saved tensor stayed on the card,
    #: since what offloading moves off it is not measured, and a note says so.
    activation_offloading: Optional[str] = None
    #: The loader whose default LoRA targets and weight dtypes are being estimated.
    backend: str = "transformers"
    #: True when a 4-bit or 8-bit run holds its embeddings, norms and untied head in fp32:
    #: a card without bf16 hardware, ``grpo_fp16``, or a run measured before #2609. False
    #: is the bf16 default (``KBIT_UNQUANTIZED_BYTES``).
    kbit_fp32: bool = False
    #: ``task: distill`` (#2909): the frozen teacher's shape, as ``read_model_shape`` read
    #: ``training.teacher_model``; ``None`` for every other task, and then the four
    #: ``distill_*`` fields below are not read.
    teacher: Optional[ModelShape] = None
    #: ``training.distill_mode`` (``DISTILL_MODES``): ``token`` keeps the teacher for the
    #: whole run; ``sequence`` has it write the training text during setup and frees it.
    distill_mode: str = "token"
    #: The loss a token-level step computes over the two models' logits
    #: (``DISTILL_KERNELS``): ``dense``, ``checkpoint`` (``distill_checkpoint: true``),
    #: ``uld`` (``uld_strategy``) or ``minillm`` (``minillm_enabled``).
    distill_kernel: str = "dense"
    #: ``training.distill_chunk_size``: tokens per chunk of the divergence; ``None`` is
    #: one chunk.
    distill_chunk_size: Optional[int] = None
    #: ``training.distill_divergence`` (``DISTILL_DIVERGENCES``).
    distill_divergence: str = "forward_kl"

    def __post_init__(self) -> None:
        if not isinstance(self.kbit_fp32, bool):
            raise TypeError(f"kbit_fp32 must be bool, got {type(self.kbit_fp32).__name__}")
        if not isinstance(self.lm_head, bool):
            raise TypeError(f"lm_head must be bool, got {type(self.lm_head).__name__}")
        if not isinstance(self.logits_loss, bool):
            raise TypeError(
                f"logits_loss must be bool, got {type(self.logits_loss).__name__}"
            )
        if not set(self.fused_loss) <= FUSED_LOSS_FLAGS:
            raise ValueError(f"unknown fused_loss {self.fused_loss!r}")
        if self.activation_offloading is not None and not (
            isinstance(self.activation_offloading, str)
            and self.activation_offloading in ACTIVATION_OFFLOADING_MODES
        ):
            raise ValueError(f"unknown activation_offloading {self.activation_offloading!r}")
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
        if self.teacher is not None:
            if not isinstance(self.teacher, ModelShape):
                raise TypeError(
                    f"teacher must be ModelShape, got {type(self.teacher).__name__}"
                )
            if self.teacher.stored_bytes_per_param not in (None, 2.0, 4.0):
                raise ValueError(
                    "the teacher's stored_bytes_per_param must be 2.0, 4.0 or None, "
                    f"got {self.teacher.stored_bytes_per_param!r}"
                )
        if self.distill_mode not in DISTILL_MODES:
            raise ValueError(f"unknown distill_mode {self.distill_mode!r}")
        if self.distill_kernel not in DISTILL_KERNELS:
            raise ValueError(f"unknown distill_kernel {self.distill_kernel!r}")
        chunk = self.distill_chunk_size
        if chunk is not None and (isinstance(chunk, bool) or not isinstance(chunk, int)):
            raise TypeError(f"distill_chunk_size must be int, got {type(chunk).__name__}")
        if chunk is not None and chunk < 1:
            raise ValueError("distill_chunk_size must be >= 1")
        if self.distill_divergence not in DISTILL_DIVERGENCES:
            raise ValueError(f"unknown distill_divergence {self.distill_divergence!r}")

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
        return lora_targets(
            self.run.target_modules, self.shape.model_type, backend=self.run.backend
        )


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
        embedding_head_bytes = (
            KBIT_UNQUANTIZED_FP32_BYTES if run.kbit_fp32 else KBIT_UNQUANTIZED_BYTES
        )
        if run.backend == "unsloth" and run.quant == "4bit":
            embedding_head_bytes = UNSLOTH_4BIT_EMBEDDING_HEAD_BYTES
        return math.ceil(linear * per + embed * embedding_head_bytes)
    return math.ceil(shape.params * BYTES_PER_PARAM_BY_QUANT[run.quant])


def activation_mode(run: RunSpec) -> str:
    """``full`` (a trained base), ``nf4`` / ``int8`` (4-bit / 8-bit loading) or ``lora``."""
    if run.peft == "full":
        return "full"
    return {"4bit": "nf4", "8bit": "int8"}.get(run.quant, "lora")


def trainable_params(shape: ModelShape, run: RunSpec) -> int:
    if run.peft == "full":
        return shape.params
    return lora_param_count(
        shape, r=run.lora_r, target_modules=run.target_modules, backend=run.backend
    )


def activations_bytes(shape: ModelShape, run: RunSpec) -> int:
    tokens = run.seq_len * run.batch_size
    mode = activation_mode(run)
    extra_hidden, extra_inter = ACT_EXTRA[mode]
    adapted_inputs = (
        0
        if mode == "full"
        else sum(
            module_dims(shape, name)[0]
            for name in lora_targets(run.target_modules, shape.model_type, backend=run.backend)
        )
    )
    per_layer = tokens * ACTIVATION_BYTES * (
        (ACT_PER_HIDDEN + extra_hidden) * shape.hidden_size
        + (ACT_PER_INTERMEDIATE + extra_inter) * shape.intermediate_size
        + ACT_PER_ADAPTED_INPUT * adapted_inputs
    )
    boundary = tokens * shape.hidden_size * BOUNDARY_BYTES[mode]
    if mode == "int8":
        kept_hidden, kept_intermediate = (
            INT8_CHECKPOINT_KEPT_FP32 if run.kbit_fp32 else INT8_CHECKPOINT_KEPT
        )
        boundary += tokens * ACTIVATION_BYTES * (
            kept_hidden * shape.hidden_size + kept_intermediate * shape.intermediate_size
        )
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
    A step with no loss over the logits (``run.logits_loss`` false: a PRM step) is
    charged the logits alone, ``LOGITS_TENSOR_BYTES_PER_ELEMENT``: the half-precision
    output and the fp32 copy Accelerate makes of it. Its forward passes
    ``logits_to_keep=1`` (#2368), so it is charged one position, except on a model
    type whose forward lacks the argument (``FULL_LOGITS_MODEL_TYPES``) or a
    remote-code base, which keep every position (``logits_positions``).
    Under 4-bit and 8-bit loading either charge adds ``KBIT_LOGITS_EXTRA_BYTES``.
    A model without an LM head builds no logits over the vocabulary.
    """
    if shape.vocab_size is None or not run.lm_head:
        return 0
    from soup_cli.utils.layer_stream import LOGITS_LOSS_BYTES_PER_ELEMENT

    per = LOGITS_LOSS_BYTES_PER_ELEMENT if run.logits_loss else LOGITS_TENSOR_BYTES_PER_ELEMENT
    if activation_mode(run) in ("nf4", "int8"):
        per += KBIT_LOGITS_EXTRA_BYTES
    return math.ceil(shape.vocab_size * logits_positions(shape, run) * run.batch_size * per)


def logits_positions(shape: ModelShape, run: RunSpec) -> int:
    """The sequence positions the step's logits cover.

    Every position for a step with a loss over the logits. A PRM step
    (``run.logits_loss`` false) passes ``logits_to_keep=1`` where the forward takes
    it, so one position, unless the model type's forward has no such argument
    (``FULL_LOGITS_MODEL_TYPES``, the outer and the nested text config's type both
    checked) or the base ships its own code (``shape.remote_code``), whose forward
    the estimate cannot see: those keep every position, the side of over-pricing.
    """
    if run.logits_loss or shape.remote_code:
        return run.seq_len
    if {shape.model_type, shape.config_model_type} & FULL_LOGITS_MODEL_TYPES:
        return run.seq_len
    return 1


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


def teacher_bytes_per_param(teacher: ModelShape) -> float:
    """What a ``dtype="auto"`` load holds per parameter: the dtype config.json stores."""
    stored = teacher.stored_bytes_per_param
    return TEACHER_DEFAULT_BYTES_PER_PARAM if stored is None else float(stored)


def teacher_weights_bytes(run: RunSpec) -> int:
    """A distillation teacher's weights at the dtype its checkpoint stores (#2909)."""
    if run.teacher is None:
        return 0
    return math.ceil(run.teacher.params * teacher_bytes_per_param(run.teacher))


def distill_bytes(shape: ModelShape, run: RunSpec) -> int:
    """What a token-level distillation step adds over the vocabulary (#2909).

    Per element of batch x sequence x the student's vocabulary, every position charged
    (the divergence covers the trained ones): ``DISTILL_KEPT_BYTES_PER_ELEMENT`` for the
    whole step plus ``DISTILL_ONE_CHUNK - KEPT`` for one chunk of ``distill_chunk_size``
    positions; unset is one chunk of every position. ``uld`` and ``minillm`` are not
    measured and are charged ``DISTILL_UNMEASURED_BYTES_PER_ELEMENT`` on every position.
    Nothing for another task, ``distill_mode: sequence`` or an unknown vocabulary.
    """
    if run.teacher is None or run.distill_mode != "token" or shape.vocab_size is None:
        return 0
    positions = run.seq_len * run.batch_size
    if run.distill_kernel in DISTILL_UNMEASURED_KERNELS:
        return math.ceil(shape.vocab_size * positions * DISTILL_UNMEASURED_BYTES_PER_ELEMENT)
    one_chunk = DISTILL_ONE_CHUNK_BYTES_PER_ELEMENT[run.distill_kernel][run.distill_divergence]
    kept = DISTILL_KEPT_BYTES_PER_ELEMENT[run.distill_kernel][run.distill_divergence]
    chunk = positions if run.distill_chunk_size is None else min(run.distill_chunk_size, positions)
    return math.ceil(shape.vocab_size * (kept * positions + (one_chunk - kept) * chunk))


def teacher_generation_bytes(run: RunSpec) -> int:
    """What ``distill_mode: sequence`` holds over the student while its teacher writes.

    The teacher's weights, its key/value cache for a prompt of ``seq_len`` tokens and
    ``min(seq_len, SEQUENCE_KD_MAX_NEW_TOKENS)`` new ones (one row at a time), its MLP's
    tensors while it reads that prompt, and the workspaces of the first matmul.
    """
    teacher = run.teacher
    if teacher is None or run.distill_mode != "sequence":
        return 0
    per = teacher_bytes_per_param(teacher)
    tokens = run.seq_len + min(run.seq_len, SEQUENCE_KD_MAX_NEW_TOKENS)
    cache = 2 * teacher.num_layers * teacher.num_kv_heads * teacher.head_dim * per * tokens
    prefill = (
        GENERATION_PREFILL_INTERMEDIATE_TENSORS * teacher.intermediate_size * per * run.seq_len
    )
    return math.ceil(
        teacher_weights_bytes(run) + cache + prefill + GENERATION_WORKSPACE_BYTES
    )


def estimate_vram(shape: ModelShape, run: RunSpec) -> VramEstimate:
    """The per-term peak for ``run`` on a model of ``shape``."""
    trainable = trainable_params(shape, run)
    token_level = run.teacher is not None and run.distill_mode == "token"
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
        # A token-level teacher is resident for the whole run (#2909).
        "teacher": teacher_weights_bytes(run) if token_level else 0,
        "distill": distill_bytes(shape, run),
    }
    # The peak is the larger of two phases (#2277). The forward/backward pass holds
    # every term above. The optimizer step holds the parameters (a token-level
    # distillation teacher's too), their gradients, the optimizer state and the step's
    # scratch, after the activations, logits, dequantized copy and divergence are freed.
    # `optimizer_step` is how far the step exceeds the pass, so the total stays a sum
    # of terms.
    forward_backward = sum(terms.values())
    step = (
        terms["weights"]
        + terms["teacher"]
        + terms["adapters"]
        + terms["gradients"]
        + terms["optimizer"]
        + math.ceil(trainable * OPTIMIZER_STEP_SCRATCH_BYTES_PER_PARAM.get(run.optimizer, 0.0))
    )
    terms["optimizer_step"] = max(step - forward_backward, 0)
    training = forward_backward + terms["optimizer_step"]
    # #2909: with distill_mode: sequence the teacher writes the training text during setup
    # and is freed before the first step, so its weights belong to a third phase, which
    # holds the student and its adapters too. `teacher_generation` is how far that phase
    # exceeds training, or 0.
    generation = (
        terms["weights"] + terms["adapters"] + teacher_generation_bytes(run)
        if run.teacher is not None and run.distill_mode == "sequence"
        else 0
    )
    terms["teacher_generation"] = max(generation - training, 0)
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
    if terms["optimizer_step"] and not terms["teacher_generation"]:
        notes.append(
            "The optimizer step, not the forward and backward pass, sets this peak: "
            "the update holds the optimizer state and its scratch after the "
            "activations are freed."
        )
    if run.fused_loss and terms["logits"]:
        flags = " and ".join(f"training.{flag}" for flag in sorted(run.fused_loss))
        notes.append(
            f"{flags} {'is' if len(run.fused_loss) == 1 else 'are'} set: if the fused loss "
            "applies, the full logits are not built, "
            f"so the logits term ({terms['logits'] / GB:.1f} GB) overstates the peak. "
            "The fused path is not measured."
        )
    if run.activation_offloading is not None:
        notes.append(activation_offloading_note(run.activation_offloading, terms["activations"]))
    notes.extend(distill_notes(run, generation=generation, training=training))
    return VramEstimate(terms=terms, shape=shape, run=run, notes=tuple(notes))


def distill_notes(run: RunSpec, *, generation: int, training: int) -> list[str]:
    """What the estimate says about a distillation teacher it priced (#2909)."""
    teacher = run.teacher
    if teacher is None:
        return []
    notes = []
    if teacher.guessed:
        notes.append(
            "No config.json could be read for training.teacher_model, so the teacher's "
            "shape is guessed from its parameter count and its terms are approximate."
        )
    if run.distill_mode == "sequence":
        larger = "writing" if generation > training else "training"
        notes.append(
            "distill_mode: sequence: the teacher writes the training text during setup and "
            "is freed before the first step. Writing holds both models, the teacher's cache "
            f"and its prefill, {generation / GB:.2f} GB; a training step holds "
            f"{training / GB:.2f} GB. The peak is the larger of the two, {larger}."
        )
    elif run.distill_kernel in DISTILL_UNMEASURED_KERNELS:
        setting = "uld_strategy" if run.distill_kernel == "uld" else "minillm_enabled"
        notes.append(
            f"training.{setting} is set: that loss is not measured, so it is priced as the "
            "largest divergence that is (JS in one chunk, "
            f"{DISTILL_UNMEASURED_BYTES_PER_ELEMENT:.0f} bytes an element of batch x "
            "sequence x vocabulary)."
        )
    return notes


def activation_offloading_note(mode: str, activation_bytes: int) -> str:
    """What the estimate says when ``training.activation_offloading`` is set.

    The resident estimate and the streaming plan panel both print it, with their own
    activation term.
    """
    return (
        f"training.activation_offloading: {mode} is set: the activation term "
        f"({activation_bytes / GB:.2f} GB) is charged as if every saved tensor stayed on "
        "the card; what offloading moves off it is not modelled."
    )


def fit_verdict(total_bytes: float, available_bytes: float) -> str:
    """``fits``, ``tight`` (fits, but not with the pre-flight's margin) or ``does not fit``."""
    if total_bytes * (1.0 + VRAM_SAFETY_MARGIN) <= available_bytes:
        return "fits"
    if total_bytes <= available_bytes:
        return "tight"
    return "does not fit"


__all__ = [
    "ACTIVATION_OFFLOADING_MODES",
    "BYTES_PER_PARAM_BY_QUANT",
    "DISTILL_DIVERGENCES",
    "DISTILL_KEPT_BYTES_PER_ELEMENT",
    "DISTILL_KERNELS",
    "DISTILL_MODES",
    "DISTILL_ONE_CHUNK_BYTES_PER_ELEMENT",
    "GB",
    "OPTIMIZER_BYTES_PER_PARAM",
    "OPTIMIZER_STEP_SCRATCH_BYTES_PER_PARAM",
    "TERM_NAMES",
    "VRAM_SAFETY_MARGIN",
    "RunSpec",
    "VramEstimate",
    "activation_offloading_note",
    "estimate_vram",
    "fit_verdict",
]
