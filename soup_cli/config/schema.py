"""Pydantic schemas for soup.yaml config — single source of truth."""

import re
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class LoraConfig(BaseModel):
    r: int = Field(default=64, description="LoRA rank")
    alpha: int = Field(default=16, description="LoRA alpha")
    dropout: float = Field(default=0.05, description="LoRA dropout")
    target_modules: Union[str, List[str]] = Field(
        default="auto",
        description="Target modules for LoRA. 'auto' = let peft decide.",
    )
    use_dora: bool = Field(
        default=False,
        description="Enable DoRA (Weight-Decomposed Low-Rank Adaptation)",
    )
    use_rslora: bool = Field(
        default=False,
        description="Enable rank-stabilized LoRA scaling (better for high ranks)",
    )
    use_vera: bool = Field(
        default=False,
        description=(
            "Enable VeRA (Vector-based Random Matrix Adaptation). "
            "Shared random matrices — much smaller memory than LoRA. "
            "Mutually exclusive with use_dora and use_olora."
        ),
    )
    use_olora: bool = Field(
        default=False,
        description=(
            "Enable OLoRA (Orthogonal LoRA init via QR decomposition). "
            "Passes init_lora_weights='olora' to peft. "
            "Mutually exclusive with use_dora and use_vera."
        ),
    )

    @model_validator(mode="after")
    def _validate_peft_exclusivity(self) -> "LoraConfig":
        enabled = [
            name for name, value in (
                ("use_dora", self.use_dora),
                ("use_vera", self.use_vera),
                ("use_olora", self.use_olora),
            )
            if value
        ]
        if len(enabled) > 1:
            raise ValueError(
                f"PEFT methods are mutually exclusive, got multiple enabled: "
                f"{', '.join(enabled)}. Pick at most one of use_dora, "
                f"use_vera, use_olora."
            )
        return self


class DataConfig(BaseModel):
    train: str = Field(..., description="Path to training data or HF dataset name")
    format: Literal[
        "alpaca", "sharegpt", "chatml", "dpo", "kto", "llava", "sharegpt4v",
        "plaintext", "embedding", "audio", "tool-calling", "auto",
    ] = Field(
        default="auto",
        description="Data format",
    )
    val_split: float = Field(default=0.1, ge=0.0, le=0.5, description="Validation split ratio")
    max_length: int = Field(
        default=2048, ge=64, le=1048576,
        description="Max sequence length in tokens",
    )
    image_dir: Optional[str] = Field(
        default=None,
        description="Base directory for resolving relative image paths in vision datasets",
    )
    audio_dir: Optional[str] = Field(
        default=None,
        description="Base directory for resolving relative audio paths in audio datasets",
    )
    train_on_responses_only: bool = Field(
        default=True,
        description=(
            "Mask non-assistant tokens with IGNORE_INDEX (-100). When True, "
            "only assistant content contributes to the SFT loss. Mirrors "
            "LlamaFactory + Axolotl default — replaces TRL's heuristic. (v0.36.0)"
        ),
    )
    train_on_messages_with_train_field: bool = Field(
        default=False,
        description=(
            "Per-message training mask via messages[i].train: bool. "
            "Mutually exclusive with train_on_responses_only. (v0.36.0)"
        ),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "Override the tokenizer chat template. Accepts a registered "
            "name (chatml, llama3, qwen2.5, mistral, gemma3, phi4, "
            "deepseek-r1) or a raw Jinja string. None = use the tokenizer's "
            "shipped template (errors loudly if absent). (v0.36.0)"
        ),
    )

    @field_validator("chat_template")
    @classmethod
    def _validate_chat_template(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("chat_template must be a string")
        if not value:
            return None
        if "\x00" in value:
            raise ValueError("chat_template must not contain null bytes")
        if len(value) > 65536:
            raise ValueError("chat_template must be <= 64KB")
        # Block Jinja directives that touch the filesystem or load arbitrary
        # modules. Only control-flow + variable interpolation are allowed
        # for raw chat-template strings (v0.36.0 security review fix).
        lower = value.lower()
        for tag in ("{%- include", "{% include", "{%- import", "{% import",
                    "{%- from", "{% from", "{%- macro", "{% macro",
                    "{%- extends", "{% extends"):
            if tag in lower:
                directive = tag.split(None, 1)[-1]
                raise ValueError(
                    f"chat_template may not use Jinja '{directive}' directive — "
                    f"only control-flow and variable interpolation are allowed."
                )
        return value

    @model_validator(mode="after")
    def _validate_loss_mask_exclusivity(self) -> "DataConfig":
        if self.train_on_responses_only and self.train_on_messages_with_train_field:
            raise ValueError(
                "train_on_responses_only and train_on_messages_with_train_field "
                "are mutually exclusive. Disable one. The per-message 'train' "
                "field is opt-in for fine-grained per-message control."
            )
        return self


class EvalGateConfig(BaseModel):
    """Eval-Gated Training config (v0.26.0 Part B).

    Runs a declarative eval suite at epoch boundaries and halts training
    if any task regresses below ``regression_threshold`` vs the baseline.
    """

    enabled: bool = Field(
        default=False,
        description="Turn the eval gate on",
    )
    suite: Optional[str] = Field(
        default=None,
        description="Path to eval-suite YAML (evals/gate.yaml)",
    )
    every_n_epochs: int = Field(
        default=1, ge=1, le=100,
        description="Run gate every N epochs (1-100)",
    )
    regression_threshold: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Max absolute drop vs baseline before regression fires",
    )
    baseline: Optional[str] = Field(
        default=None,
        description="registry://<id> | 'previous' | file path - scores to compare against",
    )
    on_regression: Literal["stop", "warn", "continue"] = Field(
        default="stop",
        description="Action on regression: stop training | warn only | continue",
    )

    @model_validator(mode="after")
    def _require_suite_when_enabled(self) -> "EvalGateConfig":
        if self.enabled and not self.suite:
            raise ValueError(
                "eval_gate.suite is required when eval_gate.enabled=true"
            )
        return self


class TrainingConfig(BaseModel):
    epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    lr: float = Field(default=2e-5, gt=0, description="Learning rate")
    batch_size: Union[int, Literal["auto"]] = Field(
        default="auto",
        description="Batch size. 'auto' = find max that fits in memory.",
    )
    auto_batch_size_strategy: Literal["auto", "static", "probe"] = Field(
        default="auto",
        description=(
            "How to pick the auto batch size: 'static' (fast formula), "
            "'probe' (real OOM try/halve loop), 'auto' (probe on CUDA, "
            "static on CPU). Default 'auto' (v0.36.0)."
        ),
    )
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    quantization: Literal[
        "4bit",
        "8bit",
        "none",
        "gptq",
        "awq",
        "hqq:1bit",
        "hqq:2bit",
        "hqq:3bit",
        "hqq:4bit",
        "hqq:5bit",
        "hqq:6bit",
        "hqq:8bit",
        "aqlm",
        "eetq",
        "mxfp4",
        "fp8",
    ] = Field(
        default="4bit",
        description=(
            "Quantization (v0.38.0 — Quant Menu): "
            "4bit (BNB QLoRA), 8bit (BNB), none, "
            "gptq / awq (load pre-quantized checkpoint, train LoRA on top), "
            "hqq:Nbit (HQQ 1-8 bit, N in {1..6, 8}), "
            "aqlm (extreme 2-bit), eetq (8-bit fast), "
            "mxfp4 (BNB 4-bit MXFP4 quant_type), "
            "fp8 (load FP8 checkpoint with dequantize-on-load)."
        ),
    )
    gptq_disable_exllama: bool = Field(
        default=True,
        description=(
            "v0.38.0 — disable exllama backend for GPTQ. PEFT requires triton "
            "backend; exllama silently breaks adapter training."
        ),
    )
    bnb_4bit_quant_storage: Optional[
        Literal["uint8", "float16", "bfloat16", "float32"]
    ] = Field(
        default=None,
        description=(
            "v0.38.0 Part G — BNB 4-bit storage dtype. Required for FSDP+QLoRA "
            "(set to 'bfloat16' or 'float16' to match compute dtype). "
            "When None, BNB picks 'uint8' (legacy default)."
        ),
    )
    quantization_aware: Union[bool, Literal["fp8"]] = Field(
        default=False,
        description=(
            "Quantization-Aware Training. False=off, True=int8 QAT (torchao), "
            "'fp8'=FP8 training on H100/B100 (v0.28.0)."
        ),
    )
    fp8_recipe: Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"] = Field(
        default="tensorwise",
        description=(
            "FP8 scaling recipe (only used when quantization_aware='fp8'). "
            "'tensorwise' (fastest, default), 'rowwise' (more accurate, CUTLASS rowwise), "
            "'rowwise_with_gw_hp' (most accurate, grad_weight in high precision). (v0.28.1)."
        ),
    )
    optimizer: str = Field(default="adamw_torch", description="Optimizer name")
    scheduler: str = Field(default="cosine", description="LR scheduler type")
    save_steps: int = Field(default=100, description="Save checkpoint every N steps")
    logging_steps: int = Field(default=10, description="Log metrics every N steps")
    # DPO-specific
    dpo_beta: float = Field(
        default=0.1, gt=0, description="DPO beta — KL penalty coefficient"
    )
    # KTO-specific
    kto_beta: float = Field(
        default=0.1, gt=0, description="KTO beta — KL penalty coefficient"
    )
    # ORPO-specific
    orpo_beta: float = Field(
        default=0.1, gt=0, description="ORPO beta — odds ratio weight"
    )
    # SimPO-specific
    simpo_gamma: float = Field(
        default=0.5, ge=0, description="SimPO gamma — reward margin term"
    )
    cpo_alpha: float = Field(
        default=1.0, gt=0, description="CPO/SimPO alpha — NLL loss weight"
    )
    # IPO-specific (uses DPO trainer with loss_type='ipo')
    ipo_tau: float = Field(
        default=0.1, gt=0, description="IPO tau — regularization strength"
    )
    # GRPO-specific
    grpo_beta: float = Field(
        default=0.1, gt=0, description="GRPO beta — KL penalty coefficient"
    )
    num_generations: int = Field(
        default=4, ge=2, description="Number of generations per prompt for GRPO"
    )
    reward_fn: Optional[str] = Field(
        default="accuracy",
        description=(
            "Reward function: 'accuracy', 'format', 'verifiable', "
            "or path to custom .py file"
        ),
    )
    # RLVR — verifiable reward domain (Part C of v0.25.0)
    verifiable_domain: Optional[Literal["math", "code", "json_schema"]] = Field(
        default=None,
        description=(
            "RLVR verifiable reward domain: math | code | json_schema. "
            "Required when reward_fn='verifiable'."
        ),
    )
    # PPO-specific
    ppo_epochs: int = Field(
        default=4, ge=1, description="Number of PPO optimization epochs per batch"
    )
    ppo_clip_ratio: float = Field(
        default=0.2, gt=0, le=1.0, description="PPO clipping range for policy ratio"
    )
    ppo_kl_penalty: float = Field(
        default=0.05, ge=0, description="KL divergence penalty coefficient for PPO"
    )
    reward_model: Optional[str] = Field(
        default=None,
        description="Path or HF ID of a trained reward model for PPO",
    )
    # LoRA+ — different learning rates for A and B matrices
    loraplus_lr_ratio: Optional[float] = Field(
        default=None,
        gt=0,
        description="LoRA+ lr ratio: lr_B = lr × ratio. None = disabled (standard LoRA).",
    )
    # GaLore — memory-efficient full-parameter training
    use_galore: bool = Field(
        default=False,
        description="Enable GaLore (Gradient Low-Rank Projection) for memory-efficient training",
    )
    galore_rank: int = Field(
        default=128, ge=1, description="GaLore projection rank"
    )
    galore_update_proj_gap: int = Field(
        default=200, ge=1, description="GaLore projection update interval (steps)"
    )
    galore_scale: float = Field(
        default=0.25, gt=0, description="GaLore gradient scaling factor"
    )
    # MoE-specific
    moe_lora: bool = Field(
        default=False,
        description="Enable MoE-aware LoRA (ScatterMoE) — applies LoRA to expert FFN layers",
    )
    moe_aux_loss_coeff: float = Field(
        default=0.01,
        ge=0,
        description="Auxiliary load-balancing loss coefficient for MoE models",
    )
    # Performance — Liger Kernel (fused operations)
    use_liger: bool = Field(
        default=False,
        description="Enable Liger Kernel fused operations (20-60% memory savings, 20-40% speedup)",
    )
    # Performance — FlashAttention
    use_flash_attn: bool = Field(
        default=False,
        description="Enable FlashAttention (auto-detects v2/v3/v4 for faster attention)",
    )
    # Performance — Ring FlashAttention (sequence parallelism)
    use_ring_attention: bool = Field(
        default=False,
        description="Enable Ring FlashAttention for sequence parallelism across GPUs",
    )
    # Long-context — RoPE scaling
    rope_scaling_type: Optional[Literal["linear", "dynamic", "yarn", "longrope"]] = Field(
        default=None,
        description="RoPE scaling method for long-context: linear, dynamic, yarn, longrope",
    )
    gradient_checkpointing: Union[
        bool, Literal["selective", "medium", "full", "auto"]
    ] = Field(
        default=False,
        description=(
            "Gradient checkpointing for memory savings on long sequences. "
            "False/True (legacy bool) or tier: 'selective' (attention only), "
            "'medium' (every other block), 'full' (all blocks), "
            "'auto' (picks based on available VRAM). (v0.28.0)."
        ),
    )
    # v0.28.0 — Cut Cross-Entropy (CCE): saves 8-24GB on large-vocab models
    use_cut_ce: bool = Field(
        default=False,
        description=(
            "Enable Cut Cross-Entropy (CCE) for large-vocab models. "
            "Saves 8-24GB VRAM on Llama 3.1 128k vocab. Requires cut_cross_entropy. "
            "Mutually exclusive with Unsloth/MLX backends."
        ),
    )
    # v0.28.0 — Kernel auto-composition (Liger + Unsloth + FlashAttn per-layer)
    kernel_auto_compose: bool = Field(
        default=False,
        description=(
            "Benchmark and auto-select the fastest kernel combination "
            "(Liger / FlashAttn / baseline) on the first few steps. (v0.28.0)."
        ),
    )
    # v0.28.0 — Cross-document attention masking for sample packing
    packing_cross_doc_attn_mask: bool = Field(
        default=False,
        description=(
            "When packing is enabled, prevent attention bleed between packed "
            "documents. Requires packing=true. (v0.28.0)."
        ),
    )
    # v0.28.0 — Activation offloading (CPU/disk) for small-VRAM large-batch
    activation_offloading: Optional[Literal["cpu", "disk"]] = Field(
        default=None,
        description=(
            "Offload activations to CPU or disk during backward pass. "
            "None=off, 'cpu'=offload to RAM, 'disk'=offload to tmp file. (v0.28.0)."
        ),
    )
    # Embedding-specific
    embedding_loss: Literal["contrastive", "triplet", "cosine"] = Field(
        default="contrastive",
        description="Loss function for embedding training: contrastive, triplet, or cosine",
    )
    embedding_margin: float = Field(
        default=0.5, gt=0,
        description="Margin for contrastive/triplet loss (higher = stricter separation)",
    )
    embedding_pooling: Literal["mean", "cls", "last"] = Field(
        default="mean",
        description="Pooling strategy for sentence embeddings: mean, cls, or last token",
    )
    embedding_temperature: float = Field(
        default=0.05, gt=0,
        description="Temperature for contrastive (InfoNCE) loss — lower = stricter similarity",
    )
    # Curriculum learning — sort dataset by difficulty
    curriculum: bool = Field(
        default=False,
        description="Enable curriculum learning (sort dataset by difficulty, easy → hard)",
    )
    curriculum_metric: Literal["length", "perplexity", "loss"] = Field(
        default="length",
        description="Metric for curriculum difficulty: length, perplexity, or loss",
    )
    curriculum_buckets: int = Field(
        default=4, ge=1, le=20,
        description="Number of difficulty stages for curriculum learning",
    )
    # Loss watchdog — auto-stop on loss spikes
    loss_watchdog: bool = Field(
        default=False,
        description="Enable loss spike detection (auto-stop if loss exceeds threshold)",
    )
    loss_watchdog_threshold: float = Field(
        default=3.0,
        gt=0,
        le=100.0,
        description="Stop training if loss exceeds this threshold",
    )
    loss_watchdog_patience: int = Field(
        default=5,
        ge=1,
        le=1000,
        description="Consecutive high-loss steps before stopping",
    )
    # Loss spike auto-recovery (v0.32.0 Part E) — extends watchdog
    loss_spike_recovery: bool = Field(
        default=False,
        description=(
            "On watchdog trigger: rollback to last checkpoint, decay LR, "
            "and resume (instead of stopping). Requires loss_watchdog=true."
        ),
    )
    loss_spike_recovery_max_attempts: int = Field(
        default=3, ge=1, le=10,
        description="Max number of spike-recovery attempts before giving up",
    )
    loss_spike_recovery_lr_decay: float = Field(
        default=0.5, gt=0.0, lt=1.0,
        description="Multiply LR by this factor on each spike recovery (0.5 = halve)",
    )
    # Convergence detection (v0.32.0 Part F)
    convergence_detection: bool = Field(
        default=False,
        description=(
            "Watch for loss plateau / oscillation and surface advice "
            "(continue / early_stop / lower_lr) at the end of training."
        ),
    )
    convergence_window: int = Field(
        default=50, ge=5, le=10_000,
        description="Number of recent losses to inspect for plateau / oscillation",
    )
    convergence_rel_tol: float = Field(
        default=0.005, gt=0.0, le=1.0,
        description="Relative range threshold below which the window is a plateau",
    )
    # Warmup auto-schedule (v0.32.0 Part D) — reuses pre-existing warmup_ratio.
    warmup_auto: bool = Field(
        default=False,
        description=(
            "Auto-pick warmup_steps from dataset_size × epochs × warmup_ratio. "
            "Overrides any manual warmup_steps in the trainer."
        ),
    )
    # Auto mixed-precision (v0.32.0 Part C)
    auto_mixed_precision: bool = Field(
        default=False,
        description=(
            "Pick bf16/fp16 based on model + GPU compute capability. "
            "Overrides manual --bf16 / --fp16 trainer flags."
        ),
    )
    # Live grad-accum monitoring (v0.32.0 Part B)
    grad_accum_auto_tune: bool = Field(
        default=False,
        description=(
            "Monitor VRAM each step; warn (and recommend new batch/accum) "
            "when memory pressure is high. Advisory in v0.32.0; live "
            "DataLoader rebuild deferred to v0.32.1."
        ),
    )
    grad_accum_pressure_threshold: float = Field(
        default=0.92, gt=0.05, lt=0.99,
        description="VRAM utilisation fraction that triggers a recommendation",
    )
    # Freeze training — freeze bottom layers for parameter-efficient training
    freeze_layers: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Freeze first N layers (from bottom). Train only remaining layers.",
    )
    freeze_ratio: Optional[float] = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Freeze this fraction of layers (0.75 = freeze 75% from bottom).",
    )
    # Sample packing — pack multiple short samples into one sequence
    packing: bool = Field(
        default=False,
        description="Pack multiple short samples into one sequence for faster training",
    )
    # v0.37.0 — Multipack First-Fit-Decreasing bin-packing sampler
    multipack: bool = Field(
        default=False,
        description=(
            "Use FFD bin-packing sampler to maximise tokens-per-batch on "
            "uneven-length data. Mutually exclusive with packing. Only "
            "supported for sft / pretrain tasks (transformers backend). "
            "(v0.37.0)."
        ),
    )
    # NEFTune — noisy embeddings for better fine-tuning
    neftune_alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=50.0,
        description="NEFTune noise alpha (0-50). Adds noise to embeddings for better chat quality.",
    )
    # Training Intelligence — Part G of v0.25.0
    # Forgetting detection
    forgetting_detection: bool = Field(
        default=False,
        description="Enable periodic general-knowledge eval to detect catastrophic forgetting",
    )
    forgetting_eval_steps: int = Field(
        default=100, ge=10, le=10000,
        description="Run forgetting eval every N steps",
    )
    forgetting_threshold: float = Field(
        default=0.10, ge=0.01, le=0.50,
        description="Warn if accuracy drops > threshold from baseline (0.01-0.50)",
    )
    forgetting_benchmark: Literal["mini_mmlu", "mini_common_sense", "mini_instruction"] = Field(
        default="mini_mmlu",
        description="Built-in mini benchmark used for forgetting detection",
    )
    forgetting_stop: bool = Field(
        default=False,
        description="Auto-stop training on severe forgetting (red-level alert)",
    )
    # Checkpoint intelligence
    checkpoint_intelligence: bool = Field(
        default=False,
        description="Enable auto-best-checkpoint tracking by quality (not just loss)",
    )
    checkpoint_eval_steps: int = Field(
        default=200, ge=50, le=10000,
        description="Run checkpoint quality eval every N steps",
    )
    checkpoint_eval_metric: Literal["judge", "mmlu", "custom", "composite"] = Field(
        default="composite",
        description="Metric used for checkpoint quality selection",
    )
    checkpoint_eval_tasks: Optional[str] = Field(
        default=None,
        description="Optional JSONL file with custom eval tasks for checkpoint scoring",
    )
    checkpoint_keep_top: int = Field(
        default=3, ge=1, le=20,
        description="Keep top-N checkpoints by quality, delete the rest",
    )
    early_stop_on_regression: bool = Field(
        default=False,
        description="Stop training when quality regresses across consecutive evals",
    )
    early_stop_patience: int = Field(
        default=2, ge=1, le=10,
        description="Consecutive regressions before early stopping (1-10)",
    )
    # Eval-Gated Training — Part B of v0.26.0
    eval_gate: Optional["EvalGateConfig"] = Field(
        default=None,
        description="Optional EvalGateConfig — block training on regressions",
    )
    # Multi-GPU Mastery — v0.27.0
    use_fsdp2_compile: bool = Field(
        default=False,
        description=(
            "Enable torch.compile on top of FSDP2 for +20-30% training speed. "
            "Requires --fsdp, CUDA, and backend=transformers."
        ),
    )
    parallelism: Literal["data", "pipeline"] = Field(
        default="data",
        description=(
            "Distributed strategy: 'data' (DDP/FSDP/DeepSpeed) or 'pipeline' "
            "(pipeline parallel, v0.27.0 wiring only)."
        ),
    )
    pipeline_stages: int = Field(
        default=1, ge=1, le=16,
        description=(
            "Number of pipeline parallel stages. Ignored when "
            "parallelism='data'."
        ),
    )

    @model_validator(mode="after")
    def _validate_verifiable_reward(self) -> "TrainingConfig":
        """RLVR: reward_fn='verifiable' requires verifiable_domain."""
        if self.reward_fn == "verifiable" and self.verifiable_domain is None:
            raise ValueError(
                "reward_fn='verifiable' requires verifiable_domain "
                "(one of: math, code, json_schema)"
            )
        return self

    @model_validator(mode="after")
    def _validate_cross_doc_attn_mask(self) -> "TrainingConfig":
        """Cross-document attention masking requires packing=True."""
        if self.packing_cross_doc_attn_mask and not self.packing:
            raise ValueError(
                "packing_cross_doc_attn_mask requires packing=true "
                "(cross-doc attention masking only applies to packed sequences)"
            )
        return self

    @model_validator(mode="after")
    def _validate_multipack_packing_exclusive(self) -> "TrainingConfig":
        """Multipack and packing are mutually exclusive — pick one (v0.37.0).

        Both rewrite the batch composition; running them together produces
        ill-defined sample boundaries. Plan: long term, multipack subsumes
        packing — but for v0.37.0 we keep them as separate opt-ins.
        """
        if self.multipack and self.packing:
            raise ValueError(
                "multipack and packing are mutually exclusive — "
                "pick one (multipack uses FFD bin-packing; packing "
                "uses TRL's basic packer). For most uses, multipack=true "
                "is the better choice."
            )
        return self

    @model_validator(mode="after")
    def _validate_spike_recovery_requires_watchdog(self) -> "TrainingConfig":
        """Spike recovery is a watchdog hook — it needs the watchdog enabled."""
        if self.loss_spike_recovery and not self.loss_watchdog:
            raise ValueError(
                "loss_spike_recovery requires loss_watchdog=true "
                "(spike recovery is triggered by the watchdog)"
            )
        return self

    @model_validator(mode="after")
    def _validate_prequantized_no_qat(self) -> "TrainingConfig":
        """v0.38.0 — Pre-quantized formats + QAT is incompatible.

        GPTQ / AWQ / HQQ / AQLM / EETQ / MXFP4 / FP8 checkpoints all carry
        their own scale; routing them through torchao QAT or float8 prepare
        would corrupt the dequantized weights. Mirrors LlamaFactory's similar
        guard at quantization.py:117 / :199 / :211.
        """
        from soup_cli.utils.quant_menu import is_quant_menu_format

        if is_quant_menu_format(self.quantization) and self.quantization_aware:
            raise ValueError(
                f"quantization={self.quantization!r} is incompatible with "
                f"quantization_aware ({self.quantization_aware!r}). "
                "Pre-quantized checkpoints carry their own scale; "
                "QAT/FP8 prepare cannot compose. Set quantization_aware: false."
            )
        return self

    @model_validator(mode="after")
    def _validate_bnb_quant_storage_only_with_4bit(self) -> "TrainingConfig":
        """v0.38.0 Part G — bnb_4bit_quant_storage applies only to BNB 4-bit
        and MXFP4 (which is a BNB 4-bit variant). Setting it on any other
        format is a silent no-op — fail fast.
        """
        if self.bnb_4bit_quant_storage is None:
            return self
        if self.quantization not in ("4bit", "mxfp4"):
            raise ValueError(
                f"bnb_4bit_quant_storage={self.bnb_4bit_quant_storage!r} "
                f"requires quantization in {{'4bit', 'mxfp4'}}, got "
                f"{self.quantization!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_fp8_recipe_requires_fp8(self) -> "TrainingConfig":
        """fp8_recipe is only meaningful when quantization_aware='fp8'."""
        if self.fp8_recipe != "tensorwise" and self.quantization_aware != "fp8":
            raise ValueError(
                f"fp8_recipe='{self.fp8_recipe}' requires quantization_aware='fp8'. "
                "Either set quantization_aware: 'fp8' or remove the fp8_recipe field."
            )
        return self


class EvalConfig(BaseModel):
    """Evaluation configuration for auto-eval after training."""

    auto_eval: bool = Field(
        default=False,
        description="Run evaluation automatically after training completes",
    )
    benchmarks: Optional[List[str]] = Field(
        default=None,
        description="lm-evaluation-harness benchmark names to run",
    )
    custom_tasks: Optional[str] = Field(
        default=None,
        description="Path to custom eval JSONL file",
    )
    judge: Optional[dict] = Field(
        default=None,
        description="LLM-as-a-judge config: model, rubric, provider",
    )


class SoupConfig(BaseModel):
    """Root config for soup.yaml."""

    base: str = Field(..., description="Base model name or path (HF model ID)")
    task: Literal[
        "sft", "dpo", "grpo", "ppo", "reward_model", "kto", "orpo", "simpo", "ipo",
        "pretrain", "embedding",
    ] = Field(
        default="sft", description="Training task type"
    )
    modality: Literal["text", "vision", "audio"] = Field(
        default="text",
        description="Training modality: text (default), vision (multimodal), or audio",
    )
    backend: Literal["transformers", "unsloth", "mlx"] = Field(
        default="transformers",
        description=(
            "Training backend: transformers (default), unsloth (2-5x faster on "
            "CUDA), or mlx (Apple Silicon M1-M4)"
        ),
    )
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: str = Field(default="./output", description="Output directory for trained model")
    experiment_name: Optional[str] = Field(default=None, description="Experiment name for tracking")
    eval: Optional[EvalConfig] = Field(
        default=None,
        description="Evaluation configuration for auto-eval after training",
    )

    @field_validator("experiment_name")
    @classmethod
    def experiment_name_safe(cls, value: Optional[str]) -> Optional[str]:
        """Disallow path separators and null bytes in experiment_name."""
        if value is None:
            return value
        if re.search(r'[/\\:\x00]', value):
            raise ValueError(
                "experiment_name must not contain path separators (/ \\ :) or null bytes"
            )
        return value

    @model_validator(mode="after")
    def _validate_v028_speed_memory_supported_tasks(self) -> "SoupConfig":
        """v0.28.0 speed/memory features: every transformer-backend trainer
        is wired in v0.35.0 (#60). MLX backend trainers are still
        unsupported. Emit a precise ValueError that names the actual reason
        (MLX backend vs unknown task) so users get the right fix.
        """
        from soup_cli.utils.v028_features import supports_v028_features

        if supports_v028_features(self.task) and self.backend != "mlx":
            return self
        tcfg = self.training
        offenders: list[str] = []
        if tcfg.use_cut_ce:
            offenders.append("use_cut_ce")
        if tcfg.quantization_aware == "fp8":
            offenders.append('quantization_aware="fp8"')
        if tcfg.activation_offloading is not None:
            offenders.append("activation_offloading")
        if tcfg.kernel_auto_compose:
            offenders.append("kernel_auto_compose")
        if not offenders:
            return self
        # Distinct reasons get distinct messages so users don't waste time
        # blaming MLX when their task is the actual offender.
        if self.backend == "mlx":
            raise ValueError(
                f"v0.28.0 features {offenders} are not supported on the "
                f"Apple Silicon mlx backend (no equivalent kernels). "
                "Switch to backend='transformers' or remove these flags."
            )
        raise ValueError(
            f"v0.28.0 features {offenders} are not wired for "
            f"task={self.task!r}. Supported tasks: see "
            "soup_cli.utils.v028_features.supports_v028_features."
        )

    @model_validator(mode="after")
    def _validate_multipack_supported_tasks(self) -> "SoupConfig":
        """v0.37.0 — multipack only ships for sft / pretrain on transformers.

        Multipack rewrites the DataLoader sampler; preference / RLHF tasks
        in v0.37.0 still use the per-pair sampler shape from TRL. MLX
        backend has its own DataLoader path and is not wired.
        """
        if not self.training.multipack:
            return self
        from soup_cli.utils.multipack import supports_multipack

        if self.backend == "mlx":
            raise ValueError(
                "multipack=true is not supported on the mlx backend "
                "in v0.37.0 (sampler injection is HF Trainer-specific). "
                "Use backend='transformers' or set multipack: false."
            )
        if not supports_multipack(self.task):
            raise ValueError(
                f"multipack=true is not supported for task={self.task!r} "
                "in v0.37.0 (only sft and pretrain are wired). "
                "Set multipack: false or switch task."
            )
        return self

    @model_validator(mode="after")
    def _validate_quant_menu_supported_tasks(self) -> "SoupConfig":
        """v0.38.0 — Quant Menu (gptq/awq/hqq:Nbit/aqlm/eetq/mxfp4/fp8) is wired
        in the SFT trainer only. Multi-trainer expansion deferred to v0.38.1
        (mirrors v0.27.0 MII / v0.37.0 multipack stub-then-live pattern).
        """
        from soup_cli.utils.quant_menu import is_quant_menu_format

        quant = self.training.quantization
        # bnb 4bit / 8bit / none always apply universally — pre-existing.
        if not is_quant_menu_format(quant):
            return self
        if self.backend == "mlx":
            raise ValueError(
                f"quantization={quant!r} is not supported on the mlx backend "
                "(no equivalent kernels). Use backend='transformers' or "
                "switch to quantization in {'4bit', '8bit', 'none'}."
            )
        if self.task != "sft":
            raise ValueError(
                f"quantization={quant!r} (Quant Menu) is wired for task='sft' "
                f"only in v0.38.0; got task={self.task!r}. Multi-trainer "
                "wiring is tracked for v0.38.1."
            )
        if self.modality != "text":
            raise ValueError(
                f"quantization={quant!r} (Quant Menu) is wired for "
                f"modality='text' only in v0.38.0; got modality={self.modality!r}. "
                "Vision/audio multi-modal wiring is tracked for v0.38.1."
            )
        return self

    @model_validator(mode="after")
    def _validate_mlx_task_support(self) -> "SoupConfig":
        """MLX backend only supports sft, dpo, and grpo tasks (v0.25.0).

        DPO and GRPO wrappers are scaffolding in v0.25.0 — they raise
        NotImplementedError at ``train()`` time because upstream mlx-lm has
        not yet shipped DPO/GRPO training helpers. Users who pick them will
        instead see this friendly error at config-load time.
        """
        if self.backend != "mlx":
            return self
        if self.task == "sft":
            return self
        raise ValueError(
            f"MLX backend only ships SFT in v0.25.0; task='{self.task}' "
            "is not yet implemented (upstream mlx-lm does not expose a "
            f"training helper). Use backend=transformers for task={self.task}."
        )


# --- Built-in templates ---

TEMPLATES: dict[str, str] = {
    "chat": """# Soup template: Chat Assistant
# Fine-tune a model for conversational chat

base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.1
  max_length: 2048

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    "code": """# Soup template: Code Model
# Fine-tune a model for code generation / completion

base: codellama/CodeLlama-7b-Instruct-hf
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/code_train.jsonl
  format: alpaca
  val_split: 0.1
  max_length: 4096

training:
  epochs: 2
  lr: 1e-5
  batch_size: auto
  lora:
    r: 128
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    "reasoning": """# Soup template: Reasoning / GRPO
# Fine-tune a model for chain-of-thought reasoning with GRPO

base: meta-llama/Llama-3.1-8B-Instruct
task: grpo
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/reasoning_train.jsonl
  format: sharegpt
  val_split: 0.1
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    "vision": """# Soup template: Vision / Multimodal
# Fine-tune a vision-language model for image understanding

base: meta-llama/Llama-3.2-11B-Vision-Instruct
task: sft
modality: vision
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/vision_train.jsonl
  format: llava
  image_dir: ./data/images
  val_split: 0.1
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    "medical": """# Soup template: Medical / Domain Expert
# Fine-tune a model with domain-specific knowledge

base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/medical_train.jsonl
  format: alpaca
  val_split: 0.15
  max_length: 2048

training:
  epochs: 5
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 128
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    "kto": """# Soup template: KTO (Kahneman-Tversky Optimization)
# Align a model using unpaired preference data (no need for chosen+rejected pairs)
#
# Data format (JSONL):
#   {"prompt": "What is 2+2?", "completion": "4", "label": true}
#   {"prompt": "What is 2+2?", "completion": "Fish", "label": false}

base: meta-llama/Llama-3.1-8B-Instruct
task: kto
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/kto_train.jsonl
  format: kto
  val_split: 0.1
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 4
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit
  kto_beta: 0.1

output: ./output
""",
    "orpo": """# Soup template: ORPO (Odds Ratio Preference Optimization)
# Align a model without a reference model — simpler than DPO
#
# Data format (JSONL):
#   {"prompt": "What is 2+2?", "chosen": "4", "rejected": "Fish"}

base: meta-llama/Llama-3.1-8B-Instruct
task: orpo
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/preference_train.jsonl
  format: dpo
  val_split: 0.1
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 4
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit
  orpo_beta: 0.1

output: ./output
""",
    "simpo": """# Soup template: SimPO (Simple Preference Optimization)
# Reference-free preference alignment with length-normalized rewards
#
# Data format (JSONL):
#   {"prompt": "What is 2+2?", "chosen": "4", "rejected": "Fish"}

base: meta-llama/Llama-3.1-8B-Instruct
task: simpo
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/preference_train.jsonl
  format: dpo
  val_split: 0.1
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 4
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit
  simpo_gamma: 0.5
  cpo_alpha: 1.0

output: ./output
""",
    "ipo": """# Soup template: IPO (Identity Preference Optimization)
# A theoretically grounded variant of DPO with stronger regularization
#
# Data format (JSONL):
#   {"prompt": "What is 2+2?", "chosen": "4", "rejected": "Fish"}

base: meta-llama/Llama-3.1-8B-Instruct
task: ipo
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/preference_train.jsonl
  format: dpo
  val_split: 0.1
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 4
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit
  ipo_tau: 0.1

output: ./output
""",
    "pretrain": """# Soup template: Continued Pre-training
# Continue pre-training a model on raw text data (domain adaptation)
#
# Data format (JSONL):
#   {"text": "Your raw text document here..."}
#
# Or plain .txt files (one document per line or entire file as one document).

base: meta-llama/Llama-3.1-8B
task: pretrain
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/corpus.jsonl
  format: plaintext
  val_split: 0.05
  max_length: 4096

training:
  epochs: 1
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit

output: ./output_pretrain
""",
    "moe": """# Soup template: MoE (Mixture of Experts) Fine-tuning
# Fine-tune a Mixture of Experts model with ScatterMoE LoRA
#
# Supported MoE models: Qwen3-30B-A3B, Mixtral-8x7B, DeepSeek-V3, etc.

base: Qwen/Qwen3-30B-A3B
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.1
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit
  moe_lora: true
  moe_aux_loss_coeff: 0.01

output: ./output
""",
    "longcontext": """# Soup template: Long-Context Fine-tuning (128k+)
# Extend model context window for long-document understanding
#
# Uses RoPE scaling + gradient checkpointing + FlashAttention for 128k tokens.
# Optionally enable Liger Kernel for additional memory savings.

base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/long_context_train.jsonl
  format: alpaca
  val_split: 0.05
  max_length: 131072

training:
  epochs: 1
  lr: 5e-6
  batch_size: 1
  gradient_accumulation_steps: 16
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit
  gradient_checkpointing: true
  rope_scaling_type: dynamic
  use_flash_attn: true
  # use_liger: true       # pip install 'soup-cli[liger]' for fused ops
  # use_ring_attention: true  # Multi-GPU sequence parallelism

output: ./output_longctx
""",
    "embedding": """# Soup template: Embedding Model Fine-tuning
# Fine-tune a sentence embedding model (BGE, E5, GTE, etc.)
#
# Data format (JSONL) — contrastive pairs:
#   {"anchor": "What is Python?", "positive": "Python is a programming language."}
#
# Data format (JSONL) — triplets:
#   {"anchor": "query", "positive": "relevant doc", "negative": "unrelated doc"}

base: BAAI/bge-base-en-v1.5
task: embedding
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/embedding_train.jsonl
  format: embedding
  val_split: 0.1
  max_length: 512

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  gradient_accumulation_steps: 4
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: none
  embedding_loss: contrastive
  embedding_margin: 0.5
  embedding_pooling: mean

output: ./output_embedding
""",
    "audio": """# Soup template: Audio / Speech
# Fine-tune an audio-language model for speech understanding
#
# Supported models: Qwen2-Audio, Whisper (via transformers)
#
# Data format (JSONL):
#   {"audio": "path/to/audio.wav", "messages": [
#     {"role": "user", "content": "Transcribe."},
#     {"role": "assistant", "content": "Hello world."}]}

base: Qwen/Qwen2-Audio-7B-Instruct
task: sft
modality: audio
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/audio_train.jsonl
  format: audio
  audio_dir: ./data/audio
  val_split: 0.1
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit

output: ./output_audio
""",
    "tool-calling": """# Soup template: Tool-Calling / Agentic Fine-tuning
# Fine-tune a model to call tools / functions correctly
#
# Data format (JSONL):
#   {
#     "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
#     "tools": [{"type": "function", "function": {
#       "name": "get_weather",
#       "description": "Get current weather for a city",
#       "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
#     }}],
#     "tool_calls": [{"function": {
#       "name": "get_weather",
#       "arguments": "{\\"city\\": \\"Tokyo\\"}"
#     }}]
#   }

base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/tool_calling_train.jsonl
  format: tool-calling
  val_split: 0.1
  max_length: 4096

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  gradient_accumulation_steps: 4
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    "rlhf": """# Soup template: Full RLHF Pipeline (SFT + Reward Model + PPO)
# Three-stage training: 1) SFT warmup, 2) Reward model, 3) PPO alignment
#
# Usage:
#   Step 1: soup train --config soup_sft.yaml       # SFT warmup
#   Step 2: soup train --config soup_rm.yaml         # Train reward model
#   Step 3: soup train --config soup_ppo.yaml        # PPO with reward model
#
# This template generates the PPO config (step 3).
# For steps 1-2, use: soup init --template chat (SFT) and edit task to reward_model.

base: meta-llama/Llama-3.1-8B-Instruct
task: ppo
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/prompts.jsonl
  format: chatml
  val_split: 0.1
  max_length: 2048

training:
  epochs: 1
  lr: 1e-6
  batch_size: auto
  gradient_accumulation_steps: 4
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit
  reward_model: ./output_rm
  ppo_epochs: 4
  ppo_clip_ratio: 0.2
  ppo_kl_penalty: 0.05

output: ./output_ppo
""",
}
