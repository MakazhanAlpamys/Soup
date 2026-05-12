"""Pydantic schemas for soup.yaml config — single source of truth."""

import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# v0.39.0 Part C — per-pattern LoRA rank/alpha bounds
_MAX_LORA_RANK_PATTERN_KEYS = 256
_MAX_LORA_RANK_PATTERN_VALUE = 1024


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
            "Mutually exclusive with use_dora and use_vera. "
            "Equivalent to init_strategy='olora'."
        ),
    )
    rank_pattern: Optional[Dict[str, int]] = Field(
        default=None,
        description=(
            "Per-target-module-pattern LoRA rank override. Maps module name "
            "patterns (e.g. 'q_proj', 'experts.*.w1') to integer rank values. "
            "Useful for MoE configs where expert FFNs need lower rank than attn. "
            "Incompatible with use_vera (VeRA shares one rank across modules)."
        ),
    )
    alpha_pattern: Optional[Dict[str, int]] = Field(
        default=None,
        description=(
            "Per-target-module-pattern LoRA alpha override. Maps module name "
            "patterns to integer alpha values. Pairs with rank_pattern. "
            "Incompatible with use_vera."
        ),
    )
    init_strategy: Literal["random", "pissa", "olora", "loftq"] = Field(
        default="random",
        description=(
            "LoRA init strategy. 'random' (default) is standard Kaiming init. "
            "'pissa' (PiSSA) initializes A/B from the SVD of the base weight — "
            "faster early convergence but adds an SVD pass on the first epoch. "
            "'olora' is equivalent to use_olora=True (orthogonal QR init). "
            "'loftq' (v0.41.0) initialises A/B + a low-bit base together, "
            "useful with QLoRA. Cannot be combined with use_dora or use_vera."
        ),
    )
    # v0.41.0 Part C — LoftQ tuning knobs (used only when init_strategy='loftq').
    loftq_iter: int = Field(
        default=1, ge=1, le=10,
        description=(
            "LoftQ iteration count (1-10). Higher = better quant-aware init "
            "at the cost of one-time setup latency. Used only when "
            "init_strategy='loftq'."
        ),
    )
    loftq_bits: Literal[2, 4, 8] = Field(
        default=4,
        description=(
            "LoftQ target bitwidth — must be one of {2, 4, 8}. Used only "
            "when init_strategy='loftq'."
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

    @model_validator(mode="before")
    @classmethod
    def _backcompat_align_olora(cls, values):
        """Back-compat: pre-validation, align init_strategy='olora' when only use_olora was set."""
        if not isinstance(values, dict):
            return values
        # Copy to avoid mutating the caller's dict (matches v0.33.0 #47
        # CrossDocCollator immutability fix).
        if values.get("use_olora") and "init_strategy" not in values:
            values = dict(values)
            values["init_strategy"] = "olora"
        return values

    @model_validator(mode="after")
    def _validate_init_strategy(self) -> "LoraConfig":
        # use_olora=True must agree with init_strategy when both are explicit
        if self.use_olora and self.init_strategy != "olora":
            raise ValueError(
                f"use_olora=True conflicts with init_strategy={self.init_strategy!r}. "
                f"Either set init_strategy='olora' (or omit it), or set use_olora=False."
            )
        # init_strategy='pissa' is incompatible with DoRA / VeRA
        if self.init_strategy == "pissa" and (self.use_dora or self.use_vera):
            other = "use_dora" if self.use_dora else "use_vera"
            raise ValueError(
                f"init_strategy='pissa' is incompatible with {other}=True. "
                f"PiSSA initializes the LoRA pair via SVD; combine with plain LoRA "
                f"(or rsLoRA) only."
            )
        # v0.41.0 Part C — init_strategy='loftq' is incompatible with DoRA / VeRA
        if self.init_strategy == "loftq" and (self.use_dora or self.use_vera):
            other = "use_dora" if self.use_dora else "use_vera"
            raise ValueError(
                f"init_strategy='loftq' is incompatible with {other}=True. "
                f"LoftQ jointly initialises A/B with quantised base weights; "
                f"combine with plain LoRA only."
            )
        return self

    @field_validator("rank_pattern", "alpha_pattern", mode="before")
    @classmethod
    def _validate_pattern_dict(cls, value) -> Optional[Dict[str, int]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("rank_pattern/alpha_pattern must be a dict[str, int]")
        if len(value) > _MAX_LORA_RANK_PATTERN_KEYS:
            raise ValueError(
                f"rank_pattern/alpha_pattern caps at {_MAX_LORA_RANK_PATTERN_KEYS} keys, "
                f"got {len(value)}"
            )
        cleaned: Dict[str, int] = {}
        for key, val in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "rank_pattern/alpha_pattern keys must be non-empty strings"
                )
            if "\x00" in key:
                raise ValueError("rank_pattern/alpha_pattern keys cannot contain null bytes")
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(
                    f"rank_pattern/alpha_pattern values must be int, "
                    f"got {type(val).__name__} for {key!r}"
                )
            if val <= 0 or val > _MAX_LORA_RANK_PATTERN_VALUE:
                raise ValueError(
                    f"rank_pattern/alpha_pattern values must be in (0, "
                    f"{_MAX_LORA_RANK_PATTERN_VALUE}], got {val} for {key!r}"
                )
            cleaned[key] = val
        return cleaned

    @model_validator(mode="after")
    def _validate_pattern_vera_exclusivity(self) -> "LoraConfig":
        if self.use_vera and self.rank_pattern:
            raise ValueError(
                "rank_pattern is incompatible with use_vera=True (VeRA shares "
                "a single rank across all target modules). Disable use_vera or "
                "remove rank_pattern."
            )
        if self.use_vera and self.alpha_pattern:
            raise ValueError(
                "alpha_pattern is incompatible with use_vera=True. Disable "
                "use_vera or remove alpha_pattern."
            )
        return self


class DataConfig(BaseModel):
    train: str = Field(..., description="Path to training data or HF dataset name")
    format: Literal[
        "alpaca", "sharegpt", "chatml", "dpo", "kto", "llava", "sharegpt4v",
        "plaintext", "embedding", "audio", "tool-calling", "auto",
        # v0.42.0 — Data Pipeline Pro
        "prm", "pre_tokenized", "input_output", "video", "multimodal",
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

    # --- v0.42.0 Data Pipeline Pro -----------------------------------------
    video_dir: Optional[str] = Field(
        default=None,
        description=(
            "Base directory for resolving relative video paths in video "
            "datasets. Mirrors image_dir / audio_dir. (v0.42.0 Part A)"
        ),
    )
    tokenized_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to a pre-tokenized cache produced by `soup data preprocess`. "
            "When set, the trainer skips the tokenize stage and reads tensors "
            "directly. Mirrors LF tokenized_path / Axolotl `empty` type. "
            "(v0.42.0 Part C)"
        ),
    )
    streaming: bool = Field(
        default=False,
        description=(
            "Pass-through to HF datasets `streaming=True`. Use for datasets "
            "that don't fit on disk. Pairs with `buffer_size`. (v0.42.0 Part B)"
        ),
    )
    buffer_size: Optional[int] = Field(
        default=None,
        description=(
            "Shuffle buffer size for streaming datasets. None = HF default. "
            "Bounds [1, 1_000_000]. (v0.42.0 Part B)"
        ),
    )
    shards: Optional[int] = Field(
        default=None,
        description=(
            "Number of shards for HF dataset splits (axolotl `shards`). "
            "Bounds [1, 1024]. (v0.42.0 Part B)"
        ),
    )
    interleave: Optional[Union[str, Dict]] = Field(
        default=None,
        description=(
            "Multi-dataset interleave strategy: 'concat' / 'under' / 'over' / "
            "{strategy: 'probs', probs: [...]}. (v0.42.0 Part D)"
        ),
    )
    mask_history: bool = Field(
        default=False,
        description=(
            "LF mask_history — mask all but the last assistant turn during "
            "loss computation. (v0.42.0 Part D)"
        ),
    )
    train_on_prompt: bool = Field(
        default=False,
        description=(
            "LF train_on_prompt — include the prompt tokens in the loss. "
            "Inverse of train_on_responses_only. (v0.42.0 Part D)"
        ),
    )
    eval_on_each_dataset: bool = Field(
        default=False,
        description=(
            "LF eval_on_each_dataset — when interleaving, run eval on every "
            "constituent dataset separately. (v0.42.0 Part D)"
        ),
    )
    split_thinking: bool = Field(
        default=False,
        description=(
            "Axolotl split_thinking — separate `<think>` reasoning blocks "
            "from the final answer for fine-grained masking. Qwen3-style. "
            "(v0.42.0 Part D)"
        ),
    )
    image_min_pixels: Optional[int] = Field(
        default=None,
        description="Per-image min pixel count for vision data. (v0.42.0 Part D)",
    )
    image_max_pixels: Optional[int] = Field(
        default=None,
        description="Per-image max pixel count for vision data. (v0.42.0 Part D)",
    )
    image_resize_algorithm: Optional[
        Literal["nearest", "bilinear", "bicubic", "lanczos"]
    ] = Field(
        default=None,
        description="Pillow resize algorithm for image preprocessing. (v0.42.0 Part D)",
    )
    video_fps: Optional[float] = Field(
        default=None,
        description=(
            "Target frames-per-second for video preprocessing. (v0.42.0 Part D)"
        ),
    )
    video_maxlen: Optional[int] = Field(
        default=None,
        description=(
            "Max number of frames per video clip. Bounds (0, 4096]. "
            "(v0.42.0 Part D)"
        ),
    )
    add_new_tokens: Optional[List[str]] = Field(
        default=None,
        description=(
            "Add these tokens to the tokenizer vocab + resize embeddings. "
            "Cap 10_000 entries; per-token <= 256 chars; no duplicates. "
            "(v0.42.0 Part E)"
        ),
    )
    new_special_tokens: Optional[List[str]] = Field(
        default=None,
        description=(
            "Like add_new_tokens but registered as additional_special_tokens "
            "so they are not split by the tokenizer. (v0.42.0 Part E)"
        ),
    )
    resize_vocab: bool = Field(
        default=False,
        description=(
            "Resize the model's input/output embedding matrix when "
            "add_new_tokens / new_special_tokens grew the vocab. "
            "(v0.42.0 Part E)"
        ),
    )
    extend_conversation: bool = Field(
        default=False,
        description=(
            "Unsloth-style conversation extension — extend the last assistant "
            "turn with N more tokens for 'continue' prompts. (v0.42.0 Part E)"
        ),
    )
    skip_prepare_dataset: bool = Field(
        default=False,
        description=(
            "Axolotl skip_prepare_dataset — escape hatch when the input is "
            "already in the trainer's expected schema. (v0.42.0 Part E)"
        ),
    )
    remove_unused_columns: bool = Field(
        default=True,
        description=(
            "HF Trainer remove_unused_columns. Set False when feeding "
            "extra cols to a custom collator. (v0.42.0 Part E)"
        ),
    )
    prompt_strategy: Optional[str] = Field(
        default=None,
        description=(
            "Axolotl-style 'module.path:function_name' Python transform. "
            "Schema-only in v0.42.0 — runtime invocation lands in v0.42.1. "
            "(v0.42.0 Part E)"
        ),
    )

    @field_validator("video_dir", "tokenized_path")
    @classmethod
    def _validate_v042_optional_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("path must be a string")
        if not value:
            return None
        if "\x00" in value:
            raise ValueError("path must not contain null bytes")
        if len(value) > 4096:
            raise ValueError("path must be <= 4096 chars")
        # Schema-level containment via shared `is_under_cwd` (os.path.realpath
        # + commonpath). Rejects arbitrary system paths at config load so a
        # crafted soup.yaml fails fast instead of at first filesystem read.
        from soup_cli.utils.paths import is_under_cwd

        if not is_under_cwd(value):
            raise ValueError(
                "path must stay under the current working directory "
                "(absolute paths outside cwd are rejected at config load)."
            )
        return value

    @field_validator("buffer_size")
    @classmethod
    def _validate_buffer_size_v042(cls, value: Optional[int]) -> Optional[int]:
        from soup_cli.utils.data_pipeline import validate_buffer_size

        return validate_buffer_size(value)

    @field_validator("shards")
    @classmethod
    def _validate_shards_v042(cls, value: Optional[int]) -> Optional[int]:
        from soup_cli.utils.data_pipeline import validate_shards

        return validate_shards(value)

    @field_validator("image_min_pixels", "image_max_pixels")
    @classmethod
    def _validate_image_pixels_v042(cls, value, info):
        from soup_cli.utils.data_pipeline import validate_image_pixels

        return validate_image_pixels(info.field_name, value)

    @field_validator("video_fps")
    @classmethod
    def _validate_video_fps_v042(cls, value: Optional[float]) -> Optional[float]:
        from soup_cli.utils.data_pipeline import validate_video_fps

        return validate_video_fps(value)

    @field_validator("video_maxlen")
    @classmethod
    def _validate_video_maxlen_v042(cls, value: Optional[int]) -> Optional[int]:
        from soup_cli.utils.data_pipeline import validate_video_maxlen

        return validate_video_maxlen(value)

    @field_validator("add_new_tokens", "new_special_tokens")
    @classmethod
    def _validate_new_tokens_v042(
        cls, value: Optional[List[str]]
    ) -> Optional[List[str]]:
        from soup_cli.utils.data_pipeline import validate_new_tokens

        return validate_new_tokens(value)

    @field_validator("prompt_strategy")
    @classmethod
    def _validate_prompt_strategy_v042(cls, value: Optional[str]) -> Optional[str]:
        from soup_cli.utils.data_pipeline import validate_prompt_strategy

        return validate_prompt_strategy(value)

    @field_validator("interleave")
    @classmethod
    def _validate_interleave_v042(cls, value):
        # Shape validation only — full ``parse_interleave`` requires
        # ``num_datasets`` which the trainer supplies at runtime. We accept
        # None / str / dict here and reject obvious type errors so a YAML
        # like ``data.interleave: 99`` fails loudly at config load.
        if value is None:
            return None
        if isinstance(value, str):
            from soup_cli.utils.data_pipeline import INTERLEAVE_STRATEGIES

            if value not in INTERLEAVE_STRATEGIES:
                raise ValueError(
                    f"interleave must be one of {sorted(INTERLEAVE_STRATEGIES)} "
                    f"or a dict — got {value!r}"
                )
            if value == "probs":
                # Probs requires the dict form so the per-dataset weights are
                # supplied — bare "probs" is meaningless.
                raise ValueError(
                    "interleave='probs' requires a 'probs' list — use "
                    "{strategy: probs, probs: [...]} dict form."
                )
            return value
        if isinstance(value, dict):
            if "strategy" not in value:
                raise ValueError(
                    "interleave dict form must include 'strategy' key"
                )
            return value
        raise ValueError(
            f"interleave must be None, a string, or a dict (got "
            f"{type(value).__name__})"
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

    @model_validator(mode="after")
    def _validate_v042_train_on_prompt(self) -> "DataConfig":
        # train_on_prompt is the inverse semantics of train_on_responses_only —
        # both True is contradictory. Match v0.36.0 loss-mask exclusivity policy.
        if self.train_on_prompt and self.train_on_responses_only:
            raise ValueError(
                "train_on_prompt and train_on_responses_only are mutually "
                "exclusive — train_on_prompt opts INTO prompt-token loss, "
                "train_on_responses_only opts OUT. Pick one."
            )
        return self

    @model_validator(mode="after")
    def _validate_v042_image_pixel_range(self) -> "DataConfig":
        if (
            self.image_min_pixels is not None
            and self.image_max_pixels is not None
            and self.image_min_pixels > self.image_max_pixels
        ):
            raise ValueError(
                "image_min_pixels must be <= image_max_pixels"
            )
        return self

    @model_validator(mode="after")
    def _validate_v042_streaming_buffer(self) -> "DataConfig":
        # buffer_size only meaningful when streaming=True — surface the
        # mismatch loudly (mirrors v0.32.0 spike-recovery / loss-watchdog
        # cross-validator policy).
        if self.buffer_size is not None and not self.streaming:
            raise ValueError(
                "buffer_size requires streaming=True (HF datasets only "
                "supports a shuffle buffer in streaming mode)."
            )
        return self

    @model_validator(mode="after")
    def _validate_v042_video_fields(self) -> "DataConfig":
        # Video-only fields must not be set when format != 'video' to avoid
        # silent no-ops (Axolotl-mode footgun this validator prevents).
        video_fields = (
            ("video_fps", self.video_fps),
            ("video_maxlen", self.video_maxlen),
            ("video_dir", self.video_dir),
        )
        any_set = any(v is not None for _, v in video_fields)
        if any_set and self.format not in ("video", "multimodal", "auto"):
            names = [n for n, v in video_fields if v is not None]
            raise ValueError(
                f"video-related fields {names} require format in "
                "{video, multimodal, auto} (got "
                f"{self.format!r})."
            )
        return self

    @model_validator(mode="after")
    def _validate_v042_resize_vocab_requires_tokens(self) -> "DataConfig":
        if self.resize_vocab and not (
            self.add_new_tokens or self.new_special_tokens
        ):
            raise ValueError(
                "resize_vocab=True requires add_new_tokens or "
                "new_special_tokens to be non-empty — otherwise the resize is "
                "a no-op."
            )
        return self

    @model_validator(mode="after")
    def _validate_v042_pre_tokenized_path(self) -> "DataConfig":
        # tokenized_path is meaningful regardless of format (Axolotl `empty`
        # type expects the cache to be the source of truth). But the
        # pre_tokenized format implies the path must be set.
        if self.format == "pre_tokenized" and not self.tokenized_path:
            raise ValueError(
                "format='pre_tokenized' requires data.tokenized_path to point "
                "at a cache directory produced by `soup data preprocess`."
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
        # v0.52.0 Part D — BitNet 1.58-bit (axolotl + onebitllms).
        "bitnet_1.58",
    ] = Field(
        default="4bit",
        description=(
            "Quantization (v0.38.0 — Quant Menu): "
            "4bit (BNB QLoRA), 8bit (BNB), none, "
            "gptq / awq (load pre-quantized checkpoint, train LoRA on top), "
            "hqq:Nbit (HQQ 1-8 bit, N in {1..6, 8}), "
            "aqlm (extreme 2-bit), eetq (8-bit fast), "
            "mxfp4 (BNB 4-bit MXFP4 quant_type), "
            "fp8 (load FP8 checkpoint with dequantize-on-load), "
            "bitnet_1.58 (BitNet ternary, v0.52.0 schema-only)."
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
    optimizer: str = Field(
        default="adamw_torch",
        description=(
            "Optimizer name. v0.41.0 expands the allowlist to cover BAdam, "
            "APOLLO, Adam-mini, lomo/adalomo, grokadamw, schedule_free, "
            "muon/dion/came_pytorch, and TorchAO ao_adamw_{fp8,4bit,8bit}. "
            "See soup_cli.utils.optimizer_zoo.SUPPORTED_OPTIMIZERS for the "
            "full list."
        ),
    )
    # v0.41.0 Part B — per-module-pattern LR override.
    lr_groups: Optional[List[Dict[str, Union[str, float]]]] = Field(
        default=None,
        description=(
            "Per-module LR override. List of {pattern, lr} entries (or a "
            "{pattern: lr} dict). First match wins; remaining params fall "
            "through to the base lr. Capped at 32 entries. (v0.41.0)"
        ),
    )
    # v0.41.0 Part C — LLaMA Pro block expansion.
    expand_layers: Optional[int] = Field(
        default=None, ge=1, le=64,
        description=(
            "LLaMA Pro: append N zero-init transformer blocks and freeze "
            "the original ones. Schema lands in v0.41.0 — full live wiring "
            "deferred to v0.41.1."
        ),
    )
    freeze_trainable_layers: Optional[int] = Field(
        default=None,
        description=(
            "LLaMA Pro: signed int. Positive = train only top-N decoder "
            "layers; negative = train only bottom-N. Magnitude capped at "
            "1000. (v0.41.0)"
        ),
    )
    # v0.41.0 Part C — Mixture-of-Depths (selective-token routing).
    use_mod: bool = Field(
        default=False,
        description=(
            "Enable Mixture-of-Depths routing patch. Schema only in v0.41.0; "
            "live patch deferred to v0.41.1 (mirrors v0.27.0 MII / v0.37.0 "
            "multipack stub-then-live pattern)."
        ),
    )
    # v0.41.0 Part C — Friendly aliases for `quantization` (LF / Axolotl users).
    load_in_8bit: Optional[bool] = Field(
        default=None,
        description=(
            "Friendly alias for quantization='8bit' / 'none'. When True, "
            "rewrites quantization to '8bit' if currently 'none'/'4bit'. "
            "Conflicts with load_in_16bit. (v0.41.0)"
        ),
    )
    load_in_16bit: Optional[bool] = Field(
        default=None,
        description=(
            "Friendly alias: when True, sets quantization='none' (full bf16/"
            "fp16 LoRA). Conflicts with load_in_8bit. (v0.41.0)"
        ),
    )
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
    # BCO-specific (Binary Classifier Optimization, v0.40.0 Part A)
    bco_beta: float = Field(
        default=0.1, gt=0, description="BCO beta — KL penalty coefficient"
    )
    # Unified preference loss dispatcher (v0.40.0 Part B).
    # Set when task='preference'. Legacy task strings ('dpo', 'simpo', ...)
    # remain first-class and are unaffected.
    preference_loss: Optional[Literal["dpo", "simpo", "orpo", "ipo", "bco"]] = Field(
        default=None,
        description=(
            "Preference loss for task='preference'. One of: dpo, simpo, orpo, "
            "ipo, bco. Mutually exclusive with task in {dpo, simpo, orpo, ipo, bco}."
        ),
    )
    # KL-controlled DPO variants (v0.40.0 Part C).
    dpo_beta_schedule: Optional[Literal["linear", "cosine", "exponential"]] = Field(
        default=None,
        description=(
            "Anneal DPO β over training. None = constant β (default). Requires "
            "dpo_beta_end. DPO-family tasks only (dpo, ipo, preference+dpo)."
        ),
    )
    dpo_beta_end: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Target β at the end of training when dpo_beta_schedule is set. "
            "Must be > 0. The starting β is dpo_beta."
        ),
    )
    dpo_ref_regen_epochs: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description=(
            "Replace the frozen ref model with the current student every N "
            "epochs. None = never regen (default). DPO-family tasks only."
        ),
    )
    # Multi-objective preference loss (v0.40.0 Part D).
    preference_loss_weights: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Weighted blend of preference losses, e.g. {'dpo': 0.7, 'bco': 0.3}. "
            "Each weight ∈ (0, 1]; weights must sum to 1.0 (±1e-6). All keys "
            "must be members of {dpo, simpo, orpo, ipo, bco}. Requires "
            "task='preference'; mutually exclusive with preference_loss (the "
            "scalar form). Capped at 5 components (the supported set)."
        ),
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
    # v0.50.0 Part A — GRPO objective variants (unsloth + axolotl parity).
    # Schema-only in v0.50.0; live loss kernels wired in v0.50.1.
    grpo_variant: Optional[Literal[
        "standard", "gspo", "dapo", "dr_grpo", "bnpo", "two_sided", "rft"
    ]] = Field(
        default=None,
        description=(
            "GRPO objective variant: standard | gspo | dapo | dr_grpo | "
            "bnpo | two_sided | rft. Defaults to None (legacy GRPO). "
            "Requires task='grpo'. Live wiring for v0.50.0 additions is "
            "deferred to v0.50.1 (schema gate only)."
        ),
    )
    grpo_delta: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Symmetric clipping radius for grpo_variant='two_sided'. "
            "Required when grpo_variant='two_sided'; rejected otherwise."
        ),
    )
    grpo_fp16: bool = Field(
        default=False,
        description=(
            "Force FP16 mixed precision for GRPO/RL. Soup currently has "
            "FP8 RL support; this flag is the explicit FP16 opt-in (unsloth "
            "parity). v0.50.0: schema-only; live mixed-precision routing "
            "deferred to v0.50.1."
        ),
    )
    # v0.50.0 Part B — Long-context + memory-efficient RL
    long_context_grpo: bool = Field(
        default=False,
        description=(
            "Enable long-context GRPO (unsloth: 380K B200 / 110K H100). "
            "Wires Tiled MLP from v0.56.0 Part A; schema-only in v0.50.0. "
            "Requires task='grpo' on a non-mlx backend and "
            "use_ring_attention=False (both rewrite attention)."
        ),
    )
    vllm_sleep_mode: bool = Field(
        default=False,
        description=(
            "Enable vLLM sleep/standby between rollouts (memory savings "
            "during the optimisation step). Requires backend in "
            "{transformers, unsloth}. Schema-only in v0.50.0; live wiring "
            "in v0.50.1."
        ),
    )
    # v0.50.0 Part C — Multi-turn agent rollout backend
    rollout_backend: Optional[Literal[
        "art", "ruler", "nemo_gym", "openenv"
    ]] = Field(
        default=None,
        description=(
            "Multi-turn agent rollout backend (unsloth / axolotl parity): "
            "art (OpenPipe ART) / ruler / nemo_gym / openenv. "
            "Requires task='grpo'. Schema-only in v0.50.0; live launcher "
            "wired in v0.50.1."
        ),
    )
    # v0.50.0 Part D — GRPO stability / efficiency knobs (axolotl + unsloth).
    # All schema-only in v0.50.0; live trainer callbacks wired in v0.50.1.
    ref_model_ema_alpha: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Exponential moving average coefficient for ref-model sync "
            "(policy → reference). Must be in (0, 1]. None = disabled. "
            "Axolotl parity."
        ),
    )
    replay_buffer_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=1_000_000,
        description=(
            "Bounded replay buffer size for GRPO rollouts. None = disabled. "
            "Axolotl parity."
        ),
    )
    async_grpo_prefetch: bool = Field(
        default=False,
        description=(
            "Overlap rollout + train via async prefetch (axolotl). "
            "Requires backend in {transformers, unsloth}."
        ),
    )
    tis_threshold: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=100.0,
        description=(
            "Truncated importance sampling threshold (unsloth, axolotl). "
            "Must be in (0, 100]. None = disabled."
        ),
    )
    mask_truncated_completions: bool = Field(
        default=False,
        description=(
            "Mask out truncated completions when computing the policy "
            "gradient (paired with tis_threshold). Unsloth + axolotl parity."
        ),
    )
    defer_rerolling: bool = Field(
        default=False,
        description=(
            "Defer re-rolling identical prompts across optimisation steps "
            "(axolotl). Saves rollouts on repeat prompts."
        ),
    )
    skip_zero_advantage: bool = Field(
        default=False,
        description=(
            "Skip backward pass on samples whose advantage is exactly zero "
            "(axolotl). Avoids wasted compute on no-signal samples."
        ),
    )
    off_policy_mask_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Off-policy mask threshold for token/sequence gating (axolotl). "
            "Must be in [0, 1]. None = disabled."
        ),
    )
    # v0.50.0 Part E — Vision-RL opt-in flag
    vision_grpo: bool = Field(
        default=False,
        description=(
            "Enable Vision RL / VLM RL — extends GRPO/PPO to vision "
            "modality (Qwen2-VL / Pixtral / InternVL). Requires "
            "modality='vision', task in {grpo, ppo}, backend in "
            "{transformers, unsloth}. Schema-only in v0.50.0; live VLM-RL "
            "rollout wiring deferred to v0.50.1."
        ),
    )
    # v0.51.0 Part E — alternative model hubs (ModelScope / Modelers)
    hub: Literal["hf", "modelscope", "modelers"] = Field(
        default="hf",
        description=(
            "Model hub for downloads + pushes. 'hf' (default), 'modelscope' "
            "(China-hosted; mirrors most Llama/Qwen/etc.), 'modelers' "
            "(Openmind hub). Schema-only in v0.51.0; live downloader / "
            "uploader wiring deferred to v0.51.1."
        ),
    )

    # ---- v0.52.0 — Modality II (schema-only; live wiring in v0.52.1) ----
    # Part A — TTS
    tts_family: Optional[Literal[
        "orpheus", "sesame_csm", "llasa", "spark", "oute"
    ]] = Field(
        default=None,
        description=(
            "TTS model family — required when task='tts'. One of: orpheus, "
            "sesame_csm, llasa, spark, oute. Schema-only in v0.52.0; live "
            "trainer wrapper deferred to v0.52.1."
        ),
    )
    tts_emotion: Optional[str] = Field(
        default=None,
        description=(
            "Optional emotion tag for emotion-conditioned families "
            "(Orpheus / Oute). Allowlisted per-family. (v0.52.0)"
        ),
    )
    # Part B — classifier / reranker / cross_encoder
    num_labels: Optional[int] = Field(
        default=None, ge=1, le=1024,
        description=(
            "Number of output labels for task in (classifier, reranker, "
            "cross_encoder). Required when task is one of those. (v0.52.0)"
        ),
    )
    classifier_kind: Optional[Literal["single_label", "multi_label"]] = Field(
        default=None,
        description=(
            "Sequence-classification head kind: single_label (default for "
            "task='classifier') or multi_label. (v0.52.0)"
        ),
    )
    label_names: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional human-readable label names. Length must match "
            "num_labels. Capped at 1024 entries. (v0.52.0)"
        ),
    )
    # Part C — knowledge distillation
    teacher_model: Optional[str] = Field(
        default=None,
        description=(
            "Teacher model HF id or local path — required when task='distill'. "
            "Null-byte rejected, capped at 512 chars. Schema-only in v0.52.0; "
            "live distill trainer deferred to v0.52.1."
        ),
    )
    distill_divergence: Optional[Literal[
        "forward_kl", "reverse_kl", "js"
    ]] = Field(
        default=None,
        description=(
            "Divergence used for distillation loss. 'kl' is an alias for "
            "'forward_kl' (canonical form). (v0.52.0)"
        ),
    )
    distill_temperature: Optional[float] = Field(
        default=None,
        description=(
            "Softmax temperature applied to teacher and student logits "
            "before the divergence. Bounded [0.05, 100.0]. (v0.52.0)"
        ),
    )
    # Part E — EBFT + GDPO
    ebft_variant: Optional[Literal["structured", "strided"]] = Field(
        default=None,
        description=(
            "Energy-Based FT variant. SFT-task-only; live loss kernel "
            "deferred to v0.52.1. (v0.52.0)"
        ),
    )
    ebft_temperature: Optional[float] = Field(
        default=None,
        description=(
            "Sampling temperature for EBFT energy proxy. Bounded "
            "[1e-4, 100.0]. (v0.52.0)"
        ),
    )
    gdpo_variant: Optional[Literal[
        "standard", "length_normalized", "margin"
    ]] = Field(
        default=None,
        description=(
            "Generalized DPO variant. DPO-family-task-only; live loss kernel "
            "deferred to v0.52.1. (v0.52.0)"
        ),
    )
    # Part F — MoE expert quantization + router-only training
    moe_expert_quant: Optional[Literal["nf4", "int8_rowwise"]] = Field(
        default=None,
        description=(
            "Per-expert quantization for fused-MoE Linear blocks. "
            "Requires moe_lora=true. (v0.52.0)"
        ),
    )
    train_router_only: bool = Field(
        default=False,
        description=(
            "Freeze every expert + train only the gating router (unsloth "
            "MoE recipe). Requires moe_lora=true. (v0.52.0)"
        ),
    )
    # Part G — gpt-oss reasoning effort + EOT control
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description=(
            "gpt-oss train-time reasoning effort level. Routes through a "
            "prompt prefix at training time; live formatter wiring deferred "
            "to v0.52.1. (v0.52.0)"
        ),
    )
    train_on_eot: bool = Field(
        default=False,
        description=(
            "Include explicit EOT / EOS control tokens in the SFT loss "
            "(axolotl ``train_on_eot``). Default False matches HF Trainer "
            "convention. (v0.52.0)"
        ),
    )
    # ---- v0.53.0 Quant Menu II — UD GGUFs + KV cache + NVFP4 ---------------
    # Part C — KV cache types (serve-side hint, captured here for round-trip).
    kv_cache_type: Optional[Literal["q8_0", "bf16", "f16", "fp8"]] = Field(
        default=None,
        description=(
            "KV-cache element type for inference (q8_0 / bf16 / f16 / fp8). "
            "Schema-only in v0.53.0; live wiring deferred to v0.53.1."
        ),
    )
    # Part D — Train-time advanced precision.
    fp8_attention: bool = Field(
        default=False,
        description=(
            "Extend the v0.28.0 FP8 menu to FP8 attention "
            "(axolotl-parity flag). Requires quantization_aware='fp8'. "
            "Schema-only in v0.53.0; live wiring deferred to v0.53.1."
        ),
    )
    nvfp4: bool = Field(
        default=False,
        description=(
            "Blackwell-only NVFP4 training (unsloth + axolotl). "
            "Schema-only in v0.53.0; live wiring deferred to v0.53.1."
        ),
    )
    unsloth_bnb_4bit: bool = Field(
        default=False,
        description=(
            "Promote Unsloth Dynamic 4-bit from 'inferable' to a native flag. "
            "Requires backend='unsloth' and quantization='4bit'. (v0.53.0)"
        ),
    )
    # Part E — LF / Axolotl parity.
    bnb_4bit_use_double_quant: bool = Field(
        default=False,
        description=(
            "Apply BNB 4-bit double-quantization (LF / Axolotl parity). "
            "Only meaningful when quantization='4bit'. (v0.53.0)"
        ),
    )
    llm_int8: bool = Field(
        default=False,
        description=(
            "Explicit 8-bit LLM.int8 alias for quantization='8bit'. "
            "When True, requires quantization='8bit'. (v0.53.0)"
        ),
    )
    quantize_ref_model: bool = Field(
        default=False,
        description=(
            "Apply the same Quant Menu config to the reference model "
            "(DPO/IPO/SimPO/ORPO/BCO ref model) — extends v0.40.5. (v0.53.0)"
        ),
    )
    quantize_reward_model: bool = Field(
        default=False,
        description=(
            "Apply the same Quant Menu config to the reward model "
            "(PPO/reward_model task) — extends v0.40.5. (v0.53.0)"
        ),
    )

    @field_validator(
        "fp8_attention",
        "nvfp4",
        "unsloth_bnb_4bit",
        "bnb_4bit_use_double_quant",
        "llm_int8",
        "quantize_ref_model",
        "quantize_reward_model",
        mode="before",
    )
    @classmethod
    def _validate_v053_bool_fields(cls, v):
        """v0.53.0 — explicit bool guard so YAML ``yes`` / ``1`` integers
        cannot silently coerce. Matches project bool-before-int policy.

        ``None`` falls through to Pydantic so the field's ``default=False``
        applies (review-fix — avoids silent ``None → False`` coercion that
        would mask YAML typos like ``fp8_attention: ~``).
        """
        if v is None:
            return v
        if isinstance(v, bool):
            return v
        raise TypeError(
            f"v0.53.0 flag must be bool, got {type(v).__name__}"
        )

    @field_validator("kv_cache_type", mode="before")
    @classmethod
    def _validate_kv_cache_type(cls, v):
        """v0.53.0 Part C — bool / null-byte / oversize / case-insensitive
        normalisation via the shared helper.
        """
        if v is None:
            return None
        from soup_cli.utils.kv_cache import validate_kv_cache_type

        return validate_kv_cache_type(v)

    @field_validator("teacher_model")
    @classmethod
    def _validate_teacher_model(cls, v: Optional[str]) -> Optional[str]:
        """v0.52.0 Part C — null-byte rejection + 512-char cap.

        Mirrors v0.40.5 ``reward_model`` field-validator policy. The bool
        rejection happens inside ``validate_teacher_model`` which lives in
        ``utils/distill.py`` so the runtime validator and schema agree on
        what's accepted.
        """
        if v is None:
            return v
        from soup_cli.utils.distill import validate_teacher_model

        return validate_teacher_model(v)

    @field_validator("distill_divergence", mode="before")
    @classmethod
    def _normalize_distill_divergence(cls, v):
        """v0.52.0 Part C — canonicalise ``kl`` → ``forward_kl``.

        Mirrors v0.51.0 ``_normalize_hub`` policy: the field validator runs
        the shared ``validate_*`` helper at ``mode='before'`` so the public
        schema and runtime validator agree on what's accepted.
        """
        if v is None:
            return None
        from soup_cli.utils.distill import validate_divergence

        return validate_divergence(v)

    @field_validator("distill_temperature", mode="before")
    @classmethod
    def _validate_distill_temperature(cls, v):
        """v0.52.0 Part C — bool/NaN-rejected float in [0.05, 100]."""
        if v is None:
            return None
        from soup_cli.utils.distill import validate_distill_temperature

        return validate_distill_temperature(v)

    @field_validator("ebft_temperature", mode="before")
    @classmethod
    def _validate_ebft_temperature(cls, v):
        """v0.52.0 Part E — bool/NaN-rejected float in [1e-4, 100]."""
        if v is None:
            return None
        from soup_cli.utils.ebft_gdpo import validate_ebft_temperature

        return validate_ebft_temperature(v)

    @field_validator("label_names")
    @classmethod
    def _validate_label_names(cls, v):
        """v0.52.0 Part B — dedup + per-entry validation."""
        if v is None:
            return None
        from soup_cli.utils.classifier import validate_label_names

        return validate_label_names(v)

    @field_validator("num_labels", mode="before")
    @classmethod
    def _validate_num_labels(cls, v):
        """v0.52.0 Part B (security review fix) — bool-before-int guard.

        Pydantic v2's ``Field(ge=1, le=1024)`` accepts ``True`` because bool
        subclasses int; explicit guard matches the project policy
        established in v0.30.0 ``Candidate`` / v0.36.0 ``make_cache_key`` /
        v0.41.0 ``expand_layers`` / v0.50.0 GRPO numeric fields.
        """
        if v is None:
            return None
        from soup_cli.utils.classifier import validate_num_labels

        return validate_num_labels(v)

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _validate_reasoning_effort(cls, v):
        """v0.52.0 Part G (security review fix) — canonicalise case +
        bool/null-byte/oversize rejection via the shared helper.

        Mirrors v0.51.0 ``_normalize_hub`` and v0.41.0 ``optimizer``
        policy of routing through the public ``validate_*`` helper at
        ``mode='before'`` so the schema and runtime helper agree on what's
        accepted.
        """
        if v is None:
            return None
        from soup_cli.utils.reasoning_effort import validate_reasoning_effort

        return validate_reasoning_effort(v)

    @field_validator("tts_emotion")
    @classmethod
    def _validate_tts_emotion_field(cls, v: Optional[str]) -> Optional[str]:
        """v0.52.0 Part A — bool / null-byte / oversize rejection (without
        family-specific allowlist; that fires in the cross-validator).
        """
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("tts_emotion must not be bool")
        if not isinstance(v, str):
            raise ValueError("tts_emotion must be str")
        if not v:
            raise ValueError("tts_emotion must be non-empty")
        if "\x00" in v:
            raise ValueError("tts_emotion must not contain null bytes")
        if len(v) > 32:
            raise ValueError("tts_emotion too long (max 32 chars)")
        return v

    @field_validator("hub", mode="before")
    @classmethod
    def _normalize_hub(cls, v):
        """v0.51.0 Part E review fix — accept any case (HF / Modelscope /
        MODELERS) and normalise to lowercase before the Literal check.
        Mirrors the v0.41.0 ``optimizer`` / v0.50.0 ``grpo_variant`` /
        ``rollout_backend`` policy of running the shared ``validate_*``
        helper at ``mode='before'`` so the public schema and the runtime
        validator agree on what's accepted.
        """
        # Lazy-import to avoid a hard dep cycle at module load.
        from soup_cli.utils.hubs import validate_hub_name
        if v is None:
            return v
        return validate_hub_name(v)
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

    @field_validator("reward_model")
    @classmethod
    def _validate_reward_model(cls, v: Optional[str]) -> Optional[str]:
        """v0.40.5 (#66 review fix) — reject null bytes and cap length on
        the reward_model string, matching the validation policy applied to
        cfg.base elsewhere. The Quant Menu loader (build_quantization_config_for_loader)
        already null-byte-rejects ref strings at training time; this is a
        defence-in-depth check at config-load so a crafted soup.yaml fails
        fast before any trainer is constructed.
        """
        if v is None:
            return v
        if "\x00" in v:
            raise ValueError("reward_model must not contain null bytes")
        if len(v) > 512:
            raise ValueError("reward_model must be <= 512 chars")
        return v
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
    rope_scaling_type: Optional[
        Literal["linear", "dynamic", "yarn", "longrope", "llama3"]
    ] = Field(
        default=None,
        description=(
            "RoPE scaling method for long-context: linear, dynamic, yarn, longrope, "
            "llama3 (v0.49.0)."
        ),
    )
    # v0.49.0 Part A — YaRN-specific tunables (only meaningful when
    # rope_scaling_type=='yarn'; cross-validator below enforces).
    yarn_factor: Optional[float] = Field(
        default=None,
        gt=1.0,
        le=1024.0,
        description=(
            "YaRN scaling factor (s). Optional — when omitted, the runtime falls "
            "back to ``target_length / original_length`` (HF default behaviour)."
        ),
    )
    yarn_attn_factor: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="YaRN attention temperature multiplier (default 1.0 in HF).",
    )
    yarn_beta_fast: Optional[int] = Field(
        default=None,
        ge=1,
        le=1024,
        description="YaRN beta_fast cutoff (HF default 32).",
    )
    yarn_beta_slow: Optional[int] = Field(
        default=None,
        ge=1,
        le=1024,
        description="YaRN beta_slow cutoff (HF default 1).",
    )
    # v0.49.0 Part C — LongLoRA S² shifted-sparse attention.
    # Schema gate only; live forward override deferred to v0.49.1.
    use_longlora: bool = Field(
        default=False,
        description=(
            "Enable LongLoRA S² shifted-sparse attention (v0.49.0, schema-only). "
            "Requires task=sft, backend=transformers, Llama-family base. "
            "Mutually exclusive with use_ring_attention."
        ),
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
    # Curriculum-Aware dynamic re-weighting (v0.48.0 Part A — BETA)
    curriculum_dynamic: bool = Field(
        default=False,
        description=(
            "BETA: dynamically re-weight curriculum buckets every N steps via "
            "online uncertainty estimation (per-sample loss + grad norm). "
            "Requires curriculum=true. Multi-rank launches must wire an "
            "all_reduce hook on per-bucket stats (see "
            "utils.curriculum_dynamic.validate_distributed_curriculum)."
        ),
    )
    curriculum_dynamic_recompute_steps: int = Field(
        default=50, ge=1, le=100_000,
        description=(
            "Recompute curriculum bucket sampler weights every N global "
            "training steps."
        ),
    )
    curriculum_dynamic_floor: float = Field(
        default=0.05, gt=0.0, le=0.5,
        description=(
            "Minimum normalised per-bucket weight after softmax. "
            "Must be in (0.0, 1/curriculum_buckets]; the cross-validator "
            "tightens this to the per-config ceiling. Prevents bucket "
            "starvation."
        ),
    )
    curriculum_dynamic_temperature: float = Field(
        default=1.0, gt=0.0, le=100.0,
        description=(
            "Softmax temperature on the uncertainty signal. Higher = flatter "
            "distribution; lower = concentrate on hardest buckets."
        ),
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
    # ReLoRA (v0.39.0 Part B)
    relora_steps: Optional[int] = Field(
        default=None, ge=1, le=10**7,
        description=(
            "Fire ReLoRA magnitude-prune + optimizer reset every N global steps. "
            "None disables. Requires a LoRA-style PEFT (not VeRA)."
        ),
    )
    relora_warmup_ratio: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Skip ReLoRA firings during the first warmup_ratio fraction of training",
    )
    relora_reset_optimizer: bool = Field(
        default=True,
        description="Clear optimizer state for pruned LoRA params on each ReLoRA fire",
    )
    relora_prune_ratio: float = Field(
        default=0.9, gt=0.0, lt=1.0,
        description=(
            "Fraction of LoRA weights to zero out by magnitude on each fire "
            "(0.9 keeps the top 10%). Must be < 1.0."
        ),
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
    def _validate_grpo_stability_pairings(self) -> "TrainingConfig":
        """v0.50.0 Part D — surfaces probable footguns in stability knobs.

        ``mask_truncated_completions=True`` without ``tis_threshold`` is a
        no-op (the mask is built from the threshold). Reject loudly rather
        than silently no-op (mirrors v0.32.0 spike-recovery + watchdog
        cross-validator policy).
        """
        if self.mask_truncated_completions and self.tis_threshold is None:
            raise ValueError(
                "mask_truncated_completions requires tis_threshold to be set "
                "(the truncation mask is derived from the importance-sampling "
                "threshold)"
            )
        return self

    @field_validator(
        "ref_model_ema_alpha",
        "tis_threshold",
        "off_policy_mask_threshold",
        "replay_buffer_size",
        "grpo_delta",
        mode="before",
    )
    @classmethod
    def _reject_bool_on_grpo_numerics(cls, v: object, info: object) -> object:
        """v0.50.0 (tdd-guide HIGH fix) — explicit bool rejection on every
        numeric stability/RL knob. Matches v0.30.0 ``Candidate`` /
        v0.41.0 Part B ``lr_groups`` / v0.43.0 Part B ``Tournament`` policy.

        Pydantic v2 coerces ``True`` → ``1`` and ``False`` → ``0`` on int /
        float fields by default; that would silently accept a misconfigured
        YAML where a user typed ``true`` instead of a numeric literal.
        """
        if isinstance(v, bool):
            raise ValueError(
                f"{getattr(info, 'field_name', 'field')} must not be bool"
            )
        return v

    @field_validator("grpo_delta", mode="after")
    @classmethod
    def _validate_grpo_delta_finite(cls, v: Optional[float]) -> Optional[float]:
        """v0.50.0 Part A (security review fix) — explicit NaN/Inf rejection.

        Pydantic's ``gt=0.0, le=1.0`` incidentally rejects NaN (since
        ``NaN > 0.0`` is False), but the rejection is implicit. Make it
        explicit so a future Pydantic change cannot regress the guard.
        Mirrors v0.32.0 ``save_lr_finder_report`` / v0.47.0 Part A
        ``build_forge_plan`` policy.
        """
        if v is None:
            return v
        import math as _math

        if not _math.isfinite(v):
            raise ValueError("grpo_delta must be finite (no NaN/Inf)")
        return v

    @model_validator(mode="after")
    def _validate_grpo_variant_delta(self) -> "TrainingConfig":
        """v0.50.0 Part A — grpo_variant='two_sided' requires grpo_delta.

        Conversely, grpo_delta is only meaningful for the two_sided variant;
        setting it on any other variant (or with no variant) is rejected
        as a probable footgun (matches v0.40.0 Part D ``preference_loss_weights``
        + ``preference_loss`` mutually-exclusive policy).
        """
        if self.grpo_variant == "two_sided" and self.grpo_delta is None:
            raise ValueError(
                "grpo_variant='two_sided' requires grpo_delta "
                "(symmetric clipping radius, (0, 1])"
            )
        if self.grpo_delta is not None and self.grpo_variant != "two_sided":
            raise ValueError(
                "grpo_delta is only valid when grpo_variant='two_sided'; "
                f"got grpo_variant={self.grpo_variant!r}"
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
    def _validate_curriculum_dynamic_requires_curriculum(self) -> "TrainingConfig":
        """v0.48.0 Part A — dynamic re-weighting layers on the static
        curriculum bucketer; it cannot run alone."""
        if self.curriculum_dynamic and not self.curriculum:
            raise ValueError(
                "curriculum_dynamic requires curriculum=true "
                "(dynamic re-weighting needs the static bucketer)."
            )
        # Cross-check: floor must leave room above uniform/N.
        if self.curriculum_dynamic:
            ceiling = 1.0 / max(self.curriculum_buckets, 1)
            if self.curriculum_dynamic_floor > ceiling:
                raise ValueError(
                    f"curriculum_dynamic_floor={self.curriculum_dynamic_floor} "
                    f"must be <= 1/curriculum_buckets ({ceiling:.4f})."
                )
        return self

    @field_validator(
        "yarn_factor",
        "yarn_attn_factor",
        "yarn_beta_fast",
        "yarn_beta_slow",
        mode="before",
    )
    @classmethod
    def _reject_bool_yarn(cls, value: Any) -> Any:
        """v0.49.0 Part A — bool is a subclass of int/float in Python; Pydantic
        would silently accept ``True``. Reject explicitly (project bool-as-int
        policy, mirrors v0.30.0 Candidate / v0.34.0 estimate_run_cost_usd)."""
        if isinstance(value, bool):
            raise ValueError(
                "bool is not a valid value for a YaRN tunable (use a real number)"
            )
        return value

    @model_validator(mode="after")
    def _validate_yarn_fields_require_yarn_type(self) -> "TrainingConfig":
        """v0.49.0 Part A — yarn_* fields are no-ops unless
        ``rope_scaling_type='yarn'``. Surface the misconfig loudly at config
        load rather than silently dropping the values.
        """
        yarn_fields = {
            "yarn_factor": self.yarn_factor,
            "yarn_attn_factor": self.yarn_attn_factor,
            "yarn_beta_fast": self.yarn_beta_fast,
            "yarn_beta_slow": self.yarn_beta_slow,
        }
        set_fields = [name for name, value in yarn_fields.items() if value is not None]
        if set_fields and self.rope_scaling_type != "yarn":
            raise ValueError(
                f"{', '.join(set_fields)} only apply when rope_scaling_type='yarn' "
                f"(got rope_scaling_type={self.rope_scaling_type!r})."
            )
        return self

    @model_validator(mode="after")
    def _validate_longlora_ring_attn_exclusive(self) -> "TrainingConfig":
        """v0.49.0 Part C — LongLoRA's S² shifted-sparse attention is a custom
        forward override that conflicts with ring/FA-v3 custom-mask attention
        paths."""
        if self.use_longlora and self.use_ring_attention:
            raise ValueError(
                "use_longlora is incompatible with use_ring_attention "
                "(both rewrite the attention kernel — pick one)."
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

    @field_validator("optimizer")
    @classmethod
    def _validate_optimizer(cls, value: str) -> str:
        """v0.41.0 Part A — optimizer allowlist."""
        from soup_cli.utils.optimizer_zoo import validate_optimizer_name

        return validate_optimizer_name(value)

    @field_validator("lr_groups", mode="before")
    @classmethod
    def _validate_lr_groups(cls, value):
        """v0.41.0 Part B — parse + validate lr_groups."""
        if value is None:
            return None
        from soup_cli.utils.lr_groups import parse_lr_groups

        parsed = parse_lr_groups(value)
        if parsed is None:
            return None
        # Re-emit as the raw schema shape (list of {pattern, lr} dicts) so
        # round-tripping through model_dump preserves user-visible structure.
        return [{"pattern": g.pattern, "lr": g.lr} for g in parsed]

    @field_validator("freeze_trainable_layers", mode="before")
    @classmethod
    def _validate_freeze_trainable_layers(cls, value):
        """v0.41.0 Part C — magnitude capped at 1000."""
        if value is None:
            return None
        from soup_cli.utils.block_expansion import (
            validate_freeze_trainable_layers,
        )

        return validate_freeze_trainable_layers(value)

    @field_validator("expand_layers", mode="before")
    @classmethod
    def _validate_expand_layers_field(cls, value):
        """v0.41.0 Part C — block expansion bounds + bool rejection.

        Pydantic's `Field(ge=1, le=64)` accepts ``True`` (subclass of int);
        the explicit validator rejects bool and routes through the shared
        helper so the int bounds stay single-source-of-truth.
        """
        if value is None:
            return None
        from soup_cli.utils.block_expansion import validate_expand_layers

        return validate_expand_layers(value)

    @model_validator(mode="after")
    def _validate_load_in_aliases(self) -> "TrainingConfig":
        """v0.41.0 Part C — load_in_8bit / load_in_16bit aliases.

        Mutually exclusive. When set to True, they override ``quantization``
        only if the user did not explicitly pick a Quant Menu format
        (gptq / awq / hqq:* / aqlm / eetq / mxfp4 / fp8). Mixing alias=True
        with Quant Menu raises rather than silently overriding the explicit
        pick. Uses ``is True`` (project policy) so an explicit ``False``
        from the user is treated as "no preference", never silently
        rewriting the field.
        """
        l8 = self.load_in_8bit
        l16 = self.load_in_16bit
        if l8 is True and l16 is True:
            raise ValueError(
                "load_in_8bit and load_in_16bit are mutually exclusive — "
                "pick one."
            )
        if l8 is not True and l16 is not True:
            return self
        # Defer the import: utils.quant_menu is loaded lazily elsewhere.
        from soup_cli.utils.quant_menu import is_quant_menu_format

        if is_quant_menu_format(self.quantization):
            raise ValueError(
                f"load_in_8bit / load_in_16bit cannot be combined with "
                f"quantization={self.quantization!r} (Quant Menu format). "
                "Either remove the alias or set quantization to '4bit', "
                "'8bit', or 'none'."
            )
        # Direct assignment routes through Pydantic v2 BaseModel.__setattr__
        # so any future field_validator on ``quantization`` still fires.
        # ``object.__setattr__`` would silently bypass that path.
        if l8 is True and self.quantization != "8bit":
            self.quantization = "8bit"
        elif l16 is True and self.quantization != "none":
            self.quantization = "none"
        return self

    @model_validator(mode="after")
    def _validate_block_expansion_pair(self) -> "TrainingConfig":
        """v0.41.0 Part C — expand_layers + freeze_trainable_layers pair."""
        if self.expand_layers is not None and self.freeze_trainable_layers is None:
            raise ValueError(
                "expand_layers requires freeze_trainable_layers (LLaMA Pro "
                "freezes the original layers and trains only the new blocks). "
                "Set freeze_trainable_layers: <signed int>."
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
        "bco", "preference", "pretrain", "embedding", "prm",
        # v0.52.0 Modality II — TTS / classifier-family / distillation.
        "tts", "classifier", "reranker", "cross_encoder", "distill",
    ] = Field(
        default="sft",
        description=(
            "Training task type. v0.50.0 Part E added 'prm'; v0.52.0 adds "
            "'tts' (TTS fine-tuning), 'classifier' / 'reranker' / "
            "'cross_encoder' (classification heads), and 'distill' "
            "(knowledge distillation)."
        ),
    )
    modality: Literal["text", "vision", "audio", "audio_out"] = Field(
        default="text",
        description=(
            "Training modality: text (default), vision (multimodal), audio "
            "(audio-input), or audio_out (audio-output — paired with task='tts', "
            "v0.52.0)."
        ),
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

    @model_validator(mode="before")
    @classmethod
    def _remap_root_level_misplaced_keys(cls, values):
        """v0.40.1 Part B — QA finding C2: users naturally write top-level
        ``lora:`` (LlamaFactory / Axolotl convention) but Soup nests it
        under ``training``. Without remap, Pydantic silently drops the
        misplaced key — including ``lora.init_strategy`` validation.

        Migrate root-level ``lora`` into ``training.lora`` so nested
        validation (Literal["random","pissa","olora"]) actually fires.

        Caller's dict is never mutated — we work on shallow copies, matching
        v0.33.0 #47 / v0.40.0 Part B immutability policy.
        """
        if not isinstance(values, dict):
            return values
        # Detect any misplaced key first so we avoid copying when not needed.
        misplaced_keys = [k for k in ("lora",) if k in values]
        if not misplaced_keys:
            return values
        new_values = dict(values)
        new_training = dict(new_values.get("training") or {})
        for misplaced in misplaced_keys:
            if misplaced in new_training:
                raise ValueError(
                    f"{misplaced!r} found at both root and training level — "
                    f"keep only one (training.{misplaced} preferred)."
                )
            new_training[misplaced] = new_values.pop(misplaced)
        new_values["training"] = new_training
        return new_values

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
    def _validate_curriculum_dynamic_supported(self) -> "SoupConfig":
        """v0.48.0 Part A — Curriculum-Aware dynamic re-weighting.

        BETA: wired for transformers backend, sft + pretrain tasks only.
        MLX backend rejected (callback is HF Trainer-specific). Other tasks
        rejected because their per-sample loss semantics differ enough that
        the bucket-level uncertainty heuristic does not transfer cleanly.
        Multi-trainer expansion tracked for v0.48.1.
        """
        if not self.training.curriculum_dynamic:
            return self
        if self.backend == "mlx":
            raise ValueError(
                "curriculum_dynamic is not supported on the mlx backend "
                "(callback is HF Trainer-specific). "
                "Use backend='transformers' or set curriculum_dynamic: false."
            )
        if self.task not in ("sft", "pretrain"):
            raise ValueError(
                f"curriculum_dynamic is not supported for task={self.task!r} "
                "in v0.48.0 (only sft and pretrain are wired). "
                "Set curriculum_dynamic: false or switch task."
            )
        return self

    @model_validator(mode="after")
    def _validate_longlora_compat(self) -> "SoupConfig":
        """v0.49.0 Part C — LongLoRA S² shifted-sparse attention requires
        ``task=sft``, ``backend=transformers``, and a Llama-family base.

        Live forward override is deferred to v0.49.1 (mirrors v0.27.0 MII /
        v0.37.0 multipack stub-then-live pattern); the schema gate prevents
        misconfiguration today.
        """
        if not self.training.use_longlora:
            return self
        from soup_cli.utils.longlora import validate_longlora_compat

        try:
            validate_longlora_compat(
                model_name=self.base,
                task=self.task,
                backend=self.backend,
                use_ring_attention=self.training.use_ring_attention,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_grpo_variant_supported(self) -> "SoupConfig":
        """v0.50.0 Part A — ``grpo_variant`` only valid on task='grpo' and
        transformers/unsloth backends. MLX rejected with distinct message
        (matches v0.34.0 review-fix policy of distinct error reasons).

        Live loss kernels for non-standard variants are deferred to v0.50.1;
        a yellow advisory at trainer construction time will name the
        deferred wiring (mirrors v0.40.0 Part D ``NotImplementedError``
        stub-then-live pattern).
        """
        if self.training.grpo_variant is None:
            return self
        if self.task != "grpo":
            raise ValueError(
                f"grpo_variant is only valid when task='grpo'; "
                f"got task={self.task!r}"
            )
        if self.backend == "mlx":
            raise ValueError(
                "grpo_variant is not supported on backend=mlx in v0.50.0 "
                "(MLX GRPO is scaffolded; new RL objectives transformers-only)"
            )
        return self

    @model_validator(mode="after")
    def _validate_long_context_grpo(self) -> "SoupConfig":
        """v0.50.0 Part B — ``long_context_grpo`` compatibility gate.

        Delegates to :func:`grpo_long_context.validate_long_context_grpo_compat`
        so the rules are single-source-of-truth (mirrors v0.49.0 LongLoRA).
        Live Tiled MLP wiring is deferred to v0.56.0.
        """
        if not self.training.long_context_grpo:
            return self
        from soup_cli.utils.grpo_long_context import (
            validate_long_context_grpo_compat,
        )

        try:
            validate_long_context_grpo_compat(
                task=self.task,
                backend=self.backend,
                use_ring_attention=self.training.use_ring_attention,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_prm_compat(self) -> "SoupConfig":
        """v0.50.0 Part E — ``task='prm'`` schema gate.

        Delegates to :func:`prm.validate_prm_compat` so the rules are
        single-source-of-truth. Live PRM trainer wrapper is deferred to
        v0.50.1.
        """
        if self.task != "prm":
            return self
        from soup_cli.utils.prm import validate_prm_compat

        try:
            validate_prm_compat(
                task=self.task,
                data_format=self.data.format,
                backend=self.backend,
                modality=self.modality,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_vision_grpo(self) -> "SoupConfig":
        """v0.50.0 Part E — ``vision_grpo=True`` compat gate."""
        if not self.training.vision_grpo:
            return self
        from soup_cli.utils.prm import validate_vision_grpo_compat

        try:
            validate_vision_grpo_compat(
                task=self.task,
                modality=self.modality,
                backend=self.backend,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_grpo_stability_task_gate(self) -> "SoupConfig":
        """v0.50.0 Part D — GRPO-specific stability knobs require task='grpo'.

        Surfaces a probable footgun where a user sets one of the seven new
        GRPO stability fields on a non-GRPO task (where they are a silent
        no-op). Mirrors v0.49.0 LongLoRA / v0.48.0 curriculum_dynamic
        task-gate policy.
        """
        tcfg = self.training
        grpo_only_fields = {
            "ref_model_ema_alpha": tcfg.ref_model_ema_alpha,
            "replay_buffer_size": tcfg.replay_buffer_size,
            "async_grpo_prefetch": tcfg.async_grpo_prefetch,
            "tis_threshold": tcfg.tis_threshold,
            "mask_truncated_completions": tcfg.mask_truncated_completions,
            "defer_rerolling": tcfg.defer_rerolling,
            "skip_zero_advantage": tcfg.skip_zero_advantage,
            "off_policy_mask_threshold": tcfg.off_policy_mask_threshold,
            "grpo_fp16": tcfg.grpo_fp16,
        }
        # Bool defaults are False; Optional defaults are None.
        active = [
            name for name, value in grpo_only_fields.items()
            if value not in (None, False)
        ]
        if not active:
            return self
        if self.task != "grpo":
            raise ValueError(
                f"GRPO stability fields {active} require task='grpo'; "
                f"got task={self.task!r}"
            )
        if self.backend == "mlx":
            raise ValueError(
                f"GRPO stability fields {active} are not supported on "
                "backend=mlx in v0.50.0"
            )
        return self

    @model_validator(mode="after")
    def _validate_hub_supported(self) -> "SoupConfig":
        """v0.51.0 Part E — ``hub`` other than ``hf`` requires a non-mlx
        backend.

        ``mlx-lm`` has no ModelScope/Modelers download integration, so a
        config that pairs ``backend: mlx`` + ``hub: modelscope`` would fail
        at runtime with a confusing ``mlx-lm`` error. Reject loudly at
        config-load with a distinct message (matches v0.34.0 review-fix
        policy).
        """
        if self.training.hub == "hf":
            return self
        if self.backend == "mlx":
            raise ValueError(
                f"hub={self.training.hub!r} is not supported on "
                "backend=mlx (mlx-lm only downloads from HF Hub). "
                "Use hub='hf' on the mlx backend."
            )
        return self

    @model_validator(mode="after")
    def _validate_tts_compat(self) -> "SoupConfig":
        """v0.52.0 Part A — ``task='tts'`` gate."""
        tcfg = self.training
        if self.task != "tts" and tcfg.tts_family is None and tcfg.tts_emotion is None:
            return self
        if self.task == "tts":
            from soup_cli.utils.tts import (
                validate_emotion_tag,
                validate_tts_compat,
            )

            try:
                validate_tts_compat(
                    task=self.task,
                    modality=self.modality,
                    backend=self.backend,
                )
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            if tcfg.tts_family is None:
                raise ValueError(
                    "task='tts' requires training.tts_family in "
                    "(orpheus, sesame_csm, llasa, spark, oute)"
                )
            if tcfg.tts_emotion is not None:
                try:
                    validate_emotion_tag(tcfg.tts_emotion, family=tcfg.tts_family)
                except ValueError as exc:
                    raise ValueError(str(exc)) from exc
            return self
        # tts_family / tts_emotion outside task='tts' is a silent-no-op
        # footgun; reject loudly (mirrors v0.50.0 GRPO stability policy).
        if tcfg.tts_family is not None:
            raise ValueError(
                f"training.tts_family={tcfg.tts_family!r} requires task='tts'; "
                f"got task={self.task!r}"
            )
        if tcfg.tts_emotion is not None:
            raise ValueError(
                f"training.tts_emotion={tcfg.tts_emotion!r} requires task='tts'; "
                f"got task={self.task!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_classifier_compat(self) -> "SoupConfig":
        """v0.52.0 Part B — classifier / reranker / cross_encoder gate.

        Lazy-import policy (code-review fix): guard the import behind the
        cheap str-only ``self.task`` check so the common ``task='sft'`` hot
        path does not pay an import cost on every config load.
        """
        tcfg = self.training
        classifier_tasks = {"classifier", "reranker", "cross_encoder"}
        classifier_fields_set = (
            tcfg.num_labels is not None
            or tcfg.classifier_kind is not None
            or tcfg.label_names is not None
        )
        if self.task not in classifier_tasks and not classifier_fields_set:
            return self
        from soup_cli.utils.classifier import (
            is_classifier_task,
            validate_classifier_compat,
        )

        if is_classifier_task(self.task):
            try:
                validate_classifier_compat(
                    task=self.task,
                    backend=self.backend,
                    modality=self.modality,
                )
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            if tcfg.num_labels is None:
                raise ValueError(
                    f"task={self.task!r} requires training.num_labels "
                    "(positive int <= 1024)"
                )
            if (
                tcfg.label_names is not None
                and len(tcfg.label_names) != tcfg.num_labels
            ):
                raise ValueError(
                    f"len(label_names)={len(tcfg.label_names)} does not "
                    f"match num_labels={tcfg.num_labels}"
                )
            return self
        # Reject classifier-only fields when task is not a classifier task.
        for field in ("num_labels", "classifier_kind", "label_names"):
            value = getattr(tcfg, field)
            if value is not None:
                raise ValueError(
                    f"training.{field} requires task in "
                    "(classifier, reranker, cross_encoder); "
                    f"got task={self.task!r}"
                )
        return self

    @model_validator(mode="after")
    def _validate_distill_compat(self) -> "SoupConfig":
        """v0.52.0 Part C — ``task='distill'`` gate."""
        tcfg = self.training
        distill_fields_set = (
            tcfg.teacher_model is not None
            or tcfg.distill_divergence is not None
            or tcfg.distill_temperature is not None
        )
        if self.task == "distill":
            from soup_cli.utils.distill import validate_distill_compat

            try:
                validate_distill_compat(
                    task=self.task,
                    backend=self.backend,
                    teacher_model=tcfg.teacher_model,
                )
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            return self
        if distill_fields_set:
            offenders = [
                name for name, value in (
                    ("teacher_model", tcfg.teacher_model),
                    ("distill_divergence", tcfg.distill_divergence),
                    ("distill_temperature", tcfg.distill_temperature),
                ) if value is not None
            ]
            raise ValueError(
                f"Distillation fields {offenders} require task='distill'; "
                f"got task={self.task!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_bitnet_compat(self) -> "SoupConfig":
        """v0.52.0 Part D — ``quantization='bitnet_1.58'`` gate."""
        if self.training.quantization != "bitnet_1.58":
            return self
        from soup_cli.utils.bitnet import validate_bitnet_compat

        try:
            validate_bitnet_compat(
                task=self.task,
                backend=self.backend,
                modality=self.modality,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_ebft_compat(self) -> "SoupConfig":
        """v0.52.0 Part E — ``ebft_variant`` requires SFT, non-MLX."""
        tcfg = self.training
        if tcfg.ebft_variant is None and tcfg.ebft_temperature is None:
            return self
        if tcfg.ebft_variant is None and tcfg.ebft_temperature is not None:
            raise ValueError(
                "training.ebft_temperature requires training.ebft_variant "
                "to be set"
            )
        from soup_cli.utils.ebft_gdpo import validate_ebft_compat

        try:
            validate_ebft_compat(task=self.task, backend=self.backend)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_gdpo_compat(self) -> "SoupConfig":
        """v0.52.0 Part E — ``gdpo_variant`` requires DPO/preference, non-MLX."""
        if self.training.gdpo_variant is None:
            return self
        from soup_cli.utils.ebft_gdpo import validate_gdpo_compat

        try:
            validate_gdpo_compat(task=self.task, backend=self.backend)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_reasoning_effort_task_gate(self) -> "SoupConfig":
        """v0.52.0 Part G (code-review fix) — surface the silent-no-op
        footgun when ``reasoning_effort`` / ``train_on_eot`` is set on a
        task they cannot influence.

        Mirrors v0.50.0 ``_validate_grpo_stability_task_gate`` policy.
        ``reasoning_effort`` only makes sense on SFT-family training
        (sft / pretrain / distill / classifier-family) because the live
        formatter (v0.52.1) will inject a system-prefix token. The other
        tasks (DPO / GRPO / KTO / ORPO / SimPO / IPO / BCO / preference /
        PPO / reward_model / embedding / prm / tts) do not consume it.

        ``train_on_eot`` is an SFT loss-mask flag; setting it on
        DPO/GRPO/etc. is a silent no-op.
        """
        tcfg = self.training
        sft_family_tasks = {
            "sft", "pretrain", "distill",
            "classifier", "reranker", "cross_encoder",
        }
        if tcfg.reasoning_effort is not None and self.task not in sft_family_tasks:
            raise ValueError(
                f"training.reasoning_effort={tcfg.reasoning_effort!r} requires "
                f"task in {sorted(sft_family_tasks)}; got task={self.task!r}"
            )
        if tcfg.train_on_eot and self.task not in sft_family_tasks:
            raise ValueError(
                f"training.train_on_eot=true requires task in "
                f"{sorted(sft_family_tasks)}; got task={self.task!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_moe_expert_quant_compat(self) -> "SoupConfig":
        """v0.52.0 Part F — ``moe_expert_quant`` + ``train_router_only`` gates."""
        tcfg = self.training
        from soup_cli.utils.moe_quant import (
            validate_moe_expert_quant_compat,
            validate_train_router_only_compat,
        )

        if tcfg.moe_expert_quant is not None:
            try:
                validate_moe_expert_quant_compat(
                    backend=self.backend, moe_lora=tcfg.moe_lora,
                )
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        if tcfg.train_router_only:
            try:
                validate_train_router_only_compat(
                    backend=self.backend, moe_lora=tcfg.moe_lora,
                )
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_rollout_backend(self) -> "SoupConfig":
        """v0.50.0 Part C — ``rollout_backend`` requires task='grpo' and a
        non-mlx backend. Live launcher wired in v0.50.1."""
        if self.training.rollout_backend is None:
            return self
        if self.task != "grpo":
            raise ValueError(
                f"rollout_backend requires task='grpo'; got task={self.task!r}"
            )
        if self.backend == "mlx":
            raise ValueError(
                "rollout_backend is not supported on backend=mlx in v0.50.0"
            )
        return self

    @model_validator(mode="after")
    def _validate_vllm_sleep_mode(self) -> "SoupConfig":
        """v0.50.0 Part B — ``vllm_sleep_mode`` requires task='grpo' and a
        vLLM-compatible backend (transformers/unsloth).

        Sleep mode is a between-rollouts feature; setting it on a non-RL
        task is a probable footgun and silently no-ops, so reject loudly.
        """
        if not self.training.vllm_sleep_mode:
            return self
        if self.task != "grpo":
            raise ValueError(
                f"vllm_sleep_mode requires task='grpo'; got task={self.task!r}"
            )
        from soup_cli.utils.grpo_long_context import (
            validate_vllm_sleep_mode_compat,
        )

        try:
            validate_vllm_sleep_mode_compat(backend=self.backend)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    # ---- v0.53.0 Quant Menu II cross-validators ----------------------------

    @model_validator(mode="after")
    def _validate_fp8_attention_compat(self) -> "SoupConfig":
        """v0.53.0 Part D — ``fp8_attention=True`` requires
        ``quantization_aware='fp8'`` and a non-mlx backend. Silent-no-op
        footgun rejection (mirrors v0.32.0 spike-recovery policy).
        """
        tcfg = self.training
        if not tcfg.fp8_attention:
            return self
        from soup_cli.utils.advanced_precision import (
            validate_fp8_attention_compat,
        )

        try:
            validate_fp8_attention_compat(
                fp8_attention=tcfg.fp8_attention,
                quantization_aware=tcfg.quantization_aware,
                backend=self.backend,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_nvfp4_compat(self) -> "SoupConfig":
        """v0.53.0 Part D — ``nvfp4=True`` requires non-mlx + text-modality.
        Blackwell SM-capability check is runtime-only (live wiring v0.53.1).
        """
        tcfg = self.training
        if not tcfg.nvfp4:
            return self
        from soup_cli.utils.advanced_precision import validate_nvfp4_compat

        try:
            validate_nvfp4_compat(
                nvfp4=tcfg.nvfp4, backend=self.backend, modality=self.modality,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_unsloth_bnb_4bit_compat(self) -> "SoupConfig":
        """v0.53.0 Part D — ``unsloth_bnb_4bit=True`` requires
        ``backend='unsloth'`` and ``quantization='4bit'``.
        """
        tcfg = self.training
        if not tcfg.unsloth_bnb_4bit:
            return self
        from soup_cli.utils.advanced_precision import (
            validate_unsloth_bnb_4bit_compat,
        )

        try:
            validate_unsloth_bnb_4bit_compat(
                unsloth_bnb_4bit=tcfg.unsloth_bnb_4bit,
                backend=self.backend,
                quantization=tcfg.quantization,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_bnb_4bit_double_quant(self) -> "SoupConfig":
        """v0.53.0 Part E — ``bnb_4bit_use_double_quant=True`` requires
        ``quantization='4bit'`` (silent-no-op footgun otherwise).
        """
        tcfg = self.training
        if not tcfg.bnb_4bit_use_double_quant:
            return self
        if tcfg.quantization != "4bit":
            raise ValueError(
                "training.bnb_4bit_use_double_quant=true requires "
                f"training.quantization='4bit'; got "
                f"quantization={tcfg.quantization!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_llm_int8_alias(self) -> "SoupConfig":
        """v0.53.0 Part E — ``llm_int8=True`` requires ``quantization='8bit'``.

        Unlike v0.41.0 ``load_in_8bit`` (which rewrites quantization), the
        ``llm_int8`` flag is a pure assertion: the user explicitly says
        "this is an LLM.int8 run" and we enforce the matching quantization
        rather than silently rewriting it.
        """
        tcfg = self.training
        if not tcfg.llm_int8:
            return self
        if tcfg.quantization != "8bit":
            raise ValueError(
                "training.llm_int8=true requires training.quantization='8bit'; "
                f"got quantization={tcfg.quantization!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_quantize_ref_reward(self) -> "SoupConfig":
        """v0.53.0 Part E — ``quantize_ref_model`` requires a ref-model task
        and ``quantize_reward_model`` requires a reward-model task.
        Silent-no-op footgun rejection.

        Ref-model tasks (review fix): all preference-family trainers PLUS
        ``grpo`` (KL to ref policy) and ``kto`` (unpaired preference, also
        keeps a frozen ref). ``ppo`` also has a ref but uses a separately
        named ``policy_ref`` checkpoint; covered by the reward path too.
        """
        tcfg = self.training
        ref_tasks = {
            "dpo", "ipo", "simpo", "orpo", "bco", "kto",
            "preference", "grpo", "ppo",
        }
        reward_tasks = {"ppo", "reward_model"}
        if tcfg.quantize_ref_model and self.task not in ref_tasks:
            raise ValueError(
                "training.quantize_ref_model=true requires a task with a "
                "reference model "
                f"(one of {sorted(ref_tasks)}); got task={self.task!r}"
            )
        if tcfg.quantize_reward_model and self.task not in reward_tasks:
            raise ValueError(
                "training.quantize_reward_model=true requires task in "
                f"{sorted(reward_tasks)}; got task={self.task!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_kv_cache_type_supported(self) -> "SoupConfig":
        """v0.53.0 Part C — ``kv_cache_type`` schema gate.

        Currently only ``fp8`` is gated (Hopper-only — MLX rejected). The
        three remaining types (``q8_0`` / ``bf16`` / ``f16``) pass through
        for every backend; v0.53.1 live wiring MAY need to narrow this
        further (e.g. MLX serve may not support ``q8_0``). The schema-only
        permissive policy is deliberate this release — kept here so the
        v0.53.1 contributor sees the gate site immediately.

        Hopper SM-capability check (compute_cap >= 9.0) is runtime-only.
        """
        kv = self.training.kv_cache_type
        if kv is None:
            return self
        if self.backend == "mlx" and kv == "fp8":
            raise ValueError(
                "training.kv_cache_type='fp8' is not supported on the mlx "
                "backend (Hopper-only). Use kv_cache_type in {q8_0,bf16,f16} "
                "or switch backend."
            )
        return self

    @model_validator(mode="after")
    def _validate_relora_supported_tasks(self) -> "SoupConfig":
        """v0.40.6 (#67) — ReLoRA callback wired in every transformer-backend
        trainer (sft / dpo / grpo / kto / orpo / simpo / ipo / ppo /
        reward_model / pretrain / embedding / bco).

        MLX backend still rejected: the callback is HF Trainer-specific.
        """
        if self.training.relora_steps is None:
            return self
        if self.backend == "mlx":
            raise ValueError(
                "relora_steps is not supported on the mlx backend "
                "(callback is HF Trainer-specific). "
                "Use backend='transformers' or remove relora_steps."
            )
        return self

    @model_validator(mode="after")
    def _validate_quant_menu_supported_tasks(self) -> "SoupConfig":
        """v0.40.5 (#66) — Quant Menu (gptq/awq/hqq:Nbit/aqlm/eetq/mxfp4/fp8)
        is wired across every transformer-backend trainer (sft / dpo / grpo /
        kto / orpo / simpo / ipo / ppo / reward_model / pretrain / embedding /
        bco). MLX backend still rejected (no equivalent kernels). Vision/audio
        modality multi-trainer wiring deferred (mirrors v0.38.1 stub-then-live
        pattern for non-text modalities).
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
        if self.modality != "text":
            raise ValueError(
                f"quantization={quant!r} (Quant Menu) is wired for "
                f"modality='text' only; got modality={self.modality!r}. "
                "Vision/audio multi-modal wiring is tracked for a follow-up patch."
            )
        return self

    @model_validator(mode="after")
    def _validate_preference_dispatcher(self) -> "SoupConfig":
        """v0.40.0 Part B — task='preference' requires preference_loss OR
        preference_loss_weights (Part D). Setting either field outside
        task='preference' is rejected to keep the two config surfaces disjoint.
        """
        loss = self.training.preference_loss
        weights = self.training.preference_loss_weights
        if self.task == "preference":
            if loss is None and weights is None:
                raise ValueError(
                    "task='preference' requires either training.preference_loss "
                    "(in {dpo, simpo, orpo, ipo, bco}) or "
                    "training.preference_loss_weights (multi-objective dict)."
                )
            return self
        if loss is not None:
            raise ValueError(
                f"training.preference_loss={loss!r} is only meaningful for "
                f"task='preference'; got task={self.task!r}. Either set "
                "task='preference' or remove preference_loss."
            )
        if weights is not None:
            raise ValueError(
                "training.preference_loss_weights is only meaningful for "
                f"task='preference'; got task={self.task!r}. Either set "
                "task='preference' or remove preference_loss_weights."
            )
        return self

    @model_validator(mode="after")
    def _validate_dpo_variants_supported_tasks(self) -> "SoupConfig":
        """v0.40.0 Part C — β-schedule + ref-model regen are DPO-family only.

        Allowed: task in {dpo, ipo} OR (task='preference' AND
        preference_loss in {dpo, ipo}). Rejected on mlx backend.
        """
        tcfg = self.training
        sched = tcfg.dpo_beta_schedule
        end = tcfg.dpo_beta_end
        regen = tcfg.dpo_ref_regen_epochs
        if sched is None and end is None and regen is None:
            return self
        # End/schedule mutual requirement.
        if sched is not None and end is None:
            raise ValueError(
                "dpo_beta_schedule requires dpo_beta_end (the target β at "
                "end of training). Set dpo_beta_end or remove dpo_beta_schedule."
            )
        if end is not None and sched is None:
            raise ValueError(
                "dpo_beta_end requires dpo_beta_schedule. Set "
                "dpo_beta_schedule in {linear, cosine, exponential} or "
                "remove dpo_beta_end."
            )
        # Backend gate.
        if self.backend == "mlx":
            raise ValueError(
                "DPO variants (dpo_beta_schedule / dpo_ref_regen_epochs) are "
                "not supported on the mlx backend in v0.40.0 (TRL trainer "
                "internals required). Use backend='transformers'."
            )
        # Task gate — DPO family only.
        family_ok = self.task in ("dpo", "ipo") or (
            self.task == "preference"
            and tcfg.preference_loss in ("dpo", "ipo")
        )
        if not family_ok:
            raise ValueError(
                f"DPO variants (dpo_beta_schedule / dpo_ref_regen_epochs) "
                f"require task in {{dpo, ipo}} or task='preference' with "
                f"preference_loss in {{dpo, ipo}}; got task={self.task!r}, "
                f"preference_loss={tcfg.preference_loss!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_preference_loss_weights(self) -> "SoupConfig":
        """v0.40.0 Part D — multi-objective preference_loss_weights gate.

        Allowed: task='preference' only. Mutually exclusive with the scalar
        preference_loss. Validates value bounds + sum-to-1 + key allowlist.
        """
        tcfg = self.training
        weights = tcfg.preference_loss_weights
        if weights is None:
            return self
        if not isinstance(weights, dict):
            raise ValueError(
                "preference_loss_weights must be a dict, e.g. "
                "{'dpo': 0.7, 'bco': 0.3}."
            )
        if self.task != "preference":
            raise ValueError(
                "preference_loss_weights requires task='preference'; got "
                f"task={self.task!r}."
            )
        if tcfg.preference_loss is not None:
            raise ValueError(
                "preference_loss_weights and (scalar) preference_loss are "
                "mutually exclusive — pick one."
            )
        if self.backend == "mlx":
            raise ValueError(
                "preference_loss_weights is not supported on the mlx backend "
                "in v0.40.0. Use backend='transformers'."
            )
        if not (2 <= len(weights) <= 5):
            raise ValueError(
                f"preference_loss_weights must have between 2 and 5 entries "
                "(single-entry blends are equivalent to the scalar "
                "preference_loss field; use that instead); got "
                f"{len(weights)}."
            )
        allowed = {"dpo", "simpo", "orpo", "ipo", "bco"}
        for key in weights:
            if not isinstance(key, str):
                raise ValueError(
                    f"preference_loss_weights keys must be strings; "
                    f"got {type(key).__name__}."
                )
            if "\x00" in key:
                raise ValueError(
                    "preference_loss_weights keys cannot contain null bytes."
                )
        unknown = set(weights.keys()) - allowed
        if unknown:
            raise ValueError(
                f"preference_loss_weights keys must be in {sorted(allowed)}; "
                f"unknown: {sorted(unknown)}."
            )
        for key, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"preference_loss_weights[{key!r}] must be a number; "
                    f"got {type(value).__name__}."
                )
            if not (0 < float(value) <= 1):
                raise ValueError(
                    f"preference_loss_weights[{key!r}]={value!r} must be in (0, 1]."
                )
        total = sum(float(v) for v in weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"preference_loss_weights must sum to 1.0 (±1e-6); got {total!r}."
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

# DEPRECATED (v0.39.0 Part E) — these inline templates are kept for back-compat.
# The canonical source is `soup_cli/templates/*.yaml` with `manifest.json`.
# Both sources are asserted equal in tests/test_templates_yaml.py — when editing
# a template, update both. Planned removal: v0.41.0+ once external consumers
# have migrated to the YAML registry.
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
    "bco": """# Soup template: BCO (Binary Classifier Optimization)
# Preference alignment via binary classification of chosen vs rejected.
#
# Data format (JSONL) — same as DPO:
#   {"prompt": "What is 2+2?", "chosen": "4", "rejected": "Fish"}

base: meta-llama/Llama-3.1-8B-Instruct
task: bco
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
  bco_beta: 0.1

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
