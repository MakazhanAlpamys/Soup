"""Pydantic schemas for soup.yaml config — single source of truth."""

import re
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


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


class DataConfig(BaseModel):
    train: str = Field(..., description="Path to training data or HF dataset name")
    format: Literal[
        "alpaca", "sharegpt", "chatml", "dpo", "kto", "llava", "sharegpt4v",
        "plaintext", "auto",
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


class TrainingConfig(BaseModel):
    epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    lr: float = Field(default=2e-5, gt=0, description="Learning rate")
    batch_size: Union[int, Literal["auto"]] = Field(
        default="auto",
        description="Batch size. 'auto' = find max that fits in memory.",
    )
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    quantization: Literal["4bit", "8bit", "none"] = Field(
        default="4bit",
        description="Quantization: 4bit (QLoRA), 8bit, or none (full precision)",
    )
    quantization_aware: bool = Field(
        default=False,
        description="Enable Quantization-Aware Training (QAT) for better post-quantization quality",
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
        description="Reward function: 'accuracy', 'format', or path to custom .py file",
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
    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing for memory savings on long sequences",
    )


class SoupConfig(BaseModel):
    """Root config for soup.yaml."""

    base: str = Field(..., description="Base model name or path (HF model ID)")
    task: Literal[
        "sft", "dpo", "grpo", "ppo", "reward_model", "kto", "orpo", "simpo", "ipo",
        "pretrain",
    ] = Field(
        default="sft", description="Training task type"
    )
    modality: Literal["text", "vision"] = Field(
        default="text",
        description="Training modality: text (default) or vision (multimodal)",
    )
    backend: Literal["transformers", "unsloth"] = Field(
        default="transformers",
        description="Training backend: transformers (default) or unsloth (2-5x faster)",
    )
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: str = Field(default="./output", description="Output directory for trained model")
    experiment_name: Optional[str] = Field(default=None, description="Experiment name for tracking")

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
