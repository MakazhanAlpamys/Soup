"""Recipe catalog — ready-made configs for popular models."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RecipeMeta:
    """Metadata for a recipe."""

    model: str
    task: str
    size: str
    tags: Tuple[str, ...]
    description: str
    yaml_str: str


def list_recipes() -> List[RecipeMeta]:
    """Return all recipes."""
    return list(RECIPES.values())


def get_recipe(name: str) -> Optional[RecipeMeta]:
    """Get a recipe by name. Returns None if not found."""
    return RECIPES.get(name)


def search_recipes(
    query: Optional[str] = None,
    task: Optional[str] = None,
    size: Optional[str] = None,
) -> List[RecipeMeta]:
    """Search recipes by keyword, task, or model size."""
    results = []
    for name, recipe in RECIPES.items():
        if task and recipe.task != task:
            continue
        if size and size.lower() not in name.lower() and size.lower() not in recipe.model.lower():
            continue
        if query:
            searchable = f"{name} {recipe.model} {recipe.task} {recipe.description} "
            searchable += " ".join(recipe.tags)
            if query.lower() not in searchable.lower():
                continue
        results.append(recipe)
    return results


# ---------------------------------------------------------------------------
# Recipe catalog (~30 recipes)
# ---------------------------------------------------------------------------

RECIPES: Dict[str, RecipeMeta] = {
    "llama3.1-8b-sft": RecipeMeta(
        model="meta-llama/Llama-3.1-8B-Instruct",
        task="sft",
        size="8B",
        tags=("llama", "sft", "chat", "instruction"),
        description="Llama 3.1 8B instruction tuning with LoRA",
        yaml_str="""\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "llama3.1-8b-dpo": RecipeMeta(
        model="meta-llama/Llama-3.1-8B-Instruct",
        task="dpo",
        size="8B",
        tags=("llama", "dpo", "alignment", "preference"),
        description="Llama 3.1 8B DPO alignment",
        yaml_str="""\
base: meta-llama/Llama-3.1-8B-Instruct
task: dpo

data:
  train: ./data/preference_train.jsonl
  format: dpo
  max_length: 2048

training:
  epochs: 3
  lr: 5e-6
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  dpo_beta: 0.1

output: ./output
""",
    ),
    "llama3.1-8b-grpo": RecipeMeta(
        model="meta-llama/Llama-3.1-8B-Instruct",
        task="grpo",
        size="8B",
        tags=("llama", "grpo", "reasoning", "deepseek"),
        description="Llama 3.1 8B GRPO reasoning training",
        yaml_str="""\
base: meta-llama/Llama-3.1-8B-Instruct
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    ),
    "llama3.1-8b-kto": RecipeMeta(
        model="meta-llama/Llama-3.1-8B-Instruct",
        task="kto",
        size="8B",
        tags=("llama", "kto", "alignment", "unpaired"),
        description="Llama 3.1 8B KTO unpaired preference alignment",
        yaml_str="""\
base: meta-llama/Llama-3.1-8B-Instruct
task: kto

data:
  train: ./data/kto_train.jsonl
  format: kto
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  kto_beta: 0.1

output: ./output
""",
    ),
    "llama3.1-70b-sft": RecipeMeta(
        model="meta-llama/Llama-3.1-70B-Instruct",
        task="sft",
        size="70B",
        tags=("llama", "sft", "large", "deepspeed"),
        description="Llama 3.1 70B SFT with DeepSpeed ZeRO-3",
        yaml_str="""\
base: meta-llama/Llama-3.1-70B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "llama3.2-3b-sft": RecipeMeta(
        model="meta-llama/Llama-3.2-3B-Instruct",
        task="sft",
        size="3B",
        tags=("llama", "sft", "small", "edge"),
        description="Llama 3.2 3B instruction tuning (edge-friendly)",
        yaml_str="""\
base: meta-llama/Llama-3.2-3B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "llama3.2-1b-sft": RecipeMeta(
        model="meta-llama/Llama-3.2-1B-Instruct",
        task="sft",
        size="1B",
        tags=("llama", "sft", "tiny", "edge", "mobile"),
        description="Llama 3.2 1B instruction tuning (mobile-friendly)",
        yaml_str="""\
base: meta-llama/Llama-3.2-1B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 3e-4
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: 8bit

output: ./output
""",
    ),
    "qwen2.5-7b-sft": RecipeMeta(
        model="Qwen/Qwen2.5-7B-Instruct",
        task="sft",
        size="7B",
        tags=("qwen", "sft", "chat", "instruction"),
        description="Qwen 2.5 7B instruction tuning with LoRA",
        yaml_str="""\
base: Qwen/Qwen2.5-7B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "qwen2.5-7b-dpo": RecipeMeta(
        model="Qwen/Qwen2.5-7B-Instruct",
        task="dpo",
        size="7B",
        tags=("qwen", "dpo", "alignment", "preference"),
        description="Qwen 2.5 7B DPO alignment",
        yaml_str="""\
base: Qwen/Qwen2.5-7B-Instruct
task: dpo

data:
  train: ./data/preference_train.jsonl
  format: dpo
  max_length: 2048

training:
  epochs: 3
  lr: 5e-6
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  dpo_beta: 0.1

output: ./output
""",
    ),
    "qwen2.5-7b-grpo": RecipeMeta(
        model="Qwen/Qwen2.5-7B-Instruct",
        task="grpo",
        size="7B",
        tags=("qwen", "grpo", "reasoning"),
        description="Qwen 2.5 7B GRPO reasoning training",
        yaml_str="""\
base: Qwen/Qwen2.5-7B-Instruct
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    ),
    "qwen2.5-72b-sft": RecipeMeta(
        model="Qwen/Qwen2.5-72B-Instruct",
        task="sft",
        size="72B",
        tags=("qwen", "sft", "large", "deepspeed"),
        description="Qwen 2.5 72B SFT with DeepSpeed ZeRO-3",
        yaml_str="""\
base: Qwen/Qwen2.5-72B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "qwen3-8b-sft": RecipeMeta(
        model="Qwen/Qwen3-8B",
        task="sft",
        size="8B",
        tags=("qwen", "sft", "qwen3"),
        description="Qwen 3 8B instruction tuning",
        yaml_str="""\
base: Qwen/Qwen3-8B
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "qwen3-30b-a3b-sft": RecipeMeta(
        model="Qwen/Qwen3-30B-A3B",
        task="sft",
        size="30B",
        tags=("qwen", "sft", "moe", "mixture-of-experts"),
        description="Qwen 3 30B-A3B MoE instruction tuning",
        yaml_str="""\
base: Qwen/Qwen3-30B-A3B
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 1e-4
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  moe_lora: true
  moe_aux_loss_coeff: 0.01

output: ./output
""",
    ),
    "mistral-7b-sft": RecipeMeta(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        task="sft",
        size="7B",
        tags=("mistral", "sft", "chat"),
        description="Mistral 7B instruction tuning",
        yaml_str="""\
base: mistralai/Mistral-7B-Instruct-v0.3
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "mistral-7b-dpo": RecipeMeta(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        task="dpo",
        size="7B",
        tags=("mistral", "dpo", "alignment"),
        description="Mistral 7B DPO alignment",
        yaml_str="""\
base: mistralai/Mistral-7B-Instruct-v0.3
task: dpo

data:
  train: ./data/preference_train.jsonl
  format: dpo
  max_length: 2048

training:
  epochs: 3
  lr: 5e-6
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  dpo_beta: 0.1

output: ./output
""",
    ),
    "gemma3-9b-sft": RecipeMeta(
        model="google/gemma-3-9b-it",
        task="sft",
        size="9B",
        tags=("gemma", "google", "sft", "chat"),
        description="Gemma 3 9B instruction tuning",
        yaml_str="""\
base: google/gemma-3-9b-it
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "gemma3-27b-sft": RecipeMeta(
        model="google/gemma-3-27b-it",
        task="sft",
        size="27B",
        tags=("gemma", "google", "sft", "deepspeed"),
        description="Gemma 3 27B SFT with DeepSpeed ZeRO-2",
        yaml_str="""\
base: google/gemma-3-27b-it
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "phi4-14b-sft": RecipeMeta(
        model="microsoft/phi-4",
        task="sft",
        size="14B",
        tags=("phi", "microsoft", "sft", "reasoning"),
        description="Phi-4 14B instruction tuning",
        yaml_str="""\
base: microsoft/phi-4
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "deepseek-r1-8b-grpo": RecipeMeta(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        task="grpo",
        size="8B",
        tags=("deepseek", "grpo", "reasoning", "r1"),
        description="DeepSeek R1 8B GRPO reasoning",
        yaml_str="""\
base: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    ),
    "deepseek-r1-32b-grpo": RecipeMeta(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        task="grpo",
        size="32B",
        tags=("deepseek", "grpo", "reasoning", "r1", "deepspeed"),
        description="DeepSeek R1 32B GRPO with DeepSpeed",
        yaml_str="""\
base: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    ),
    "llama3.1-8b-orpo": RecipeMeta(
        model="meta-llama/Llama-3.1-8B-Instruct",
        task="orpo",
        size="8B",
        tags=("llama", "orpo", "alignment", "reference-free"),
        description="Llama 3.1 8B ORPO reference-free alignment",
        yaml_str="""\
base: meta-llama/Llama-3.1-8B-Instruct
task: orpo

data:
  train: ./data/preference_train.jsonl
  format: dpo
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  orpo_beta: 0.1

output: ./output
""",
    ),
    "llama3.1-8b-simpo": RecipeMeta(
        model="meta-llama/Llama-3.1-8B-Instruct",
        task="simpo",
        size="8B",
        tags=("llama", "simpo", "alignment", "simple"),
        description="Llama 3.1 8B SimPO length-normalized alignment",
        yaml_str="""\
base: meta-llama/Llama-3.1-8B-Instruct
task: simpo

data:
  train: ./data/preference_train.jsonl
  format: dpo
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  simpo_gamma: 0.5

output: ./output
""",
    ),
    "llama3.1-8b-embed": RecipeMeta(
        model="meta-llama/Llama-3.1-8B",
        task="embedding",
        size="8B",
        tags=("llama", "embedding", "sentence", "cosine"),
        description="Llama 3.1 8B sentence embedding with cosine loss",
        yaml_str="""\
base: meta-llama/Llama-3.1-8B
task: embedding

data:
  train: ./data/embedding_train.jsonl
  format: embedding
  max_length: 512

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: 4bit
  embedding_loss: cosine

output: ./output
""",
    ),
    "qwen2.5-7b-pretrain": RecipeMeta(
        model="Qwen/Qwen2.5-7B",
        task="pretrain",
        size="7B",
        tags=("qwen", "pretrain", "continued", "domain"),
        description="Qwen 2.5 7B continued pre-training",
        yaml_str="""\
base: Qwen/Qwen2.5-7B
task: pretrain

data:
  train: ./data/corpus.jsonl
  format: plaintext
  max_length: 4096

training:
  epochs: 1
  lr: 1e-4
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "llama3.2-11b-vision": RecipeMeta(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        task="sft",
        size="11B",
        tags=("llama", "vision", "multimodal", "image"),
        description="Llama 3.2 11B Vision multimodal fine-tuning",
        yaml_str="""\
base: meta-llama/Llama-3.2-11B-Vision-Instruct
task: sft
modality: vision

data:
  train: ./data/vision_train.jsonl
  format: llava
  image_dir: ./data/images
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "qwen2.5-7b-reward": RecipeMeta(
        model="Qwen/Qwen2.5-7B-Instruct",
        task="reward_model",
        size="7B",
        tags=("qwen", "reward", "rlhf", "stage2"),
        description="Qwen 2.5 7B reward model (RLHF stage 2)",
        yaml_str="""\
base: Qwen/Qwen2.5-7B-Instruct
task: reward_model

data:
  train: ./data/preference_train.jsonl
  format: dpo
  max_length: 2048

training:
  epochs: 1
  lr: 1e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output_rm
""",
    ),
    "llama3.1-8b-ppo": RecipeMeta(
        model="meta-llama/Llama-3.1-8B-Instruct",
        task="ppo",
        size="8B",
        tags=("llama", "ppo", "rlhf", "stage3"),
        description="Llama 3.1 8B PPO (RLHF stage 3)",
        yaml_str="""\
base: meta-llama/Llama-3.1-8B-Instruct
task: ppo

data:
  train: ./data/prompts.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 1
  lr: 1e-6
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  reward_model: ./output_rm
  ppo_epochs: 4
  ppo_clip_ratio: 0.2
  ppo_kl_penalty: 0.05

output: ./output_ppo
""",
    ),
    "llama3.1-8b-longctx": RecipeMeta(
        model="meta-llama/Llama-3.1-8B-Instruct",
        task="sft",
        size="8B",
        tags=("llama", "sft", "longcontext", "yarn", "rope"),
        description="Llama 3.1 8B long-context (32k) with YaRN RoPE scaling",
        yaml_str="""\
base: meta-llama/Llama-3.1-8B-Instruct
task: sft

data:
  train: ./data/long_context_train.jsonl
  format: auto
  max_length: 32768

training:
  epochs: 1
  lr: 5e-6
  batch_size: 1
  gradient_accumulation_steps: 16
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  gradient_checkpointing: true
  rope_scaling_type: yarn
  use_flash_attn: true

output: ./output
""",
    ),
    # ---------------- v0.25.0: Llama 4 / Qwen 3 / Gemma 3 / DeepSeek V3 ----------------
    "llama4-scout-17b-sft": RecipeMeta(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        task="sft",
        size="17B",
        tags=("llama", "llama4", "sft", "chat", "instruction"),
        description="Llama 4 Scout 17B SFT with LoRA (4bit)",
        yaml_str="""\
base: meta-llama/Llama-4-Scout-17B-16E-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "llama4-scout-17b-dpo": RecipeMeta(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        task="dpo",
        size="17B",
        tags=("llama", "llama4", "dpo", "alignment", "preference"),
        description="Llama 4 Scout 17B DPO alignment",
        yaml_str="""\
base: meta-llama/Llama-4-Scout-17B-16E-Instruct
task: dpo

data:
  train: ./data/preference_train.jsonl
  format: dpo
  max_length: 2048

training:
  epochs: 3
  lr: 5e-6
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  dpo_beta: 0.1

output: ./output
""",
    ),
    "llama4-scout-17b-grpo": RecipeMeta(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        task="grpo",
        size="17B",
        tags=("llama", "llama4", "grpo", "reasoning"),
        description="Llama 4 Scout 17B GRPO reasoning training",
        yaml_str="""\
base: meta-llama/Llama-4-Scout-17B-16E-Instruct
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    ),
    "qwen3-14b-sft": RecipeMeta(
        model="Qwen/Qwen3-14B",
        task="sft",
        size="14B",
        tags=("qwen", "qwen3", "sft", "chat"),
        description="Qwen 3 14B instruction tuning with LoRA",
        yaml_str="""\
base: Qwen/Qwen3-14B
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "qwen3-32b-sft": RecipeMeta(
        model="Qwen/Qwen3-32B",
        task="sft",
        size="32B",
        tags=("qwen", "qwen3", "sft", "large", "deepspeed"),
        description="Qwen 3 32B SFT with DeepSpeed ZeRO-2",
        yaml_str="""\
base: Qwen/Qwen3-32B
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "qwen3-8b-grpo": RecipeMeta(
        model="Qwen/Qwen3-8B",
        task="grpo",
        size="8B",
        tags=("qwen", "qwen3", "grpo", "reasoning"),
        description="Qwen 3 8B GRPO reasoning training",
        yaml_str="""\
base: Qwen/Qwen3-8B
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    ),
    "gemma3-12b-sft": RecipeMeta(
        model="google/gemma-3-12b-it",
        task="sft",
        size="12B",
        tags=("gemma", "gemma3", "google", "sft", "chat"),
        description="Gemma 3 12B instruction tuning",
        yaml_str="""\
base: google/gemma-3-12b-it
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "gemma3-27b-dpo": RecipeMeta(
        model="google/gemma-3-27b-it",
        task="dpo",
        size="27B",
        tags=("gemma", "gemma3", "google", "dpo", "alignment"),
        description="Gemma 3 27B DPO alignment",
        yaml_str="""\
base: google/gemma-3-27b-it
task: dpo

data:
  train: ./data/preference_train.jsonl
  format: dpo
  max_length: 2048

training:
  epochs: 3
  lr: 5e-6
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  dpo_beta: 0.1

output: ./output
""",
    ),
    "deepseek-v3-7b-sft": RecipeMeta(
        model="deepseek-ai/DeepSeek-V3-0324",
        task="sft",
        size="7B",
        tags=("deepseek", "sft", "moe", "mixture-of-experts"),
        description="DeepSeek V3 SFT with MoE LoRA",
        yaml_str="""\
base: deepseek-ai/DeepSeek-V3-0324
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 1e-4
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  moe_lora: true
  moe_aux_loss_coeff: 0.01

output: ./output
""",
    ),
    # ---------------- v0.25.0: Apple Silicon MLX recipes ----------------
    "llama3.1-8b-sft-mlx": RecipeMeta(
        model="mlx-community/Llama-3.1-8B-Instruct-4bit",
        task="sft",
        size="8B",
        tags=("llama", "mlx", "apple-silicon", "sft"),
        description="Llama 3.1 8B SFT on Apple Silicon via MLX (M2+ 16GB)",
        yaml_str="""\
base: mlx-community/Llama-3.1-8B-Instruct-4bit
task: sft
backend: mlx

data:
  train: ./data/train.jsonl
  format: chatml
  max_length: 2048

training:
  epochs: 3
  lr: 1e-4
  batch_size: 2
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "qwen3-8b-sft-mlx": RecipeMeta(
        model="mlx-community/Qwen3-8B-Instruct-4bit",
        task="sft",
        size="8B",
        tags=("qwen", "qwen3", "mlx", "apple-silicon", "sft"),
        description="Qwen 3 8B SFT on Apple Silicon via MLX (M2+ 16GB)",
        yaml_str="""\
base: mlx-community/Qwen3-8B-Instruct-4bit
task: sft
backend: mlx

data:
  train: ./data/train.jsonl
  format: chatml
  max_length: 2048

training:
  epochs: 3
  lr: 1e-4
  batch_size: 2
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "gemma3-9b-sft-mlx": RecipeMeta(
        model="mlx-community/gemma-3-9b-it-4bit",
        task="sft",
        size="9B",
        tags=("gemma", "gemma3", "mlx", "apple-silicon", "sft"),
        description="Gemma 3 9B SFT on Apple Silicon via MLX (M2+ 16GB)",
        yaml_str="""\
base: mlx-community/gemma-3-9b-it-4bit
task: sft
backend: mlx

data:
  train: ./data/train.jsonl
  format: chatml
  max_length: 2048

training:
  epochs: 3
  lr: 1e-4
  batch_size: 2
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "qwen3-8b-tools": RecipeMeta(
        model="Qwen/Qwen3-8B",
        task="sft",
        size="8B",
        tags=("qwen", "qwen3", "sft", "tool-calling", "agentic", "function-calling"),
        description="Qwen 3 8B tool-calling / function-calling SFT",
        yaml_str="""\
base: Qwen/Qwen3-8B
task: sft

data:
  train: ./data/tool_calling_train.jsonl
  format: tool-calling
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
    ),
    "llama4-scout-tools": RecipeMeta(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        task="sft",
        size="17B",
        tags=("llama", "llama4", "sft", "tool-calling", "agentic", "function-calling"),
        description="Llama 4 Scout 17B tool-calling / function-calling SFT",
        yaml_str="""\
base: meta-llama/Llama-4-Scout-17B-16E-Instruct
task: sft

data:
  train: ./data/tool_calling_train.jsonl
  format: tool-calling
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
    ),
    "llama3.1-8b-ipo": RecipeMeta(
        model="meta-llama/Llama-3.1-8B-Instruct",
        task="ipo",
        size="8B",
        tags=("llama", "ipo", "alignment", "regularized"),
        description="Llama 3.1 8B IPO regularized preference alignment",
        yaml_str="""\
base: meta-llama/Llama-3.1-8B-Instruct
task: ipo

data:
  train: ./data/preference_train.jsonl
  format: dpo
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  ipo_tau: 0.1

output: ./output
""",
    ),
    # ------------------------------------------------------------------
    # Multi-GPU Mastery recipes (v0.27.0)
    # ------------------------------------------------------------------
    "llama3-70b-fsdp2": RecipeMeta(
        model="meta-llama/Llama-3.1-70B-Instruct",
        task="sft",
        size="70B",
        tags=("llama", "sft", "fsdp2", "multi-gpu", "torch-compile"),
        description=(
            "Llama 3.1 70B SFT with FSDP2 full shard + torch.compile. "
            "Requires 8 x A100/H100 80GB."
        ),
        yaml_str="""\
base: meta-llama/Llama-3.1-70B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 2
  lr: 1e-4
  batch_size: 1
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  use_fsdp2_compile: true
  gradient_checkpointing: true

output: ./output
""",
    ),
    "qwen3-32b-zeropp": RecipeMeta(
        model="Qwen/Qwen3-32B",
        task="sft",
        size="32B",
        tags=("qwen", "sft", "zeropp", "deepspeed", "multi-gpu"),
        description=(
            "Qwen3 32B SFT with DeepSpeed ZeRO++ (quantized gradients + "
            "hierarchical partitioning). Launch with --deepspeed zero++."
        ),
        yaml_str="""\
base: Qwen/Qwen3-32B
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 2e-4
  batch_size: 1
  gradient_accumulation_steps: 16
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit
  gradient_checkpointing: true

output: ./output
""",
    ),
    "deepseek-v3-pipeline": RecipeMeta(
        model="deepseek-ai/DeepSeek-V3",
        task="sft",
        size="671B",
        tags=("deepseek", "sft", "pipeline", "multi-gpu", "moe"),
        description=(
            "DeepSeek V3 SFT scaffold with pipeline parallelism (4 stages). "
            "Pipeline execution wiring ships in v0.27.1."
        ),
        yaml_str="""\
base: deepseek-ai/DeepSeek-V3
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 1
  lr: 5e-5
  batch_size: 1
  gradient_accumulation_steps: 32
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  moe_lora: true
  parallelism: pipeline
  pipeline_stages: 4
  gradient_checkpointing: true

output: ./output
""",
    ),
    # ------------------------------------------------------------------
    # v0.31.0 Part A — Vision recipes (expand)
    # ------------------------------------------------------------------
    "llama3.2-vision-90b-sft": RecipeMeta(
        model="meta-llama/Llama-3.2-90B-Vision-Instruct",
        task="sft",
        size="90B",
        tags=("llama", "vision", "multimodal", "image", "large"),
        description="Llama 3.2 90B Vision multimodal SFT (8 x A100/H100 80GB)",
        yaml_str="""\
base: meta-llama/Llama-3.2-90B-Vision-Instruct
task: sft
modality: vision

data:
  train: ./data/vision_train.jsonl
  format: llava
  image_dir: ./data/images
  max_length: 4096

training:
  epochs: 1
  lr: 1e-5
  batch_size: 1
  gradient_accumulation_steps: 16
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit
  gradient_checkpointing: true

output: ./output
""",
    ),
    "pixtral-12b-sft": RecipeMeta(
        model="mistralai/Pixtral-12B-2409",
        task="sft",
        size="12B",
        tags=("mistral", "pixtral", "vision", "multimodal", "image"),
        description="Pixtral 12B vision-language SFT with LoRA",
        yaml_str="""\
base: mistralai/Pixtral-12B-2409
task: sft
modality: vision

data:
  train: ./data/vision_train.jsonl
  format: llava
  image_dir: ./data/images
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "qwen2-vl-7b-sft": RecipeMeta(
        model="Qwen/Qwen2-VL-7B-Instruct",
        task="sft",
        size="7B",
        tags=("qwen", "qwen2", "vision", "multimodal", "image"),
        description="Qwen2-VL 7B vision-language SFT",
        yaml_str="""\
base: Qwen/Qwen2-VL-7B-Instruct
task: sft
modality: vision

data:
  train: ./data/vision_train.jsonl
  format: sharegpt4v
  image_dir: ./data/images
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "qwen2-vl-72b-sft": RecipeMeta(
        model="Qwen/Qwen2-VL-72B-Instruct",
        task="sft",
        size="72B",
        tags=("qwen", "qwen2", "vision", "multimodal", "image", "large"),
        description="Qwen2-VL 72B vision-language SFT (multi-GPU recommended)",
        yaml_str="""\
base: Qwen/Qwen2-VL-72B-Instruct
task: sft
modality: vision

data:
  train: ./data/vision_train.jsonl
  format: sharegpt4v
  image_dir: ./data/images
  max_length: 4096

training:
  epochs: 1
  lr: 5e-6
  batch_size: 1
  gradient_accumulation_steps: 16
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit
  gradient_checkpointing: true

output: ./output
""",
    ),
    "internvl-2.5-8b-sft": RecipeMeta(
        model="OpenGVLab/InternVL2_5-8B",
        task="sft",
        size="8B",
        tags=("internvl", "vision", "multimodal", "image"),
        description="InternVL 2.5 8B vision-language SFT",
        yaml_str="""\
base: OpenGVLab/InternVL2_5-8B
task: sft
modality: vision

data:
  train: ./data/vision_train.jsonl
  format: llava
  image_dir: ./data/images
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "minicpm-v-2.6-sft": RecipeMeta(
        model="openbmb/MiniCPM-V-2_6",
        task="sft",
        size="8B",
        tags=("minicpm", "vision", "multimodal", "image", "edge"),
        description="MiniCPM-V 2.6 vision-language SFT (edge-friendly multimodal)",
        yaml_str="""\
base: openbmb/MiniCPM-V-2_6
task: sft
modality: vision

data:
  train: ./data/vision_train.jsonl
  format: llava
  image_dir: ./data/images
  max_length: 2048

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    # ------------------------------------------------------------------
    # v0.31.0 Part B — Audio recipes
    # ------------------------------------------------------------------
    "qwen2-audio-7b-sft": RecipeMeta(
        model="Qwen/Qwen2-Audio-7B-Instruct",
        task="sft",
        size="7B",
        tags=("qwen", "qwen2", "audio", "multimodal", "speech"),
        description="Qwen2-Audio 7B audio-language SFT",
        yaml_str="""\
base: Qwen/Qwen2-Audio-7B-Instruct
task: sft
modality: audio

data:
  train: ./data/audio_train.jsonl
  format: audio
  audio_dir: ./data/audio
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "seamlessm4t-v2-sft": RecipeMeta(
        model="facebook/seamless-m4t-v2-large",
        task="sft",
        size="2.3B",
        tags=("meta", "seamless", "audio", "translation", "multilingual"),
        description="SeamlessM4T v2 multilingual speech-to-text SFT",
        yaml_str="""\
base: facebook/seamless-m4t-v2-large
task: sft
modality: audio

data:
  train: ./data/audio_train.jsonl
  format: audio
  audio_dir: ./data/audio
  max_length: 1024

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "whisper-large-v3-ft": RecipeMeta(
        model="openai/whisper-large-v3",
        task="sft",
        size="1.5B",
        tags=("openai", "whisper", "audio", "asr", "transcription"),
        description="Whisper Large v3 ASR fine-tuning",
        yaml_str="""\
base: openai/whisper-large-v3
task: sft
modality: audio

data:
  train: ./data/audio_train.jsonl
  format: audio
  audio_dir: ./data/audio
  max_length: 448

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 8bit

output: ./output
""",
    ),
    # ------------------------------------------------------------------
    # v0.31.0 Part C — Reasoning recipes (R1 distills + Qwen3-Coder + Phi-4)
    # ------------------------------------------------------------------
    "r1-distill-qwen-1.5b-grpo": RecipeMeta(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        task="grpo",
        size="1.5B",
        tags=("deepseek", "r1", "qwen", "grpo", "reasoning", "small", "distill"),
        description="DeepSeek-R1-Distill Qwen 1.5B GRPO reasoning training",
        yaml_str="""\
base: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    ),
    "r1-distill-qwen-7b-grpo": RecipeMeta(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        task="grpo",
        size="7B",
        tags=("deepseek", "r1", "qwen", "grpo", "reasoning", "distill"),
        description="DeepSeek-R1-Distill Qwen 7B GRPO reasoning training",
        yaml_str="""\
base: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    ),
    "r1-distill-qwen-14b-grpo": RecipeMeta(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        task="grpo",
        size="14B",
        tags=("deepseek", "r1", "qwen", "grpo", "reasoning", "distill"),
        description="DeepSeek-R1-Distill Qwen 14B GRPO reasoning training",
        yaml_str="""\
base: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    ),
    "r1-distill-llama-70b-grpo": RecipeMeta(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        task="grpo",
        size="70B",
        tags=("deepseek", "r1", "llama", "grpo", "reasoning", "distill", "large"),
        description="DeepSeek-R1-Distill Llama 70B GRPO reasoning (multi-GPU)",
        yaml_str="""\
base: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 1
  lr: 5e-6
  batch_size: 1
  gradient_accumulation_steps: 16
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy
  gradient_checkpointing: true

output: ./output
""",
    ),
    "qwen3-coder-30b-sft": RecipeMeta(
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        task="sft",
        size="30B",
        tags=("qwen", "qwen3", "coder", "code", "sft", "moe"),
        description="Qwen3-Coder 30B (A3B MoE) code-specialist SFT",
        yaml_str="""\
base: Qwen/Qwen3-Coder-30B-A3B-Instruct
task: sft

data:
  train: ./data/code_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  moe_lora: true

output: ./output
""",
    ),
    "qwen3-30b-a3b-reasoning-grpo": RecipeMeta(
        model="Qwen/Qwen3-30B-A3B",
        task="grpo",
        size="30B",
        tags=("qwen", "qwen3", "grpo", "reasoning", "moe", "thinking"),
        description="Qwen3 30B-A3B GRPO reasoning training (MoE thinking model)",
        yaml_str="""\
base: Qwen/Qwen3-30B-A3B
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 8192

training:
  epochs: 3
  lr: 1e-5
  batch_size: 1
  gradient_accumulation_steps: 16
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy
  moe_lora: true
  gradient_checkpointing: true

output: ./output
""",
    ),
    "phi4-reasoning-grpo": RecipeMeta(
        model="microsoft/phi-4",
        task="grpo",
        size="14B",
        tags=("microsoft", "phi", "phi4", "grpo", "reasoning"),
        description="Phi-4 14B GRPO reasoning training",
        yaml_str="""\
base: microsoft/phi-4
task: grpo

data:
  train: ./data/reasoning_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    ),
    # ------------------------------------------------------------------
    # v0.31.0 Part D — Small / edge recipes
    # ------------------------------------------------------------------
    "qwen2.5-0.5b-sft": RecipeMeta(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        task="sft",
        size="0.5B",
        tags=("qwen", "qwen2.5", "sft", "tiny", "edge", "mobile"),
        description="Qwen 2.5 0.5B SFT (mobile / edge)",
        yaml_str="""\
base: Qwen/Qwen2.5-0.5B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 5e-4
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: 8bit

output: ./output
""",
    ),
    "qwen2.5-1.5b-sft": RecipeMeta(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        task="sft",
        size="1.5B",
        tags=("qwen", "qwen2.5", "sft", "tiny", "edge"),
        description="Qwen 2.5 1.5B SFT (edge-friendly)",
        yaml_str="""\
base: Qwen/Qwen2.5-1.5B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 3e-4
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: 8bit

output: ./output
""",
    ),
    "qwen2.5-3b-sft": RecipeMeta(
        model="Qwen/Qwen2.5-3B-Instruct",
        task="sft",
        size="3B",
        tags=("qwen", "qwen2.5", "sft", "small", "edge"),
        description="Qwen 2.5 3B SFT (small / edge)",
        yaml_str="""\
base: Qwen/Qwen2.5-3B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "gemma2-2b-sft": RecipeMeta(
        model="google/gemma-2-2b-it",
        task="sft",
        size="2B",
        tags=("gemma", "gemma2", "google", "sft", "small", "edge"),
        description="Gemma 2 2B SFT (edge-friendly)",
        yaml_str="""\
base: google/gemma-2-2b-it
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "smollm2-135m-sft": RecipeMeta(
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        task="sft",
        size="135M",
        tags=("smollm", "smollm2", "huggingface", "sft", "tiny", "edge", "mobile"),
        description="SmolLM2 135M SFT (ultra-tiny / mobile)",
        yaml_str="""\
base: HuggingFaceTB/SmolLM2-135M-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 5e-4
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: none

output: ./output
""",
    ),
    "smollm2-360m-sft": RecipeMeta(
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        task="sft",
        size="360M",
        tags=("smollm", "smollm2", "huggingface", "sft", "tiny", "edge", "mobile"),
        description="SmolLM2 360M SFT (tiny / mobile)",
        yaml_str="""\
base: HuggingFaceTB/SmolLM2-360M-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 5e-4
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: none

output: ./output
""",
    ),
    "smollm2-1.7b-sft": RecipeMeta(
        model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        task="sft",
        size="1.7B",
        tags=("smollm", "smollm2", "huggingface", "sft", "small", "edge"),
        description="SmolLM2 1.7B SFT (small / edge)",
        yaml_str="""\
base: HuggingFaceTB/SmolLM2-1.7B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 3e-4
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: 8bit

output: ./output
""",
    ),
    "phi3.5-mini-sft": RecipeMeta(
        model="microsoft/Phi-3.5-mini-instruct",
        task="sft",
        size="3.8B",
        tags=("microsoft", "phi", "phi3.5", "sft", "small", "edge"),
        description="Phi-3.5-mini 3.8B SFT (small / edge)",
        yaml_str="""\
base: microsoft/Phi-3.5-mini-instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 2e-4
  batch_size: auto
  lora:
    r: 8
    alpha: 16
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    # ------------------------------------------------------------------
    # v0.31.0 Part E — Domain specialists (medical / code / finance / math)
    # ------------------------------------------------------------------
    "biomistral-7b-sft": RecipeMeta(
        model="BioMistral/BioMistral-7B",
        task="sft",
        size="7B",
        tags=("biomistral", "mistral", "medical", "biomedical", "sft", "domain"),
        description="BioMistral 7B medical/biomedical domain SFT",
        yaml_str="""\
base: BioMistral/BioMistral-7B
task: sft

data:
  train: ./data/medical_train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 5e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "meditron-7b-sft": RecipeMeta(
        model="epfl-llm/meditron-7b",
        task="sft",
        size="7B",
        tags=("meditron", "epfl", "medical", "clinical", "sft", "domain"),
        description="Meditron 7B medical / clinical domain SFT",
        yaml_str="""\
base: epfl-llm/meditron-7b
task: sft

data:
  train: ./data/medical_train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 5e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "codellama-70b-sft": RecipeMeta(
        model="codellama/CodeLlama-70b-Instruct-hf",
        task="sft",
        size="70B",
        tags=("codellama", "code", "sft", "domain", "large", "deepspeed"),
        description="Code Llama 70B code-specialist SFT (multi-GPU)",
        yaml_str="""\
base: codellama/CodeLlama-70b-Instruct-hf
task: sft

data:
  train: ./data/code_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 1
  lr: 5e-6
  batch_size: 1
  gradient_accumulation_steps: 16
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit
  gradient_checkpointing: true

output: ./output
""",
    ),
    "codellama-13b-sft": RecipeMeta(
        model="codellama/CodeLlama-13b-Instruct-hf",
        task="sft",
        size="13B",
        tags=("codellama", "code", "sft", "domain"),
        description="Code Llama 13B code-specialist SFT",
        yaml_str="""\
base: codellama/CodeLlama-13b-Instruct-hf
task: sft

data:
  train: ./data/code_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "magicoder-7b-sft": RecipeMeta(
        model="ise-uiuc/Magicoder-S-DS-6.7B",
        task="sft",
        size="6.7B",
        tags=("magicoder", "deepseek", "code", "sft", "domain"),
        description="Magicoder S-DS 6.7B code-specialist SFT",
        yaml_str="""\
base: ise-uiuc/Magicoder-S-DS-6.7B
task: sft

data:
  train: ./data/code_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 5e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "nemotron-4-340b-sft": RecipeMeta(
        model="nvidia/Nemotron-4-340B-Instruct",
        task="sft",
        size="340B",
        tags=("nvidia", "nemotron", "sft", "large", "domain", "deepspeed"),
        description="Nemotron-4 340B SFT (massive multi-node deployment)",
        yaml_str="""\
base: nvidia/Nemotron-4-340B-Instruct
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 1
  lr: 5e-6
  batch_size: 1
  gradient_accumulation_steps: 32
  lora:
    r: 32
    alpha: 64
    target_modules: auto
  quantization: 4bit
  gradient_checkpointing: true

output: ./output
""",
    ),
    "llama2-13b-finance-sft": RecipeMeta(
        model="meta-llama/Llama-2-13b-hf",
        task="sft",
        size="13B",
        tags=("llama", "llama2", "finance", "financial", "sft", "domain"),
        description="Llama 2 13B finance-domain SFT (FinGPT-style starter recipe)",
        yaml_str="""\
base: meta-llama/Llama-2-13b-hf
task: sft

data:
  train: ./data/finance_train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    "mathstral-7b-sft": RecipeMeta(
        model="mistralai/Mathstral-7B-v0.1",
        task="sft",
        size="7B",
        tags=("mistral", "mathstral", "math", "stem", "sft", "domain"),
        description="Mathstral 7B math/STEM-specialist SFT",
        yaml_str="""\
base: mistralai/Mathstral-7B-v0.1
task: sft

data:
  train: ./data/math_train.jsonl
  format: auto
  max_length: 4096

training:
  epochs: 3
  lr: 5e-5
  batch_size: auto
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    ),
    # ------------------------------------------------------------------
    # v0.31.0 Part F — Multimodal reasoning
    # ------------------------------------------------------------------
    "llama3.2-vision-grpo": RecipeMeta(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        task="grpo",
        size="11B",
        tags=("llama", "vision", "multimodal", "grpo", "reasoning"),
        description="Llama 3.2 11B Vision GRPO multimodal reasoning training",
        yaml_str="""\
base: meta-llama/Llama-3.2-11B-Vision-Instruct
task: grpo
modality: vision

data:
  train: ./data/vision_reasoning_train.jsonl
  format: llava
  image_dir: ./data/images
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: 1
  gradient_accumulation_steps: 16
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy
  gradient_checkpointing: true

output: ./output
""",
    ),
    "pixtral-dpo": RecipeMeta(
        model="mistralai/Pixtral-12B-2409",
        task="dpo",
        size="12B",
        tags=("mistral", "pixtral", "vision", "multimodal", "dpo", "alignment"),
        description="Pixtral 12B DPO multimodal preference alignment",
        yaml_str="""\
base: mistralai/Pixtral-12B-2409
task: dpo
modality: vision

data:
  train: ./data/vision_preference_train.jsonl
  format: llava
  image_dir: ./data/images
  max_length: 4096

training:
  epochs: 3
  lr: 5e-6
  batch_size: 1
  gradient_accumulation_steps: 16
  lora:
    r: 16
    alpha: 32
    target_modules: auto
  quantization: 4bit
  dpo_beta: 0.1
  gradient_checkpointing: true

output: ./output
""",
    ),
}
