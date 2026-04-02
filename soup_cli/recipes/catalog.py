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
}
