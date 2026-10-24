"""Common parameter aliases from Hugging Face TrainingArguments, TRL and related tools (#2099).

Maps foreign or misplaced config key paths to canonical Soup schema field paths.
A test in ``tests/test_unknown_key_aliases.py`` verifies that every target field exists
in the schema, guarding against drift on field renames.
"""

from __future__ import annotations

#: Common aliases mapping foreign / HF TrainingArguments / TRL parameter names
#: to canonical Soup config field paths.
CONFIG_ALIASES: dict[str, str] = {
    # Under training:
    "training.learning_rate": "training.lr",
    "training.num_train_epochs": "training.epochs",
    "training.per_device_train_batch_size": "training.batch_size",
    "training.per_gpu_train_batch_size": "training.batch_size",
    "training.train_batch_size": "training.batch_size",
    "training.gradient_accumulation": "training.gradient_accumulation_steps",
    "training.grad_accum_steps": "training.gradient_accumulation_steps",
    "training.warmup_steps_ratio": "training.warmup_ratio",
    # Top-level (flat / root-level) misplacements:
    "learning_rate": "training.lr",
    "lr": "training.lr",
    "num_train_epochs": "training.epochs",
    "epochs": "training.epochs",
    "batch_size": "training.batch_size",
    "per_device_train_batch_size": "training.batch_size",
    "per_gpu_train_batch_size": "training.batch_size",
    "train_batch_size": "training.batch_size",
    "gradient_accumulation_steps": "training.gradient_accumulation_steps",
    "warmup_ratio": "training.warmup_ratio",
}
