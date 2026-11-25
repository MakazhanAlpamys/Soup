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


def _at_both_levels(notes: dict[str, str]) -> dict[str, str]:
    """``notes`` for the key under ``training:`` and for the same key at the top level."""
    return {**{f"training.{key}": text for key, text in notes.items()}, **notes}


#: Names from Hugging Face ``TrainingArguments`` that look like an alias above but mean
#: something Soup has no setting for. They get this sentence instead of a "use X"
#: pointing at the nearest-looking field, which would run the job with a different
#: setting than the one the user asked for. Keyed by the path as written, so the
#: top level and ``training:`` give the same answer.
CONFIG_NOTES: dict[str, str] = _at_both_levels(
    {
        "per_device_eval_batch_size": (
            "Soup has no separate evaluation batch size; evaluation runs at the "
            "training batch size."
        ),
        "per_gpu_eval_batch_size": (
            "Soup has no separate evaluation batch size; evaluation runs at the "
            "training batch size."
        ),
        "eval_batch_size": (
            "Soup has no separate evaluation batch size; evaluation runs at the "
            "training batch size."
        ),
        "warmup_steps": (
            "Soup has no step-count warmup setting; warmup is a fraction of the "
            "training steps, set with training.warmup_ratio (0 to 0.5)."
        ),
        "warmup_step": (
            "Soup has no step-count warmup setting; warmup is a fraction of the "
            "training steps, set with training.warmup_ratio (0 to 0.5)."
        ),
    }
)
