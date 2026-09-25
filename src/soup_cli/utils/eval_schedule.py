"""Evaluate the validation split that every task wrapper receives (#1223).

``data.val_split`` (default 0.1) moves that share of the rows out of training,
and the task wrappers hand the result to their trainer as ``eval_dataset``. No
wrapper set ``eval_strategy``, whose Hugging Face default is ``"no"``, so
``Trainer.evaluate()`` never ran. The rows were withheld and used for nothing,
and the validation-loss sinks #713 added (the metrics column,
``TrainEvent.val_loss``, the display row) never received a value on the
transformers backend.

The policy this module encodes:

* A trainer that receives a non-empty validation split evaluates it at the end
  of every epoch, or every ``training.eval_steps`` optimizer steps when that is
  set. Hugging Face also evaluates once more at the last step when that step
  is not a multiple of ``eval_steps``, so a value larger than the run still
  evaluates once.
* The evaluation batch is the resolved per-device TRAIN batch. Hugging Face's
  default of 8 would run out of memory on a card sized to train at 1.
* Generation-based tasks (:data:`GENERATION_EVAL_TASKS`) do not evaluate by
  default, because an evaluation there means generating completions. They do
  not withhold rows either: ``data.val_split`` is ignored for them, with a
  notice, unless ``training.eval_steps`` asks for evaluation.
* ``backend: mlx`` is untouched. mlx-lm runs its own validation loop over the
  split (#739), so the loader keeps withholding it there.

Import-light on purpose: ``config/schema.py`` imports the task sets, so nothing
here may pull torch, transformers or the data stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from soup_cli.config.schema import DataConfig, SoupConfig

#: Tasks where an evaluation pass means generating completions, so it costs as
#: much per row as training does. They evaluate only when asked to.
GENERATION_EVAL_TASKS: frozenset[str] = frozenset({"grpo", "ppo", "online_dpo"})

#: Tasks that refuse ``training.eval_steps`` at config load, each with the
#: reason. The setting would be read by nothing on them.
EVAL_STEPS_UNSUPPORTED_TASKS: dict[str, str] = {
    "ppo": "TRL's PPO training loop never runs an evaluation pass",
    "online_dpo": (
        "TRL's online-DPO trainer has no evaluation step for prompt-only rows "
        "(its evaluate() fails on them)"
    ),
    "unlearn": (
        "unlearning trains on data.forget_set and data.retain_set and never "
        "evaluates a validation split"
    ),
}


def evaluates_by_default(task: str) -> bool:
    """Whether a run of ``task`` evaluates its validation split unasked."""
    return task not in GENERATION_EVAL_TASKS


def holds_out_validation(cfg: SoupConfig) -> bool:
    """Whether the loader should carve ``data.val_split`` out of the training rows.

    A split that nothing evaluates is not carved out: those rows train instead.
    """
    if cfg.data.val_split <= 0:
        return False
    if getattr(cfg, "backend", None) == "mlx":
        return True
    return evaluates_by_default(cfg.task) or cfg.training.eval_steps is not None


def loader_data_config(cfg: SoupConfig) -> DataConfig:
    """The ``DataConfig`` to hand ``load_dataset`` for this run.

    ``cfg.data`` itself, unless the run would withhold rows that nothing
    evaluates; then a copy with ``val_split`` at 0, so every row trains. A
    dataset's own validation split (an HF-hub ``validation`` split) is not
    withheld from training, so it still arrives either way.
    """
    if cfg.data.val_split <= 0 or holds_out_validation(cfg):
        return cfg.data
    return cfg.data.model_copy(update={"val_split": 0.0})


def effective_val_split(cfg: SoupConfig) -> float:
    """The fraction of the rows this run actually withholds from training."""
    return loader_data_config(cfg).val_split


def validation_notice(cfg: SoupConfig) -> Optional[str]:
    """One line explaining an ignored ``data.val_split``, or ``None``."""
    if cfg.data.val_split <= 0 or holds_out_validation(cfg):
        return None
    notice = (
        f"data.val_split is ignored for task={cfg.task} because evaluation "
        "means generation, so every row trains."
    )
    if cfg.task not in EVAL_STEPS_UNSUPPORTED_TASKS:
        notice += " Set training.eval_steps to evaluate on a held-out split."
    return notice


def _has_rows(dataset: Any) -> bool:
    if dataset is None:
        return False
    try:
        return len(dataset) > 0
    except TypeError:  # an iterable dataset of unknown length
        return True


def training_eval_kwargs(
    cfg: SoupConfig, eval_dataset: Any, *, batch_size: int
) -> dict[str, Any]:
    """Evaluation kwargs for any ``TrainingArguments`` subclass.

    ``eval_dataset`` is what the wrapper is about to hand its trainer, after its
    own row preparation, and ``batch_size`` the resolved per-device train batch
    (never ``"auto"``). Raises ``ValueError`` when ``training.eval_steps`` asks
    for an evaluation that has no rows to run on.
    """
    eval_steps = getattr(cfg.training, "eval_steps", None)
    kwargs: dict[str, Any] = {"per_device_eval_batch_size": int(batch_size)}
    if not _has_rows(eval_dataset):
        if eval_steps is not None:
            if cfg.data.val_split > 0:
                why = "every validation row was dropped while the dataset was prepared"
            else:
                why = (
                    "data.val_split is 0 and the dataset carries no validation "
                    "split of its own"
                )
            raise ValueError(
                f"training.eval_steps={eval_steps} asks for evaluation, but this "
                f"run has no validation rows: {why}. Set data.val_split above 0, "
                "or remove training.eval_steps (#1223)."
            )
        kwargs["eval_strategy"] = "no"
        return kwargs
    if eval_steps is not None:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = int(eval_steps)
    elif evaluates_by_default(cfg.task):
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "no"
    return kwargs
