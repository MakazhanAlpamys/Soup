"""The resolved L2L run recipe and the snapshot on disk.

``training.gradient_accumulation_steps`` is what the user asks for. The effective k is what the
run uses. Every consumer (the HF training arguments, the loss scale the HF loop derives from
them, the activation store capacity, the log) reads the effective k from one
:class:`L2LRecipe`, so none of them can disagree. The recipe is recorded with checkpoints
and checked on resume, because a different k changes the optimizer-step boundaries.
"""

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import MISSING, asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional

RECIPE_NAME = "l2l_recipe.json"
_ORDINARY_CHECKPOINT = {"l2l": False}
_CHECKPOINT_NAME = re.compile(r"checkpoint-(\d+)")

#: Codecs the launch accepts. ``fp8_e4m3`` stays closed until its gradients are checked
#: against the accumulation baseline; the trainer and planner accept it through the API
#: (``L2LState``), but a ``soup train`` config cannot reach it while it is absent here.
AVAILABLE_CODECS = ("none",)


@dataclass(frozen=True)
class L2LRecipe:
    requested_k: int
    effective_k: int
    activation_codec: str
    auto_k: bool
    #: The global step at which this run began: 0 for a fresh run, the resumed checkpoint's step
    #: otherwise. A recipe in a run directory covers only the checkpoints after it; an older
    #: one belongs to the run that wrote it (a recipe without the field counts as 0).
    start_step: int = 0


def validate_l2l_options(tcfg: Any) -> None:
    """Check options that can be refused before loading the model."""
    codec = str(tcfg.l2l_activation_codec)
    if codec not in AVAILABLE_CODECS:
        raise ValueError(
            f"training.l2l_activation_codec={codec!r} is not available yet: the launch "
            f"supports {', '.join(repr(c) for c in AVAILABLE_CODECS)}. Set it to 'none'."
        )
    spill_dir = getattr(tcfg, "l2l_spill_dir", None)
    if spill_dir is not None:
        # Checked here, before the trainer's `train` sweeps the directory: the spill may live
        # only under $HOME, $CWD or $TMPDIR (realpath + commonpath, see utils/l2l_spill.py).
        from soup_cli.utils.l2l_spill import _outside_allowed_roots

        reason = _outside_allowed_roots(spill_dir)
        if reason is not None:
            raise ValueError(
                f"training.l2l_spill_dir is refused: {reason}. Set it to a directory under "
                "$HOME, the working directory or the temp directory."
            )
    if tcfg.l2l_auto_k:
        raise ValueError(
            "training.l2l_auto_k is not available yet for this model: the launch cannot plan "
            "the window automatically. Unset training.l2l_auto_k and set "
            "training.gradient_accumulation_steps to the k you want: L2L runs that many "
            "micro-batches per layer load."
        )


def refuse_distributed_l2l(distributed: bool) -> None:
    """Refuse ``training.l2l`` on a distributed layer stream.

    The data-parallel stream trainer (#3267) averages the adapter gradients between backward
    and clipping in its own ``training_step``; the L2L step runs its own backward, so the
    average would be skipped and each rank would train an unsynchronised copy.
    """
    if distributed:
        raise ValueError(
            "training.l2l cannot run on a distributed layer stream: the L2L step runs its "
            "own backward, so the adapter-gradient average between ranks would be skipped. "
            "Run one process, or set training.l2l to false."
        )


def resolve_l2l_recipe(tcfg: Any, *, effective_k: int | None = None) -> L2LRecipe:
    """Use the automatic plan when requested, otherwise keep the configured k."""
    validate_l2l_options(tcfg)
    requested = int(tcfg.gradient_accumulation_steps)
    auto_k = bool(tcfg.l2l_auto_k)
    if not auto_k:
        effective_k = requested
    if type(effective_k) is not int or not 1 <= effective_k <= requested:
        raise ValueError("training.l2l_auto_k requires a resolved, RAM-fitting effective k")
    return L2LRecipe(
        requested_k=requested,
        effective_k=effective_k,
        activation_codec=str(tcfg.l2l_activation_codec),
        auto_k=auto_k,
    )


def write_recipe(output_dir: "str | os.PathLike[str]", recipe: Optional[L2LRecipe]) -> str:
    """Write a recipe or ordinary checkpoint record; return its path."""
    from soup_cli.utils.paths import atomic_write_text

    path = os.path.join(os.fspath(output_dir), RECIPE_NAME)
    data = _ORDINARY_CHECKPOINT if recipe is None else asdict(recipe)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    return atomic_write_text(text, path, field="l2l recipe")


def load_recipe(output_dir: "str | os.PathLike[str]") -> Optional[L2LRecipe]:
    """The saved recipe, or None for a missing or explicitly ordinary record."""
    path = Path(output_dir) / RECIPE_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and len(data) == 1 and data.get("l2l") is False:
            return None
        values = {}
        for field in fields(L2LRecipe):
            if field.name in data:
                values[field.name] = data[field.name]
            elif field.default is not MISSING:
                values[field.name] = field.default
            else:
                raise ValueError(f"recipe is missing required field {field.name!r}")
        return L2LRecipe(**values)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path} is not a valid L2L recipe: {exc}") from exc


def validate_resume(output_dir: "str | os.PathLike[str]", recipe: L2LRecipe) -> None:
    """Refuse to resume a run whose saved recipe differs from ``recipe``.

    No saved recipe (a run from before the recipe existed) is accepted.
    """
    saved = load_recipe(output_dir)
    if saved is None:
        return
    for name in ("effective_k", "activation_codec"):
        before, now = getattr(saved, name), getattr(recipe, name)
        if before != now:
            raise ValueError(
                f"cannot resume: the checkpoint was trained with {name}={before!r}, but this "
                f"run has {name}={now!r}. Restore the old value in the config."
            )


def checkpoint_step(checkpoint: "str | os.PathLike[str]") -> Optional[int]:
    """The global step of a checkpoint directory, or None when it cannot be read.

    ``trainer_state.json`` wins; the ``checkpoint-N`` directory name is the fallback.
    """
    path = os.path.realpath(os.fspath(checkpoint))
    state = os.path.join(path, "trainer_state.json")
    try:
        with open(state, encoding="utf-8") as handle:
            step = json.load(handle).get("global_step")
        if type(step) is int and step >= 0:
            return step
    except (OSError, ValueError, AttributeError):
        pass
    match = _CHECKPOINT_NAME.fullmatch(os.path.basename(path))
    return int(match.group(1)) if match else None


def _source_recipe(checkpoint: "str | os.PathLike[str]") -> Optional[L2LRecipe]:
    """The recipe that covers ``checkpoint``, or None for a checkpoint without L2L provenance.

    A checkpoint's own record is authoritative, including an explicit ordinary marker.
    A recipe in the parent run directory covers the checkpoint only when its step exceeds the
    ``start_step``: the run that wrote the recipe began at ``start_step`` (a legacy checkpoint
    resumed with ``training.l2l``), so that checkpoint and the older ones predate it. A step
    that cannot be read counts as covered.
    """
    path = os.path.realpath(os.fspath(checkpoint))
    if os.path.isfile(os.path.join(path, RECIPE_NAME)):
        return load_recipe(path)
    parent = os.path.dirname(path)
    saved = load_recipe(parent)
    if saved is None:
        return None
    step = checkpoint_step(path)
    if step is not None and step <= saved.start_step:
        return None
    return saved


def write_run_recipe(output_dir: "str | os.PathLike[str]", recipe: L2LRecipe) -> None:
    """Preserve retained checkpoints' provenance before publishing a new run recipe."""
    output = Path(os.path.realpath(output_dir))
    if load_recipe(output) == recipe:
        return
    if output.is_dir():
        for checkpoint in output.iterdir():
            if not checkpoint.is_dir() or (checkpoint / RECIPE_NAME).is_file():
                continue
            if not (
                _CHECKPOINT_NAME.fullmatch(checkpoint.name)
                or (checkpoint / "trainer_state.json").is_file()
            ):
                continue
            # A link can refer to a checkpoint in another output directory.
            if os.path.normcase(os.path.realpath(checkpoint)) != os.path.normcase(checkpoint):
                continue
            write_recipe(checkpoint, _source_recipe(checkpoint))
    write_recipe(output_dir, recipe)


@contextmanager
def checkpoint_recipe_context(
    trainer: Any, output_dir: str, recipe: Optional[L2LRecipe]
) -> Iterator[None]:
    """Record the training mode with each checkpoint in an output that uses L2L."""
    if recipe is None and not (Path(output_dir) / RECIPE_NAME).is_file():
        yield
        return

    from transformers import TrainerCallback
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    class CheckpointRecipeCallback(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if args.should_save:
                checkpoint = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
                if checkpoint.is_dir():
                    write_recipe(checkpoint, recipe)

    callback = CheckpointRecipeCallback()
    trainer.add_callback(callback)
    try:
        yield
    finally:
        trainer.remove_callback(callback)


def validate_resume_source(
    checkpoint: "str | os.PathLike[str]", recipe: Optional[L2LRecipe]
) -> Optional[L2LRecipe]:
    """Refuse a resume whose SOURCE checkpoint was trained under another L2L recipe.

    ``checkpoint`` is the path handed to ``resume_from_checkpoint``; it may sit in another run
    directory than this launch's output, so its own recipe is the one that counts. ``recipe`` is
    this launch's, or None when ``training.l2l`` is off. Returns the recipe to write for this
    run (None when ``training.l2l`` is off): ``recipe`` with ``start_step`` set to where this
    run begins, which is the covering recipe's ``start_step`` when there is one and the
    checkpoint's own step for a checkpoint without L2L provenance.

    - A source recipe that differs in ``effective_k`` or ``activation_codec`` is refused.
    - A source with a recipe resumed with ``training.l2l`` off is refused.
    - A source without a recipe is accepted with either setting: it is a run without L2L (or
      from before the recipe existed), and the optimizer-step boundaries come from
      ``gradient_accumulation_steps`` in both modes, because ``l2l_auto_k`` is refused.
    """
    saved = _source_recipe(checkpoint)
    if saved is None:
        if recipe is None:
            return None
        return replace(recipe, start_step=checkpoint_step(checkpoint) or 0)
    if recipe is None:
        raise ValueError(
            f"cannot resume: the checkpoint was trained with training.l2l "
            f"(effective_k={saved.effective_k!r}, activation_codec={saved.activation_codec!r}), "
            "but this run has training.l2l off. Set training.l2l to true with the same "
            "settings."
        )
    for name in ("effective_k", "activation_codec"):
        before, now = getattr(saved, name), getattr(recipe, name)
        if before != now:
            raise ValueError(
                f"cannot resume: the checkpoint was trained with {name}={before!r}, but this "
                f"run has {name}={now!r}. Restore the old value in the config."
            )
    return replace(recipe, start_step=saved.start_step)
