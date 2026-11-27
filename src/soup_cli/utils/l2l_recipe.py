"""The resolved L2L run recipe and the snapshot on disk.

``training.gradient_accumulation_steps`` is what the user asks for. The effective k is what the
run uses. Every consumer (the HF training arguments, the loss scale the HF loop derives from
them, the activation store capacity, the log) reads the effective k from one
:class:`L2LRecipe`, so none of them can disagree. The recipe is written to the output
directory and checked on resume, because a different k changes the optimizer-step boundaries.
"""

import json
import os
import re
from dataclasses import MISSING, asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional

RECIPE_NAME = "l2l_recipe.json"

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
    two_read_experts: bool = False
    grouped_expert_routes: bool = False
    # A run-root recipe covers checkpoints made after this step.
    start_step: int = 0
    route_preserving_mbr: bool = False
    mbr_group_size: Optional[int] = None
    mbr_split_backward: bool = False
    mbr_ordered_backward: bool = False


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
    mbr_group_size = getattr(tcfg, "l2l_mbr_group_size", None)
    if mbr_group_size is not None and mbr_group_size > effective_k:
        raise ValueError(
            f"training.l2l_mbr_group_size={mbr_group_size} exceeds the "
            f"effective L2L window of {effective_k} micro-batches"
        )
    return L2LRecipe(
        requested_k=requested,
        effective_k=effective_k,
        activation_codec=str(tcfg.l2l_activation_codec),
        auto_k=auto_k,
        two_read_experts=bool(getattr(tcfg, "l2l_two_read_experts", False)),
        grouped_expert_routes=bool(getattr(tcfg, "l2l_grouped_expert_routes", False)),
        route_preserving_mbr=getattr(tcfg, "l2l_route_preserving_mbr", False),
        mbr_group_size=mbr_group_size,
        mbr_split_backward=getattr(tcfg, "l2l_mbr_split_backward", False),
        mbr_ordered_backward=getattr(tcfg, "l2l_mbr_ordered_backward", False),
    )


def write_recipe(output_dir: "str | os.PathLike[str]", recipe: L2LRecipe | None) -> str:
    """Write the L2L recipe or an explicit record of ordinary training to a checkpoint."""
    from soup_cli.utils.paths import atomic_write_text

    path = os.path.join(os.fspath(output_dir), RECIPE_NAME)
    data = asdict(recipe) if recipe is not None else {"l2l": False}
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    return atomic_write_text(text, path, field="l2l recipe")


def load_recipe(output_dir: "str | os.PathLike[str]") -> Optional[L2LRecipe]:
    """The recipe saved in ``output_dir``, or None if there is none."""
    path = Path(output_dir) / RECIPE_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("l2l") is False:
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
    if saved is not None:
        _validate_recipe_match(saved, recipe)


def _validate_recipe_match(saved: L2LRecipe, recipe: L2LRecipe) -> None:
    for name in (
        "effective_k",
        "activation_codec",
        "two_read_experts",
        "grouped_expert_routes",
        "route_preserving_mbr",
        "mbr_group_size",
        "mbr_split_backward",
        "mbr_ordered_backward",
    ):
        before, now = getattr(saved, name), getattr(recipe, name)
        if before != now:
            raise ValueError(
                f"cannot resume: the checkpoint was trained with {name}={before!r}, but this "
                f"run has {name}={now!r}. Restore the old value in the config."
            )


def checkpoint_step(checkpoint: str | os.PathLike[str]) -> int | None:
    """Read the trainer's global step, falling back to the standard checkpoint name."""
    path = Path(os.path.realpath(checkpoint))
    try:
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = None
    step = state.get("global_step") if isinstance(state, dict) else None
    if type(step) is int and step >= 0:
        return step
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def validate_resume_source(
    checkpoint: str | os.PathLike[str], recipe: L2LRecipe | None,
) -> L2LRecipe | None:
    """Validate the checkpoint or its run-root recipe, then return the recipe to write.

    Accept legacy checkpoints with no covering recipe. When resuming a legacy checkpoint,
    the new run-root recipe starts at that checkpoint's step, so it cannot reclassify the
    checkpoint itself or earlier ones. A checkpoint-local recipe takes precedence.
    """
    path = Path(os.path.realpath(checkpoint))
    step = None
    if (path / RECIPE_NAME).is_file():
        saved = load_recipe(path)
    else:
        saved = load_recipe(path.parent)
        if saved is not None:
            step = checkpoint_step(path)
            if step is not None and step <= saved.start_step:
                saved = None
    if saved is None:
        if recipe is None:
            return None
        if step is None:
            step = checkpoint_step(path)
        return replace(recipe, start_step=step or 0)
    if recipe is None:
        raise ValueError(
            "cannot resume: the checkpoint was trained with training.l2l "
            f"(effective_k={saved.effective_k!r}, activation_codec={saved.activation_codec!r}), "
            "but this run has training.l2l off. Set training.l2l to true with the same settings."
        )
    _validate_recipe_match(saved, recipe)
    return replace(recipe, start_step=saved.start_step)


def build_l2l_recipe_callback(output_dir: str, recipe: L2LRecipe | None) -> Any:
    """Preserve the training mode and recipe beside each saved checkpoint."""
    from transformers import TrainerCallback

    class L2LRecipeCallback(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if args.should_save:
                checkpoint = Path(args.output_dir or output_dir) / f"checkpoint-{state.global_step}"
                write_recipe(checkpoint, recipe)

    return L2LRecipeCallback()
