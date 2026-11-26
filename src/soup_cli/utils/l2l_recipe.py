"""The resolved L2L run recipe and the snapshot on disk.

``training.gradient_accumulation_steps`` is what the user asks for. The effective k is what the
run uses. Every consumer (the HF training arguments, the loss scale the HF loop derives from
them, the activation store capacity, the log) reads the effective k from one
:class:`L2LRecipe`, so none of them can disagree. The recipe is written to the output
directory and checked on resume, because a different k changes the optimizer-step boundaries.
"""

import json
import os
from dataclasses import MISSING, asdict, dataclass, fields
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


def validate_l2l_options(tcfg: Any) -> None:
    """Check options that can be refused before loading the model."""
    codec = str(tcfg.l2l_activation_codec)
    if codec not in AVAILABLE_CODECS:
        raise ValueError(
            f"training.l2l_activation_codec={codec!r} is not available yet: the launch "
            f"supports {', '.join(repr(c) for c in AVAILABLE_CODECS)}. Set it to 'none'."
        )
    if not tcfg.l2l_auto_k:
        return
    if tcfg.stream_expert_layout != "per_expert" or type(tcfg.batch_size) is not int:
        raise ValueError(
            "training.l2l_auto_k requires DeepSeek-V3 per_expert streaming and a fixed batch_size"
        )
    if tcfg.l2l_spill_dir is not None:
        raise ValueError("training.l2l_auto_k uses RAM only; unset training.l2l_spill_dir")
    raise ValueError(
        "training.l2l_auto_k plans the window of a MoE model, and L2L on a MoE model is not "
        "available in this version. Unset training.l2l_auto_k."
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


def write_recipe(output_dir: "str | os.PathLike[str]", recipe: L2LRecipe) -> str:
    """Write the recipe to ``<output_dir>/l2l_recipe.json``; returns the path."""
    from soup_cli.utils.paths import atomic_write_text

    path = os.path.join(os.fspath(output_dir), RECIPE_NAME)
    text = json.dumps(asdict(recipe), indent=2, sort_keys=True) + "\n"
    return atomic_write_text(text, path, field="l2l recipe")


def load_recipe(output_dir: "str | os.PathLike[str]") -> Optional[L2LRecipe]:
    """The recipe saved in ``output_dir``, or None if there is none."""
    path = Path(output_dir) / RECIPE_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
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
