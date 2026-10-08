"""What a cloud run can actually carry (#1430).

``soup train --cloud`` ships the remote job as a rendered stub holding one
base64 blob: the config. The stub reaches the instance with that config and
nothing else, so two things can be lost between the local run and the remote
one, and both used to fail silently after the paid boot:

1. **The local inputs the config points at.** All 21 built-in templates set
   ``data.train: ./data/*.jsonl``. That file is not uploaded, so the remote
   ``soup train`` dies with ``FileNotFoundError: Data file not found``.
2. **The CLI overrides.** ``--reward-hack-detector`` and friends are applied to
   the loaded config before the cloud branch, then printed locally as enabled —
   but the stub embedded the *file's* text, so the override was absent from
   the remote run.

The fix for (2) is to embed the effective config rather than the file; that
lives in ``commands/train.py``. This module answers (1): which fields would
arrive on the remote machine pointing at nothing, so the run can refuse
before it costs anything.

Uploading the files instead is the other half of the fix and is a much larger
change (per-backend upload, size limits, an image or a volume). Until that
lands, refusing is the honest behaviour: a paid run that cannot possibly
succeed is worse than a plan-time error naming the field.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Tuple

from soup_cli.mcp_server.registry import PLAN_INPUT_FIELDS

if TYPE_CHECKING:  # pragma: no cover — typing only
    from soup_cli.config.schema import SoupConfig

#: Dotted paths the remote run reads. Sourced from
#: ``mcp_server.registry.PLAN_INPUT_FIELDS`` so the two lists cannot drift, with
#: ``data.train`` (handled entry by entry through the schema's own classifier)
#: and ``training.checkpoint_eval_tasks``, which ``soup train`` never reads,
#: removed.
_PATH_FIELDS: Tuple[str, ...] = tuple(
    field
    for field in PLAN_INPUT_FIELDS
    if field not in {"data.train", "training.checkpoint_eval_tasks"}
)


_LOCAL_ESCAPE_PREFIXES: Tuple[str, ...] = (
    "./",
    "../",
    "~/",        # home
    "/",         # absolute path
    "\\\\",      # Windows UNC
    "\\",        # Windows root-relative
)

_COMMA_SEPARATED_PLAN_INPUTS: frozenset[str] = frozenset(
    {"training.reward_fn"}
)


def _resolve_plan_input(cfg: "SoupConfig", dotted: str) -> object:
    """Read a dotted schema path; a missing intermediate model yields None."""
    value: object = cfg
    for part in dotted.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _entries_for_plan_input(value: object, dotted: str) -> List[str]:
    """Normalise a field value into the list of strings the loader opens.

    ``data.train`` and ``data.replay`` already arrive as a list, and
    ``training.reward_fn`` accepts a comma-separated ensemble. Every other
    field is a single string.
    """
    if isinstance(value, (list, tuple)):
        return [str(entry) for entry in value]
    if dotted in _COMMA_SEPARATED_PLAN_INPUTS and isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value)]


def _is_remote(value: str) -> bool:
    """True for a URI the loader fetches over the network."""
    return "://" in value


def _train_entry_is_local(entry: str) -> bool:
    """True when a ``data.train`` entry names a file the stub does not carry.

    Delegates to the schema's own classifier rather than repeating the
    suffix + ``"://"`` rule; a second copy here is how the two existing copies
    drifted apart in the first place. It is the right rule *for train
    entries*: ``teknium/OpenHermes-2.5`` classifies ``hub`` even though
    ``Path(...).suffix`` is ``.5``.
    """
    from soup_cli.config.schema import _classify_data_train_entry

    return _classify_data_train_entry(entry) == "local"


def _path_field_is_local(value: str) -> bool:
    """True when a directory/file field points at a local file the stub cannot
    carry.

    The rule is existence or escape-hatch, not shape. A Hugging Face hub id
    (owner/name, with a slash) is never a local path, so a model id such as
    sentence-transformers/all-mpnet-base-v2 (the schema's own example) now
    plans fine instead of being refused at exit 2. A plain name such as
    teacher is left alone on purpose: without a separator or a leading slash
    there is nothing to distinguish it from a dataset id, and writing ./teacher
    is the safe way to say this is local.

    ponytail: existence checked at plan time, not real-path-checked. A value
    whose parent directory was deleted falls through and reaches the remote
    run, which then fails; fixing that properly is walking every field the
    loader opens and giving each its own rule, as _train_entry_is_local does
    for data.train. Upgrade path: the os.path.exists call is the only part that
    can be wrong tonight, and it can be guarded with the field type once that
    rule is written once.
    """
    if _is_remote(value):
        return False
    if value.startswith(_LOCAL_ESCAPE_PREFIXES):
        return True
    if len(value) >= 2 and value[1] == ":" and value[0].isalpha():
        return True
    return os.path.exists(value)


def unshipped_inputs(cfg: "SoupConfig") -> List[str]:
    """Dotted paths whose local file the stub would not carry, sorted.

    data.train may be a single entry or a list, and each is classified
    separately: a Hub id in the list is fine and a ./data file beside it is
    not, so the whole field cannot be judged by its first entry. Every other
    field is walked as a dotted schema path so a list or a comma-separated
    value is judged entry by entry.
    """
    found: List[str] = []
    train = getattr(getattr(cfg, "data", None), "train", None)
    if train is not None:
        entries = [train] if isinstance(train, str) else list(train or [])
        if any(_train_entry_is_local(e) for e in entries if isinstance(e, str)):
            found.append("data.train")

    for dotted in _PATH_FIELDS:
        value = _resolve_plan_input(cfg, dotted)
        if value is None:
            continue
        entries = _entries_for_plan_input(value, dotted)
        if any(_path_field_is_local(str(entry)) for entry in entries):
            found.append(dotted)
    return sorted(set(found))


def refuse_unshipped_inputs(cfg: "SoupConfig") -> None:
    """Raise ``ValueError`` naming every input the remote machine would lack.

    Called at plan time, before any paid call, so the failure costs nothing to
    surface. The message names the Hub-dataset escape hatch because that is the
    one form of each field the remote run can actually resolve.
    """
    fields = unshipped_inputs(cfg)
    if not fields:
        return
    listed = ", ".join(fields)
    raise ValueError(
        f"--cloud ships only the config: {listed} would leave local files behind, "
        "so the remote run would fail after the instance boots. Use a Hugging Face "
        "Hub dataset id instead, or run locally. See issue #1430."
    )
