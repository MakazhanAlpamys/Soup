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
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:  # pragma: no cover — typing only
    from soup_cli.config.schema import SoupConfig

#: Fields whose value is a path the remote run reads. Mirrors the subset of
#: `mcp_server.registry.PLAN_INPUT_FIELDS` that is a file the loader opens
#: rather than a model or suite id.
_PATH_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("data", "image_dir"),
    ("data", "audio_dir"),
    ("data", "video_dir"),
    ("data", "replay"),
    ("data", "tokenized_path"),
    ("data", "forget_set"),
    ("data", "retain_set"),
    ("training", "reward_fn"),
    ("training", "prm_reward"),
    ("training", "minillm_pretrain_anchor_path"),
    ("training", "mole_task_adapters"),
    ("training", "ra_dit_retriever_model"),
    ("eval", "custom_tasks"),
)


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
    """True for a directory/file field whose value names a local path.

    ponytail: shape-based, not existence-based. A bare name with no separator
    and no suffix (``images``) is not detected — it is genuinely ambiguous
    with a Hub id, and the escape hatch is to write ``./images``. Upgrade path:
    walk the fields the loader opens and check each with its own rule, as
    ``_train_entry_is_local`` does for train.
    """
    text = value.strip()
    if _is_remote(text):
        return False
    if text.startswith(("./", "../", ".\\", "..\\", "~", "/", "\\")):
        return True
    return bool(os.sep in text or "/" in text or "\\" in text or Path(text).suffix)


def unshipped_inputs(cfg: "SoupConfig") -> List[str]:
    """Dotted paths whose local file the stub would not carry, sorted.

    ``data.train`` may be a single entry or a list, and each is classified
    separately: a Hub id in the list is fine and a ``./data`` file beside it is
    not, so the whole field cannot be judged by its first entry.
    """
    found: List[str] = []
    train = getattr(getattr(cfg, "data", None), "train", None)
    if train is not None:
        entries = [train] if isinstance(train, str) else list(train or [])
        if any(_train_entry_is_local(e) for e in entries if isinstance(e, str)):
            found.append("data.train")
    for section, field in _PATH_FIELDS:
        value = getattr(getattr(cfg, section, None), field, None)
        if isinstance(value, str) and _path_field_is_local(value):
            found.append(f"{section}.{field}")
    # `training.eval_gate.suite` is a nested path, walked explicitly.
    gate = getattr(getattr(cfg, "training", None), "eval_gate", None)
    for field in ("suite", "baseline"):
        value = getattr(gate, field, None)
        if isinstance(value, str) and _path_field_is_local(value):
            found.append(f"training.eval_gate.{field}")
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
        f"--cloud ships only the config: {listed} point at local files that "
        "are not uploaded, so the remote run would fail after the instance "
        "boots. Use a Hugging Face Hub dataset id instead, or run locally. "
        "See issue #1430."
    )
