"""Which ``*.safetensors`` files make up a Hugging Face checkpoint directory (#1806).

Mistral's Hub repositories ship the same weights twice: the Hugging Face shards
with ``model.safetensors.index.json``, and one ``consolidated.safetensors`` with
mistral-inference parameter names. transformers follows the index and never
opens the consolidated file. Soup's file filters took every top-level
``*.safetensors`` as part of the checkpoint, so on such a repo the parameter
count doubled (hardware-fit pre-flight, ``soup profile``, ``soup cost``, batch
estimation), the weight download fetched both copies, the cache-snapshot copy
held both, ``soup spectrum scan`` scored both name sets, and the layer-stream
sharder kept every consolidated tensor resident.

This module is the one answer to "which files are the checkpoint", read by
every one of those sites:

- with an index, exactly the files its ``weight_map`` names, as transformers loads;
- without one, ``model.safetensors`` if present, else the ``model-*-of-*.safetensors``
  shards;
- otherwise every ``*.safetensors`` except ``consolidated*.safetensors``, so an
  unconventionally named single file still counts and the mistral-inference copy
  never does.

No torch here: the helper lists names and reads one small JSON file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
_SHARD_RE = re.compile(r"^model-\d+-of-\d+\.safetensors$")
_CONSOLIDATED_PREFIX = "consolidated"
_MAX_INDEX_BYTES = 64 * 1024 * 1024  # a 70B index is ~70 KB; bound a crafted file

#: ``allow_patterns`` for a Hub download's first call: the Hugging Face spellings,
#: which are the ``single`` and ``shards`` rules below, plus the JSON files (the index
#: among them). Never ``*.safetensors``: that is what fetched Mistral's consolidated
#: copy beside the shards (29.0 GB for a 14.5 GB model).
HUB_WEIGHT_PATTERNS_WITHOUT_INDEX: Tuple[str, ...] = (
    SINGLE_FILE,
    "model-*-of-*.safetensors",
)
#: The ``fallback`` rule as Hub patterns: every ``*.safetensors`` except
#: ``consolidated*``, so a repository whose only weight file has another name is
#: fetched just as the local rule would count it (#1806 review).
HUB_FALLBACK_ALLOW_PATTERNS: Tuple[str, ...] = ("*.safetensors",)
HUB_FALLBACK_IGNORE_PATTERNS: Tuple[str, ...] = (
    f"{_CONSOLIDATED_PREFIX}*.safetensors",
    f"*/{_CONSOLIDATED_PREFIX}*.safetensors",
)


@dataclass(frozen=True)
class CheckpointFiles:
    """The checkpoint's safetensors files, by basename, sorted."""

    weights: Tuple[str, ...]
    #: ``*.safetensors`` present beside the checkpoint that are not part of it.
    ignored: Tuple[str, ...]
    #: Files the index names that the directory lacks.
    missing: Tuple[str, ...]
    #: How the set was chosen: ``index`` / ``single`` / ``shards`` / ``fallback`` / ``none``.
    rule: str

    def why_empty(self, directory: str) -> str:
        """A refusal message for a directory with no checkpoint file, naming what is there."""
        if self.ignored:
            names = ", ".join(self.ignored)
            return (
                f"no Hugging Face checkpoint in {directory}: the only safetensors file(s) "
                f"there, {names}, are not part of one (a consolidated.safetensors holds the "
                f"mistral-inference copy of the weights, which transformers does not load). "
                f"A checkpoint is model.safetensors, or model-*-of-*.safetensors shards with "
                f"model.safetensors.index.json."
            )
        return f"no .safetensors weight files found in {directory}"


def _is_safetensors(name: str) -> bool:
    return name.endswith(".safetensors")


def select_checkpoint_files(
    names: Iterable[str],
    index_weight_map: Optional[Mapping[str, str]] = None,
) -> CheckpointFiles:
    """Choose the checkpoint's files from a directory's top-level file names.

    ``index_weight_map`` is the parsed ``weight_map`` of ``model.safetensors.index.json``
    when the directory has one (``None`` otherwise).
    """
    present = sorted({name for name in names if _is_safetensors(name)})
    if index_weight_map is not None:
        named = sorted({str(value) for value in index_weight_map.values()})
        present_set = set(present)
        weights = tuple(name for name in named if name in present_set)
        missing = tuple(name for name in named if name not in present_set)
        ignored = tuple(name for name in present if name not in set(named))
        return CheckpointFiles(weights, ignored, missing, "index")
    if SINGLE_FILE in present:
        rest = tuple(name for name in present if name != SINGLE_FILE)
        return CheckpointFiles((SINGLE_FILE,), rest, (), "single")
    shards = tuple(name for name in present if _SHARD_RE.match(name))
    if shards:
        rest = tuple(name for name in present if not _SHARD_RE.match(name))
        return CheckpointFiles(shards, rest, (), "shards")
    consolidated = tuple(name for name in present if name.startswith(_CONSOLIDATED_PREFIX))
    others = tuple(name for name in present if not name.startswith(_CONSOLIDATED_PREFIX))
    return CheckpointFiles(others, consolidated, (), "fallback" if others else "none")


def read_index_weight_map(directory: str) -> Optional[Mapping[str, str]]:
    """The index's ``weight_map`` (basename values), or ``None`` without an index.

    Raises ``ValueError`` for an index that is present but unreadable, so a
    corrupt index is a named failure rather than a silent fall-through to a
    filter that would count the consolidated copy again.
    """
    path = os.path.join(directory, INDEX_FILE)
    if not os.path.isfile(path):
        return None
    if os.path.getsize(path) > _MAX_INDEX_BYTES:
        raise ValueError(f"{INDEX_FILE} in {directory} exceeds {_MAX_INDEX_BYTES} bytes")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{INDEX_FILE} in {directory} is not valid JSON: {exc}") from exc
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{INDEX_FILE} in {directory} has no weight_map")
    # transformers stores basenames; tolerate a relative path by keeping its basename.
    return {str(key): os.path.basename(str(value)) for key, value in weight_map.items()}


def checkpoint_files(directory: str) -> CheckpointFiles:
    """The checkpoint's files in ``directory`` (top level only, names not resolved)."""
    names = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if _is_safetensors(entry.name):
                names.append(entry.name)
    return select_checkpoint_files(names, read_index_weight_map(directory))


def hub_allow_patterns(index_weight_map: Optional[Mapping[str, str]]) -> list[str]:
    """``allow_patterns`` for the weight download: the indexed files, or the HF spellings.

    Never ``*.safetensors``: that pattern is what fetched Mistral's consolidated
    copy beside the shards (29.0 GB for a 14.5 GB model).
    """
    if index_weight_map is not None:
        return sorted({os.path.basename(str(value)) for value in index_weight_map.values()})
    return list(HUB_WEIGHT_PATTERNS_WITHOUT_INDEX)


def hub_second_call(selected: CheckpointFiles) -> Optional[Tuple[list[str], list[str]]]:
    """What a second download call must ask for, after the first fetched the HF spellings.

    ``(allow_patterns, ignore_patterns)``, or ``None`` when the first call already
    holds the checkpoint. The local rule and the download agree by construction:
    with an index, the files it names that are not there yet (an index naming
    ``weights-part1.safetensors``); without one, when neither ``model.safetensors``
    nor shards came, the ``fallback`` rule's own patterns, every ``*.safetensors``
    except ``consolidated*`` (#1806 review).
    """
    if selected.rule == "index":
        if selected.missing:
            return list(selected.missing), []
        return None
    if selected.rule == "none":
        return list(HUB_FALLBACK_ALLOW_PATTERNS), list(HUB_FALLBACK_IGNORE_PATTERNS)
    return None


__all__ = [
    "HUB_FALLBACK_ALLOW_PATTERNS",
    "HUB_FALLBACK_IGNORE_PATTERNS",
    "HUB_WEIGHT_PATTERNS_WITHOUT_INDEX",
    "INDEX_FILE",
    "SINGLE_FILE",
    "CheckpointFiles",
    "checkpoint_files",
    "hub_allow_patterns",
    "hub_second_call",
    "read_index_weight_map",
    "select_checkpoint_files",
]
