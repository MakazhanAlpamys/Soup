"""``quantization: auto`` of the loop's sft recipe: ``none`` when the base fits in free VRAM.

The recipe's ``quantization`` is ``auto`` by default. The train stage resolves it right before
it writes the soup.yaml and starts ``soup train``, so the figure is the card's free memory at
that moment:

- need = parameter count x 2 bytes (bf16) x 5 (weights, LoRA, optimizer, gradients and
  activations, with room to spare); free = free VRAM of the CUDA device, measured in a
  short-lived subprocess (``free_vram_bytes``);
- CUDA available and need <= free: ``none``; otherwise ``4bit`` (the safe side);
- ``4bit`` and ``none`` written in the recipe are explicit and never measured or overridden.

The parameter count is read without loading weights: the safetensors headers of a local model
directory, else the Hub's header-only metadata (skipped when ``HF_HUB_OFFLINE`` or
``TRANSFORMERS_OFFLINE`` is set), else what the Hugging Face cache holds of the repo (its cached
safetensors headers, else its cached ``config.json``; ``model_shape.read_model_shape`` with
``allow_network=False``), else unknown, which is also ``4bit``. A size guessed from the repo
name is not a count. The note says where a cached count came from.

The loop's watch daemon never imports torch for this: a CUDA context held by a process that
lives for days would cost the card's memory for good (and ``soup serve`` often shares the card).
The probe subprocess imports torch, asks, prints one JSON line and exits, which frees the context.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Optional

from soup_cli.utils.loop_spawn import child_popen_kwargs, python_child_argv

#: bf16: two bytes per parameter.
BYTES_PER_PARAM = 2
#: What a bf16 LoRA run holds per byte of base weights: the weights, the adapter and its
#: optimizer state, gradients and activations, with room to spare.
WORKING_SET_FACTOR = 5

#: Seconds the probe subprocess may take (importing torch and opening the card).
PROBE_TIMEOUT_S = 60
#: What the probe subprocess runs. It drops the working directory from ``sys.path`` first, so a
#: ``torch.py`` or ``json.py`` next to the loop cannot be imported in its place. One JSON line
#: on stdout: ``{"free": <bytes>, "total": <bytes>}``, or ``{"free": null}`` without a card.
PROBE_SCRIPT = (
    "import sys\n"
    "sys.path[:] = [entry for entry in sys.path if entry not in ('', '.')]\n"
    "import json\n"
    "try:\n"
    "    import torch\n"
    "    if torch.cuda.is_available():\n"
    "        free, total = torch.cuda.mem_get_info()\n"
    "        print(json.dumps({'free': int(free), 'total': int(total)}))\n"
    "    else:\n"
    "        print(json.dumps({'free': None}))\n"
    "except Exception:\n"
    "    print(json.dumps({'free': None}))\n"
)

_GB = 1024**3
#: How every note of this module starts; the train stage finds its note by it.
NOTE_PREFIX = "sft quantization auto"
_PREFIX = NOTE_PREFIX

#: ``read_model_shape``'s ``params_source`` values that are counts, and how a note names them.
#: ``"name"`` (a size guessed from the repo name) is not one.
_CACHED_SOURCES = {
    "safetensors": "count from the cached safetensors headers",
    "config.json": "count from the cached config.json, an estimate without biases",
}


_PARTIAL_CACHE = "the cache holds only part of the safetensors weights"
_NUMBERED_SHARD = re.compile(r"^model-(\d+)-of-(\d+)\.safetensors$")


def _holds_every_weight_file(snapshot: str) -> bool:
    """Whether a cached snapshot has every weight file of its checkpoint on disk.

    ``param_count_from_local_safetensors`` sums the files that are present, so a snapshot
    whose index names a shard it lacks, whose ``model-0000x-of-0000N`` shards are fewer
    than ``N``, or that lists a weight file that does not resolve to a file (a cache symlink
    whose blob is gone) gives a subtotal. An unreadable index counts as incomplete.
    """
    from soup_cli.utils.checkpoint_files import checkpoint_files

    try:
        selected = checkpoint_files(snapshot)
    except (OSError, ValueError):
        return False
    if selected.missing:
        return False
    # ``os.path.isfile`` follows the cache's symlinks into ``blobs/``, and refuses a dangling one.
    if not all(os.path.isfile(os.path.join(snapshot, name)) for name in selected.weights):
        return False
    if selected.rule == "shards":
        numbers = set()
        totals = set()
        for name in selected.weights:
            match = _NUMBERED_SHARD.match(name)
            if match is None:
                return False
            numbers.add(int(match.group(1)))
            totals.add(int(match.group(2)))
        if len(totals) != 1 or numbers != set(range(1, totals.pop() + 1)):
            return False
    return True


def estimated_need_bytes(params: int) -> int:
    """VRAM a bf16 LoRA run on ``params`` parameters is expected to need, in bytes."""
    return params * BYTES_PER_PARAM * WORKING_SET_FACTOR


def parameter_count(base: str) -> Optional[int]:
    """Parameter count of ``base`` without loading weights, or None when it cannot be told.

    See :func:`count_parameters`, which also says where the count came from.
    """
    return count_parameters(base)[0]


def count_parameters(base: str) -> tuple[Optional[int], str]:
    """``(count, where it came from)``, or ``(None, why it is unknown)``.

    A local model directory is counted from its safetensors headers (a symlinked directory
    or weights file is refused); a Hub repo id from the Hub's header-only metadata, unless
    the environment is offline, else from what the Hugging Face cache holds of it (never the
    network). The second item is empty for a local directory and the Hub's metadata (exact
    counts), and names the cached file otherwise.
    """
    if not isinstance(base, str) or not base:
        return None, "no base model"
    if os.path.isdir(base):
        from soup_cli.utils.gpu import param_count_from_local_safetensors

        count = param_count_from_local_safetensors(base, refuse_symlinks=True)
        return count, "" if count is not None else "no safetensors weights could be read"
    from soup_cli.utils import model_shape

    if not model_shape._REPO_ID.fullmatch(base) or ".." in base:
        return None, "not a local directory or a Hub model id"
    count = model_shape._hub_param_count(base, allow_network=True)
    if count is not None:
        return count, ""
    shape = model_shape.read_model_shape(base, allow_network=False)
    if shape.params_source == "safetensors":
        config, snapshot = model_shape._hub_lookup(base, allow_network=False)
        if snapshot is None or not _holds_every_weight_file(snapshot):
            # The headers summed only the shards on disk: a subtotal, not the model.
            estimate = None
            if config is not None:
                estimate = model_shape.shape_from_config(
                    config, params=None, params_source="config.json"
                )
            if estimate is None:
                return None, _PARTIAL_CACHE
            return estimate.params, f"{_CACHED_SOURCES['config.json']}; {_PARTIAL_CACHE}"
    source = _CACHED_SOURCES.get(shape.params_source)
    if source is not None and shape.params > 0:
        return shape.params, source
    where = "the Hub is offline" if model_shape.hub_offline() else "the Hub gave no count"
    return None, f"{where} and the cache has no config.json or weights to count"


def free_vram_bytes() -> Optional[int]:
    """Free memory of the CUDA device in bytes, or None when it cannot be told.

    Measured in a short-lived subprocess (``PROBE_SCRIPT``, argv list, no shell, killed after
    ``PROBE_TIMEOUT_S``): this process never imports torch, so it never holds a CUDA context.
    A timeout, a failure to start, a non-zero exit, output that is not one JSON line with an
    int ``free`` of at least 0, no CUDA device and no torch are all None.
    """
    argv = python_child_argv("-c", PROBE_SCRIPT)
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            argv,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
            check=False,
            **child_popen_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):  # a timeout is a SubprocessError
        return None
    if result.returncode != 0:
        return None
    lines = [line for line in str(result.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        answer = json.loads(lines[-1])
    except (ValueError, RecursionError):
        return None
    free = answer.get("free") if isinstance(answer, dict) else None
    if isinstance(free, bool) or not isinstance(free, int) or free < 0:
        return None
    return free


def resolve_sft_quantization(setting: str, base: str) -> tuple[str, Optional[str]]:
    """``(quantization, note)`` for the soup.yaml of an sft run on ``base``.

    ``4bit`` and ``none`` come back as they are with no note. ``auto`` is measured:
    ``none`` when ``estimated_need_bytes`` fits in ``free_vram_bytes``, otherwise ``4bit``;
    the note says which and why, for the train stage's notes.
    """
    if setting in ("4bit", "none"):
        return setting, None
    if setting != "auto":
        raise ValueError(f'quantization must be one of "4bit", "none", "auto", got {setting!r}')
    params, source = count_parameters(base)
    if params is None:
        return "4bit", f"{_PREFIX}: 4bit (parameter count unknown: {source})"
    free = free_vram_bytes()
    if free is None:
        return "4bit", f"{_PREFIX}: 4bit (free VRAM unknown: no usable CUDA device)"
    need = estimated_need_bytes(params)
    chosen = "none" if need <= free else "4bit"
    where = f"; {source}" if source else ""
    return chosen, (
        f"{_PREFIX}: {chosen} (need {need / _GB:.1f} GB, free {free / _GB:.1f} GB{where})"
    )
