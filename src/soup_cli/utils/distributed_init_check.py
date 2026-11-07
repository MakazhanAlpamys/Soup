"""Cross-rank init consistency for data-parallel layer streaming (C5).

Data parallelism over the LoRA adapters is only correct if every rank starts from the
same trainable tensors AND the same frozen base, and Soup's gradient average
(:class:`~soup_cli.utils.adapter_grad_sync.AdapterGradSync`) assumes it without checking.
DDP gives the first half for free (it broadcasts the parameters from rank 0 when it
wraps the model); a streamed model is never wrapped (see ``adapter_sync_trainer``), so
this module does both halves once, after the model and trainer are built and before the
first step:

1. **Check (an error on mismatch).** Every rank builds a small fingerprint, the ranks
   all-gather it, and any difference from rank 0 is refused BY NAME (which item, which
   rank). The items:

   * ``trainable layout``: the names, shapes and dtypes of the trainable tensors, in
     ``named_parameters()`` order. A different LoRA ``r`` or target-module list gives a
     different layout, and a flat gradient buffer of a different length would hang or
     corrupt the collective. Gathered as a digest first; only on a mismatch are the full
     layouts gathered, to name the first differing tensor.
   * ``shard index``: what the sharder wrote (architecture, dtype, layer keys, parameter
     count, source file names and sizes, expert layout). NOT the source files' mtimes
     (two hosts that downloaded the same repository have different mtimes) and not the
     Soup version or any local path.
   * ``quantisation``: ``quant`` (NF4 or none), the double-quant flag, the device the
     offline quantisation ran on and a digest of the per-weight specs.
   * ``first layer shard`` / ``last layer shard``: SHA-256 of the two layer files. NF4
     statistics can differ between GPU generations (``layer_shard`` notes that CPU and
     CUDA agree on the nibbles but not on every float32 nested statistic), and the cache
     key only records the device kind, so these digests are what would show it. They cost
     one read of two files (about a second per GB). The embeddings and the remaining
     layers are not hashed.
   * ``base revision``: the Hub commit the run bound its base to (rank 0 resolves it and
     broadcasts it, ``run_provenance``), ``unknown`` for a local directory.

2. **Broadcast (a value mismatch is fixed, not an error).** The trainable tensors are
   copied from rank 0, so ranks launched with different ``seed`` values (LoRA ``A`` is
   drawn from a seeded generator) cannot start apart. The caller prints one line that
   says so, and which ranks had different values.

A rank that cannot build its own fingerprint (an unreadable shard) sends the error in
place of the fingerprint, so the other ranks refuse by name instead of waiting for it
until the group timeout.

Every collective goes through a :class:`~soup_cli.utils.adapter_grad_transport.GradGroup`:
``torch.distributed`` (the default, unchanged) or, for B1, the TCP star. Objects travel as
JSON there, which every value this module gathers survives.

Torch is imported inside the functions: ``soup`` stays importable without it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from soup_cli.utils.adapter_grad_sync import (
    DEFAULT_CHUNK_BYTES,
    collect_trainable,
    validate_chunk_bytes,
)
from soup_cli.utils.adapter_grad_transport import resolve_group

ITEM_LAYOUT = "trainable layout"
ITEM_SHARD_INDEX = "shard index"
ITEM_QUANT = "quantisation"
ITEM_FIRST_SHARD = "first layer shard"
ITEM_LAST_SHARD = "last layer shard"
ITEM_REVISION = "base revision"

#: The order a refusal lists the items in.
ITEMS = (
    ITEM_LAYOUT,
    ITEM_SHARD_INDEX,
    ITEM_QUANT,
    ITEM_FIRST_SHARD,
    ITEM_LAST_SHARD,
    ITEM_REVISION,
)

#: The frozen-base items (everything but the layout, which comes from the model).
BASE_ITEMS = ITEMS[1:]

_HASH_CHUNK_BYTES = 16 * 1024 * 1024
_ELEMENT_BYTES = 4

#: A tensor in a layout: ``[name, shape, dtype]`` (lists, so it pickles and JSON-dumps alike).
LayoutEntry = List[Any]


class DistributedInitMismatchError(ValueError):
    """The ranks do not start from the same trainable layout or the same frozen base."""


@dataclass(frozen=True)
class Mismatch:
    """One item on which ``rank`` differs from rank 0."""

    item: str
    rank: int
    detail: str


@dataclass(frozen=True)
class InitReport:
    """What :func:`verify_and_sync_init` did on this rank."""

    world_size: int
    rank: int
    n_tensors: int
    n_elements: int
    #: Ranks (other than 0) whose adapter VALUES differed from rank 0's before the broadcast.
    differing_ranks: Tuple[int, ...]
    broadcast_bytes: int
    seconds: float


# ---------------------------------------------------------------------------
# what one rank holds
# ---------------------------------------------------------------------------


def hash_file(path: str, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """SHA-256 of a file's bytes, read in ``chunk_bytes`` pieces."""
    sha = hashlib.sha256()
    buffer = bytearray(chunk_bytes)
    view = memoryview(buffer)
    with open(path, "rb") as handle:
        while True:
            count = handle.readinto(buffer)
            if not count:
                break
            sha.update(view[:count])
    return sha.hexdigest()


def _digest_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shard_index_digest(index: Any) -> str:
    """Digest of what a shard index says about the frozen base, host-independent.

    Left out on purpose: ``source_fingerprint`` and the source files' mtimes (they differ
    between two hosts that downloaded the same repository), ``soup_version`` (a cache
    written by an older release is read by a newer one), and every path or placement of a
    local disk (``stripe_roots``, ``layer_roots``, the external tensors' files).
    """
    source_files = sorted(
        (str(entry[0]), int(entry[1])) for entry in (getattr(index, "source_files", ()) or ())
    )
    expert_shapes = {
        str(key): [int(dim) for dim in shape]
        for key, shape in sorted((getattr(index, "expert_shapes", {}) or {}).items())
    }
    payload = {
        "format_version": getattr(index, "format_version", None),
        "arch": getattr(index, "arch", ""),
        "dtype": getattr(index, "dtype", ""),
        "n_layers": getattr(index, "n_layers", None),
        "layer_keys": list(getattr(index, "layer_keys", ()) or ()),
        "extra_keys": list(getattr(index, "extra_keys", ()) or ()),
        "large_keys": list(getattr(index, "large_keys", ()) or ()),
        "total_params": getattr(index, "total_params", None),
        "source_files": source_files,
        "expert_layout": getattr(index, "expert_layout", ""),
        "expert_layers": list(getattr(index, "expert_layers", ()) or ()),
        "expert_count": getattr(index, "expert_count", 0),
        "expert_shapes": expert_shapes,
        "external_mode": getattr(index, "external_mode", ""),
        "external_tensors": sorted(
            str(key) for key in (getattr(index, "external_tensors", {}) or {})
        ),
    }
    return _digest_json(payload)


def quantisation_label(index: Any) -> str:
    """How the frozen base was quantised, readable: ``none`` or ``nf4 (double_quant=...)``."""
    quant = str(getattr(index, "quant", "") or "none")
    if quant == "none":
        return "none"
    specs = {
        "quant_specs": sorted(
            (str(key), repr(spec))
            for key, spec in (getattr(index, "quant_specs", {}) or {}).items()
        ),
        "expert_quant_specs": sorted(
            (str(key), repr(spec))
            for key, spec in (getattr(index, "expert_quant_specs", {}) or {}).items()
        ),
    }
    device = str(getattr(index, "quant_device", "") or "n/a")
    return (
        f"{quant} (double_quant={bool(getattr(index, 'double_quant', False))}, "
        f"quantised on {device}, specs {_digest_json(specs)[:12]})"
    )


def frozen_base_items(
    shard_dir: str, index: Any, *, revision: Optional[str] = None
) -> Dict[str, str]:
    """The frozen-base items of this rank: ``{item name: value}`` (see the module docstring)."""
    from soup_cli.utils.layer_shard import layer_paths

    paths = layer_paths(shard_dir, index)
    if not paths:
        raise ValueError("the shard index lists no decoder layer to fingerprint")
    first = hash_file(paths[0])
    last = first if len(paths) == 1 else hash_file(paths[-1])
    return {
        ITEM_SHARD_INDEX: shard_index_digest(index),
        ITEM_QUANT: quantisation_label(index),
        ITEM_FIRST_SHARD: first,
        ITEM_LAST_SHARD: last,
        ITEM_REVISION: revision or "unknown",
    }


def trainable_layout(params: Sequence[Tuple[str, Any]]) -> List[LayoutEntry]:
    """``[name, shape, dtype]`` per trainable tensor, in the order given."""
    return [[name, [int(dim) for dim in param.shape], str(param.dtype)] for name, param in params]


def layout_digest(layout: Sequence[LayoutEntry]) -> str:
    return _digest_json(list(layout))


def values_digest(params: Sequence[Tuple[str, Any]]) -> str:
    """SHA-256 over the trainable tensors' values (fp32 bytes, in order)."""
    import torch

    sha = hashlib.sha256()
    with torch.no_grad():
        for _name, param in params:
            host = param.detach().to(device="cpu", dtype=torch.float32).contiguous()
            sha.update(host.numpy().tobytes())
    return sha.hexdigest()


def describe_local(
    params: Sequence[Tuple[str, Any]], base_items: Optional[Mapping[str, str]]
) -> Dict[str, Any]:
    """What this rank sends to the all-gather."""
    layout = trainable_layout(params)
    return {
        "error": None,
        "base": dict(base_items or {}),
        "layout": {
            "digest": layout_digest(layout),
            "n_tensors": len(layout),
            "n_elements": sum(param.numel() for _name, param in params),
        },
        "values": values_digest(params),
    }


# ---------------------------------------------------------------------------
# comparing the ranks
# ---------------------------------------------------------------------------


def _short(value: Any) -> str:
    """A 64-hex digest as its first 12 characters; anything else as is."""
    text = str(value)
    is_digest = len(text) == 64 and all(char in "0123456789abcdef" for char in text)
    return text[:12] + "..." if is_digest else text


def first_layout_difference(
    reference: Sequence[LayoutEntry], other: Sequence[LayoutEntry], rank: int
) -> str:
    """Name the first tensor on which rank 0's layout and ``rank``'s disagree."""
    from soup_cli.utils.terminal import for_terminal

    for position in range(max(len(reference), len(other))):
        left = reference[position] if position < len(reference) else None
        right = other[position] if position < len(other) else None
        if left == right:
            continue
        if left is None:
            return f"tensor {position} '{for_terminal(str(right[0]))}' only exists on rank {rank}"
        if right is None:
            return f"tensor {position} '{for_terminal(str(left[0]))}' only exists on rank 0"
        if left[0] != right[0]:
            return (
                f"tensor {position} is '{for_terminal(str(left[0]))}' on rank 0 and "
                f"'{for_terminal(str(right[0]))}' on rank {rank}"
            )
        return (
            f"tensor {position} '{for_terminal(str(left[0]))}' is {tuple(left[1])} {left[2]} "
            f"on rank 0 and {tuple(right[1])} {right[2]} on rank {rank}"
        )
    return "the layouts are equal"


def _layout_detail(
    gathered: Sequence[Mapping[str, Any]], rank: int, layouts: Optional[Sequence[Any]]
) -> str:
    reference, other = gathered[0]["layout"], gathered[rank]["layout"]
    counts = (
        f"rank 0 has {reference['n_tensors']} tensors / {reference['n_elements']:,} elements, "
        f"rank {rank} has {other['n_tensors']} tensors / {other['n_elements']:,} elements"
    )
    if layouts is None:
        return counts
    difference = first_layout_difference(layouts[0], layouts[rank], rank)
    return f"{counts}; first difference: {difference}"


def find_mismatches(
    gathered: Sequence[Mapping[str, Any]], layouts: Optional[Sequence[Any]] = None
) -> List[Mismatch]:
    """Every item on which a rank differs from rank 0, in the order of :data:`ITEMS`.

    ``gathered`` is one :func:`describe_local` result per rank (none with an error).
    ``layouts``, the full layout of every rank, only sharpens the layout message.
    """
    found: List[Mismatch] = []
    reference = gathered[0]
    for item in ITEMS:
        for rank in range(1, len(gathered)):
            if item == ITEM_LAYOUT:
                if gathered[rank]["layout"]["digest"] != reference["layout"]["digest"]:
                    found.append(Mismatch(item, rank, _layout_detail(gathered, rank, layouts)))
                continue
            wanted = reference["base"].get(item)
            have = gathered[rank]["base"].get(item)
            if have != wanted:
                found.append(
                    Mismatch(
                        item,
                        rank,
                        f"{_short(have)} on rank {rank}, {_short(wanted)} on rank 0",
                    )
                )
    return found


def format_mismatches(found: Sequence[Mismatch]) -> str:
    lines = [
        "distributed layer streaming: the ranks do not start from the same trainable "
        "layout and frozen base, so their gradients would not mean the same thing. "
        "Refusing to train:"
    ]
    lines.extend(
        f"  - {entry.item}: rank {entry.rank} differs from rank 0 ({entry.detail})"
        for entry in found
    )
    lines.append(
        "Launch every rank with the same config (lora.r, lora.target_modules, quantization), "
        "the same base revision, and the same GPU model."
    )
    return "\n".join(lines)


def _format_errors(errors: Sequence[Tuple[int, str]]) -> str:
    from soup_cli.utils.terminal import for_terminal

    parts = "; ".join(f"rank {rank}: {for_terminal(text)}" for rank, text in errors)
    return (
        "distributed layer streaming: a rank could not fingerprint its model for the "
        f"cross-rank init check ({parts}). Refusing to train."
    )


# ---------------------------------------------------------------------------
# the broadcast
# ---------------------------------------------------------------------------


def broadcast_trainable(
    params: Sequence[Tuple[str, Any]],
    *,
    group: Optional[Any] = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    out: Optional[Any] = None,
) -> int:
    """Copy the trainable tensors from rank 0 to every rank, in place; return the bytes.

    One flat fp32 buffer, broadcast in pieces of at most ``chunk_bytes``. Every rank must
    hold the same layout (``collect_trainable`` ordering); the caller checked it. ``out``
    is that buffer when the caller allocated it already (``verify_and_sync_init`` does,
    before the gather, so an allocation failure on one rank is reported to all of them
    instead of leaving the others inside ``broadcast``).
    """
    import torch

    transport = resolve_group(group)
    if transport is None:
        raise RuntimeError("broadcast_trainable needs a process group, and there is none")
    step = validate_chunk_bytes(chunk_bytes) // _ELEMENT_BYTES
    total = sum(param.numel() for _name, param in params)
    if out is None:
        flat = torch.empty(total, dtype=torch.float32, device=params[0][1].device)
    elif out.dtype != torch.float32 or out.numel() != total:
        raise ValueError(
            f"broadcast buffer must be {total} fp32 elements, got {out.numel()} {out.dtype}"
        )
    else:
        flat = out
    with torch.no_grad():
        offset = 0
        for _name, param in params:
            count = param.numel()
            flat[offset : offset + count].copy_(param.detach().reshape(-1))
            offset += count
        for begin in range(0, total, step):
            transport.broadcast_(flat[begin : begin + step])
        offset = 0
        for _name, param in params:
            count = param.numel()
            param.copy_(flat[offset : offset + count].view(param.shape))
            offset += count
    return _ELEMENT_BYTES * total


def verify_and_sync_init(
    model: Any,
    base_items: Optional[Mapping[str, str]] = None,
    *,
    group: Optional[Any] = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> InitReport:
    """Check that every rank starts alike, then broadcast the adapters from rank 0 (C5).

    Raises :class:`DistributedInitMismatchError` (on every rank, with the same message) when the
    trainable layout or the frozen base differs, and ``RuntimeError`` when there is no
    process group although the run was launched as distributed (each rank would train its
    own copy, silently). In a world of one it only reports.
    """
    transport = resolve_group(group)
    if transport is None:
        raise RuntimeError(
            "distributed layer streaming has no process group, although the run was "
            "launched with several ranks: each rank would train its own copy of the "
            "adapters. The launcher (torchrun / accelerate launch) must create the group."
        )
    world_size = transport.world_size
    rank = transport.rank
    start = time.perf_counter()

    params: List[Tuple[str, Any]] = []
    local: Dict[str, Any]
    flat = None
    problem = (base_items or {}).get("error")
    try:
        if problem:
            raise RuntimeError(f"the frozen-base fingerprint failed: {problem}")
        params = collect_trainable(model)
        local = describe_local(params, base_items)
        # The broadcast buffer is allocated HERE, before the gather: an out-of-memory on
        # one rank then travels in the gather like any other error, instead of striking
        # after it while the other ranks already wait inside broadcast.
        import torch

        flat = torch.empty(
            sum(param.numel() for _name, param in params),
            dtype=torch.float32,
            device=params[0][1].device,
        )
    except Exception as exc:  # noqa: BLE001 -- every rank must reach the gather
        local = {"error": f"{type(exc).__name__}: {exc}"}

    gathered: List[Any] = transport.all_gather_object(local)

    errors = [(index, entry["error"]) for index, entry in enumerate(gathered) if entry["error"]]
    if errors:
        raise DistributedInitMismatchError(_format_errors(errors))

    layouts: Optional[List[Any]] = None
    digests = {entry["layout"]["digest"] for entry in gathered}
    if len(digests) > 1:
        layouts = transport.all_gather_object(trainable_layout(params))
    found = find_mismatches(gathered, layouts)
    if found:
        raise DistributedInitMismatchError(format_mismatches(found))

    differing = tuple(
        index
        for index in range(1, world_size)
        if gathered[index]["values"] != gathered[0]["values"]
    )
    sent = broadcast_trainable(params, group=transport, chunk_bytes=chunk_bytes, out=flat)
    return InitReport(
        world_size=world_size,
        rank=rank,
        n_tensors=len(params),
        n_elements=sum(param.numel() for _name, param in params),
        differing_ranks=differing,
        broadcast_bytes=sent,
        seconds=time.perf_counter() - start,
    )
