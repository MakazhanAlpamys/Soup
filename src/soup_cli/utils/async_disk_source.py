"""Tier 2, asynchronous: the base stays on NVMe and a background thread reads it.

The shipped ``DiskSource`` calls ``safe_open(...).get_tensor`` from
``LayerBufferPool.load_async``, which is only "async" about the copy to the GPU:
the read happens on the COMPUTE thread before the copy is enqueued, so the
Python thread blocks, the GPU starves, and the starvation shows up as wall time
rather than as a stall. Measured cold on a 36 GB 70B-shaped store: 0.57 GB/s
average and 22 MB/s at worst, from a drive that reads 3.5+ GB/s.

This source reads ahead on its own thread into pre-allocated pinned host
buffers, so ``get`` is a handoff. It never memory-maps: a mapping charges
Windows commit for the file's whole size (#926), and holding one per decoder
layer costs ~35 GB of charge for a 70B run.

``get(idx, name)`` keeps the interface ``RamSource`` and ``DiskSource`` share,
so the prefetcher and the layer wrapper are untouched and the v0.72.0
correctness gates carry over rather than being re-derived. ONE call is added
rather than kept: ``release(idx, event)``. A reusable staging buffer is exactly
what the other two sources do not have — ``RamSource`` holds every layer for the
whole run and ``DiskSource`` returns a freshly allocated tensor per call, so
neither can be recycled underneath an in-flight copy. This source can, and out
of PINNED host memory ``dst.copy_(..., non_blocking=True)`` is still draining
when ``load_async`` returns while ``pool.wait`` is a GPU-side ``wait_event``
that does not block the Python thread at all. Measured through the real pool
before ``release`` existed: 7 of 8 layers reached the device holding another
layer's weights at ``read_ahead=1``, 6 of 8 at the default 2, 4 of 8 at 4.

One layer spec group is assumed to occupy a CONTIGUOUS run of indices, which
is what ``_build_source`` produces: the decoder layers in order, then the
vocabulary-sized shards one group each. ``_group_bounds`` takes a group's span
from its lowest and highest index, so on a synthetic index whose groups
alternate the span is wider than the membership and the effective lookahead
window halves. That costs depth, never correctness — no wedge, no wrong bytes.

NO top-level torch: this module is imported by the trainer path only.
"""

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from soup_cli.utils.safetensors_reader import (
    SECTOR_BYTES,
    ShardIdentity,
    TensorRange,
    aligned_span,
    identity_of,
    open_direct,
    plan_ranges,
    read_header_with_identity,
    read_range_into,
)

logger = logging.getLogger(__name__)

MIN_STREAM_READ_AHEAD = 1
MAX_STREAM_READ_AHEAD = 8
DEFAULT_STREAM_READ_AHEAD = 2

#: How many byte ranges a layer's data section is read as, each by its own
#: worker thread through its own direct-I/O handle. Measured cold on the
#: 70B-shaped NF4 store (#974; fresh layers per configuration, Samsung PM9B1
#: NVMe, Windows 11): unbuffered 3.9-4.2 GB/s at one range, 5.1-5.65 GB/s at
#: two to four, no better at eight; the buffered per-tensor reads this replaced
#: managed 1.5-2.2 GB/s at any width, and the async source was already there
#: (gate-971 §10: 1.46-2.32 GB/s, 84.7% of a cold 70B step inside the read).
#: A range is one sector-aligned request into a sector-aligned slice of the
#: slot's region.
MIN_STREAM_READ_RANGES = 1
MAX_STREAM_READ_RANGES = 16
DEFAULT_STREAM_READ_RANGES = 4

# How long ONE layer's read may be in flight before `get` calls it a wedge.
#
# Derived from the measurement, not guessed. `benchmarks/gate-971-async-nvme-
# source.md` §8 and `benchmarks/results/probe-rtx5070/layer0_wait_cold_synth70b
# _nf4.json` are the only per-layer record: over 478 load brackets on the cold
# 70B-shaped fixture the slowest SINGLE one is 662 ms (the vocabulary matrix on
# its first touch), against a 213-252 ms mean. Ten times that is 6.6 s, so the
# 300 s floor is what binds — and it binds with room: this project's own
# worst-case source rate is 22 MB/s (the module docstring above), which is ~20 s
# for one 441 MB decoder layer, so 300 s is ~15x the slowest read anyone here
# has measured. A limit that fires on a merely slow read would be worse than no
# limit at all, because the first thing an operator would do is delete it.
_MAX_READ_SECONDS = 300.0

# How many DISTINCT layer specs this source will stage for.
#
# Staging is per distinct spec, so the bound matters: an index in which every
# layer's spec differs yields n_layers groups of one slot each, i.e. the WHOLE
# MODEL in page-locked host memory — precisely what the disk tier exists to
# avoid, and unbounded by `read_ahead` because a group of one takes a full slot
# at any depth. A real model has 1-3 (decoder, embed, untied head); a hybrid
# architecture alternating two decoder shapes has 4. Eight leaves room for a
# shape nobody has shipped yet and still refuses the pathological one, before
# the allocation rather than after.
_MAX_SPEC_GROUPS = 8

# How often a blocked `get` wakes to re-run its liveness checks. Not a deadline:
# the checks below are what raise, and this only decides how promptly. A
# consumer parked on a 40 s cold read wakes ~80 times to look at two fields.
_LIVENESS_POLL_SECONDS = 0.5


def _spec_key(layer_spec: Mapping[str, Tuple[Tuple[int, ...], str]]) -> tuple:
    """A hashable identity for one layer's tensor names, shapes and dtypes.

    Two layers share staging only if they agree on all three: a buffer sized for
    a decoder layer cannot hold a vocabulary matrix, and a buffer keyed on
    ``self_attn.q_proj.weight`` cannot answer ``model.embed_tokens.weight``.
    """
    return tuple(
        (name, tuple(shape), dtype)
        for name, (shape, dtype) in sorted(layer_spec.items())
    )


@dataclass(frozen=True)
class _LayerPlan:
    """One layer's wanted tensors, relative to the sector-aligned start of its read.

    ``start``/``end`` bound the aligned superset of the wanted tensors' byte
    range; ``expected`` is how much of it the file actually holds (the last
    sector runs past EOF); ``tensors`` is ``(name, offset from start, bytes,
    shape, dtype)``. Per LAYER rather than per spec group: a sibling whose
    header is longer, or which carries a tensor the spec does not want, keeps
    the same tensors at different offsets.
    """

    start: int
    end: int
    expected: int
    tensors: Tuple[Tuple[str, int, int, Tuple[int, ...], str], ...]

    @property
    def span(self) -> int:
        return self.end - self.start


class _RangeJob:
    __slots__ = ("done", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.error: Optional[BaseException] = None


class _RangeReaders:
    """``count`` daemon threads that read byte ranges on request.

    Daemon, deliberately: a range read wedged inside the kernel must not keep
    the interpreter from exiting once ``get``'s progress check has refused the
    run — the single reader thread this widens is daemon for the same reason,
    which is why this is not a ``ThreadPoolExecutor`` (its workers are joined
    at interpreter exit). ``run`` hands every job out and waits for all of
    them, then re-raises the first failure, so a layer read is either whole
    or an error.
    """

    def __init__(self, count: int, name: str = "soup-layer-range"):
        self._jobs: "queue.Queue[Optional[Tuple[Any, _RangeJob]]]" = queue.Queue()
        self._threads = [
            threading.Thread(target=self._loop, name=f"{name}-{index}", daemon=True)
            for index in range(count)
        ]
        for thread in self._threads:
            thread.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            work, outcome = job
            try:
                work()
            except BaseException as exc:  # noqa: BLE001 — handed back to the reader
                outcome.error = exc
            finally:
                outcome.done.set()

    def run(self, works: Sequence[Any]) -> None:
        outcomes = [_RangeJob() for _ in works]
        for work, outcome in zip(works, outcomes):
            self._jobs.put((work, outcome))
        for outcome in outcomes:
            # A job queued behind the close sentinels is never taken; polling
            # the workers' liveness is what turns that into an error rather
            # than a reader parked forever.
            while not outcome.done.wait(timeout=1.0):
                if not any(thread.is_alive() for thread in self._threads):
                    raise RuntimeError("layer-stream disk source is closed")
        for outcome in outcomes:
            if outcome.error is not None:
                raise outcome.error

    def close(self) -> None:
        for _ in self._threads:
            self._jobs.put(None)


class AsyncDiskSource:
    """Read layers ahead on a background thread; ``get`` hands over the result.

    ONE consumer thread. ``get`` is not safe to call concurrently from two,
    and the failure is a wedge rather than an error: each demand push REPLACES
    the queue, so two consumers would drop each other's request in turn and
    neither would ever be served. The shipped consumer is ``StreamPrefetcher``
    on the compute thread, which is single-threaded by construction; nothing
    here enforces it, so a second caller is a caller bug.
    """

    def __init__(
        self,
        shard_dir: str,
        n_layers: int,
        spec: Union[
            Mapping[str, Tuple[Tuple[int, ...], str]],
            Sequence[Mapping[str, Tuple[Tuple[int, ...], str]]],
        ],
        *,
        shard_paths: Optional[Sequence[str]] = None,
        read_ahead: int = DEFAULT_STREAM_READ_AHEAD,
        pin: bool = True,
        read_ranges: int = DEFAULT_STREAM_READ_RANGES,
    ):
        from soup_cli.utils.layer_stream_runtime import RamSource

        if isinstance(read_ahead, bool):
            raise ValueError("training.stream_read_ahead must be an int, not bool")
        read_ahead = int(read_ahead)
        if read_ahead < MIN_STREAM_READ_AHEAD or read_ahead > MAX_STREAM_READ_AHEAD:
            raise ValueError(
                f"training.stream_read_ahead must be between {MIN_STREAM_READ_AHEAD} "
                f"and {MAX_STREAM_READ_AHEAD}; got {read_ahead}. Each level costs one "
                f"layer of pinned host memory."
            )
        if isinstance(read_ranges, bool):
            raise ValueError("read_ranges must be an int, not bool")
        read_ranges = int(read_ranges)
        if read_ranges < MIN_STREAM_READ_RANGES or read_ranges > MAX_STREAM_READ_RANGES:
            raise ValueError(
                f"read_ranges must be between {MIN_STREAM_READ_RANGES} and "
                f"{MAX_STREAM_READ_RANGES}; got {read_ranges}. Each is one worker "
                f"thread and one direct-I/O request per layer read."
            )

        self._layer_specs = RamSource._normalize_layer_specs(spec, n_layers)
        self._paths = RamSource._normalize_shard_paths(shard_dir, n_layers, shard_paths)
        self.n_layers = int(n_layers)
        self.read_ahead = read_ahead
        self.read_ranges = read_ranges
        self.pinned = bool(pin)

        # Headers once, up front: a shard that disagrees with the index would
        # otherwise read the right byte count from the wrong offsets and train
        # on garbage with no error. Each header's file identity comes with it,
        # so the same failure arriving LATER — a shard replaced in place while
        # the run holds these ranges — is refused at the next open rather than
        # read at stale offsets (see ShardIdentity).
        self._ranges: List[Dict[str, TensorRange]] = []
        self._identities: List[ShardIdentity] = []
        for idx in range(self.n_layers):
            header, identity = read_header_with_identity(self._paths[idx])
            self._identities.append(identity)
            for name, (shape, dtype) in self._layer_specs[idx].items():
                entry = header.get(name)
                if entry is None:
                    raise ValueError(
                        f"{self._paths[idx]}: layer {idx} is missing tensor {name!r}"
                    )
                if entry.shape != tuple(shape) or entry.dtype != dtype:
                    raise ValueError(
                        f"{self._paths[idx]}: tensor {name!r} disagrees with the "
                        f"index — shard has {entry.shape} of {entry.dtype}, the "
                        f"index expects {tuple(shape)} of {dtype}"
                    )
            self._ranges.append(header)

        # The exact bytes that will be read, taken from the headers rather than
        # recomputed from the spec: `read_header` already cross-checks each
        # range against shape x itemsize, so a second dtype-size table here
        # could only ever disagree with the one authority. `layer_stream.
        # dtype_bytes` is THE table for everything that still needs one.
        self.disk_bytes = sum(
            self._ranges[idx][name].nbytes
            for idx in range(self.n_layers)
            for name in self._layer_specs[idx]
        )

        # Each layer's read, planned once: the wanted tensors' byte range,
        # widened to sector boundaries (direct I/O reads whole sectors), and
        # where each tensor lands relative to that start. Per layer, because
        # the header length — and so the data section's offset — can differ
        # between siblings, and a shard may carry tensors the spec does not
        # want, sorted among the ones it does.
        self._plans: List[_LayerPlan] = []
        for idx in range(self.n_layers):
            header = self._ranges[idx]
            wanted = [header[name] for name in self._layer_specs[idx]]
            lo = min(entry.start for entry in wanted)
            hi = max(entry.end for entry in wanted)
            start, end = aligned_span(lo, hi)
            tensors = []
            for name, (shape, dtype) in self._layer_specs[idx].items():
                entry = header[name]
                offset = entry.start - start
                elements = math.prod(shape)
                itemsize = entry.nbytes // elements if elements else 0
                if itemsize and offset % itemsize:
                    raise ValueError(
                        f"{self._paths[idx]}: tensor {name!r} starts at byte "
                        f"{entry.start}, which is not aligned to its {itemsize}-byte "
                        f"elements; a staged view cannot be placed there. Soup's own "
                        f"sharder writes every tensor aligned — re-shard the base."
                    )
                tensors.append((name, offset, entry.nbytes, tuple(shape), dtype))
            self._plans.append(
                _LayerPlan(
                    start=start,
                    end=end,
                    expected=min(end, self._identities[idx].size) - start,
                    tensors=tuple(tensors),
                )
            )

        # Staging is allocated per DISTINCT layer spec, NOT from layer 0's.
        # `_build_source` hands this source the decoder layers followed by the
        # vocabulary-sized embed / lm_head shards, whose tensor keys are
        # entirely different; sizing every slot from layer 0 made the reader
        # thread raise KeyError on the first forward turnaround of any real
        # model and poisoned the source permanently, where `DiskSource` simply
        # returns the tensor.
        groups: Dict[tuple, int] = {}
        specs_by_group: List[Mapping[str, Tuple[Tuple[int, ...], str]]] = []
        members: List[int] = []
        self._group_of: List[int] = []
        for idx in range(self.n_layers):
            key = _spec_key(self._layer_specs[idx])
            group = groups.get(key)
            if group is None:
                group = len(groups)
                groups[key] = group
                specs_by_group.append(self._layer_specs[idx])
                members.append(0)
            members[group] += 1
            self._group_of.append(group)
        if len(groups) > _MAX_SPEC_GROUPS:
            raise ValueError(
                f"layer streaming's disk tier was handed {len(groups)} distinct "
                f"layer shapes across {self.n_layers} layers, above the "
                f"{_MAX_SPEC_GROUPS} this source stages for. Staging is allocated "
                f"PER SHAPE and a shape with one member takes a full slot at any "
                f"training.stream_read_ahead, so this would hold most of the "
                f"model in host memory — page-locked when the box allows and "
                f"pin=True, pageable otherwise — which is what the disk tier "
                f"exists to avoid. A real model has 1-3 shapes (decoder, "
                f"embedding, untied lm_head). Refusing before the allocation "
                f"rather than after."
            )

        # The index span each group's walk lives in. The decoder layers are one
        # contiguous run; each vocabulary weight is a group of one, where there
        # is no walk to speak of.
        self._group_bounds: List[Tuple[int, int]] = []
        for group in range(len(groups)):
            indices = [
                idx for idx in range(self.n_layers) if self._group_of[idx] == group
            ]
            self._group_bounds.append((min(indices), max(indices)))

        # Staging is ONE region per slot — the layer's whole data section,
        # widened to sector boundaries — with every tensor a view at its file
        # offset from the region's start. Pinned, the regions are packed into
        # the same power-of-two arenas as the RAM store (#901): pinning one
        # tensor at a time cost 1.7-1.9x, one region at a time would still
        # round each layer up (441 MB -> 512 MiB), and the packer puts the
        # 70B store's four slots into one 2 GiB arena for 1.96 GB.
        region_sizes: List[int] = []
        first_member: List[int] = []
        self._group_slots: List[List[int]] = []
        for group in range(len(specs_by_group)):
            members_of_group = [
                idx for idx in range(self.n_layers) if self._group_of[idx] == group
            ]
            span = max(self._plans[idx].span for idx in members_of_group)
            # Depth beyond the number of layers sharing a spec buys nothing and
            # costs a whole vocabulary matrix of pinned host memory: embed and
            # an untied lm_head are one layer each.
            flat: List[int] = []
            for _ in range(min(read_ahead, members[group])):
                flat.append(len(region_sizes))
                region_sizes.append(span)
                first_member.append(members_of_group[0])
            self._group_slots.append(flat)
        self._arenas, self._regions = self._allocate_staging(region_sizes)
        self._slots: List[Dict[str, Any]] = [
            self._views(self._plans[first_member[slot]], self._regions[slot])
            for slot in range(len(region_sizes))
        ]
        self.staging_bytes = sum(region_sizes)
        self.nbytes = sum(
            dst.numel() * dst.element_size() for slot in self._slots for dst in slot.values()
        )

        # Direct I/O when the platform and the volume allow it, decided once
        # and said out loud: the two paths differ 2.4x cold (#974).
        self.direct_io = False
        aligned = all(region.data_ptr() % SECTOR_BYTES == 0 for region in self._regions)
        if not aligned:  # pragma: no cover — cudaHostAlloc hands out page-aligned blocks
            logger.warning(
                "layer streaming's staging is not sector-aligned on this box; the "
                "disk tier reads through the page cache instead of direct I/O"
            )
        elif self.n_layers:
            try:
                open_direct(self._paths[0]).close()
            except OSError as exc:
                logger.info(
                    "layer streaming's disk tier reads through the page cache: direct "
                    "I/O is unavailable here (%s)",
                    exc,
                )
            else:
                self.direct_io = True
        self._open = open_direct if self.direct_io else self._open_buffered
        self._readers = _RangeReaders(self.read_ranges)

        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        # The reader works a QUEUE, not a single request: with one pending
        # target it can only ever be ONE layer ahead of the consumer, so
        # `read_ahead=8` charged eight layers of pinned host memory for a
        # one-deep pipeline. On the cold-disk regime this source exists for,
        # the read is ~12x the compute, and one layer of lookahead can only
        # hide one layer's read behind one layer's compute.
        self._queue: List[int] = [0]
        self._slot_of: Dict[int, int] = {}
        self._in_flight: Optional[int] = None
        # Set and cleared in lockstep with `_in_flight`, under the lock, so the
        # two can never disagree about which layer has been in flight how long.
        self._read_started_at: Optional[float] = None
        self._next_slot: List[int] = [0] * len(self._group_slots)
        # Per staging slot: handed to the consumer and not released yet, and
        # the event that says when its copy has drained.
        self._live: List[bool] = [False] * len(self._slots)
        self._drain: List[Any] = [None] * len(self._slots)
        # Anchor and direction are PER SPEC GROUP. A single shared anchor was
        # poisoned by the very first `get` of a real run: `install_streaming`
        # primes the output weight BEFORE the decoder walk, so the first index
        # ever seen was the embed's — in its own group, above every decoder
        # layer. The old cross-group guard then made every later decoder `get`
        # return early, and the anchor never moved again for the rest of the
        # run. Measured on a hetero source over a full forward AND backward:
        # `_last_get` stuck at the embed index and `_direction` stuck at +1,
        # with `_still_wanted` true 0 times out of 16 — the eviction preference
        # degenerating to the plain FIFO it exists to replace, and the backward
        # pass prefetching the wrong way from end to end.
        #
        # Per-group state removes the guard rather than strengthening it: a
        # group that keeps its own anchor cannot be spoken for by another.
        self._last_get: Dict[int, int] = {}
        self._direction: Dict[int, int] = {}
        self._error: Optional[BaseException] = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="soup-layer-reader", daemon=True
        )
        self._thread.start()

    # -- staging ---------------------------------------------------------
    def _allocate_staging(self, region_sizes: Sequence[int]) -> Tuple[List[Any], List[Any]]:
        """Regions on sector boundaries; pinned ones as views of power-of-two arenas."""
        import torch

        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        arenas: List[Any] = []
        regions: List[Any] = []
        if self.pinned:
            plan = plan_pinned_arenas(region_sizes, align=SECTOR_BYTES)
            for size in plan.arena_sizes:
                arena = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
                if arena.device.type != "cpu":
                    raise RuntimeError(
                        "layer streaming's async disk source requested a CPU "
                        f"tensor, but torch returned {arena.device}."
                    )
                # Mirrors RamSource: a box that hands back pageable memory
                # would otherwise report pinned=True while silently paying
                # the ~97% -> ~79% GPU-utilisation cost of a synchronous
                # host-to-device copy.
                if not arena.is_pinned():
                    raise RuntimeError(
                        "layer streaming requested pinned CPU RAM, but torch "
                        "returned pageable memory; retry with pin=False."
                    )
                arenas.append(arena)
            for (arena_index, offset), size in zip(plan.placements, region_sizes):
                regions.append(arenas[arena_index][offset : offset + size])
            self.pinned_bytes = plan.pinned_bytes
        else:
            # The CPU allocator neither rounds nor aligns: a sector of slack
            # per region buys the alignment direct I/O needs.
            for size in region_sizes:
                buffer = torch.empty(size + SECTOR_BYTES, dtype=torch.uint8, device="cpu")
                pad = (-buffer.data_ptr()) % SECTOR_BYTES
                arenas.append(buffer)
                regions.append(buffer[pad : pad + size])
            self.pinned_bytes = 0
        return arenas, regions

    @staticmethod
    def _views(plan: _LayerPlan, region: Any) -> Dict[str, Any]:
        """The layer's tensors as views into ``region``, at the plan's offsets."""
        import torch

        return {
            name: region[offset : offset + nbytes].view(getattr(torch, dtype)).view(shape)
            for name, offset, nbytes, shape, dtype in plan.tensors
        }

    @staticmethod
    def _open_buffered(path: str):
        return open(path, "rb")

    # -- the reader ------------------------------------------------------
    def _read_layer(self, idx: int, region: Any) -> None:
        """Read layer ``idx``'s data section into ``region`` as parallel ranges.

        The one seam a profiler should time: one call per layer read, whatever
        the ranges do inside it. Every worker opens its own handle — the ranges
        were taken minutes or hours ago off a file this source deliberately
        does not keep open — and re-checks the file's identity before reading
        at those offsets (see ShardIdentity): a same-size replacement would
        otherwise be read at stale offsets and trained on with no error
        anywhere; only a SHORTER one surfaces, as a short read.
        """
        plan = self._plans[idx]
        path = self._paths[idx]
        expected_identity = self._identities[idx]

        def read_range(start: int, end: int):
            def work() -> None:
                with self._open(path) as handle:
                    if identity_of(handle) != expected_identity:
                        raise RuntimeError(
                            f"{path}: layer {idx}'s shard changed on disk since its "
                            f"header was read — the byte ranges this source holds no "
                            f"longer describe it. Refusing rather than reading at "
                            f"stale offsets, which would train on whatever is now at "
                            f"those bytes. Re-shard and restart, and do not re-shard "
                            f"a base while a run is reading it."
                        )
                    view = region[start - plan.start : end - plan.start]
                    read_range_into(
                        handle, start, view, min(end, expected_identity.size) - start
                    )

            return work

        ranges = plan_ranges(plan.start, plan.end, self.read_ranges)
        self._readers.run([read_range(start, end) for start, end in ranges])

    def _window_span(self, group: int) -> int:
        """How many of ``group``'s layers its staging can hold at once.

        THE one width, read by both the producer of the window and its
        defender. ``_plan_queue`` spends this as a per-group budget and
        ``_still_wanted`` compares against it; before they shared it, a group
        with fewer slots than ``read_ahead`` (a short model, or a vocabulary
        weight, which is a group of one) had an evictor defending a wider
        window than the planner could ever fill.
        """
        return len(self._group_slots[group])

    def _still_wanted(self, layer: int) -> bool:
        """Is ``layer`` still inside the lookahead window its walk is heading into?

        Anchored on ``layer``'s OWN group, which is what ``_plan_queue`` walks
        from. Reading a shared anchor here is what made this predicate answer
        about the decoder walk using the embed's index, and say False every
        time. A layer behind the walk, or beyond the far edge, has already been
        used or is not wanted yet.

        ``k = 0`` counts as wanted where ``_plan_queue`` starts at ``k = 1``:
        the layer the consumer is standing on is the one thing that must not be
        evicted, and is also the one thing there is no point fetching.
        """
        group = self._group_of[layer]
        anchor = self._last_get.get(group)
        if anchor is None:
            return False
        step = (layer - anchor) * self._direction.get(group, 1)
        return 0 <= step < self._window_span(group)

    def _claim_slot(self, idx: int) -> Optional[int]:
        """Choose a staging slot to refill with layer ``idx``. Lock held.

        Only slots in ``idx``'s spec group are eligible — a decoder buffer
        cannot hold a vocabulary matrix. A slot the consumer is still holding
        is never chosen: overwriting one is what put another layer's weights on
        the device through the real buffer pool. ``None`` means every slot in
        the group is held, and the caller waits for a release rather than
        picking a victim anyway — a fallback that overwrites a live slot is the
        defect, not a relief valve for it.

        Among the slots it MAY take, it prefers one holding a layer the walk is
        done with. Plain FIFO rotation is only right while the rotation
        direction matches the walk direction; across the forward/backward
        turnaround it goes out of phase and starts evicting layers still inside
        the window — measured, 4 of 177 claims in one traced run, including
        ``dir=-1 reading layer 20, evicted layer 19``, the very next layer
        wanted. Re-reading it is not incorrect, but it is depth the setting
        charged for and did not deliver: 2 of 3 sustained at ``read_ahead=4``,
        5 of 7 at 8.

        An empty slot counts as done-with, so a cold source still fills in
        rotation order. The FIFO fallback stands when every takeable slot is
        still wanted, which in steady state coincides with having nothing to
        fetch — the window being fully resident is what makes ``_plan_queue``
        return empty.
        """
        group = self._group_of[idx]
        flat = self._group_slots[group]
        start = self._next_slot[group]
        layer_in_slot = {held: lay for lay, held in self._slot_of.items()}
        takeable = [
            (offset, flat[(start + offset) % len(flat)])
            for offset in range(len(flat))
            if not self._live[flat[(start + offset) % len(flat)]]
        ]
        if not takeable:
            return None
        chosen_offset, chosen = takeable[0]
        for offset, candidate in takeable:
            held = layer_in_slot.get(candidate)
            if held is None or not self._still_wanted(held):
                chosen_offset, chosen = offset, candidate
                break
        self._next_slot[group] = (start + chosen_offset + 1) % len(flat)
        for layer in [lay for lay, held in self._slot_of.items() if held == chosen]:
            del self._slot_of[layer]
        return chosen

    def _hold(self, idx: int) -> None:
        """Mark ``idx``'s slot as in use and release the rest. Lock held.

        What "release the rest" may mean depends on how the staging was
        allocated, and the split is the whole safety argument:

        * PAGEABLE staging keeps the implicit release. ``dst.copy_(...,
          non_blocking=True)`` out of pageable memory is host-synchronous, so no
          copy can still be in flight and the contract this design always had —
          a reference from ``get`` is valid until your next ``get`` for a
          different layer — is still true. Every caller that predates this
          source relies on it, including the byte-identity gate, which calls
          ``get`` directly; without it their slots would never come back and the
          reader would stall until ``get``'s 30 s timeout fired. The implicit
          release carries NO drain event, which is correct: a consumer that
          never calls ``release`` never enqueued an asynchronous copy either.
        * PINNED staging REFUSES instead. There the copy is genuinely still
          draining when ``load_async`` returns, so reclaiming the buffer is the
          corruption ``release`` exists to prevent, not a tidy-up. Silently
          doing it is how 6 of 8 layers reached the device holding another
          layer's weights. A loud stop is the only honest answer, because the
          alternative is a wrong gradient with no error anywhere.
        """
        keep = self._slot_of.get(idx)
        borrowed = [
            slot
            for slot in range(len(self._slots))
            if slot != keep and self._live[slot]
        ]
        if borrowed and self.pinned:
            layer_in_slot = {held: lay for lay, held in self._slot_of.items()}
            # Sort the layer NUMBERS, then stringify: sorting the strings puts
            # "10" before "2" in operator-facing text.
            on_loan = [
                str(layer)
                for layer in sorted(
                    layer_in_slot[slot] for slot in borrowed if slot in layer_in_slot
                )
            ]
            raise RuntimeError(
                f"layer-stream staging for layer(s) {', '.join(on_loan) or '?'} "
                f"is still on loan: get() handed it out and release() was never "
                f"called, and now layer {idx} has been requested. With PINNED "
                f"staging the host-to-device copy is still draining when "
                f"load_async returns, so recycling that buffer puts another "
                f"layer's weights on the device with no error at all — measured "
                f"6 of 8 layers at the default depth. Call release(idx, event) "
                f"once the copy is enqueued; layer_stream_runtime._release_source "
                f"is what the buffer pools use. (Pageable staging permits the "
                f"implicit release, because its copy is host-synchronous.)"
            )
        for slot in borrowed:
            self._live[slot] = False
        if keep is not None:
            self._live[keep] = True
        if borrowed:
            self._ready.notify_all()

    def _plan_queue(self, idx: int) -> List[int]:
        """The upcoming layers worth staging, deepest-first order. Lock held.

        Walks ``read_ahead - 1`` steps in the consumer's CURRENT direction —
        one slot in ``idx``'s own group is spoken for by ``idx`` itself, so the
        rest is what genuine lookahead can occupy. The budget is per spec
        group and per real slot, not a flat count, for two reasons: a target
        whose group has no free slot could never be claimed and would only
        block the queue behind it, and the vocabulary group holds one member,
        so it can absorb exactly one target no matter how deep the decoder
        runs.

        At ``read_ahead=1`` the home budget is zero and the plan is empty,
        which is the same "no arming with a single slot" the explicit guard
        used to spell out: the only slot is the one being handed back.

        Already-resident layers consume budget (they are occupying a slot) but
        are not re-queued.
        """
        budget = [self._window_span(group) for group in range(len(self._group_slots))]
        budget[self._group_of[idx]] -= 1
        plan: List[int] = []
        direction = self._direction.get(self._group_of[idx], 1)
        nxt = idx + direction
        for _ in range(self.read_ahead):
            if not 0 <= nxt < self.n_layers:
                break
            group = self._group_of[nxt]
            if budget[group] <= 0:
                break
            budget[group] -= 1
            if nxt not in self._slot_of:
                plan.append(nxt)
            nxt += direction
        return plan

    def _run(self) -> None:
        try:
            while True:
                with self._ready:
                    while not self._queue and not self._closed:
                        self._ready.wait()
                    if self._closed:
                        return
                    idx = self._queue[0]
                    if idx in self._slot_of:
                        self._queue.pop(0)
                        continue
                    claimed = self._claim_slot(idx)
                    if claimed is None:
                        # Every slot in this group is still handed out. Put the
                        # request back and wait for a release: the alternative
                        # is overwriting a buffer the consumer is reading, which
                        # is the whole defect. This park is not a wedge `get`'s
                        # two liveness checks would catch — (a) a reader thread
                        # that exited without recording an error, (b) a single
                        # read in flight past `_MAX_READ_SECONDS` — because
                        # nothing is in flight here and the thread stays alive.
                        # By design: it resumes the moment whoever holds the
                        # slot calls `release()`, which clears `_live` and
                        # notifies this same condition, so no check fires for
                        # it. The target stays at the FRONT of the queue: it is
                        # still the next thing wanted, it just has nowhere to
                        # land yet.
                        logger.debug(
                            "layer-stream reader parked: every staging slot in "
                            "layer %d's spec group is still on loan",
                            idx,
                        )
                        self._ready.wait(timeout=1.0)
                        continue
                    self._queue.pop(0)
                    slot_index = claimed
                    draining = self._drain[slot_index]
                    self._drain[slot_index] = None
                    self._in_flight = idx
                    self._read_started_at = time.monotonic()
                    # Resolved HERE, under the lock that claimed it: `close()`
                    # empties `_regions` after a join that can time out, and
                    # the reader must not index a list the closer has cleared —
                    # the IndexError lands in `_error` and a later `get`
                    # reports "list index out of range" instead of "closed".
                    # The tensor stays alive through this local reference.
                    region = self._regions[slot_index]
                # OUTSIDE the lock: the compute thread must be able to call
                # get() while this waits. The event was recorded on a stream
                # that already waited on the compute stream, so it depends only
                # on work the GPU has been handed — never on this process
                # taking another Python step, which is what makes waiting here
                # safe rather than a deadlock.
                if draining is not None:
                    draining.synchronize()
                self._read_layer(idx, region)
                with self._ready:
                    self._in_flight = None
                    self._read_started_at = None
                    # Not after close(): the closer clears `_slot_of` under this
                    # same lock once the join returns, and re-populating it
                    # would resurrect a slot whose buffers are gone.
                    if not self._closed:
                        # Views are per LAYER: where each tensor sits in the
                        # region follows this layer's own header length.
                        self._slots[slot_index] = self._views(self._plans[idx], region)
                        self._slot_of[idx] = slot_index
                    self._ready.notify_all()
        except BaseException as exc:  # noqa: BLE001 — handed to the consumer
            self._fail(exc)

    def _fail(self, exc: BaseException) -> None:
        """Record a reader failure and wake everyone waiting on it."""
        with self._ready:
            self._error = exc
            self._in_flight = None
            self._read_started_at = None
            self._ready.notify_all()

    def _note_direction(self, idx: int) -> None:
        """Track which way the consumer is walking ``idx``'s group. Lock held.

        ``StreamPrefetcher`` walks 0..L-1 on the forward pass and L-1..0 on the
        backward recompute, so arming ``idx + 1`` unconditionally spent the
        whole backward half fetching a layer the consumer had already passed —
        wasted I/O that also evicted a slot the next ``get`` wanted. A repeat of
        the same index carries no direction information and leaves it alone.

        Each group is tracked separately, so a vocabulary fetch simply updates
        the vocabulary group and says nothing about the decoder walk. That is
        what makes the three orderings that actually occur harmless: the
        ``_prime`` output weight before the decoder walk, the tail prefetch at
        the forward turnaround, and the step wrap — none of them writes the
        decoder group's anchor, because none of them is a decoder index.

        The edge rule is the other half. A walk that has reached the end of its
        group's span can only continue the other way, and the index it arrives
        on is often a REPEAT (``prime()`` re-reads layer 0; the backward pass
        re-reads the last forward layer), which carries no direction of its own.
        Without this the first layer after every turnaround and every step
        boundary is demanded before it is staged. A group of one has no walk, so
        it is left alone.
        """
        group = self._group_of[idx]
        last = self._last_get.get(group)
        self._last_get[group] = idx
        if last is not None and idx != last:
            self._direction[group] = 1 if idx > last else -1
        low, high = self._group_bounds[group]
        if low < high and not low <= idx + self._direction.get(group, 1) <= high:
            self._direction[group] = -self._direction.get(group, 1)

    # -- the interface ---------------------------------------------------
    def get(self, idx: int, name: str):
        """Hand back layer ``idx``'s ``name``, staged in a REUSABLE host buffer.

        This is a borrow, not a copy — which is the point, and which is the one
        way this source differs from ``RamSource`` (holds everything forever)
        and ``DiskSource`` (allocates per call). What you owe in return depends
        on how the staging was allocated:

        * With PINNED staging (``pin=True``, the default and what production
          runs on) ``release(idx, event)`` is REQUIRED. The copy out of the
          buffer is still draining when ``load_async`` returns, so without it
          the reader refills the slot underneath the copy and another layer's
          weights land on the device. Asking for a different layer while one is
          still on loan RAISES rather than recycling it silently.
        * With PAGEABLE staging the copy is host-synchronous, so nothing can be
          in flight and the older contract still holds: the reference is valid
          until your next ``get`` for a different layer, and the implicit
          release is sound.

        ``LayerBufferPool`` and ``LargeLayerBufferPool`` both release; see
        ``layer_stream_runtime._release_source``, and the scan in
        ``TestEveryRuntimeConsumerReleases`` that keeps it that way.
        """
        with self._ready:
            while True:
                if self._error is not None:
                    raise self._error
                if self._closed:
                    raise RuntimeError("layer-stream disk source is closed")
                slot_index = self._slot_of.get(idx)
                if slot_index is not None:
                    tensor = self._slots[slot_index][name]
                    # Held from here until release() says the consumer's copy
                    # has drained. Without this the window between handing the
                    # reference out and the copy being enqueued is enough for
                    # the reader to overwrite it.
                    self._hold(idx)
                    self._note_direction(idx)
                    # Re-plan on every hit rather than arming one target. The
                    # plan is recomputed from the CURRENT direction, so the
                    # backward pass discards a forward queue at the turnaround
                    # instead of spending the whole recompute fetching layers
                    # the consumer has already gone past.
                    self._queue = self._plan_queue(idx)
                    if self._queue:
                        self._ready.notify_all()
                    return tensor
                # Asking for a layer that is not resident takes the SAME branch
                # `_hold` takes on a hit, and it is the branch that matters:
                # under PAGEABLE staging it ends the implicit hold on the other
                # slots, so the reader always has one to claim and a
                # release-unaware consumer cannot deadlock a reader that refuses
                # to overwrite a live slot; under PINNED staging it REFUSES,
                # because a copy out of an unreleased buffer is still draining
                # and recycling it is the corruption `release` exists to
                # prevent. Missing rather than hitting changes nothing about
                # that split — see `_hold`.
                self._hold(idx)
                if self._in_flight != idx and (
                    not self._queue or self._queue[0] != idx
                ):
                    # Demand goes to the FRONT: a blocked consumer outranks any
                    # lookahead, including one this same walk planned. The rest
                    # of the queue is dropped — wanting a layer the plan did not
                    # have is the plan being wrong, not a reason to finish it.
                    self._queue = [idx]
                    self._ready.notify_all()
                self._ready.wait(timeout=_LIVENESS_POLL_SECONDS)
                self._check_the_reader_is_making_progress(idx)

    def _check_the_reader_is_making_progress(self, idx: int) -> None:
        """Raise if the reader cannot serve ``idx``. Lock held.

        This replaces a guard that could not fire. The old one required ``idx``
        to be in none of ``_queue`` / ``_in_flight`` / ``_slot_of``, and the
        demand push two lines above puts it in ``_queue`` before the wait while
        the reader moves it queue -> ``_in_flight`` -> ``_slot_of`` under this
        lock — so on the single-consumer path it was always in exactly one of
        them and the conjunction was never true. It was the spec's NAMED
        mitigation for the spec's named risk ("a background thread in the
        training loop is new surface for a hang"), and it was dead: a wedged
        reader made the suite HANG instead of fail.

        Two checks, each for a state that really is reachable:

        * The reader thread is gone and said nothing. A backstop, deliberately:
          ``_run`` is one ``try`` around the whole loop whose only ``return`` is
          guarded by ``self._closed``, and every other exit runs ``_fail``, so
          the thread cannot exit with ``_error`` unset and ``_closed`` False
          through any path in this file. It exists for the ones that are not in
          this file — a monkeypatched ``_run`` in a test, a future edit adding a
          second ``return``, an interpreter that kills the thread — where the
          alternative is a consumer blocking forever on a reader that is not
          there.
        * One read has been in flight past ``_MAX_READ_SECONDS``. THIS is the
          reachable wedge: a reader blocked inside ``read_into`` (a stalled
          drive, a disconnected network path) keeps ``idx`` legitimately
          queued, so no amount of state-inspection can tell it from a slow
          read — only elapsed time can. The stamp covers the drain wait as well
          as the read itself, because it is taken with ``_in_flight``: both are
          time the consumer spends unable to proceed, and charging the wider
          window errs towards firing, which is the safe direction for a
          hang detector carrying a 15x margin.
        """
        if self._error is None and not self._closed and not self._thread.is_alive():
            raise RuntimeError(
                f"layer-stream reader thread exited without recording an error, "
                f"with layer {idx} still wanted. Refusing rather than blocking: a "
                f"training run that stops without an error is worse than one that "
                f"fails."
            )
        started = self._read_started_at
        if started is None:
            return
        elapsed = time.monotonic() - started
        if elapsed > _MAX_READ_SECONDS:
            raise RuntimeError(
                f"layer-stream reader has been reading layer {self._in_flight} for "
                f"{elapsed:.0f} s, past the {_MAX_READ_SECONDS:.0f} s limit (layer "
                f"{idx} is waiting behind it). The slowest single layer read ever "
                f"measured on this tier is 0.662 s, so this is a wedged read, not a "
                f"slow one. Refusing rather than blocking: a training run that stops "
                f"without an error is worse than one that fails."
            )

    def release(self, idx: int, event: Any = None) -> None:
        """Say the consumer is done reading layer ``idx`` out of its staging slot.

        ``LayerBufferPool.load_async`` enqueues ``dst.copy_(..., non_blocking=
        True)`` on a side stream out of PINNED host memory, so the copy is still
        draining when the call returns, and ``pool.wait`` is a GPU-side
        ``wait_event`` that never blocks the Python thread. Recycling the
        staging buffer at that point rewrites the bytes the copy is reading:
        measured through the real pool, 6 of 8 layers reached the device holding
        another layer's weights at the default depth.

        ``event`` is a CUDA event recorded after those copies — the reader waits
        on it before reusing the slot. ``None`` means the copy was already
        synchronous (no side stream, or a non-CUDA device) and the slot is free
        immediately. A layer that is no longer resident is ignored: it can only
        have been evicted, which requires having been released already.
        """
        with self._ready:
            slot_index = self._slot_of.get(idx)
            if slot_index is None:
                return
            self._drain[slot_index] = event
            self._live[slot_index] = False
            # The reader may be parked because every slot in this group was
            # held; this is the event it was waiting for.
            self._ready.notify_all()

    def close(self) -> None:
        """Stop the reader and release the staging buffers. Idempotent.

        The teardown runs UNDER the lock, and drops four things rather than two.
        The join has a timeout, so a reader still inside a cold read outlives
        this call; clearing shared state unlocked let that reader observe a
        half-torn source and store an ``IndexError`` a later ``get`` would
        report as "list index out of range" instead of "closed". ``_live`` and
        ``_drain`` go too: they hold the buffer pools' ``torch.cuda.Event``
        objects, and a closed source has no business keeping them alive.
        """
        with self._ready:
            if self._closed:
                return
            self._closed = True
            self._ready.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)
        with self._ready:
            self._slots = []
            self._regions = []
            self._arenas = []
            self._slot_of = {}
            self._live = []
            self._drain = []
        self._readers.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:  # noqa: BLE001 — interpreter teardown
            pass
