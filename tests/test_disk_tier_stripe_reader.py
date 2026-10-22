"""R4 — every drive keeps a read in flight; one drive still reads one layer at a time."""

import io
import threading
import time
from pathlib import Path

import pytest
from rich.console import Console

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

import soup_cli.utils.async_disk_source as mod  # noqa: E402
from soup_cli.utils.async_disk_source import AsyncDiskSource, _RangeReaders  # noqa: E402
from soup_cli.utils.layer_shard import layer_shard_path  # noqa: E402
from soup_cli.utils.layer_stream_runtime import (  # noqa: E402
    DiskSource,
    RamSource,
    StreamRuntime,
    _build_source,
    _say_page_cache_roots,
)
from tests.conftest import strip_ansi  # noqa: E402

N = 8


def _striped_shards(tmp_path: Path, n: int = N):
    """Layer i under folder root{i % 2}: two folders standing in for two drives."""
    roots = [tmp_path / "root0", tmp_path / "root1"]
    for root in roots:
        root.mkdir()
    torch.manual_seed(928)
    paths = []
    for idx in range(n):
        path = layer_shard_path(str(roots[idx % 2]), idx)
        save_file(
            {
                "self_attn.q_proj.weight": torch.randint(0, 255, (64, 32), dtype=torch.uint8),
                "self_attn.q_proj.weight::absmax": torch.rand(16, dtype=torch.float32),
                "input_layernorm.weight": torch.rand(64, dtype=torch.bfloat16),
            },
            path,
        )
        paths.append(path)
    return str(roots[0]), paths, [idx % 2 for idx in range(n)]


def _raw(tensor):
    return tensor.reshape(-1).view(torch.uint8)


def _timed_reads(monkeypatch, delay=0.05, block_root=None, release=None):
    """Wrap every range read: record (layer, root folder, region pointer, start, end), slow
    it down, and optionally hold every read from one folder until ``release`` is set."""
    real = AsyncDiskSource._range_works
    log = []
    lock = threading.Lock()

    def wrapped(self, idx, region):
        root = Path(self._paths[idx]).parent.name
        works = real(self, idx, region)

        def timed(work):
            def run():
                began = time.perf_counter()
                if root == block_root:
                    release.wait(timeout=30)
                time.sleep(delay)
                work()
                with lock:
                    log.append((idx, root, region.data_ptr(), began, time.perf_counter()))

            return run

        return [timed(work) for work in works]

    monkeypatch.setattr(AsyncDiskSource, "_range_works", wrapped)
    return log


def _per_layer(log):
    """One (layer, root, region, first start, last end) per layer, over its ranges."""
    spans = {}
    for idx, root, region, began, ended in log:
        if idx in spans:
            _, _, _, first, last = spans[idx]
            began, ended = min(first, began), max(last, ended)
        spans[idx] = (idx, root, region, began, ended)
    return spans


def _overlap(a, b):
    return a[3] < b[4] and b[3] < a[4]


def _walk(source, spec, order, compute=0.0):
    """``get`` every tensor of every layer in ``order``; ``compute`` seconds per layer stand
    in for the forward/backward work a real consumer does between layers."""
    for idx in order:
        for name in spec[idx]:
            source.get(idx, name)
        if compute:
            time.sleep(compute)


def _source(tmp_path, *, striped=True, n=N, **kwargs):
    shard_dir, paths, roots = _striped_shards(tmp_path, n)
    spec = RamSource.layer_specs_from_paths(paths)
    extra = {"layer_roots": roots} if striped else {}
    source = AsyncDiskSource(shard_dir, n, spec, pin=False, shard_paths=paths, **extra, **kwargs)
    return source, spec, shard_dir, paths


#: The steady-state walk: 16 layers over two drives, read_ahead 3. Gated, not timed.
STEADY_LAYERS = 16
GATE_WAIT_S = 30.0
START_WAIT_S = 10.0


def _gated_reads(monkeypatch, gates, started):
    """Every range read of layer ``idx`` sets ``started[idx]`` on entry, then blocks until
    ``gates[idx]`` is set. The test, not the clock, decides when a read lands."""
    real = AsyncDiskSource._range_works

    def wrapped(self, idx, region):
        works = real(self, idx, region)

        def gated(work):
            def run():
                started[idx].set()
                gates[idx].wait(timeout=GATE_WAIT_S)
                work()

            return run

        return [gated(work) for work in works]

    monkeypatch.setattr(AsyncDiskSource, "_range_works", wrapped)


def test_every_drive_stays_busy_in_steady_state(tmp_path, monkeypatch):
    """The test a synchronous-batch reader fails, with no clock in it.

    The reader plans its lookahead on every HIT: once the consumer has layer k, the queue
    becomes k+1.. and every idle drive claims its next layer. Consecutive layers live on
    different drives, so with layer k landed BOTH k+1 and k+2 must be in flight at once, one
    per drive, while their reads are still open. The main thread lands layers one at a time
    (opening gate k), then requires started[k+1] and started[k+2] with both gates still
    closed. A batch reader claims nothing while any read is pending, so it starts k+1 and
    never k+2 until k+1 lands: the wait times out. Scheduling cannot change the answer: the
    waits are on events the reader sets, so a late thread only delays, and the gates keep
    both reads in flight however long that takes.
    """
    gates = [threading.Event() for _ in range(STEADY_LAYERS)]
    started = [threading.Event() for _ in range(STEADY_LAYERS)]
    _gated_reads(monkeypatch, gates, started)
    source, spec, _, _ = _source(tmp_path, n=STEADY_LAYERS, read_ahead=3, read_ranges=1)
    errors = []

    def consume():
        try:
            _walk(source, spec, range(STEADY_LAYERS))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    stalled = None
    try:
        for idx in range(STEADY_LAYERS - 2):
            gates[idx].set()
            for ahead in (idx + 1, idx + 2):
                if not started[ahead].wait(timeout=START_WAIT_S):
                    stalled = (idx, ahead)
                    break
            if stalled:
                break
    finally:
        for gate in gates:
            gate.set()
        consumer.join(timeout=GATE_WAIT_S)
        source.close()
    assert stalled is None, (
        f"after the consumer took layer {stalled[0]}, layer {stalled[1]} never started "
        f"although the reads already in flight were held open: a drive sat idle"
        if stalled
        else ""
    )
    assert errors == [], errors


def test_one_drive_never_has_two_layers_in_flight(tmp_path, monkeypatch):
    log = _timed_reads(monkeypatch)
    source, spec, _, _ = _source(tmp_path, striped=False, read_ahead=3, read_ranges=1)
    try:
        _walk(source, spec, range(N))
    finally:
        source.close()
    spans = sorted(_per_layer(log).values(), key=lambda span: span[3])
    assert all(not _overlap(a, b) for a, b in zip(spans, spans[1:])), spans


def test_reads_in_flight_together_never_share_a_staging_region(tmp_path, monkeypatch):
    log = _timed_reads(monkeypatch)
    source, spec, _, _ = _source(tmp_path, read_ahead=3, read_ranges=2)
    try:
        _walk(source, spec, list(range(N)) + list(reversed(range(N))))
    finally:
        source.close()
    # Per RANGE READ, not per layer: the walk turns around with three slots, so layers 0-4
    # are read twice, and one span merged over both reads of a layer covers everything read
    # in between — seven "clashes" on a reader with none.
    clashes = [
        (a[0], b[0])
        for position, a in enumerate(log)
        for b in log[position + 1 :]
        if a[0] != b[0] and a[2] == b[2] and _overlap(a, b)
    ]
    assert clashes == [], clashes


def test_a_claim_never_lands_in_the_slot_of_a_read_still_in_flight(tmp_path, monkeypatch):
    """Posed directly, with the reader stubbed out (test_issue971 does the same for `_hold`):
    no timed walk reaches this state on demand — a mutant that ignores ``exclude`` passes
    every test above. Root 0 is reading layer 2 into slot C; slots A and B hold layers 0 and
    1, both inside the window of a consumer standing on layer 0; layer 5 is demanded on idle
    root 1. An in-flight slot is in neither ``_slot_of`` nor ``_live``, so without
    ``exclude`` it looks EMPTY — the preferred victim — and layer 5 would be read into the
    region layer 2 is still being read into."""
    monkeypatch.setattr(AsyncDiskSource, "_run", lambda self: None)
    source, _, _, _ = _source(tmp_path, read_ahead=3)
    try:
        slot_a, slot_b, slot_c = source._group_slots[0]
        with source._ready:
            source._slot_of = {0: slot_a, 1: slot_b}
            source._last_get[0] = 0
            source._direction[0] = 1
            source._next_slot[0] = 0
            source._pending = {
                0: mod._PendingRead(
                    idx=2, slot=slot_c, region=source._regions[slot_c], started_at=time.monotonic()
                )
            }
            source._sync_in_flight()
            source._queue = [5]
            started = source._claim_reads()
            assert [(read.idx, read.slot) for read, _ in started] == [(5, slot_a)]
            # The wedge check reads the OLDEST pending read.
            assert source._in_flight == 2 and source._in_flight_batch == (2, 5)
    finally:
        source.close()


@pytest.mark.parametrize("read_ahead", [2, 3, 4])
def test_both_directions_read_the_right_bytes(tmp_path, read_ahead):
    source, spec, shard_dir, paths = _source(tmp_path, read_ahead=read_ahead)
    shipped = DiskSource(shard_dir, N, spec, shard_paths=paths)
    try:
        for idx in list(range(N)) + list(reversed(range(N))) + list(range(N)):
            for name in spec[idx]:
                assert torch.equal(_raw(source.get(idx, name)), _raw(shipped.get(idx, name))), (
                    idx,
                    name,
                )
    finally:
        source.close()
        shipped.close()


def test_a_read_wedged_on_one_drive_is_reported(tmp_path, monkeypatch):
    release = threading.Event()
    _timed_reads(monkeypatch, delay=0.0, block_root="root1", release=release)
    monkeypatch.setattr(mod, "_MAX_READ_SECONDS", 0.3)
    monkeypatch.setattr(mod, "_LIVENESS_POLL_SECONDS", 0.05)
    source, spec, _, _ = _source(tmp_path, read_ahead=3)
    try:
        with pytest.raises(RuntimeError, match="has been reading layer"):
            _walk(source, spec, range(N))
    finally:
        release.set()
        source.close()


def test_close_stops_every_drives_pool(tmp_path):
    source, _, _, _ = _source(tmp_path)
    pools = list(source._pools.values())
    assert len(pools) == 2
    source.close()
    assert all(pool._closed for pool in pools)


def test_direct_io_is_decided_per_drive(tmp_path, monkeypatch):
    """One volume without direct I/O must not drag the other drive to the page cache."""
    real = mod.open_direct

    def refuse_root1(path):
        if Path(path).parent.name == "root1":
            raise OSError("no direct I/O on this volume")
        return real(path)

    monkeypatch.setattr(mod, "open_direct", refuse_root1)
    source, spec, _, _ = _source(tmp_path, read_ahead=3)
    try:
        assert source._open_of[1] == source._open_buffered
        _walk(source, spec, range(N))
    finally:
        source.close()


def test_build_source_reports_each_drive_mode_and_page_cache_reason(tmp_path, monkeypatch):
    """The console and runtime stats expose the same per-drive decision."""
    def refuse_root1(path):
        if Path(path).parent.name == "root1":
            raise OSError("no direct I/O on this volume")
        return open(path, "rb", buffering=0)

    monkeypatch.setattr(mod, "open_direct", refuse_root1)
    shard_dir, paths, roots = _striped_shards(tmp_path)
    spec = RamSource.layer_specs_from_paths(paths)
    output = io.StringIO()
    source, pinned = _build_source(
        shard_dir,
        N,
        spec,
        pin=False,
        console=Console(file=output, width=200, color_system=None),
        tier="disk",
        shard_paths=paths,
        layer_roots=roots,
        read_ahead=3,
    )
    try:
        assert source._open_of[0] is refuse_root1
        assert source._open_of[1] == source._open_buffered
        pool = type("Pool", (), {"n": 2, "nbytes": 0, "loads": 0})()
        stats = StreamRuntime(
            pool=pool,
            source=source,
            prefetcher=None,
            n_layers=N,
            pinned=pinned,
            device="cpu",
            tier="disk",
        ).stats()
        assert stats["drive_count"] == 2
        assert stats["direct_io_drives"] == 1
        assert sorted(stats["direct_io_roots"].values()) == [False, True]
        said = " ".join(strip_ansi(output.getvalue()).split())
        assert "root1" in said
        assert "page cache" in said
        assert "no direct I/O on this volume" in said
    finally:
        source.close()


def test_page_cache_warning_sanitizes_external_root_and_reason():
    output = io.StringIO()
    source = type(
        "Source",
        (),
        {"page_cache_reasons": {"[/]root\x1b": "[bold]refused[/]\x07"}},
    )()

    _say_page_cache_roots(
        source,
        Console(file=output, width=200, color_system=None),
    )

    said = strip_ansi(output.getvalue())
    assert "[/]root" in said
    assert "[bold]refused[/]" in said
    assert "\x1b" not in said
    assert "\x07" not in said


def test_submit_calls_on_done_once_per_job_after_its_outcome_is_set():
    pool = _RangeReaders(2)
    finished, calls = [], []
    try:
        outcomes = pool.submit(
            [lambda: finished.append(1), lambda: finished.append(2)],
            on_done=lambda: calls.append(len(finished)),
        )
        pool.wait(outcomes)
        deadline = time.monotonic() + 5
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        pool.close()
    assert sorted(finished) == [1, 2]
    assert len(calls) == 2 and all(count >= 1 for count in calls)


def test_on_done_sees_the_outcome_already_published():
    """The reader's no-lost-wake-up property rests on this order: the worker sets `done` and
    only then calls the reader's wake-up. Swapped, `_wake` can fire before the read is
    visible, the reader finds nothing to publish and sleeps out its 1 s poll — per layer. The
    test above only shows the callback runs after the WORK; this pins it to after `done`."""
    pool = _RangeReaders(1)
    release = threading.Event()
    holder, seen = [], []
    try:
        outcomes = pool.submit(
            [lambda: release.wait(5)],
            on_done=lambda: seen.append(holder[0].done.is_set()),
        )
        holder.append(outcomes[0])
        release.set()
        pool.wait(outcomes)
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        pool.close()
    assert seen == [True]
