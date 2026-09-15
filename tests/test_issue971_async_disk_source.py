"""#971 — the disk tier reads on a background thread, not on the compute thread."""

import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from soup_cli.utils.async_disk_source import AsyncDiskSource  # noqa: E402
from soup_cli.utils.layer_stream_runtime import DiskSource, RamSource  # noqa: E402

N_LAYERS = 4


def _shards(tmp_path: Path, n_layers: int = N_LAYERS) -> str:
    """A shard per layer, NF4-shaped: mixed uint8 and float32, plus a bf16 norm.

    ``::nested_offset`` is a SCALAR (shape ``()``), and it belongs here because
    it is the shape production has: under double quantisation — the default —
    every quantised weight carries one, 7 of the 30 tensors in a real
    decoder-layer shard. This fixture was NF4-shaped in dtype only, and the
    gap was not academic — ``read_into`` viewed to uint8 before flattening,
    torch refuses that on a 0-dim tensor, and the whole 4-bit disk tier failed
    on layer 0 the moment `_build_source` started using this source.
    """
    from soup_cli.utils.layer_shard import layer_shard_path

    out = tmp_path / "shards"
    out.mkdir()
    torch.manual_seed(927)
    for idx in range(n_layers):
        save_file(
            {
                "self_attn.q_proj.weight": torch.randint(
                    0, 255, (64, 32), dtype=torch.uint8
                ),
                "self_attn.q_proj.weight::absmax": torch.rand(16, dtype=torch.float32),
                "self_attn.q_proj.weight::nested_offset": torch.tensor(
                    0.125 * (idx + 1), dtype=torch.float32
                ),
                "input_layernorm.weight": torch.rand(64, dtype=torch.bfloat16),
            },
            layer_shard_path(str(out), idx),
        )
    return str(out)


def _spec(shard_dir: str):
    return RamSource.layer_specs_from_shards(shard_dir, N_LAYERS)


def _raw_bytes(tensor):
    """A tensor's bytes, comparable across dtypes AND ranks.

    Flatten first, then view: torch refuses a dtype-``view`` on a 0-dim tensor,
    so the bare ``tensor.view(torch.uint8)`` this file used could not compare a
    scalar at all — the same restriction that made ``read_into`` unable to fill
    one. Same idiom as test_issue971_safetensors_reader.py.
    """
    return tensor.reshape(-1).view(torch.uint8)


class TestByteIdentityAgainstTheShippedSource:
    """THE gate: same bytes as DiskSource, or the change is wrong."""

    @pytest.mark.parametrize("read_ahead", [1, 2, 4])
    def test_every_tensor_matches_disk_source(self, tmp_path, read_ahead):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        shipped = DiskSource(shard_dir, N_LAYERS, spec)
        ours = AsyncDiskSource(
            shard_dir, N_LAYERS, spec, read_ahead=read_ahead, pin=False
        )
        try:
            for idx in range(N_LAYERS):
                for name in spec[idx]:
                    theirs = shipped.get(idx, name)
                    mine = ours.get(idx, name)
                    assert mine.dtype == theirs.dtype, (idx, name)
                    assert mine.shape == theirs.shape, (idx, name)
                    assert torch.equal(
                        _raw_bytes(mine), _raw_bytes(theirs)
                    ), (idx, name)
        finally:
            ours.close()
            shipped.close()

    def test_a_scalar_sidecar_matches_disk_source_too(self, tmp_path):
        """The shape production has, named so it cannot be lost silently.

        `_shards` now carries a 0-dim `::nested_offset` beside the matrices, so
        the parametrised gate above already covers it — but only implicitly. If
        someone trims the fixture back, that gate goes on passing and this one
        fails, which is the point: the whole defect was a fixture that was
        NF4-shaped in dtype and not in rank. `DiskSource` is the reference
        precisely because it reads through `safe_open` and never had the
        restriction.
        """
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        scalars = [n for n, (shape, _) in spec[0].items() if shape == ()]
        assert scalars == ["self_attn.q_proj.weight::nested_offset"], (
            f"the fixture no longer carries exactly one scalar: {scalars}"
        )
        name = scalars[0]

        shipped = DiskSource(shard_dir, N_LAYERS, spec)
        ours = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            for idx in range(N_LAYERS):
                theirs, mine = shipped.get(idx, name), ours.get(idx, name)
                assert mine.shape == theirs.shape == ()
                assert torch.equal(_raw_bytes(mine), _raw_bytes(theirs)), idx
                # Per-layer values, so a source that staged one layer's scalar
                # and handed it back for every layer would fail here.
                assert float(mine) == pytest.approx(0.125 * (idx + 1))
        finally:
            ours.close()
            shipped.close()

    def test_depth_is_performance_never_semantics(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        shallow = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=1, pin=False)
        deep = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=4, pin=False)
        try:
            for idx in range(N_LAYERS):
                for name in spec[idx]:
                    mine, theirs = shallow.get(idx, name), deep.get(idx, name)
                    # `_raw_bytes` flattens, so bytes alone would pass for a
                    # source that returned the right data in the wrong shape.
                    # The parametrised gate above keeps this; so must this.
                    assert mine.shape == theirs.shape, (idx, name)
                    assert torch.equal(_raw_bytes(mine), _raw_bytes(theirs))
        finally:
            shallow.close()
            deep.close()


class TestItHoldsNoMapping:
    def test_no_safe_open_handle_is_ever_created(self, tmp_path, monkeypatch):
        """The commit charge #926 is about comes from mappings; we must hold none."""
        import safetensors

        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        opens = []

        real = safetensors.safe_open

        def counting(*args, **kwargs):
            opens.append(args[0] if args else kwargs.get("filename"))
            return real(*args, **kwargs)

        monkeypatch.setattr(safetensors, "safe_open", counting)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, pin=False)
        try:
            for idx in range(N_LAYERS):
                source.get(idx, "input_layernorm.weight")
        finally:
            source.close()
        assert opens == [], f"mapped {len(opens)} shard(s) after all"

    def test_nbytes_reports_staging_not_the_store(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            one_layer = sum(
                torch.empty(shape, dtype=getattr(torch, dtype)).numel()
                * torch.empty((), dtype=getattr(torch, dtype)).element_size()
                for shape, dtype in spec[0].values()
            )
            assert source.nbytes == 2 * one_layer
            assert source.disk_bytes > source.nbytes
        finally:
            source.close()


class TestFailuresAreLoudAndNeverHang:
    def test_a_read_error_surfaces_at_the_get_that_wanted_it(self, tmp_path):
        """A REAL read failure, not an injected one: the shard is gone.

        It is a deletion rather than the corruption this test used to write,
        because a shard rewritten after construction is now caught one step
        earlier by the identity check (see
        ``TestAShardThatChangesUnderTheRunIsRefused``) and never reaches
        ``read_into``. A missing file fails at ``open``, which is the read path
        proper, and is what an evicted cache or an unmounted share looks like.
        """
        from soup_cli.utils.layer_shard import layer_shard_path

        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=1, pin=False)
        try:
            source.get(0, "input_layernorm.weight")
            Path(layer_shard_path(shard_dir, 3)).unlink()
            with pytest.raises(OSError, match="layer_003"):
                for idx in range(1, N_LAYERS):
                    for name in spec[idx]:
                        source.get(idx, name)
        finally:
            source.close()

    def test_every_later_get_raises_the_same_stored_error(self, tmp_path):
        """Spec §Errors item 4's second half, against a REAL reader death.

        The first raise is the easy half. What an operator actually meets is
        the SECOND one, and the interesting case is a layer that is still
        RESIDENT: ``get`` must refuse it too rather than hand back bytes from a
        source whose reader is dead, and it must refuse with the error that
        killed the reader rather than a fresh one about closing or progress.
        """
        from soup_cli.utils.layer_shard import layer_shard_path

        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        # Depth N_LAYERS so every layer has its own slot and NOTHING is
        # evicted: the resident control below is vacuous at depth 1, where the
        # group holds one slot and layer 0 is gone by the time layer 3 fails.
        source = AsyncDiskSource(
            shard_dir, N_LAYERS, spec, read_ahead=N_LAYERS, pin=False
        )
        try:
            # Before the first get, so the only thing queued is the layer-0
            # prime: the reader cannot have read layer 3 yet.
            Path(layer_shard_path(shard_dir, 3)).unlink()
            resident = source.get(0, "input_layernorm.weight")
            assert resident is not None

            with pytest.raises(OSError, match="layer_003") as first:
                source.get(3, "input_layernorm.weight")
            stored = str(first.value)

            # The same layer again.
            with pytest.raises(OSError, match="layer_003") as again:
                source.get(3, "input_layernorm.weight")
            assert str(again.value) == stored

            # And a DIFFERENT layer, one that is still RESIDENT: it raises the
            # stored error rather than returning its (perfectly good) bytes.
            # The assertion above it is what stops this being vacuous.
            assert 0 in source._slot_of, "layer 0 must still be staged"
            with pytest.raises(OSError, match="layer_003") as other:
                source.get(0, "input_layernorm.weight")
            assert str(other.value) == stored
        finally:
            source.close()

    def test_a_dead_reader_refuses_instead_of_blocking(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=1, pin=False)
        try:
            source._fail(RuntimeError("reader died"))
            done = threading.Event()
            captured = {}

            def call():
                try:
                    source.get(1, "input_layernorm.weight")
                except BaseException as exc:  # noqa: BLE001 — recorded for the assert
                    captured["exc"] = exc
                done.set()

            threading.Thread(target=call, daemon=True).start()
            assert done.wait(timeout=10), "get() blocked after the reader died"
            exc = captured.get("exc")
            assert isinstance(exc, RuntimeError), repr(exc)
            # The stored error, not "closed" and not "no progress" — a bare
            # isinstance passes for both of those, which are different failures.
            assert "reader died" in str(exc), str(exc)
        finally:
            source.close()

    def test_close_is_idempotent_and_stops_the_thread(self, tmp_path):
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=False)
        source.get(0, "input_layernorm.weight")
        source.close()
        source.close()
        assert not source._thread.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            source.get(1, "input_layernorm.weight")

    def test_close_is_a_barrier_and_drops_the_pools_events(self, tmp_path):
        """``close()`` used to clear two fields unlocked and leave two behind.

        The two it left hold the buffer pools' ``torch.cuda.Event`` objects, so
        a closed source kept them alive; and clearing the other two outside the
        lock let a reader that outlived the 10 s join observe a half-torn source
        and store an ``IndexError`` that a later ``get`` reported as "list index
        out of range" instead of "closed".
        """

        class _Event:
            def synchronize(self):
                pass

        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(
            shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=1, pin=False
        )
        event = _Event()
        source.get(0, "input_layernorm.weight")
        source.release(0, event)
        assert event in source._drain, "the drain event must have been parked"

        source.close()
        assert source._slots == []
        assert source._slot_of == {}
        assert source._live == []
        assert source._drain == [], "a closed source still holds a pool event"

        # A release arriving after close must not resurrect anything.
        source.release(0, event)
        assert source._slot_of == {}

    def test_a_reader_that_outlives_the_join_cannot_resurrect_a_closed_source(
        self, tmp_path, monkeypatch
    ):
        """``close()``'s join has a 10 s timeout, so a reader still inside a
        cold read outlives the teardown. Posed directly rather than waited out:
        the flag is set and the state cleared while the reader is mid-read,
        which is exactly the ordering a timed-out join produces.

        Two properties at once — the reader must not publish into the cleared
        ``_slot_of``, and it must not blow up on the cleared ``_slots`` (which
        it did, storing an ``IndexError`` a later ``get`` reported as "list
        index out of range" rather than "closed").
        """
        import soup_cli.utils.async_disk_source as mod

        gate = threading.Event()
        real = mod.read_into

        def held(handle, entry, dst):
            assert gate.wait(timeout=10.0), "the test never released the reader"
            real(handle, entry, dst)

        monkeypatch.setattr(mod, "read_into", held)
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(
            shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=1, pin=False
        )
        try:
            deadline = time.monotonic() + 10.0
            while source._in_flight is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert source._in_flight == 0, "the reader never started layer 0"

            with source._ready:
                source._closed = True
                source._slots = []
                source._slot_of = {}
                source._live = []
                source._drain = []
            gate.set()
            source._thread.join(timeout=10.0)
            assert not source._thread.is_alive()
            assert source._slot_of == {}, "the reader published into a closed source"
            assert source._error is None, repr(source._error)
        finally:
            gate.set()
            source.close()

    def test_a_reader_waiting_on_a_drain_survives_the_teardown_too(self, tmp_path):
        """The other side of the same window, and the one that actually bit.

        A reader parked in ``draining.synchronize()`` has NOT yet resolved its
        staging slot. If it resolves it after the teardown it indexes an empty
        list, and the ``IndexError`` is stored as the source's error — so the
        next ``get`` reports "list index out of range" where it should report
        "closed". No ``read_into`` stub here: the block is a drain event whose
        ``synchronize`` waits, which is the real shape of the 10 s join timing
        out on a cold shard.
        """
        gate = threading.Event()

        class _SlowEvent:
            def synchronize(self):
                assert gate.wait(timeout=10.0), "the test never released the drain"

        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(
            shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=1, pin=False
        )
        try:
            source.get(0, "input_layernorm.weight")
            source.release(0, _SlowEvent())
            with source._ready:
                source._queue = [1]
                source._ready.notify_all()

            deadline = time.monotonic() + 10.0
            while source._in_flight != 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert source._in_flight == 1, "the reader never took layer 1"

            with source._ready:
                source._closed = True
                source._slots = []
                source._slot_of = {}
                source._live = []
                source._drain = []
            gate.set()
            source._thread.join(timeout=10.0)
            assert not source._thread.is_alive()
            assert source._error is None, repr(source._error)
        finally:
            gate.set()
            source.close()

    def test_a_spec_that_disagrees_with_the_header_is_refused(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        spec[0]["input_layernorm.weight"] = ((128,), "bfloat16")
        with pytest.raises(ValueError, match="disagrees with"):
            AsyncDiskSource(shard_dir, N_LAYERS, spec, pin=False)

    def test_read_ahead_is_bounded(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        for bad in (0, 9):
            with pytest.raises(ValueError, match="stream_read_ahead"):
                AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=bad, pin=False)
        with pytest.raises(ValueError, match="must be an int"):
            AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=True, pin=False)


def _get_on_a_thread(source, idx, name, timeout=20.0):
    """Call ``get`` where a REGRESSION fails the test instead of hanging it.

    Every test below drives a reader that is deliberately wedged or dead. If
    the liveness checks stop working, a direct ``get`` would block the whole
    suite forever; on a worker thread it times out and the assertion names it.
    """
    done = threading.Event()
    captured = {}

    def call():
        try:
            captured["value"] = source.get(idx, name)
        except BaseException as exc:  # noqa: BLE001 — recorded for the assert
            captured["exc"] = exc
        done.set()

    threading.Thread(target=call, daemon=True).start()
    assert done.wait(timeout=timeout), (
        f"get({idx}, {name!r}) never returned — the liveness checks in "
        f"AsyncDiskSource.get did not fire"
    )
    return captured


class TestTheLivenessChecksCanActuallyFire:
    """The guard this replaced could not fire, and that was the whole problem.

    The old conjunction required ``idx`` to be in none of ``_queue`` /
    ``_in_flight`` / ``_slot_of``, but the demand push puts it in the queue
    before the wait and the reader moves it through those three states under
    the lock — so on the single-consumer path it was always in exactly one of
    them. Dead code standing where the spec names its anti-hang mitigation: a
    wedged reader made these very tests HANG rather than fail.
    """

    def test_a_read_that_never_returns_is_refused_by_the_limit(
        self, tmp_path, monkeypatch
    ):
        """The reachable wedge. A reader blocked inside ``read_into`` keeps the
        layer legitimately in flight, so no amount of state-inspection can tell
        it from a slow read — only elapsed time can."""
        import soup_cli.utils.async_disk_source as mod

        blocked = threading.Event()

        def never_returns(handle, entry, dst):
            blocked.wait(timeout=30.0)

        monkeypatch.setattr(mod, "read_into", never_returns)
        monkeypatch.setattr(mod, "_MAX_READ_SECONDS", 0.25)

        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=False)
        try:
            captured = _get_on_a_thread(source, 0, "input_layernorm.weight")
            exc = captured.get("exc")
            assert isinstance(exc, RuntimeError), repr(captured)
            message = str(exc)
            assert "0 s limit" in message, message
            assert "reading layer 0" in message, message
            assert "is waiting behind it" in message, message
        finally:
            blocked.set()
            source.close()

    def test_a_slow_read_under_the_limit_is_not_refused(self, tmp_path, monkeypatch):
        """The control that makes the limit a wedge detector rather than a
        timeout: a read that is slow but finishes must still be served."""
        import soup_cli.utils.async_disk_source as mod

        real = mod.read_into

        def slow(handle, entry, dst):
            time.sleep(0.2)
            real(handle, entry, dst)

        monkeypatch.setattr(mod, "read_into", slow)
        monkeypatch.setattr(mod, "_MAX_READ_SECONDS", 5.0)

        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=False)
        try:
            captured = _get_on_a_thread(source, 0, "input_layernorm.weight")
            assert "exc" not in captured, repr(captured["exc"])
            assert captured["value"] is not None
        finally:
            source.close()

    def test_a_reader_thread_that_exits_silently_is_refused(
        self, tmp_path, monkeypatch
    ):
        """A backstop, and it is labelled one in the docstring: no path in the
        shipped ``_run`` can exit without ``_fail`` or ``_closed``. It exists
        for the paths that are not in that file — this monkeypatch stands in
        for a future second ``return`` or a thread killed from outside."""
        monkeypatch.setattr(AsyncDiskSource, "_run", lambda self: None)

        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=False)
        try:
            assert source._error is None, "the reader must have said nothing"
            captured = _get_on_a_thread(source, 0, "input_layernorm.weight")
            exc = captured.get("exc")
            assert isinstance(exc, RuntimeError), repr(captured)
            message = str(exc)
            assert "exited without recording an error" in message, message
            assert "layer 0 still wanted" in message, message
        finally:
            source.close()


class TestAShardThatChangesUnderTheRunIsRefused:
    """Headers are parsed once; the file is re-opened per read.

    ``read_into`` checks how many bytes arrived, never that they came from the
    same file, so a shard replaced in place by one of the SAME SIZE and a
    different layout was read at stale offsets and trained on with no error
    anywhere. ``layer_shard`` re-shards into the same directory with
    ``os.replace`` whenever the base's fingerprint moves, and this source keeps
    no handle to block that, so the window is the whole run.
    """

    @staticmethod
    def _rewrite_same_size(path: Path) -> None:
        """Flip one DATA byte and bump mtime; size, inode and device unchanged.

        Deliberately still a valid safetensors file of the same length: the
        point is that a file which is perfectly readable is refused because it
        is not the file whose byte ranges this source holds. That also leaves
        mtime as the only field that differs, which pins the weakest of the
        four rather than letting size carry the test.
        """
        import os

        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        was = path.stat()
        path.write_bytes(bytes(data))
        os.utime(path, ns=(was.st_mtime_ns + 10**9, was.st_mtime_ns + 10**9))
        assert path.stat().st_size == was.st_size, "the rewrite changed the size"
        assert path.stat().st_mtime_ns != was.st_mtime_ns

    def test_a_same_size_rewrite_is_refused_by_name(self, tmp_path):
        from soup_cli.utils.layer_shard import layer_shard_path

        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=1, pin=False)
        try:
            source.get(0, "input_layernorm.weight")
            self._rewrite_same_size(Path(layer_shard_path(shard_dir, 3)))
            with pytest.raises(RuntimeError, match="changed on disk since its header"):
                for idx in range(1, N_LAYERS):
                    for name in spec[idx]:
                        source.get(idx, name)
        finally:
            source.close()

    def test_the_check_runs_before_the_read_so_a_truncation_says_what_happened(
        self, tmp_path
    ):
        """A SHORTER replacement used to surface as "short read", which names
        the symptom. The identity check runs first and names the cause."""
        from soup_cli.utils.layer_shard import layer_shard_path

        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=1, pin=False)
        try:
            source.get(0, "input_layernorm.weight")
            Path(layer_shard_path(shard_dir, 3)).write_bytes(b"corrupt")
            with pytest.raises(RuntimeError, match="changed on disk since its header"):
                for idx in range(1, N_LAYERS):
                    for name in spec[idx]:
                        source.get(idx, name)
        finally:
            source.close()

    def test_an_unchanged_shard_never_trips_it(self, tmp_path):
        """The control. Every layer, twice, forward then backward — the reader
        re-opens each shard on every read, so a check that compared the wrong
        thing would fire here."""
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            for sweep in (range(N_LAYERS), reversed(range(N_LAYERS))):
                for idx in sweep:
                    for name in spec[idx]:
                        assert source.get(idx, name) is not None
        finally:
            source.close()


class TestPinnedStagingRefusesAnUnreleasedBorrowWithoutAGpu:
    """The CPU-runnable half of Task 5's safety contract.

    The CUDA class below needs a card because the HAZARD is a draining device
    copy. The REFUSAL does not: it reads ``self.pinned``, a plain attribute.
    Without this class every test that runs in CI constructs with
    ``pin=False``, so deleting the refusal branch outright left CI green —
    and that branch is the difference between a silently wrong gradient and a
    loud stop.
    """

    @staticmethod
    def _as_if_pinned(tmp_path):
        """Pageable staging that reports itself pinned.

        ``pin=True`` would need a CUDA device to allocate; the branch under
        test never looks at the memory, only at the flag. The mirror image of
        ``TestPinningRefusesPageableMemory``, which monkeypatches the other way.
        """
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(
            shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=False
        )
        source.pinned = True
        return source

    def test_a_second_get_without_release_is_refused(self, tmp_path):
        source = self._as_if_pinned(tmp_path)
        try:
            source.get(0, "input_layernorm.weight")
            with pytest.raises(RuntimeError, match="release") as excinfo:
                source.get(1, "input_layernorm.weight")
            message = str(excinfo.value)
            assert "_release_source" in message, message
            assert "0" in message and "1" in message, message
        finally:
            source.close()

    def test_the_compliant_sequence_is_unaffected(self, tmp_path):
        source = self._as_if_pinned(tmp_path)
        try:
            first = source.get(0, "input_layernorm.weight")
            source.release(0, None)
            second = source.get(1, "input_layernorm.weight")
            assert second.shape == first.shape
        finally:
            source.close()

    def test_the_pageable_control_still_takes_the_implicit_release(self, tmp_path):
        """Genuinely ``pin=False``: the same sequence must NOT raise, or the
        refusal is a block on all traffic rather than on the hazard."""
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(
            shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=False
        )
        try:
            assert not source.pinned
            source.get(0, "input_layernorm.weight")
            assert source.get(1, "input_layernorm.weight") is not None
        finally:
            source.close()

    def test_the_refusal_orders_layers_numerically(self, tmp_path, monkeypatch):
        """Operator-facing text: ``sorted`` over the STRINGS prints "10, 2".

        Two simultaneous borrows cannot be reached through ``get`` — the first
        unreleased one is what the refusal is about — so the reader is stubbed
        out and the state posed directly. ``_hold`` is a pure function of it.
        """
        monkeypatch.setattr(AsyncDiskSource, "_run", lambda self: None)
        shard_dir = _shards(tmp_path, n_layers=12)
        spec = RamSource.layer_specs_from_shards(shard_dir, 12)
        source = AsyncDiskSource(shard_dir, 12, spec, read_ahead=8, pin=False)
        try:
            source.pinned = True
            flat = source._group_slots[0]
            assert len(flat) >= 2, flat
            source._slot_of = {2: flat[0], 10: flat[1]}
            source._live[flat[0]] = True
            source._live[flat[1]] = True
            with source._ready:
                with pytest.raises(RuntimeError, match="release") as excinfo:
                    source._hold(1)
            message = str(excinfo.value)
            assert "layer(s) 2, 10" in message, message
        finally:
            source.close()


# ==========================================================================
# the hazards pin=False cannot express
# ==========================================================================
_NO_CUDA = not torch.cuda.is_available()


def _uniform_shards(tmp_path: Path, n_layers: int, size: int) -> str:
    """One tensor per layer, every byte equal to the layer index.

    A leak therefore names the layer it came from, which a random fixture
    cannot do: `torch.equal` says "different", this says "layer 6 is sitting in
    layer 0's buffer".
    """
    from soup_cli.utils.layer_shard import layer_shard_path

    out = tmp_path / "uniform"
    out.mkdir()
    for idx in range(n_layers):
        save_file(
            {"w": torch.full((size,), idx, dtype=torch.uint8)},
            layer_shard_path(str(out), idx),
        )
    return str(out)


@pytest.mark.skipif(_NO_CUDA, reason="the hazard is a CUDA copy draining out of pinned host memory")
class TestTheDeviceGetsTheLayerItAskedFor:
    """THE gate for the recycle-under-an-in-flight-copy defect.

    Every other test in this file runs ``pin=False``, where a host-to-device
    copy is synchronous and the hazard cannot exist — so the suite was blind to
    it by construction while ``pin=True`` is the default AND the production
    setting. This drives the REAL consumer (``LayerBufferPool`` +
    ``StreamPrefetcher``) and checks the bytes that reach the DEVICE.
    """

    @pytest.mark.parametrize("read_ahead", [1, 2, 4, 8])
    def test_no_layer_reaches_the_device_holding_another_layers_weights(
        self, tmp_path, read_ahead
    ):
        from soup_cli.utils.layer_shard import layer_shard_path
        from soup_cli.utils.layer_stream_runtime import LayerBufferPool, StreamPrefetcher

        n_layers = 8
        shard_dir = _uniform_shards(tmp_path, n_layers, 8 * 1024 * 1024)
        spec = RamSource.layer_specs_from_shards(shard_dir, n_layers)
        source = AsyncDiskSource(
            shard_dir, n_layers, spec, read_ahead=read_ahead, pin=True
        )
        try:
            assert source.pinned, "the hazard needs genuinely pinned staging"
            pool = LayerBufferPool(spec[0], n_buffers=2, device="cuda")
            stream = torch.cuda.Stream()
            prefetcher = StreamPrefetcher(pool, source, n_layers, stream)

            # Warm the page cache: a reader blocked on cold I/O cannot run
            # ahead far enough to overwrite anything, which would hide the bug.
            for idx in range(n_layers):
                Path(layer_shard_path(shard_dir, idx)).read_bytes()

            # Put the GPU behind. Prefetching only pays off when it is, and the
            # copy only stays in flight long enough to be clobbered when it is.
            hog = torch.randn(4096, 4096, device="cuda")
            for _ in range(200):
                hog = hog @ hog.clamp(-1, 1)

            seen = []
            prefetcher.prime()
            for idx in range(n_layers):
                buffers = pool.wait(idx)  # a GPU-side wait_event, not a host one
                prefetcher.advance(idx)  # -> load_async(idx+1) -> source.get(...)
                seen.append(buffers["w"].clone())
            torch.cuda.synchronize()

            wrong = {
                idx: torch.unique(got).tolist()
                for idx, got in enumerate(seen)
                if torch.unique(got).tolist() != [idx]
            }
            assert not wrong, (
                f"read_ahead={read_ahead}: {len(wrong)} of {n_layers} layers reached "
                f"the device holding another layer's weights — {wrong} (each value is "
                f"the layer the bytes actually came from). The staging buffer was "
                f"recycled while its copy was still draining."
            )
        finally:
            source.close()

    def test_the_same_harness_is_clean_through_the_shipped_sources(self, tmp_path):
        """A control: if this ever fails, the harness is wrong, not the source."""
        from soup_cli.utils.layer_stream_runtime import LayerBufferPool, StreamPrefetcher

        n_layers = 8
        shard_dir = _uniform_shards(tmp_path, n_layers, 8 * 1024 * 1024)
        spec = RamSource.layer_specs_from_shards(shard_dir, n_layers)
        # Built one at a time and closed: `DiskSource` holds a `safe_open`
        # mapping per layer, and on Windows a live mapping keeps the file open
        # against `tmp_path` cleanup — the very situation #926 is about.
        for make in (
            lambda: DiskSource(shard_dir, n_layers, spec),
            lambda: RamSource(shard_dir, n_layers, spec, pin=True),
        ):
            source = make()
            try:
                pool = LayerBufferPool(spec[0], n_buffers=2, device="cuda")
                stream = torch.cuda.Stream()
                prefetcher = StreamPrefetcher(pool, source, n_layers, stream)
                hog = torch.randn(4096, 4096, device="cuda")
                for _ in range(200):
                    hog = hog @ hog.clamp(-1, 1)
                seen = []
                prefetcher.prime()
                for idx in range(n_layers):
                    buffers = pool.wait(idx)
                    prefetcher.advance(idx)
                    seen.append(buffers["w"].clone())
                torch.cuda.synchronize()
                for idx, got in enumerate(seen):
                    assert torch.unique(got).tolist() == [idx], (
                        f"{type(source).__name__} layer {idx} is wrong — the "
                        f"harness, not the source under test, is at fault"
                    )
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    close()

    def test_pinned_is_measured_not_asserted(self, tmp_path):
        """``pinned=True`` must mean the staging really is page-locked.

        ``RamSource`` checks ``dst.is_pinned()`` because a box that hands back
        pageable memory would otherwise report the fast path while silently
        paying the ~97% -> ~79% GPU-utilisation cost of a synchronous copy.
        """
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=True)
        try:
            assert source.pinned is True
            staged = [dst for slot in source._slots for dst in slot.values()]
            assert staged, "no staging was allocated"
            assert all(dst.is_pinned() for dst in staged), (
                "pinned=True but torch returned pageable memory"
            )
        finally:
            source.close()


class TestTheReadHappensAhead:
    """A synchronous implementation passes every other test in this file.

    ``get`` returning the right bytes is necessary and not sufficient: the
    point of the whole source is that the read is OFF the compute thread and
    already done before the consumer asks. Nothing pinned that.
    """

    def test_reads_run_on_the_reader_thread_not_the_caller(self, tmp_path, monkeypatch):
        import soup_cli.utils.async_disk_source as module

        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        threads = []
        real = module.read_into

        def recording(handle, entry, tensor):
            threads.append(threading.current_thread())
            return real(handle, entry, tensor)

        monkeypatch.setattr(module, "read_into", recording)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            source.get(0, "input_layernorm.weight")
            assert threads, "nothing was read at all"
            caller = threading.current_thread()
            offenders = sorted({t.name for t in threads if t is caller})
            assert not offenders, (
                f"the read ran on the calling thread ({offenders}) — this source "
                f"exists to keep it off the compute thread"
            )
            assert {t.name for t in threads} == {"soup-layer-reader"}
        finally:
            source.close()

    def test_the_next_layer_arrives_before_anyone_asks_for_it(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            source.get(0, "input_layernorm.weight")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and 1 not in source._slot_of:
                time.sleep(0.005)
            assert 1 in source._slot_of, (
                "layer 1 was never staged, though nothing asked for it yet — the "
                "source read on demand instead of ahead"
            )
        finally:
            source.close()


class TestTheShapeBuildSourceActuallyPasses:
    """``_build_source`` hands this source per-layer specs whose trailing
    entries are the vocabulary-sized embed / lm_head shards, with ENTIRELY
    different tensor keys (layer_stream_runtime.py, ``source_specs``). Sizing
    every staging slot from layer 0 made the reader raise ``KeyError`` on the
    reader thread at the first forward turnaround of any real model, poisoning
    the source permanently, where ``DiskSource`` simply returns the tensor.
    """

    @staticmethod
    def _hetero(tmp_path: Path):
        from soup_cli.utils.layer_shard import layer_shard_path

        out = tmp_path / "hetero"
        out.mkdir()
        torch.manual_seed(927)
        for idx in range(2):
            save_file(
                {"self_attn.q_proj.weight": torch.rand(8, 4, dtype=torch.float32)},
                layer_shard_path(str(out), idx),
            )
        big = out / "embed.safetensors"
        save_file(
            {"model.embed_tokens.weight": torch.rand(64, 4, dtype=torch.float32)},
            str(big),
        )
        specs = [
            {"self_attn.q_proj.weight": ((8, 4), "float32")},
            {"self_attn.q_proj.weight": ((8, 4), "float32")},
            {"model.embed_tokens.weight": ((64, 4), "float32")},
        ]
        paths = [
            layer_shard_path(str(out), 0),
            layer_shard_path(str(out), 1),
            str(big),
        ]
        return str(out), specs, paths

    def test_the_large_layer_comes_back_exactly_as_disk_source_returns_it(
        self, tmp_path
    ):
        shard_dir, specs, paths = self._hetero(tmp_path)
        shipped = DiskSource(shard_dir, 3, specs, shard_paths=paths)
        ours = AsyncDiskSource(
            shard_dir, 3, specs, shard_paths=paths, read_ahead=2, pin=False
        )
        try:
            for idx, per_layer in enumerate(specs):
                for name in per_layer:
                    theirs = shipped.get(idx, name)
                    mine = ours.get(idx, name)
                    assert mine.shape == theirs.shape, (idx, name)
                    assert torch.equal(mine, theirs), (idx, name)
        finally:
            ours.close()
            shipped.close()

    def test_staging_is_reported_honestly_for_every_distinct_spec(self, tmp_path):
        """Extra host memory is fine; misreporting it is not.

        One slot per distinct spec beyond the decoder's own depth: the embed
        shard is one layer, so depth past 1 there would buy nothing and cost a
        whole vocabulary matrix.
        """
        shard_dir, specs, paths = self._hetero(tmp_path)
        source = AsyncDiskSource(
            shard_dir, 3, specs, shard_paths=paths, read_ahead=2, pin=False
        )
        try:
            decoder = 8 * 4 * 4
            embed = 64 * 4 * 4
            assert source.nbytes == 2 * decoder + embed
            observed = sum(
                dst.numel() * dst.element_size()
                for slot in source._slots
                for dst in slot.values()
            )
            assert source.nbytes == observed, "nbytes disagrees with what was allocated"
        finally:
            source.close()

    def test_disk_bytes_counts_what_the_headers_say(self, tmp_path):
        """Not recomputed from the spec through a fourth dtype-size table."""
        shard_dir, specs, paths = self._hetero(tmp_path)
        source = AsyncDiskSource(
            shard_dir, 3, specs, shard_paths=paths, read_ahead=2, pin=False
        )
        try:
            assert source.disk_bytes == 2 * (8 * 4 * 4) + 64 * 4 * 4
        finally:
            source.close()


# ==========================================================================
# read_ahead must actually read ahead
# ==========================================================================
def _settle(source, timeout: float = 10.0) -> None:
    """Wait until the reader has nothing queued and nothing in flight.

    Depth is a property of the pipeline at rest. Sampling while the reader is
    mid-read would measure this machine's timing, not the source's design.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not source._queue and source._in_flight is None:
            return
        time.sleep(0.002)
    raise AssertionError("the reader never went idle")


def _deep_shards(tmp_path: Path, n_layers: int) -> str:
    from soup_cli.utils.layer_shard import layer_shard_path

    out = tmp_path / "deep"
    out.mkdir()
    for idx in range(n_layers):
        save_file(
            {"w": torch.full((256,), idx % 251, dtype=torch.uint8)},
            layer_shard_path(str(out), idx),
        )
    return str(out)


class TestReadAheadActuallyReadsAhead:
    """``read_ahead`` is a DEPTH, and depth has to be measured, not declared.

    The parent commit armed exactly one target — ``idx + direction`` — so the
    reader was at most ONE layer ahead of the consumer whatever the setting
    said, and ``read_ahead=8`` charged eight layers of pinned host memory for a
    one-deep pipeline. Every test in this file passed. That is what makes a
    setting that does not do what it says invisible (#748), and it matters most
    exactly where this source is used: on a cold disk the read is roughly 12x
    the compute, so one layer of lookahead can only hide one layer's read.

    The reachable depth is ``read_ahead - 1``, not ``read_ahead``: one slot is
    always the one the consumer is holding.
    """

    N_LAYERS = 16

    @pytest.mark.parametrize("read_ahead", [1, 2, 4, 8])
    def test_forward_depth_scales_with_the_setting(self, tmp_path, read_ahead):
        shard_dir = _deep_shards(tmp_path, self.N_LAYERS)
        spec = RamSource.layer_specs_from_shards(shard_dir, self.N_LAYERS)
        source = AsyncDiskSource(
            shard_dir, self.N_LAYERS, spec, read_ahead=read_ahead, pin=False
        )
        try:
            deepest = 0
            for idx in range(self.N_LAYERS - read_ahead):
                source.get(idx, "w")
                _settle(source)
                ahead = sum(1 for layer in source._slot_of if layer > idx)
                deepest = max(deepest, ahead)
            assert deepest == read_ahead - 1, (
                f"read_ahead={read_ahead} staged at most {deepest} layers ahead of "
                f"demand, expected {read_ahead - 1}. The setting charges "
                f"{read_ahead} layers of host memory for the depth it promises."
            )
        finally:
            source.close()

    def test_depth_is_a_window_that_slides_not_a_one_off_burst(self, tmp_path):
        """Depth has to be SUSTAINED, or the pipeline drains after one step."""
        shard_dir = _deep_shards(tmp_path, self.N_LAYERS)
        spec = RamSource.layer_specs_from_shards(shard_dir, self.N_LAYERS)
        source = AsyncDiskSource(shard_dir, self.N_LAYERS, spec, read_ahead=4, pin=False)
        try:
            for idx in range(self.N_LAYERS - 4):
                source.get(idx, "w")
                _settle(source)
                staged = sorted(layer for layer in source._slot_of if layer > idx)
                assert staged == [idx + 1, idx + 2, idx + 3], (
                    f"at layer {idx} the lookahead window was {staged}, not the "
                    f"three consecutive layers the consumer is about to ask for"
                )
        finally:
            source.close()


class TestTheDirectionIsFollowedNotAssumed:
    """The direction half had no test: reverting ``_note_direction`` to
    always-forward passed all 21. On the backward recompute an always-forward
    plan targets layers the consumer has just been through, which are still
    resident, so it queues nothing and the pipeline runs dry exactly half the
    time.
    """

    N_LAYERS = 16

    def test_the_backward_walk_is_prefetched_as_deeply_as_the_forward_one(
        self, tmp_path
    ):
        shard_dir = _deep_shards(tmp_path, self.N_LAYERS)
        spec = RamSource.layer_specs_from_shards(shard_dir, self.N_LAYERS)
        source = AsyncDiskSource(shard_dir, self.N_LAYERS, spec, read_ahead=4, pin=False)
        try:
            deepest = 0
            for idx in range(self.N_LAYERS - 1, 3, -1):
                source.get(idx, "w")
                _settle(source)
                behind = sum(1 for layer in source._slot_of if layer < idx)
                deepest = max(deepest, behind)
            assert deepest == 3, (
                f"walking DOWN, the deepest lookahead was {deepest} layers, expected "
                f"3. The reader is prefetching in the direction the consumer came "
                f"from, not the one it is going."
            )
        finally:
            source.close()

    def test_every_backward_target_is_staged_before_it_is_asked_for(self, tmp_path):
        """The reviewer's shape: after get(idx) on the way down, is idx-1 there?"""
        shard_dir = _deep_shards(tmp_path, 8)
        spec = RamSource.layer_specs_from_shards(shard_dir, 8)
        source = AsyncDiskSource(shard_dir, 8, spec, read_ahead=2, pin=False)
        try:
            for idx in range(8):
                source.get(idx, "w")
            missed = []
            for idx in range(7, 0, -1):
                source.get(idx, "w")
                _settle(source)
                if (idx - 1) not in source._slot_of:
                    missed.append(idx - 1)
            assert not missed, (
                f"layers {missed} were not staged before the backward walk reached "
                f"them — each one is a read the consumer had to wait on"
            )
        finally:
            source.close()


class TestPinningRefusesPageableMemory:
    """The mirror of RamSource's guard (tests/test_qwen35_streaming.py) — the
    branch existed with no test, so a box quietly handing back pageable memory
    would have reported the fast path while paying the ~97% -> ~79% cost.
    """

    def test_a_pageable_allocation_is_refused_not_reported_as_pinned(
        self, tmp_path, monkeypatch
    ):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        real_empty = torch.empty

        def _pageable_empty(*args, **kwargs):
            allocation = dict(kwargs)
            allocation["pin_memory"] = False
            return real_empty(*args, **allocation)

        monkeypatch.setattr(torch, "empty", _pageable_empty)
        with pytest.raises(RuntimeError, match="returned pageable memory"):
            AsyncDiskSource(shard_dir, N_LAYERS, spec, pin=True)


class TestDepthSurvivesTheTurnaround:
    """Depth measured on a fresh one-way walk is not the depth training gets.

    Training walks forward, backward, forward, forever. The single-direction
    tests above pass while delivered depth decays across the turnaround: FIFO
    rotation is only in phase with the walk while the walk keeps going the same
    way, so after a reversal it starts evicting layers still inside the
    lookahead window -- the very next layer wanted, in the traced case. Measured
    over 4 sweeps of 32 layers before the eviction policy was fixed: 2 of 3
    sustained at read_ahead=4, and 5 of 7 at 8 with demand misses in both
    directions.

    This drives the PRODUCTION access pattern -- ``get`` immediately followed by
    ``release``, which is what all four ``_release_source`` call sites do.
    """

    N_LAYERS = 32
    SWEEPS = 4

    @staticmethod
    def _sweep(source, read_ahead, n_layers, sweeps):
        """Walk up and down, returning (sustained depth, demand misses)."""
        depths = []
        misses = 0
        for sweep in range(sweeps):
            descending = sweep % 2 == 1
            walk = range(n_layers - 1, -1, -1) if descending else range(n_layers)
            for idx in walk:
                staged_before_demand = idx in source._slot_of
                source.get(idx, "w")
                source.release(idx, None)
                _settle(source)
                if descending:
                    depth = sum(1 for layer in source._slot_of if layer < idx)
                    room = idx >= read_ahead - 1
                else:
                    depth = sum(1 for layer in source._slot_of if layer > idx)
                    room = idx + read_ahead - 1 < n_layers
                # The first sweep is cold, and the ends of a walk simply run
                # out of layers to stage -- neither is the steady state.
                if sweep >= 1 and room:
                    depths.append(depth)
                    if not staged_before_demand:
                        misses += 1
        return min(depths), misses

    @pytest.mark.parametrize("read_ahead", [2, 4, 8])
    def test_full_depth_is_sustained_across_repeated_reversals(
        self, tmp_path, read_ahead
    ):
        shard_dir = _deep_shards(tmp_path, self.N_LAYERS)
        spec = RamSource.layer_specs_from_shards(shard_dir, self.N_LAYERS)
        source = AsyncDiskSource(
            shard_dir, self.N_LAYERS, spec, read_ahead=read_ahead, pin=False
        )
        try:
            sustained, misses = self._sweep(
                source, read_ahead, self.N_LAYERS, self.SWEEPS
            )
            assert sustained == read_ahead - 1, (
                f"read_ahead={read_ahead} peaks at the depth it promises on a "
                f"one-way walk but only SUSTAINS {sustained} of {read_ahead - 1} "
                f"across {self.SWEEPS} reversals. The eviction policy is dropping "
                f"layers still inside the lookahead window."
            )
            assert misses == 0, (
                f"{misses} layers were demanded before they were staged, across "
                f"{self.SWEEPS} sweeps -- each one is a read the consumer waited on "
                f"that the configured depth had already paid for"
            )
        finally:
            source.close()

    def test_a_layer_inside_the_window_is_not_the_eviction_victim(self, tmp_path):
        """The mechanism, asserted directly rather than through its symptom."""
        shard_dir = _deep_shards(tmp_path, self.N_LAYERS)
        spec = RamSource.layer_specs_from_shards(shard_dir, self.N_LAYERS)
        source = AsyncDiskSource(shard_dir, self.N_LAYERS, spec, read_ahead=4, pin=False)
        try:
            for idx in range(self.N_LAYERS):
                source.get(idx, "w")
                source.release(idx, None)
            evicted_while_wanted = []
            for idx in range(self.N_LAYERS - 1, -1, -1):
                source.get(idx, "w")
                source.release(idx, None)
                _settle(source)
                window = {idx - step for step in range(4) if idx - step >= 0}
                staged = set(source._slot_of)
                missing = sorted(window - staged)
                if missing and idx >= 3:
                    evicted_while_wanted.append((idx, missing))
            assert not evicted_while_wanted, (
                f"walking down, these layers were inside the lookahead window but "
                f"not staged: {evicted_while_wanted[:4]}"
            )
        finally:
            source.close()


class TestDepthOnTheShapeProductionActuallyHas:
    """Every depth test above uses ONE spec group. A real model has three.

    ``install_streaming._prime`` calls ``large_pool.load_async(embed_key, ...)``
    BEFORE ``prefetcher.prime()``, so the first ``source.get`` of a run is the
    embed -- its own spec group, at an index above every decoder layer. With a
    single shared anchor that first call set it forever: every later decoder
    ``get`` took the cross-group early return, ``_direction`` stuck at +1 through
    the whole backward pass, and the eviction preference degenerated to the
    plain FIFO it exists to replace. Measured on this shape before the per-group
    anchor: backward sustained depth 0 at read_ahead 4 AND 8, with 50 of 58 and
    34 of 50 layers demanded before they were staged.

    Homogeneous tests cannot see any of it, which is the point of this class.
    """

    N_DECODER = 20
    STEPS = 2

    @staticmethod
    def _hetero_fixture(tmp_path: Path, n_decoder: int):
        """Decoder shards plus embed and lm_head, as ``_build_source`` builds it."""
        from soup_cli.utils.layer_shard import layer_shard_path

        out = tmp_path / "production"
        out.mkdir()
        for idx in range(n_decoder):
            save_file(
                {"w": torch.full((256,), idx % 251, dtype=torch.uint8)},
                layer_shard_path(str(out), idx),
            )
        large = [
            (n_decoder, "model.embed_tokens.weight"),
            (n_decoder + 1, "lm_head.weight"),
        ]
        specs = [{"w": ((256,), "uint8")} for _ in range(n_decoder)]
        paths = [layer_shard_path(str(out), idx) for idx in range(n_decoder)]
        for index, key in large:
            path = out / f"large_{index}.safetensors"
            save_file({key: torch.full((1024,), 7, dtype=torch.uint8)}, str(path))
            specs.append({key: ((1024,), "uint8")})
            paths.append(str(path))
        return str(out), specs, paths, large

    @pytest.mark.parametrize("read_ahead", [2, 4, 8])
    def test_both_passes_keep_full_depth_with_the_embed_primed_first(
        self, tmp_path, read_ahead
    ):
        decoders = self.N_DECODER
        shard_dir, specs, paths, large = self._hetero_fixture(tmp_path, decoders)
        (embed_idx, embed_key), (head_idx, head_key) = large
        source = AsyncDiskSource(
            shard_dir,
            decoders + 2,
            specs,
            shard_paths=paths,
            read_ahead=read_ahead,
            pin=False,
        )
        try:
            forward, backward = [], []
            missed_forward, missed_backward = 0, 0
            for step in range(self.STEPS):
                # install_streaming._prime: the OUTPUT weight, before the walk.
                source.get(embed_idx, embed_key)
                source.release(embed_idx, None)
                for idx in range(decoders):
                    staged = idx in source._slot_of
                    source.get(idx, "w")
                    source.release(idx, None)
                    _settle(source)
                    if step >= 1 and idx + read_ahead - 1 < decoders:
                        forward.append(
                            sum(1 for lay in source._slot_of if idx < lay < decoders)
                        )
                        if not staged:
                            missed_forward += 1
                # StreamPrefetcher's tail prefetch at the forward turnaround.
                source.get(head_idx, head_key)
                source.release(head_idx, None)
                for idx in range(decoders - 1, -1, -1):
                    staged = idx in source._slot_of
                    source.get(idx, "w")
                    source.release(idx, None)
                    _settle(source)
                    if step >= 1 and idx >= read_ahead - 1:
                        backward.append(sum(1 for lay in source._slot_of if lay < idx))
                        if not staged:
                            missed_backward += 1
            assert min(forward) == read_ahead - 1, (
                f"forward pass sustained {min(forward)} of {read_ahead - 1} on the "
                f"production shape"
            )
            assert min(backward) == read_ahead - 1, (
                f"BACKWARD pass sustained {min(backward)} of {read_ahead - 1} on the "
                f"production shape. A cross-group fetch has frozen the decoder walk's "
                f"direction, so the recompute is prefetching the way it came."
            )
            assert (missed_forward, missed_backward) == (0, 0), (
                f"{missed_forward} forward and {missed_backward} backward layers were "
                f"demanded before they were staged"
            )
        finally:
            source.close()

    def test_a_vocabulary_fetch_never_speaks_for_the_decoder_walk(self, tmp_path):
        """The mechanism, named: the anchor the eviction policy reads."""
        decoders = 8
        shard_dir, specs, paths, large = self._hetero_fixture(tmp_path, decoders)
        (embed_idx, embed_key), _head = large
        source = AsyncDiskSource(
            shard_dir, decoders + 2, specs, shard_paths=paths, read_ahead=4, pin=False
        )
        try:
            source.get(embed_idx, embed_key)
            decoder_group = source._group_of[0]
            assert decoder_group not in source._last_get, (
                "the embed fetch wrote the DECODER group's anchor; the first decoder "
                "get can no longer establish it"
            )
            for idx in range(decoders):
                source.get(idx, "w")
            assert source._last_get.get(decoder_group) == decoders - 1, (
                "the decoder group's anchor was never established: a cross-group "
                "fetch is still speaking for it"
            )
            for idx in range(decoders - 1, decoders - 4, -1):
                source.get(idx, "w")
            assert source._direction.get(decoder_group) == -1, (
                "walking down, the decoder group's direction is still +1"
            )
            assert source._still_wanted(decoders - 4), (
                "the next layer the backward walk wants is not recognised as wanted, "
                "so the eviction preference has degenerated to plain FIFO"
            )
        finally:
            source.close()


class TestTheSettingReachesTheSource:
    """#748's lesson: a field read by nothing is the defect, not the feature.

    ``training.stream_read_ahead`` is validated, documented and bounded — and
    until the disk tier actually constructs ``AsyncDiskSource`` with it, setting
    it changes the config fingerprint and nothing else.
    """

    def test_build_source_returns_the_async_source_on_the_disk_tier(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import _build_source

        shard_dir = _shards(tmp_path)
        source, pinned = _build_source(
            shard_dir, N_LAYERS, _spec(shard_dir), False, None, "disk", read_ahead=3
        )
        try:
            assert isinstance(source, AsyncDiskSource)
            assert source.read_ahead == 3
            # pin=False was asked for, so the staging is pageable and the flag
            # — "the host-side source memory is page-locked" — says so.
            assert pinned is False
        finally:
            source.close()

    @pytest.mark.skipif(_NO_CUDA, reason="pinned staging needs a CUDA device")
    def test_the_disk_tier_pins_its_staging_when_asked(self, tmp_path):
        """The flag means page-locked on BOTH tiers, so the disk tier must be
        able to return True. Before pinned staging existed it was hardcoded
        False, which is now a lie rather than a simplification."""
        from soup_cli.utils.layer_stream_runtime import _build_source

        shard_dir = _shards(tmp_path)
        source, pinned = _build_source(
            shard_dir, N_LAYERS, _spec(shard_dir), True, None, "disk", read_ahead=2
        )
        try:
            assert source.pinned is True
            assert pinned is True
        finally:
            source.close()

    def test_the_ram_tier_is_untouched(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import _build_source

        shard_dir = _shards(tmp_path)
        source, _ = _build_source(
            shard_dir, N_LAYERS, _spec(shard_dir), False, None, "ram"
        )
        assert isinstance(source, RamSource)

    def test_every_stream_setup_call_site_passes_read_ahead(self):
        """Matched on the CALLEE NAME, not on "any call with a buffers= keyword":
        two of the three ``buffers=`` call sites in that file are
        ``build_stream_plan`` (the pure planner) and ``estimate_stream_peak_vram``
        (host staging is not VRAM), neither of which takes a read_ahead.
        """
        import ast

        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "soup_cli"
            / "trainer"
            / "stream_setup.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def _callee(node: ast.Call) -> str:
            func = node.func
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                return func.attr
            return ""

        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _callee(node) == "build_streamed_model"
        ]
        # Positive control: a scan that finds nothing must fail, not pass.
        assert calls, (
            "no build_streamed_model call site found in stream_setup.py — the "
            "scan cannot see the real one, so it proves nothing"
        )
        for call in calls:
            names = {kw.arg for kw in call.keywords}
            assert "read_ahead" in names, (
                f"the build_streamed_model call at line {call.lineno} passes "
                f"buffers but not read_ahead, so training.stream_read_ahead "
                f"never reaches the source"
            )


class TestEveryRuntimeConsumerReleases:
    """A consumer that forgets ``release`` gets pre-fix behaviour SILENTLY.

    The next ``get`` implicitly releases, the reader refills the slot, and the
    still-draining copy reads another layer's bytes — measured through the real
    pool at 6 of 8 layers on the default depth. ``_release_source`` is what the
    consumers call and it cannot detect NOT being called, so the contract is
    enforced from both ends: this scan is the static half, and
    ``AsyncDiskSource._hold``'s refusal under pinned staging is the dynamic one.
    Either alone leaves a gap — the scan cannot see a consumer written
    elsewhere, and the refusal cannot fire on a code path no test exercises.
    """

    @staticmethod
    def _functions_that_borrow():
        """``Class.method`` -> whether it also calls ``_release_source``."""
        import ast

        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "soup_cli"
            / "utils"
            / "layer_stream_runtime.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def _borrows(node: ast.AST) -> bool:
            return any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "get"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "source"
                for call in ast.walk(node)
            )

        def _releases(node: ast.AST) -> bool:
            return any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_release_source"
                for call in ast.walk(node)
            )

        found = {}
        for parent in ast.walk(tree):
            # Methods are named `Class.method`, so a scan that only walked
            # module-level defs — where neither consumer lives — cannot pass
            # by finding nothing.
            if isinstance(parent, ast.ClassDef):
                bodies = [(f"{parent.name}.{n.name}", n) for n in parent.body
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            elif isinstance(parent, ast.Module):
                bodies = [(n.name, n) for n in parent.body
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            else:
                continue
            for name, node in bodies:
                if _borrows(node):
                    found[name] = _releases(node)
        return found

    def test_the_scan_sees_both_shipped_consumers(self):
        """Positive control: a scan that finds nothing must FAIL, not pass."""
        found = self._functions_that_borrow()
        assert "LayerBufferPool.load_async" in found, found
        assert "LargeLayerBufferPool.load_async" in found, found

    def test_every_borrower_also_releases(self):
        found = self._functions_that_borrow()
        missing = sorted(name for name, releases in found.items() if not releases)
        assert not missing, (
            f"these functions call source.get() without calling "
            f"_release_source(): {missing}. Out of PINNED host staging the "
            f"copy is still draining when load_async returns, so the buffer "
            f"must not be recycled until an event says it has landed."
        )


@pytest.mark.skipif(_NO_CUDA, reason="pinned staging needs a CUDA device")
class TestPinnedStagingRefusesAnUnreleasedBorrow:
    """The dynamic half of the contract (see the scan above).

    Under PINNED staging the implicit release is unsound: the copy out of the
    borrowed buffer is still in flight, so quietly reclaiming it is the exact
    corruption ``release`` exists to prevent. Refusing is the only honest
    answer — a silently wrong gradient is worse than a loud stop.
    """

    def test_a_second_get_without_release_is_refused(self, tmp_path):
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=True)
        try:
            assert source.pinned, "the hazard needs genuinely pinned staging"
            source.get(0, "input_layernorm.weight")
            with pytest.raises(RuntimeError, match="release") as excinfo:
                source.get(1, "input_layernorm.weight")
            message = str(excinfo.value)
            assert "_release_source" in message, message
            # Both layers named: which one is on loan, and which was wanted.
            assert "0" in message and "1" in message, message
        finally:
            source.close()

    def test_the_compliant_sequence_is_unaffected(self, tmp_path):
        """The control that makes the refusal meaningful rather than a block on
        all traffic: released, the very same sequence works."""
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=True)
        try:
            first = source.get(0, "input_layernorm.weight")
            assert first is not None
            # None is the TRUE event here: a direct get enqueues no copy.
            source.release(0, None)
            second = source.get(1, "input_layernorm.weight")
            assert second.shape == first.shape
        finally:
            source.close()

    def test_repeated_gets_for_the_same_layer_are_not_a_borrow_violation(
        self, tmp_path
    ):
        """`load_async` calls `get` once per tensor NAME before releasing the
        layer once. If that read as a violation the refusal would fire on the
        real consumer's normal path."""
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, pin=True)
        try:
            for name in spec[0]:
                assert source.get(0, name) is not None
            source.release(0, None)
        finally:
            source.close()

    def test_pageable_staging_still_permits_the_implicit_release(self, tmp_path):
        """The carve-out, and why it is sound: out of PAGEABLE memory a
        `non_blocking` copy is host-synchronous, so no copy can be in flight and
        the documented borrow contract ("valid until your next get for a
        different layer") still holds. Every direct-get test in this file and
        the byte-identity gate rely on it."""
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=False)
        try:
            source.get(0, "input_layernorm.weight")
            assert source.get(1, "input_layernorm.weight") is not None
        finally:
            source.close()
