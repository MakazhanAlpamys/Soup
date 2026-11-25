"""CPU reproductions of activation-transfer, startup, and raw-I/O failures."""

from __future__ import annotations

import ctypes
import io
import threading
import weakref

import pytest

from soup_cli.utils import l2l_activations as acts


def _store(tmp_path, *, spill_layers=1, k=1, ring=1):
    import torch

    return acts.ActivationStore(
        n_layers=2,
        k=k,
        chunk_shape=(1, 8, 128),
        dtype=torch.float32,
        device="cpu",
        pin=False,
        spill_layers=spill_layers,
        spill_factory=lambda size: acts.BufferedSpillFile(str(tmp_path / "spill.bin"), size),
        ring=ring,
    )


def _defer_host_copies(monkeypatch):
    """Model an accelerator that returns before an explicitly nonblocking host copy.

    The deferred operation reads its source only when completed. The real store,
    CPU tensors, staging rings, worker threads and spill bytes remain in use.
    """
    import torch

    pending = []
    copy = torch.Tensor.copy_

    def copy_later(dst, src, non_blocking=False):
        if not non_blocking:
            return copy(dst, src)
        dst.zero_()
        pending.append(lambda: copy(dst, src))
        return dst

    monkeypatch.setattr(torch.Tensor, "copy_", copy_later)
    return pending


def _defer_device_copies(monkeypatch):
    import torch

    pending = []
    to = torch.Tensor.to
    copy = torch.Tensor.copy_

    def to_later(src, *args, **kwargs):
        out = to(src, *args, **kwargs)
        if kwargs.get("non_blocking") and kwargs.get("copy"):
            out.zero_()
            pending.append(lambda: copy(out, src))
        return out

    monkeypatch.setattr(torch.Tensor, "to", to_later)
    return pending


def _complete(pending):
    for transfer in pending:
        transfer()
    pending.clear()


@pytest.mark.parametrize("spill_layers", [0, 1], ids=["resident", "spilled"])
def test_noncuda_put_completes_host_transfer_before_consumer_reads(
    tmp_path, monkeypatch, spill_layers
):
    import torch

    store = _store(tmp_path, spill_layers=spill_layers)
    pending = _defer_host_copies(monkeypatch)
    expected = torch.arange(1024, dtype=torch.float32).view(1, 8, 128)
    try:
        store.put(0, 0, expected)
        got = store.get(0, 0)
        _complete(pending)
        assert torch.equal(got, expected)
        store.end_step()
    finally:
        _complete(pending)
        store.close()


def test_noncuda_resident_get_completes_before_host_slot_is_reused(tmp_path, monkeypatch):
    import torch

    store = _store(tmp_path, spill_layers=0)
    expected = torch.full((1, 8, 128), 17.0)
    store.put(0, 0, expected)
    pending = _defer_device_copies(monkeypatch)
    try:
        got = store.get(0, 0)
        store.put(0, 0, torch.full_like(expected, 29.0))
        _complete(pending)
        assert torch.equal(got, expected)
    finally:
        _complete(pending)
        store.close()


def test_noncuda_spilled_get_completes_before_read_ring_is_reused(tmp_path, monkeypatch):
    import torch

    store = _store(tmp_path, k=2, ring=1)
    first = torch.full((1, 8, 128), 17.0)
    second = torch.full_like(first, 29.0)
    store.put(0, 0, first)
    store.put(0, 1, second)
    pending = _defer_device_copies(monkeypatch)
    try:
        store.begin_backward()
        got_first = store.get(0, 0)
        got_second = store.get(0, 1)
        _complete(pending)
        assert torch.equal(got_first, first)
        assert torch.equal(got_second, second)
        store.end_step()
    finally:
        _complete(pending)
        store.close()


def test_variable_capacity_resident_allocation_failure_releases_partial_store(
    tmp_path, monkeypatch
):
    import torch

    original_empty = torch.empty
    stores = []
    buffers = []
    attempts = 0

    def empty(*args, **kwargs):
        nonlocal attempts
        if "pin_memory" in kwargs:
            attempts += 1
            if attempts == 3:
                raise MemoryError("injected resident allocation failure")
            buf = original_empty(*args, **kwargs)
            buffers.append((weakref.ref(buf), buf.numel()))
            return buf
        return original_empty(*args, **kwargs)

    class CapturedStore(acts.ActivationStore):
        def __new__(cls, **kwargs):
            store = super().__new__(cls)
            stores.append(store)
            return store

    path = tmp_path / "partial-resident.bin"
    monkeypatch.setattr(torch, "empty", empty)
    with pytest.raises(MemoryError, match="injected resident allocation failure"):
        CapturedStore(
            n_layers=3, k=2, chunk_shape=(1, 8, 512),
            mb_shapes=[(1, 3, 512), (1, 8, 512)], dtype=torch.float32, device="cpu", pin=False,
            spill_layers=1, spill_factory=lambda size: acts.BufferedSpillFile(str(path), size),
        )
    assert [size for _, size in buffers] == [1536, 4096]
    assert all(ref() is None for ref, _ in buffers)
    assert stores and all(not store._resident for store in stores)
    assert not path.exists()


@pytest.mark.parametrize("fail_at", [1, 2], ids=["writer", "reader"])
@pytest.mark.parametrize("after_start", [False, True], ids=["before-start", "after-start"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_constructor_thread_start_failure_releases_file_workers_and_buffers(
    tmp_path, monkeypatch, fail_at, after_start, failure
):
    original_thread = threading.Thread
    original_buffer = acts.aligned_buffer
    made_threads = []
    stores = []
    buffers = []
    handles = []
    attempts = 0

    class FailingThread(original_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made_threads.append(self)
            stores.append(kwargs["target"].__self__)

        def start(self):
            nonlocal attempts
            attempts += 1
            if attempts == fail_at:
                if after_start:
                    super().start()
                raise failure("injected worker startup failure")
            super().start()

    def aligned_buffer(*args, **kwargs):
        buf = original_buffer(*args, **kwargs)
        buffers.extend((weakref.ref(buf), weakref.ref(buf._base)))
        return buf

    class TrackingFile(acts.BufferedSpillFile):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            handles.extend((self._writer, self._reader))

    monkeypatch.setattr(acts.threading, "Thread", FailingThread)
    monkeypatch.setattr(acts, "aligned_buffer", aligned_buffer)
    monkeypatch.setattr(acts, "BufferedSpillFile", TrackingFile)
    try:
        with pytest.raises(failure, match="injected worker startup failure"):
            _store(tmp_path)
        assert not (tmp_path / "spill.bin").exists()
        assert all(handle.closed for handle in handles)
        assert all(not thread.is_alive() for thread in made_threads)
        assert stores and all(not store._resident for store in stores)
        assert buffers and all(ref() is None for ref in buffers)
    finally:
        # A pre-fix constructor cannot be reached by its caller's finally block.
        # Keep the regression itself from leaking its deliberately orphaned workers.
        for store in stores:
            store._closing = True
            for name in ("_writes", "_reads"):
                jobs = getattr(store, name, None)
                if jobs is not None:
                    jobs.put(None)
            cond = getattr(store, "_cond", None)
            if cond is not None:
                with cond:
                    cond.notify_all()
        for thread in made_threads:
            if thread.ident is not None:
                thread.join(timeout=5)
        for handle in handles:
            handle.close()
        (tmp_path / "spill.bin").unlink(missing_ok=True)


@pytest.mark.parametrize("fail_at", [1, 2], ids=["write-queue", "read-queue"])
def test_constructor_queue_failure_releases_file_and_partial_allocations(
    tmp_path, monkeypatch, fail_at
):
    original_queue = acts.queue.Queue
    original_buffer = acts.aligned_buffer
    original_ring = acts._Ring
    stores = []
    buffers = []
    attempts = 0
    allocating_ring = False

    def aligned_buffer(*args, **kwargs):
        buf = original_buffer(*args, **kwargs)
        buffers.extend((weakref.ref(buf), weakref.ref(buf._base)))
        return buf

    def ring(*args, **kwargs):
        nonlocal allocating_ring
        allocating_ring = True
        try:
            return original_ring(*args, **kwargs)
        finally:
            allocating_ring = False

    def jobs(*args, **kwargs):
        nonlocal attempts
        if not allocating_ring:
            attempts += 1
            if attempts == fail_at:
                raise KeyboardInterrupt("injected queue construction failure")
        return original_queue(*args, **kwargs)

    class CapturedStore(acts.ActivationStore):
        def __new__(cls, **kwargs):
            store = super().__new__(cls)
            stores.append(store)
            return store

    monkeypatch.setattr(acts, "ActivationStore", CapturedStore)
    monkeypatch.setattr(acts, "aligned_buffer", aligned_buffer)
    monkeypatch.setattr(acts, "_Ring", ring)
    monkeypatch.setattr(acts.queue, "Queue", jobs)
    try:
        with pytest.raises(KeyboardInterrupt, match="injected queue construction failure"):
            _store(tmp_path)
        assert not (tmp_path / "spill.bin").exists()
        assert stores and all(not store._resident for store in stores)
        assert buffers and all(ref() is None for ref in buffers)
    finally:
        for store in stores:
            backend = getattr(store, "_spill", None)
            if backend is not None:
                backend.close()


@pytest.mark.parametrize("fail_at", [1, 2, 3], ids=["first-write", "second-write", "read"])
def test_constructor_ring_failure_releases_file_and_partial_buffers(
    tmp_path, monkeypatch, fail_at
):
    original_buffer = acts.aligned_buffer
    stores = []
    buffers = []
    attempts = 0

    def aligned_buffer(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == fail_at:
            raise MemoryError("injected staging allocation failure")
        buf = original_buffer(*args, **kwargs)
        buffers.extend((weakref.ref(buf), weakref.ref(buf._base)))
        return buf

    class CapturedStore(acts.ActivationStore):
        def __new__(cls, **kwargs):
            store = super().__new__(cls)
            stores.append(store)
            return store

    monkeypatch.setattr(acts, "ActivationStore", CapturedStore)
    monkeypatch.setattr(acts, "aligned_buffer", aligned_buffer)
    with pytest.raises(MemoryError, match="injected staging allocation failure"):
        _store(tmp_path, ring=2)
    assert not (tmp_path / "spill.bin").exists()
    assert stores and all(not store._resident for store in stores)
    assert all(ref() is None for ref in buffers)


@pytest.mark.parametrize("spill_layers", [0, 1], ids=["resident", "spilled"])
def test_close_releases_allocations_while_store_reference_remains(
    tmp_path, monkeypatch, spill_layers
):
    import torch

    original_buffer = acts.aligned_buffer
    buffers = []

    def aligned_buffer(*args, **kwargs):
        buf = original_buffer(*args, **kwargs)
        buffers.extend((weakref.ref(buf), weakref.ref(buf._base)))
        return buf

    monkeypatch.setattr(acts, "aligned_buffer", aligned_buffer)
    store = _store(tmp_path, spill_layers=spill_layers, k=2, ring=2)
    buffers.extend(weakref.ref(buf) for buf in store._resident.values())
    for mb in range(2):
        store.put(0, mb, torch.full((1, 8, 128), float(mb + 1)))
    store.begin_backward()
    store.get(0, 0)
    stats = store.stats
    store.close()
    store.close()
    assert buffers and all(ref() is None for ref in buffers)
    assert store.stats is stats
    assert store.stats.resident_chunks == (4 if spill_layers == 0 else 2)
    assert not (tmp_path / "spill.bin").exists()


def test_close_completes_read_ring_event_before_releasing_source(tmp_path):
    import torch

    store = _store(tmp_path)
    buf = store._read_ring.acquire(lambda: False)
    buf.fill_(37)
    source = weakref.ref(buf)
    allocation = weakref.ref(buf._base)
    copied = torch.zeros_like(buf)
    expected = torch.full_like(buf, 37)

    class DeferredRead:
        def synchronize(self):
            copied.copy_(source())

    store._read_ring.release(buf, DeferredRead())
    del buf
    store.close()
    assert torch.equal(copied, expected)
    assert source() is None
    assert allocation() is None


def test_ring_acquire_failure_retains_source_until_deferred_copy_completes():
    import torch

    buf = acts.aligned_buffer(4096, pin=False)
    buf.fill_(41)
    source = weakref.ref(buf)
    allocation = weakref.ref(buf._base)
    copied = torch.zeros_like(buf)
    expected = torch.full_like(buf, 41)
    ring = acts._Ring([])
    failed = False

    class DeferredRead:
        def synchronize(self):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("injected acquire event failure")
            copied.copy_(source())

    ring.release(buf, DeferredRead())
    del buf
    with pytest.raises(RuntimeError, match="injected acquire event failure"):
        ring.acquire(lambda: False)
    assert source() is not None
    assert allocation() is not None
    ring.close()
    assert torch.equal(copied, expected)
    assert source() is None
    assert allocation() is None


def test_close_can_retry_after_interrupted_worker_join(tmp_path, monkeypatch):
    original_thread = threading.Thread
    threads = []
    interrupted = False

    class InterruptedJoin(original_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            threads.append(self)

        def join(self, *args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt("injected join interruption")
            return super().join(*args, **kwargs)

    monkeypatch.setattr(acts.threading, "Thread", InterruptedJoin)
    store = _store(tmp_path)
    resident = [weakref.ref(buf) for buf in store._resident.values()]
    try:
        with pytest.raises(KeyboardInterrupt, match="injected join interruption"):
            store.close()
        store.close()
        assert not (tmp_path / "spill.bin").exists()
        assert all(not thread.is_alive() for thread in threads)
        assert all(ref() is None for ref in resident)
    finally:
        for thread in threads:
            original_thread.join(thread, timeout=5)
        if store._spill is not None:
            store._spill.close()


def test_close_waits_for_worker_completion_when_interrupted_join_returns_early(
    tmp_path, monkeypatch
):
    import torch

    original_thread = threading.Thread
    entered_write = threading.Event()
    release_write = threading.Event()
    observation = threading.Condition()
    outcomes = []
    threads = []
    backends = []
    written = []
    failures = []
    interrupted = False

    def observed(outcome):
        with observation:
            outcomes.append(outcome)
            observation.notify_all()

    class InterruptedJoin(original_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            threads.append(self)

        def join(self, *args, **kwargs):
            nonlocal interrupted
            if self.name == "l2l-spill-writer":
                if not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt("injected native join interruption")
                # CPython can report the thread stopped after an interrupted join,
                # even though its target still owns the backend and staging buffer.
                return None
            return super().join(*args, **kwargs)

    class BlockedFile(acts.BufferedSpillFile):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            backends.append(self)

        def write(self, offset, buf):
            entered_write.set()
            if not release_write.wait(5):
                raise RuntimeError("test did not release the blocked writer")
            super().write(offset, buf)
            got = torch.empty_like(buf)
            super().read(offset, got)
            written.append(got)

        def close(self, remove=True):
            observed("backend-closed")
            super().close(remove=remove)

    class ObservedCompletion:
        def __init__(self, done):
            self.done = done

        def set(self):
            self.done.set()

        def wait(self, timeout=None):
            observed("waiting-for-worker")
            return self.done.wait(timeout)

    monkeypatch.setattr(acts.threading, "Thread", InterruptedJoin)
    monkeypatch.setattr(acts, "BufferedSpillFile", BlockedFile)
    store = _store(tmp_path)
    expected = torch.arange(1024, dtype=torch.float32).view(1, 8, 128)
    resident = [weakref.ref(buf) for buf in store._resident.values()]
    retry = None
    try:
        store.put(0, 0, expected)
        assert entered_write.wait(5)
        with pytest.raises(KeyboardInterrupt, match="injected native join interruption"):
            store.close()
        completions = getattr(store, "_worker_done", [])
        if completions:
            completions[0] = ObservedCompletion(completions[0])

        def retry_close():
            try:
                store.close()
            except BaseException as exc:
                failures.append(exc)
                observed("close-failed")

        retry = original_thread(target=retry_close)
        retry.start()
        with observation:
            assert observation.wait_for(lambda: bool(outcomes), timeout=5)
            assert outcomes[0] == "waiting-for-worker"
        assert (tmp_path / "spill.bin").exists()
        assert backends[0]._writer is not None and not backends[0]._writer.closed
        assert all(ref() is not None for ref in resident)
        release_write.set()
        retry.join(timeout=5)
        assert not retry.is_alive()
        assert not failures
        assert len(written) == 1
        assert torch.equal(written[0], expected.view(torch.uint8).view(-1))
        assert not (tmp_path / "spill.bin").exists()
        assert all(ref() is None for ref in resident)
    finally:
        release_write.set()
        if retry is not None:
            retry.join(timeout=5)
        for thread in threads:
            original_thread.join(thread, timeout=5)
        store.close()


@pytest.mark.parametrize("pending_in", ["read-ring", "write-queue"])
def test_close_retains_pending_sources_when_event_synchronization_fails(
    tmp_path, pending_in
):
    import torch

    store = _store(tmp_path, ring=2)
    failed = False
    sources = []
    allocations = []
    copied = [torch.zeros(4096, dtype=torch.uint8) for _ in range(2)]

    class DeferredCopy:
        def __init__(self, source, destination, fail_once):
            self.source = source
            self.destination = destination
            self.fail_once = fail_once

        def synchronize(self):
            nonlocal failed
            if self.fail_once and not failed:
                failed = True
                raise RuntimeError("injected event synchronization failure")
            self.destination.copy_(self.source())

    if pending_in == "write-queue":
        # Stop the real writer first; these tasks model an already-failed worker's
        # queued host transfers that teardown must finish without consuming bytes.
        store._writes.put(None)
        store._threads[0].join()
    for index in range(2):
        ring = store._read_ring if pending_in == "read-ring" else store._write_ring
        buf = ring.acquire(lambda: False)
        buf.fill_(17 + index)
        source = weakref.ref(buf)
        allocations.append(weakref.ref(buf._base))
        sources.append(source)
        event = DeferredCopy(source, copied[index], index == 0)
        if pending_in == "read-ring":
            ring.release(buf, event)
        else:
            store._writes.put(((0, index), buf, event, 4096))
        del buf
    try:
        with pytest.raises(RuntimeError, match="injected event synchronization failure"):
            store.close()
        assert not (tmp_path / "spill.bin").exists()
        assert all(source() is not None for source in sources)
        assert all(allocation() is not None for allocation in allocations)
        store.close()
        assert torch.equal(copied[0], torch.full_like(copied[0], 17))
        assert torch.equal(copied[1], torch.full_like(copied[1], 18))
        assert all(source() is None for source in sources)
        assert all(allocation() is None for allocation in allocations)
    finally:
        store.close()


def test_close_completes_transfer_after_writer_event_failure(tmp_path):
    import torch

    store = _store(tmp_path)
    buf = store._write_ring.acquire(lambda: False)
    buf.fill_(23)
    source = weakref.ref(buf)
    allocation = weakref.ref(buf._base)
    copied = torch.zeros_like(buf)
    expected = torch.full_like(buf, 23)
    failed = False

    class InterruptedTransfer:
        def synchronize(self):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("injected writer event failure")
            copied.copy_(source())

    store._writes.put(((0, 0), buf, InterruptedTransfer(), 4096))
    del buf
    try:
        with store._cond:
            assert store._cond.wait_for(lambda: store._error is not None, timeout=5)
        store.close()
        assert torch.equal(copied, expected)
        assert source() is None
        assert allocation() is None
        assert not (tmp_path / "spill.bin").exists()
    finally:
        store.close()


def test_close_releases_buffers_retained_by_chained_worker_error(tmp_path, monkeypatch):
    import torch

    original_buffer = acts.aligned_buffer
    buffers = []

    def aligned_buffer(*args, **kwargs):
        buf = original_buffer(*args, **kwargs)
        buffers.append(weakref.ref(buf._base))
        return buf

    class BrokenFile(acts.BufferedSpillFile):
        def write(self, offset, buf):
            try:
                raise OSError("underlying write failure")
            except OSError as exc:
                raise RuntimeError("wrapped write failure") from exc

    monkeypatch.setattr(acts, "aligned_buffer", aligned_buffer)
    monkeypatch.setattr(acts, "BufferedSpillFile", BrokenFile)
    store = _store(tmp_path)
    try:
        store.put(0, 0, torch.ones((1, 8, 128)))
        with store._cond:
            assert store._cond.wait_for(lambda: store._error is not None, timeout=5)
        store.close()
        assert isinstance(store._error, RuntimeError)
        assert isinstance(store._error.__cause__, OSError)
        assert buffers and all(ref() is None for ref in buffers)
        assert not (tmp_path / "spill.bin").exists()
    finally:
        store.close()


class _CappedFile:
    """FileIO's Windows request cap, with a small controllable sector boundary."""

    def __init__(self, path, cap, *, write_address, read_address, partial=False):
        self.file = io.FileIO(path, "w+b")
        self.cap = cap
        self.partial = partial
        self.write_address = write_address
        self.read_address = read_address

    def seek(self, offset):
        self.offset = offset
        return self.file.seek(offset)

    def _request(self, view, address):
        if not view.contiguous or view.itemsize != 1:
            raise AssertionError("raw I/O must retain its contiguous byte view")
        if len(view) % 4096:
            raise OSError("unbuffered request is not sector aligned")
        expected_address = address + self.file.tell() - self.offset
        if ctypes.addressof(ctypes.c_ubyte.from_buffer(view)) != expected_address:
            raise AssertionError("raw I/O must not copy its source or destination")
        nbytes = min(len(view), self.cap)
        if nbytes % 4096:
            raise OSError("unbuffered request is not sector aligned")
        if self.partial and nbytes > 4096:
            nbytes -= 4096
        return view[:nbytes]

    def write(self, view):
        return self.file.write(self._request(view, self.write_address))

    def readinto(self, view):
        return self.file.readinto(self._request(view, self.read_address))


@pytest.mark.parametrize("partial", [False, True], ids=["whole", "partial"])
def test_raw_io_splits_at_sector_aligned_platform_limit_without_copying(
    tmp_path, monkeypatch, partial
):
    import torch

    monkeypatch.setattr(acts, "_RAW_IO_MAX_BYTES", 3 * 4096 - 1, raising=False)
    source = acts.aligned_buffer(5 * 4096, pin=False)
    source.copy_(torch.arange(source.numel(), dtype=torch.int64).remainder(251).to(torch.uint8))
    destination = acts.aligned_buffer(source.numel(), pin=False)
    handle = _CappedFile(
        tmp_path / "raw.bin",
        3 * 4096 - 1,
        write_address=source.data_ptr(),
        read_address=destination.data_ptr(),
        partial=partial,
    )
    try:
        acts._write_all(handle, 4096, source)
        acts._read_all(handle, 4096, destination)
        assert torch.equal(destination, source)
    finally:
        handle.file.close()


@pytest.mark.parametrize("direction", ["read", "write"])
def test_raw_io_zero_progress_still_raises_short_transfer(tmp_path, direction):
    buf = acts.aligned_buffer(4096, pin=False)
    with io.FileIO(tmp_path / "empty.bin", "w+b") as handle:
        if direction == "read":
            with pytest.raises(OSError, match="short read"):
                acts._read_all(handle, 0, buf)
        else:
            class StalledWrite:
                def seek(self, offset):
                    handle.seek(offset)

                def write(self, view):
                    return 0

            with pytest.raises(OSError, match="short write"):
                acts._write_all(StalledWrite(), 0, buf)
