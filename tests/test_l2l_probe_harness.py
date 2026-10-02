"""CPU tests for the L2L step-0 probe harness (benchmarks/harness/l2l_*.py).

The harness is benchmark code, not shipped; these tests pin the parts a GPU run
cannot check by itself: the schedule's bit-exactness against gradient
accumulation, the activation store's byte round trip and spill ordering, the
suspend watch, and the rule's verdict logic.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent.parent / "benchmarks" / "harness"


def _load(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# l2l_activations
# --------------------------------------------------------------------------
@pytest.fixture()
def acts():
    return _load("l2l_activations")


def _chunk(seed: int, shape=(1, 8, 128)):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen)


def _store(acts, tmp_path, *, n_layers=4, k=3, spill_layers=0, backend=None, ring=8):
    import torch

    factory = None
    if spill_layers:
        made = backend or acts.BufferedSpillFile

        def factory(size):
            return made(str(tmp_path / "spill.bin"), size)

    return acts.ActivationStore(
        n_layers=n_layers,
        k=k,
        chunk_shape=(1, 8, 128),
        dtype=torch.float32,
        device="cpu",
        pin=False,
        spill_layers=spill_layers,
        spill_factory=factory,
        ring=ring,
    )


def _forward_puts(store, n_layers, k):
    for layer in range(n_layers):
        for mb in range(k):
            store.put(layer, mb, _chunk(layer * 100 + mb))


def test_aligned_buffer_is_sector_aligned_and_exact(acts):
    buf = acts.aligned_buffer(3 * 4096, pin=False)
    assert buf.data_ptr() % 4096 == 0
    assert buf.numel() == 3 * 4096


def test_resident_round_trip_returns_a_copy(acts, tmp_path):
    import torch

    store = _store(acts, tmp_path)
    _forward_puts(store, 4, 3)
    store.begin_backward()
    for layer in range(3, -1, -1):
        for mb in range(3):
            got = store.get(layer, mb)
            assert torch.equal(got, _chunk(layer * 100 + mb))
            got.zero_()  # a caller scribbling on its tensor must not reach the store
            assert torch.equal(store.get(layer, mb), _chunk(layer * 100 + mb))
    store.end_step()
    store.close()


def test_spill_round_trip_picks_the_lowest_layers(acts, tmp_path):
    import torch

    store = _store(acts, tmp_path, spill_layers=2)
    assert [store.is_spilled(layer) for layer in range(4)] == [True, True, False, False]
    assert store.stats.resident_chunks == 2 * 3
    assert store.stats.spilled_chunks == 2 * 3
    _forward_puts(store, 4, 3)
    store.begin_backward()
    for layer in range(3, -1, -1):
        for mb in range(3):
            assert torch.equal(store.get(layer, mb), _chunk(layer * 100 + mb))
    store.end_step()
    assert store.stats.bytes_written == 2 * 3 * store.chunk_bytes
    assert store.stats.bytes_read == 2 * 3 * store.chunk_bytes
    store.close()
    assert not (tmp_path / "spill.bin").exists()


def test_spill_survives_two_steps_with_new_values(acts, tmp_path):
    import torch

    store = _store(acts, tmp_path, spill_layers=2)
    for step in range(2):
        for layer in range(4):
            for mb in range(3):
                store.put(layer, mb, _chunk(step * 1000 + layer * 100 + mb))
        store.begin_backward()
        for layer in range(3, -1, -1):
            for mb in range(3):
                assert torch.equal(store.get(layer, mb), _chunk(step * 1000 + layer * 100 + mb))
        store.end_step()
    store.close()


def test_get_without_begin_backward_reads_on_demand(acts, tmp_path):
    import torch

    store = _store(acts, tmp_path, spill_layers=1)
    _forward_puts(store, 4, 3)
    assert torch.equal(store.get(0, 2), _chunk(2))
    store.close()


def test_a_ring_of_one_back_pressures_instead_of_overwriting(acts, tmp_path):
    import torch

    class SlowFile(acts.BufferedSpillFile):
        def write(self, offset, buf):
            time.sleep(0.02)  # the staging buffer must not be refilled meanwhile
            super().write(offset, buf)

    store = _store(acts, tmp_path, spill_layers=3, backend=SlowFile, ring=1)
    _forward_puts(store, 4, 3)
    store.begin_backward()
    for layer in range(3, -1, -1):
        for mb in range(3):
            assert torch.equal(store.get(layer, mb), _chunk(layer * 100 + mb))
    store.end_step()
    assert store.stats.put_wait_s > 0.0
    store.close()


def test_spill_write_failure_surfaces_on_the_next_call(acts, tmp_path):
    class BrokenFile(acts.BufferedSpillFile):
        def write(self, offset, buf):
            raise OSError("disk full")

    store = _store(acts, tmp_path, spill_layers=2, backend=BrokenFile)
    store.put(0, 0, _chunk(0))
    with pytest.raises(RuntimeError, match="spill.bin"):
        deadline = time.time() + 5
        while time.time() < deadline:
            store.put(0, 1, _chunk(1))
            time.sleep(0.01)
    store.close()


def test_end_step_refuses_unconsumed_spilled_chunks(acts, tmp_path):
    store = _store(acts, tmp_path, spill_layers=2)
    _forward_puts(store, 4, 3)
    store.begin_backward()
    store.get(3, 0)
    with pytest.raises(RuntimeError, match="not consumed"):
        store.end_step()
    store.close()


def test_spill_layers_without_a_factory_is_refused(acts):
    import torch

    with pytest.raises(ValueError, match="spill_factory"):
        acts.ActivationStore(
            n_layers=4, k=2, chunk_shape=(1, 8, 128), dtype=torch.float32,
            device="cpu", pin=False, spill_layers=1,
        )


def test_direct_spill_refuses_an_unaligned_size(acts, tmp_path):
    with pytest.raises(ValueError, match="multiple of 4096"):
        acts.DirectSpillFile(str(tmp_path / "x.bin"), 4096 + 512)


# --------------------------------------------------------------------------
# l2l_box
# --------------------------------------------------------------------------
@pytest.fixture()
def box():
    return _load("l2l_box")


class _FakeClock:
    def __init__(self, times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0)


def test_steady_samples_are_not_void(box):
    watch = box.SuspendWatch(clock=_FakeClock([0, 2, 4, 6]), power=lambda: True)
    for _ in range(4):
        watch.sample()
    report = watch.report()
    assert report["samples"] == 4
    assert report["max_gap_s"] == pytest.approx(2.0)
    assert report["void"] is False


def test_gap_over_limit_voids(box):
    watch = box.SuspendWatch(clock=_FakeClock([0, 2, 33]), power=lambda: True)
    for _ in range(3):
        watch.sample()
    assert watch.report()["max_gap_s"] == pytest.approx(31.0)
    assert watch.report()["void"] is True


def test_battery_sample_voids(box):
    power = iter([True, False, True])
    watch = box.SuspendWatch(clock=_FakeClock([0, 2, 4]), power=lambda: next(power))
    for _ in range(3):
        watch.sample()
    assert watch.report()["battery_samples"] == 1
    assert watch.report()["void"] is True


def test_unknown_power_is_recorded_not_void(box):
    watch = box.SuspendWatch(clock=_FakeClock([0, 2]), power=lambda: None)
    watch.sample()
    watch.sample()
    assert watch.report()["unknown_power_samples"] == 2
    assert watch.report()["void"] is False


def test_thread_start_stop_takes_samples(box):
    watch = box.SuspendWatch(interval=0.01, power=lambda: True)
    watch.start()
    time.sleep(0.1)
    report = watch.stop()
    assert report["samples"] >= 3
    assert not any(t.name == "l2l-suspend-watch" for t in threading.enumerate())


def test_box_stamp_has_the_keys(box):
    stamp = box.box_stamp()
    for key in ("unix", "avail_phys_gb", "commit_avail_gb", "ac", "gpu_pids",
                "gpu_mem_used_mib", "python_processes"):
        assert key in stamp

def test_foreign_readers_flags_heavy_other_processes(box):
    before = {10: {"name": "a.exe", "read": 0}, 11: {"name": "b.exe", "read": 5},
              99: {"name": "python.exe", "read": 0}}
    after = {10: {"name": "a.exe", "read": 2_000_000_000}, 11: {"name": "b.exe", "read": 6},
             99: {"name": "python.exe", "read": 3_000_000_000},
             12: {"name": "SearchIndexer.exe", "read": 1_500_000_000}}
    heavy = box.foreign_readers(before, after, own_pids={99}, threshold=1_000_000_000)
    assert [(row["pid"], row["name"]) for row in heavy] == [
        (10, "a.exe"), (12, "SearchIndexer.exe"),
    ]
    assert heavy[0]["read_bytes"] == 2_000_000_000


def test_foreign_readers_is_none_without_snapshots(box):
    assert box.foreign_readers(None, {}, own_pids=set()) is None


def test_process_read_bytes_sees_this_process(box):
    import os

    snapshot = box.process_read_bytes()
    if snapshot is None:
        pytest.skip("psutil is not installed")
    assert os.getpid() in snapshot
    assert snapshot[os.getpid()]["read"] >= 0
