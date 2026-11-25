"""The L2L activation spill without a trainer: the planner, the store's slots, the spill
file, the stale sweep, the volume checks, the disk-tier rule and the host-RAM budget.

Seams: ``soup_cli.utils.l2l_spill`` (``plan_spill``, ``open_spill_file``, ``sweep_stale``,
``spill_files``, ``prepare_spill_dir``), ``soup_cli.utils.l2l_activations.ActivationStore`` with
the real ``DirectSpillFile``, and ``ram_budget`` on fake v1/v2 cgroup trees.
"""

from __future__ import annotations

import errno
import logging
import os
import subprocess
import sys
import types

import pytest

MIB, GIB = 2**20, 2**30
#: psutil's answer when a budget test wants the cgroup to decide.
PLENTY = 10**15
#: Write a directory in place of a file: it exists and cannot be read.
DIRECTORY = object()


@pytest.fixture()
def direct_dir(tmp_path):
    """A directory on the temp volume, skipped where unbuffered I/O is refused (tmpfs)."""
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "direct"
    directory.mkdir()
    problem = l2l_spill.filesystem_refusal(os.path.realpath(directory))
    problem = problem or l2l_spill._round_trip(str(directory))
    if problem:
        pytest.skip(f"the temp directory cannot hold an unbuffered spill: {problem}")
    return str(directory)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------
def test_a_pinned_17_mib_chunk_costs_32_mib_and_the_rings_are_charged():
    from soup_cli.utils.l2l_spill import plan_spill

    plan = plan_spill(10 * GIB, n_layers=80, k=16, chunk_bytes=17 * MIB, pin=True)
    assert plan.ring == 4
    assert plan.chunk_cost == 32 * MIB
    assert plan.slot_bytes == 17 * MIB
    # one staging buffer asks a slot plus one sector, and that request is page-locked as
    # the next power of two
    assert plan.ring_buffer_cost == 32 * MIB
    assert plan.minimum_bytes == 2 * 4 * 32 * MIB
    assert plan.all_resident_bytes == 80 * 16 * 32 * MIB
    assert plan.spill_layers == 61
    assert plan.host_bytes == 19 * 16 * 32 * MIB + 2 * 4 * 32 * MIB == 10_468_982_784
    assert plan.host_bytes <= 10 * GIB
    assert plan.spill_file_bytes == 61 * 16 * 17 * MIB


def test_pageable_memory_is_charged_at_its_request():
    from soup_cli.utils.l2l_spill import plan_spill

    plan = plan_spill(10 * GIB, n_layers=80, k=16, chunk_bytes=17 * MIB, pin=False)
    assert plan.chunk_cost == 17 * MIB
    assert plan.ring_buffer_cost == 17 * MIB + 4096
    assert plan.minimum_bytes == 8 * (17 * MIB + 4096)
    assert plan.spill_layers == 43
    assert plan.host_bytes == 37 * 16 * 17 * MIB + 8 * (17 * MIB + 4096)


def test_pinned_rounding_decides_whether_the_store_fits():
    """bf16, 80 layers, hidden 8192, seq 1088, k=16: a 17 MiB chunk occupies a 32 MiB pinned
    block, so the store needs 40 GiB, not the 21.25 GiB of its logical bytes."""
    from soup_cli.utils.l2l_spill import plan_spill

    budget = int(32 * GIB * 0.7)
    kw = dict(n_layers=80, k=16, chunk_bytes=17 * MIB)
    assert plan_spill(budget, pin=False, **kw).spill_layers == 0
    pinned = plan_spill(budget, pin=True, **kw)
    assert pinned.spill_layers == 36
    assert pinned.host_bytes <= budget


def test_what_fits_stays_resident_and_no_budget_means_no_spill():
    from soup_cli.utils.l2l_spill import plan_spill

    kw = dict(n_layers=2, k=3, chunk_bytes=100, pin=False)
    fits = plan_spill(1000, **kw)
    assert fits.spill_layers == 0 and fits.host_bytes == 600
    assert plan_spill(None, **{**kw, "chunk_bytes": 10**15}).spill_layers == 0


def test_a_budget_below_every_layer_spilled_is_refused_with_what_to_change():
    from soup_cli.utils.l2l_spill import SpillRefusedError, plan_spill

    with pytest.raises(SpillRefusedError) as caught:
        plan_spill(1000, n_layers=2, k=4, chunk_bytes=200, pin=False)
    message = str(caught.value)
    assert "activations" in message
    for fix in ("gradient_accumulation_steps", "data.max_length", "batch_size", "free RAM"):
        assert fix in message
    # 10,000-byte chunks: slots of 12,288, ring buffers of 16,384, rings of 131,072 bytes,
    # the smallest budget that runs once 10 x 4 chunks no longer fit
    kw = dict(n_layers=10, k=4, chunk_bytes=10_000, pin=False)
    assert plan_spill(131_072, **kw).spill_layers == 10
    with pytest.raises(SpillRefusedError):
        plan_spill(131_071, **kw)


@pytest.mark.parametrize("pin", [False, True])
def test_every_budget_gets_the_fewest_spilled_layers_within_it(pin):
    from soup_cli.utils.l2l_spill import plan_spill

    kw = dict(n_layers=12, k=3, chunk_bytes=5 * MIB + 123, pin=pin)
    top = plan_spill(None, **kw)
    for budget in range(top.minimum_bytes, top.all_resident_bytes, 3 * MIB + 7):
        plan = plan_spill(budget, **kw)
        assert 1 <= plan.spill_layers <= 12
        assert plan.host_bytes <= budget, budget
        # one layer fewer spilled would not fit (or every other layer is already resident)
        assert plan.resident_layers == 11 or plan.host_bytes + 3 * plan.chunk_cost > budget


# --------------------------------------------------------------------------
# the store: slots, alignment, a smaller tensor in a larger slot
# --------------------------------------------------------------------------
def _direct_store(directory, *, n_layers, spill_layers, shape, dtype, k=2):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore, DirectSpillFile

    path = os.path.join(directory, "spill.bin")
    return ActivationStore(
        n_layers=n_layers, k=k, chunk_shape=shape, dtype=dtype, device="cpu", pin=False,
        spill_layers=spill_layers, spill_factory=lambda size: DirectSpillFile(path, size),
        ring=4,
    ), path, torch


def test_a_bf16_chunk_off_the_sector_round_trips_through_the_direct_file(direct_dir):
    import torch

    shape = (1, 7, 5120)  # bf16, odd seq: 71,680 bytes, 17.5 sectors
    store, path, _ = _direct_store(
        direct_dir, n_layers=3, spill_layers=2, shape=shape, dtype=torch.bfloat16
    )
    try:
        assert store.chunk_bytes == 71_680 and store.slot_bytes == 73_728
        assert os.path.getsize(path) == 2 * 2 * 73_728
        gen = torch.Generator().manual_seed(0)
        sent = {
            (layer, mb): torch.randn(shape, generator=gen).to(torch.bfloat16)
            for layer in range(3) for mb in range(2)
        }
        for (layer, mb), tensor in sent.items():
            store.put(layer, mb, tensor)
        store.begin_backward()
        for layer in range(2, -1, -1):
            for mb in range(2):
                got = store.get(layer, mb)
                assert got.dtype == torch.bfloat16 and got.shape == shape
                assert torch.equal(got, sent[(layer, mb)])
        store.end_step()
        assert store.stats.bytes_written == store.stats.bytes_read == 2 * 2 * 73_728
    finally:
        store.close()
    assert not os.path.exists(path)


def test_a_smaller_tensor_lives_in_the_front_of_a_larger_slot(direct_dir):
    import torch

    store, path, _ = _direct_store(
        direct_dir, n_layers=3, spill_layers=2, shape=(1, 7, 5120), dtype=torch.bfloat16
    )
    try:
        for step, short in enumerate([(1, 3, 5120), (1, 7, 5120), (1, 1, 5120)]):
            sent = {
                (layer, mb): torch.full(short, float(step * 10 + layer * 2 + mb)).bfloat16()
                for layer in range(3) for mb in range(2)
            }
            for (layer, mb), tensor in sent.items():
                store.put(layer, mb, tensor)
            store.begin_backward()
            for layer in range(2, -1, -1):
                for mb in range(2):
                    got = store.get(layer, mb)
                    assert got.shape == short
                    assert torch.equal(got, sent[(layer, mb)])
            store.end_step()
        with pytest.raises(ValueError, match="does not fit"):
            store.put(2, 0, torch.zeros((1, 8, 5120), dtype=torch.bfloat16))
    finally:
        store.close()


def test_per_microbatch_slots_keep_each_layers_shape_and_spill_only_its_bytes(tmp_path):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore, BufferedSpillFile

    path = tmp_path / "mixed-shapes.bin"
    store = ActivationStore(
        n_layers=4, k=2, chunk_shape=(1, 8, 512),
        mb_shapes=[(1, 3, 512), (1, 8, 512)], dtype=torch.float32, device="cpu", pin=False,
        spill_layers=2, spill_factory=lambda size: BufferedSpillFile(str(path), size), ring=1,
    )
    try:
        assert path.stat().st_size == 65_536
        windows = [
            ((1, 7), (3, 4), (2, 8), (3, 1)),
            ((3, 2), (2, 8), (1, 6), (2, 5)),
        ]
        for step, lengths in enumerate(windows):
            sent = {
                (layer, mb): torch.full((1, seq, 512), float(step * 10 + layer * 2 + mb))
                for layer, rows in enumerate(lengths)
                for mb, seq in enumerate(rows)
            }
            for (layer, mb), tensor in sent.items():
                store.put(layer, mb, tensor)
            store.begin_backward()
            for layer in range(3, -1, -1):
                for mb in range(2):
                    got = store.get(layer, mb)
                    expected = sent[(layer, mb)]
                    assert got.shape == expected.shape
                    assert torch.equal(got, expected)
                    got.zero_()
                    if layer >= 2:
                        assert torch.equal(store.get(layer, mb), expected)
            store.end_step()
            assert store.stats.bytes_written == store.stats.bytes_read == (36_864, 69_632)[step]
    finally:
        store.close()
    assert not path.exists()


@pytest.mark.parametrize("layer", [0, 1], ids=["spilled", "resident"])
def test_put_refuses_a_microbatch_over_its_own_capacity_without_replacing_it(tmp_path, layer):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore, BufferedSpillFile

    store = ActivationStore(
        n_layers=2, k=2, chunk_shape=(1, 8, 512),
        mb_shapes=[(1, 3, 512), (1, 8, 512)], dtype=torch.float32, device="cpu", pin=False,
        spill_layers=1,
        spill_factory=lambda size: BufferedSpillFile(str(tmp_path / "bounded.bin"), size),
    )
    expected = torch.full((1, 3, 512), 17.0)
    largest = torch.full((1, 8, 512), 29.0)
    try:
        store.put(layer, 0, expected)
        store.put(layer, 1, largest)
        with pytest.raises(ValueError, match="micro-batch 0"):
            store.put(layer, 0, torch.zeros((1, 4, 512)))
        store.begin_backward()
        for mb, tensor in enumerate((expected, largest)):
            got = store.get(layer, mb)
            assert got.shape == tensor.shape
            assert torch.equal(got, tensor)
        store.end_step()
    finally:
        store.close()


@pytest.mark.parametrize(
    "mb_shapes, message",
    [
        ([(1, 3, 512)], "mb_shapes for k=2"),
        ([(1, 3, 512), (1, 9, 512)], "larger than chunk_shape"),
    ],
)
def test_microbatch_capacities_must_match_k_and_fit_the_spill_slot(tmp_path, mb_shapes, message):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore, BufferedSpillFile

    path = tmp_path / "invalid-shapes.bin"
    with pytest.raises(ValueError, match=message):
        ActivationStore(
            n_layers=2, k=2, chunk_shape=(1, 8, 512), mb_shapes=mb_shapes,
            dtype=torch.float32, device="cpu", pin=False, spill_layers=1,
            spill_factory=lambda size: BufferedSpillFile(str(path), size),
        )
    assert not path.exists()


def test_a_short_window_reads_only_what_it_put(tmp_path):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore, BufferedSpillFile

    store = ActivationStore(
        n_layers=2, k=3, chunk_shape=(1, 4, 8), dtype=torch.float32, device="cpu", pin=False,
        spill_layers=1, spill_factory=lambda size: BufferedSpillFile(str(tmp_path / "s"), size),
    )
    try:
        for k in (3, 1, 2):
            for layer in range(2):
                for mb in range(k):
                    store.put(layer, mb, torch.full((1, 4, 8), float(k * 10 + mb)))
            store.begin_backward()
            for layer in (1, 0):
                for mb in range(k):
                    assert torch.equal(store.get(layer, mb), torch.full((1, 4, 8), k * 10.0 + mb))
            store.end_step()
    finally:
        store.close()


def test_a_ring_that_cannot_be_allocated_removes_the_file(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils import l2l_activations

    path = tmp_path / "spill.bin"

    def no_memory(nbytes, *, pin):
        raise MemoryError("no staging")

    monkeypatch.setattr(l2l_activations, "aligned_buffer", no_memory)
    with pytest.raises(MemoryError):
        l2l_activations.ActivationStore(
            n_layers=2, k=2, chunk_shape=(1, 8, 128), dtype=torch.float32, device="cpu",
            pin=False, spill_layers=1,
            spill_factory=lambda size: l2l_activations.BufferedSpillFile(str(path), size),
        )
    assert not path.exists()


# --------------------------------------------------------------------------
# a full drive
# --------------------------------------------------------------------------
def _enospc(*_a, **_k):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_a_full_drive_while_creating_the_file_names_size_and_directory(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_activations
    from soup_cli.utils.l2l_spill import SpillDiskFullError, open_spill_file, spill_file_name

    path = str(tmp_path / spill_file_name())
    monkeypatch.setattr(l2l_activations, "_write_all", _enospc)
    with pytest.raises(SpillDiskFullError) as caught:
        open_spill_file(path, 5 * 4096)
    message = str(caught.value)
    assert str(tmp_path) in message and "20,480 bytes" in message
    assert "training.l2l_spill_dir" in message
    assert not os.path.exists(path)


def test_a_full_drive_in_the_writer_thread_removes_the_file(tmp_path, monkeypatch):
    import time

    import torch

    from soup_cli.utils.l2l_activations import ActivationStore, BufferedSpillFile
    from soup_cli.utils.l2l_spill import SpillDiskFullError, spill_file_name

    path = str(tmp_path / spill_file_name())
    monkeypatch.setattr(BufferedSpillFile, "write", lambda self, offset, buf: _enospc())
    store = ActivationStore(
        n_layers=3, k=2, chunk_shape=(1, 8, 128), dtype=torch.float32, device="cpu", pin=False,
        spill_layers=2, spill_factory=lambda size: BufferedSpillFile(path, size),
    )
    with pytest.raises(SpillDiskFullError) as caught:
        deadline = time.time() + 10
        while time.time() < deadline:
            store.put(0, 0, torch.zeros(1, 8, 128))
            time.sleep(0.01)
    message = str(caught.value)
    assert str(tmp_path) in message and f"{2 * 2 * 4096:,} bytes" in message
    assert "spill file was removed" in message
    assert not os.path.exists(path)
    store.close()


# --------------------------------------------------------------------------
# file names and the stale sweep
# --------------------------------------------------------------------------
def test_the_sweep_removes_only_files_of_dead_runs(tmp_path):
    from soup_cli.utils.l2l_spill import SPILL_FILE_GLOB, spill_files, sweep_stale

    assert SPILL_FILE_GLOB == "soup-l2l-spill-*.bin"
    dead, live = _dead_pid(), os.getpid()
    directory = tmp_path / "spill"
    directory.mkdir()
    names = {
        "dead": f"soup-l2l-spill-{dead}-0123abcd.bin",
        "dead_probe": f"soup-l2l-probe-{dead}-89abcdef.bin",
        "live": f"soup-l2l-spill-{live}-0123abcd.bin",
        "foreign": "notes.bin",
        "near_miss": f"soup-l2l-spill-{dead}.bin",
        "upper_hex": f"soup-l2l-spill-{dead}-FEDCBA98.bin",
    }
    for name in names.values():
        (directory / name).write_bytes(b"x")
    lines = []
    removed = sweep_stale(str(directory), notify=lines.append)
    gone = sorted(os.path.realpath(directory / names[key]) for key in ("dead", "dead_probe"))
    assert removed == gone
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        names[key] for key in ("live", "foreign", "near_miss", "upper_hex")
    )
    assert len(lines) == 1 and all(path in lines[0] for path in gone)
    assert spill_files(str(directory)) == [os.path.realpath(directory / names["live"])]
    assert sweep_stale(str(tmp_path / "missing")) == []


def _link_fixture(tmp_path):
    """A spill directory and, outside it, a file named like ours whose process is gone."""
    dead = _dead_pid()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / f"soup-l2l-spill-{dead}-00000000.bin"
    target.write_bytes(b"keep me")
    directory = tmp_path / "spill"
    directory.mkdir()
    return outside, target, directory, directory / f"soup-l2l-spill-{dead}-11111111.bin"


def _assert_link_untouched(directory, link, target):
    from soup_cli.utils.l2l_spill import spill_files, sweep_stale

    assert sweep_stale(str(directory)) == []
    assert spill_files(str(directory)) == []
    assert os.path.lexists(link)
    assert target.read_bytes() == b"keep me"


@pytest.mark.requires_symlink
def test_the_sweep_never_follows_or_removes_a_symlink(tmp_path):
    _outside, target, directory, link = _link_fixture(tmp_path)
    os.symlink(target, link)
    _assert_link_untouched(directory, link, target)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are a Windows construct")
def test_the_sweep_never_follows_or_removes_a_junction(tmp_path):
    import _winapi

    outside, target, directory, link = _link_fixture(tmp_path)
    _winapi.CreateJunction(str(outside), str(link))
    _assert_link_untouched(directory, link, target)


def _race_after_entry_inspection(monkeypatch, name, replace):
    """Keep the real directory/stat results; replace the file after a cached inspection."""
    real_scandir = os.scandir
    replaced = False

    def swap():
        nonlocal replaced
        if not replaced:
            replaced = True
            replace()

    class Entry:
        def __init__(self, entry):
            self.entry = entry

        def __getattr__(self, attribute):
            return getattr(self.entry, attribute)

        def is_file(self, **kwargs):
            answer = self.entry.is_file(**kwargs)
            if self.entry.name == name:
                swap()
            return answer

        def stat(self, **kwargs):
            answer = self.entry.stat(**kwargs)
            if self.entry.name == name:
                swap()
            return answer

    class Scan:
        def __init__(self, directory):
            self.scan = real_scandir(directory)

        def __iter__(self):
            return (Entry(entry) for entry in self.scan)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.scan.close()

    monkeypatch.setattr(os, "scandir", Scan)


@pytest.mark.requires_symlink
def test_the_sweep_keeps_an_inside_target_when_a_checked_file_becomes_a_symlink(
    tmp_path, monkeypatch,
):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"stale activation")
    target = directory / "notes.bin"
    target.write_bytes(b"foreign notes")

    def replace():
        candidate.unlink()
        candidate.symlink_to(target)

    _race_after_entry_inspection(monkeypatch, candidate.name, replace)
    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    assert l2l_spill.sweep_stale(str(directory)) == []
    assert target.read_bytes() == b"foreign notes"
    assert candidate.is_symlink()


def test_the_sweep_keeps_a_regular_file_replaced_after_entry_inspection(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"stale activation")
    retired = directory / "retired.bin"

    def replace():
        candidate.rename(retired)
        candidate.write_bytes(b"a replacement owned by another writer")

    _race_after_entry_inspection(monkeypatch, candidate.name, replace)
    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    assert l2l_spill.sweep_stale(str(directory)) == []
    assert candidate.read_bytes() == b"a replacement owned by another writer"
    assert retired.read_bytes() == b"stale activation"


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are a Windows construct")
def test_the_sweep_keeps_a_file_replaced_by_a_junction_after_inspection(tmp_path, monkeypatch):
    import _winapi

    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"stale activation")
    target = directory / "notes"
    target.mkdir()
    (target / "notes.bin").write_bytes(b"foreign notes")

    def replace():
        candidate.unlink()
        _winapi.CreateJunction(str(target), str(candidate))

    _race_after_entry_inspection(monkeypatch, candidate.name, replace)
    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    assert l2l_spill.sweep_stale(str(directory)) == []
    assert os.path.lexists(candidate)
    assert (target / "notes.bin").read_bytes() == b"foreign notes"


def test_the_sweep_cannot_delete_from_a_replacement_directory(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    name = "soup-l2l-spill-1234567890-0123abcd.bin"
    (directory / name).write_bytes(b"stale activation")
    retired = tmp_path / "retired"
    replaced = False

    def replace_directory(pid):
        nonlocal replaced
        try:
            directory.rename(retired)
        except PermissionError:
            # Windows may lock the scanned directory against replacement.
            return False
        replaced = True
        directory.mkdir()
        (directory / name).write_bytes(b"foreign replacement")
        return False

    monkeypatch.setattr(l2l_spill, "process_alive", replace_directory)
    removed = l2l_spill.sweep_stale(str(directory))
    if replaced:
        assert (directory / name).read_bytes() == b"foreign replacement"
        assert (retired / name).read_bytes() == b"stale activation"
        assert removed == []
    else:
        assert not (directory / name).exists()
        assert removed == [os.path.realpath(directory / name)]


@pytest.mark.skipif(os.name != "nt", reason="native Windows sharing semantics")
@pytest.mark.parametrize("replace_ancestor", [False, True])
def test_windows_sweep_blocks_directory_replacement_at_the_disposition_boundary(
    tmp_path, monkeypatch, replace_ancestor,
):
    from soup_cli.utils import l2l_spill

    ancestor = tmp_path / "owned"
    directory = ancestor / "spill"
    directory.mkdir(parents=True)
    name = "soup-l2l-spill-1234567890-0123abcd.bin"
    (directory / name).write_bytes(b"stale activation")
    moving = ancestor if replace_ancestor else directory
    retired = tmp_path / "retired"
    replacement = tmp_path / "incoming"
    replacement_spill = replacement / "spill" if replace_ancestor else replacement
    replacement_spill.mkdir(parents=True)
    (replacement_spill / name).write_bytes(b"foreign replacement")
    # Establish that these exact directories are renameable before the guard opens.
    moving.rename(retired)
    retired.rename(moving)
    real_dispose = l2l_spill._windows_dispose_file
    blocked = []

    def replace_then_dispose(kernel32, handle):
        try:
            moving.rename(retired)
        except PermissionError as exc:
            blocked.append(exc.winerror)
        else:
            replacement.rename(moving)
        return real_dispose(kernel32, handle)

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(l2l_spill, "_windows_dispose_file", replace_then_dispose)
    removed = l2l_spill.sweep_stale(str(directory), notify=lambda line: None)
    assert len(blocked) == 1 and blocked[0] in (5, 32)  # Access denied/sharing violation.
    assert removed == [os.path.realpath(directory / name)]
    assert not (directory / name).exists()
    assert (replacement_spill / name).read_bytes() == b"foreign replacement"
    assert not retired.exists()
    # Closing the guard releases the share lock, including any ancestor handle.
    moving.rename(retired)
    retired.rename(moving)


@pytest.mark.skipif(os.name != "nt", reason="native Windows file handle identity")
def test_windows_sweep_rejects_a_file_replaced_before_the_delete_handle_opens(
    tmp_path, monkeypatch,
):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"stale activation")
    retired = directory / "retired.bin"
    real_remove = l2l_spill._windows_remove_stale

    def replace_then_open(path, identity):
        candidate.rename(retired)
        candidate.write_bytes(b"foreign replacement")
        return real_remove(path, identity)

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(l2l_spill, "_windows_remove_stale", replace_then_open)
    assert l2l_spill.sweep_stale(str(directory)) == []
    assert candidate.read_bytes() == b"foreign replacement"
    assert retired.read_bytes() == b"stale activation"
    candidate.rename(directory / "replacement-after-sweep.bin")  # No leaked delete handle.


@pytest.mark.skipif(os.name != "nt", reason="native Windows sharing semantics")
def test_windows_sweep_blocks_basename_replacement_at_the_disposition_boundary(
    tmp_path, monkeypatch,
):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"stale activation")
    retired = directory / "retired.bin"
    foreign = directory / "incoming.bin"
    foreign.write_bytes(b"foreign replacement")
    real_dispose = l2l_spill._windows_dispose_file
    blocked = []

    def replace_then_dispose(kernel32, handle):
        try:
            candidate.rename(retired)
        except PermissionError as exc:
            blocked.append(exc.winerror)
        else:
            foreign.rename(candidate)
        return real_dispose(kernel32, handle)

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(l2l_spill, "_windows_dispose_file", replace_then_dispose)
    assert l2l_spill.sweep_stale(str(directory), notify=lambda line: None) == [
        os.path.realpath(candidate),
    ]
    assert len(blocked) == 1 and blocked[0] in (5, 32)
    assert foreign.read_bytes() == b"foreign replacement"
    assert not candidate.exists() and not retired.exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows file handle cleanup")
@pytest.mark.parametrize("interrupt", [False, True])
def test_windows_failed_disposition_keeps_the_file_and_closes_the_verified_handle(
    tmp_path, monkeypatch, interrupt,
):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"keep after failed disposition")
    calls = []

    def fail_dispose(kernel32, handle):
        calls.append(handle)
        if interrupt:
            raise KeyboardInterrupt
        raise OSError(errno.EIO, "disposition failed")

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(l2l_spill, "_windows_dispose_file", fail_dispose)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            l2l_spill.sweep_stale(str(directory))
    else:
        assert l2l_spill.sweep_stale(str(directory)) == []
    assert len(calls) == 1
    assert candidate.read_bytes() == b"keep after failed disposition"
    candidate.rename(directory / "after-sweep.bin")  # The native handle was closed.


@pytest.mark.skipif(os.name == "nt", reason="relative directory-fd unlink is POSIX only")
def test_the_sweep_unlinks_in_the_anchored_directory_when_it_moves_at_unlink(
    tmp_path, monkeypatch,
):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    name = "soup-l2l-spill-1234567890-0123abcd.bin"
    (directory / name).write_bytes(b"stale activation")
    retired = tmp_path / "retired"
    real_unlink = os.unlink

    def replace_then_unlink(path, *args, **kwargs):
        directory.rename(retired)
        directory.mkdir()
        (directory / name).write_bytes(b"foreign replacement")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(os, "unlink", replace_then_unlink)
    monkeypatch.setattr(os, "remove", replace_then_unlink)
    l2l_spill.sweep_stale(str(directory), notify=lambda line: None)
    assert (directory / name).read_bytes() == b"foreign replacement"
    assert not (retired / name).exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine capture")
def test_the_sweep_keeps_a_foreign_basename_created_at_the_unlink_boundary(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"stale activation")
    retired = directory / "retired.bin"
    real_unlink = os.unlink

    def replace_then_unlink(path, *args, **kwargs):
        if candidate.exists():
            candidate.rename(retired)
        candidate.write_bytes(b"foreign replacement")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(os, "unlink", replace_then_unlink)
    l2l_spill.sweep_stale(str(directory), notify=lambda line: None)
    assert candidate.read_bytes() == b"foreign replacement"
    assert list(directory.iterdir()) == [candidate]


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine capture")
@pytest.mark.parametrize("occupy_original", [False, True])
def test_the_sweep_restores_or_preserves_a_foreign_file_captured_during_rename(
    tmp_path, monkeypatch, caplog, occupy_original,
):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"stale activation")
    retired = directory / "retired.bin"
    real_rename = os.rename

    def replace_then_capture(src, dst, *args, **kwargs):
        real_rename(candidate, retired)
        candidate.write_bytes(b"foreign captured file")
        answer = real_rename(src, dst, *args, **kwargs)
        if occupy_original:
            candidate.write_bytes(b"another foreign replacement")
        return answer

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(os, "rename", replace_then_capture)
    with caplog.at_level("WARNING", logger="soup_cli.utils.l2l_spill"):
        assert l2l_spill.sweep_stale(str(directory)) == []
    assert retired.read_bytes() == b"stale activation"
    preserved = [path for path in directory.iterdir() if path.is_dir()]
    if occupy_original:
        assert candidate.read_bytes() == b"another foreign replacement"
        assert len(preserved) == 1
        captured = list(preserved[0].iterdir())
        assert len(captured) == 1 and captured[0].read_bytes() == b"foreign captured file"
        assert str(captured[0]) in caplog.text
    else:
        assert candidate.read_bytes() == b"foreign captured file"
        assert preserved == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine capture")
def test_the_sweep_preserves_a_foreign_directory_captured_during_rename(
    tmp_path, monkeypatch, caplog,
):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"stale activation")
    retired = directory / "retired.bin"
    real_rename = os.rename

    def replace_then_capture(src, dst, *args, **kwargs):
        real_rename(candidate, retired)
        candidate.mkdir()
        (candidate / "notes.bin").write_bytes(b"foreign directory content")
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(os, "rename", replace_then_capture)
    with caplog.at_level("WARNING", logger="soup_cli.utils.l2l_spill"):
        assert l2l_spill.sweep_stale(str(directory)) == []
    assert retired.read_bytes() == b"stale activation"
    preserved = [path for path in directory.iterdir() if path.is_dir()]
    assert len(preserved) == 1
    captured = list(preserved[0].iterdir())
    assert len(captured) == 1
    assert (captured[0] / "notes.bin").read_bytes() == b"foreign directory content"
    assert str(captured[0]) in caplog.text


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine capture")
def test_a_failed_capture_inspection_preserves_the_file_and_closes_directory_fds(
    tmp_path, monkeypatch, caplog,
):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"keep an unverified captured file")
    real_open, real_stat = os.open, os.stat
    opened = []

    def track_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def fail_captured_stat(path, *args, **kwargs):
        if len(opened) == 2 and kwargs.get("dir_fd") == opened[1]:
            raise OSError(errno.EIO, "capture inspection failed")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "stat", fail_captured_stat)
    with caplog.at_level("WARNING", logger="soup_cli.utils.l2l_spill"):
        assert l2l_spill.sweep_stale(str(directory)) == []
    preserved = [path for path in directory.iterdir() if path.is_dir()]
    assert len(preserved) == 1
    captured = list(preserved[0].iterdir())
    assert len(captured) == 1
    assert captured[0].read_bytes() == b"keep an unverified captured file"
    assert str(captured[0]) in caplog.text
    assert len(opened) == 2
    for descriptor in opened:
        with pytest.raises(OSError) as caught:
            os.fstat(descriptor)
        assert caught.value.errno == errno.EBADF


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine capture")
def test_an_interrupted_capture_keeps_the_claimed_file_and_closes_directory_fds(
    tmp_path, monkeypatch, caplog,
):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"preserve on interruption")
    real_open, real_stat = os.open, os.stat
    opened = []

    def track_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def interrupt_captured_stat(path, *args, **kwargs):
        if len(opened) == 2 and kwargs.get("dir_fd") == opened[1]:
            raise KeyboardInterrupt
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "stat", interrupt_captured_stat)
    with caplog.at_level("WARNING", logger="soup_cli.utils.l2l_spill"):
        with pytest.raises(KeyboardInterrupt):
            l2l_spill.sweep_stale(str(directory))
    preserved = [path for path in directory.iterdir() if path.is_dir()]
    assert len(preserved) == 1
    captured = list(preserved[0].iterdir())
    assert len(captured) == 1 and captured[0].read_bytes() == b"preserve on interruption"
    assert str(captured[0]) in caplog.text
    for descriptor in opened:
        with pytest.raises(OSError) as caught:
            os.fstat(descriptor)
        assert caught.value.errno == errno.EBADF


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine capture")
def test_a_failed_quarantine_open_keeps_the_original_and_removes_the_empty_directory(
    tmp_path, monkeypatch,
):
    from soup_cli.utils import l2l_spill

    directory = tmp_path / "spill"
    directory.mkdir()
    candidate = directory / "soup-l2l-spill-1234567890-0123abcd.bin"
    candidate.write_bytes(b"never captured")
    real_open = os.open
    opened = []

    def fail_quarantine_open(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None:
            raise OSError(errno.EACCES, "quarantine cannot be opened")
        descriptor = real_open(path, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(l2l_spill, "process_alive", lambda pid: False)
    monkeypatch.setattr(os, "open", fail_quarantine_open)
    assert l2l_spill.sweep_stale(str(directory)) == []
    assert list(directory.iterdir()) == [candidate]
    assert candidate.read_bytes() == b"never captured"
    assert len(opened) == 1
    with pytest.raises(OSError) as caught:
        os.fstat(opened[0])
    assert caught.value.errno == errno.EBADF


def test_a_process_is_alive_until_it_exits():
    from soup_cli.utils.l2l_spill import process_alive

    assert process_alive(os.getpid())
    assert not process_alive(_dead_pid())


# --------------------------------------------------------------------------
# the spill directory
# --------------------------------------------------------------------------
def _fake_mountinfo(tmp_path, monkeypatch, directory, fstype, source="none"):
    from soup_cli.utils import l2l_spill

    proc = tmp_path / "proc" / "self"
    proc.mkdir(parents=True)
    mount = os.path.realpath(directory).replace(os.sep, "/").replace(" ", "\\040")
    (proc / "mountinfo").write_text(
        "22 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n"
        f"40 22 0:50 / {mount} rw,relatime - {fstype} {source} rw\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(l2l_spill, "PROC_ROOT", str(tmp_path / "proc"))


def _prepare(directory, **kw):
    from soup_cli.utils.l2l_spill import prepare_spill_dir

    options = dict(explicit=True, spill_layers=3, file_bytes=12 * MIB)
    options.update(kw)
    return prepare_spill_dir(str(directory), **options)


@pytest.mark.parametrize(
    "fstype,source,reason",
    [
        ("tmpfs", "tmpfs", "tmpfs"),
        ("overlay", "overlay", "overlay"),
        ("nfs4", "server:/export", "network filesystem"),
        ("cifs", "//server/share", "network filesystem"),
    ],
)
def test_ram_stacked_and_network_filesystems_are_refused(tmp_path, monkeypatch, fstype,
                                                         source, reason):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    directory = tmp_path / "spill"
    directory.mkdir()
    _fake_mountinfo(tmp_path, monkeypatch, directory, fstype, source)
    with pytest.raises(SpillRefusedError) as caught:
        _prepare(directory)
    message = str(caught.value)
    assert str(directory) in message and reason in message
    assert "training.l2l_spill_dir" in message


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    """(allowed, outside): two real directories under tmp_path; only ``allowed`` is an allowed
    root, so a run without the guard still writes nowhere outside tmp_path."""
    from soup_cli.utils import l2l_spill

    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setattr(l2l_spill, "_allowed_roots", lambda: (str(allowed),))
    return allowed, outside


def _short_name(path) -> str:
    """The 8.3 form of ``path`` (Windows), or a skip when the volume has none."""
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer)):
        pytest.skip("8.3 short names are disabled on this volume")
    if buffer.value.lower() == str(path).lower():
        pytest.skip("8.3 short names are disabled on this volume")
    return buffer.value


def test_a_spill_dir_outside_the_allowed_roots_is_refused_and_not_created(roots):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    _allowed, outside = roots
    target = outside / "spill"
    with pytest.raises(SpillRefusedError) as caught:
        _prepare(target)
    message = str(caught.value)
    assert str(target) in message
    assert "outside $HOME, $CWD and $TMPDIR" in message
    assert "training.l2l_spill_dir" in message
    assert not target.exists()


def test_a_dir_inside_the_allowed_root_is_accepted(monkeypatch, direct_dir):
    from soup_cli.utils import l2l_spill

    monkeypatch.setattr(l2l_spill, "_allowed_roots", lambda: (direct_dir,))
    target = os.path.join(direct_dir, "spill")
    assert _prepare(target) == os.path.realpath(target)


def test_a_symlink_out_of_the_allowed_roots_is_refused(roots):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    allowed, outside = roots
    link = allowed / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    with pytest.raises(SpillRefusedError, match=r"outside \$HOME, \$CWD and \$TMPDIR"):
        _prepare(link / "spill")
    assert not (outside / "spill").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="8.3 short names exist on Windows only")
def test_a_windows_short_name_of_a_dir_outside_the_roots_is_refused(roots):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    _allowed, outside = roots
    long_outside = outside / "a-directory-with-a-long-name"
    long_outside.mkdir()
    short = _short_name(long_outside)
    with pytest.raises(SpillRefusedError, match=r"outside \$HOME, \$CWD and \$TMPDIR"):
        _prepare(os.path.join(short, "spill"))
    assert not (long_outside / "spill").exists()


def test_the_refusal_escapes_control_bytes_in_the_path(roots):
    from soup_cli.utils.l2l_spill import SpillRefusedError
    from soup_cli.utils.terminal import for_terminal

    _allowed, outside = roots
    hostile = str(outside / "a\x1b[31mb")
    with pytest.raises(SpillRefusedError) as caught:
        _prepare(hostile)
    message = str(caught.value)
    assert "\x1b" not in message
    assert for_terminal(hostile) in message
    assert not os.path.exists(hostile)


def test_a_null_byte_in_the_spill_dir_is_refused(tmp_path):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    with pytest.raises(SpillRefusedError, match="null byte"):
        _prepare(str(tmp_path / "sp\x00ill"))


@pytest.mark.skipif(os.name != "nt", reason="8.3 short names exist on Windows only")
def test_a_windows_short_name_of_an_allowed_dir_is_accepted(tmp_path, direct_dir):
    import ctypes

    long_dir = os.path.join(direct_dir, "a-directory-with-a-long-name")
    os.makedirs(long_dir)
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(long_dir, buffer, len(buffer)):
        pytest.skip("8.3 short names are disabled on this volume")
    short = buffer.value
    if short.lower() == long_dir.lower():
        pytest.skip("8.3 short names are disabled on this volume")
    assert _prepare(short) == os.path.realpath(long_dir)


def test_a_dir_under_cwd_is_accepted(tmp_path, monkeypatch, direct_dir):
    monkeypatch.chdir(direct_dir)
    assert _prepare("spill") == os.path.realpath(os.path.join(direct_dir, "spill"))


def _fake_darwin_volume(tmp_path, monkeypatch, *, fstype="apfs", flags=0x1000,
                        info=None, stores=None):
    import plistlib

    from soup_cli.utils import l2l_spill, layer_stream

    monkeypatch.setattr(l2l_spill.sys, "platform", "darwin")
    monkeypatch.setattr(l2l_spill, "PROC_ROOT", str(tmp_path / "no-proc"))
    monkeypatch.setattr(l2l_spill, "_darwin_mount_info", lambda path: {
        "fstype": fstype, "flags": flags, "mount_point": "/Volumes/spill",
        "source": "/dev/disk7s1",
    }, raising=False)
    monkeypatch.setattr(layer_stream, "_resolve_tool", lambda name, *fallbacks: (
        "/usr/sbin/diskutil" if name == "diskutil" else None
    ))
    volumes = {"/dev/disk7s1": {"BusProtocol": "Apple Fabric"} if info is None else info}
    volumes.update(stores or {})

    def run(args, **kwargs):
        assert args[:-1] == ["/usr/sbin/diskutil", "info", "-plist"]
        assert kwargs.get("shell") is False
        payload = volumes[args[-1]]
        return types.SimpleNamespace(
            returncode=0,
            stdout=payload if isinstance(payload, bytes) else plistlib.dumps(payload),
        )

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(l2l_spill, "_round_trip", lambda directory: None)
    monkeypatch.setattr(l2l_spill.shutil, "disk_usage", lambda path: types.SimpleNamespace(
        free=10**15,
    ))


@pytest.mark.parametrize("fstype", ["nfs", "smbfs", "afpfs"])
def test_darwin_refuses_nonlocal_volumes_even_when_unbuffered_io_would_succeed(
    tmp_path, monkeypatch, fstype,
):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    _fake_darwin_volume(tmp_path, monkeypatch, fstype=fstype, flags=0)
    with pytest.raises(SpillRefusedError, match="network filesystem"):
        _prepare(tmp_path / "spill")


@pytest.mark.parametrize("fstype,info,stores", [
    ("tmpfs", {}, {}),
    ("hfs", {"BusProtocol": "Disk Image", "MediaName": "Apple RAM Disk Media"}, {}),
    ("apfs", {"BusProtocol": "Apple Fabric", "VirtualOrPhysical": "Virtual",
              "APFSPhysicalStores": [{"APFSPhysicalStore": "disk2s1"}]},
     {"/dev/disk2": {"BusProtocol": "Disk Image", "DiskImage": True}}),
])
def test_darwin_refuses_ram_and_ram_backed_apfs_volumes(
    tmp_path, monkeypatch, fstype, info, stores,
):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    _fake_darwin_volume(tmp_path, monkeypatch, fstype=fstype, info=info, stores=stores)
    with pytest.raises(SpillRefusedError, match="RAM"):
        _prepare(tmp_path / "spill")


def test_darwin_local_apfs_without_a_physical_disk_id_keeps_the_explicit_warning_policy(
    tmp_path, monkeypatch,
):
    _fake_darwin_volume(tmp_path, monkeypatch, info={
        "FilesystemType": "apfs", "BusProtocol": "Apple Fabric", "VirtualOrPhysical": "Virtual",
    })
    lines = []
    directory = tmp_path / "spill"
    assert _prepare(directory, tier="disk", weight_dirs=[str(tmp_path / "weights")],
                    notify=lines.append) == os.path.realpath(directory)
    assert len(lines) == 1 and "could not verify" in lines[0]


@pytest.mark.parametrize("info", [b"not a diskutil plist", {}])
def test_darwin_cannot_accept_unknown_volume_backing_as_a_local_disk(tmp_path, monkeypatch, info):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    _fake_darwin_volume(tmp_path, monkeypatch, info=info)
    with pytest.raises(SpillRefusedError, match="backing could not be verified"):
        _prepare(tmp_path / "spill")


@pytest.fixture()
def local_volume(monkeypatch):
    """Supply known local backing for tests of subsequent spill checks."""
    from soup_cli.utils import l2l_spill

    monkeypatch.setattr(l2l_spill, "filesystem_refusal", lambda path: None)


def test_a_volume_without_unbuffered_io_is_refused(tmp_path, monkeypatch, local_volume):
    from soup_cli.utils import l2l_activations, l2l_spill

    directory = tmp_path / "spill"

    def refuse(path):
        raise OSError(errno.EINVAL, "Invalid argument")

    monkeypatch.setattr(l2l_activations, "_open_unbuffered", refuse)
    monkeypatch.setattr(l2l_spill.shutil, "disk_usage", lambda p: types.SimpleNamespace(
        free=10**15))
    with pytest.raises(l2l_spill.SpillRefusedError, match=r"(?s)unbuffered.*l2l_spill_dir"):
        _prepare(directory)
    assert os.listdir(directory) == []  # the probe file is gone


def test_a_drive_short_of_the_file_plus_the_margin_is_refused(
    tmp_path, monkeypatch, local_volume,
):
    from soup_cli.utils import l2l_spill

    monkeypatch.setattr(l2l_spill.shutil, "disk_usage", lambda p: types.SimpleNamespace(
        free=GIB + 12 * MIB - 1))
    with pytest.raises(l2l_spill.SpillRefusedError) as caught:
        _prepare(tmp_path / "spill")
    message = str(caught.value)
    assert "1 GiB" in message and str(tmp_path / "spill") in message
    assert l2l_spill.space_margin_bytes(100 * GIB) == 10 * GIB


def test_a_usable_directory_is_created_and_returned(direct_dir):
    path = os.path.join(direct_dir, "nested", "spill")
    assert _prepare(path) == os.path.realpath(path)
    assert os.listdir(path) == []


# --------------------------------------------------------------------------
# the disk tier
# --------------------------------------------------------------------------
@pytest.fixture()
def disks(monkeypatch, local_volume):
    """``physical_disks`` answered from a table keyed by directory name."""
    from soup_cli.utils import l2l_spill

    table = {}

    def physical(path):
        answer = table.get(os.path.basename(os.path.normpath(path)))
        if answer is None:
            return None, "not read on this platform"
        return frozenset(answer), ", ".join(answer)

    monkeypatch.setattr(l2l_spill, "physical_disks", physical)
    monkeypatch.setattr(l2l_spill, "_round_trip", lambda directory: None)
    monkeypatch.setattr(l2l_spill.shutil, "disk_usage", lambda p: types.SimpleNamespace(
        free=10**15))
    return table


@pytest.mark.parametrize("layout", ["same", "unknown"])
def test_the_default_dir_on_the_weights_disk_or_an_unknown_one_stops(tmp_path, disks, layout):
    from soup_cli.utils.l2l_spill import SpillRefusedError

    disks["shards"] = ["nvme0n1"]
    if layout == "same":
        disks["_l2l_spill"] = ["nvme0n1"]
    with pytest.raises(SpillRefusedError) as caught:
        _prepare(tmp_path / "_l2l_spill", explicit=False, tier="disk",
                 weight_dirs=[str(tmp_path / "shards")])
    message = str(caught.value)
    assert "training.l2l_spill_dir" in message and "another disk" in message
    assert ("same physical disk" in message) == (layout == "same")


@pytest.mark.parametrize("layout", ["same", "unknown"])
def test_an_explicit_dir_on_the_weights_disk_or_an_unknown_one_warns(tmp_path, disks, layout):
    disks["shards"] = ["nvme0n1"]
    if layout == "same":
        disks["mine"] = ["nvme0n1"]
    lines = []
    _prepare(tmp_path / "mine", explicit=True, tier="disk",
             weight_dirs=[str(tmp_path / "shards")], notify=lines.append)
    assert len(lines) == 1
    if layout == "same":
        assert "adds time to every step" in lines[0]
        assert "13%" not in lines[0] and "3090" not in lines[0]
    else:
        assert "could not verify" in lines[0]


def test_another_disk_or_the_ram_tier_is_silent(tmp_path, disks):
    disks["shards"] = ["nvme0n1"]
    disks["_l2l_spill"] = ["nvme1n1"]
    lines = []
    _prepare(tmp_path / "_l2l_spill", explicit=False, tier="disk",
             weight_dirs=[str(tmp_path / "shards")], notify=lines.append)
    _prepare(tmp_path / "elsewhere", explicit=False, tier="ram", notify=lines.append)
    assert lines == []


def _link_dir(target, link):
    """A directory link: a symlink, or a junction where symlinks need a privilege."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def test_a_partition_and_a_device_mapper_volume_resolve_to_their_whole_disks(tmp_path):
    """The /sys/devices layout: a partition sits inside its disk and has a ``partition``
    file; a dm device lists the block devices under it in ``slaves``."""
    from soup_cli.utils.l2l_spill import _whole_disks

    devices = tmp_path / "devices"
    nvme = devices / "nvme0n1"
    (nvme / "nvme0n1p2").mkdir(parents=True)
    (nvme / "nvme0n1p2" / "partition").write_text("2")
    sda = devices / "sda"
    (sda / "sda1").mkdir(parents=True)
    (sda / "sda1" / "partition").write_text("1")
    dm = devices / "dm-0"
    (dm / "slaves").mkdir(parents=True)
    _link_dir(nvme / "nvme0n1p2", dm / "slaves" / "nvme0n1p2")
    _link_dir(sda / "sda1", dm / "slaves" / "sda1")

    assert _whole_disks(str(nvme / "nvme0n1p2")) == ["nvme0n1"]
    assert _whole_disks(str(nvme)) == ["nvme0n1"]
    assert _whole_disks(str(dm)) == ["nvme0n1", "sda"]


def test_the_physical_disk_is_unknown_off_linux(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_spill

    for platform in ("win32", "darwin"):
        monkeypatch.setattr(l2l_spill.sys, "platform", platform)
        disks, why = l2l_spill.physical_disks(str(tmp_path))
        assert disks is None and platform in why


@pytest.mark.skipif(not hasattr(os, "major"), reason="st_dev major/minor is POSIX only")
def test_the_physical_disk_is_read_from_sys_dev_block(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_spill

    monkeypatch.setattr(l2l_spill.sys, "platform", "linux")
    st_dev = os.stat(tmp_path).st_dev
    sys_root = tmp_path / "sys"
    disk = sys_root / "devices" / "nvme1n1"
    (disk / "nvme1n1p1").mkdir(parents=True)
    (disk / "nvme1n1p1" / "partition").write_text("1")
    (sys_root / "dev" / "block").mkdir(parents=True)
    _link_dir(disk / "nvme1n1p1",
              sys_root / "dev" / "block" / f"{os.major(st_dev)}:{os.minor(st_dev)}")
    monkeypatch.setattr(l2l_spill, "SYS_ROOT", str(sys_root))
    disks, _why = l2l_spill.physical_disks(str(tmp_path / "not-yet" / "created"))
    assert disks == frozenset({"nvme1n1"})


# --------------------------------------------------------------------------
# the host-RAM budget
# --------------------------------------------------------------------------
def _stat(inactive_file=0, shmem=0, *, v1=False):
    key = "total_inactive_file" if v1 else "inactive_file"
    return f"anon 0\nfile 0\n{'total_shmem' if v1 else 'shmem'} {shmem}\n{key} {inactive_file}\n"


def cgroup_tree(tmp_path, monkeypatch, levels, *, v1=False, own="/user.slice/run.scope",
                psutil=PLENTY, name="tree"):
    """A fake /proc/self/cgroup naming ``own`` and a cgroup mount whose directories hold the
    files in ``levels`` ({"" (root) | "user.slice" | ...: {file: text | DIRECTORY}}).
    Returns the mount root as the product spells level paths (root + "/part"...)."""
    from soup_cli.utils import l2l_spill

    base = tmp_path / name
    proc = base / "proc" / "self"
    proc.mkdir(parents=True)
    line = f"9:cpuset,memory:{own}" if v1 else f"0::{own}"
    (proc / "cgroup").write_text(f"12:pids:{own}\n{line}\n", encoding="utf-8")
    root = base / ("v1" if v1 else "v2")
    root.mkdir()
    for rel, files in levels.items():
        directory = root / rel if rel else root
        directory.mkdir(parents=True, exist_ok=True)
        for file_name, content in files.items():
            if content is DIRECTORY:
                (directory / file_name).mkdir()
            else:
                (directory / file_name).write_text(str(content), encoding="utf-8")
    monkeypatch.setattr(l2l_spill, "PROC_ROOT", str(base / "proc"))
    monkeypatch.setattr(l2l_spill, "CGROUP_V2_ROOT", str(base / "v2"))
    monkeypatch.setattr(l2l_spill, "CGROUP_V1_ROOT", str(base / "v1"))
    monkeypatch.setattr("soup_cli.utils.layer_stream.free_ram_bytes", lambda: psutil)
    return str(root)


def _v2(limit="max", current=0, *, high=None, stat=None):
    files = {"memory.max": limit, "memory.current": current,
             "memory.stat": _stat() if stat is None else stat}
    if high is not None:
        files["memory.high"] = high
    return files


def _budget():
    from soup_cli.utils.l2l_spill import ram_budget

    return ram_budget()


# --------------------------------------------------------------------------
# the share: 70%, or what the reserve leaves, whichever is smaller
# --------------------------------------------------------------------------
MIB = 2**20
#: Where 0.7 x available and available - reserve meet: reserve / 0.3, about 1.79 GB.
SWITCH = 512 * MIB * 10 // 3


@pytest.mark.parametrize("available, budget, rule", [
    (4 * GIB, int(4 * GIB * 0.7), "70% of"),
    (SWITCH + MIB, int((SWITCH + MIB) * 0.7), "70% of"),  # above it: the 70% side
    (SWITCH - MIB, SWITCH - MIB - 512 * MIB, "reserve"),  # below it: the reserve side
    (840 * MIB, 328 * MIB, "reserve"),  # A3's calibrated limit for 16 of Mistral's 32 layers
    (512 * MIB, 0, "reserve"),
    (100 * MIB, 0, "floored at 0"),
])
def test_the_budget_never_leaves_less_than_the_reserve(tmp_path, monkeypatch, available,
                                                       budget, rule):
    from soup_cli.utils.l2l_spill import RAM_RESERVE_BYTES

    assert RAM_RESERVE_BYTES == 512 * MIB
    cgroup_tree(tmp_path, monkeypatch, {}, psutil=available)
    seen = _budget()
    assert seen.available_bytes == available
    assert seen.budget_bytes == budget
    assert available - seen.budget_bytes >= min(available, RAM_RESERVE_BYTES)
    assert rule in seen.describe()


def test_the_refusal_threshold_is_the_reserve_plus_every_layer_spilled(tmp_path, monkeypatch):
    """Mistral-7B at k 4, seq 512, pinned: every layer spilled still needs the 64 MiB of
    staging rings, so below 512 + 64 MiB available the run stops and names the reserve."""
    from soup_cli.utils.l2l_spill import SpillRefusedError, plan_spill

    def plan(available):
        cgroup_tree(tmp_path, monkeypatch, {}, psutil=available, name=f"t{available}")
        budget = _budget()
        return plan_spill(budget.budget_bytes, n_layers=32, k=4, chunk_bytes=4 * MIB,
                          pin=True, budget_source=budget.describe())

    assert plan(576 * MIB).spill_layers == 32
    with pytest.raises(SpillRefusedError) as caught:
        plan(576 * MIB - 1)
    message = str(caught.value)
    assert "512 MiB reserve" in message and "67.1 MB" in message
    for fix in ("gradient_accumulation_steps", "data.max_length", "batch_size", "free RAM"):
        assert fix in message


def test_session_1s_check_child_is_refused_instead_of_spilling(tmp_path, monkeypatch):
    """A3 stage 2 session 1: 0.43 GB available at the plan point. The 0.7 share planned 18
    of 32 layers and the child was OOM-killed in its first step; now the run stops."""
    from soup_cli.utils.l2l_spill import SpillRefusedError, plan_spill

    cgroup_tree(tmp_path, monkeypatch, {}, psutil=430_000_000)
    budget = _budget()
    assert budget.budget_bytes == 0
    with pytest.raises(SpillRefusedError):
        plan_spill(budget.budget_bytes, n_layers=32, k=4, chunk_bytes=4 * MIB, pin=True)

# --------------------------------------------------------------------------
# v2
# --------------------------------------------------------------------------
def test_a_child_without_a_limit_is_limited_by_its_parent(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_spill import plan_spill

    root = cgroup_tree(tmp_path, monkeypatch, {
        "": {},
        "user.slice": _v2(limit=4 * GIB, current=1 * GIB),
        "user.slice/run.scope": _v2(limit="max", current=1 * GIB),
    })
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice"
    assert budget.available_bytes == 3 * GIB
    assert budget.budget_bytes == int(3 * GIB * 0.7)
    assert budget.limiting_numbers == {
        "limit": 4 * GIB, "limit_file": "memory.max", "used": 1 * GIB, "inactive_file": 0,
        "remainder": 3 * GIB,
    }
    # a store above that budget spills; psutil alone (10^15 bytes free) would keep it resident
    plan = plan_spill(budget.budget_bytes, n_layers=40, k=8, chunk_bytes=16 * 2**20, pin=False)
    assert plan.all_resident_bytes > budget.budget_bytes
    assert plan.spill_layers > 0
    assert plan.host_bytes <= budget.budget_bytes


def test_a_parent_with_a_larger_limit_but_a_smaller_remainder_decides(tmp_path, monkeypatch):
    root = cgroup_tree(tmp_path, monkeypatch, {
        "user.slice": _v2(limit=8 * GIB, current=7 * GIB),  # nearly full of siblings
        "user.slice/run.scope": _v2(limit=4 * GIB, current=1 * GIB),
    })
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice"
    assert budget.available_bytes == 1 * GIB


def test_the_own_group_decides_when_it_is_the_tightest(tmp_path, monkeypatch):
    root = cgroup_tree(tmp_path, monkeypatch, {
        "user.slice": _v2(limit=16 * GIB, current=2 * GIB),
        "user.slice/run.scope": _v2(limit=4 * GIB, current=2 * GIB),
    })
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice/run.scope"
    assert budget.available_bytes == 2 * GIB


def test_without_any_limit_psutil_decides(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, {
        "": {"memory.current": 5 * GIB},
        "user.slice": _v2(),
        "user.slice/run.scope": _v2(),
    }, psutil=6 * GIB)
    budget = _budget()
    assert budget.limiting_level == "psutil"
    assert budget.limiting_numbers == {"psutil_available": 6 * GIB}
    assert budget.budget_bytes == int(6 * GIB * 0.7)
    assert "psutil available" in budget.describe()


def test_psutil_decides_when_it_is_below_every_remainder(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, {"user.slice": _v2(limit=8 * GIB)}, psutil=1 * GIB)
    assert _budget().limiting_level == "psutil"


def test_without_psutil_and_without_a_limit_no_budget_is_known(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_spill import plan_spill

    cgroup_tree(tmp_path, monkeypatch, {"user.slice": _v2()}, psutil=None)
    budget = _budget()
    assert budget.budget_bytes is None and budget.limiting_level is None
    # nothing spills: everything stays resident, as before the spill existed
    assert plan_spill(None, n_layers=80, k=16, chunk_bytes=2**30, pin=True).spill_layers == 0


def test_without_psutil_a_cgroup_limit_still_gives_a_budget(tmp_path, monkeypatch):
    root = cgroup_tree(tmp_path, monkeypatch, {"user.slice": _v2(limit=2 * GIB)}, psutil=None)
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice"
    assert budget.budget_bytes == int(2 * GIB * 0.7)


@pytest.mark.parametrize("broken", ["garbage", DIRECTORY])
def test_an_unreadable_limit_is_not_no_limit(tmp_path, monkeypatch, caplog, broken):
    root = cgroup_tree(tmp_path, monkeypatch, {
        "user.slice": _v2(limit=3 * GIB, current=1 * GIB),
        "user.slice/run.scope": {**_v2(current=1 * GIB), "memory.max": broken},
    })
    with caplog.at_level(logging.WARNING, logger="soup_cli.utils.l2l_spill"):
        budget = _budget()
    own = root + "/user.slice/run.scope"
    assert budget.limiting_level == root + "/user.slice"
    assert budget.available_bytes == 2 * GIB
    assert [row["level"] for row in budget.failed_levels] == [own]
    assert any(own in record.getMessage() for record in caplog.records)
    assert own in budget.describe()


def test_inactive_file_cache_does_not_shrink_the_budget(tmp_path, monkeypatch):
    levels = {"user.slice/run.scope": _v2(limit=8 * GIB, current=2 * GIB, stat=_stat(0))}
    cgroup_tree(tmp_path, monkeypatch, levels, name="clean")
    clean = _budget()
    cached = {"user.slice/run.scope": _v2(limit=8 * GIB, current=6 * GIB, stat=_stat(4 * GIB))}
    cgroup_tree(tmp_path, monkeypatch, cached, name="cached")
    full_of_cache = _budget()
    assert clean.available_bytes == full_of_cache.available_bytes == 6 * GIB
    assert full_of_cache.limiting_numbers["inactive_file"] == 4 * GIB


def test_shmem_shrinks_the_budget(tmp_path, monkeypatch):
    """Pinned host memory is charged as shmem, on the anon LRU: it raises memory.current
    and never shows up as inactive_file."""
    base = {"user.slice/run.scope": _v2(limit=8 * GIB, current=3 * GIB, stat=_stat(1 * GIB))}
    cgroup_tree(tmp_path, monkeypatch, base, name="before")
    before = _budget()
    pinned = {"user.slice/run.scope": _v2(
        limit=8 * GIB, current=5 * GIB, stat=_stat(1 * GIB, shmem=2 * GIB))}
    cgroup_tree(tmp_path, monkeypatch, pinned, name="after")
    after = _budget()
    assert before.available_bytes == 6 * GIB
    assert after.available_bytes == 4 * GIB
    assert after.budget_bytes < before.budget_bytes


def test_memory_high_below_memory_max_is_the_limit(tmp_path, monkeypatch):
    root = cgroup_tree(tmp_path, monkeypatch, {
        "user.slice/run.scope": _v2(limit=8 * GIB, high=3 * GIB, current=1 * GIB),
    })
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice/run.scope"
    assert budget.limiting_numbers["limit"] == 3 * GIB
    assert budget.limiting_numbers["limit_file"] == "memory.high"
    assert budget.available_bytes == 2 * GIB


def test_memory_high_of_max_leaves_memory_max(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, {
        "user.slice/run.scope": _v2(limit=4 * GIB, high="max", current=1 * GIB),
    })
    assert _budget().limiting_numbers["limit_file"] == "memory.max"


@pytest.mark.parametrize("stat", [DIRECTORY, None])
def test_an_unreadable_memory_stat_counts_limit_minus_used(tmp_path, monkeypatch, caplog, stat):
    files = _v2(limit=8 * GIB, current=5 * GIB)
    if stat is DIRECTORY:
        files["memory.stat"] = DIRECTORY
    else:
        del files["memory.stat"]
    root = cgroup_tree(tmp_path, monkeypatch, {"user.slice/run.scope": files})
    with caplog.at_level(logging.WARNING, logger="soup_cli.utils.l2l_spill"):
        budget = _budget()
    own = root + "/user.slice/run.scope"
    assert budget.available_bytes == 3 * GIB
    assert budget.limiting_numbers["inactive_file"] is None
    assert any(own in r.getMessage() and "memory.stat" in r.getMessage() for r in caplog.records)
    assert "limit - used" in budget.describe()


def test_a_level_missing_from_this_mount_namespace_is_skipped(tmp_path, monkeypatch):
    # a container sees its host path in /proc/self/cgroup; only the mount root exists
    root = cgroup_tree(tmp_path, monkeypatch, {"": _v2(limit=2 * GIB)},
                       own="/docker/0123456789ab")
    assert _budget().limiting_level == root


def _mounted_cgroup_tree(tmp_path, monkeypatch, levels, *, v1=False, own="/tenant/job",
                         mount_root="/"):
    """An actual mountinfo target, deliberately different from both conventional roots."""
    from pathlib import Path

    from soup_cli.utils import l2l_spill

    conventional = Path(cgroup_tree(tmp_path, monkeypatch, levels, v1=v1, own=own))
    mounted = conventional.with_name("cpu,cpuacct,memory" if v1 else "unified memory")
    conventional.rename(mounted)
    mount_point = str(mounted).replace(os.sep, "/").replace(" ", "\\040")
    fstype, controllers = ("cgroup", "cpu,cpuacct,memory") if v1 else ("cgroup2", "")
    (tmp_path / "tree" / "proc" / "self" / "mountinfo").write_text(
        "22 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n"
        f"40 22 0:50 {mount_root} {mount_point} rw,relatime - "
        f"{fstype} cgroup rw{',' + controllers if controllers else ''}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(l2l_spill, "CGROUP_V1_ROOT", str(tmp_path / "absent-v1"))
    monkeypatch.setattr(l2l_spill, "CGROUP_V2_ROOT", str(tmp_path / "absent-v2"))
    return str(mounted).replace(os.sep, "/")


def test_a_nondefault_v2_mount_keeps_a_nested_limit_out_of_the_host_budget(
    tmp_path, monkeypatch,
):
    from soup_cli.utils.l2l_spill import plan_spill

    root = _mounted_cgroup_tree(tmp_path, monkeypatch, {
        "": _v2(limit=16 * GIB),
        "tenant": _v2(limit=8 * GIB),
        "tenant/job": _v2(limit=4 * GIB, current=GIB),
    })
    budget = _budget()
    assert budget.limiting_level == root + "/tenant/job"
    assert budget.available_bytes == 3 * GIB
    assert budget.budget_bytes == int(3 * GIB * 0.7)
    plan = plan_spill(budget.budget_bytes, n_layers=40, k=8, chunk_bytes=16 * MIB, pin=False)
    assert plan.spill_layers > 0


@pytest.mark.parametrize("v1", [False, True])
def test_a_cgroup_subtree_mount_translates_membership_and_reads_nested_ancestry(
    tmp_path, monkeypatch, v1,
):
    files = _v1 if v1 else _v2
    root = _mounted_cgroup_tree(tmp_path, monkeypatch, {
        "": files(16 * GIB, GIB),
        "service": files(4 * GIB, 3 * GIB),
        "service/job": files(8 * GIB, GIB),
    }, v1=v1, own="/host/tenant/service/job", mount_root="/host/tenant")
    budget = _budget()
    assert budget.limiting_level == root + "/service"
    assert budget.available_bytes == GIB
    assert budget.budget_bytes == 512 * MIB
    assert {level.path for level in budget.levels} == {
        root, root + "/service", root + "/service/job",
    }


@pytest.mark.parametrize("v1", [False, True])
@pytest.mark.parametrize("mount_root,own", [("/..", "/"), ("/../..", "/job")])
def test_an_inherited_namespace_mount_keeps_its_known_ancestor_limit(
    tmp_path, monkeypatch, caplog, v1, mount_root, own,
):
    files = _v1 if v1 else _v2
    root = _mounted_cgroup_tree(tmp_path, monkeypatch, {
        "": files(4 * GIB, 3 * GIB),
        # The hidden namespace root may be another child: root/job is not self/job.
        "job": files(512 * MIB, 0),
    }, v1=v1, own=own, mount_root=mount_root)
    with caplog.at_level(logging.WARNING, logger="soup_cli.utils.l2l_spill"):
        budget = _budget()
    assert budget.limiting_level == root
    assert budget.available_bytes == GIB
    assert budget.budget_bytes == 512 * MIB
    assert [level.path for level in budget.levels] == [root]
    assert any(
        "tighter nested memory limit may be unknown" in record.getMessage()
        and root in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("v1", [False, True])
def test_an_unrelated_namespace_bind_mount_is_not_a_memory_ancestor(tmp_path, monkeypatch, v1):
    files = _v1 if v1 else _v2
    _mounted_cgroup_tree(tmp_path, monkeypatch, {
        "": files(512 * MIB, 0), "job": files(512 * MIB, 0),
    }, v1=v1, own="/job", mount_root="/../sibling")
    budget = _budget()
    assert budget.levels == []
    assert budget.limiting_level == "psutil"
    assert budget.available_bytes == PLENTY


@pytest.mark.parametrize("v1", [False, True])
def test_a_shadowing_cgroup_bind_cannot_charge_an_unrelated_group(tmp_path, monkeypatch, v1):
    files = _v1 if v1 else _v2
    root = _mounted_cgroup_tree(tmp_path, monkeypatch, {
        "": files(4 * GIB, GIB),
        # These visible directories belong to /other, not this process's /tenant.
        "tenant": files(GIB, 0),
        "tenant/job": files(512 * MIB, 0),
    }, v1=v1)
    mount_point = (root + "/tenant").replace(" ", "\\040")
    fstype, options = ("cgroup", "rw,cpu,cpuacct,memory") if v1 else ("cgroup2", "rw")
    mountinfo = tmp_path / "tree" / "proc" / "self" / "mountinfo"
    with mountinfo.open("a", encoding="utf-8") as handle:
        handle.write(f"41 40 0:50 /other {mount_point} rw - {fstype} cgroup {options}\n")
    budget = _budget()
    assert [level.path for level in budget.levels] == [root]
    assert budget.limiting_level == root
    assert budget.available_bytes == 3 * GIB
    assert budget.budget_bytes == int(3 * GIB * 0.7)


@pytest.mark.parametrize("v1", [False, True])
def test_a_stacked_overmount_cannot_expose_a_hidden_child_cgroup_limit(
    tmp_path, monkeypatch, caplog, v1,
):
    files = _v1 if v1 else _v2
    root = _mounted_cgroup_tree(tmp_path, monkeypatch, {
        "": files(4 * GIB, GIB),
        # The old child mount is hidden below the later overmount. These visible
        # files now belong to /other/subtree, not this process's /tenant.
        "subtree": files(GIB, 0), "subtree/job": files(512 * MIB, 0),
    }, v1=v1)
    known = tmp_path / "known-ancestor"
    known.mkdir()
    for name, content in files(4 * GIB, GIB).items():
        (known / name).write_text(str(content), encoding="utf-8")
    known_path = str(known).replace(os.sep, "/")
    mount_point = root.replace(" ", "\\040")
    known_point = known_path.replace(" ", "\\040")
    fstype, options = ("cgroup", "rw,cpu,cpuacct,memory") if v1 else ("cgroup2", "rw")
    mountinfo = tmp_path / "tree" / "proc" / "self" / "mountinfo"
    with mountinfo.open("a", encoding="utf-8") as handle:
        handle.write(
            f"41 40 0:50 /tenant {mount_point}/subtree rw - {fstype} cgroup {options}\n"
            f"42 40 0:50 /other {mount_point} rw - {fstype} cgroup {options}\n"
            f"43 22 0:50 / {known_point} rw - {fstype} cgroup {options}\n"
        )
    with caplog.at_level(logging.WARNING, logger="soup_cli.utils.l2l_spill"):
        budget = _budget()
    assert [level.path for level in budget.levels] == [known_path]
    assert budget.limiting_level == known_path
    assert budget.available_bytes == 3 * GIB
    assert budget.budget_bytes == int(3 * GIB * 0.7)
    assert any("stacked mounts" in record.getMessage() for record in caplog.records)


def test_explicit_cgroup_roots_keep_the_fixture_path_convention(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_spill

    root = cgroup_tree(tmp_path, monkeypatch, {"user.slice": _v2(limit=2 * GIB)})
    proc = tmp_path / "tree" / "proc"
    (proc / "self" / "mountinfo").write_text(
        "40 22 0:50 / /unrelated rw - cgroup2 cgroup rw\n", encoding="utf-8",
    )
    budget = l2l_spill.ram_budget(
        proc_root=str(proc), v2_root=root, v1_root=str(tmp_path / "absent"),
    )
    assert budget.limiting_level == root + "/user.slice"
    assert budget.available_bytes == 2 * GIB


# --------------------------------------------------------------------------
# v1
# --------------------------------------------------------------------------
def _v1(limit, usage=0, inactive=0, shmem=0):
    return {
        "memory.limit_in_bytes": limit,
        "memory.usage_in_bytes": usage,
        "memory.stat": _stat(inactive, shmem, v1=True),
    }


def test_v1_infinite_is_skipped_and_the_finite_parent_applies(tmp_path, monkeypatch):
    infinite = 9223372036854771712
    root = cgroup_tree(tmp_path, monkeypatch, {
        "": _v1(infinite, usage=50 * GIB),
        "user.slice": _v1(3 * GIB, usage=1 * GIB),
        "user.slice/run.scope": _v1(infinite, usage=1 * GIB),
    }, v1=True)
    budget = _budget()
    assert budget.limiting_level == root + "/user.slice"
    assert budget.limiting_numbers["limit_file"] == "memory.limit_in_bytes"
    assert budget.available_bytes == 2 * GIB


def test_v1_total_inactive_file_is_reclaimable_and_shmem_is_not(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch, {
        "user.slice/run.scope": _v1(8 * GIB, usage=6 * GIB, inactive=4 * GIB),
    }, v1=True, name="cache")
    assert _budget().available_bytes == 6 * GIB
    cgroup_tree(tmp_path, monkeypatch, {
        "user.slice/run.scope": _v1(8 * GIB, usage=6 * GIB, inactive=1 * GIB, shmem=3 * GIB),
    }, v1=True, name="shmem")
    assert _budget().available_bytes == 3 * GIB


def test_a_nondefault_v1_memory_comount_limits_its_nested_group(tmp_path, monkeypatch):
    root = _mounted_cgroup_tree(tmp_path, monkeypatch, {
        "": _v1(16 * GIB),
        "tenant": _v1(8 * GIB),
        "tenant/job": _v1(4 * GIB, GIB),
    }, v1=True)
    budget = _budget()
    assert budget.limiting_level == root + "/tenant/job"
    assert budget.available_bytes == 3 * GIB
    assert budget.budget_bytes == int(3 * GIB * 0.7)


@pytest.mark.parametrize("pin,layer_bytes", [(False, 24 * MIB), (True, 42 * MIB)])
def test_mixed_length_plan_sums_resident_blocks_but_keeps_largest_spill_slots(
    tmp_path, monkeypatch, pin, layer_bytes
):
    from soup_cli.utils.l2l_spill import RAM_RESERVE_BYTES, plan_spill

    sizes = [17 * MIB, 5 * MIB, MIB, MIB]
    options = dict(n_layers=12, k=4, chunk_bytes=max(sizes), mb_bytes=sizes, pin=pin)
    layout = plan_spill(None, **options)
    # Pinned allocations round independently: 32 + 8 + 1 + 1 MiB, not 4 x 32 MiB.
    assert layout.resident_layer_cost == layer_bytes
    assert layout.all_resident_bytes == 12 * layer_bytes
    assert layout.slot_bytes == 17 * MIB
    assert layout.ring_buffer_cost == (32 * MIB if pin else 17 * MIB + 4096)

    target = layout.minimum_bytes + 3 * layer_bytes + 1
    cgroup_tree(tmp_path, monkeypatch, {}, psutil=RAM_RESERVE_BYTES + target)
    budget = _budget()
    assert budget.budget_bytes == target
    plan = plan_spill(budget.budget_bytes, budget_source=budget.describe(), **options)
    assert plan.spill_layers == 9
    assert plan.host_bytes == target - 1
    assert plan.host_bytes + layer_bytes > budget.budget_bytes
    assert plan.spill_file_bytes == 9 * 4 * 17 * MIB
    assert plan.as_dict()["mb_bytes"] == sizes


@pytest.mark.parametrize("sizes", [[100], [100, 101], [100, -1]])
def test_micro_batch_capacities_match_the_window_and_largest_slot(sizes):
    from soup_cli.utils.l2l_spill import plan_spill

    with pytest.raises(ValueError, match="mb_bytes"):
        plan_spill(None, n_layers=2, k=2, chunk_bytes=100, mb_bytes=sizes, pin=False)
