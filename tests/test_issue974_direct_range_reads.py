"""#974 — the disk tier reads a layer as sector-aligned byte ranges, through direct I/O,
into ONE staging region per slot.

Measured on the dev box (RTX 5070, Windows 11, Samsung PM9B1 NVMe, 2026-09-15; the
probes and JSON are in the #974 record): every BUFFERED read primitive tops out at
1.2-2.9 GB/s cold — per-tensor ``readinto`` (the shipped ``AsyncDiskSource`` path), one
``readinto`` per byte range, any open flag — because the cache manager, not the request
shape, is the ceiling; and the async source was already there (``gate-971`` §10:
1.46-2.32 GB/s at the source, 84.7% of a cold 70B step inside the read). Unbuffered
I/O (``FILE_FLAG_NO_BUFFERING`` / ``O_DIRECT``) into a 4 KiB-aligned pinned buffer reads
the same layers at 3.5-5.65 GB/s, best at 2-4 parallel ranges. Warm, with the store in
the page cache, the async arm was 0.91-0.97x the synchronous control position-matched:
the "regression" #974 was filed over was a run-order effect (and one 2.4x-slow block
that did not reproduce), not a cost of the design.

So the reader changes in three ways, each pinned here:

* a layer's whole data section (contiguous in a safetensors file: the tensors are
  dtype-descending with zero gap bytes) is read as K sector-aligned byte ranges by K
  worker threads, each through its own direct-I/O handle, with buffered ``open`` as
  the fallback where direct I/O is unavailable (tmpfs, an unsupported platform);
* every staging slot is ONE contiguous region — tensors are views at their file offset
  minus the aligned start — packed into power-of-two pinned arenas by the #901 packer,
  so the disk tier's staging stops paying the per-tensor rounding the RAM store stopped
  paying in #901 (1.7-1.9x measured);
* ``_plan_queue`` / ``_claim_slot`` / ``get`` are untouched; the byte-identity gate
  against the shipped ``DiskSource`` is the proof that nothing on that path moved.
"""

from __future__ import annotations

import json
import struct
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
SECTOR = 4096


def _cuda() -> bool:
    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not _cuda(), reason="needs a CUDA device")

#: Far beyond any box's RAM, so ``cuMemHostAlloc`` refuses it at once (#901).
_IMPOSSIBLE_PIN_BYTES = 2**40


class _Console:
    def __init__(self):
        self.printed = []

    def print(self, msg):
        self.printed.append(str(msg))


def _shards(
    tmp_path: Path,
    n_layers: int = N_LAYERS,
    *,
    long_header_on: int | None = None,
    extra_tensor_on: int | None = None,
    extra_elements: int = 33,
    empty_tensor: bool = False,
    big: bool = False,
) -> str:
    """NF4-shaped shards (mixed uint8 / float32 / bf16 plus a 0-dim scalar), as in the
    #971 suite, with two optional per-layer variations that a range reader must
    survive and a per-tensor reader never saw:

    * ``long_header_on``: that layer's header carries 5000 bytes of metadata, so its
      data section starts PAST the first 4 KiB sector — the only fixture in which
      the aligned read start is not 0 and the ``entry.start - aligned_start`` term
      of every view offset is actually exercised (a parallel session's mutation
      run found the absolute offset survived every other fixture, including the
      real 70B store, whose data section starts at byte 2976);
    * ``extra_tensor_on``: that layer's shard holds a tensor the spec does not ask
      for, sorted INTO the middle of the wanted ones, so the wanted span is not the
      whole data section and the wanted tensors sit at different offsets.
    * ``extra_elements``: how large that foreign float32 tensor is — small by default,
      large to make the wanted span mostly gap;
    * ``empty_tensor``: every layer also carries a zero-element float32 tensor;
    * ``big``: a 256 KiB packed weight, so a layer spans many sectors and can be
      split across several ranges.
    """
    from soup_cli.utils.layer_shard import layer_shard_path

    out = tmp_path / "shards"
    out.mkdir()
    torch.manual_seed(927)
    rows = 4096 if big else 64
    for idx in range(n_layers):
        blob = {
            "self_attn.q_proj.weight": torch.randint(0, 255, (rows, 64), dtype=torch.uint8),
            "self_attn.q_proj.weight::absmax": torch.rand(16, dtype=torch.float32),
            "self_attn.q_proj.weight::nested_offset": torch.tensor(
                0.125 * (idx + 1), dtype=torch.float32
            ),
            "input_layernorm.weight": torch.rand(64, dtype=torch.bfloat16),
        }
        if extra_tensor_on == idx:
            blob["self_attn.q_proj.weight::absmax_extra"] = torch.rand(
                extra_elements, dtype=torch.float32
            )
        if empty_tensor:
            blob["self_attn.q_proj.weight::empty"] = torch.empty(0, dtype=torch.float32)
        metadata = {"note": "x" * 5000} if long_header_on == idx else None
        save_file(blob, layer_shard_path(str(out), idx), metadata=metadata)
    return str(out)


def _spec(shard_dir: str, n_layers: int = N_LAYERS):
    """Layer 0's spec for every layer — what ``_build_source`` hands a decoder group."""
    return [RamSource.spec_from_shard(shard_dir, 0)] * n_layers


def _raw_bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8)


def _assert_identical_to_disk_source(source, shard_dir, spec, n_layers=N_LAYERS):
    shipped = DiskSource(shard_dir, n_layers, spec)
    try:
        for idx in range(n_layers):
            for name in spec[idx]:
                theirs = shipped.get(idx, name)
                mine = source.get(idx, name)
                assert mine.dtype == theirs.dtype, (idx, name)
                assert mine.shape == theirs.shape, (idx, name)
                assert torch.equal(_raw_bytes(mine), _raw_bytes(theirs)), (idx, name)
            # Pinned staging is a borrow: give the slot back before the next layer.
            source.release(idx, None)
    finally:
        shipped.close()


# ==========================================================================
# The read plan: sector-aligned ranges over a layer's data section
# ==========================================================================
class TestRangePlanning:
    """Pure arithmetic. Direct I/O needs offset, length and address on sector
    boundaries, and the safetensors data section starts wherever the header ends
    (byte 2976 on the 70B store), so the plan covers the aligned superset."""

    def test_aligned_span_floors_the_start_and_ceils_the_end(self):
        from soup_cli.utils.safetensors_reader import aligned_span

        assert aligned_span(2976, 441433020) == (0, 441434112)
        assert aligned_span(4096, 8192) == (4096, 8192)
        assert aligned_span(4097, 8193) == (4096, 12288)

    def test_ranges_are_sector_aligned_contiguous_and_cover_the_span(self):
        from soup_cli.utils.safetensors_reader import plan_ranges

        lo, hi = 4096, 4096 + SECTOR * 10
        ranges = plan_ranges(lo, hi, 4)
        assert len(ranges) == 4
        assert ranges[0][0] == lo and ranges[-1][1] == hi
        for (_a, b), (c, _d) in zip(ranges, ranges[1:]):
            assert b == c, "ranges must be contiguous"
        assert all(a % SECTOR == 0 and b % SECTOR == 0 and b > a for a, b in ranges)

    def test_more_parts_than_sectors_collapses_to_what_the_span_can_carry(self):
        from soup_cli.utils.safetensors_reader import plan_ranges

        assert plan_ranges(0, SECTOR, 4) == [(0, SECTOR)]
        assert len(plan_ranges(0, 2 * SECTOR, 4)) == 2

    def test_a_malformed_plan_is_refused_by_name(self):
        from soup_cli.utils.safetensors_reader import plan_ranges

        with pytest.raises(ValueError, match="parts"):
            plan_ranges(0, SECTOR, 0)
        with pytest.raises(ValueError, match="aligned"):
            plan_ranges(1, SECTOR, 1)
        with pytest.raises(ValueError, match="empty"):
            plan_ranges(SECTOR, SECTOR, 1)


# ==========================================================================
# One range into the front of a view
# ==========================================================================
class TestReadRangeInto:
    @staticmethod
    def _file(tmp_path: Path, size: int):
        data = bytes(i % 251 for i in range(size))
        path = tmp_path / "blob.bin"
        path.write_bytes(data)
        return str(path), data

    def test_fills_exactly_the_requested_bytes_at_the_offset(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_range_into

        path, data = self._file(tmp_path, 3 * SECTOR + 100)
        view = torch.zeros(2 * SECTOR, dtype=torch.uint8)
        with open(path, "rb") as handle:
            read_range_into(handle, SECTOR, view, 2 * SECTOR)
        assert bytes(view.numpy()) == data[SECTOR : 3 * SECTOR]

    def test_a_view_running_past_the_end_of_the_file_gets_the_tail_and_stops(self, tmp_path):
        """The aligned superset's last range runs past EOF; the file's tail arrives
        and the read stops at ``expected`` rather than demanding the padding."""
        from soup_cli.utils.safetensors_reader import read_range_into

        path, data = self._file(tmp_path, 3 * SECTOR + 100)
        view = torch.zeros(2 * SECTOR, dtype=torch.uint8)
        with open(path, "rb") as handle:
            read_range_into(handle, 2 * SECTOR, view, SECTOR + 100)
        assert bytes(view[: SECTOR + 100].numpy()) == data[2 * SECTOR :]

    def test_a_file_shorter_than_expected_is_a_named_short_read(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_range_into

        path, _ = self._file(tmp_path, SECTOR + 10)
        view = torch.zeros(2 * SECTOR, dtype=torch.uint8)
        with open(path, "rb") as handle:
            with pytest.raises(OSError, match="short read"):
                read_range_into(handle, 0, view, 2 * SECTOR)

    def test_a_view_that_cannot_hold_expected_is_refused(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_range_into

        path, _ = self._file(tmp_path, SECTOR)
        with open(path, "rb") as handle:
            with pytest.raises(ValueError, match="holds"):
                read_range_into(handle, 0, torch.zeros(10, dtype=torch.uint8), 100)

    def test_a_zero_return_before_expected_is_a_short_read(self):
        """The one signal that ends the loop, driven directly: a handle that has
        nothing at all must not let the read return as if it had filled the view."""
        from soup_cli.utils.safetensors_reader import read_range_into

        class Empty:
            def seek(self, pos: int) -> None:
                pass

            def readinto(self, buffer) -> int:
                return 0

        with pytest.raises(OSError, match="short read, 0 of 4096"):
            read_range_into(Empty(), 0, torch.zeros(SECTOR, dtype=torch.uint8), SECTOR)

    def test_a_negative_expected_is_refused(self, tmp_path):
        """A negative count would return without reading and leave the view
        undefined while claiming success (security review of #974)."""
        from soup_cli.utils.safetensors_reader import read_range_into

        path, _ = self._file(tmp_path, SECTOR)
        with open(path, "rb") as handle:
            with pytest.raises(ValueError, match="expected"):
                read_range_into(handle, 0, torch.zeros(SECTOR, dtype=torch.uint8), -1)

    def test_a_partial_read_that_is_not_the_end_of_the_file_is_retried(self):
        """``io.FileIO.readinto`` is one syscall, and POSIX lets it return fewer
        bytes than asked with more still to come (Linux documents exactly that for
        O_DIRECT). Only a ZERO return is the end of the data; a short one means
        ask again (python review of #974)."""
        from soup_cli.utils.safetensors_reader import read_range_into

        data = bytes(i % 251 for i in range(3 * SECTOR))

        class Dribbling:
            """Hands back half of every request, never zero until the data is gone."""

            def __init__(self) -> None:
                self.pos = 0
                self.calls = 0

            def seek(self, pos: int) -> None:
                self.pos = pos

            def readinto(self, buffer) -> int:
                self.calls += 1
                chunk = data[self.pos : self.pos + max(1, len(buffer) // 2)]
                buffer[: len(chunk)] = chunk
                self.pos += len(chunk)
                return len(chunk)

        view = torch.zeros(2 * SECTOR, dtype=torch.uint8)
        handle = Dribbling()
        read_range_into(handle, SECTOR, view, 2 * SECTOR)
        assert bytes(view.numpy()) == data[SECTOR : 3 * SECTOR]
        assert handle.calls >= 2, "the short return was accepted without asking again"


# ==========================================================================
# Direct I/O, where the platform has it
# ==========================================================================
class TestOpenDirect:
    @staticmethod
    def _file(tmp_path: Path, size: int):
        data = bytes(i % 253 for i in range(size))
        path = tmp_path / "direct.bin"
        path.write_bytes(data)
        return str(path), data

    @staticmethod
    def _aligned_view(numel: int):
        buffer = torch.empty(numel + SECTOR, dtype=torch.uint8)
        pad = (-buffer.data_ptr()) % SECTOR
        return buffer, buffer[pad : pad + numel]

    def _open_or_skip(self, path: str):
        from soup_cli.utils.safetensors_reader import open_direct

        try:
            return open_direct(path)
        except OSError as exc:
            pytest.skip(f"direct I/O is unavailable on this filesystem/platform: {exc}")

    def test_a_direct_handle_reads_the_same_bytes_as_a_buffered_one(self, tmp_path):
        from soup_cli.utils.safetensors_reader import read_range_into

        path, data = self._file(tmp_path, 5 * SECTOR + 7)
        handle = self._open_or_skip(path)
        keep, view = self._aligned_view(6 * SECTOR)
        with handle:
            read_range_into(handle, 0, view, len(data))
        assert bytes(view[: len(data)].numpy()) == data
        del keep

    def test_a_direct_handle_carries_the_files_identity(self, tmp_path):
        """The shard-replaced-under-the-run check (#971) fstat's the handle it
        reads through; a direct handle must answer the same way."""
        from soup_cli.utils.safetensors_reader import identity_of

        path, _ = self._file(tmp_path, SECTOR)
        handle = self._open_or_skip(path)
        with handle, open(path, "rb") as buffered:
            assert identity_of(handle) == identity_of(buffered)


# ==========================================================================
# Staging: one region per slot, tensors as views
# ==========================================================================
class TestStagingIsOneRegionPerSlot:
    def test_every_staged_tensor_is_a_view_into_its_slots_region(self, tmp_path):
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=False)
        try:
            assert len(source._regions) == len(source._slots) == 2
            for region, slot in zip(source._regions, source._slots):
                lo = region.data_ptr()
                hi = lo + region.numel()
                assert slot, "an empty slot stages nothing"
                for dst in slot.values():
                    start = dst.data_ptr()
                    assert lo <= start < hi, "a staged tensor lives outside its region"
                    assert start + dst.numel() * dst.element_size() <= hi
        finally:
            source.close()

    def test_regions_start_on_a_sector_boundary_even_when_pageable(self, tmp_path):
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=False)
        try:
            assert all(region.data_ptr() % SECTOR == 0 for region in source._regions)
        finally:
            source.close()

    def test_staging_is_accounted_three_ways(self, tmp_path):
        """``nbytes`` stays what it was (the staged tensors' own bytes, the figure
        the pre-flight budgets); ``staging_bytes`` is the regions, which cover the
        aligned superset; ``pinned_bytes`` is what the box page-locked, 0 here."""
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=False)
        try:
            observed = sum(
                dst.numel() * dst.element_size()
                for slot in source._slots
                for dst in slot.values()
            )
            assert source.nbytes == observed
            assert source.staging_bytes == sum(r.numel() for r in source._regions)
            assert source.staging_bytes >= source.nbytes
            assert source.pinned_bytes == 0
        finally:
            source.close()

    def test_read_ranges_is_validated_by_name(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        with pytest.raises(ValueError, match="read_ranges.*not bool"):
            AsyncDiskSource(shard_dir, N_LAYERS, spec, pin=False, read_ranges=True)
        for bad in (0, -1, 17):
            with pytest.raises(ValueError, match="read_ranges"):
                AsyncDiskSource(shard_dir, N_LAYERS, spec, pin=False, read_ranges=bad)

    def test_the_default_depth_of_ranges_is_the_measured_one(self, tmp_path):
        from soup_cli.utils.async_disk_source import DEFAULT_STREAM_READ_RANGES

        assert DEFAULT_STREAM_READ_RANGES == 4
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), pin=False)
        try:
            assert source.read_ranges == DEFAULT_STREAM_READ_RANGES
            assert isinstance(source.direct_io, bool)
        finally:
            source.close()


# ==========================================================================
# THE gate: the same bytes as DiskSource through the range reader
# ==========================================================================
class TestByteIdentityThroughTheRangeReader:
    @pytest.mark.parametrize("read_ranges", [1, 3, 4])
    def test_every_tensor_matches_disk_source(self, tmp_path, read_ranges):
        shard_dir = _shards(tmp_path, big=True)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(
            shard_dir, N_LAYERS, spec, read_ahead=2, pin=False, read_ranges=read_ranges
        )
        try:
            _assert_identical_to_disk_source(source, shard_dir, spec)
        finally:
            source.close()

    def test_identity_holds_without_direct_io(self, tmp_path, monkeypatch):
        """tmpfs, a network share, a platform without the flag: the buffered
        fallback must read the same bytes, and the source must SAY it fell back."""
        import soup_cli.utils.async_disk_source as mod

        def unavailable(path):
            raise OSError("no direct I/O here")

        monkeypatch.setattr(mod, "open_direct", unavailable)
        shard_dir = _shards(tmp_path, big=True)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            assert source.direct_io is False
            _assert_identical_to_disk_source(source, shard_dir, spec)
        finally:
            source.close()

    def test_a_layer_whose_header_is_longer_than_its_siblings_is_placed_correctly(
        self, tmp_path
    ):
        """Views are per LAYER, not per slot: the sector pad differs when the
        header length does, and a slot reused for such a layer must re-derive
        where each tensor landed."""
        shard_dir = _shards(tmp_path, long_header_on=2)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            starts = [plan.start for plan in source._plans]
            assert starts[2] >= SECTOR and all(s == 0 for i, s in enumerate(starts) if i != 2), (
                f"the fixture no longer moves layer 2's data section past the first "
                f"sector (aligned starts {starts}); the offset term is not exercised"
            )
            _assert_identical_to_disk_source(source, shard_dir, spec)
        finally:
            source.close()

    def test_a_shard_carrying_a_tensor_the_spec_does_not_want_is_still_read_correctly(
        self, tmp_path
    ):
        shard_dir = _shards(tmp_path, extra_tensor_on=1)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            _assert_identical_to_disk_source(source, shard_dir, spec)
            with pytest.raises(KeyError):
                source.get(1, "self_attn.q_proj.weight::absmax_extra")
        finally:
            source.close()

    def test_a_shard_spreading_the_wanted_tensors_over_a_far_larger_span_is_refused(
        self, tmp_path
    ):
        """Staging is sized to the span between the first and last wanted tensor,
        so a foreign tensor BETWEEN them is staged with them. A small one is the
        cost of the design; a large one is a page-locked allocation the spec never
        asked for (the security review of #974 built a shard whose two 16-byte
        tensors cost an 8 MiB region). Refused at construction, by name."""
        shard_dir = _shards(tmp_path, extra_tensor_on=1, extra_elements=2_000_000)
        with pytest.raises(ValueError, match="span"):
            AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=False)

    def test_a_layer_that_wants_no_tensors_is_refused_by_name(self, tmp_path):
        shard_dir = _shards(tmp_path)
        spec = [dict(layer) for layer in _spec(shard_dir)]
        spec[1] = {}
        with pytest.raises(ValueError, match="no tensors"):
            AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)

    def test_a_layer_whose_wanted_tensors_hold_no_bytes_is_refused_by_name(self, tmp_path):
        """Otherwise the empty span is refused by ``plan_ranges`` at the first
        read, i.e. at the ``get`` that wanted it, instead of at construction."""
        shard_dir = _shards(tmp_path, empty_tensor=True)
        spec = [{"self_attn.q_proj.weight::empty": ((0,), "float32")}] * N_LAYERS
        with pytest.raises(ValueError, match="no bytes"):
            AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)

    def test_a_zero_element_tensor_rides_along_intact(self, tmp_path):
        """An empty tensor has no bytes to read and no alignment to check; it must
        still come back as an empty tensor of the right dtype from every layer."""
        shard_dir = _shards(tmp_path, empty_tensor=True)
        spec = _spec(shard_dir)
        assert spec[0]["self_attn.q_proj.weight::empty"] == ((0,), "float32")
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=False)
        try:
            _assert_identical_to_disk_source(source, shard_dir, spec)
            empty = source.get(0, "self_attn.q_proj.weight::empty")
            assert empty.numel() == 0 and empty.dtype == torch.float32
        finally:
            source.close()

    def test_a_tensor_offset_not_aligned_to_its_dtype_is_refused_at_construction(
        self, tmp_path
    ):
        """A view needs its byte offset on the dtype's boundary. safetensors' own
        writer orders tensors so that holds; a hand-built shard that breaks it is
        refused by name rather than read into a view torch cannot make."""
        from soup_cli.utils.layer_shard import layer_shard_path

        out = tmp_path / "misaligned"
        out.mkdir()
        header = {
            "a": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [3, 11]},
        }
        body = json.dumps(header).encode()
        # The refusal is about 'b' starting off a 4-byte boundary from the aligned
        # read start (0): if the header length ever makes 8 + len + 3 a multiple of
        # four, this fixture stops testing anything — say so rather than pass.
        assert (8 + len(body) + 3) % 4 != 0, "fixture no longer misaligns 'b'"
        for idx in range(2):
            with open(layer_shard_path(str(out), idx), "wb") as handle:
                handle.write(struct.pack("<Q", len(body)) + body + bytes(11))
        spec = [{"a": ((3,), "uint8"), "b": ((2,), "float32")}] * 2
        with pytest.raises(ValueError, match="aligned"):
            AsyncDiskSource(str(out), 2, spec, read_ahead=1, pin=False)


# ==========================================================================
# The ranges are read by workers, in parallel, off the calling thread
# ==========================================================================
class TestTheReadIsSplitAcrossWorkers:
    def test_ranges_of_one_layer_are_read_concurrently(self, tmp_path, monkeypatch):
        import soup_cli.utils.async_disk_source as mod

        real = mod.read_range_into
        lock = threading.Lock()
        live = {"now": 0, "peak": 0}

        def counting(handle, start, view, expected):
            with lock:
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
            try:
                time.sleep(0.02)
                return real(handle, start, view, expected)
            finally:
                with lock:
                    live["now"] -= 1

        monkeypatch.setattr(mod, "read_range_into", counting)
        shard_dir = _shards(tmp_path, big=True)
        source = AsyncDiskSource(
            shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=1, pin=False, read_ranges=4
        )
        try:
            source.get(0, "input_layernorm.weight")
            assert live["peak"] >= 2, "the ranges of one layer were read one after another"
        finally:
            source.close()

    def test_the_reads_never_run_on_the_calling_thread(self, tmp_path, monkeypatch):
        import soup_cli.utils.async_disk_source as mod

        real = mod.read_range_into
        seen = []

        def recording(handle, start, view, expected):
            seen.append(threading.current_thread())
            return real(handle, start, view, expected)

        monkeypatch.setattr(mod, "read_range_into", recording)
        shard_dir = _shards(tmp_path, big=True)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=False)
        try:
            source.get(0, "input_layernorm.weight")
            assert seen, "nothing was read at all"
            assert threading.current_thread() not in seen
            assert {t.name.rsplit("-", 1)[0] for t in seen} == {"soup-layer-range"}
            assert all(t.daemon for t in seen), "a wedged range read must not block exit"
        finally:
            source.close()

    def test_the_workers_are_gone_after_close(self, tmp_path):
        """THIS source's workers, not every ``soup-layer-range`` thread in the
        process: in a full test session other sources' workers are still
        winding down, and a process-wide sweep blamed them on this one."""
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=False)
        source.get(0, "input_layernorm.weight")
        workers = list(source._readers._threads)
        assert workers and all(t.is_alive() for t in workers), "no range workers were started"
        assert all(t.name.startswith("soup-layer-range") for t in workers)
        source.close()
        # No join here: ``close()`` returning is the promise that the workers are
        # gone, the same promise it makes for the reader thread.
        assert not any(t.is_alive() for t in workers)

    def test_close_releases_the_staging_buffers(self, tmp_path):
        """The regions and their arenas are gigabytes of page-locked memory on a
        real run; a ``close()`` that dropped the slots but kept them referenced
        would hold that for the process's lifetime with nothing to say so (a
        parallel session's mutation run: no test looked)."""
        shard_dir = _shards(tmp_path)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=False)
        source.get(0, "input_layernorm.weight")
        assert source._regions and source._arenas
        source.close()
        assert source._slots == [] and source._regions == [] and source._arenas == []

    def test_the_reader_reads_each_layer_through_read_layer(self, tmp_path, monkeypatch):
        """The one seam the harness times (``reader_read`` in
        ``benchmarks/harness/issue974_warm_stages.py``): a layer read is exactly one
        ``_read_layer`` call, whatever the ranges do inside it."""
        calls = []
        real = AsyncDiskSource._read_layer

        def recording(self, idx, region):
            calls.append(idx)
            return real(self, idx, region)

        monkeypatch.setattr(AsyncDiskSource, "_read_layer", recording)
        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=1, pin=False)
        try:
            for idx in range(N_LAYERS):
                source.get(idx, "input_layernorm.weight")
            assert calls == list(range(N_LAYERS))
        finally:
            source.close()

    def test_an_error_in_one_range_surfaces_at_the_get_that_wanted_it(
        self, tmp_path, monkeypatch
    ):
        import soup_cli.utils.async_disk_source as mod

        real = mod.read_range_into
        count = {"n": 0}

        def failing(handle, start, view, expected):
            count["n"] += 1
            if count["n"] == 2:
                raise OSError("range two went away")
            return real(handle, start, view, expected)

        monkeypatch.setattr(mod, "read_range_into", failing)
        shard_dir = _shards(tmp_path, big=True)
        source = AsyncDiskSource(
            shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=1, pin=False, read_ranges=4
        )
        try:
            with pytest.raises(OSError, match="range two went away"):
                source.get(0, "input_layernorm.weight")
        finally:
            source.close()


# ==========================================================================
# The ready line says what the disk tier page-locked
# ==========================================================================
class TestTheReadyLineNamesThePageLockedStaging:
    @staticmethod
    def _stats(pinned: bool, pinned_bytes):
        return {
            "tier": "disk",
            "pinned": pinned,
            "pinned_bytes": pinned_bytes,
            "disk_bytes": 36.39e9,
            "read_ahead": 2,
            "store_bytes": 1.96e9,
        }

    def test_pinned_staging_prints_the_page_locked_figure(self):
        from soup_cli.trainer.stream_setup import _stream_source_line

        line = _stream_source_line(self._stats(True, 2**31))
        assert "1960 MB pinned host staging" in line
        assert "(2.15 GB page-locked)" in line

    def test_pageable_staging_prints_no_second_figure(self):
        from soup_cli.trainer.stream_setup import _stream_source_line

        line = _stream_source_line(self._stats(False, 0))
        assert "pageable host staging" in line
        assert "page-locked" not in line


# ==========================================================================
# The arena plan on the real 70B staging
# ==========================================================================
class TestTheArenaPlanForARealStore:
    def test_the_70b_staging_page_locks_within_ten_percent_of_its_bytes(self):
        """Two decoder slots (441.4 MB aligned spans) plus the embed and the untied
        head (536.9 MB each) on the 70B-shaped NF4 store: one 2 GiB arena for
        1.96 GB, where per-tensor pinning paid 1.7-1.9x."""
        from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

        spans = [441434112, 441434112, 536875008, 536875008]
        plan = plan_pinned_arenas(spans, align=SECTOR)
        assert plan.pinned_bytes / sum(spans) <= 1.10
        assert all(offset % SECTOR == 0 for _arena, offset in plan.placements)


# ==========================================================================
# Real hardware: pinned arenas, and the disk tier's own failed page-lock
# ==========================================================================
@requires_cuda
class TestOnRealHardware:
    def test_every_region_is_a_pinned_sector_aligned_view_of_a_power_of_two_arena(
        self, tmp_path
    ):
        shard_dir = _shards(tmp_path, big=True)
        source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=True)
        try:
            assert source.pinned is True
            assert source._arenas, "pinned staging must live in arenas"
            for arena in source._arenas:
                assert arena.is_pinned()
                assert arena.numel() & (arena.numel() - 1) == 0, "arena is not a power of two"
            assert source.pinned_bytes == sum(a.numel() for a in source._arenas)
            for region in source._regions:
                assert region.is_pinned()
                assert region.data_ptr() % SECTOR == 0
            staged = [dst for slot in source._slots for dst in slot.values()]
            assert all(dst.is_pinned() for dst in staged)
        finally:
            source.close()

    def test_byte_identity_holds_with_pinned_staging(self, tmp_path):
        shard_dir = _shards(tmp_path, big=True)
        spec = _spec(shard_dir)
        source = AsyncDiskSource(shard_dir, N_LAYERS, spec, read_ahead=2, pin=True, read_ranges=4)
        try:
            _assert_identical_to_disk_source(source, shard_dir, spec)
        finally:
            source.close()

    def test_the_disk_tier_falls_back_to_pageable_staging_after_a_real_failed_page_lock(
        self, tmp_path, monkeypatch
    ):
        """The #901 round trip, on the DISK tier: a genuine driver refusal of the
        first arena, the genuine stale error, the shipped fallback — and the first
        kernel launch after ``_build_source`` returns must work. The RAM tier had
        this test; the disk tier did not (TDD review of #989)."""
        import soup_cli.utils.layer_stream_runtime as rt

        shard_dir = _shards(tmp_path)
        spec = _spec(shard_dir)
        real_empty = torch.empty

        def _impossible_when_pinned(*args, **kwargs):
            if kwargs.get("pin_memory"):
                return real_empty(_IMPOSSIBLE_PIN_BYTES, dtype=torch.uint8, pin_memory=True)
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", _impossible_when_pinned)
        console = _Console()
        source, pinned = rt._build_source(
            shard_dir, N_LAYERS, spec, True, console, "disk", read_ahead=2
        )
        monkeypatch.undo()
        try:
            assert pinned is False
            assert source.pinned is False
            assert any("PAGEABLE" in msg for msg in console.printed)
            # The first kernel launch of the run, where #901 died.
            torch.ones(1, device="cuda")
            torch.cuda.synchronize()
            _assert_identical_to_disk_source(source, shard_dir, spec)
        finally:
            source.close()
