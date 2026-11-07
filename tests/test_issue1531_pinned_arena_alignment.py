"""#1531 — a pinned staging arena whose base is off a 4096-byte boundary no longer turns
direct I/O off for the whole store.

``plan_pinned_arenas`` keeps every placement on a sector boundary RELATIVE to its
arena, so the regions are aligned only when the arena's base is. torch's caching host
allocator does not promise that: on torch 2.14.0+cu130 a probe of 400 pinned
allocations with frees in between got 159 bases at a multiple of 512 but not of 4096.
The source now aligns inside the arena when its slack allows, and otherwise takes the
next power of two up, which always has room for the shift.

No GPU here: the arena allocation is replaced with pageable CPU memory at a forced
512-byte offset, the shape the probe saw.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

import soup_cli.utils.async_disk_source as mod  # noqa: E402
from soup_cli.utils.async_disk_source import AsyncDiskSource  # noqa: E402
from tests.test_issue974_direct_range_reads import N_LAYERS, _shards, _spec  # noqa: E402

SECTOR = 4096
OFF_BY = 512


def _misaligned_arena(size: int, off_by: int = OFF_BY):
    """``size`` bytes whose first byte sits ``off_by`` past a sector boundary."""
    buffer = torch.empty(size + 2 * SECTOR, dtype=torch.uint8)
    pad = (off_by - buffer.data_ptr()) % SECTOR
    return buffer[pad : pad + size]


@pytest.fixture
def misaligned(monkeypatch):
    sizes = []

    def allocate(size: int):
        sizes.append(size)
        return _misaligned_arena(size)

    monkeypatch.setattr(mod, "_allocate_pinned_arena", allocate)
    return sizes


def _bare_pinned_source() -> AsyncDiskSource:
    source = AsyncDiskSource.__new__(AsyncDiskSource)
    source.pinned = True
    return source


class TestAllocateStaging:
    def test_the_fake_arena_is_off_by_512(self):
        assert _misaligned_arena(16 * SECTOR).data_ptr() % SECTOR == OFF_BY

    def test_a_misaligned_base_with_slack_is_aligned_inside_the_same_arena(self, misaligned):
        # Three sectors pack into a 16 KiB arena: one sector of slack covers the shift.
        arenas, regions = _bare_pinned_source()._allocate_staging([3 * SECTOR])
        assert misaligned == [4 * SECTOR]
        assert [a.numel() for a in arenas] == [4 * SECTOR]
        assert all(r.data_ptr() % SECTOR == 0 for r in regions)
        assert [r.numel() for r in regions] == [3 * SECTOR]

    def test_a_full_misaligned_arena_is_replaced_by_the_next_size_up(self, misaligned):
        # Two regions fill a 32 KiB arena exactly: no room to shift inside it.
        source = _bare_pinned_source()
        arenas, regions = source._allocate_staging([4 * SECTOR, 4 * SECTOR])
        assert misaligned == [8 * SECTOR, 16 * SECTOR]
        assert [a.numel() for a in arenas] == [16 * SECTOR]
        assert all(r.data_ptr() % SECTOR == 0 for r in regions)
        assert [r.numel() for r in regions] == [4 * SECTOR, 4 * SECTOR]
        # The page-locked total counts the arena actually held, not the plan's.
        assert source.pinned_bytes == 16 * SECTOR

    def test_the_shift_is_measured_again_on_the_bigger_arena(self, monkeypatch):
        # The replacement arena comes back at a different offset from the first.
        offsets = iter([OFF_BY, 2 * OFF_BY])
        monkeypatch.setattr(
            mod, "_allocate_pinned_arena", lambda size: _misaligned_arena(size, next(offsets))
        )
        arenas, regions = _bare_pinned_source()._allocate_staging([4 * SECTOR, 4 * SECTOR])
        assert arenas[0].data_ptr() % SECTOR == 2 * OFF_BY
        assert all(r.data_ptr() % SECTOR == 0 for r in regions)

    def test_the_regions_of_one_arena_do_not_overlap_after_the_shift(self, misaligned):
        arenas, regions = _bare_pinned_source()._allocate_staging([SECTOR, 2 * SECTOR, SECTOR])
        assert len(arenas) == 1
        spans = sorted((r.data_ptr(), r.data_ptr() + r.numel()) for r in regions)
        assert all(end <= start for (_, end), (start, _) in zip(spans, spans[1:]))
        lo, hi = arenas[0].data_ptr(), arenas[0].data_ptr() + arenas[0].numel()
        assert all(lo <= start and end <= hi for start, end in spans)

    def test_an_aligned_base_is_used_as_planned(self, monkeypatch):
        sizes = []

        def aligned(size: int):
            sizes.append(size)
            buffer = torch.empty(size + SECTOR, dtype=torch.uint8)
            pad = (-buffer.data_ptr()) % SECTOR
            return buffer[pad : pad + size]

        monkeypatch.setattr(mod, "_allocate_pinned_arena", aligned)
        source = _bare_pinned_source()
        arenas, regions = source._allocate_staging([4 * SECTOR, 4 * SECTOR])
        assert sizes == [8 * SECTOR]
        assert regions[0].data_ptr() == arenas[0].data_ptr()
        assert source.pinned_bytes == 8 * SECTOR


class TestTheSource:
    def test_a_misaligned_arena_keeps_the_regions_aligned_and_says_nothing(
        self, tmp_path, misaligned, caplog
    ):
        shard_dir = _shards(tmp_path)
        with caplog.at_level(logging.WARNING, logger=mod.__name__):
            source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=True)
        try:
            assert misaligned, "the pinned arenas did not go through the patched allocation"
            assert all(r.data_ptr() % SECTOR == 0 for r in source._regions)
            assert not [r for r in caplog.records if "not sector-aligned" in r.getMessage()]
            assert source.pinned_bytes == sum(a.numel() for a in source._arenas)
        finally:
            source.close()

    def test_misaligned_regions_still_fall_back_to_the_page_cache(
        self, tmp_path, monkeypatch, caplog
    ):
        """The fallback stays for staging that is misaligned anyway: reading it with
        direct I/O would fail, so the whole source reads through the page cache."""
        real = AsyncDiskSource._allocate_staging

        def shifted(self, region_sizes):
            arenas, regions = real(self, region_sizes)
            return arenas, [_misaligned_arena(r.numel()) for r in regions]

        monkeypatch.setattr(AsyncDiskSource, "_allocate_staging", shifted)
        opened = []
        monkeypatch.setattr(mod, "open_direct", lambda path: opened.append(path))
        shard_dir = _shards(tmp_path)
        with caplog.at_level(logging.WARNING, logger=mod.__name__):
            source = AsyncDiskSource(shard_dir, N_LAYERS, _spec(shard_dir), read_ahead=2, pin=False)
        try:
            assert source.direct_io is False
            assert opened == [], "direct I/O was probed for misaligned staging"
            assert [r for r in caplog.records if "not sector-aligned" in r.getMessage()]
        finally:
            source.close()


class _Console:
    def __init__(self):
        self.lines = []

    def print(self, text):
        self.lines.append(text)


@pytest.fixture
def unalignable(monkeypatch):
    """Staging that is misaligned whatever the arenas were: the fallback's one case."""
    real = AsyncDiskSource._allocate_staging

    def shifted(self, region_sizes):
        arenas, regions = real(self, region_sizes)
        return arenas, [_misaligned_arena(r.numel()) for r in regions]

    monkeypatch.setattr(AsyncDiskSource, "_allocate_staging", shifted)


class TestTheConsoleLine:
    """A `soup train` user sees the Rich console, not the logger."""

    def _build(self, tmp_path, console, *, pin):
        from soup_cli.utils.layer_stream_runtime import _build_source

        shard_dir = _shards(tmp_path)
        return _build_source(
            shard_dir, N_LAYERS, _spec(shard_dir), pin, console, tier="disk", read_ahead=2
        )

    @staticmethod
    def _fallback_lines(console):
        return [line for line in console.lines if "page cache" in line]

    def test_a_page_cache_fallback_is_said_on_the_console(self, tmp_path, unalignable):
        console = _Console()
        source, pinned = self._build(tmp_path, console, pin=False)
        try:
            assert source.direct_io is False and pinned is False
            lines = self._fallback_lines(console)
            assert len(lines) == 1
            assert "not sector-aligned" in lines[0] and "direct I/O" in lines[0]
        finally:
            source.close()

    def test_a_pinned_source_says_it_too(self, tmp_path, misaligned, unalignable):
        console = _Console()
        source, pinned = self._build(tmp_path, console, pin=True)
        try:
            assert source.direct_io is False and pinned is True
            assert len(self._fallback_lines(console)) == 1
        finally:
            source.close()

    def test_aligned_staging_says_nothing(self, tmp_path):
        console = _Console()
        source, _ = self._build(tmp_path, console, pin=False)
        try:
            assert source.staging_aligned is True
            assert self._fallback_lines(console) == []
        finally:
            source.close()

    def test_without_a_console_the_logger_says_it_once(self, tmp_path, unalignable, caplog):
        with caplog.at_level(logging.WARNING, logger=mod.__name__):
            source, _ = self._build(tmp_path, None, pin=False)
        try:
            assert len([r for r in caplog.records if "not sector-aligned" in r.getMessage()]) == 1
        finally:
            source.close()
