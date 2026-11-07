"""#1702 — the pinned store's arenas are packed largest-first when that page-locks less.

``plan_pinned_arenas`` walked the tensors in allocation order and only ever looked at
the arena it opened last. A Qwen3-8B NF4 store is 3.58 GB of decoder tensors followed
by two 1.24 GB vocabulary matrices: neither fits behind the decoder, so each opened a
2 GiB arena of its own and 8 GiB was page-locked for a 6.07 GB store. Measured
2026-09-28 on an RTX 5070 Laptop (torch 2.14.0+cu130, ``main`` at d7a4f7a7 plus this
change): the RAM tier's arenas went from 8,589,934,592 to 6,442,450,944 bytes, and
three deterministic optimizer steps were byte-identical before and after.

The planner now also plans stable first-fit decreasing and returns it ONLY when it
page-locks strictly fewer bytes. A tie keeps the in-order plan, placement for
placement, so a store that gains nothing is laid out exactly as before.

Everything but the last class is arithmetic, or pageable CPU memory, and needs no GPU.
"""

from __future__ import annotations

import functools
import hashlib
import random
from itertools import pairwise

import pytest

import soup_cli.utils.layer_stream_runtime as runtime
from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

GiB = 2**30
SECTOR = 4096


def _nf4_sizes(parameters: int) -> list:
    """One NF4 double-quant linear: packed nibbles, absmax, nested absmax, nested offset."""
    return [parameters // 2, parameters // 64, parameters // 16384 * 4, 4]


#: One Qwen3-8B decoder layer in the runtime's module order: q / k / v / o, the q and k
#: norms, gate / up / down, the two layernorms. 32 tensors, 99.5 MB.
_QWEN3_8B_LAYER_BYTES = (
    _nf4_sizes(4096 * 4096)
    + _nf4_sizes(1024 * 4096) * 2
    + _nf4_sizes(4096 * 4096)
    + [128 * 2] * 2
    + _nf4_sizes(12288 * 4096) * 3
    + [4096 * 2] * 2
)
#: The whole RAM-tier store: 36 layers, then the embedding and the untied head in bf16.
_QWEN3_8B_STORE_BYTES = _QWEN3_8B_LAYER_BYTES * 36 + [151936 * 4096 * 2] * 2

#: Qwen3-8B on the disk tier: two decoder staging slots, then the two vocabulary regions.
_QWEN3_8B_DISK_REGIONS = [99_553_280] * 2 + [1_244_663_808] * 2


def _in_order(sizes, **kwargs):
    """The plan before #1702, through the public function: the candidate cannot win."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "_plan_largest_first", runtime._plan_in_order)
        return plan_pinned_arenas(sizes, **kwargs)


def _assert_arena_layout(plan, sizes, *, align: int) -> None:
    assert plan.requested_bytes == sum(sizes)
    assert len(plan.placements) == len(sizes)
    assert all(size > 0 and size & (size - 1) == 0 for size in plan.arena_sizes)
    spans = []
    for (arena, offset), size in zip(plan.placements, sizes):
        assert 0 <= arena < len(plan.arena_sizes)
        assert offset >= 0 and offset % align == 0, (offset, align)
        assert offset + size <= plan.arena_sizes[arena]
        if size:
            spans.append((arena, offset, offset + size))
    spans.sort()
    for (left_arena, _lo, left_end), (right_arena, right_start, _hi) in pairwise(spans):
        if left_arena == right_arena:
            assert left_end <= right_start


@pytest.fixture
def small_arenas(monkeypatch):
    """A 1 KiB ceiling, so a hand-sized list of tensors spans several arenas."""
    monkeypatch.setattr(runtime, "PINNED_ARENA_MAX_BYTES", 1024)
    return {"arena_bytes": 256, "align": 16}


class TestTheQwen3Store:
    """The geometry the saving was measured on, as arithmetic."""

    def test_the_geometry_is_the_measured_store(self):
        assert len(_QWEN3_8B_LAYER_BYTES) == 32
        assert len(_QWEN3_8B_STORE_BYTES) == 1154
        assert sum(_QWEN3_8B_STORE_BYTES) == 6_073_035_760

    def test_in_allocation_order_it_page_locked_four_arenas(self):
        plan = _in_order(_QWEN3_8B_STORE_BYTES)
        assert plan.arena_sizes == (2**31,) * 4
        assert plan.pinned_bytes == 8 * GiB
        # Each vocabulary matrix alone in an arena, behind a decoder it could not follow.
        assert plan.placements[-2:] == ((2, 0), (3, 0))

    def test_late_vocabulary_weights_reuse_the_decoder_arenas_slack(self):
        plan = plan_pinned_arenas(_QWEN3_8B_STORE_BYTES)
        assert plan.arena_sizes == (2**31,) * 3
        assert plan.pinned_bytes == 6 * GiB
        assert plan.pinned_bytes == 6_442_450_944
        assert plan.placements[-2:] == ((0, 0), (1, 0))
        _assert_arena_layout(plan, _QWEN3_8B_STORE_BYTES, align=runtime.PINNED_ARENA_ALIGN)

    def test_the_14b_stores_of_issue_901_pack_tighter_too(self):
        from tests.test_issue901_pinned_arenas import _QWEN14B_NF4_LAYER_BYTES

        nf4 = _QWEN14B_NF4_LAYER_BYTES * 48
        assert _in_order(nf4).pinned_bytes == 7_381_975_040
        assert plan_pinned_arenas(nf4).pinned_bytes == 6_845_104_128
        bf16 = ([141_557_760] * 3 + [52_428_800] * 2 + [10_485_760] * 2 + [10_240] * 2) * 48
        assert _in_order(bf16).pinned_bytes == 28_991_029_248
        packed = plan_pinned_arenas(bf16)
        assert packed.pinned_bytes == 26_843_545_600
        # The request size did not grow: still the 1 GiB the bf16 store already asked for.
        assert set(packed.arena_sizes) == {2**30}
        _assert_arena_layout(packed, bf16, align=runtime.PINNED_ARENA_ALIGN)


class TestATieKeepsTheInOrderPlan:
    def test_equal_cost_disk_staging_retains_original_placements(self):
        """Either order costs two 2 GiB arenas, so nothing about the disk tier moves."""
        sizes = _QWEN3_8B_DISK_REGIONS
        plan = plan_pinned_arenas(sizes, align=SECTOR)
        assert plan.arena_sizes == (2**31, 2**31)
        assert plan.placements == ((0, 0), (0, sizes[0]), (0, sizes[0] * 2), (1, 0))
        assert plan == _in_order(sizes, align=SECTOR)

    def test_the_70b_disk_staging_is_unchanged(self):
        sizes = [441434112, 441434112, 536875008, 536875008]
        assert plan_pinned_arenas(sizes, align=SECTOR) == _in_order(sizes, align=SECTOR)

    def test_the_same_arena_count_is_not_a_saving_when_the_trim_rounds_up(self, small_arenas):
        """Largest-first needs four arenas here as well, but its last one holds 334 bytes
        and trims to 512 where the in-order one holds 249 and trims to 256."""
        sizes = [503, 94, 121, 250, 345, 211, 444, 170, 138, 578, 142, 105]
        candidate = runtime._plan_largest_first(sizes, 1024, 16)
        assert candidate.arena_sizes == (1024, 1024, 1024, 512)
        plan = plan_pinned_arenas(sizes, **small_arenas)
        assert plan.arena_sizes == (1024, 1024, 1024, 256)
        assert plan.placements == (
            *((0, 0), (0, 512), (0, 608), (0, 736)),
            *((1, 0), (1, 352), (1, 576)),
            *((2, 0), (2, 176), (2, 320)),
            *((3, 0), (3, 144)),
        )

    def test_a_single_arena_store_is_left_alone(self):
        sizes = [3_000_000, 700_000, 5, 12_345_678, 1, 0, 999_999]
        assert plan_pinned_arenas(sizes) == _in_order(sizes)


class TestThePackedLayout:
    def test_placements_stay_indexed_by_allocation_order(self, small_arenas):
        """Bytes written through the plan come back through the plan: the tensor at
        index 6 is the first one placed, and is still the one read at index 6."""
        sizes = [256] * 6 + [600, 600, 4, 0]
        assert _in_order(sizes, **small_arenas).pinned_bytes == 3584
        plan = plan_pinned_arenas(sizes, **small_arenas)
        assert plan.pinned_bytes == 3072
        assert plan.placements[6] == (0, 0)
        assert plan.placements[7] == (1, 0)
        _assert_arena_layout(plan, sizes, align=16)
        arenas = [bytearray(size) for size in plan.arena_sizes]
        payloads = [bytes([index + 1]) * size for index, size in enumerate(sizes)]
        for payload, (arena, offset) in zip(payloads, plan.placements):
            arenas[arena][offset : offset + len(payload)] = payload
        for payload, (arena, offset) in zip(payloads, plan.placements):
            assert bytes(arenas[arena][offset : offset + len(payload)]) == payload

    def test_equal_sizes_keep_their_allocation_order(self, small_arenas):
        sizes = [256] * 6 + [600, 600, 4, 0]
        plan = plan_pinned_arenas(sizes, **small_arenas)
        equal = list(plan.placements[:6])
        assert equal == sorted(equal)
        assert plan == plan_pinned_arenas(list(sizes), **small_arenas)

    @pytest.mark.parametrize("align", [1, 256, 4096])
    def test_zero_byte_and_oversized_tensors_keep_their_contracts(self, align):
        """A tensor past the 2 GiB ceiling still gets an arena that fits it, and the
        small ones now fill that arena's tail instead of opening one of their own."""
        sizes = [4, 0, 2**31 + 1, 0, 2**31 + 33, 1, 0]
        assert len(_in_order(sizes, align=align).arena_sizes) == 3
        plan = plan_pinned_arenas(sizes, align=align)
        assert plan.arena_sizes == (2**32, 2**32)
        _assert_arena_layout(plan, sizes, align=align)

    def test_every_plan_is_valid_and_never_costs_more_than_the_in_order_one(self, small_arenas):
        rng = random.Random(1702)

        def draw() -> int:
            # Half the tensors tiny or empty, a third ordinary, a sixth past the ceiling.
            kind = rng.randrange(6)
            if kind < 3:
                return (0, 1, 4)[kind]
            return rng.randint(1, 700) if kind < 5 else rng.randint(900, 2600)

        packed = kept = 0
        for _case in range(400):
            sizes = [draw() for _ in range(rng.randint(1, 40))]
            align = rng.choice((1, 16, 64))
            plan = plan_pinned_arenas(sizes, arena_bytes=256, align=align)
            before = _in_order(sizes, arena_bytes=256, align=align)
            _assert_arena_layout(plan, sizes, align=align)
            assert plan.pinned_bytes <= before.pinned_bytes, sizes
            if plan.pinned_bytes == before.pinned_bytes:
                assert plan == before, sizes
                kept += 1
            else:
                packed += 1
        # Both branches are exercised, or the loop above proves nothing about one of them.
        assert packed >= 100 and kept >= 20, (packed, kept)


# ==========================================================================
# The disk tier reads the same plan (CPU: pageable memory stands in for the arenas)
# ==========================================================================
def _aligned_arena(size: int, off_by: int = 0):
    """``size`` bytes whose first byte sits ``off_by`` past a sector boundary."""
    torch = pytest.importorskip("torch")
    buffer = torch.empty(size + 2 * SECTOR, dtype=torch.uint8)
    pad = (off_by - buffer.data_ptr()) % SECTOR
    return buffer[pad : pad + size]


@pytest.fixture
def sector_arenas(monkeypatch):
    """Staging planned at a 4-sector ceiling, so eight small regions span three arenas."""
    monkeypatch.setattr(runtime, "PINNED_ARENA_MAX_BYTES", 4 * SECTOR)
    monkeypatch.setattr(
        runtime, "plan_pinned_arenas", functools.partial(plan_pinned_arenas, arena_bytes=4 * SECTOR)
    )


def _stage(monkeypatch, region_sizes, *, off_by: int):
    import soup_cli.utils.async_disk_source as disk

    allocated = []

    def allocate(size: int):
        allocated.append(size)
        return _aligned_arena(size, off_by)

    monkeypatch.setattr(disk, "_allocate_pinned_arena", allocate)
    source = disk.AsyncDiskSource.__new__(disk.AsyncDiskSource)
    source.pinned = True
    arenas, regions = source._allocate_staging(region_sizes)
    return source, arenas, regions, allocated


@pytest.mark.usefixtures("sector_arenas")
class TestTheDiskTierStagesAPackedPlan:
    #: Six one-sector regions, then two of three sectors. In order: 4 + 2 + 4 + 4
    #: sectors of arena. Largest-first: three arenas of 4, every one exactly full.
    REGIONS = [SECTOR] * 6 + [3 * SECTOR, 3 * SECTOR]

    def test_the_fixture_selects_the_packed_plan(self):
        before = _in_order(self.REGIONS, arena_bytes=4 * SECTOR, align=SECTOR)
        assert [size // SECTOR for size in before.arena_sizes] == [4, 2, 4, 4]
        plan = plan_pinned_arenas(self.REGIONS, arena_bytes=4 * SECTOR, align=SECTOR)
        assert [size // SECTOR for size in plan.arena_sizes] == [4, 4, 4]

    def test_each_region_is_carved_at_its_own_index(self, monkeypatch):
        source, arenas, regions, allocated = _stage(monkeypatch, self.REGIONS, off_by=0)
        assert allocated == [4 * SECTOR] * 3
        assert source.pinned_bytes == 12 * SECTOR
        assert [region.numel() for region in regions] == self.REGIONS
        assert all(region.data_ptr() % SECTOR == 0 for region in regions)
        # The first three-sector region was placed first, and is still region 6.
        assert regions[6].data_ptr() == arenas[0].data_ptr()
        assert regions[7].data_ptr() == arenas[1].data_ptr()
        spans = sorted((r.data_ptr(), r.data_ptr() + r.numel()) for r in regions)
        assert all(end <= start for (_, end), (start, _) in pairwise(spans))

    def test_a_full_packed_arena_off_a_sector_boundary_takes_the_next_size_up(self, monkeypatch):
        """#1531 meets #1702: a packed arena has no slack to shift in, so a base that
        comes back 512 bytes off a sector boundary costs the next power of two, exactly
        as an in-order arena that happens to be full does. The regions stay aligned."""
        source, arenas, regions, allocated = _stage(monkeypatch, self.REGIONS, off_by=512)
        assert allocated == [4 * SECTOR, 8 * SECTOR] * 3
        assert [arena.numel() for arena in arenas] == [8 * SECTOR] * 3
        assert source.pinned_bytes == 24 * SECTOR
        assert [region.numel() for region in regions] == self.REGIONS
        assert all(region.data_ptr() % SECTOR == 0 for region in regions)
        spans = sorted((r.data_ptr(), r.data_ptr() + r.numel()) for r in regions)
        assert all(end <= start for (_, end), (start, _) in pairwise(spans))


# ==========================================================================
# Real hardware: a store that DOES select the packed plan
# ==========================================================================
@pytest.fixture
def _release_packed_fixture_pinned_cache():
    yield
    # The test's locals are gone by teardown: hand its cached arenas back, so the
    # hardware tests that run after it do not inherit them.
    import gc

    import torch

    from soup_cli.utils.layer_stream_runtime import release_cached_pinned_memory

    torch.cuda.synchronize()
    gc.collect()
    release_cached_pinned_memory()


@pytest.mark.gpu
@pytest.mark.usefixtures("_release_packed_fixture_pinned_cache")
class TestOnRealHardware:
    def test_a_packed_store_is_bit_identical_pinned_and_copies_to_the_gpu(
        self, tmp_path, monkeypatch
    ):
        """Every hardware fixture of #901 fits one arena or ties, so none of them runs
        a reordered plan. This one does: six 256 KiB tensors, then two of 600 KiB and a
        scalar, under a 1 MiB ceiling. In order: 1 MiB + 512 KiB + 1 MiB + 1 MiB.
        Largest-first: three arenas of 1 MiB."""
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        from soup_cli.utils.layer_shard import layer_shard_path

        monkeypatch.setattr(runtime, "PINNED_ARENA_MAX_BYTES", 2**20)
        arena_floor = 2**18
        expected_sizes = [2**18] * 6 + [600 * 1024, 600 * 1024, 4]
        assert _in_order(expected_sizes, arena_bytes=arena_floor).pinned_bytes == 7 * 2**19
        plan = plan_pinned_arenas(expected_sizes, arena_bytes=arena_floor)
        assert plan.arena_sizes == (2**20, 2**20, 2**20)
        assert plan.requested_bytes == 2_801_668
        assert plan.placements[6] == (0, 0)
        assert plan.placements[0] != (0, 0)

        weights = {
            "a_00_packed": torch.arange(2**18, dtype=torch.int64).remainder(251).to(torch.uint8),
            "a_01_absmax": torch.linspace(-1.0, 1.0, 2**16, dtype=torch.float32),
            "a_02_norm": torch.arange(2**17, dtype=torch.int32).remainder(997).to(torch.bfloat16),
            "a_03_packed": torch.full((2**18,), 37, dtype=torch.uint8),
            "a_04_absmax": torch.linspace(0.25, 2.0, 2**16, dtype=torch.float32),
            "a_05_norm": torch.full((2**17,), 3.25, dtype=torch.bfloat16),
            "z_00_vocab": torch.arange(300 * 1024, dtype=torch.int32)
            .remainder(97)
            .to(torch.bfloat16)
            .reshape(600, 512),
            "z_01_matrix": torch.arange(150 * 1024, dtype=torch.float32).reshape(300, 512) * 0.125,
            "z_02_nested_offset": torch.tensor(0.03125, dtype=torch.float32),
        }
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        path = layer_shard_path(str(shard_dir), 0)
        save_file(weights, path)
        with open(path, "rb") as handle:
            source_hash = hashlib.sha256(handle.read()).hexdigest()
        spec = runtime.RamSource.spec_from_shard(str(shard_dir))
        assert list(spec) == list(weights)
        actual_sizes = [weights[name].numel() * weights[name].element_size() for name in spec]
        assert actual_sizes == expected_sizes

        torch.cuda.init()
        torch.cuda.synchronize()
        before = int(torch.cuda.host_memory_stats()["active_bytes.current"])
        source = runtime.RamSource(str(shard_dir), 1, spec, pin=True, arena_bytes=arena_floor)
        after = int(torch.cuda.host_memory_stats()["active_bytes.current"])
        assert source.pinned_bytes == plan.pinned_bytes == after - before
        assert source.arena_sizes == plan.arena_sizes
        assert source.nbytes == sum(expected_sizes)

        spans = []
        storages = set()
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in spec:
                got = source.get(0, name)
                expected = handle.get_tensor(name)
                assert got.device.type == "cpu" and got.is_pinned()
                assert got.shape == expected.shape and got.dtype == expected.dtype
                assert got.is_contiguous()
                assert got.data_ptr() % runtime.PINNED_ARENA_ALIGN == 0
                # Flattened first, so the 0-dim sidecar can be reinterpreted as bytes too.
                expected_bytes = expected.reshape(-1).view(torch.uint8)
                assert torch.equal(got.reshape(-1).view(torch.uint8), expected_bytes), name
                storages.add(got.untyped_storage().data_ptr())
                spans.append((got.data_ptr(), got.data_ptr() + got.numel() * got.element_size()))
                copied = got.to(device="cuda", non_blocking=True)
                torch.cuda.synchronize()
                assert torch.equal(copied.cpu().reshape(-1).view(torch.uint8), expected_bytes), name
        assert len(storages) == 3
        spans.sort()
        assert all(end <= next_start for (_, end), (next_start, _) in pairwise(spans))
        with open(path, "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == source_hash
