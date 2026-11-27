"""Pinned disk staging accounts for the alignment of actual allocation addresses."""

import pytest


@pytest.mark.parametrize("base_offset", [0, 1, 256, 4095])
def test_pinned_staging_aligns_absolute_addresses(monkeypatch, base_offset):
    import torch

    from soup_cli.utils.async_disk_source import AsyncDiskSource
    from soup_cli.utils.safetensors_reader import SECTOR_BYTES

    original_empty = torch.empty
    allocations = []

    def unaligned_empty(size, *, dtype, device, pin_memory):
        assert pin_memory and device == "cpu" and dtype == torch.uint8
        backing = original_empty(size + 2 * SECTOR_BYTES, dtype=dtype, device=device)
        start = (-backing.data_ptr()) % SECTOR_BYTES + base_offset
        arena = backing[start : start + size]
        allocations.append(arena)
        return arena

    monkeypatch.setattr(torch, "empty", unaligned_empty)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    source = AsyncDiskSource.__new__(AsyncDiskSource)
    source.pinned = True
    sizes = [SECTOR_BYTES, 2 * SECTOR_BYTES, SECTOR_BYTES]
    arenas, regions = source._allocate_staging(sizes, base_slack_bytes=SECTOR_BYTES - 1)
    expected = source._staging_allocation_bytes(sizes)
    assert len(allocations) == len(arenas)
    assert source.pinned_bytes == expected == sum(arena.numel() for arena in allocations)
    for region, size in zip(regions, sizes, strict=True):
        assert region.data_ptr() % SECTOR_BYTES == 0
        assert region.numel() == size
        assert any(
            arena.data_ptr() <= region.data_ptr()
            and region.data_ptr() + size <= arena.data_ptr() + arena.numel()
            for arena in arenas
        )
    for index, region in enumerate(regions):
        region.fill_(index + 1)
    for index, region in enumerate(regions):
        assert torch.all(region == index + 1), "aligned regions overlap"


def test_pinned_alignment_slack_is_budgeted_before_allocation(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.async_disk_source import AsyncDiskSource
    from soup_cli.utils.layer_shard import layer_shard_path

    save_file({"weight": torch.ones(64)}, layer_shard_path(str(tmp_path), 0))
    allocations = []

    def forbid_allocation(*args, **kwargs):
        allocations.append(args)
        raise AssertionError("must refuse before allocating pinned memory")

    monkeypatch.setattr(torch, "empty", forbid_allocation)
    # One 4096-byte span needs up to 4095 extra bytes for its base alignment.
    with pytest.raises(ValueError) as excinfo:
        AsyncDiskSource(
            str(tmp_path),
            1,
            {"weight": ((64,), "float32")},
            pin=True,
            read_ahead=1,
            max_host_bytes=4096,
        )
    assert str(excinfo.value) == (
        "disk staging needs 8192 bytes, exceeding remaining host RAM budget 4096"
    )
    assert allocations == []


def test_arena_plan_reserves_base_slack_without_overlapping_or_oversizing():
    from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

    sizes = [4096] * 5
    plan = plan_pinned_arenas(sizes, arena_bytes=4096, align=4096, base_slack_bytes=4095)
    assert plan.arena_sizes == (16384, 16384)
    assert plan.placements == ((0, 0), (0, 4096), (0, 8192), (1, 0), (1, 4096))
    assert plan.requested_bytes == sum(sizes)
    for (arena, offset), size in zip(plan.placements, sizes, strict=True):
        assert offset + size + 4095 <= plan.arena_sizes[arena]


@pytest.mark.parametrize("slack", [-1, True, 0.5])
def test_arena_plan_rejects_invalid_base_slack(slack):
    from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

    with pytest.raises(ValueError) as excinfo:
        plan_pinned_arenas([4096], base_slack_bytes=slack)
    assert str(excinfo.value) == "base_slack_bytes must be a non-negative integer"


def test_staging_padding_preserves_large_arena_efficiency_and_boundaries():
    from soup_cli.utils.layer_stream_runtime import plan_pinned_arenas

    # Existing 70B-shaped example still fits one 2GiB allocation with padding.
    sizes = [441_434_112, 441_434_112, 536_875_008, 536_875_008]
    plan = plan_pinned_arenas(sizes, align=4096, base_slack_bytes=4095)
    assert plan.arena_sizes == (2**31,)
    assert plan.pinned_bytes / sum(sizes) < 1.10
    for sizes in ([], [0], [4096], [2**31 - 4096, 4096], [2**31]):
        plan = plan_pinned_arenas(sizes, align=4096, base_slack_bytes=4095)
        ends = {}
        for (arena, offset), size in zip(plan.placements, sizes, strict=True):
            assert offset % 4096 == 0
            assert offset >= ends.get(arena, 0)
            assert offset + size + 4095 <= plan.arena_sizes[arena]
            ends[arena] = offset + size
