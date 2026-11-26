"""Byte layout and the static host hotset of the packed NF4 expert transfer (CPU logic)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _tensors() -> dict:
    gen = torch.Generator().manual_seed(11)
    return {
        "weights": torch.randint(0, 255, (101, 1), dtype=torch.uint8, generator=gen),
        "absmax": torch.randn(7, generator=gen),
        "nested_absmax": torch.randint(0, 255, (13,), dtype=torch.uint8, generator=gen),
        "offset": torch.randn((), generator=gen),
    }


def test_packing_layout_aligns_every_tensor_and_keeps_its_bytes_exact() -> None:
    from soup_cli.utils.expert_transfer import pack_into, packing_layout

    tensors = _tensors()
    rows, size = packing_layout(tensors)
    assert [row[0] for row in rows] == list(tensors)
    ends = [offset + count for _, offset, count, _, _ in rows]
    assert all(offset % 256 == 0 for _, offset, _, _, _ in rows)
    assert all(end <= row[1] for end, row in zip(ends, rows[1:]))
    assert size >= ends[-1]
    assert dict((row[0], row[2]) for row in rows)["offset"] == 4  # a scalar keeps its bytes
    block = torch.zeros(size, dtype=torch.uint8)
    assert pack_into(block, rows, tensors) == sum(row[2] for row in rows)
    for name, offset, count, dtype, shape in rows:
        view = block[offset : offset + count].view(dtype).view(shape)
        assert torch.equal(view, tensors[name])
    covered = torch.zeros(size, dtype=torch.bool)
    for _, offset, count, _, _ in rows:
        covered[offset : offset + count] = True
    assert not block[~covered].any()  # the padding stays zero


def test_a_stored_tensor_that_does_not_match_its_row_is_refused_not_converted() -> None:
    from soup_cli.utils.expert_transfer import pack_into, packing_layout

    tensors = _tensors()
    rows, size = packing_layout(tensors)
    wrong = dict(tensors, absmax=tensors["absmax"].to(torch.float64))
    with pytest.raises(ValueError, match="absmax: stored tensor does not match its packed layout"):
        pack_into(torch.zeros(size, dtype=torch.uint8), rows, wrong)
    reshaped = dict(tensors, absmax=torch.zeros(8))
    with pytest.raises(ValueError, match="absmax: stored tensor does not match"):
        pack_into(torch.zeros(size, dtype=torch.uint8), rows, reshaped)


def test_packing_refuses_an_empty_or_non_contiguous_transfer() -> None:
    from soup_cli.utils.expert_transfer import layout_rows, packing_layout

    with pytest.raises(ValueError, match="an expert transfer must be nonempty"):
        packing_layout({})
    with pytest.raises(ValueError, match="an expert transfer must be nonempty"):
        layout_rows([])
    with pytest.raises(ValueError, match="nonempty contiguous CPU tensors"):
        packing_layout({"x": torch.zeros(4, 4).t()})
    with pytest.raises(ValueError, match="nonempty contiguous CPU tensors"):
        packing_layout({"x": torch.zeros(0)})


def test_layout_rows_aligns_wide_elements_to_their_own_width() -> None:
    from soup_cli.utils.expert_transfer import layout_rows

    rows, size = layout_rows([("a", (3,), torch.uint8), ("b", (2,), torch.float32)])
    assert rows[0] == ("a", 0, 3, torch.uint8, (3,))
    assert rows[1] == ("b", 256, 8, torch.float32, (2,)) and size == 264


def test_the_host_hotset_charges_whole_blocks_and_never_evicts() -> None:
    from soup_cli.utils.expert_transfer import PackedHostHotset

    cache = PackedHostHotset(300)
    first = torch.arange(200, dtype=torch.uint8)
    assert cache.put("a", first) and not cache.put("a", first)  # a key is admitted once
    assert not cache.put("b", torch.zeros(101, dtype=torch.uint8))  # 200 + 101 > 300
    assert cache.put("c", torch.zeros(100, dtype=torch.uint8))  # exactly fills the budget
    assert not cache.put("d", torch.zeros(1, dtype=torch.uint8))
    first.fill_(0)  # the cache holds an owned copy
    got = cache.get("a")
    assert torch.equal(got, torch.arange(200, dtype=torch.uint8))
    assert cache.get("missing") is None
    assert cache.holds("a") and not cache.holds("b")
    assert cache.stats() == {
        "budget_bytes": 300, "payload_bytes": 300, "entries": 2, "hits": 1, "misses": 1,
        "rejected": 3,
    }


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "8"])
def test_the_host_hotset_budget_must_be_a_positive_integer(bad) -> None:
    from soup_cli.utils.expert_transfer import PackedHostHotset

    with pytest.raises(ValueError, match="budget_bytes must be a positive integer"):
        PackedHostHotset(bad)


def test_holds_does_not_touch_the_counters() -> None:
    from soup_cli.utils.expert_transfer import PackedHostHotset

    cache = PackedHostHotset(64)
    cache.put("a", torch.zeros(8, dtype=torch.uint8))
    cache.holds("a")
    cache.holds("z")
    assert cache.hits == 0 and cache.misses == 0
