"""Installing the expert pool through ``install_streaming`` / ``build_streamed_model`` (CPU).

A tiny DeepSeek-V3 is sharded per expert and streamed. The new switches are API keywords
(``expert_pool``, ``grouped_expert_routes`` and the pool's ``expert_transfer``,
``expert_host_cache_bytes``, ``expert_prefetch``); with none of them set the run is the one it
always was. The model-level bit-equality of the levers is in ``test_streamed_experts_levers.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tests.test_expert_stream import _lora, _shard, _tiny_deepseek_v3  # noqa: E402


def _build(tmp_path: Path, index: Any, weights: str, **kwargs: Any) -> Any:
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    return build_streamed_model(
        model_id=weights, shard_dir=str(tmp_path / "shards"), index=index, lora_config=_lora(),
        device="cpu", dtype="float32", buffers=2, pin=False, seed=3, **kwargs,
    )


def _moe_experts(model: Any, idx: int) -> Any:
    return model.get_submodule("base_model.model.model.layers")[idx].inner.mlp.experts


def test_without_a_switch_the_experts_read_through_the_fetcher_as_before(tmp_path: Path) -> None:
    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    model, runtime = _build(tmp_path, index, weights)
    try:
        experts = _moe_experts(model, 1)
        assert runtime.expert_pool is None and experts.pool is None
        assert experts.fetcher is not None and not experts.grouped_expert_routes
        assert runtime.stats()["expert_buffer_bytes"] == 0
    finally:
        runtime.close()


def test_the_pool_and_the_grouped_lever_reach_every_moe_layer_and_the_runtime(
    tmp_path: Path,
) -> None:
    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    model, runtime = _build(
        tmp_path, index, weights, expert_pool=True, grouped_expert_routes=True,
        two_read_experts=True,
    )
    try:
        pool = runtime.expert_pool
        assert pool is not None and len(runtime.expert_readers) == 2
        for idx in (1, 2):
            experts = _moe_experts(model, idx)
            assert experts.pool is pool and experts.fetcher is None
            assert experts.grouped_expert_routes and experts.two_reads
        stats = runtime.stats()
        assert stats["expert_buffer_bytes"] == pool.plan.weight_bytes > 0
        assert stats["expert_fetches"] == 0 and stats["expert_bytes_read"] == 0
        model.eval()
        ids = torch.arange(24, dtype=torch.int64).reshape(2, 12) % 128
        model(input_ids=ids, labels=ids).loss.backward()
        stats = runtime.stats()
        assert stats["expert_fetches"] == pool.fetches > 0
        assert stats["expert_bytes_read"] == pool.logical_bytes_read > 0
    finally:
        runtime.close()
    assert runtime.expert_pool.stats()["closed"] is True
    assert all(reader._handle is None for reader in runtime.expert_readers)


def test_the_pool_options_tune_the_pool_and_need_the_pool(tmp_path: Path) -> None:
    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    for option in (
        {"expert_transfer": "packed"}, {"expert_host_cache_bytes": 1 << 20},
        {"expert_prefetch": True},
    ):
        (name,) = option
        with pytest.raises(ValueError, match=f"{name} needs expert_pool=True"):
            _build(tmp_path, index, weights, **option)
    model, runtime = _build(tmp_path, index, weights, expert_pool=True, expert_prefetch=True)
    try:
        assert runtime.expert_pool.prefetch_enabled is True
        assert _moe_experts(model, 1).pool is runtime.expert_pool
    finally:
        runtime.close()


def test_a_pool_that_cannot_be_built_closes_the_readers_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from soup_cli.utils import expert_shard

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    closed: set = set()
    original = expert_shard.ExpertShardReader.close

    def counting_close(self: Any) -> None:
        closed.add(id(self))  # close() may run more than once per reader
        original(self)

    monkeypatch.setattr(expert_shard.ExpertShardReader, "close", counting_close)
    # The plan refuses packed transfer on dense experts before the pool is built; the readers
    # opened for it must still be closed.
    with pytest.raises(ValueError, match="packed transfer needs NF4 in every expert matrix"):
        _build(tmp_path, index, weights, expert_pool=True, expert_transfer="packed")
    assert len(closed) == 2  # every reader opened, one per MoE layer


@pytest.mark.parametrize("lever", ["two_read_experts", "grouped_expert_routes", "expert_pool"])
def test_every_lever_needs_a_per_expert_cache_instead_of_being_ignored(
    tmp_path: Path, lever: str
) -> None:
    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights)
    assert index.expert_layout != "per_expert"
    with pytest.raises(ValueError, match=f"{lever} needs a cache sharded with"):
        _build(tmp_path, index, weights, **{lever: True})


@pytest.mark.parametrize("name", ["expert_pool", "expert_prefetch"])
@pytest.mark.parametrize("bad", [1, 0, "no", None])
def test_the_pool_switches_take_only_a_bool(name: str, bad: Any) -> None:
    from soup_cli.utils.layer_stream_runtime import _install_expert_streaming

    with pytest.raises(ValueError, match=f"{name} must be a bool"):
        _install_expert_streaming(
            [], "unused", object(), device="cpu", codes=None, **{name: bad}
        )
