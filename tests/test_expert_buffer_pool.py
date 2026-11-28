"""``ExpertBufferPool``: one bounded expert slot, leased one expert at a time.

Seam: ``plan_expert_buffer`` + ``expert_allocation_budget`` + ``ExpertBufferPool`` over an
``ExpertShardReader``. The pool is not wired into ``StreamedExperts`` yet, so every test drives
``lease`` directly and compares the leased weights with the stored ones (dense), or with what
``ExpertFetcher`` dequantises from the same file (NF4). The packed-transfer H2D copy needs a
card; its tests carry the ``gpu`` marker and skip on CPU.
"""

from __future__ import annotations

import threading
import types

import pytest

torch = pytest.importorskip("torch")

HIDDEN = 32
INTER = 16


def _experts(count=8, seed=5):
    """``count`` dense ``(gate_up, down)`` pairs of one DeepSeek-V3-class expert shape."""
    gen = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(2 * INTER, HIDDEN, generator=gen) * 0.2,
            torch.randn(HIDDEN, INTER, generator=gen) * 0.2,
        )
        for _ in range(count)
    ]


def _store(tmp_path, experts, *, nf4=False, nested=True, device="cpu", with_pool=True, **pool_kw):
    from soup_cli.utils.expert_shard import ExpertShardReader, write_expert_shard
    from soup_cli.utils.expert_stream import ExpertBufferPool, plan_expert_buffer
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    path = str(tmp_path / "experts.safetensors")
    info = write_expert_shard(path, iter(experts), quantise=nf4, double_quant=nested, device=device)
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs=info.quant_specs,
        dtype="float32", direct=False,
    )
    codes = {}
    if nf4:
        codes = {NF4_CODE_KEY: info.codes[0]}
        if nested:
            codes[NF4_NESTED_CODE_KEY] = info.codes[1]
    if not with_pool:
        return reader, info, codes
    transfer = pool_kw.get("transfer", "sync")
    plan = plan_expert_buffer(reader, dtype="float32", transfer=transfer)
    budget = plan.weight_bytes + (32 * 2**20 if device == "cuda" else 0)
    pool = ExpertBufferPool(
        reader, device=device, dtype="float32", weight_budget_bytes=budget, codes=codes,
        **pool_kw,
    )
    return reader, pool


def _close(reader, pool):
    reader.close()
    pool.close()


# ----------------------------------------------------------------------------------- plan
def test_weight_budget_is_independent_of_expert_count(tmp_path):
    from soup_cli.utils.expert_stream import plan_expert_buffer

    plans = []
    for count in (8, 64):
        folder = tmp_path / str(count)
        folder.mkdir()
        reader, pool = _store(folder, _experts(count))
        plans.append(plan_expert_buffer(reader, dtype="float32"))
        _close(reader, pool)
    assert plans[0] == plans[1]


def test_the_plan_adds_up_and_is_512_aligned(tmp_path):
    from soup_cli.utils.expert_stream import plan_expert_buffer

    reader, pool = _store(tmp_path, _experts(), nf4=True)
    try:
        plan = plan_expert_buffer(reader, dtype="float32")
        parts = (
            plan.dense_bytes + plan.packed_and_stats_bytes + plan.code_bytes
            + plan.scratch_bytes + plan.padding_bytes
        )
        assert parts == plan.weight_bytes and plan.weight_bytes % 512 == 0
        assert plan.dense_bytes == 4 * (2 * INTER * HIDDEN + HIDDEN * INTER)
        assert plan.packed_and_stats_bytes > 0 and plan.code_bytes > 0
        assert plan.packed_region_bytes == 0 and plan.host_pinned_bytes == 0
    finally:
        _close(reader, pool)


def test_the_packed_plan_puts_every_stored_component_in_one_aligned_region_at_the_front(tmp_path):
    from soup_cli.utils.expert_stream import plan_expert_buffer

    reader, pool = _store(tmp_path, _experts(), nf4=True)
    try:
        sync = plan_expert_buffer(reader, dtype="float32")
        packed = plan_expert_buffer(reader, dtype="float32", transfer="packed")
        assert packed.packed_region_bytes >= packed.packed_and_stats_bytes > 0
        assert packed.host_pinned_bytes == packed.packed_region_bytes
        assert (packed.dense_bytes, packed.packed_and_stats_bytes, packed.code_bytes) == (
            sync.dense_bytes, sync.packed_and_stats_bytes, sync.code_bytes,
        )
        assert packed.weight_bytes >= packed.packed_region_bytes + packed.dense_bytes
        assert packed.weight_bytes % 512 == 0
    finally:
        _close(reader, pool)


def test_a_dense_store_has_nothing_to_pack_and_a_bad_mode_is_refused(tmp_path):
    from soup_cli.utils.expert_stream import plan_expert_buffer

    reader, pool = _store(tmp_path, _experts())
    try:
        with pytest.raises(ValueError, match="packed transfer needs NF4"):
            plan_expert_buffer(reader, dtype="float32", transfer="packed")
        with pytest.raises(ValueError, match="transfer must be 'sync' or 'packed'"):
            plan_expert_buffer(reader, dtype="float32", transfer="prefetch")
        with pytest.raises(ValueError, match="compute dtype must be float32, float16 or bfloat16"):
            plan_expert_buffer(reader, dtype="float64")
    finally:
        _close(reader, pool)


def test_nf4_plan_refuses_a_compute_dtype_that_differs_from_the_stored_spec(tmp_path):
    from soup_cli.utils.expert_stream import plan_expert_buffer

    reader, pool = _store(tmp_path, _experts(), nf4=True)
    try:
        with pytest.raises(ValueError, match="must match its stored NF4 spec"):
            plan_expert_buffer(reader, dtype="bfloat16")
    finally:
        _close(reader, pool)


@pytest.mark.parametrize("device,extra", [("cpu", 0), ("cuda:0", 32 * 2**20)])
def test_allocation_budget_reserves_an_allocator_allowance_only_on_cuda(
    tmp_path, device, extra
):
    from soup_cli.utils.expert_stream import expert_allocation_budget, plan_expert_buffer

    reader, pool = _store(tmp_path, _experts())
    try:
        plan = plan_expert_buffer(reader, dtype="float32")
        assert expert_allocation_budget(plan, device=device) == plan.weight_bytes + extra
    finally:
        _close(reader, pool)


# ----------------------------------------------------------------------------------- budget
def test_budget_is_checked_before_allocating_or_reading(tmp_path, monkeypatch):
    from soup_cli.utils.expert_stream import ExpertBufferPool

    reader, pool = _store(tmp_path, _experts())
    needed = pool.plan.weight_bytes
    pool.close()

    def forbidden(*args, **kwargs):
        pytest.fail("an insufficient budget must fail before allocation or a disk read")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(reader, "read_host", forbidden)
    message = f"expert slot needs {needed} bytes; budget is {needed - 1}"
    with pytest.raises(ValueError, match=message):
        ExpertBufferPool(reader, device="cpu", dtype="float32", weight_budget_bytes=needed - 1)
    assert reader.fetches == 0
    reader.close()


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "10"])
def test_weight_budget_must_be_a_positive_integer(tmp_path, bad):
    from soup_cli.utils.expert_stream import ExpertBufferPool

    reader, pool = _store(tmp_path, _experts())
    pool.close()
    try:
        with pytest.raises(ValueError, match="weight_budget_bytes must be a positive integer"):
            ExpertBufferPool(reader, device="cpu", dtype="float32", weight_budget_bytes=bad)
    finally:
        reader.close()


@pytest.mark.parametrize("reserve,overhead,accepted", [
    (False, 1024, False), (True, 1024, True), (True, 33 * 2**20, False),
])
def test_observed_rounding_is_checked_against_explicit_budget(
    tmp_path, monkeypatch, reserve, overhead, accepted
):
    """CUDA block rounding must be budgeted, with an observed hard check retained."""
    from soup_cli.utils.expert_stream import ExpertBufferPool, expert_allocation_budget

    reader, initial = _store(tmp_path, _experts())
    plan = initial.plan
    initial.close()
    original_empty = torch.empty

    def cpu_storage(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    samples = iter([0, plan.weight_bytes + overhead])
    monkeypatch.setattr(torch, "empty", cpu_storage)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: next(samples))
    monkeypatch.setattr(ExpertBufferPool, "_synchronize", lambda self: None)
    budget = (expert_allocation_budget(plan, device="cuda:0") if reserve
              else plan.weight_bytes)
    try:
        if not accepted:
            with pytest.raises(ValueError, match="rounded the expert slot.*observed"):
                ExpertBufferPool(reader, device="cuda:0", dtype="float32",
                                 weight_budget_bytes=budget)
        else:
            pool = ExpertBufferPool(reader, device="cuda:0", dtype="float32",
                                    weight_budget_bytes=budget)
            assert pool.cuda_allocation_delta <= budget
            assert pool._arena.untyped_storage().nbytes() == plan.weight_bytes
            pool.close()
    finally:
        reader.close()


# ----------------------------------------------------------------------------------- lease
def test_pool_reuses_one_arena_across_experts_and_blocks_nested_lease(tmp_path):
    experts = _experts()
    reader, pool = _store(tmp_path, experts)
    arena = pool._arena.data_ptr()
    try:
        with pool.lease(reader, 0) as (gate_up, down):
            assert gate_up.untyped_storage().data_ptr() == arena
            assert down.untyped_storage().data_ptr() == arena
            assert gate_up.untyped_storage().data_ptr() != (
                pool._staging.untyped_storage().data_ptr()
            )
            assert torch.equal(gate_up, experts[0][0]) and torch.equal(down, experts[0][1])
            snapshot = gate_up.clone()
            with pytest.raises(RuntimeError, match="pool already has an active lease"):
                with pool.lease(reader, 1):
                    pass
            with pytest.raises(RuntimeError, match="cannot close an expert pool with an active"):
                pool.close()
            assert torch.equal(gate_up, snapshot)
        with pool.lease(reader, 1) as (gate_up, down):
            assert pool._arena.data_ptr() == arena
            assert torch.equal(gate_up, experts[1][0]) and torch.equal(down, experts[1][1])
        assert pool.stats()["slot_count"] == 1 and pool.fetches == 2
    finally:
        _close(reader, pool)
    assert pool._arena is None and not pool._buffers
    with pytest.raises(RuntimeError, match="expert buffer pool is closed"):
        with pool.lease(reader, 1):
            pass


def test_a_lease_from_another_thread_is_refused_while_one_is_held(tmp_path):
    reader, pool = _store(tmp_path, _experts())
    seen = []

    def other():
        try:
            with pool.lease(reader, 1):
                seen.append("leased")
        except RuntimeError as error:
            seen.append(str(error))

    try:
        with pool.lease(reader, 0):
            thread = threading.Thread(target=other)
            thread.start()
            thread.join()
        assert seen == ["expert buffer pool already has an active lease"]
    finally:
        _close(reader, pool)


def test_the_lease_is_released_when_the_body_raises(tmp_path):
    reader, pool = _store(tmp_path, _experts())
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with pool.lease(reader, 0):
                raise RuntimeError("boom")
        assert reader._handle is None  # a failed lease closes the reader it was reading
        with pool.lease(reader, 1):
            pass
    finally:
        _close(reader, pool)


def test_load_error_closes_the_reader_and_the_lease_can_be_reused(tmp_path, monkeypatch):
    experts = _experts()
    reader, pool = _store(tmp_path, experts)
    original = reader.read_host

    def broken(*args):
        raise OSError("deliberate short read")

    monkeypatch.setattr(reader, "read_host", broken)
    try:
        with pytest.raises(OSError, match="deliberate"):
            with pool.lease(reader, 0):
                pass
        assert reader._handle is None and pool.fetches == 0
        monkeypatch.setattr(reader, "read_host", original)
        with pool.lease(reader, 2) as (gate_up, _down):
            assert torch.equal(gate_up, experts[2][0])
    finally:
        _close(reader, pool)


def test_pool_close_releases_buffers_even_if_device_sync_fails(tmp_path, monkeypatch):
    reader, pool = _store(tmp_path, _experts())

    def failed_sync():
        raise RuntimeError("injected device synchronization failure")

    monkeypatch.setattr(pool, "_synchronize", failed_sync)
    with pytest.raises(RuntimeError, match="injected"):
        pool.close()
    assert pool.stats()["closed"] and pool._arena is None and not pool._buffers
    reader.close()


def test_a_reader_with_another_storage_contract_is_refused(tmp_path):
    reader, pool = _store(tmp_path, _experts())
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other, info, _ = _store(other_dir, _experts(), nf4=True, with_pool=False)
    try:
        with pytest.raises(ValueError, match="expert reader does not match the buffer pool"):
            with pool.lease(other, 0):
                pass
        assert pool.fetches == 0
    finally:
        other.close()
        _close(reader, pool)


def test_lease_phase_and_layer_must_be_given_together_and_be_known(tmp_path):
    reader, pool = _store(tmp_path, _experts())
    try:
        with pytest.raises(ValueError, match="unsupported expert-read phase: 'nope'"):
            with pool.lease(reader, 0, phase="nope", layer_idx=0):
                pass
        with pytest.raises(ValueError, match="phase and layer_idx must be provided together"):
            with pool.lease(reader, 0, phase="init_forward"):
                pass
        with pytest.raises(ValueError, match="phase and layer_idx must be provided together"):
            with pool.lease(reader, 0, layer_idx=1):
                pass
        assert pool.fetches == 0
        with pool.lease(reader, 0):  # none of the refusals left the lock held
            pass
    finally:
        _close(reader, pool)


def test_phase_read_snapshot_counts_only_when_tracking_is_on(tmp_path):
    reader, pool = _store(tmp_path, _experts())
    try:
        with pool.lease(reader, 0, phase="init_forward", layer_idx=3):
            pass
        assert pool.phase_read_snapshot() == {}
        pool.phase_read_tracking = True
        for expert, phase in ((1, "init_forward"), (2, "recompute_forward"), (3, "backward_dX"),
                              (4, "backward_dX")):
            with pool.lease(reader, expert, phase=phase, layer_idx=3):
                pass
        snapshot = pool.phase_read_snapshot()
        assert set(snapshot) == {3}
        assert snapshot[3]["init_forward"]["expert_fetches"] == 1
        assert snapshot[3]["recompute_forward"]["expert_fetches"] == 1
        assert snapshot[3]["backward_dX"]["expert_fetches"] == 2
        assert snapshot[3]["backward_dX"]["expert_bytes_read"] == 2 * pool.last_lease_bytes
        snapshot[3]["backward_dX"]["expert_fetches"] = 99  # a copy, not the live counters
        assert pool.phase_read_snapshot()[3]["backward_dX"]["expert_fetches"] == 2
    finally:
        _close(reader, pool)


def test_stats_describe_the_slot(tmp_path):
    reader, pool = _store(tmp_path, _experts())
    try:
        with pool.lease(reader, 0):
            pass
        stats = pool.stats()
        assert stats["slot_count"] == 1 and stats["fetches"] == 1
        assert stats["logical_bytes_read"] == pool.last_lease_bytes > 0
        assert stats["transfer"] == "sync" and stats["host_cache"] is None
        assert stats["prefetch"] == {"enabled": False, "hits": 0, "discards": 0}
        assert stats["plan"]["weight_bytes"] == pool.plan.weight_bytes
        assert stats["closed"] is False and stats["cpu_nf4_temporaries"] is False
    finally:
        _close(reader, pool)


# ----------------------------------------------------------------------------------- NF4
@pytest.mark.parametrize("nested", [False, True])
def test_nf4_lease_equals_what_the_fetcher_dequantises_from_the_same_file(tmp_path, nested):
    pytest.importorskip("bitsandbytes")
    from soup_cli.utils.expert_stream import ExpertFetcher

    reader, pool = _store(tmp_path, _experts(), nf4=True, nested=nested)
    try:
        # the fetcher needs the same code tables the pool was given
        fetcher = ExpertFetcher(device="cpu", dtype="float32", codes=_codes_of(pool))
        for expert in (0, 3, 7):
            want_gate_up, want_down = fetcher.fetch(reader, expert)
            with pool.lease(reader, expert) as (gate_up, down):
                assert torch.equal(gate_up, want_gate_up) and torch.equal(down, want_down)
        assert pool.stats()["cpu_nf4_temporaries"] is True
    finally:
        _close(reader, pool)


def _codes_of(pool):
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    codes = {NF4_CODE_KEY: pool._buffers["code"]}
    if "nested_code" in pool._buffers:
        codes[NF4_NESTED_CODE_KEY] = pool._buffers["nested_code"]
    return codes


def test_an_nf4_pool_refuses_a_missing_or_malformed_code_table(tmp_path):
    from soup_cli.utils.expert_stream import ExpertBufferPool, plan_expert_buffer
    from soup_cli.utils.layer_shard import NF4_CODE_KEY

    reader, info, codes = _store(tmp_path, _experts(), nf4=True, nested=False, with_pool=False)
    budget = plan_expert_buffer(reader, dtype="float32").weight_bytes
    common = dict(device="cpu", dtype="float32", weight_budget_bytes=budget)
    try:
        for bad in (None, torch.zeros(15), torch.zeros(16, dtype=torch.float16),
                    torch.full((16,), float("nan"))):
            table = {} if bad is None else {NF4_CODE_KEY: bad}
            with pytest.raises(ValueError, match="missing or invalid NF4 code table"):
                ExpertBufferPool(reader, codes=table, **common)
        pool = ExpertBufferPool(reader, codes=codes, **common)
        pool.close()
    finally:
        reader.close()


# ----------------------------------------------------------------------------------- transfer
def test_the_pool_refuses_packed_transfer_without_cuda_and_unsupported_combinations(tmp_path):
    from soup_cli.utils.expert_stream import ExpertBufferPool, plan_expert_buffer
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    reader, info, _codes = _store(tmp_path, _experts(), nf4=True, with_pool=False)
    plan = plan_expert_buffer(reader, dtype="float32", transfer="packed")
    codes = {
        NF4_CODE_KEY: torch.linspace(-1, 1, 16),
        NF4_NESTED_CODE_KEY: torch.linspace(-1, 1, 256),
    }
    common = dict(device="cpu", dtype="float32", weight_budget_bytes=plan.weight_bytes, codes=codes)
    try:
        with pytest.raises(ValueError, match="packed expert transfer requires CUDA"):
            ExpertBufferPool(reader, transfer="packed", **common)
        with pytest.raises(ValueError, match="host_cache_bytes needs transfer='packed'"):
            ExpertBufferPool(reader, host_cache_bytes=1 << 20, **common)
        for bad in (-1, True, 1.5):
            with pytest.raises(
                ValueError, match="host_cache_bytes must be a non-negative integer"
            ):
                ExpertBufferPool(reader, host_cache_bytes=bad, **common)
        with pytest.raises(ValueError, match="transfer must be 'sync' or 'packed'"):
            ExpertBufferPool(reader, transfer="prefetch", **common)
        with pytest.raises(ValueError, match="expert forward supports CPU or CUDA only"):
            ExpertBufferPool(reader, **dict(common, device="meta"))
    finally:
        reader.close()


# ----------------------------------------------------------------------------------- prefetch
def test_prefetch_is_off_by_default_and_starts_no_thread(tmp_path):
    reader, pool = _store(tmp_path, _experts())
    try:
        pool.prefetch(reader, 1)  # a hint on a pool built without prefetch does nothing
        assert pool._pending is None and pool._prefetch_executor is None
        assert not [t for t in threading.enumerate() if t.name.startswith("expert-prefetch")]
    finally:
        _close(reader, pool)


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_prefetch_must_be_a_bool(tmp_path, bad):
    from soup_cli.utils.expert_stream import ExpertBufferPool

    reader, pool = _store(tmp_path, _experts())
    budget = pool.plan.weight_bytes
    pool.close()
    try:
        with pytest.raises(ValueError, match="prefetch must be a bool"):
            ExpertBufferPool(
                reader, device="cpu", dtype="float32", weight_budget_bytes=budget, prefetch=bad
            )
    finally:
        reader.close()


def test_a_hinted_expert_is_used_and_a_hint_for_another_is_discarded(tmp_path):
    experts = _experts()
    reader, pool = _store(tmp_path, experts, prefetch=True)
    try:
        pool.prefetch(reader, 1)
        pool.prefetch(reader, 1)  # the same read is already pending: no second one
        with pool.lease(reader, 1) as (gate_up, down):
            assert torch.equal(gate_up, experts[1][0]) and torch.equal(down, experts[1][1])
        assert pool.prefetch_hits == 1 and pool.prefetch_discards == 0
        assert pool.last_lease_bytes > 0  # the lease reports what the background read took
        pool.prefetch(reader, 2)
        with pool.lease(reader, 3) as (gate_up, _down):
            assert torch.equal(gate_up, experts[3][0])
        assert pool.prefetch_hits == 1 and pool.prefetch_discards == 1
        pool.prefetch(reader, 4)
        pool.prefetch(reader, 5)  # replaces the pending 4: one read in flight at a time
        assert pool.prefetch_discards == 2
        with pool.lease(reader, 5) as (gate_up, _down):
            assert torch.equal(gate_up, experts[5][0])
        assert pool.stats()["prefetch"] == {"enabled": True, "hits": 2, "discards": 2}
        pool.prefetch_enabled = False
        pool.prefetch(reader, 6)
        assert pool._pending is None
    finally:
        _close(reader, pool)
    assert not [t for t in threading.enumerate() if t.name.startswith("expert-prefetch")]


def test_a_failed_background_read_raises_in_the_lease_that_asked_for_it(tmp_path, monkeypatch):
    reader, pool = _store(tmp_path, _experts(), prefetch=True)
    original = reader.read_host

    def broken(expert, staging):
        raise OSError("deliberate background failure")

    monkeypatch.setattr(reader, "read_host", broken)
    try:
        pool.prefetch(reader, 1)
        with pytest.raises(OSError, match="deliberate background failure"):
            with pool.lease(reader, 1):
                pass
        monkeypatch.setattr(reader, "read_host", original)
        with pool.lease(reader, 1):
            pass
    finally:
        _close(reader, pool)


def test_a_prefetch_after_close_does_nothing(tmp_path):
    reader, pool = _store(tmp_path, _experts(), prefetch=True)
    pool.close()
    pool.prefetch(reader, 1)
    assert pool._pending is None and pool._prefetch_executor is None
    reader.close()


# ----------------------------------------------------------------------------------- async lease
def _counting_pool(tmp_path):
    reader, pool = _store(tmp_path, _experts())
    calls = []
    pool._synchronize = lambda: calls.append(1)
    return reader, pool, calls


def test_async_lease_is_off_by_default_and_a_lease_syncs_the_device(tmp_path):
    from soup_cli.utils.expert_stream import ExpertBufferPool

    assert ExpertBufferPool.async_lease is False
    reader, pool, calls = _counting_pool(tmp_path)
    try:
        with pool.lease(reader, 0):
            pass
        assert calls == [1]
    finally:
        _close(reader, pool)


@pytest.mark.parametrize(
    "transfer,device,async_lease,syncs",
    [
        ("packed", "cuda", True, 0),      # the one case that skips the device-wide wait
        ("packed", "cuda", False, 1),
        ("sync", "cuda", True, 1),
        ("packed", "cpu", True, 1),
    ],
)
def test_lease_end_skips_the_device_sync_only_for_async_packed_cuda_pools(
    tmp_path, transfer, device, async_lease, syncs
):
    reader, pool, calls = _counting_pool(tmp_path)
    try:
        pool.transfer = transfer
        pool.device = types.SimpleNamespace(type=device)
        pool.async_lease = async_lease
        pool._lease_end()
        assert len(calls) == syncs
    finally:
        pool.device = torch.device("cpu")
        _close(reader, pool)


def test_a_failing_lease_still_syncs_the_device_even_when_async(tmp_path):
    reader, pool, calls = _counting_pool(tmp_path)
    try:
        pool.async_lease = True
        with pytest.raises(RuntimeError, match="boom"):
            with pool.lease(reader, 0):
                raise RuntimeError("boom")
        assert calls  # the error path keeps the full wait
    finally:
        _close(reader, pool)


def test_the_host_cache_key_resolves_the_path_every_time_by_default_and_once_when_async(
    tmp_path, monkeypatch
):
    import os

    reader, pool, _ = _counting_pool(tmp_path)
    try:
        calls = []
        real = os.path.realpath
        monkeypatch.setattr(os.path, "realpath", lambda p, **k: (calls.append(p), real(p, **k))[1])
        assert pool._host_key(reader, 3) == (real(reader.path), 3)
        pool._host_key(reader, 4)
        assert calls.count(reader.path) == 2
        pool.async_lease = True
        calls.clear()
        assert pool._host_key(reader, 3) == (real(reader.path), 3)
        pool._host_key(reader, 4)
        assert calls.count(reader.path) == 1
    finally:
        _close(reader, pool)


# ----------------------------------------------------------------------------------- laziness
def test_the_pool_module_import_keeps_torch_out_of_a_fresh_interpreter():
    import subprocess
    import sys

    code = (
        "import sys; import soup_cli.utils.expert_stream, soup_cli.utils.expert_transfer, "
        "soup_cli.utils.expert_cache_budget; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ----------------------------------------------------------------------------------- CUDA
@pytest.mark.gpu
@pytest.mark.parametrize("nested", [False, True])
def test_cuda_packed_transfer_leases_the_same_weights_as_the_default_pool(tmp_path, nested):
    pytest.importorskip("bitsandbytes")
    experts = _experts()
    got = {}
    pools = []
    try:
        for transfer in ("sync", "packed"):
            folder = tmp_path / transfer
            folder.mkdir()
            reader, pool = _store(folder, experts, nf4=True, nested=nested, device="cuda",
                                  transfer=transfer)
            pools.append((reader, pool))
            rows = []
            for expert in (0, 3, 3, 7):
                with pool.lease(reader, expert) as (gate_up, down):
                    rows.append((gate_up.clone(), down.clone()))
            got[transfer] = rows
        for (sync_gate, sync_down), (packed_gate, packed_down) in zip(
            got["sync"], got["packed"]
        ):
            assert torch.equal(sync_gate, packed_gate) and torch.equal(sync_down, packed_down)
        reader, pool = pools[1]
        stats = pool.stats()
        assert stats["transfer"] == "packed" and stats["h2d_copies"] == pool.fetches == 4
        assert pool.cuda_allocation_delta <= pool.weight_budget_bytes
    finally:
        for reader, pool in pools:
            _close(reader, pool)


@pytest.mark.gpu
def test_cuda_a_host_cache_hit_reads_nothing_and_changes_nothing(tmp_path):
    pytest.importorskip("bitsandbytes")
    reader, pool = _store(tmp_path, _experts(), nf4=True, device="cuda", transfer="packed",
                          host_cache_bytes=64 * 2**20)
    try:
        first = []
        for expert in (1, 2):
            with pool.lease(reader, expert) as (gate_up, down):
                first.append((gate_up.clone(), down.clone()))
        reads = reader.fetches
        for expert, (want_gate, want_down) in zip((1, 2), first):
            with pool.lease(reader, expert) as (gate_up, down):
                assert torch.equal(gate_up, want_gate) and torch.equal(down, want_down)
        assert reader.fetches == reads  # served from the host hotset, nothing read
        assert pool.stats()["host_cache"]["hits"] == 2
        assert pool.stats()["host_cache"]["rejected"] == 0
    finally:
        _close(reader, pool)


@pytest.mark.gpu
def test_cuda_async_lease_leases_the_same_weights_and_skips_the_device_sync(tmp_path):
    pytest.importorskip("bitsandbytes")
    experts = _experts()
    syncs, got, pools = {}, {}, []
    real_sync = torch.cuda.synchronize
    try:
        for name, async_lease in (("default", False), ("async", True)):
            folder = tmp_path / name
            folder.mkdir()
            reader, pool = _store(folder, experts, nf4=True, device="cuda", transfer="packed",
                                  host_cache_bytes=64 * 2**20)
            pools.append((reader, pool))
            pool.async_lease = async_lease
            counter = []
            torch.cuda.synchronize = lambda *a, **k: (counter.append(1), real_sync(*a, **k))[1]
            try:
                rows = []
                for expert in (0, 1, 2, 3, 0, 1):
                    with pool.lease(reader, expert) as (gate_up, down):
                        rows.append((gate_up.clone(), down.clone()))
                got[name] = rows
            finally:
                torch.cuda.synchronize = real_sync
            syncs[name] = len(counter)
        for (a_gate, a_down), (b_gate, b_down) in zip(got["default"], got["async"]):
            assert torch.equal(a_gate, b_gate) and torch.equal(a_down, b_down)
        assert syncs["default"] >= 6
        assert syncs["async"] < 3
    finally:
        torch.cuda.synchronize = real_sync
        for reader, pool in pools:
            _close(reader, pool)
