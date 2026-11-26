"""FP8 boundary codec parity, ragged capacity, accounting, and resource lifetimes."""

import weakref

import pytest

from soup_cli.utils.host_memory_budget import HostMemoryBudget


def test_fp8_allocation_charges_each_pinned_payload_and_one_scale_slab():
    from soup_cli.utils.fp8_activations import fp8_activation_allocation

    shapes = [(1, 3, 5), (2, 2, 5)]
    raw = fp8_activation_allocation(n_layers=2, mb_shapes=shapes, pin=False)
    pinned = fp8_activation_allocation(n_layers=2, mb_shapes=shapes, pin=True)
    # Four payloads request 15, 20, 15, 20 bytes. The single scale slab requests 56.
    assert raw.payload_bytes == pinned.payload_bytes == 70
    assert raw.scale_bytes == pinned.scale_bytes == 56
    assert raw.pinned_overhead_bytes == 0
    assert pinned.pinned_overhead_bytes == 34  # 16 + 32 + 16 + 32 + 64 - 126
    assert raw.host_transient_bytes == pinned.host_transient_bytes == 5
    assert raw.host_bytes == 131
    assert pinned.host_bytes == 165
    # Eight float32 tile matrices/row vectors plus one largest bf16 output and flag.
    assert raw.device_transient_bytes == pinned.device_transient_bytes == 809
    assert pinned.as_dict() == {
        "host_bytes": 165,
        "payload_bytes": 70,
        "scale_bytes": 56,
        "pinned_overhead_bytes": 34,
        "host_transient_bytes": 5,
        "device_transient_bytes": 809,
    }


def test_cpu_codec_scratch_is_host_memory_not_device_memory():
    from soup_cli.utils.fp8_activations import fp8_activation_allocation

    allocation = fp8_activation_allocation(
        n_layers=2, mb_shapes=[(1, 3, 5), (2, 2, 5)], pin=False,
        compute_dtype="float32", device_type="cpu",
    )
    assert allocation.host_transient_bytes == 849
    assert allocation.host_bytes == 975
    assert allocation.device_transient_bytes == 0


def test_codec_tile_scratch_does_not_grow_with_full_window_rows():
    from soup_cli.utils.fp8_activations import fp8_activation_allocation

    allocation = fp8_activation_allocation(
        n_layers=2, mb_shapes=[(1, 1024, 8)], pin=False, device_type="cuda",
    )
    assert allocation.payload_bytes == 16_384
    assert allocation.scale_bytes == 8_192
    assert allocation.host_transient_bytes == 257
    assert allocation.device_transient_bytes == 90_113


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_layers": 0}, {"n_layers": True}, {"mb_shapes": []},
        {"mb_shapes": [(0, 8)]}, {"mb_shapes": [(1, 8), (2, 4)]},
        {"mb_shapes": [(True, 8)]}, {"pin": 1},
        {"compute_dtype": "float64"}, {"device_type": "mps"},
    ],
)
def test_invalid_estimator_inputs_are_refused(overrides):
    from soup_cli.utils.fp8_activations import fp8_activation_allocation

    args = {"n_layers": 2, "mb_shapes": [(1, 3, 5)], "pin": False}
    args.update(overrides)
    with pytest.raises(ValueError):
        fp8_activation_allocation(**args)


def _torch():
    return pytest.importorskip("torch")


def _store(budget=None, **overrides):
    torch = _torch()
    from soup_cli.utils.fp8_activations import FP8ActivationStore

    args = dict(
        host_budget=budget or HostMemoryBudget(1 << 24), n_layers=2, k=2,
        chunk_shape=(4, 5), mb_shapes=[(1, 3, 5), (2, 2, 5)],
        dtype=torch.float32, device="cpu", pin=False,
    )
    args.update(overrides)
    return FP8ActivationStore(**args)


def _reference_fp8(tensor):
    """G4's original operation order, independent of the shared codec entry points."""
    torch = _torch()
    values = tensor.float()
    absmax = values.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    payload = (values / safe * 448).clamp_(-448, 448).to(torch.float8_e4m3fn)
    return (payload.float() / 448 * absmax).to(tensor.dtype)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_shared_fp8_matches_the_original_g4_operation_order(dtype):
    torch = _torch()
    from soup_cli.utils import activation_codec as codec

    tensor = torch.randn(3, 7, 19, generator=torch.Generator().manual_seed(51))
    tensor = tensor.to(getattr(torch, dtype))
    expected = _reference_fp8(tensor)
    assert torch.equal(codec.round_trip(tensor, codec.FP8), expected)
    assert codec.payload_chunk_bytes((1, 512, 4096), codec.FP8) == 2_097_152
    assert codec.scale_chunk_bytes((1, 512, 4096), codec.FP8) == 2048


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
def test_shared_g4_codecs_keep_zero_denormal_and_extreme_absmax_exact(name):
    torch = _torch()
    from soup_cli.utils import activation_codec as codec

    maximum = torch.finfo(torch.bfloat16).max
    tiny = 2.0 ** -133
    tensor = torch.tensor(
        [[0, 0, 0, 0, 0], [maximum, -maximum, 1e30, -1e30, 0],
         [tiny, -tiny, 0, tiny, 0], [-7.25, -7.25, -7.25, -7.25, -7.25]],
        dtype=torch.bfloat16,
    )
    payload, scale = codec.encode(tensor, name)
    restored = codec.decode(payload, scale, name, tensor.dtype, 5)
    assert torch.isfinite(restored).all()
    index = tensor.float().abs().argmax(dim=-1, keepdim=True)
    assert torch.equal(restored.gather(-1, index), tensor.gather(-1, index))
    assert torch.equal(restored[0], torch.zeros_like(restored[0]))
    assert torch.equal(scale, tensor.float().abs().amax(dim=-1))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_shared_fp8_refuses_nonfinite_values(bad):
    torch = _torch()
    from soup_cli.utils.activation_codec import FP8, NonFiniteActivationError, encode

    tensor = torch.zeros((2, 3, 5))
    tensor[1, 2, 4] = bad
    with pytest.raises(NonFiniteActivationError, match="1 of 6 rows"):
        encode(tensor, FP8)


def test_owned_host_storage_and_live_lease_match_the_shared_breakdown():
    budget = HostMemoryBudget(1 << 24, reserved_bytes=37)
    store = _store(budget)
    try:
        assert store.allocation.host_bytes == 975
        assert budget.used_bytes == 1012
        payload = sum(tensor.untyped_storage().nbytes() for tensor in store._payload.values())
        scales = store._scales.untyped_storage().nbytes()
        assert payload == 70
        assert scales == 56
        measured = store.host_bytes()
        assert measured["host_bytes"] == 975
        assert measured["resident_payload_bytes"] == 70
        assert measured["scale_bytes"] == 56
        assert measured["owned_host_storage_bytes"] == 126
        assert measured["owned_device_storage_bytes"] == 0
        assert measured["host_transient_bytes"] == 849
        assert store.stats.resident_chunks == 4
        assert store.stats.spilled_chunks == store.stats.spill_layers == 0
    finally:
        store.close()
    assert budget.used_bytes == 37
    assert store.host_bytes()["host_bytes"] == 0
    assert store.host_bytes()["owned_host_storage_bytes"] == 0


def test_ragged_slots_restore_each_original_shape_and_return_independent_outputs():
    torch = _torch()
    from soup_cli.utils.activation_codec import FP8, round_trip

    store = _store()
    chunks = {
        (0, 0): torch.tensor([[[1., -2., 0., 4., 0.5], [3., 2., 1., 0., -1.]]]),
        (0, 1): torch.arange(15, dtype=torch.float32).reshape(3, 5) / 3,
        (1, 0): torch.full((5,), -7.25),
        (1, 1): torch.arange(20, dtype=torch.float32).reshape(2, 2, 5) / 7,
    }
    try:
        for key, tensor in chunks.items():
            before = tensor.clone()
            store.put(*key, tensor)
            assert torch.equal(tensor, before), "compression must not change the forward input"
        store.begin_backward()
        for key, tensor in chunks.items():
            expected = round_trip(tensor, FP8)
            restored = store.get(*key)
            assert restored.shape == tensor.shape
            assert restored.dtype == tensor.dtype
            assert torch.equal(restored, expected)
            restored.zero_()
            assert torch.equal(store.get(*key), expected)
        store.end_step()
        with pytest.raises(KeyError, match="this step"):
            store.get(0, 0)
    finally:
        store.close()


def test_noncontiguous_input_is_encoded_in_bounded_row_tiles():
    torch = _torch()
    from soup_cli.utils.activation_codec import FP8, round_trip

    store = _store(
        k=1, chunk_shape=(1, 600, 5), mb_shapes=None,
    )
    tensor = torch.arange(3000, dtype=torch.float32).reshape(2, 300, 5).permute(1, 0, 2)
    assert not tensor.is_contiguous()
    with pytest.raises(RuntimeError):
        tensor.view(600, 5)
    try:
        store.put(0, 0, tensor)
        assert torch.equal(store.get(0, 0), round_trip(tensor, FP8))
        store.end_step()
    finally:
        store.close()


def test_short_windows_clear_old_shapes_and_nonfinite_status_between_steps():
    torch = _torch()
    from soup_cli.utils.activation_codec import NonFiniteActivationError

    store = _store(k=4, chunk_shape=(1, 4, 5), mb_shapes=None)
    try:
        bad = torch.ones((1, 4, 5))
        bad[0, 0, 0] = float("nan")
        store.put(1, 3, bad)
        store.begin_backward()
        store.get(1, 3)
        with pytest.raises(NonFiniteActivationError, match=r"\(1, 3\)"):
            store.end_step()
        for _step in range(3):
            tensor = torch.full((1, 2, 5), 2.0)
            store.put(0, 0, tensor)
            store.begin_backward()
            assert torch.equal(store.get(0, 0), tensor)
            with pytest.raises(KeyError, match="this step"):
                store.get(1, 3)
            store.end_step()
            assert not store._shapes
    finally:
        store.close()


def test_a_finite_overwrite_cannot_hide_an_earlier_nonfinite_boundary():
    torch = _torch()
    from soup_cli.utils.activation_codec import NonFiniteActivationError

    store = _store()
    try:
        store.put(0, 0, torch.full((1, 5), float("inf")))
        store.put(0, 0, torch.ones((1, 5)))
        assert torch.equal(store.get(0, 0), torch.ones((1, 5)))
        with pytest.raises(NonFiniteActivationError, match=r"\(0, 0\)"):
            store.end_step()
        store.put(0, 0, torch.ones((1, 5)))
        store.end_step()
    finally:
        store.close()


@pytest.mark.parametrize("shape", [(1, 4, 5), (1, 2, 10), (0, 5)])
def test_put_rejects_capacity_overflow_or_changed_hidden_width(shape):
    torch = _torch()

    store = _store()
    try:
        with pytest.raises(ValueError, match="capacity|hidden|positive"):
            store.put(0, 0, torch.zeros(shape))
        assert not store._shapes
    finally:
        store.close()


@pytest.mark.parametrize("key", [(-1, 0), (2, 0), (0, -1), (0, 2), (True, 0)])
def test_keys_cannot_escape_the_reserved_layer_and_microbatch_slots(key):
    torch = _torch()

    store = _store()
    try:
        with pytest.raises((ValueError, IndexError)):
            store.put(*key, torch.zeros((1, 5)))
        with pytest.raises((ValueError, IndexError)):
            store.get(*key)
    finally:
        store.close()


def test_insufficient_budget_is_rejected_before_the_first_tensor_allocation(monkeypatch):
    torch = _torch()

    budget = HostMemoryBudget(974)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("allocation happened before reservation")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    with pytest.raises(MemoryError, match="needs 975"):
        _store(budget)
    assert budget.used_bytes == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"k": 3}, {"n_layers": True}, {"ring": 0}, {"ring": True},
        {"spill_layers": 1}, {"spill_layers": True},
        {"chunk_shape": (1, 2, 5)}, {"mb_shapes": [(1, 3, 5), (1, 1, 7)]},
    ],
)
def test_invalid_store_layout_leaves_no_reservation(overrides):
    budget = HostMemoryBudget(1 << 24)
    with pytest.raises(ValueError):
        _store(budget, **overrides)
    assert budget.used_bytes == 0


def test_partial_constructor_failure_drops_storage_before_releasing_the_lease(monkeypatch):
    torch = _torch()

    budget = HostMemoryBudget(1 << 24)
    allocated = []
    original = torch.empty

    def fail_scale(*args, **kwargs):
        if kwargs.get("dtype") is torch.float32:
            raise MemoryError("injected scale allocation failure")
        tensor = original(*args, **kwargs)
        allocated.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(torch, "empty", fail_scale)
    with pytest.raises(MemoryError, match="injected scale"):
        _store(budget)
    assert allocated and all(reference() is None for reference in allocated)
    assert budget.used_bytes == 0


def test_compression_exception_preserves_the_lease_until_explicit_close(monkeypatch):
    torch = _torch()
    from soup_cli.utils import activation_codec

    budget = HostMemoryBudget(1 << 24)
    store = _store(budget)
    held = budget.used_bytes
    original = activation_codec.encode

    def fail_encode(*args, **kwargs):
        raise RuntimeError("injected codec failure")

    try:
        monkeypatch.setattr(activation_codec, "encode", fail_encode)
        with pytest.raises(RuntimeError, match="injected codec"):
            store.put(0, 0, torch.ones((1, 5)))
        assert budget.used_bytes == held
        assert not store._shapes
        monkeypatch.setattr(activation_codec, "encode", original)
        store.put(0, 0, torch.ones((1, 5)))
        assert torch.equal(store.get(0, 0), torch.ones((1, 5)))
        store.end_step()
    finally:
        store.close()
    assert budget.used_bytes == 0


def test_decode_exception_tracebacks_do_not_keep_owned_storage_after_close(monkeypatch):
    torch = _torch()
    from soup_cli.utils import activation_codec

    budget = HostMemoryBudget(1 << 24)
    store = _store(budget)
    store.put(0, 0, torch.ones((1, 5)))
    storage_refs = [weakref.ref(tensor) for tensor in store._payload.values()]
    storage_refs.append(weakref.ref(store._scales))

    def fail_decode(*args, **kwargs):
        raise RuntimeError("injected decode failure")

    monkeypatch.setattr(activation_codec, "decode", fail_decode)
    with pytest.raises(RuntimeError, match="injected decode") as held_error:
        store.get(0, 0)
    store.close()
    assert held_error.value.__traceback__ is not None
    assert all(reference() is None for reference in storage_refs)
    assert budget.used_bytes == 0


def test_failed_cuda_drain_retains_storage_and_lease_until_close_retry(monkeypatch):
    torch = _torch()

    budget = HostMemoryBudget(1 << 24)
    store = _store(budget)
    held = budget.used_bytes
    store.is_cuda = True  # Exercise failed cleanup without requiring CUDA hardware.
    store._cuda_used = True

    def failed_sync(device):
        raise RuntimeError("injected CUDA drain failure")

    monkeypatch.setattr(torch.cuda, "synchronize", failed_sync)
    with pytest.raises(RuntimeError, match="injected CUDA drain"):
        store.close()
    assert budget.used_bytes == held
    assert store._payload and store._scales is not None
    with pytest.raises(RuntimeError, match="clos"):
        store.put(0, 0, torch.ones((1, 5)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    store.close()
    store.close()
    assert budget.used_bytes == 0
    assert not store._payload and store._scales is None
    with pytest.raises(RuntimeError, match="closed"):
        store.begin_backward()


@pytest.mark.gpu
def test_pinned_cuda_storage_rounding_and_cross_stream_close():
    torch = _torch()
    from soup_cli.utils.activation_codec import FP8, round_trip

    budget = HostMemoryBudget(1 << 24)
    store = _store(budget, pin=True, device="cuda")
    source_stream = torch.cuda.Stream()
    read_stream = torch.cuda.Stream()
    try:
        assert store.allocation.host_bytes == 165
        assert budget.used_bytes == 165
        assert all(tensor.is_pinned() for tensor in store._payload.values())
        assert store._scales.is_pinned()
        measured = store.host_bytes()
        assert measured["owned_host_storage_bytes"] == 126
        assert measured["pinned_overhead_bytes"] == 34
        with torch.cuda.stream(source_stream):
            tensor = torch.arange(15, dtype=torch.float32, device="cuda").reshape(1, 3, 5)
            expected = round_trip(tensor, FP8)
            store.put(0, 0, tensor)
        with torch.cuda.stream(read_stream):
            restored = store.get(0, 0)
        # close must drain transfers even if no end_step or caller sync happened.
        store.close()
        assert torch.equal(restored.cpu(), expected.cpu())
        assert budget.used_bytes == 0
    finally:
        store.close()


@pytest.mark.gpu
def test_cuda_steps_do_not_retain_codec_tensors_or_timing_events():
    torch = _torch()

    store = _store(
        k=1, chunk_shape=(1, 600, 17), mb_shapes=None, pin=True, device="cuda",
    )
    try:
        for _warmup in range(2):
            tensor = torch.ones((1, 600, 17), device="cuda")
            store.put(0, 0, tensor)
            restored = store.get(0, 0)
            store.end_step()
            del tensor, restored
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        for _step in range(4):
            tensor = torch.ones((1, 257, 17), device="cuda")
            store.put(0, 0, tensor)
            restored = store.get(0, 0)
            assert torch.equal(restored, tensor)
            store.end_step()
            del tensor, restored
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() <= baseline
            assert not store._shapes
            assert len(store._events) == 1
    finally:
        store.close()


@pytest.mark.gpu
def test_cuda_decode_exception_releases_scratch_when_traceback_is_retained(monkeypatch):
    torch = _torch()
    from soup_cli.utils import activation_codec

    budget = HostMemoryBudget(1 << 24)
    store = _store(budget, pin=True, device="cuda")
    tensor = torch.ones((1, 3, 5), device="cuda")
    store.put(0, 0, tensor)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    def fail_decode(*args, **kwargs):
        raise RuntimeError("injected CUDA codec failure")

    monkeypatch.setattr(activation_codec, "decode", fail_decode)
    with pytest.raises(RuntimeError, match="injected CUDA codec") as held_error:
        store.get(0, 0)
    store.close()
    assert held_error.value.__traceback__ is not None
    assert torch.cuda.memory_allocated() <= baseline
    assert budget.used_bytes == 0


@pytest.mark.gpu
def test_cuda_slot_reuse_waits_for_prior_reader_before_overwriting_host_payload():
    torch = _torch()
    from soup_cli.utils.activation_codec import FP8, round_trip

    store = _store(pin=True, device="cuda")
    forward = torch.cuda.Stream()
    backward = torch.cuda.Stream()
    try:
        with torch.cuda.stream(forward):
            first = torch.arange(15, dtype=torch.float32, device="cuda").reshape(1, 3, 5)
            expected = round_trip(first, FP8)
            store.put(0, 0, first)
        with torch.cuda.stream(backward):
            restored = store.get(0, 0)
        with torch.cuda.stream(forward):
            store.put(0, 0, torch.full((1, 3, 5), 2.0, device="cuda"))
            second = store.get(0, 0)
        store.end_step()
        assert torch.equal(restored, expected)
        assert torch.equal(second, torch.full_like(second, 2.0))
    finally:
        store.close()
