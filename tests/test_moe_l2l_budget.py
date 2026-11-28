"""CPU checks for the MoE L2L host-memory plan."""

import pytest

from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.moe_l2l_budget import plan_moe_l2l

PLAN_ARGS = {
    "n_layers": 3,
    "n_moe_layers": 2,
    "max_shape": (2, 5, 7),
    "experts_per_token": 3,
    "requested_k": 4,
    "dtype": "bfloat16",
    "pin": False,
}


@pytest.mark.parametrize(
    ("dtype", "pin", "expected_activations"),
    [
        ("bfloat16", False, 1680),
        ("float32", False, 3360),
        ("bfloat16", True, 3072),
        ("float32", True, 6144),
    ],
)
def test_exact_fit_keeps_requested_window(dtype, pin, expected_activations):
    # Each routed layer retains 10 tokens x 3 int64 expert indices per micro-batch.
    # All eight records can be live while one more is copied for the first comparison.
    args = {**PLAN_ARGS, "dtype": dtype, "pin": pin}
    budget = HostMemoryBudget(expected_activations + 2160)

    plan = plan_moe_l2l(budget, **args)

    assert plan.effective_k == plan.requested_k == 4
    assert plan.activation_bytes == expected_activations
    assert plan.routing_record_bytes == 1920
    assert plan.routing_comparison_bytes == 240
    assert plan.host_bytes == plan.available_host_bytes == budget.limit_bytes
    assert plan.saved_expert_input_bytes is None
    assert budget.used_bytes == 0


def test_expert_input_dtype_is_independent_of_host_storage():
    budget = HostMemoryBudget(3840)
    plan = plan_moe_l2l(budget, expert_input_dtype="float32", **PLAN_ARGS)

    assert plan.saved_expert_input_bytes == 840  # 10 tokens x 3 routes x 7 values x 4 B
    assert plan.effective_k == 4
    assert plan.host_bytes == 3840


@pytest.mark.parametrize(("pin", "oracle_bytes"), [(False, 1760), (True, 2048)])
def test_oracle_slabs_are_charged_in_addition_to_mbr_records(pin, oracle_bytes):
    args = {**PLAN_ARGS, "pin": pin}
    baseline = plan_moe_l2l(HostMemoryBudget(100_000), **args)
    limit = baseline.host_bytes + oracle_bytes
    plan = plan_moe_l2l(HostMemoryBudget(limit), **args, routing_oracle=True, num_experts=8)
    assert plan.effective_k == 4
    assert plan.host_bytes == limit
    assert plan.routing_record_bytes == baseline.routing_record_bytes
    assert plan.routing_oracle_bytes == oracle_bytes
    short = plan_moe_l2l(HostMemoryBudget(limit - 1), **args, routing_oracle=True, num_experts=8)
    assert short.effective_k == 3


def test_block64_plan_charges_padding_and_oracle_at_exact_fit():
    args = {
        **PLAN_ARGS, "n_layers": 2, "n_moe_layers": 1, "max_shape": (1, 3, 65),
        "experts_per_token": 2, "requested_k": 2, "codec": "fp8_e4m3_block64",
        "device_type": "cpu", "routing_oracle": True, "num_experts": 8,
    }
    # 12 rows of 128 payload bytes and 2 float32 scales. Scratch is 12,871 bytes.
    # The routing plan adds 144 bytes of route records/scratch and 96 oracle bytes.
    limit = 14_743
    plan = plan_moe_l2l(HostMemoryBudget(limit), **args)
    assert plan.effective_k == 2
    assert plan.activation_bytes == 14_503
    assert plan.routing_oracle_bytes == 96
    assert plan.host_bytes == limit
    assert plan_moe_l2l(HostMemoryBudget(limit - 1), **args).effective_k == 1


def test_block64_plan_requires_oracle():
    with pytest.raises(ValueError) as caught:
        plan_moe_l2l(HostMemoryBudget(100_000), **PLAN_ARGS, codec="fp8_e4m3_block64")
    assert str(caught.value) == (
        "training.l2l_activation_codec='fp8_e4m3_block64' requires routing_oracle"
    )


def test_selection_uses_remaining_budget_and_all_integer_window_sizes():
    budget = HostMemoryBudget(40_000, reserved_bytes=961)
    staging = budget.reserve("reader staging", 4600)
    args = {**PLAN_ARGS, "requested_k": 64}

    plan = plan_moe_l2l(budget, **args)

    # 38 micro-batches need 34,440 bytes. The existing consumers leave one byte less.
    assert plan.requested_k == 64
    assert plan.effective_k == 37
    assert plan.available_host_bytes == 34_439
    assert plan.host_bytes == 33_540
    assert budget.used_bytes == 5561
    staging.close()
    assert plan_moe_l2l(budget, **args).effective_k == 43
    assert plan_moe_l2l(budget, **PLAN_ARGS).effective_k == 4


@pytest.mark.parametrize("available", [0, 1139])
def test_rejects_budget_too_small_for_one_microbatch(available):
    budget = HostMemoryBudget(1140, reserved_bytes=1140 - available)

    with pytest.raises(MemoryError, match=rf"k=1 needs 1140 host bytes.*{available} available"):
        plan_moe_l2l(budget, **PLAN_ARGS)

    assert budget.available_bytes == available


@pytest.mark.parametrize(
    "invalid",
    [
        {"requested_k": True},
        {"max_shape": (2, 5)},
        {"max_shape": (2, 0, 7)},
        {"n_moe_layers": 4},
        {"experts_per_token": 0},
        {"dtype": "float16"},
        {"expert_input_dtype": "uint8"},
    ],
)
def test_rejects_unsupported_dimensions_and_dtype(invalid):
    with pytest.raises(ValueError):
        plan_moe_l2l(HostMemoryBudget(10_000), **{**PLAN_ARGS, **invalid})


def test_plan_covers_cpu_store_and_route_tensors_under_one_budget():
    import torch

    from soup_cli.utils.budgeted_activations import BudgetedActivationStore

    budget = HostMemoryBudget(3840)
    plan = plan_moe_l2l(budget, **PLAN_ARGS)
    routing = budget.reserve(
        "routing", plan.routing_record_bytes + plan.routing_comparison_bytes
    )
    records = comparison = None
    try:
        records = torch.empty((plan.effective_k, 2, 10, 3), dtype=torch.int64)
        comparison = torch.empty((10, 3), dtype=torch.int64)
        assert records.untyped_storage().nbytes() == plan.routing_record_bytes
        assert comparison.untyped_storage().nbytes() == plan.routing_comparison_bytes
        store = BudgetedActivationStore(
            host_budget=budget,
            n_layers=3,
            k=plan.effective_k,
            chunk_shape=(2, 5, 7),
            dtype=torch.bfloat16,
            device="cpu",
            pin=False,
        )
        try:
            activation = torch.ones((2, 5, 7), dtype=torch.bfloat16)
            store.put(2, 3, activation)
            assert torch.equal(store.get(2, 3), activation)
            assert budget.used_bytes == plan.host_bytes
            assert budget.available_bytes == 0
        finally:
            store.close()
    finally:
        del records, comparison
        routing.close()
    assert budget.used_bytes == 0


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_fp8_plan_selects_largest_fitting_window_including_scratch(device_type):
    from soup_cli.utils.fp8_activations import fp8_activation_allocation

    allocation = fp8_activation_allocation(
        n_layers=3, mb_shapes=[(2, 5, 7)] * 4, pin=False,
        compute_dtype="bfloat16", device_type=device_type,
    )
    exact_bytes = allocation.host_bytes + 4 * 2 * 10 * 3 * 8 + 10 * 3 * 8
    args = {
        **PLAN_ARGS, "requested_k": 64, "codec": "fp8_e4m3", "device_type": device_type,
    }
    budget = HostMemoryBudget(exact_bytes + 100, reserved_bytes=100)
    plan = plan_moe_l2l(budget, **args)
    assert plan.effective_k == 4
    assert plan.host_bytes == exact_bytes
    assert plan.activation_bytes == allocation.host_bytes
    assert budget.used_bytes == 100
    smaller = plan_moe_l2l(HostMemoryBudget(exact_bytes - 1), **args)
    assert smaller.effective_k == 3


def test_fp8_plan_reserves_real_variable_shape_store_and_releases_it():
    import torch

    from soup_cli.utils.fp8_activations import FP8ActivationStore

    budget = HostMemoryBudget(100_000)
    plan = plan_moe_l2l(budget, **PLAN_ARGS, codec="fp8_e4m3", device_type="cpu")
    shapes = [(2, 5, 7), (1, 2, 7), (1, 4, 7), (2, 3, 7)]
    plan.validate_microbatches(
        n_layers=3, shapes=shapes, dtype="bfloat16", pin=False, codec="fp8_e4m3"
    )
    routing = budget.reserve(
        "routing", plan.routing_record_bytes + plan.routing_comparison_bytes
    )
    store = FP8ActivationStore(
        host_budget=budget, n_layers=3, k=4, chunk_shape=(2, 5, 7),
        mb_shapes=shapes, dtype=torch.bfloat16, device="cpu", pin=False,
    )
    try:
        for mb, shape in enumerate(shapes):
            value = torch.ones(shape, dtype=torch.bfloat16)
            store.put(2, mb, value)
            assert torch.equal(store.get(2, mb), value)
        assert budget.used_bytes <= plan.host_bytes
        store.end_step()
    finally:
        store.close()
        routing.close()
    assert budget.used_bytes == 0


@pytest.mark.parametrize(
    "change",
    [
        {"shapes": [(2, 5, 7)] * 5},
        {"shapes": [(2, 6, 7)]},
        {"shapes": [(2, 5, 8)]},
        {"dtype": "float32"},
        {"codec": "none"},
        {"pin": True},
        {"n_layers": 4},
    ],
)
def test_memory_plan_refuses_out_of_envelope_window(change):
    plan = plan_moe_l2l(
        HostMemoryBudget(100_000), **PLAN_ARGS, codec="fp8_e4m3", device_type="cpu"
    )
    window = dict(
        n_layers=3, shapes=[(2, 5, 7)], dtype="bfloat16", pin=False, codec="fp8_e4m3"
    )
    with pytest.raises(ValueError, match="plan"):
        plan.validate_microbatches(**{**window, **change})


@pytest.mark.parametrize("device_type,pin,minimum", [("cpu", False, 237), ("cuda", True, 42)])
def test_fp8_minimum_includes_scratch_and_routing_before_admission(device_type, pin, minimum):
    # Two 5-byte payloads, 8 scale bytes; CPU BF16 scratch is 203 bytes.
    # CUDA pins the payloads to 8 bytes each and reserves 2 host scratch bytes.
    # One 8-byte route record and one comparison are required on both devices.
    options = dict(
        n_layers=2, n_moe_layers=1, max_shape=(1, 1, 5), experts_per_token=1,
        requested_k=64, dtype="bfloat16", pin=pin, codec="fp8_e4m3",
        device_type=device_type,
    )
    plan = plan_moe_l2l(HostMemoryBudget(minimum), **options)
    assert plan.effective_k == 1
    assert plan.host_bytes == minimum
    with pytest.raises(MemoryError, match="k=1 needs"):
        plan_moe_l2l(HostMemoryBudget(minimum - 1), **options)


@pytest.mark.parametrize("limit,expected_k", [(113, 2), (114, 3)])
def test_fp8_scale_slab_rounding_changes_the_admitted_window(limit, expected_k):
    # k=3: six 5-byte payloads round to 48; the 24-byte scale slab rounds to 32.
    # Add 2 host scratch bytes, 24 route-record bytes and an 8-byte comparison.
    plan = plan_moe_l2l(
        HostMemoryBudget(limit), n_layers=2, n_moe_layers=1, max_shape=(1, 1, 5),
        experts_per_token=1, requested_k=64, dtype="bfloat16", pin=True,
        codec="fp8_e4m3", device_type="cuda",
    )
    assert plan.effective_k == expected_k
    assert plan.host_bytes == (74 if expected_k == 2 else 114)
