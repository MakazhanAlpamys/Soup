"""Measured residency must respect caps and leave training state untouched."""

from types import SimpleNamespace

import pytest

from soup_cli.utils.stream_residency import (
    centered_layer_indices,
    configure_stream_residency,
    plan_centered_layers,
    residency_budget,
)


@pytest.mark.parametrize(
    ("free", "override", "bound", "cap", "available"),
    [(10000, None, "card", 8500, 7100),
     (3000, None, "free", 3000, 1600),
     (10000, 2500, "override", 2500, 1100),
     (10000, 1000, "override", 1000, 0)],
)
def test_shared_cap_accounts_for_reserved_peak_and_missing_optimizer(
    free, override, bound, cap, available,
):
    budget = residency_budget(
        total_bytes=10000, free_bytes=free, reserved_bytes=0, peak_reserved_bytes=1000,
        reserve_bytes=100, optimizer_reserve_bytes=300, override_bytes=override,
    )
    assert (budget.bound, budget.cap_bytes, budget.available_bytes) == (bound, cap, available)


def test_reserved_memory_is_already_owned_and_can_be_reused():
    budget = residency_budget(
        total_bytes=10000, free_bytes=1000, reserved_bytes=2000, peak_reserved_bytes=2500,
        reserve_bytes=100, optimizer_reserve_bytes=0,
    )
    assert budget.cap_bytes == 3000
    assert budget.available_bytes == 400


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_bad_byte_counts_refused(value):
    with pytest.raises(ValueError, match="byte counts"):
        residency_budget(
            total_bytes=10000, free_bytes=value, reserved_bytes=0, peak_reserved_bytes=1,
        )


@pytest.mark.parametrize("value", [-1, 0, True, 1.5])
def test_invalid_override_refused(value):
    with pytest.raises(ValueError, match="override_bytes"):
        residency_budget(
            total_bytes=10000, free_bytes=10000, reserved_bytes=0,
            peak_reserved_bytes=1, override_bytes=value,
        )


def test_selection_charges_actual_layer_sizes_and_allocator_overhead():
    # The centre one weighs 300; the centre pair 500. A 1.5x allocator
    # ratio prevents the nominally fitting 500-byte pair from being selected.
    assert plan_centered_layers([100, 300, 200, 100], 550) == (1, 2)
    assert plan_centered_layers(
        [100, 300, 200, 100], 550, peak_allocated_bytes=100, peak_reserved_bytes=150,
    ) == (1,)
    assert plan_centered_layers([100, 300, 200, 100], 99) == ()
    assert centered_layer_indices(36, 4) == (16, 17, 18, 19)


@pytest.mark.parametrize("count", [-1, True, 7, 1.5])
def test_manual_count_validation(count):
    with pytest.raises(ValueError):
        centered_layer_indices(6, count)


class FakeMemory:
    def __init__(self, free=3000):
        self.free_bytes = free
        self.synchronizations = 0
        self.releases = 0

    def total(self):
        return 10000

    def free(self):
        return self.free_bytes

    def reserved(self):
        return 0

    def synchronize(self):
        self.synchronizations += 1

    def release(self):
        self.releases += 1


class FakePool:
    is_cuda = True
    device = "cpu"  # CPU executes the state test; FakeMemory supplies the VRAM instrument.

    def __init__(self):
        self.resident_slots = {}
        self.clears = 0
        self.failure = None

    def resident_slot_bytes(self, idx):
        return 500

    def reserve_layers(self, indices, source, stream=None):
        if self.failure is not None:
            raise self.failure
        self.resident_slots = dict.fromkeys(indices, 1)
        return {"resident_bytes": len(indices) * 500}

    def clear_resident_layers(self):
        self.resident_slots.clear()
        self.clears += 1


def fixture():
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.2))
    runtime = SimpleNamespace(
        pool=FakePool(), n_layers=6, tier="ram", source=object(),
        prefetcher=SimpleNamespace(stream=None),
    )
    memory = FakeMemory()
    calls = []

    def measure(model, **shape):
        import torch

        from soup_cli.utils.layer_stream_runtime import StepPeak

        calls.append(shape)
        # Exercise RNG consumption, actual backward, and mixed module modes.
        output = model(torch.rand(2, 2))
        output.sum().backward()
        model[0].training = not model[0].training
        count = len(runtime.pool.resident_slots)
        return StepPeak(
            peak_bytes=1000 + count * 500, reserved_bytes=1000 + count * 500,
            seconds=1, rows=shape["rows"], seq_len=shape["seq_len"],
        )

    return model, runtime, memory, calls, measure


def configure(model, runtime, memory, measure, mode="auto", **kwargs):
    return configure_stream_residency(
        model, runtime, mode, rows=2, seq_len=128, vocab_size=10,
        memory=memory, measure=measure, reserve_bytes=100, optimizer_reserve_bytes=300,
        **kwargs,
    )


def test_disabled_is_a_true_no_op_without_model_or_runtime():
    plan = configure_stream_residency(None, None, 0, rows=0, seq_len=0, vocab_size=0)
    assert plan.reason == "disabled"
    assert not plan.enabled


def test_auto_preserves_rng_weights_gradients_and_mixed_training_flags():
    import torch

    model, runtime, memory, calls, measure = fixture()
    model[1].eval()
    initial_rng = torch.get_rng_state().clone()
    initial_values = [param.detach().clone() for param in model.parameters()]
    initial_modes = [module.training for module in model.modules()]
    messages = []
    plan = configure(model, runtime, memory, measure, notify=messages.append)
    assert plan.indices == (1, 2, 3)
    assert plan.selected_indices == plan.indices
    assert plan.enabled
    assert plan.resident_bytes == 1500
    assert plan.candidate_peak_reserved_bytes + 400 <= plan.cap_bytes
    assert plan.optimizer_reserve_bytes == 300
    assert plan.preload_seconds >= 0
    assert plan.setup_seconds >= plan.preload_seconds
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert all(
        torch.equal(before, after) for before, after in zip(initial_values, model.parameters())
    )
    assert all(param.grad is None for param in model.parameters())
    assert [module.training for module in model.modules()] == initial_modes
    assert len(calls) == 2
    assert all(call["rows"] == 2 and call["seq_len"] == 128 for call in calls)
    assert "optimizer state estimated" in messages[0]


def test_auto_does_not_destroy_existing_accumulated_gradients():
    import torch

    model, runtime, memory, calls, measure = fixture()
    param = next(model.parameters())
    param.grad = torch.ones_like(param)
    original = param.grad
    with pytest.raises(ValueError, match="pristine gradients"):
        configure(model, runtime, memory, measure)
    assert param.grad is original
    assert calls == []
    assert runtime.pool.resident_slots == {}


def test_probe_failure_restores_adapter_values_rng_and_modes():
    import torch

    model, runtime, memory, calls, good = fixture()
    values = [param.detach().clone() for param in model.parameters()]
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()

    def measure(model, **shape):
        good(model, **shape)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(3)
        model.eval()
        raise RuntimeError("probe device failure")

    with pytest.raises(RuntimeError, match="probe device failure"):
        configure(model, runtime, memory, measure)
    assert all(torch.equal(before, after) for before, after in zip(values, model.parameters()))
    assert all(param.grad is None for param in model.parameters())
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("reason", ["oom", "cap", "none"])
def test_candidate_failure_releases_private_slots_and_cached_views(reason):
    from soup_cli.utils.layer_stream_runtime import StepPeak

    model, runtime, memory, calls, good = fixture()
    model._cached_substitution_buffers = {"fake": object()}
    model._cached_substitution_weights = {"fake": object()}

    def measure(model, **shape):
        peak = good(model, **shape)
        if len(calls) == 1:
            return peak
        model._cached_substitution_buffers = {"private": object()}
        model._cached_substitution_weights = {"private": object()}
        if reason == "none":
            return None
        return StepPeak(
            peak_bytes=4000, reserved_bytes=4000, seconds=1,
            rows=shape["rows"], seq_len=shape["seq_len"], oom=reason == "oom",
        )

    plan = configure(model, runtime, memory, measure)
    assert not plan.enabled
    assert runtime.pool.resident_slots == {}
    assert model._cached_substitution_buffers is None
    assert model._cached_substitution_weights is None
    assert memory.releases >= 1


def test_allocation_oom_falls_back_but_other_errors_propagate():
    import torch

    model, runtime, memory, calls, measure = fixture()
    runtime.pool.failure = torch.OutOfMemoryError("out of memory")
    plan = configure(model, runtime, memory, measure)
    assert not plan.enabled
    assert "out of memory" in plan.reason
    assert runtime.pool.resident_slots == {}
    runtime.pool.failure = RuntimeError("owner corruption")
    with pytest.raises(RuntimeError, match="owner corruption"):
        configure(model, runtime, memory, measure)


def test_reported_non_oom_probe_failure_never_becomes_a_safe_fallback():
    from soup_cli.utils.layer_stream_runtime import StepPeak

    model, runtime, memory, calls, good = fixture()

    def measure(model, **shape):
        if not calls:
            return good(model, **shape)
        return StepPeak(
            peak_bytes=0, reserved_bytes=0, seconds=1, rows=2, seq_len=128,
            failed=True, error="AcceleratorError",
        )

    with pytest.raises(RuntimeError, match="AcceleratorError"):
        configure(model, runtime, memory, measure)
    assert runtime.pool.resident_slots == {}


@pytest.mark.parametrize("tier", ["disk", "unknown"])
def test_auto_unsupported_tier_falls_back_and_manual_refuses(tier):
    model, runtime, memory, calls, measure = fixture()
    runtime.tier = tier
    assert not configure(model, runtime, memory, measure).enabled
    with pytest.raises(ValueError, match="RAM tier"):
        configure(model, runtime, memory, measure, mode=1)
    assert calls == []


def test_no_complete_layer_fits_and_manual_never_silently_reduces_count():
    model, runtime, memory, calls, measure = fixture()
    memory.free_bytes = 1400
    plan = configure(model, runtime, memory, measure)
    assert not plan.enabled
    assert "no complete decoder" in plan.reason
    with pytest.raises(ValueError, match="exceed"):
        configure(model, runtime, memory, measure, mode=2)
    assert runtime.pool.resident_slots == {}


def test_default_optimizer_reserve_counts_only_trainable_parameters():
    import torch

    model, runtime, memory, calls, measure = fixture()
    model.register_parameter("frozen", torch.nn.Parameter(torch.zeros(100), requires_grad=False))
    plan = configure_stream_residency(
        model, runtime, "auto", rows=2, seq_len=128, vocab_size=10,
        memory=memory, measure=measure, reserve_bytes=100,
    )
    assert plan.optimizer_reserve_bytes == 6 * 12
