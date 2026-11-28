"""Research selection and checkpoint correctness before any CUDA experiment."""

import pytest


def test_capacity_selection_preserves_explicit_workspace_reserve():
    from benchmarks.harness.streaming_residency_compile_bench import select_resident_layers

    assert select_resident_layers(32, 100, 300, 1000, 200) == [13, 14, 15, 16, 17]
    assert select_resident_layers(32, 100, 900, 1000, 200) == []
    assert len(select_resident_layers(8, 1, 0, 1000, 0)) == 4
    with pytest.raises(ValueError):
        select_resident_layers(32, 0, 100, 1000, 0)


def test_selected_compilation_preserves_checkpointed_updates():
    import torch

    from benchmarks.harness.streaming_residency_candidate import install_persistent_layers
    from benchmarks.harness.streaming_residency_compile_bench import compile_resident_layers
    from tests.test_streaming_residency_candidate import fixture, trajectory

    reference, native = fixture()
    candidate, runtime = fixture()
    install_persistent_layers(native, [2, 3])
    install_persistent_layers(runtime, [2, 3])
    assert compile_resident_layers(candidate, [2, 3], backend="aot_eager") == 2
    assert not getattr(candidate[1].inner, "_soup_resident_compiled", False)
    expected = trajectory(reference, native)
    actual = trajectory(candidate, runtime)
    assert len(actual) == len(expected)
    assert all(torch.equal(first, second) for first, second in zip(expected, actual, strict=True))
    assert runtime.pool.n == 2
    assert len(runtime.pool.buffers) == 4
    with pytest.raises(ValueError, match="already compiled"):
        compile_resident_layers(candidate, [2, 3], backend="aot_eager")


def test_compilation_refuses_rotating_layers_before_mutation():
    from benchmarks.harness.streaming_residency_compile_bench import compile_resident_layers
    from tests.test_streaming_residency_candidate import fixture

    model, _runtime = fixture()
    with pytest.raises(ValueError, match="not resident"):
        compile_resident_layers(model, [2], backend="aot_eager")
    assert not any(getattr(layer.inner, "_soup_resident_compiled", False) for layer in model)
