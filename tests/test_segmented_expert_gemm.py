"""Segmented expert calls use the current MBR capture and retain their own VJP geometry."""

from collections import Counter

import pytest

from tests.test_l2l_mbr import (
    LEVERS,
    MBR,
    PREREQUISITES,
    _assert_same,
    _build,
    _capture_window,
    _load,
    _run,
)

MODE = "segmented"


@pytest.mark.parametrize(
    "override",
    [{"l2l": False}, {"stream_expert_layout": "fused"}, {MBR: False}],
)
def test_segmented_config_requires_l2l_per_expert_and_mbr(override):
    options = {**PREREQUISITES, MBR: True, "stream_expert_gemm": MODE, **override}
    with pytest.raises(ValueError) as error:
        _load(**options)
    assert str(error.value) == (
        ": Value error, training.stream_expert_gemm='segmented' requires training.l2l=true, "
        "training.stream_expert_layout='per_expert' and training.l2l_route_preserving_mbr=true"
    )


@pytest.mark.parametrize("backend", ["fused_nf4", "triton"])
def test_segmented_config_refuses_non_native_group_backends(backend):
    with pytest.raises(ValueError) as error:
        _load(
            **PREREQUISITES, **{MBR: True}, stream_expert_gemm=MODE, quantization="4bit",
            stream_experts={"backend": backend, "native_rows_from": None},
        )
    assert str(error.value) == (
        ": Value error, training.stream_expert_gemm='segmented' requires "
        "training.stream_experts.backend='sequential' or CPU 'grouped'"
    )


def test_segmented_cuda_grouped_is_refused_during_backend_validation():
    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils.expert_training import validate_expert_training

    with pytest.raises(ValueError) as error:
        validate_expert_training(
            ExpertStreamConfig(), device="cuda", dtype="bfloat16", expert_gemm=MODE,
        )
    assert str(error.value) == (
        "training.stream_expert_gemm='segmented' requires "
        "training.stream_experts.backend='sequential' or CPU 'grouped'"
    )


def test_segmented_mode_is_recorded_in_recipe(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe, validate_resume, write_recipe

    monkeypatch.chdir(tmp_path)
    options = {**PREREQUISITES, MBR: True}
    native = resolve_l2l_recipe(_load(**options))
    segmented = resolve_l2l_recipe(_load(**options, stream_expert_gemm=MODE))
    assert native.expert_gemm == "native"
    assert segmented.expert_gemm == MODE
    write_recipe(tmp_path, segmented)
    validate_resume(tmp_path, segmented)
    with pytest.raises(ValueError) as error:
        validate_resume(tmp_path, native)
    assert str(error.value) == (
        "cannot resume: the checkpoint was trained with expert_gemm='segmented', but this "
        "run has expert_gemm='native'. Restore the old value in the config."
    )


@pytest.mark.parametrize("missing", ["pool", "two_reads", "grouped_expert_routes"])
def test_segmented_experts_require_pooled_two_read_grouped_routes(tmp_path, missing):
    pytest.importorskip("torch")
    from tests.test_streamed_experts_pool import _experts, _store, _streamed

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        with pytest.raises(ValueError) as error:
            _streamed(
                module, reader, None if missing == "pool" else pool,
                two_reads=missing != "two_reads",
                grouped_expert_routes=missing != "grouped_expert_routes", expert_gemm=MODE,
            )
        assert str(error.value) == (
            "training.stream_expert_gemm='segmented' requires pooled experts, "
            "two_reads=True and grouped_expert_routes=True"
        )
        assert pool.fetches == 0
    finally:
        pool.close()
        reader.close()


@pytest.mark.parametrize("sizes", [(), (0, 6), (-1, 7), (2, 3), (True, 5), (6.0,)])
def test_invalid_group_geometry_fails_before_reads(tmp_path, sizes):
    pytest.importorskip("torch")
    from tests.test_streamed_experts_pool import _experts, _inputs, _store, _streamed

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        experts = _streamed(
            module, reader, pool, two_reads=True, grouped_expert_routes=True, expert_gemm=MODE,
        )
        hidden, indices, weights, _ = _inputs(tokens=6)
        with pytest.raises(ValueError) as error:
            experts(hidden, indices, weights, group_sizes=sizes)
        assert str(error.value) == "group_sizes must be positive integers covering all 6 rows"
        assert pool.fetches == reader.bytes_read == 0
    finally:
        pool.close()
        reader.close()


def test_segmented_backward_retains_its_plan_across_same_layer_calls(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from soup_cli.utils import direct_expert_backward
    from tests.test_streamed_experts_pool import _experts, _inputs, _store, _streamed

    module = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        experts = _streamed(
            module, reader, pool, two_reads=True, grouped_expert_routes=True, expert_gemm=MODE,
        )
        hidden, indices, weights, grad = _inputs(tokens=6, spread=1, top_k=1)
        hidden.requires_grad_(True)
        weights.requires_grad_(True)
        original = direct_expert_backward.expert_vjp
        vjp_rows = []

        def record(rows, *args, **kwargs):
            vjp_rows.append(len(rows))
            return original(rows, *args, **kwargs)

        monkeypatch.setattr(direct_expert_backward, "expert_vjp", record)
        first = experts(hidden, indices, weights, group_sizes=[2, 4])
        later = experts(hidden, indices, weights, group_sizes=[3, 3])
        ordinary = experts(hidden, indices, weights)
        ordinary.backward(grad)
        assert vjp_rows == []
        later.backward(grad)
        first.backward(grad, retain_graph=True)
        first.backward(grad)
        assert vjp_rows == [3, 3, 2, 4, 2, 4]
        assert pool.fetches == 7
        with pytest.raises(RuntimeError) as error:
            torch.autograd.grad(
                experts(hidden, indices, weights, group_sizes=[4, 2]).sum(), hidden,
                create_graph=True,
            )
        assert str(error.value) == "expert streaming does not support higher-order gradients"
    finally:
        pool.close()
        reader.close()


@pytest.mark.parametrize(
    "bf16,grouped,ordered", [(False, False, False), (True, True, False), (False, True, True)],
)
def test_segmented_mbr_preserves_native_shapes_and_cpu_bits(
    tmp_path, monkeypatch, bf16, grouped, ordered,
):
    torch = pytest.importorskip("torch")
    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils import direct_expert_backward, expert_stream, l2l_step

    options = dict(LEVERS["pool-all-levers"], expert_gemm=MODE)
    if grouped:
        options["stream_experts"] = ExpertStreamConfig()
    model, runtime = _build(tmp_path, **options)
    try:
        window = _capture_window(model, runtime)
        stepper = window[1]
        cls = expert_stream.streamed_experts_class()
        calls = []
        in_experts = False
        original = cls.forward

        def forward(self, hidden, indices, weights, *, group_sizes=None):
            nonlocal in_experts
            if torch.is_grad_enabled():
                calls.append((self.layer_idx, None if group_sizes is None else tuple(group_sizes)))
            in_experts = True
            try:
                return original(self, hidden, indices, weights, group_sizes=group_sizes)
            finally:
                in_experts = False

        monkeypatch.setattr(cls, "forward", forward)
        original_vjp = direct_expert_backward.expert_vjp
        vjp_rows = []

        def record_vjp(rows, *args, **kwargs):
            vjp_rows.append(len(rows))
            return original_vjp(rows, *args, **kwargs)

        monkeypatch.setattr(direct_expert_backward, "expert_vjp", record_vjp)
        split_groups = []
        original_split = l2l_step._split_group_backward

        def record_split(*args):
            split_groups.append(True)
            return original_split(*args)

        monkeypatch.setattr(l2l_step, "_split_group_backward", record_split)
        linear_rows = []
        original_linear = torch.nn.functional.linear

        def record_linear(rows, weight, bias=None):
            if in_experts:
                linear_rows.append(len(rows))
            return original_linear(rows, weight, bias)

        # Both the expert owner and the grouped eager backend call native linear.
        monkeypatch.setattr(torch.nn.functional, "linear", record_linear)
        for expert in model.modules():
            if isinstance(expert, cls):
                monkeypatch.setattr(expert, "linear", record_linear)

        reference = _run(window, bf16)
        baseline_rows = Counter(linear_rows)
        baseline_reads = stepper.counters.expert_fetches
        assert baseline_rows and not vjp_rows
        calls.clear()
        linear_rows.clear()

        actual = _run(
            window, bf16, route_preserving_mbr=True, mbr_group_size=3,
            mbr_ordered_backward=ordered, collect_phase_stats=True,
        )
        _assert_same(actual, reference)
        assert Counter(linear_rows) == baseline_rows
        assert Counter({rows: count * 2 for rows, count in Counter(vjp_rows).items()}) == (
            baseline_rows
        )
        assert len(split_groups) == (0 if ordered else 4)
        per_layer = [(5, 3), (3, 6, 4)] if ordered else [(3, 6, 4), (5, 3)]
        assert all([sizes for index, sizes in calls if index == layer] == per_layer
                   for layer in (1, 2))
        for layer in (1, 2):
            stats = stepper.route_preserving_mbr_stats["layers"][layer]
            assert stats["recompute_forward"]["expert_fetches"] == 0
        assert 0 < stepper.counters.expert_fetches < baseline_reads
        if grouped:
            assert runtime.expert_pool.groups_loaded > 0
    finally:
        runtime.close()
