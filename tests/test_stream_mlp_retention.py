"""MLP retention is a separate experiment with exactness and memory gates."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _helper():
    directory = Path(__file__).parents[1] / "benchmarks/harness"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(
        "stream_mlp_retention", directory / "stream_mlp_retention.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _census():
    return dict(measured=True, k=11, full_layer_charge=482015738,
                budget_bytes=5756682240, peak_reserved=2833252352,
                peak_allocated=2468564480, retained_indices=list(range(21, 32)))


def test_fixed_candidate_a_has_no_memory_double_charge():
    plan = _helper().candidate_plan(_census())
    assert plan["effective_k"] == 8
    assert plan["retained_indices"] == list(range(24, 32))
    assert plan["qkvo_indices"] == list(range(24))
    assert plan["mlp_indices"] == list(range(19, 24))
    assert plan["raw_bytes"] == 1520 * 2**20
    assert plan["charged_bytes"] == 1829297278
    assert plan["slack_bytes"] == 71259058
    control = _helper().candidate_plan(_census(), include_mlp=False)
    assert control["effective_k"] == 8 and control["mlp_indices"] == []
    assert control["raw_bytes"] == 24 * 41943040


@pytest.mark.parametrize("change", [dict(k=10), dict(measured=False),
                                   dict(budget_bytes=5656682240),
                                   dict(retained_indices=list(range(22, 32)))])
def test_changed_native_census_is_not_an_adaptive_selection(change):
    with pytest.raises(ValueError, match="census"):
        _helper().candidate_plan(_census() | change)


@pytest.mark.parametrize("quant", ["none", "nf4"])
def test_mixed_mlp_cache_matches_native_with_recycled_slots_and_two_updates(tmp_path, quant):
    torch = pytest.importorskip("torch")
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute, streamed_decoder_layers
    from tests.test_stream_projection_retention import _mistral_stream

    helper = _helper()
    records = []
    for arm in ("native", "qkvo", "mixed"):
        model, runtime = _mistral_stream(tmp_path / arm, quant)
        controller = observer = None
        try:
            set_stream_recompute(model, False, retain=2 if arm == "native" else 1)
            layers = streamed_decoder_layers(model)
            original = [layer.forward for layer in layers]
            if arm != "native":
                controller = helper.install(model, [0, 1, 2], [1, 2] if arm == "mixed" else [])
                assert layers[3].forward == original[3]
            observer = helper.ProjectionObserver(model)
            parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
            optimizer = torch.optim.AdamW(parameters.values(), lr=1e-4)
            snapshots = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(5))
                with observer.phase("forward"):
                    output = model(input_ids=ids, labels=ids)
                if controller:
                    expected = 3 * 8 * 4 * (64 + 32 + 32 + 64)
                    expected += (2 * 2 * 8 * 4 * 128) if arm == "mixed" else 0
                    assert controller.cache_bytes() == expected
                    assert len(controller.cached_tensors()) == (16 if arm == "mixed" else 12)
                with observer.phase("backward"):
                    output.loss.backward()
                gradients = {n: p.grad.clone() for n, p in parameters.items()}
                assert all(bool(g.abs().sum() > 0) for g in gradients.values())
                optimizer.step()
                updated = {n: p.detach().clone() for n, p in parameters.items()}
                snapshots.append((output.logits.detach().clone(), output.loss.detach().clone(),
                                  gradients, updated))
            records.append(snapshots)
            observer.remove()
            if controller:
                controller.remove()
            assert [layer.forward for layer in layers] == original
        finally:
            if observer:
                observer.remove()
            if controller:
                controller.remove()
            runtime.close()
    for candidate in records[1:]:
        for expected, actual in zip(records[0], candidate):
            assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])
            for left, right in zip(expected[2:], actual[2:]):
                assert left.keys() == right.keys()
                assert all(torch.equal(left[name], right[name]) for name in left)


def test_bad_rosters_fail_before_patch_and_mode_change_fails_at_use(tmp_path):
    torch = pytest.importorskip("torch")
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute
    from tests.test_stream_projection_retention import _mistral_stream

    helper = _helper()
    model, runtime = _mistral_stream(tmp_path, "none")
    try:
        set_stream_recompute(model, False, retain=1)
        for qkvo, mlp in [([0, 3], []), ([0, 0], []), ([0], [1]), ([8], [])]:
            with pytest.raises(ValueError):
                helper.install(model, qkvo, mlp)
        controller = helper.install(model, [0, 1, 2], [2])
        set_stream_recompute(model, False, retain=4)
        with pytest.raises(RuntimeError, match="mode changed"):
            model(input_ids=torch.ones(1, 8, dtype=torch.long))
        controller.remove()
    finally:
        runtime.close()


def test_mutated_mlp_cache_is_refused(tmp_path):
    torch = pytest.importorskip("torch")
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute
    from tests.test_stream_projection_retention import _mistral_stream

    model, runtime = _mistral_stream(tmp_path, "none")
    controller = None
    try:
        set_stream_recompute(model, False, retain=1)
        controller = _helper().install(model, [0, 1, 2], [2])
        output = model(input_ids=torch.ones(1, 8, dtype=torch.long))
        tensors = [t for t in controller.cached_tensors() if t.shape[-1] == 128]
        assert len(tensors) == 2
        with torch.no_grad():
            tensors[0].add_(1)
        with pytest.raises(RuntimeError, match="mutat"):
            output.logits.square().sum().backward()
    finally:
        if controller:
            controller.remove()
        runtime.close()


def test_profile_counts_innermost_mm_only_and_never_gradient_mm_as_recompute():
    helper = _helper()
    scope = SimpleNamespace(name="soup2162|backward|19|gate_proj", cpu_parent=None)
    outer = SimpleNamespace(name="aten::mm", cpu_parent=scope, cpu_children=[],
                            device_time_total=7.0, self_cpu_time_total=2.0)
    inner = SimpleNamespace(name="aten::mm", cpu_parent=outer, cpu_children=[],
                            device_time_total=7.0, self_cpu_time_total=1.0)
    outer.cpu_children = [inner]
    gradient = SimpleNamespace(name="aten::mm", cpu_parent=None, cpu_children=[],
                               device_time_total=10.0, self_cpu_time_total=1.0)
    hit = SimpleNamespace(name="aten::mm", cpu_parent=scope,
                          cpu_children=[SimpleNamespace(name="PythonDispatchMode")],
                          device_time_total=0.0, self_cpu_time_total=1.0)
    rows = helper.summarize_events([outer, inner, gradient, hit])
    assert rows == [{"phase": "backward", "layer": 19, "projection": "gate_proj",
                     "op": "aten::mm", "count": 1, "device_us": 7.0, "self_cpu_us": 1.0}]


def test_real_profile_attributes_gate_up_recompute_without_double_counting(tmp_path):
    torch = pytest.importorskip("torch")
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute
    from tests.test_stream_projection_retention import _mistral_stream

    helper = _helper()
    counts = []
    for arm in ("qkvo", "mixed"):
        model, runtime = _mistral_stream(tmp_path / arm, "none")
        set_stream_recompute(model, False, retain=1)
        controller = helper.install(model, [0, 1, 2], [1, 2] if arm == "mixed" else [])
        observer = helper.ProjectionObserver(model)
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                with observer.phase("forward"):
                    output = model(input_ids=torch.ones(1, 8, dtype=torch.long))
                with observer.phase("backward"):
                    output.logits.square().mean().backward()
            rows = helper.summarize_events(trace.events())
            counts.append(sum(row["count"] for row in rows if row["phase"] == "backward"
                              and row["projection"] in ("gate_proj", "up_proj")))
        finally:
            observer.remove()
            controller.remove()
            runtime.close()
    assert counts == [6, 2]
