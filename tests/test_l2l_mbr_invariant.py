"""Compare grouped MBR with per-micro-batch L2L on the same expert GEMM."""

import json
from types import SimpleNamespace

import pytest

from tests.test_invariant_gemm import rowwise_dot, rowwise_matmul
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

KEY = "stream_expert_gemm"
MODE = "batch_invariant"


@pytest.mark.parametrize(
    ("overrides", "gate"),
    [
        ({"l2l": False}, "training.l2l=true"),
        ({"stream_expert_layout": "fused"}, "training.stream_expert_layout"),
        ({MBR: False}, f"training.{MBR}=true"),
    ],
)
def test_batch_invariant_gemm_requires_l2l_per_expert_mbr(overrides, gate):
    with pytest.raises(ValueError) as caught:
        _load(**{**PREREQUISITES, MBR: True, KEY: MODE, **overrides})
    assert f"training.{KEY}" in str(caught.value)
    assert gate in str(caught.value)


@pytest.mark.parametrize("value", [True, False, 0, 1, "triton"])
def test_expert_gemm_refuses_values_outside_the_enum(value):
    with pytest.raises(ValueError, match=KEY):
        _load(**{**PREREQUISITES, MBR: True, KEY: value})


def test_expert_gemm_recipe_round_trip_and_resume(tmp_path, monkeypatch):
    # The recipe writer keeps its file under the working directory.
    monkeypatch.chdir(tmp_path)
    from soup_cli.utils.l2l_recipe import (
        RECIPE_NAME,
        resolve_l2l_recipe,
        validate_resume,
        write_recipe,
    )

    options = {**PREREQUISITES, MBR: True}
    native = resolve_l2l_recipe(_load(**options))
    invariant = resolve_l2l_recipe(_load(**{**options, KEY: MODE}))
    path = tmp_path / RECIPE_NAME
    write_recipe(tmp_path, invariant)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["expert_gemm"] == MODE
    validate_resume(tmp_path, invariant)
    with pytest.raises(ValueError, match="expert_gemm"):
        validate_resume(tmp_path, native)
    del data["expert_gemm"]
    path.write_text(json.dumps(data), encoding="utf-8")
    validate_resume(tmp_path, native)
    with pytest.raises(ValueError, match="expert_gemm"):
        validate_resume(tmp_path, invariant)


def test_batch_invariant_expert_gemm_refuses_cpu(tmp_path):
    with pytest.raises(ValueError, match="stream_expert_gemm.*CUDA"):
        _build(tmp_path, **LEVERS["pool-all-levers"], expert_gemm=MODE)


def test_launch_refuses_cpu_before_loading_the_base():
    pytest.importorskip("torch")
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    training = _load(**{**PREREQUISITES, MBR: True, KEY: MODE})
    owner = SimpleNamespace(device="cpu")
    with pytest.raises(ValueError, match="stream_expert_gemm.*CUDA"):
        StreamingSetupMixin._setup_streaming_transformers(owner, None, training)


def test_native_experts_keep_the_native_linear_and_scaling(tmp_path):
    torch = pytest.importorskip("torch")
    from soup_cli.utils.expert_stream import streamed_experts_class

    model, runtime = _build(tmp_path, **LEVERS["pool-all-levers"])
    try:
        cls = streamed_experts_class()
        experts = [module for module in model.modules() if isinstance(module, cls)]
        assert experts
        for expert in experts:
            assert expert.expert_gemm == "native"
            assert expert.linear is torch.nn.functional.linear
            assert expert.scale is torch.mul
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["native", MODE])
def test_trainer_logs_the_configured_expert_gemm(monkeypatch, mode):
    pytest.importorskip("torch")
    from soup_cli.trainer import l2l_trainer
    from soup_cli.utils.l2l_recipe import resolve_l2l_recipe
    from tests.test_l2l_sft_dispatch import _builder

    attach = l2l_trainer.attach_l2l_state
    owner, build = _builder(monkeypatch, "none", [], k=5)
    owner.config.training = _load(**{**PREREQUISITES, MBR: True, KEY: mode})
    owner._l2l_recipe = resolve_l2l_recipe(owner.config.training)
    owner._stream_runtime = SimpleNamespace(expert_pool=object())
    monkeypatch.setattr(l2l_trainer, "attach_l2l_state", attach)
    monkeypatch.setattr(l2l_trainer, "make_l2l_trainer_class", lambda _base: SimpleNamespace)
    assert build()._soup_l2l.log_fields()["l2l_expert_gemm"] == mode


@pytest.mark.parametrize("ordered", [False, True])
def test_grouped_mbr_matches_plain_l2l_with_invariant_expert_gemm(
    tmp_path, monkeypatch, ordered,
):
    from soup_cli.utils import invariant_gemm, l2l_step

    monkeypatch.setattr(invariant_gemm, "check_supported", lambda device, dtype: None)
    calls = []

    def matmul(left, right):
        calls.append(left.shape[0])
        return rowwise_matmul(left, right)

    monkeypatch.setattr(invariant_gemm, "matmul", matmul)
    monkeypatch.setattr(invariant_gemm, "row_dot", rowwise_dot)
    monkeypatch.setattr(invariant_gemm, "scale_rows", invariant_gemm._scale_function().apply)
    splits = []
    split = l2l_step._split_group_backward

    def record_split(*args):
        splits.append(True)
        return split(*args)

    monkeypatch.setattr(l2l_step, "_split_group_backward", record_split)
    model, runtime = _build(tmp_path, **LEVERS["pool-all-levers"], expert_gemm=MODE)
    try:
        window = _capture_window(model, runtime)
        expected = _run(window)
        assert calls and not splits
        baseline_rows = max(calls)
        for group_size in (1, 2, 5):
            calls.clear()
            splits.clear()
            actual = _run(
                window,
                route_preserving_mbr=True,
                mbr_group_size=group_size,
                mbr_ordered_backward=ordered,
            )
            _assert_same(actual, expected)
            assert bool(splits) is (not ordered)
            if group_size == 5:
                assert max(calls) > baseline_rows
    finally:
        runtime.close()


@pytest.mark.gpu
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16-autocast"])
def test_cuda_grouped_mbr_matches_plain_l2l(tmp_path, monkeypatch, bf16):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if bf16 and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 needs compute capability 8.0+")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model, runtime = _build(
            tmp_path, device="cuda", **LEVERS["pool-all-levers"], expert_gemm=MODE,
        )
        try:
            window = _capture_window(model, runtime, device="cuda")
            expected = _run(window, bf16)
            for group_size in (2, 5):
                actual = _run(window, bf16, route_preserving_mbr=True, mbr_group_size=group_size)
                _assert_same(actual, expected)
        finally:
            runtime.close()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
