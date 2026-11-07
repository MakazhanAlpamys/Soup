"""Checkpoint-visible NF4 fused-forward groundwork (#842, folding #841)."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys

import pytest

from soup_cli.utils.layer_stream_runtime import checkpoint_visible_nf4_linear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
bnb_functional = pytest.importorskip("bitsandbytes.functional")


@pytest.mark.parametrize("nested", [False, True])
def test_fused_forward_and_input_gradient_match_dequant_linear(nested):
    torch.manual_seed(17)
    weight = torch.randn(16, 16)
    packed, state = bnb_functional.quantize_4bit(
        weight, quant_type="nf4", compress_statistics=nested
    )
    x = torch.randn(3, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)

    actual = checkpoint_visible_nf4_linear(x, packed, state)
    dense = bnb_functional.dequantize_4bit(packed, state).to(reference_x.dtype)
    expected = torch.nn.functional.linear(reference_x, dense)

    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(x.grad, reference_x.grad)


def _non_nested_state(absmax, template):
    return bnb_functional.QuantState(
        absmax=absmax,
        shape=template.shape,
        dtype=template.dtype,
        blocksize=template.blocksize,
        quant_type=template.quant_type,
    )


def test_save_for_backward_makes_pool_recycling_recompute_and_preserve_gradients():
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(23)
    weight0 = torch.randn(16, 16)
    weight1 = torch.randn(16, 16)
    packed0, state0 = bnb_functional.quantize_4bit(weight0, quant_type="nf4")
    packed1, state1 = bnb_functional.quantize_4bit(weight1, quant_type="nf4")

    shared_packed = packed0.clone()
    shared_absmax = state0.absmax.clone()
    shared_state = _non_nested_state(shared_absmax, state0)
    calls = {"n": 0}

    def streamed_body(value):
        calls["n"] += 1

        with torch.no_grad():
            shared_packed.copy_(packed0)
            shared_absmax.copy_(state0.absmax)
        return checkpoint_visible_nf4_linear(value, shared_packed, shared_state)

    x = torch.randn(3, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    output = checkpoint(streamed_body, x, use_reentrant=False)

    # Recycle the pool slot to another layer before backward.
    with torch.no_grad():
        shared_packed.copy_(packed1)
        shared_absmax.copy_(state1.absmax)

    output.square().sum().backward()
    dense0 = bnb_functional.dequantize_4bit(packed0, state0).to(reference_x.dtype)
    reference = torch.nn.functional.linear(reference_x, dense0)
    reference.square().sum().backward()

    assert calls["n"] == 2, "checkpoint did not recompute the custom NF4 op"
    assert torch.equal(x.grad, reference_x.grad)


def test_plain_ctx_control_is_detectably_wrong_after_pool_recycling():
    """Negative control: reproduces the #331 lifetime bug on a tiny CPU fixture."""
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(29)
    weight0 = torch.randn(16, 16)
    weight1 = torch.randn(16, 16)

    packed0, state0 = bnb_functional.quantize_4bit(weight0, quant_type="nf4")
    packed1, state1 = bnb_functional.quantize_4bit(weight1, quant_type="nf4")
    shared_packed = packed0.clone()
    shared_absmax = state0.absmax.clone()
    shared_state = _non_nested_state(shared_absmax, state0)

    class PlainCtx4Bit(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, packed):
            ctx.packed = packed
            ctx.state = shared_state
            return torch.ops.bitsandbytes.gemm_4bit.default(
                value,
                packed,
                shared_state.shape,
                shared_state.absmax,
                shared_state.blocksize,
                shared_state.quant_type,
            )

        @staticmethod
        def backward(ctx, grad_output):
            dense = bnb_functional.dequantize_4bit(ctx.packed, ctx.state).to(
                grad_output.dtype
            )
            return torch.matmul(grad_output, dense), None

    calls = {"n": 0}

    def broken_body(value):
        calls["n"] += 1
        with torch.no_grad():
            shared_packed.copy_(packed0)
            shared_absmax.copy_(state0.absmax)
        return PlainCtx4Bit.apply(value, shared_packed)

    x = torch.randn(3, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    output = checkpoint(broken_body, x, use_reentrant=False)
    with torch.no_grad():
        shared_packed.copy_(packed1)
        shared_absmax.copy_(state1.absmax)
    output.square().sum().backward()

    dense0 = bnb_functional.dequantize_4bit(packed0, state0).to(reference_x.dtype)
    reference = torch.nn.functional.linear(reference_x, dense0)
    reference.square().sum().backward()

    assert calls["n"] == 1, "negative control unexpectedly became checkpoint-visible"
    assert not torch.equal(x.grad, reference_x.grad)
    assert (x.grad - reference_x.grad).abs().max().item() > 1e-3


def test_production_capability_gate_keeps_cpu_on_the_old_path():
    from soup_cli.utils.layer_stream_runtime import _can_use_checkpoint_visible_nf4_gemm

    packed, state = bnb_functional.quantize_4bit(torch.randn(16, 16), quant_type="nf4")
    assert packed.device.type == "cpu"
    assert _can_use_checkpoint_visible_nf4_gemm(torch.randn(2, 16), state) is False


_BNB_DISPATCH_NAMES = ("_gemm_4bit_use_custom_fn", "_gemm_4bit_custom_max_m")


def _module_level_names(source):
    """Names a module source binds at its top level (Assign / AnnAssign / def / class).

    Walks into top-level ``if`` / ``try`` / ``with`` blocks, because bitsandbytes binds
    the dispatch names under ``if torch.version.hip is None:`` / ``else:``. Function and
    class bodies are not entered: a local variable is not a module attribute.
    """
    bound = set()

    def add_target(target):
        if isinstance(target, ast.Name):
            bound.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                add_target(element)
        elif isinstance(target, ast.Starred):
            add_target(target.value)

    def visit(statements):
        for node in statements:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    add_target(target)
            elif isinstance(node, ast.AnnAssign):
                if node.value is not None:
                    add_target(node.target)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, (ast.Try, ast.TryStar)):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                visit(node.body)

    visit(ast.parse(source).body)
    return bound


def _missing_module_level_names(source, names):
    """The entries of ``names`` that ``source`` does not bind at module level."""
    bound = _module_level_names(source)
    return [name for name in names if name not in bound]


def _locate_bnb_cuda_ops_source():
    """Path of ``bitsandbytes/backends/cuda/ops`` (module or package), without importing it.

    Returns ``None`` only when bitsandbytes is not installed (the caller skips). When it
    IS installed and the module is not where we expect it, this fails instead of
    skipping: bitsandbytes is an unconditional ``[train]`` dependency, so a moved or
    renamed module is exactly the upstream change this canary exists to report.
    """
    spec = importlib.util.find_spec("bitsandbytes")
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    candidates = [
        os.path.join(root, "backends", "cuda", leaf)
        for root in locations
        for leaf in ("ops.py", os.path.join("ops", "__init__.py"))
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    expected = " or ".join(candidates) or "backends/cuda/ops.py under the bitsandbytes package"
    pytest.fail(
        f"bitsandbytes is installed (search locations {locations}) but its CUDA ops module "
        f"is not at the expected path {expected}. "
        "src/soup_cli/utils/layer_stream_runtime.py (_can_use_checkpoint_visible_nf4_gemm) "
        f"reads {' / '.join(_BNB_DISPATCH_NAMES)} from that module, so find where "
        "bitsandbytes moved them and update this canary and that reader together.",
        pytrace=False,
    )


def _canary_body():
    # Read the source instead of importing ``bitsandbytes.backends.cuda.ops``: that module
    # reads ``torch._C._cuda_getCurrentRawStream`` at import (bitsandbytes 0.50.2,
    # ops.py:69), which a CPU-only torch wheel does not have, so an import would skip
    # this canary on every CPU-wheel CI cell.
    ops_path = _locate_bnb_cuda_ops_source()
    if ops_path is None:
        pytest.skip("bitsandbytes is not installed")

    with open(ops_path, encoding="utf-8") as handle:
        source = handle.read()

    missing = _missing_module_level_names(source, _BNB_DISPATCH_NAMES)
    assert not missing, (
        f"bitsandbytes {ops_path} no longer binds {missing} at module level; "
        "layer_stream_runtime._can_use_checkpoint_visible_nf4_gemm reads them"
    )


def test_bitsandbytes_private_dispatch_contract_is_visible_to_cpu_ci():
    _canary_body()


_SAMPLE_NAMES = ("_gemm_4bit_use_custom_fn", "_gemm_4bit_custom_max_m")


def test_dispatch_name_check_accepts_names_bound_the_way_bitsandbytes_binds_them():
    source = (
        "import torch\n"
        "if torch.version.hip is None:\n"
        "    _gemm_4bit_use_custom_fn = _custom_cuda\n"
        "    _gemm_4bit_custom_max_m: int = 1536\n"
        "else:\n"
        "    _gemm_4bit_use_custom_fn = _custom_rocm\n"
        "    _gemm_4bit_custom_max_m = 256\n"
    )
    assert _missing_module_level_names(source, _SAMPLE_NAMES) == []
    defined = "def _gemm_4bit_use_custom_fn():\n    pass\n_gemm_4bit_custom_max_m = 1\n"
    assert _missing_module_level_names(defined, _SAMPLE_NAMES) == []


@pytest.mark.parametrize(
    ("source", "expected_missing"),
    [
        ("_gemm_4bit_use_custom_fn = 1\n", ["_gemm_4bit_custom_max_m"]),
        ("_gemm_4bit_custom_max_m = 1\n", ["_gemm_4bit_use_custom_fn"]),
        ("", ["_gemm_4bit_use_custom_fn", "_gemm_4bit_custom_max_m"]),
        (
            "def helper():\n    _gemm_4bit_use_custom_fn = 1\n    _gemm_4bit_custom_max_m = 2\n",
            ["_gemm_4bit_use_custom_fn", "_gemm_4bit_custom_max_m"],
        ),
        (
            "_gemm_4bit_use_custom_fn: int\n_gemm_4bit_custom_max_m = 1\n",
            ["_gemm_4bit_use_custom_fn"],
        ),
        (
            "# _gemm_4bit_use_custom_fn = 1\n_gemm_4bit_custom_max_m = 1\n",
            ["_gemm_4bit_use_custom_fn"],
        ),
    ],
    ids=[
        "missing-max-m",
        "missing-use-custom-fn",
        "empty-module",
        "bound-only-inside-a-function",
        "annotation-without-value",
        "mentioned-only-in-a-comment",
    ],
)
def test_dispatch_name_check_reports_the_name_that_is_missing(source, expected_missing):
    assert _missing_module_level_names(source, _SAMPLE_NAMES) == expected_missing


_FIRST, _SECOND = _SAMPLE_NAMES


@pytest.mark.parametrize(
    "source",
    [
        f"if flag:\n    {_FIRST} = 1\n{_SECOND} = 1\n",
        f"if flag:\n    pass\nelse:\n    {_FIRST} = 1\n{_SECOND} = 1\n",
        f"if flag:\n    pass\nelif other:\n    {_FIRST} = 1\n{_SECOND} = 1\n",
        f"try:\n    {_FIRST} = 1\nexcept ImportError:\n    pass\n{_SECOND} = 1\n",
        f"try:\n    pass\nexcept ImportError:\n    {_FIRST} = 1\n{_SECOND} = 1\n",
        f"try:\n    pass\nexcept ImportError:\n    pass\nelse:\n    {_FIRST} = 1\n{_SECOND} = 1\n",
        f"try:\n    pass\nfinally:\n    {_FIRST} = 1\n{_SECOND} = 1\n",
        f"try:\n    pass\nexcept* ValueError:\n    {_FIRST} = 1\n{_SECOND} = 1\n",
        f"with context():\n    {_FIRST} = 1\n{_SECOND} = 1\n",
        f"async with context():\n    {_FIRST} = 1\n{_SECOND} = 1\n",
        (
            "with context():\n"
            "    try:\n"
            "        if flag:\n"
            "            pass\n"
            "        else:\n"
            f"            {_FIRST} = 1\n"
            "    finally:\n"
            "        pass\n"
            f"{_SECOND} = 1\n"
        ),
    ],
    ids=[
        "only-under-if-body",
        "only-under-else",
        "only-under-elif",
        "only-under-try-body",
        "only-under-except",
        "only-under-try-else",
        "only-under-finally",
        "only-under-except-star",
        "only-under-with",
        "only-under-async-with",
        "only-under-else-nested-in-try-in-with",
    ],
)
def test_dispatch_name_check_enters_every_block_that_can_bind_a_module_name(source):
    # One name is bound at the top level, the other ONLY inside the block under test, so
    # a walker that stops descending into that block reports the second name as missing.
    assert _missing_module_level_names(source, _SAMPLE_NAMES) == []


def _fake_bitsandbytes_find_spec(monkeypatch, spec):
    """Make ``importlib.util.find_spec("bitsandbytes")`` return ``spec``; others are real."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "bitsandbytes":
            return spec
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _installed_bitsandbytes(root):
    from types import SimpleNamespace

    return SimpleNamespace(submodule_search_locations=[str(root)])


def _outcome_of(call):
    """Run ``call``; report ``("failed" | "skipped" | "passed", message)``.

    ``pytest.raises`` would let a ``skip`` escape, and a skipped test is not a red one: a
    canary that went back to skipping would then read as "not failed" instead of failing
    this test. Catching both outcomes lets the test assert on the outcome itself.
    """
    try:
        call()
    except pytest.fail.Exception as error:
        return "failed", str(error)
    except pytest.skip.Exception as error:
        return "skipped", str(error)
    return "passed", ""


def test_ops_locator_skips_only_when_bitsandbytes_is_not_installed(monkeypatch):
    _fake_bitsandbytes_find_spec(monkeypatch, None)
    assert _locate_bnb_cuda_ops_source() is None
    outcome, message = _outcome_of(_canary_body)
    assert (outcome, message) == ("skipped", "bitsandbytes is not installed")


def test_ops_locator_fails_when_bitsandbytes_is_installed_but_ops_module_moved(
    monkeypatch, tmp_path
):
    cuda_dir = tmp_path / "backends" / "cuda"
    cuda_dir.mkdir(parents=True)
    (cuda_dir / "ops_renamed.py").write_text(
        f"{_FIRST} = 1\n{_SECOND} = 2\n", encoding="utf-8"
    )
    _fake_bitsandbytes_find_spec(monkeypatch, _installed_bitsandbytes(tmp_path))

    outcome, message = _outcome_of(_locate_bnb_cuda_ops_source)

    assert outcome == "failed", (outcome, message)
    assert os.path.join(str(tmp_path), "backends", "cuda", "ops.py") in message
    assert "src/soup_cli/utils/layer_stream_runtime.py" in message
    assert _FIRST in message and _SECOND in message
    # The canary itself must fail the same way, not skip.
    assert _outcome_of(_canary_body)[0] == "failed"


@pytest.mark.parametrize("locations", [None, []], ids=["no-locations", "empty-locations"])
def test_ops_locator_fails_when_an_installed_bitsandbytes_has_no_search_locations(
    monkeypatch, locations
):
    from types import SimpleNamespace

    spec = SimpleNamespace(submodule_search_locations=locations)
    _fake_bitsandbytes_find_spec(monkeypatch, spec)

    outcome, message = _outcome_of(_locate_bnb_cuda_ops_source)

    assert outcome == "failed", (outcome, message)
    assert "backends/cuda/ops.py" in message
    assert "src/soup_cli/utils/layer_stream_runtime.py" in message


@pytest.mark.parametrize(
    "relative_path",
    [("backends", "cuda", "ops.py"), ("backends", "cuda", "ops", "__init__.py")],
    ids=["ops-module", "ops-package"],
)
def test_ops_locator_finds_the_module_as_a_file_or_a_package(
    monkeypatch, tmp_path, relative_path
):
    ops_file = tmp_path.joinpath(*relative_path)
    ops_file.parent.mkdir(parents=True)
    ops_file.write_text("", encoding="utf-8")
    _fake_bitsandbytes_find_spec(monkeypatch, _installed_bitsandbytes(tmp_path))

    assert _locate_bnb_cuda_ops_source() == str(ops_file)


def test_canary_reads_the_names_from_the_located_ops_source(monkeypatch, tmp_path):
    ops_file = tmp_path / "backends" / "cuda" / "ops.py"
    ops_file.parent.mkdir(parents=True)
    _fake_bitsandbytes_find_spec(monkeypatch, _installed_bitsandbytes(tmp_path))

    ops_file.write_text(f"{_FIRST} = 1\n{_SECOND} = 2\n", encoding="utf-8")
    assert _outcome_of(_canary_body) == ("passed", "")

    ops_file.write_text(f"{_FIRST} = 1\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=_SECOND) as excinfo:
        _canary_body()
    assert str(ops_file) in str(excinfo.value)


@pytest.mark.parametrize(
    ("m", "k", "dispatch", "expected"),
    [
        (3, 64, True, True),
        (3, 64, False, False),
        (97, 64, True, False),
        (3, 65, True, False),
    ],
    ids=["fused", "bnb-dequant", "above-m-cap", "misaligned-k"],
)
def test_cuda_route_follows_bitsandbytes_shape_decision_on_cpu(
    monkeypatch, m, k, dispatch, expected
):
    from types import SimpleNamespace

    import bitsandbytes.backends.cuda as bnb_cuda_package

    from soup_cli.utils.layer_stream_runtime import _can_use_checkpoint_visible_nf4_gemm

    calls = []

    def use_custom_fn(device_index, dtype, rows, out_features, in_features):
        calls.append((device_index, dtype, rows, out_features, in_features))
        return dispatch

    fake_ops = SimpleNamespace(
        _gemm_4bit_custom_max_m=96,
        _gemm_4bit_use_custom_fn=use_custom_fn,
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes.backends.cuda.ops", fake_ops)
    monkeypatch.setattr(bnb_cuda_package, "ops", fake_ops, raising=False)
    x = SimpleNamespace(
        device=SimpleNamespace(type="cuda", index=0),
        dtype=torch.bfloat16,
        shape=(m, k),
        numel=lambda: m * k,
    )
    state = SimpleNamespace(
        nested=False,
        shape=(128, k),
        blocksize=64,
    )

    assert _can_use_checkpoint_visible_nf4_gemm(x, state) is expected
    if m <= 96 and k % 64 == 0:
        assert calls == [(0, torch.bfloat16, m, 128, k)]
    else:
        assert calls == []


@pytest.mark.parametrize(
    ("shape", "expected", "expected_m"),
    [
        ((1, 97, 64), False, None),
        ((2, 3, 64), True, 6),
    ],
    ids=["three-dimensional-above-m-cap", "three-dimensional-fused"],
)
def test_cuda_route_flattens_all_leading_dimensions_on_cpu(
    monkeypatch, shape, expected, expected_m
):
    from types import SimpleNamespace

    import bitsandbytes.backends.cuda as bnb_cuda_package

    from soup_cli.utils.layer_stream_runtime import _can_use_checkpoint_visible_nf4_gemm

    calls = []

    def use_custom_fn(device_index, dtype, rows, out_features, in_features):
        calls.append((device_index, dtype, rows, out_features, in_features))
        return True

    fake_ops = SimpleNamespace(
        _gemm_4bit_custom_max_m=96,
        _gemm_4bit_use_custom_fn=use_custom_fn,
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes.backends.cuda.ops", fake_ops)
    monkeypatch.setattr(bnb_cuda_package, "ops", fake_ops, raising=False)
    x = SimpleNamespace(
        device=SimpleNamespace(type="cuda", index=0),
        dtype=torch.bfloat16,
        shape=shape,
        numel=lambda: shape[0] * shape[1] * shape[2],
    )
    state = SimpleNamespace(nested=False, shape=(128, 64), blocksize=64)

    assert _can_use_checkpoint_visible_nf4_gemm(x, state) is expected
    if expected_m is None:
        assert calls == []
    else:
        assert calls == [(0, torch.bfloat16, expected_m, 128, 64)]


def test_backward_refuses_a_recycled_stream_slot():
    torch.manual_seed(31)
    packed, state = bnb_functional.quantize_4bit(
        torch.randn(16, 16), quant_type="nf4"
    )
    owner = {"current": 0}

    def assert_owner():
        if owner["current"] != 0:
            raise RuntimeError("recycled weight slot")

    packed._soup_stream_owner_check = assert_owner
    x = torch.randn(3, 16, requires_grad=True)
    output = checkpoint_visible_nf4_linear(x, packed, state)
    owner["current"] = 1
    with pytest.raises(RuntimeError, match="recycled weight slot"):
        output.square().sum().backward()


def test_the_production_owner_check_refuses_a_recycled_slot(tmp_path):
    from test_v07202 import _nf4_stream

    import soup_cli.utils.layer_stream_runtime as runtime_module

    model, runtime, _, _, _ = _nf4_stream(tmp_path, n_layers=4, buffers=2)
    layer = runtime_module.decoder_owner(model).layers[0]
    slot = runtime.pool.slot_for(layer.idx)

    runtime.pool.owner[slot] = layer.idx
    layer._assert_nf4_slot_owner()

    runtime.pool.owner[slot] = layer.idx + 2
    with pytest.raises(RuntimeError, match="recycled weight slot"):
        layer._assert_nf4_slot_owner()


def test_streamed_layer_caches_substitution_views_for_one_pool_mapping(
    tmp_path, monkeypatch
):
    from test_v07202 import _nf4_stream

    import soup_cli.utils.layer_stream_runtime as runtime_module

    model, runtime, _, _, _ = _nf4_stream(tmp_path)
    layer = runtime_module.decoder_owner(model).layers[0]
    buffers = runtime.pool.buffers[runtime.pool.slot_for(layer.idx)]

    # Start from a known-empty cache: model construction must not be what makes
    # this test pass.
    layer._cached_substitution_buffers = None
    layer._cached_substitution_weights = None

    real_rebuild = runtime_module.rebuild_params4bit
    calls = {"n": 0}

    def counting_rebuild(*args, **kwargs):
        calls["n"] += 1
        return real_rebuild(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "rebuild_params4bit", counting_rebuild)

    quantized = sum(
        ckpt in layer.quant_specs for ckpt in layer.name_map.values()
    )
    assert quantized > 0

    first = layer._substituted_weights(buffers)
    second = layer._substituted_weights(buffers)
    assert second is first
    assert calls["n"] == quantized

    # A different mapping object represents a different pool view and must not
    # inherit the cache, even when it currently points at the same tensors.
    other_mapping = dict(buffers)
    third = layer._substituted_weights(other_mapping)
    assert third is not first
    assert calls["n"] == 2 * quantized


def test_missing_bnb_private_dispatch_symbols_fail_toward_safe_function_on_cpu(
    monkeypatch,
):
    from types import SimpleNamespace

    import bitsandbytes.backends.cuda as bnb_cuda_package

    from soup_cli.utils.layer_stream_runtime import _can_use_checkpoint_visible_nf4_gemm

    fake_ops = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "bitsandbytes.backends.cuda.ops", fake_ops)
    monkeypatch.setattr(bnb_cuda_package, "ops", fake_ops, raising=False)
    x = SimpleNamespace(
        device=SimpleNamespace(type="cuda", index=0),
        dtype=torch.bfloat16,
        shape=(512, 64),
        numel=lambda: 512 * 64,
    )
    state = SimpleNamespace(nested=False, shape=(128, 64), blocksize=64)

    assert _can_use_checkpoint_visible_nf4_gemm(x, state) is True


def test_forced_function_in_a_recycling_stream_matches_unpatched_resident(
    tmp_path, monkeypatch
):
    from test_v07202 import (
        _nf4_stream,
        _randomise_lora_b,
        _resident_nf4,
        _sync_adapters,
    )

    import soup_cli.utils.layer_stream_runtime as runtime_module

    calls = {"function": 0, "owner_checks": 0}
    real_function = runtime_module.checkpoint_visible_nf4_linear

    def counting_function(x, packed, quant_state, bias=None):
        calls["function"] += 1
        check = getattr(packed, "_soup_stream_owner_check", None)
        assert check is not None, "streamed NF4 weight carries no slot-owner check"

        def counted_check():
            calls["owner_checks"] += 1
            check()

        packed._soup_stream_owner_check = counted_check
        try:
            return real_function(x, packed, quant_state, bias)
        finally:
            packed._soup_stream_owner_check = check

    monkeypatch.setattr(
        runtime_module, "checkpoint_visible_nf4_linear", counting_function
    )
    monkeypatch.setattr(
        runtime_module,
        "_can_use_checkpoint_visible_nf4_gemm",
        lambda *_args, **_kwargs: True,
    )

    # Four layers over two buffers: every slot is refilled by another layer mid-step.
    streamed, runtime, weights, _index, _shards = _nf4_stream(
        tmp_path, n_layers=4, buffers=2
    )
    resident = _resident_nf4(weights)
    _randomise_lora_b(resident)
    assert _sync_adapters(streamed, resident) > 0

    input_ids = torch.randint(
        0, 64, (1, 16), generator=torch.Generator().manual_seed(5)
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids,
    }
    streamed.train()
    resident.train()
    streamed_loss = streamed(**batch).loss
    streamed_loss.backward()
    resident_loss = resident(**batch).loss
    resident_loss.backward()

    assert calls["function"] > 0 and calls["owner_checks"] > 0
    assert runtime.pool.loads > 4, "no slot was recycled"
    assert torch.equal(streamed_loss, resident_loss)

    def lora_grads(model):
        return {
            name.replace(".inner.", "."): parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.grad is not None
        }

    mine, theirs = lora_grads(streamed), lora_grads(resident)
    assert mine and mine.keys() == theirs.keys()
    for name, grad in mine.items():
        assert torch.equal(grad, theirs[name]), name


@pytest.mark.gpu
def test_missing_bnb_private_dispatch_symbols_fail_toward_the_safe_function(
    monkeypatch,
):
    from bitsandbytes.backends.cuda import ops as bnb_cuda_ops

    from soup_cli.utils.layer_stream_runtime import (
        _can_use_checkpoint_visible_nf4_gemm,
    )

    weight = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    _packed, state = bnb_functional.quantize_4bit(weight, quant_type="nf4")
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    monkeypatch.delattr(bnb_cuda_ops, "_gemm_4bit_use_custom_fn")
    assert _can_use_checkpoint_visible_nf4_gemm(x, state) is True


@pytest.mark.gpu
def test_fused_cuda_recycling_matches_private_buffer_and_plain_ctx_fails():
    from torch.utils.checkpoint import checkpoint

    from soup_cli.utils.layer_stream_runtime import (
        _can_use_checkpoint_visible_nf4_gemm,
    )

    torch.manual_seed(37)
    weight0 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    weight1 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    packed0, state0 = bnb_functional.quantize_4bit(weight0, quant_type="nf4")
    packed1, state1 = bnb_functional.quantize_4bit(weight1, quant_type="nf4")
    shared_packed = packed0.clone()
    shared_absmax = state0.absmax.clone()
    shared_state = _non_nested_state(shared_absmax, state0)
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert _can_use_checkpoint_visible_nf4_gemm(x, shared_state) is True

    calls = {"visible": 0, "plain": 0}

    def visible_body(value):
        calls["visible"] += 1
        with torch.no_grad():
            shared_packed.copy_(packed0)
            shared_absmax.copy_(state0.absmax)
        return checkpoint_visible_nf4_linear(value, shared_packed, shared_state)

    output = checkpoint(visible_body, x, use_reentrant=False)
    with torch.no_grad():
        shared_packed.copy_(packed1)
        shared_absmax.copy_(state1.absmax)
    output.float().square().sum().backward()

    private_x = x.detach().clone().requires_grad_(True)
    private = checkpoint_visible_nf4_linear(private_x, packed0, state0)
    private.float().square().sum().backward()
    assert calls["visible"] == 2
    assert torch.equal(x.grad, private_x.grad)

    class PlainCtx4Bit(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, packed):
            ctx.packed = packed
            ctx.state = shared_state
            return torch.ops.bitsandbytes.gemm_4bit.default(
                value,
                packed,
                shared_state.shape,
                shared_state.absmax,
                shared_state.blocksize,
                shared_state.quant_type,
            )

        @staticmethod
        def backward(ctx, grad_output):
            dense = bnb_functional.dequantize_4bit(ctx.packed, ctx.state).to(
                grad_output.dtype
            )
            return torch.matmul(grad_output, dense), None

    def plain_body(value):
        calls["plain"] += 1
        with torch.no_grad():
            shared_packed.copy_(packed0)
            shared_absmax.copy_(state0.absmax)
        return PlainCtx4Bit.apply(value, shared_packed)

    plain_x = x.detach().clone().requires_grad_(True)
    plain = checkpoint(plain_body, plain_x, use_reentrant=False)
    with torch.no_grad():
        shared_packed.copy_(packed1)
        shared_absmax.copy_(state1.absmax)
    plain.float().square().sum().backward()
    assert calls["plain"] == 1
    assert not torch.equal(plain_x.grad, private_x.grad)


@pytest.mark.gpu
@pytest.mark.parametrize("seq_len", [8, 128], ids=["fused-m", "training-m"])
def test_streamed_matches_unpatched_resident_logits_and_all_lora_grads(
    tmp_path, monkeypatch, seq_len
):
    import bitsandbytes as bnb
    from test_v07202 import (
        _nf4_stream,
        _randomise_lora_b,
        _resident_nf4,
        _sync_adapters,
    )

    import soup_cli.utils.layer_stream_runtime as runtime_module

    streamed, _runtime, weights, _index, _shards = _nf4_stream(
        tmp_path, n_layers=4, buffers=2, device="cuda", dtype="bfloat16"
    )
    resident = _resident_nf4(weights, dtype="bfloat16", device=0)
    _randomise_lora_b(resident)
    assert _sync_adapters(streamed, resident) > 0

    calls = {"function": 0, "streamed_matmul": 0, "resident_matmul": 0}
    phase = {"name": "streamed"}
    real_function = runtime_module.checkpoint_visible_nf4_linear
    real_matmul = bnb.matmul_4bit
    real_gate = runtime_module._can_use_checkpoint_visible_nf4_gemm
    gate_decisions = []

    def counting_function(*args, **kwargs):
        calls["function"] += 1
        return real_function(*args, **kwargs)

    def counting_matmul(*args, **kwargs):
        calls[phase["name"] + "_matmul"] += 1
        return real_matmul(*args, **kwargs)

    def recording_gate(*args, **kwargs):
        decision = real_gate(*args, **kwargs)
        gate_decisions.append(decision)
        return decision

    monkeypatch.setattr(
        runtime_module, "checkpoint_visible_nf4_linear", counting_function
    )
    monkeypatch.setattr(
        runtime_module, "_can_use_checkpoint_visible_nf4_gemm", recording_gate
    )
    monkeypatch.setattr(bnb, "matmul_4bit", counting_matmul)

    generator = torch.Generator(device="cuda").manual_seed(41)
    input_ids = torch.randint(
        0, 64, (1, seq_len), device="cuda", generator=generator
    )
    streamed_logits = streamed(input_ids=input_ids, use_cache=False).logits
    streamed_logits.float().square().mean().backward()
    phase["name"] = "resident"
    resident_logits = resident(input_ids=input_ids, use_cache=False).logits
    resident_logits.float().square().mean().backward()

    assert calls["streamed_matmul"] == 0
    assert calls["resident_matmul"] > 0
    assert gate_decisions
    assert (calls["function"] > 0) is any(gate_decisions)
    assert torch.equal(streamed_logits, resident_logits)

    def adapter_grads(model):
        return {
            name.replace(".inner.", "."): parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.grad is not None
        }

    streamed_grads = adapter_grads(streamed)
    resident_grads = adapter_grads(resident)
    assert streamed_grads and streamed_grads.keys() == resident_grads.keys()
    for name, grad in streamed_grads.items():
        assert torch.equal(grad, resident_grads[name]), name
