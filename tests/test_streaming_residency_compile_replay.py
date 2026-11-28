"""Chunked independent replay must agree with the preserved original comparator."""

import ast
import hashlib
import importlib.util
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.harness import liger_streaming_parity
from benchmarks.harness.streaming_residency_controls import compare_streamed, compile_bench

REPO = Path(__file__).parents[1]

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_bench_preserves_bytes_recorded_in_gpu_provenance():
    raw = Path(compile_bench.__file__).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "080b4de75b8f228f367d7f9de2e4b7be7793b2b3c1e6c06c27ec0b0638b10f1b"
    )


@pytest.mark.parametrize("name", [
    "liger_streaming_bench", "liger_streaming_parity", "streaming_optimization_bench",
])
def test_published_runtime_dependencies_preserve_frozen_execution_ast(name):
    archive = REPO / "tests/fixtures/streaming_residency/planned-harness.zip"
    with zipfile.ZipFile(archive) as source:
        member = next(member for member in source.namelist() if member.endswith(name + ".py"))
        frozen = source.read(member)
    public = REPO / "benchmarks/harness" / (name + ".py")
    assert ast.dump(ast.parse(public.read_bytes())) == ast.dump(ast.parse(frozen))
    load_module(public, "runtime_dependency_" + name)


def test_frozen_wrapper_metadata_compatibility_never_enables_liger():
    wrapper = SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace(use_liger=False)))
    compile_bench.initialize_patch_metadata(wrapper)
    assert wrapper._streaming_liger_patches == {}
    wrapper._streaming_liger_patches = {"unexpected": 1}
    compile_bench.initialize_patch_metadata(wrapper)
    assert wrapper._streaming_liger_patches == {"unexpected": 1}
    wrapper.config.training.use_liger = True
    with pytest.raises(ValueError, match="ordinary unpatched"):
        compile_bench.initialize_patch_metadata(wrapper)


def test_incomplete_capture_cannot_pass_without_loss(tmp_path):
    import torch
    from safetensors.torch import save_file

    for name in ("first.safetensors", "second.safetensors"):
        save_file({"initial/weight": torch.ones(3)}, tmp_path / name)
    with pytest.raises(KeyError, match="loss"):
        compare_streamed.compare(tmp_path / "first.safetensors", tmp_path / "second.safetensors")


@pytest.mark.parametrize("mutation", ["none", "batch", "initial", "logits", "nan", "loss"])
def test_chunked_gate_matches_original_comparator(tmp_path, mutation):
    import torch
    from safetensors.torch import save_file

    values = {
        "batch/input_ids": torch.arange(8).reshape(1, 8),
        "initial/weight": torch.tensor([1., 2., 0.]),
        "logits/0": torch.linspace(-2, 2, 34 * 10001).reshape(1, 34, 10001),
        "loss": torch.tensor([1.23449, 1.23449], dtype=torch.float64),
    }
    changed = {name: value.clone() for name, value in values.items()}
    if mutation == "batch":
        changed["batch/input_ids"][0, 0] += 1
    elif mutation == "initial":
        changed["initial/weight"][0] += .00001
    elif mutation == "logits":
        changed["logits/0"][0, 33, 10000] += .2
    elif mutation == "nan":
        changed["logits/0"][0, 18, 100] = float("nan")
    elif mutation == "loss":
        changed["loss"][0] = 1.23451
    for folder, tensors in [("reference", values), ("candidate", changed)]:
        target = tmp_path / folder
        target.mkdir()
        save_file(tensors, target / "trace.safetensors")
    expected = liger_streaming_parity.compare(tmp_path / "reference", tmp_path / "candidate")
    actual = compare_streamed.compare(
        tmp_path / "reference/trace.safetensors", tmp_path / "candidate/trace.safetensors"
    )
    for field in ("passed", "loss_three_decimals", "groups", "rtol", "atol"):
        assert actual[field] == expected[field]
    assert actual["passed"] == (mutation == "none")
