"""Local runner controls. CPU evidence cannot become a CUDA performance verdict."""

import json

import pytest

from benchmarks.harness.streaming_controls.compare import run


def _settings():
    import torch

    return (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


@pytest.fixture
def caller_settings():
    import torch

    original = _settings()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(False, warn_only=True)
        yield _settings()
    finally:
        threads, deterministic, warn_only, matmul_tf32, cudnn_tf32 = original
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def test_cpu_comparison_preserves_exactness_and_measures_reads(tmp_path, caller_settings):
    result = run(tmp_path / "run", "cpu-smoke")
    assert _settings() == caller_settings
    assert result["status"] == "NUMERICAL_PASS"
    assert result["exact_training_match"]
    assert result["performance_verdict"] == "NO_VERDICT"
    before, after = result["records"]
    assert 0 < after["range_reads_total"]["bytes"] < before["range_reads_total"]["bytes"]
    assert 0 < len(after["resident_indices"]) < 4
    assert 0 < after["initial_stats"]["resident_bytes"] <= 45_000
    assert after["final_stats"]["resident_range_bytes_served"] > 0
    assert before["validity"] is after["validity"] is None
    assert len(before["updates"]) == len(after["updates"]) == 3


def test_existing_output_cannot_be_reused(tmp_path):
    (tmp_path / "sentinel").write_text("keep")
    with pytest.raises(
        FileExistsError, match="^output must be a new directory; no retry or overwrite$"
    ):
        run(tmp_path, "cpu-smoke")
    assert (tmp_path / "sentinel").read_text() == "keep"


def test_setup_failure_is_preserved(tmp_path, monkeypatch, caller_settings):
    from soup_cli.utils import layer_shard

    def fail(*args, **kwargs):
        raise OSError("injected sharding failure")

    monkeypatch.setattr(layer_shard, "shard_checkpoint", fail)
    target = tmp_path / "failed"
    with pytest.raises(OSError, match="^injected sharding failure$"):
        run(target, "cpu-smoke")
    assert _settings() == caller_settings
    result = json.loads((target / "result.json").read_text())
    assert result["status"] == "FAILED"
    assert result["error"] == "OSError('injected sharding failure')"
    assert result["performance_verdict"] == "NO_VERDICT"
