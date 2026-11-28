"""Prepared workload identity, real isolated CPU updates, and bf16 evidence bytes."""

import copy
import hashlib
import json
import os

import pytest

from benchmarks.harness.streaming_controls.representative_workload import (
    load_prepared,
    tensor_digest,
    write_prepared,
)


@pytest.fixture
def prepared(tmp_path):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
    )
    weights = tmp_path / "weights"
    model.save_pretrained(weights, safe_serialization=True)
    tokens = [[(x + step) % 64 for x in range(8)] for step in range(6)]
    return write_prepared(
        weights, tmp_path / "prepared", tokens, {"kind": "synthetic CPU fixture"}, smoke=True
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
def test_prepared_cpu_series_has_identical_real_updates_and_inputs(tmp_path, prepared):
    from benchmarks.harness.streaming_controls.isolated_compare import compare_results, run

    result = run(tmp_path / "series", "cpu-smoke", prepared=prepared)
    assert result["status"] == "NUMERICAL_PASS", result
    assert result["performance_verdict"] == "NO_VERDICT"
    assert len(result["arms"]) == 2
    first, second = result["arms"]
    assert first["pid"] != second["pid"]
    assert first["workload_sha256"] == hashlib.sha256(prepared.read_bytes()).hexdigest()
    assert first["effective_tokens_per_step"] == 7
    for arm in result["arms"]:
        assert arm["records"][0]["changed_adapter"]
        assert len(arm["records"][0]["updates"]) == 6
    assert (
        second["records"][0]["range_reads_total"]["bytes"]
        < (first["records"][0]["range_reads_total"]["bytes"])
    )
    tampered = copy.deepcopy(result["arms"])
    tampered[1]["workload_sha256"] = "0" * 64
    assert compare_results(tampered, ["A", "B"])["status"] == "BLOCKED"


def test_cpu_fixture_cannot_be_used_for_cuda(prepared):
    with pytest.raises(ValueError, match="kind"):
        load_prepared(prepared, cuda=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", 0),
        ("warmup", 6),
        ("cache_bytes", 0),
        ("reserved_bytes", True),
        ("dtype", "bfloat16"),
        ("effective_tokens_per_step", 8),
        ("tokens", [[-1] * 8] * 6),
        ("files", {}),
    ],
)
def test_changed_policy_or_incomplete_inputs_are_refused(prepared, field, value):
    data = json.loads(prepared.read_text())
    data[field] = value
    prepared.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_prepared(prepared, cuda=False)


def test_modified_shard_is_rejected(prepared):
    data = json.loads(prepared.read_text())
    name = next(name for name in data["files"] if name.endswith(".safetensors"))
    shard = prepared.parent / name
    with shard.open("r+b") as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="shard changed"):
        load_prepared(prepared, cuda=False)


def test_modified_config_is_rejected(prepared):
    data = json.loads(prepared.read_text())
    from pathlib import Path

    config = Path(data["weights"]) / "config.json"
    config.write_text(config.read_text() + " ")
    with pytest.raises(ValueError, match="configuration"):
        load_prepared(prepared, cuda=False)


@pytest.mark.parametrize(
    "mode,has_prepared", [("cuda-representative", False), ("cuda-pilot", True)]
)
def test_mode_mismatch_refused_before_output(tmp_path, mode, has_prepared):
    from benchmarks.harness.streaming_controls.isolated_compare import run

    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="prepared inputs"):
        run(output, mode, prepared=tmp_path / "missing.json" if has_prepared else None)
    assert not output.exists()


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_tensor_digest_preserves_dtype_shape_name_and_bits(dtype_name):
    import torch

    from benchmarks.harness.streaming_controls.compare import tensor_digest as historical_digest

    value = torch.arange(4).to(getattr(torch, dtype_name))
    reference = tensor_digest({"weight": value})
    assert reference == tensor_digest({"weight": value.clone()})
    assert reference != tensor_digest({"weight": value.reshape(2, 2)})
    assert reference != tensor_digest({"other": value})
    assert reference != tensor_digest({"weight": value + 1})
    if dtype_name == "float32":
        assert reference == historical_digest({"weight": value})


@pytest.mark.parametrize("corrupt", [False, True])
def test_bounded_snapshot_download_checks_hash_and_never_retries(tmp_path, monkeypatch, corrupt):
    import io
    import time

    from benchmarks.harness.streaming_controls import representative_workload as module

    data = b"controlled bytes"
    manifest = {
        "repo": "example/model",
        "files": {"one.bin": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}},
    }
    (tmp_path / "representative_model.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "__file__", str(tmp_path / "representative_workload.py"))
    calls = []

    class Response(io.BytesIO):
        status = 200

    def open_response(url, timeout):
        calls.append((url, timeout))
        return Response(b"bad" if corrupt else data)

    monkeypatch.setattr(module.urllib.request, "urlopen", open_response)
    dest = tmp_path / "snapshot"
    if corrupt:
        with pytest.raises(ValueError, match="hash or size"):
            module.fetch_snapshot(dest, time.time() + 60)
    else:
        module.fetch_snapshot(dest, time.time() + 60)
        assert (dest / "one.bin").read_bytes() == data
    assert len(calls) == 1
    receipt = json.loads((dest / "download_receipt.json").read_text())
    assert receipt["status"] == ("FAILED" if corrupt else "PASS_DOWNLOAD_SHA256")
    with pytest.raises(FileExistsError):
        module.fetch_snapshot(dest, time.time() + 60)
    assert len(calls) == 1


@pytest.mark.parametrize("deadline", [True, -1, float("nan"), float("inf")])
def test_bad_download_deadline_refuses_before_network_and_output(tmp_path, monkeypatch, deadline):
    from benchmarks.harness.streaming_controls import representative_workload as module

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("network"))
    with pytest.raises(TimeoutError):
        module.fetch_snapshot(tmp_path / "new", deadline)
    assert not (tmp_path / "new").exists()
