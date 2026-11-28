"""A tighter disk plan must bind to the actual source and writer, before writes."""

import copy
import hashlib
import json
import re
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from benchmarks.harness import giant_moe_acceptance as acceptance
from benchmarks.harness import giant_moe_source_plan as planning


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = b'{"model_type":"deepseek_v3"}'
    (source / "config.json").write_bytes(config)
    header = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": {
            "shape": [8, 8],
            "dtype": "F32",
            "data_offsets": [0, 256],
        }
    }
    raw = json.dumps(header).encode()
    name = "model-00001-of-00001.safetensors"
    payload = len(raw).to_bytes(8, "little") + raw + bytes(256)
    (source / name).write_bytes(payload)
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget

    bound = new_expert_cache_budget([str(source / name)])
    plan = dict(
        version=1,
        source_revision="pinned",
        dtype="float32",
        n_layers=1,
        quant_suffixes=[],
        cache_budget_bytes=bound,
        generic_cache_budget_bytes=bound,
        config_sha256=hashlib.sha256(config).hexdigest(),
        source_bytes=len(payload),
        files=[dict(name=name, size=len(payload), header_sha256=hashlib.sha256(raw).hexdigest())],
    )
    return source, plan, payload


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("none", None),
        ("revision", "source plan revision/config mismatch"),
        ("config", "source plan revision/config mismatch"),
        ("header", "downloaded source differs from planned header/size"),
        ("size", "downloaded source differs from planned header/size"),
        ("bound", "source plan cache bound mismatch"),
        ("generic_bound", "source plan cache bound mismatch"),
        ("byte_total", "source plan byte total mismatch"),
        ("extra_shard", "source plan shard selection mismatch"),
        ("writer_selection", "source plan shard selection mismatch"),
    ],
)
def test_plan_binds_to_source_headers_sizes_config_and_revision(tmp_path, defect, message):
    source, plan, _ = _source(tmp_path)
    if defect == "revision":
        plan["source_revision"] = "other"
    elif defect == "config":
        (source / "config.json").write_text("{}")
    elif defect == "header":
        plan["files"][0]["header_sha256"] = "0" * 64
    elif defect == "size":
        plan["files"][0]["size"] += 1
    elif defect == "bound":
        plan["cache_budget_bytes"] -= 1
    elif defect == "generic_bound":
        plan["generic_cache_budget_bytes"] -= 1
    elif defect == "byte_total":
        plan["source_bytes"] += 1
    elif defect == "extra_shard":
        (source / "extra.safetensors").write_bytes(b"")
    elif defect == "writer_selection":
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"test.weight": "missing.safetensors"}}),
            encoding="utf-8",
        )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    if defect == "none":
        assert planning.validate_plan(source, path, "pinned") == plan
    else:
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            planning.validate_plan(source, path, "pinned")


@pytest.mark.parametrize("defect", ["none", "suffixes", "layers", "dtype"])
def test_plan_must_match_the_actual_writer_selection(defect):
    plan = dict(quant_suffixes=["self_attn.q_proj.weight"], n_layers=61, dtype="float32")
    actual = copy.deepcopy(plan)
    if defect == "suffixes":
        actual["quant_suffixes"] = []
    elif defect == "layers":
        actual["n_layers"] = 62
    elif defect == "dtype":
        actual["dtype"] = "bfloat16"
    if defect == "none":
        planning.validate_writer_plan(
            plan, actual["quant_suffixes"], actual["n_layers"], actual["dtype"]
        )
    else:
        with pytest.raises(
            ValueError,
            match="^source plan quantization/layout differs from the actual writer$",
        ):
            planning.validate_writer_plan(
                plan, actual["quant_suffixes"], actual["n_layers"], actual["dtype"]
            )


def test_remote_header_reads_only_the_two_exact_ranges(tmp_path, monkeypatch):
    _, _, payload = _source(tmp_path)
    requests = []

    def open_range(request, timeout):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"])
        start, end = map(int, match.groups())
        requests.append((start, end))
        raw = payload[start : end + 1]
        response = SimpleNamespace(
            status=206,
            headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
            read=lambda maximum: raw[:maximum],
        )
        return nullcontext(response)

    monkeypatch.setattr(planning.urllib.request, "urlopen", open_range)
    raw, header = planning.read_remote_header("https://example.test/model", len(payload))
    assert requests == [(0, 7), (8, 7 + len(raw))]
    assert header["model.layers.0.mlp.experts.0.gate_proj.weight"]["shape"] == [8, 8]


@pytest.mark.parametrize("status,content_range", [(200, None), (206, "bytes 0-7/99999")])
def test_ignored_or_wrong_range_is_refused_before_reading_body(monkeypatch, status, content_range):
    response = SimpleNamespace(
        status=status,
        headers={"Content-Range": content_range},
        read=lambda count: pytest.fail("unbounded weight body read"),
    )
    monkeypatch.setattr(
        planning.urllib.request, "urlopen", lambda *args, **kwargs: nullcontext(response)
    )
    with pytest.raises(ValueError, match="^source server did not honor the bounded header range$"):
        planning.read_remote_header("https://example.test/model", 1000)


@pytest.mark.parametrize(
    "name", ["../outside.safetensors", "a/model.safetensors", "a\\model.safetensors", "config.json"]
)
def test_plan_cannot_read_or_create_paths_outside_its_shards(tmp_path, name):
    with pytest.raises(ValueError, match="^source plan requires top-level model shard names$"):
        planning.source_file(tmp_path, name)


@pytest.mark.requires_symlink
def test_plan_refuses_an_escaping_shard_symlink(tmp_path):
    source, plan, payload = _source(tmp_path)
    shard = source / plan["files"][0]["name"]
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(payload)
    shard.unlink()
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="^planned source escapes checkpoint directory$"):
        planning.source_file(source, shard.name)


@pytest.mark.parametrize("received", [7, 9])
def test_remote_header_refuses_short_or_oversized_range(monkeypatch, received):
    response = SimpleNamespace(
        status=206, headers={"Content-Range": "bytes 0-7/1000"}, read=lambda count: bytes(received)
    )
    monkeypatch.setattr(
        planning.urllib.request, "urlopen", lambda *args, **kwargs: nullcontext(response)
    )
    with pytest.raises(
        ValueError, match="^source server returned a truncated or oversized header range$"
    ):
        planning.read_remote_header("https://example.test/model", 1000)


def test_complete_source_requires_exact_count_and_size():
    sizes = [1] * (planning.SOURCE_SHARD_COUNT - 1)
    sizes.append(planning.SOURCE_BYTES - sum(sizes))
    planning.require_complete_source(sizes)
    for invalid in (sizes[:-1], [*sizes[:-1], sizes[-1] + 1]):
        with pytest.raises(
            ValueError, match="^giant MoE requires the complete pinned 163-file DeepSeek-V3 source$"
        ):
            planning.require_complete_source(invalid)


@pytest.mark.parametrize(
    ("model_type", "layers", "message"),
    [
        ("other", 61, "this pilot accepts DeepSeek-V3 FP8 only"),
        ("deepseek_v3", 60, "giant MoE requires the 61-layer DeepSeek-V3 decoder"),
    ],
)
def test_writer_refuses_wrong_architecture_or_depth_before_model_setup(
    tmp_path, model_type, layers, message
):
    config = {
        "model_type": model_type,
        "num_hidden_layers": layers,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        planning.writer_geometry(tmp_path, "float32")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"steps": 0}, "--steps must be positive"),
        ({"checkpoint_every": -1}, "--checkpoint-every must be nonnegative"),
        (
            {"checkpoint_every": 1, "evaluation_protocol": "protocol.json"},
            "checkpoint and resume options apply to the standalone giant MoE pilot",
        ),
        (
            {"resume": "checkpoint.pt", "evaluation_data": "frozen.json"},
            "checkpoint and resume options apply to the standalone giant MoE pilot",
        ),
        (
            {"resume": "checkpoint.pt", "restart_incomplete_cache": True},
            "--resume requires a committed cache and cannot use --restart-incomplete-cache",
        ),
        (
            {"evaluation_protocol": "protocol.json"},
            "both evaluation protocol and frozen data are required",
        ),
    ],
)
def test_runner_refuses_invalid_checkpoint_modes_before_setup(override, message):
    args = dict(
        steps=2,
        checkpoint_every=0,
        resume=None,
        evaluation_protocol=None,
        evaluation_data=None,
        restart_incomplete_cache=False,
    )
    args.update(override)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        acceptance.run(SimpleNamespace(**args))


@pytest.mark.parametrize(
    ("available", "devices", "vram_mib", "message"),
    [
        (False, 0, 0, "giant MoE requires a real CUDA device"),
        (True, 2, 8192, "giant MoE pilot requires one visible CUDA GPU in the 8 GB class"),
        (True, 1, 4096, "giant MoE pilot requires one visible CUDA GPU in the 8 GB class"),
        (True, 1, 16384, "giant MoE pilot requires one visible CUDA GPU in the 8 GB class"),
    ],
)
def test_runner_refuses_cuda_outside_the_pilot_scope(
    tmp_path, monkeypatch, available, devices, vram_mib, message
):
    cuda = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: devices,
        get_device_properties=lambda device: SimpleNamespace(total_memory=vram_mib * 2**20),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = SimpleNamespace(
        steps=2,
        checkpoint_every=0,
        resume=None,
        evaluation_protocol=None,
        evaluation_data=None,
        restart_incomplete_cache=False,
        reuse_cache=False,
        source=tmp_path / "source",
        cache=tmp_path / "cache",
        output=tmp_path / "output",
    )
    with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
        acceptance.run(args)
    assert not args.output.exists()


def test_cache_reuse_and_restart_require_the_right_index_state(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    assert not acceptance.reuse_cache(cache, reuse_requested=False, restart_incomplete=True)
    with pytest.raises(
        ValueError, match="^--reuse-cache and --resume require a committed cache index$"
    ):
        acceptance.reuse_cache(cache, reuse_requested=True, restart_incomplete=False)
    (cache / "index.json").write_text("{}", encoding="utf-8")
    assert acceptance.reuse_cache(cache, reuse_requested=True, restart_incomplete=False)
    with pytest.raises(ValueError, match="^cache already exists: use --reuse-cache$"):
        acceptance.reuse_cache(cache, reuse_requested=False, restart_incomplete=True)


@pytest.mark.parametrize(
    ("override", "budget", "message"),
    [
        ({}, 3, None),
        ({"total_params": 4096}, 3, "giant MoE pilot requires at least 500B source parameters"),
        ({"expert_layers": ()}, 3, "unexpected DeepSeek-V3 cache geometry"),
        ({}, 2, "cache exceeds planned budget: 3 > 2 bytes"),
    ],
)
def test_runner_checks_cache_geometry_and_disk_bound_before_training(
    tmp_path, override, budget, message
):
    (tmp_path / "index.json").write_bytes(b"{}\n")
    values = dict(
        arch="deepseek_v3",
        n_layers=61,
        expert_count=256,
        expert_layers=tuple(range(3, 61)),
        total_params=671_000_000_000,
    )
    values.update(override)
    index = SimpleNamespace(**values)
    if message is None:
        assert acceptance.validate_cache(tmp_path, index, 61, budget) == 3
    else:
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            acceptance.validate_cache(tmp_path, index, 61, budget)
