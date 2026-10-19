"""``training.stream_expert_layout``: expert-granularity streaming from ``soup train``.

``per_expert`` shards a DeepSeek-V3-family base with each MoE layer's routed
experts in their own file and streams one expert at a time
(``utils/expert_stream.py``), so the VRAM bound no longer grows with the expert
count. The setup passes the layout to the sharder and to the pre-flight's cache
check, charges the expert path's VRAM, and refuses an architecture the layout is
not gated on before anything is downloaded or written.
"""

from types import SimpleNamespace

import pytest

from soup_cli.config.loader import load_config_from_string


def _cfg(**training):
    body = "\n".join(f"  {k}: {v}" for k, v in training.items())
    return (
        "base: HuggingFaceTB/SmolLM2-135M\n"
        "task: sft\n"
        "data:\n"
        "  train: d.jsonl\n"
        "  max_length: 512\n"
        "training:\n"
        "  batch_size: 1\n"
        "  lora:\n"
        "    r: 8\n" + body + "\n"
    )


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
class TestSchema:
    def test_default_is_fused(self):
        cfg = load_config_from_string(_cfg(stream_layers="true"))
        assert cfg.training.stream_expert_layout == "fused"

    def test_per_expert_parses_with_streaming_on(self):
        cfg = load_config_from_string(
            _cfg(stream_layers="true", quantization="4bit", stream_expert_layout="per_expert")
        )
        assert cfg.training.stream_expert_layout == "per_expert"

    def test_per_expert_without_streaming_is_the_stream_footgun(self):
        with pytest.raises(ValueError) as exc:
            load_config_from_string(_cfg(stream_expert_layout="per_expert"))
        text = str(exc.value)
        assert "training.stream_expert_layout" in text
        assert "stream_layers is false" in text

    def test_an_explicit_fused_without_streaming_is_not_the_footgun(self):
        cfg = load_config_from_string(_cfg(stream_expert_layout="fused"))
        assert cfg.training.stream_expert_layout == "fused"

    def test_an_unknown_layout_is_refused(self):
        with pytest.raises(ValueError, match="stream_expert_layout"):
            load_config_from_string(_cfg(stream_layers="true", stream_expert_layout="experts"))

    @pytest.mark.parametrize("quantization", ["none", "4bit", "8bit"])
    def test_per_expert_refuses_adapters_on_the_routed_experts(self, quantization):
        with pytest.raises(ValueError) as exc:
            load_config_from_string(
                _cfg(
                    stream_layers="true",
                    stream_expert_layout="per_expert",
                    moe_lora="true",
                    quantization=quantization,
                )
            )
        text = str(exc.value)
        assert "training.stream_expert_layout='per_expert'" in text
        assert "moe_lora" in text


# --------------------------------------------------------------------------
# setup wiring (the shared driver in tests/test_issue602_qwen4_ple_streaming.py)
# --------------------------------------------------------------------------
def _deepseek_cfg(model_type="deepseek_v3"):
    return SimpleNamespace(
        model_type=model_type,
        hidden_size=4,
        num_hidden_layers=1,
        vocab_size=8,
        intermediate_size=8,
    )


@pytest.mark.parametrize(
    ("layout", "expected"),
    [("per_expert", "per_expert"), ("fused", "")],
    ids=["per_expert", "fused"],
)
def test_setup_hands_the_layout_to_the_sharder(tmp_path, monkeypatch, layout, expected):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    calls = []
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=_deepseek_cfg(),
        shard_calls=calls, tcfg_overrides={"stream_expert_layout": layout},
    )
    assert len(calls) == 1
    assert calls[0]["expert_layout"] == expected


def test_a_fake_training_config_without_the_field_still_shards_fused(tmp_path, monkeypatch):
    """Older trainer configs and test fakes have no attribute: fused, as before."""
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    calls = []
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=_deepseek_cfg(), shard_calls=calls
    )
    assert calls[0]["expert_layout"] == ""


def test_setup_refuses_per_expert_for_an_ungated_arch_before_sharding(tmp_path, monkeypatch):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    def _resolve_must_not_run(*_args, **_kwargs):
        raise AssertionError("weights were resolved before the refusal")

    calls = []
    with pytest.raises(ValueError) as exc:
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, resolve_weights=_resolve_must_not_run, arch="llama",
            model_cfg=_deepseek_cfg("llama"), shard_calls=calls,
            tcfg_overrides={"stream_expert_layout": "per_expert"},
        )
    text = str(exc.value)
    assert "training.stream_expert_layout='per_expert'" in text
    assert "deepseek_v3" in text and "llama" in text
    assert calls == []


def _captured_setup_output(tmp_path, monkeypatch, *, layout, disk_kind, **driver_kwargs):
    import io
    import re

    from rich.console import Console

    import soup_cli.trainer.stream_setup as stream_setup
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    buffer = io.StringIO()
    monkeypatch.setattr(stream_setup, "console", Console(file=buffer, width=400))
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=_deepseek_cfg(),
        tcfg_overrides={"stream_expert_layout": layout}, disk_kind=disk_kind,
        **driver_kwargs,
    )
    plain = re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue())
    return " ".join(plain.split())


@pytest.mark.parametrize("disk_kind", ["ssd", "hdd", "unknown"])
def test_a_non_nvme_cache_volume_is_warned_about_for_per_expert(
    tmp_path, monkeypatch, disk_kind
):
    text = _captured_setup_output(
        tmp_path, monkeypatch, layout="per_expert", disk_kind=disk_kind
    )
    assert "reads every routed expert from the cache volume" in text
    assert f"'{disk_kind}', not NVMe" in text


def test_the_volume_probe_reads_the_cache_dir_with_the_users_override(tmp_path, monkeypatch):
    """The experts are read from the shard cache, not the weights, and a
    training.stream_disk_kind override wins there as it does for the tier."""
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    seen = []
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=_deepseek_cfg(),
        tcfg_overrides={"stream_expert_layout": "per_expert", "stream_disk_kind": "hdd"},
        disk_kind="hdd", disk_calls=seen,
    )
    assert (str(tmp_path / "shards"), "hdd") in seen, seen


def test_an_nvme_cache_volume_is_not_warned_about(tmp_path, monkeypatch):
    text = _captured_setup_output(tmp_path, monkeypatch, layout="per_expert", disk_kind="nvme")
    assert "not NVMe" not in text


def test_the_fused_layout_never_warns_about_the_volume(tmp_path, monkeypatch):
    text = _captured_setup_output(tmp_path, monkeypatch, layout="fused", disk_kind="hdd")
    assert "routed expert from the cache volume" not in text


def test_the_planner_checks_the_cache_for_the_requested_layout(tmp_path, monkeypatch):
    """A cached fused store must read as a miss for a per_expert request, or the
    disk pre-flight under-counts the re-shard it is about to write."""
    import types

    import torch
    from safetensors.torch import save_file

    import soup_cli.utils.layer_shard as layer_shard
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    seen = []
    monkeypatch.setattr(
        layer_shard,
        "inspect_shard_cache",
        lambda *args, **kwargs: (seen.append(kwargs), (None, "miss"))[1],
    )
    weights = tmp_path / "weights"
    weights.mkdir()
    blob = weights / "model.safetensors"
    save_file({"model.layers.0.proj.weight": torch.ones(2, 2)}, str(blob))
    (weights / "config.json").write_text('{"model_type": "deepseek_v3"}', encoding="utf-8")
    stat = blob.stat()
    plan = types.SimpleNamespace(
        weights_dir=str(weights),
        source_bytes=stat.st_size,
        materialized_copy_bytes=0,
        materialize_bytes=0,
        needs_materialization=False,
        source_files=((str(blob), stat.st_size, int(stat.st_mtime)),),
    )

    def _resolve(_base, *, before_materialize=None):
        if before_materialize is not None:
            before_materialize(plan)
        return str(weights)

    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, resolve_weights=_resolve, arch="deepseek_v3",
        model_cfg=_deepseek_cfg(), tcfg_overrides={"stream_expert_layout": "per_expert"},
    )
    assert [call.get("expert_layout") for call in seen] == ["per_expert"]


# --------------------------------------------------------------------------
# the pre-flight charges the expert path
# --------------------------------------------------------------------------
class _Lora:
    r = 8
    target_modules = ["q_proj", "v_proj"]


class _ModelConfig:
    vocab_size = 64
    hidden_size = 8
    intermediate_size = 16
    num_hidden_layers = 4


class _Config:
    data = SimpleNamespace(max_length=4)


class _TrainConfig:
    batch_size = 1
    stream_buffers = 2
    stream_vram_probe = False
    stream_nf4_scale_cache = False
    lora = _Lora()


class _ScaleCacheTrainConfig(_TrainConfig):
    stream_nf4_scale_cache = True


_SMALL_SHAPES = {"gate_up": (64, 32), "down": (32, 32)}


def _index(layout, *, dtype="bfloat16", quantised=False, shapes=None, quant_specs=None):
    shapes = (shapes or _SMALL_SHAPES) if layout else {}
    return SimpleNamespace(
        n_layers=4,
        quant_specs=quant_specs or {},
        dtype=dtype,
        expert_layout=layout,
        expert_shapes=shapes,
        expert_quant_specs={"gate_up": object(), "down": object()} if quantised else {},
    )


def _charged(monkeypatch, layout, *, tcfg=None, **index_kwargs):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils import layer_stream

    real = layer_stream.estimate_stream_peak_vram
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(layer_stream, "estimate_stream_peak_vram", capture)
    lines, _plan = StreamingSetupMixin()._stream_budget_lines(
        _Config(),
        tcfg or _TrainConfig(),
        model_config=_ModelConfig(),
        layer_bytes=1000,
        embed_bytes=0,
        index=_index(layout, **index_kwargs),
        on_cuda=False,
    )
    return seen.get("persistent_bytes", 0), "\n".join(lines)


def test_the_preflight_charges_one_expert_in_flight_for_per_expert(monkeypatch):
    from soup_cli.utils.expert_stream import expert_stream_vram_bytes

    charged, text = _charged(monkeypatch, "per_expert")
    want = expert_stream_vram_bytes(
        {"gate_up": (64, 32), "down": (32, 32)}, dtype_bytes=2, quantised=False
    )
    assert want > 0
    assert charged == want
    assert "experts" in text and "stream_expert_layout" in text


def test_the_experts_line_prints_the_charge_in_gb(monkeypatch):
    from soup_cli.utils.expert_stream import expert_stream_vram_bytes

    shapes = {"gate_up": (4096, 2048), "down": (2048, 2048)}
    charged, text = _charged(monkeypatch, "per_expert", shapes=shapes)
    want = expert_stream_vram_bytes(shapes, dtype_bytes=2, quantised=False)
    assert charged == want
    assert 0.001 < want / 1e9 < 1.0
    assert f"experts      {want / 1e9:.3f} GB for the routed expert in flight" in text


def test_the_charge_adds_the_scale_cache_to_the_expert_in_flight(monkeypatch):
    from soup_cli.utils import layer_stream
    from soup_cli.utils.expert_stream import expert_stream_vram_bytes

    quant_specs = {"w": SimpleNamespace(shape=(128, 128), blocksize=64, nested=True)}
    charged, text = _charged(
        monkeypatch, "per_expert", tcfg=_ScaleCacheTrainConfig(), quant_specs=quant_specs
    )
    experts = expert_stream_vram_bytes(_SMALL_SHAPES, dtype_bytes=2, quantised=False)
    cache = layer_stream.nf4_scale_cache_bytes(quant_specs, 4)
    assert cache > 0 and experts > 0
    assert charged == cache + experts
    assert "scale cache" in text and "experts" in text


@pytest.mark.parametrize(
    ("dtype", "width", "quantised"),
    [("float32", 4, False), ("bfloat16", 2, True), ("float32", 4, True)],
    ids=["fp32", "nf4-bf16", "nf4-fp32"],
)
def test_the_expert_charge_follows_the_store_dtype_and_quantisation(
    monkeypatch, dtype, width, quantised
):
    from soup_cli.utils.expert_stream import expert_stream_vram_bytes

    charged, _text = _charged(monkeypatch, "per_expert", dtype=dtype, quantised=quantised)
    shapes = {"gate_up": (64, 32), "down": (32, 32)}
    assert charged == expert_stream_vram_bytes(shapes, dtype_bytes=width, quantised=quantised)
    assert charged != expert_stream_vram_bytes(
        shapes, dtype_bytes=2 if width == 4 else 4, quantised=quantised
    )


def test_a_fused_store_is_charged_nothing_extra(monkeypatch):
    charged, text = _charged(monkeypatch, "")
    assert charged == 0
    assert "stream_expert_layout" not in text
