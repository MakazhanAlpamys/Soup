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

    def test_an_unknown_layout_is_refused(self):
        with pytest.raises(ValueError, match="stream_expert_layout"):
            load_config_from_string(_cfg(stream_layers="true", stream_expert_layout="experts"))

    def test_per_expert_refuses_adapters_on_the_routed_experts(self):
        with pytest.raises(ValueError) as exc:
            load_config_from_string(
                _cfg(
                    stream_layers="true",
                    stream_expert_layout="per_expert",
                    moe_lora="true",
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

    calls = []
    with pytest.raises(ValueError) as exc:
        _drive_qwen4_streaming_setup(
            tmp_path, monkeypatch, arch="llama", model_cfg=_deepseek_cfg("llama"),
            shard_calls=calls, tcfg_overrides={"stream_expert_layout": "per_expert"},
        )
    text = str(exc.value)
    assert "training.stream_expert_layout='per_expert'" in text
    assert "deepseek_v3" in text and "llama" in text
    assert calls == []


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


def _index(layout):
    shapes = {"gate_up": (64, 32), "down": (32, 32)} if layout else {}
    return SimpleNamespace(
        n_layers=4,
        quant_specs={},
        dtype="bfloat16",
        expert_layout=layout,
        expert_shapes=shapes,
        expert_quant_specs={},
    )


def _charged(monkeypatch, layout):
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
        _TrainConfig(),
        model_config=_ModelConfig(),
        layer_bytes=1000,
        embed_bytes=0,
        index=_index(layout),
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


def test_a_fused_store_is_charged_nothing_extra(monkeypatch):
    charged, text = _charged(monkeypatch, "")
    assert charged == 0
    assert "stream_expert_layout" not in text
