"""An FP8 block-quantised base sharded as NF4 uses double-quant OFF unless told otherwise."""

from __future__ import annotations

import types

import pytest

from soup_cli.utils.fp8_block import resolve_fp8_double_quant

BLOCK = (128, 128)
DEEPSEEK_FP8 = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}
DOUBLE_QUANT_REFUSAL = (
    "training.bnb_4bit_use_double_quant=true is not supported for an FP8 "
    "block-quantised base sharded as NF4: with double-quant on, the NF4 codec "
    "exceeds the per-block reconstruction bound on a block-scaled weight whose "
    "block scales span a wide range, and stays within it with double-quant off. "
    "Leave it unset or set it to false; the cost is about 0.37 bit per parameter "
    "of extra NF4 statistics."
)


@pytest.mark.parametrize(
    ("requested", "fp8_block", "quant_nf4", "expected"),
    [
        pytest.param(None, BLOCK, True, False, id="fp8-nf4-unset-is-off"),
        pytest.param(False, BLOCK, True, False, id="fp8-nf4-false-is-off"),
        pytest.param(None, None, True, True, id="plain-nf4-unset-keeps-on"),
        pytest.param(True, None, True, True, id="plain-nf4-true-stays-on"),
        pytest.param(False, None, True, False, id="plain-nf4-false-stays-off"),
        pytest.param(None, BLOCK, False, True, id="fp8-without-nf4-keeps-the-default"),
        pytest.param(True, BLOCK, False, True, id="fp8-without-nf4-true-is-not-refused"),
    ],
)
def test_the_resolution_table(requested, fp8_block, quant_nf4, expected):
    assert (
        resolve_fp8_double_quant(requested, fp8_block=fp8_block, quant_nf4=quant_nf4)
        is expected
    )


def test_an_explicit_true_is_refused_for_an_fp8_nf4_source():
    with pytest.raises(ValueError) as exc_info:
        resolve_fp8_double_quant(True, fp8_block=BLOCK, quant_nf4=True)
    assert str(exc_info.value) == DOUBLE_QUANT_REFUSAL


def _drive(tmp_path, monkeypatch, *, requested, quantization="4bit", **overrides):
    from tests.test_issue602_qwen4_ple_streaming import _drive_qwen4_streaming_setup

    shard_calls: list = []
    # The harness fakes ``transformers``: no meta skeleton and no BitsAndBytesConfig. Probe
    # and NF4 config are replaced as the serialized-source setup tests do.
    monkeypatch.setattr(
        "soup_cli.trainer.stream_setup.StreamingSetupMixin._stream_meta_skeleton",
        classmethod(lambda cls, base, *, dtype, quant: types.SimpleNamespace()),
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.quantised_layer_suffixes",
        lambda _model: frozenset({"self_attn.q_proj.weight"}),
    )
    monkeypatch.setattr(
        "soup_cli.utils.moe.resolve_moe_lora_targets",
        lambda _probe, _tcfg, targets, **_kwargs: targets,
    )
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.build_nf4_config",
        lambda dtype, *, double_quant=True: types.SimpleNamespace(double_quant=double_quant),
    )
    model_cfg = types.SimpleNamespace(
        model_type="deepseek_v3", hidden_size=4, num_hidden_layers=1, vocab_size=8,
        intermediate_size=8, quantization_config=DEEPSEEK_FP8,
    )
    tcfg_overrides = {
        "quantization": quantization,
        "bnb_4bit_use_double_quant": requested,
        "double_quant_on": requested is not False,
        **overrides,
    }
    _drive_qwen4_streaming_setup(
        tmp_path, monkeypatch, arch="deepseek_v3", model_cfg=model_cfg,
        shard_calls=shard_calls, tcfg_overrides=tcfg_overrides,
    )
    return shard_calls


@pytest.mark.parametrize("requested", [None, False])
def test_the_setup_shards_an_fp8_nf4_base_without_double_quant(tmp_path, monkeypatch, requested):
    calls = _drive(tmp_path, monkeypatch, requested=requested)
    assert [call["double_quant"] for call in calls] == [False]


def test_the_setup_refuses_double_quant_true_for_an_fp8_nf4_base(tmp_path, monkeypatch):
    with pytest.raises(ValueError) as exc_info:
        _drive(tmp_path, monkeypatch, requested=True)
    assert str(exc_info.value) == DOUBLE_QUANT_REFUSAL


def test_the_setup_refuses_the_scale_cache_for_an_fp8_nf4_base(tmp_path, monkeypatch):
    with pytest.raises(ValueError) as exc_info:
        _drive(tmp_path, monkeypatch, requested=None, stream_nf4_scale_cache=True)
    assert str(exc_info.value) == (
        "training.stream_nf4_scale_cache caches the expanded nested scales of "
        "double quantisation, and an FP8 block-quantised base is sharded "
        "without it; unset stream_nf4_scale_cache for this base."
    )


def test_an_fp8_base_that_is_not_nf4_keeps_the_shipped_default(tmp_path, monkeypatch):
    calls = _drive(tmp_path, monkeypatch, requested=None, quantization="none")
    assert [call["double_quant"] for call in calls] == [True]
