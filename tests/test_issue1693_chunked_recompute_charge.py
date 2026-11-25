"""#1693 — the pre-flight charges the last decoder layer's backward, not only the loss.

With ``training.stream_loss_chunk`` the loss stops being the peak of a streamed step.
A CUDA memory snapshot of one chunked Llama-3.1-8B NF4 step at 1024 tokens (RTX 3060
Laptop) put it in the backward of decoder layer 31, at the end of its checkpoint
recompute: the layer's seven dequantised bf16 weights (436,207,616 bytes), the 31
earlier layers' saved hidden states and the layer's recomputed activations, with
the chunked loss buffers already freed. The pre-flight charged none of that, and
under-predicted all three chunked rows of the issue's table.

``peak = resident + boundaries + max(loss phase, recompute phase)``: the loss phase
is the transient, logits and chunk terms as before; the recompute phase is
``estimate_recompute_bytes``. The six rows below are the issue's table (two passes,
one card, one stack; 2026-10-28), predicted from the inputs that run's pre-flight
saw.
"""

import math
from types import SimpleNamespace

import pytest

from soup_cli.trainer.stream_setup import StreamingSetupMixin
from soup_cli.utils import layer_stream
from soup_cli.utils.layer_stream import (
    estimate_activation_bytes,
    estimate_logits_bytes,
    estimate_loss_chunk_bytes,
    estimate_recompute_bytes,
    estimate_stream_peak_vram,
)

_GB = 1e9

# Llama-3.1-8B: hidden, intermediate, layers, vocabulary, heads, KV heads.
_H, _I, _L, _V, _HEADS, _KV_HEADS, _HEAD_DIM = 4096, 14_336, 32, 128_256, 32, 8, 128
_KV = _KV_HEADS * _HEAD_DIM

# One decoder layer's NF4 weights, by the name the sharder's index keys them.
_NF4_SHAPES = {
    "self_attn.q_proj.weight": (_H, _H),
    "self_attn.k_proj.weight": (_KV, _H),
    "self_attn.v_proj.weight": (_KV, _H),
    "self_attn.o_proj.weight": (_H, _H),
    "mlp.gate_proj.weight": (_I, _H),
    "mlp.up_proj.weight": (_I, _H),
    "mlp.down_proj.weight": (_H, _I),
}
_DENSE = sum(math.prod(shape) for shape in _NF4_SHAPES.values())


def _nf4_layer_bytes():
    """One layer's buffer as the sharder stores it under double quantisation:
    packed nibbles, a uint8 absmax per 64-block, an fp32 absmax per 256 of those
    and an fp32 offset per weight, plus the two bf16 RMSNorm weights."""
    total = 2 * _H * 2
    for shape in _NF4_SHAPES.values():
        elements = math.prod(shape)
        total += elements // 2 + elements // 64 + 4 * -(-elements // 64 // 256) + 4
    return total


# (seq, chunk) -> (predicted bytes, real peak GB, the prediction before #1693 in GB).
# The real peaks are the harness's rows; 1024/128 is the snapshot's exact peak.
_ROWS = {
    (512, None): (2_447_654_552, 2.222, 2.448),
    (512, 128): (2_058_960_536, 1.985, 1.767),
    (1024, None): (3_538_960_024, 3.153, 3.539),
    (1024, 128): (2_325_364_376, 2_230_982_656 / _GB, 1.947),
    (2048, None): (5_721_570_968, 5.015, 5.722),
    (2048, 128): (2_858_172_056, 2.722, 2.308),
}
_CHUNKED = [row for row in _ROWS if row[1] is not None]
_UNCHUNKED = [row for row in _ROWS if row[1] is None]


def _model_config(**overrides):
    fields = dict(
        vocab_size=_V,
        hidden_size=_H,
        intermediate_size=_I,
        num_hidden_layers=_L,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
    )
    fields.update(overrides)
    return SimpleNamespace(**{key: value for key, value in fields.items() if value is not None})


def _tcfg(chunk=None, lora=None, **overrides):
    """The harness's arm: batch 1, two buffers, LoRA r=8 on q_proj / v_proj, the
    schema's default dropout of 0.05."""
    fields = dict(
        batch_size=1,
        stream_buffers=2,
        stream_vram_probe=False,
        stream_vram_override=1 << 40,
        gradient_accumulation_steps=1,
        stream_loss_chunk=chunk,
        lora=lora
        if lora is not None
        else SimpleNamespace(r=8, target_modules=["q_proj", "v_proj"], dropout=0.05),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _index(nf4=True):
    specs = {key: SimpleNamespace(shape=shape) for key, shape in _NF4_SHAPES.items()}
    return SimpleNamespace(n_layers=_L, total_params=0, quant_specs=specs if nf4 else {})


def _preflight(
    monkeypatch,
    seq=1024,
    chunk=128,
    *,
    mixin=None,
    model_config=None,
    tcfg=None,
    index=None,
    target_modules=("q_proj", "v_proj"),
):
    """Run the real pre-flight off CUDA; return what both estimates were called with."""
    seen = {}
    real_recompute = layer_stream.estimate_recompute_bytes
    real_peak = layer_stream.estimate_stream_peak_vram

    def recompute(**kwargs):
        seen["recompute"] = kwargs
        seen["recompute_bytes"] = real_recompute(**kwargs)
        return seen["recompute_bytes"]

    def peak(**kwargs):
        seen["peak"] = kwargs
        seen["predicted"] = real_peak(**kwargs)
        return seen["predicted"]

    monkeypatch.setattr(layer_stream, "estimate_recompute_bytes", recompute)
    monkeypatch.setattr(layer_stream, "estimate_stream_peak_vram", peak)
    # The harness's card measured no more than the shipped constant.
    monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 14.0)
    mixin = mixin if mixin is not None else StreamingSetupMixin()
    mixin.device = "cpu"
    mixin._stream_budget_lines(
        SimpleNamespace(data=SimpleNamespace(max_length=seq)),
        tcfg if tcfg is not None else _tcfg(chunk),
        model_config=model_config if model_config is not None else _model_config(),
        layer_bytes=_nf4_layer_bytes(),
        embed_bytes=2 * _H,  # the final norm
        index=index if index is not None else _index(),
        on_cuda=False,
        large_layer_bytes=_V * _H * 2,  # one embed_tokens / lm_head slot
        target_modules=None if target_modules is None else list(target_modules),
    )
    return seen


def _loss_phase(kwargs):
    """The transient, logits and chunk terms of ``estimate_stream_peak_vram``."""
    seq, batch, chunk = kwargs["seq_len"], kwargs["batch_size"], kwargs["loss_chunk"]
    shape = dict(hidden_size=kwargs["hidden_size"], seq_len=seq, batch_size=batch)
    activations = estimate_activation_bytes(
        intermediate_size=kwargs["intermediate_size"], n_layers=kwargs["n_layers"], **shape
    )
    logits = estimate_logits_bytes(
        vocab_size=kwargs["vocab_size"],
        seq_len=seq if chunk is None else min(seq, chunk),
        batch_size=batch,
        bytes_per_element=kwargs["logits_bytes_per_element"],
    )
    kept = 0 if chunk is None else estimate_loss_chunk_bytes(**shape)
    return activations - _boundaries(kwargs) + logits + kept


def _boundaries(kwargs):
    return 2 * kwargs["n_layers"] * kwargs["hidden_size"] * kwargs["seq_len"] * kwargs[
        "batch_size"
    ]


class TestTheMeasuredRows:
    def test_the_inputs_are_the_runs(self, monkeypatch):
        """The panel of that run: 2 x 113 MB decoder buffers + 1 x 1051 MB slot =
        1276 MB, and 2.448 / 1.767 GB at 512 tokens before #1693."""
        assert round(_nf4_layer_bytes() / 1e6) == 113
        assert round((2 * _nf4_layer_bytes() + _V * _H * 2) / 1e6) == 1276
        for chunk, before in ((None, 2.448), (128, 1.767)):
            kwargs = _preflight(monkeypatch, 512, chunk)["peak"]
            old = estimate_stream_peak_vram(**{**kwargs, "recompute_bytes": 0})
            assert round(old / _GB, 3) == before

    @pytest.mark.parametrize("row", list(_ROWS))
    def test_each_row_is_predicted_exactly(self, monkeypatch, row):
        predicted, _real, _before = _ROWS[row]

        assert _preflight(monkeypatch, *row)["predicted"] == predicted

    @pytest.mark.parametrize("row", list(_ROWS))
    def test_no_row_under_predicts(self, monkeypatch, row):
        _predicted, real, _before = _ROWS[row]

        assert _preflight(monkeypatch, *row)["predicted"] >= real * _GB

    @pytest.mark.parametrize("row", _CHUNKED)
    def test_the_chunked_rows_keep_the_headroom_the_docs_state(self, monkeypatch, row):
        """1.037 to 1.050, from one card and one stack; no separate margin."""
        _predicted, real, _before = _ROWS[row]
        ratio = _preflight(monkeypatch, *row)["predicted"] / (real * _GB)

        assert 1.037 <= ratio < 1.0505

    @pytest.mark.parametrize("row", _CHUNKED)
    def test_the_chunked_rows_were_under_predicted_before(self, monkeypatch, row):
        _predicted, real, before = _ROWS[row]
        kwargs = _preflight(monkeypatch, *row)["peak"]

        old = estimate_stream_peak_vram(**{**kwargs, "recompute_bytes": 0})
        assert round(old / _GB, 3) == before < real

    @pytest.mark.parametrize("row", _UNCHUNKED)
    def test_the_unchunked_rows_are_the_predictions_they_were(self, monkeypatch, row):
        predicted, _real, before = _ROWS[row]
        kwargs = _preflight(monkeypatch, *row)["peak"]

        assert estimate_stream_peak_vram(**{**kwargs, "recompute_bytes": 0}) == predicted
        assert round(predicted / _GB, 3) == before

    @pytest.mark.parametrize("row", list(_ROWS))
    def test_chunked_rows_peak_in_the_recompute_and_unchunked_in_the_loss(
        self, monkeypatch, row
    ):
        seen = _preflight(monkeypatch, *row)
        kwargs, recompute = seen["peak"], seen["recompute_bytes"]

        if row[1] is None:
            assert _loss_phase(kwargs) > recompute
        else:
            assert recompute > _loss_phase(kwargs)
            resident = seen["predicted"] - _boundaries(kwargs) - recompute
            assert resident == estimate_stream_peak_vram(
                **{**kwargs, "recompute_bytes": 0}
            ) - _boundaries(kwargs) - _loss_phase(kwargs)


def _charge(**overrides):
    kwargs = dict(
        hidden_size=_H,
        intermediate_size=_I,
        kv_size=_KV,
        n_heads=_HEADS,
        seq_len=1,
        lora_targets=2,
        lora_dropout=0.05,
    )
    kwargs.update(overrides)
    return estimate_recompute_bytes(**kwargs)


class TestTheRecomputeCharge:
    def test_one_token_of_the_measured_layer(self):
        """8H + e(3H + 2KV) + 4 heads + t(1 + e)H + 6eI, at e = 2 and t = 2."""
        assert _charge() == 8 * _H + 2 * (3 * _H + 2 * _KV) + 4 * _HEADS + 2 * 3 * _H + 12 * _I
        assert _charge() == 258_176

    @pytest.mark.parametrize(
        "field, step",
        [
            ("hidden_size", 8 + 2 * 3 + 2 * (1 + 2)),  # norms, q/out/layer output, LoRA
            ("intermediate_size", 6 * 2),
            ("kv_size", 2 * 2),
            ("n_heads", 4),
            ("lora_targets", (1 + 2) * _H),
        ],
    )
    def test_each_term_moves_the_charge_by_its_coefficient(self, field, step):
        base = {"hidden_size": _H, "intermediate_size": _I, "kv_size": _KV}
        base.update({"n_heads": _HEADS, "lora_targets": 2})

        assert _charge(**{field: base[field] + 1}) - _charge() == step

    def test_it_scales_with_every_token_of_every_row(self):
        assert _charge(seq_len=1024) == 1024 * _charge()
        assert _charge(seq_len=1024, batch_size=3) == 3 * 1024 * _charge()

    def test_the_dense_weights_are_charged_in_the_stream_dtype(self):
        assert _charge(dense_layer_elements=_DENSE) - _charge() == 436_207_616
        assert _charge(dense_layer_elements=7, dtype="float32") - _charge(dtype="float32") == 28

    def test_a_dropout_of_zero_saves_no_mask(self):
        assert _charge() - _charge(lora_dropout=0.0) == 2 * _H
        assert _charge(lora_dropout=0.5) == _charge()

    def test_more_lora_targets_raise_it(self):
        assert _charge(lora_targets=4) > _charge() > _charge(lora_targets=0)

    def test_the_stream_dtype_sizes_every_tensor_but_the_fp32_ones(self):
        per_token = 8 * _H + 4 * (3 * _H + 2 * _KV) + 4 * _HEADS + 2 * 5 * _H + 24 * _I

        assert _charge(dtype="float32") == per_token

    @pytest.mark.parametrize(
        "field",
        ["hidden_size", "intermediate_size", "kv_size", "n_heads", "seq_len", "batch_size"],
    )
    def test_a_shape_that_is_not_positive_is_refused(self, field):
        with pytest.raises(ValueError, match=f"{field} must be positive"):
            _charge(**{field: 0})


def _peak(**overrides):
    shape = {
        "layer_bytes": 300_000_000,
        "buffers": 2,
        "extras_bytes": 0,
        "adapter_params": 10_000_000,
        "vocab_size": _V,
        "hidden_size": _H,
        "intermediate_size": _I,
        "n_layers": _L,
        "seq_len": 2048,
        "batch_size": 1,
        "loss_chunk": 128,
        "logits_bytes_per_element": None,
    }
    shape.update(overrides)
    return estimate_stream_peak_vram(**shape), {**shape, "logits_bytes_per_element": 14.0}


class TestTheLargerPhaseIsCharged:
    def test_unset_it_is_the_prediction_it_was(self):
        assert _peak(recompute_bytes=0)[0] == _peak()[0]

    @pytest.mark.parametrize("chunk", [None, 128])
    def test_a_recompute_phase_up_to_the_loss_phase_changes_nothing(self, chunk):
        base, kwargs = _peak(loss_chunk=chunk)

        assert _peak(loss_chunk=chunk, recompute_bytes=_loss_phase(kwargs))[0] == base

    @pytest.mark.parametrize("chunk", [None, 128])
    def test_a_larger_one_replaces_the_loss_phase(self, chunk):
        base, kwargs = _peak(loss_chunk=chunk)
        larger = _loss_phase(kwargs) + 12_345

        assert _peak(loss_chunk=chunk, recompute_bytes=larger)[0] == base + 12_345

    def test_the_boundaries_are_charged_beside_either_phase(self):
        """One more layer is one more saved hidden state, whichever phase peaks."""
        boundary = 2 * _H * 2048
        for recompute in (0, 10 * _GB):
            assert (
                _peak(n_layers=_L + 1, recompute_bytes=int(recompute))[0]
                - _peak(recompute_bytes=int(recompute))[0]
                == boundary
            )

    def test_the_recompute_phase_sits_on_the_resident_terms_and_the_boundaries(self):
        charge = int(10 * _GB)
        predicted, kwargs = _peak(recompute_bytes=charge)

        resident = _peak()[0] - _boundaries(kwargs) - _loss_phase(kwargs)
        assert predicted == resident + _boundaries(kwargs) + charge


class TestTheCallSite:
    def test_it_passes_the_shape_the_lora_and_the_dense_weights(self, monkeypatch):
        seen = _preflight(monkeypatch, 2048, 128)

        assert seen["recompute"] == dict(
            hidden_size=_H,
            intermediate_size=_I,
            kv_size=1024,
            n_heads=32,
            seq_len=2048,
            batch_size=1,
            lora_targets=2,
            lora_dropout=0.05,
            dense_layer_elements=218_103_808,
        )
        assert seen["peak"]["recompute_bytes"] == seen["recompute_bytes"] > 0

    def test_a_bf16_base_is_charged_no_dense_weights(self, monkeypatch):
        seen = _preflight(monkeypatch, index=_index(nf4=False))

        assert seen["recompute"]["dense_layer_elements"] == 0

    def test_an_index_without_quant_specs_is_a_bf16_base(self, monkeypatch):
        seen = _preflight(monkeypatch, index=SimpleNamespace(n_layers=_L, total_params=0))

        assert seen["recompute"]["dense_layer_elements"] == 0

    def test_the_kv_width_reads_head_dim_when_the_config_has_one(self, monkeypatch):
        seen = _preflight(monkeypatch, model_config=_model_config(head_dim=256))

        assert seen["recompute"]["kv_size"] == 8 * 256

    def test_without_head_dim_it_is_hidden_over_the_heads(self, monkeypatch):
        seen = _preflight(monkeypatch, model_config=_model_config(head_dim=None))

        assert seen["recompute"]["kv_size"] == 8 * (_H // 32)

    def test_without_kv_heads_it_is_multi_head_attention(self, monkeypatch):
        seen = _preflight(monkeypatch, model_config=_model_config(num_key_value_heads=None))

        assert seen["recompute"]["kv_size"] == 32 * _HEAD_DIM

    def test_without_a_head_count_it_is_one_head_as_wide_as_hidden(self, monkeypatch):
        config = _model_config(num_attention_heads=None, num_key_value_heads=None, head_dim=None)
        seen = _preflight(monkeypatch, model_config=config)

        assert (seen["recompute"]["n_heads"], seen["recompute"]["kv_size"]) == (1, _H)

    def test_the_resolved_targets_are_counted(self, monkeypatch):
        seen = _preflight(monkeypatch, target_modules=("q_proj", "k_proj", "v_proj"))

        assert seen["recompute"]["lora_targets"] == 3

    def test_without_resolved_targets_the_configured_list_is_counted(self, monkeypatch):
        lora = SimpleNamespace(r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "x"])
        seen = _preflight(monkeypatch, tcfg=_tcfg(128, lora=lora), target_modules=None)

        assert seen["recompute"]["lora_targets"] == 5

    def test_a_target_set_that_is_not_a_list_counts_as_four(self, monkeypatch):
        lora = SimpleNamespace(r=8, target_modules="auto", dropout=0.05)
        seen = _preflight(monkeypatch, tcfg=_tcfg(128, lora=lora), target_modules=None)

        assert seen["recompute"]["lora_targets"] == 4

    def test_no_lora_block_counts_four_targets_with_a_mask(self, monkeypatch):
        tcfg = _tcfg(128)
        del tcfg.lora
        mixin = StreamingSetupMixin()
        mixin._estimate_adapter_params = lambda *args, **kwargs: 0
        seen = _preflight(monkeypatch, mixin=mixin, tcfg=tcfg, target_modules=None)

        assert seen["recompute"]["lora_targets"] == 4
        assert seen["recompute"]["lora_dropout"] > 0

    def test_the_configured_dropout_is_passed(self, monkeypatch):
        lora = SimpleNamespace(r=8, target_modules=["q_proj", "v_proj"], dropout=0.0)
        seen = _preflight(monkeypatch, tcfg=_tcfg(128, lora=lora))

        assert seen["recompute"]["lora_dropout"] == 0.0

    def test_a_lora_block_without_a_dropout_is_charged_the_mask(self, monkeypatch):
        lora = SimpleNamespace(r=8, target_modules=["q_proj", "v_proj"])
        seen = _preflight(monkeypatch, tcfg=_tcfg(128, lora=lora))

        assert seen["recompute"]["lora_dropout"] > 0

    def test_a_paired_loss_charges_both_rows(self, monkeypatch):
        class _Paired(StreamingSetupMixin):
            _STREAM_ROWS_PER_EXAMPLE = 2

        seen = _preflight(monkeypatch, mixin=_Paired())

        assert seen["recompute"]["batch_size"] == 2 == seen["peak"]["batch_size"]

    def test_a_real_setup_passes_its_models_shape(self, tmp_path, monkeypatch):
        """The tiny Llama of the streamed suite: hidden 64, 4 heads, 2 KV heads of
        16, LoRA on q_proj / v_proj at the default dropout, a bf16-sized base."""
        from tests.test_v07204 import _build_streamed_wrapper

        real = layer_stream.estimate_recompute_bytes
        seen = {}

        def recording(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(layer_stream, "estimate_recompute_bytes", recording)
        wrapper, _, _ = _build_streamed_wrapper(tmp_path, monkeypatch, task="sft")
        wrapper._close_stream_runtime()

        assert seen == dict(
            hidden_size=64,
            intermediate_size=128,
            kv_size=32,
            n_heads=4,
            seq_len=64,
            batch_size=1,
            lora_targets=2,
            lora_dropout=0.05,
            dense_layer_elements=0,
        )


def test_a_real_nf4_setup_charges_its_layers_dense_weights(tmp_path, monkeypatch):
    """The sharder's index of the tiny Llama: q/o 64 x 64, k/v 32 x 64 and
    gate/up/down 128 x 64, all NF4 (on the CPU: bitsandbytes has no 4-bit MPS
    kernels)."""
    pytest.importorskip("bitsandbytes")
    from tests.test_v07204 import _build_streamed_wrapper

    real = layer_stream.estimate_recompute_bytes
    seen = {}

    def recording(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(layer_stream, "estimate_recompute_bytes", recording)
    wrapper, _, _ = _build_streamed_wrapper(
        tmp_path, monkeypatch, task="sft", quantization="4bit", device="cpu"
    )
    wrapper._close_stream_runtime()

    assert seen["dense_layer_elements"] == 2 * 64 * 64 + 2 * 32 * 64 + 3 * 128 * 64
