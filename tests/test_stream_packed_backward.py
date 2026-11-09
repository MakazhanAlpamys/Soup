"""Packed-state NF4 backward under the checkpoint recompute (layer streaming).

A streamed NF4 linear on bitsandbytes' dequantise + ``F.linear`` arm builds a dense
weight in the forward, and ``F.linear`` saves it for the backward. Under the default
per-layer checkpoint that is harmless in the forward (the checkpoint discards it) but
the layer's recompute rebuilds ALL of the layer's dense weights together and holds
them until the layer's backward has used each one: for Mistral Large 123B that is
2.77 GB per layer. ``packed_dequant_nf4_linear`` saves only the packed bytes and the
quantisation tensors and dequantises one matrix at a time, so a layer then holds at
most its largest matrix (704,643,072 bytes) dense.

There is no config key: ``_stream_budget_lines`` decides from the VRAM figures it
already computes. The dense-retaining arm fits -> nothing changes. It does not fit
and the packed arm is smaller -> the layers' NF4 linears are routed to the packed
Function and the prediction is charged on the largest matrix.

The Mistral Large rows are the shapes of ``unsloth/Mistral-Large-Instruct-2407-bnb-4bit``
(hidden 12288, intermediate 28672, 96 heads, 8 KV heads, head_dim 128, 88 layers) and
the harness' declared 6.25 GiB per-process cap (6,710,886,400 bytes).
"""

from __future__ import annotations

import contextlib
import io
import math
import re
import weakref
from types import SimpleNamespace

import pytest

_N_LAYERS = 4
_BUFFERS = 2
_TOKENS = 8
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text).split())


# --------------------------------------------------------------------------
# Mistral Large 123B, the shapes the pre-flight is asked about
# --------------------------------------------------------------------------
_H, _I, _L, _V, _HEADS, _KV_HEADS, _HEAD_DIM = 12_288, 28_672, 88, 32_768, 96, 8, 128
_KV = _KV_HEADS * _HEAD_DIM
_ML_SHAPES = {
    "self_attn.q_proj.weight": (_H, _H),
    "self_attn.k_proj.weight": (_KV, _H),
    "self_attn.v_proj.weight": (_KV, _H),
    "self_attn.o_proj.weight": (_H, _H),
    "mlp.gate_proj.weight": (_I, _H),
    "mlp.up_proj.weight": (_I, _H),
    "mlp.down_proj.weight": (_H, _I),
}
_ML_ELEMENTS = [math.prod(shape) for shape in _ML_SHAPES.values()]
_ML_DENSE = sum(_ML_ELEMENTS)  # 1,384,120,320 elements = 2.77 GB in bf16
_ML_LARGEST = max(_ML_ELEMENTS)  # one 28672 x 12288 matrix = 704,643,072 bytes in bf16
_ML_LAYER_BYTES = 714_074_140  # FULL35.txt: two decoder slots = 1,428,148,280
_ML_VOCAB_SLOT = _V * _H * 2  # 805,306,368
_ML_ADAPTER_PARAMS = _L * 2 * 2 * 8 * _H  # what _estimate_adapter_params charges q/v r=8
_HARNESS_CAP = 6_710_886_400  # 6.25 GiB, the harness' per-process torch cap
_CARD_8151_MIB = 8151 * 1024 * 1024
_GB = 1e9


def _ml_recompute(seq, elements):
    from soup_cli.utils.layer_stream import estimate_recompute_bytes

    return estimate_recompute_bytes(
        hidden_size=_H,
        intermediate_size=_I,
        kv_size=_KV,
        n_heads=_HEADS,
        seq_len=seq,
        batch_size=1,
        lora_targets=2,
        lora_dropout=0.05,
        dense_layer_elements=elements,
    )


def _ml_peak(seq, elements):
    from soup_cli.utils.layer_stream import estimate_stream_peak_vram

    return estimate_stream_peak_vram(
        layer_bytes=_ML_LAYER_BYTES,
        buffers=2,
        extras_bytes=2 * _H,
        adapter_params=_ML_ADAPTER_PARAMS,
        vocab_size=_V,
        hidden_size=_H,
        intermediate_size=_I,
        n_layers=_L,
        seq_len=seq,
        batch_size=1,
        logits_bytes_per_element=None,
        large_layer_bytes=_ML_VOCAB_SLOT,
        recompute_bytes=_ml_recompute(seq, elements),
    )


def _ml_index(nf4=True):
    specs = {key: SimpleNamespace(shape=shape) for key, shape in _ML_SHAPES.items()}
    return SimpleNamespace(n_layers=_L, total_params=0, quant_specs=specs if nf4 else {})


def _ml_model_config():
    return SimpleNamespace(
        vocab_size=_V,
        hidden_size=_H,
        intermediate_size=_I,
        num_hidden_layers=_L,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
    )


def _ml_budget(monkeypatch, *, seq, free, nf4=True, on_cuda=True, probe=False, mixin=None):
    """The REAL ``_stream_budget_lines`` for Mistral Large on a card with ``free`` bytes.

    Returns ``(mixin, lines, probe_plan, seen)``; ``seen`` has the kwargs each
    estimator was last called with. The device queries are patched, not skipped, so
    the same code runs on this box and on a GPU-less runner.
    """
    import torch

    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils import layer_stream

    seen = {}
    real_recompute = layer_stream.estimate_recompute_bytes

    def recompute(**kwargs):
        seen["recompute"] = kwargs
        seen.setdefault("recompute_calls", []).append(kwargs)
        return real_recompute(**kwargs)

    monkeypatch.setattr(layer_stream, "estimate_recompute_bytes", recompute)
    monkeypatch.setattr(layer_stream, "calibrated_logits_bytes_per_element", lambda: 14.0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a, **_k: (10**12, 10**12))
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.measure_gemm_tflops", lambda *_a, **_k: None
    )
    tcfg = SimpleNamespace(
        batch_size=1,
        stream_buffers=2,
        stream_vram_probe=probe,
        stream_vram_override=free,
        gradient_accumulation_steps=1,
        stream_loss_chunk=None,
        lora=SimpleNamespace(r=8, target_modules=["q_proj", "v_proj"], dropout=0.05),
    )
    mixin = mixin if mixin is not None else StreamingSetupMixin()
    mixin.device = "cuda" if on_cuda else "cpu"
    lines, plan = mixin._stream_budget_lines(
        SimpleNamespace(data=SimpleNamespace(max_length=seq)),
        tcfg,
        model_config=_ml_model_config(),
        layer_bytes=_ML_LAYER_BYTES,
        embed_bytes=2 * _H,
        index=_ml_index(nf4),
        on_cuda=on_cuda,
        large_layer_bytes=_ML_VOCAB_SLOT,
        target_modules=["q_proj", "v_proj"],
    )
    return mixin, lines, plan, seen


# --------------------------------------------------------------------------
# the arithmetic: whole layer -> largest matrix
# --------------------------------------------------------------------------
class TestDenseWeightCharge:
    def test_the_dense_arm_holds_every_matrix_of_the_layer(self):
        from soup_cli.utils.layer_stream import dense_weight_charge_elements

        assert dense_weight_charge_elements(_ML_ELEMENTS, packed_backward=False) == 1_384_120_320

    def test_the_packed_arm_holds_the_largest_matrix_only(self):
        from soup_cli.utils.layer_stream import dense_weight_charge_elements

        charge = dense_weight_charge_elements(_ML_ELEMENTS, packed_backward=True)
        assert charge == 352_321_536 == _I * _H
        assert charge * 2 == 704_643_072  # FULL35.txt "largest local BF16 matrix"

    def test_the_default_is_the_dense_arm(self):
        from soup_cli.utils.layer_stream import dense_weight_charge_elements

        assert dense_weight_charge_elements([3, 5, 7]) == 15

    @pytest.mark.parametrize("packed", [False, True])
    def test_a_base_without_nf4_charges_nothing(self, packed):
        from soup_cli.utils.layer_stream import dense_weight_charge_elements

        assert dense_weight_charge_elements([], packed_backward=packed) == 0

    @pytest.mark.parametrize("bad", [[-1], [True], [2.5]])
    def test_a_bad_matrix_size_is_refused(self, bad):
        from soup_cli.utils.layer_stream import dense_weight_charge_elements

        with pytest.raises(ValueError, match="matrix_elements"):
            dense_weight_charge_elements(bad, packed_backward=True)


class TestDecidePackedBackward:
    def _decide(self, dense, packed, available):
        from soup_cli.utils.layer_stream import decide_packed_backward

        return decide_packed_backward(
            dense_predicted_bytes=dense,
            packed_predicted_bytes=packed,
            available_bytes=available,
        )

    def test_a_dense_arm_that_fits_stays_dense(self):
        decision = self._decide(dense=6_000, packed=4_000, available=7_000)
        assert decision.packed is False
        assert decision.predicted_bytes == 6_000

    def test_the_boundary_is_the_pre_flights_own(self):
        """``decide_stream_fit`` accepts ``predicted == available``; so does this."""
        assert self._decide(6_000, 4_000, available=6_000).packed is False
        assert self._decide(6_000, 4_000, available=5_999).packed is True

    def test_a_dense_arm_that_does_not_fit_goes_packed(self):
        decision = self._decide(dense=6_000, packed=4_000, available=5_000)
        assert decision.packed is True
        assert decision.predicted_bytes == 4_000
        assert decision.fits is True

    def test_packed_is_chosen_even_when_it_does_not_fit_either(self):
        """It is the smaller peak, so it is the arm the refusal (or the probe) judges."""
        decision = self._decide(dense=6_000, packed=4_000, available=3_000)
        assert decision.packed is True
        assert decision.predicted_bytes == 4_000
        assert decision.fits is False

    def test_nothing_to_save_stays_dense(self):
        """No NF4 weight, or a single matrix per layer: both predictions are equal."""
        assert self._decide(dense=6_000, packed=6_000, available=1_000).packed is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dense": -1, "packed": 0, "available": 0},
            {"dense": 1, "packed": True, "available": 0},
            {"dense": 1, "packed": 0, "available": 1.5},
        ],
    )
    def test_bad_byte_counts_are_refused(self, kwargs):
        with pytest.raises(ValueError, match="byte counts must be non-negative ints"):
            self._decide(**kwargs)


class TestMistralLargeCharge:
    """The pre-flight's own estimators, on the shapes of the 123B run."""

    def test_the_recompute_charge_drops_by_the_difference_of_the_dense_terms(self):
        for seq in (128, 512):
            dense = _ml_recompute(seq, _ML_DENSE)
            packed = _ml_recompute(seq, _ML_LARGEST)
            assert dense - packed == 2 * (_ML_DENSE - _ML_LARGEST) == 2_063_597_568
        assert _ml_recompute(128, _ML_DENSE) == 2_844_311_552
        assert _ml_recompute(128, _ML_LARGEST) == 780_713_984
        assert _ml_recompute(512, _ML_DENSE) == 3_072_524_288
        assert _ml_recompute(512, _ML_LARGEST) == 1_008_926_720

    def test_predicted_peak_before_and_after(self):
        assert _ml_peak(128, _ML_DENSE) == 5_921_762_968
        assert _ml_peak(128, _ML_LARGEST) == 3_858_165_400
        assert _ml_peak(512, _ML_DENSE) == 6_980_447_896
        assert _ml_peak(512, _ML_LARGEST) == 4_916_850_328

    def test_512_tokens_fit_the_harness_cap_only_on_the_packed_arm(self):
        assert _ml_peak(512, _ML_DENSE) > _HARNESS_CAP
        assert _ml_peak(512, _ML_LARGEST) <= _HARNESS_CAP
        # 128 tokens fit either way, which is why the harness' 128 run needed no change.
        assert _ml_peak(128, _ML_DENSE) <= _HARNESS_CAP

    def test_the_8151_mib_card_is_not_the_discriminator(self):
        """By the formula the dense arm is under the card's whole 8151 MiB at 512
        tokens too (6657 MiB); what it cannot satisfy is a cap or a busy display."""
        assert _ml_peak(512, _ML_DENSE) < _CARD_8151_MIB
        assert _ml_peak(512, _ML_LARGEST) < _ml_peak(512, _ML_DENSE) - 1_900_000_000


# --------------------------------------------------------------------------
# the decision, from the real pre-flight
# --------------------------------------------------------------------------
class TestThePreflightDecides:
    def test_a_dense_arm_that_fits_changes_nothing(self, monkeypatch):
        mixin, lines, plan, seen = _ml_budget(monkeypatch, seq=512, free=_CARD_8151_MIB)
        assert mixin._stream_packed_backward is False
        assert seen["recompute"]["dense_layer_elements"] == _ML_DENSE
        assert lines[0] == (
            "  peak VRAM    ~6.98 GB at batch 1 x seq 512 (logits 0.23 GB)"
        )
        assert not any("packed" in line for line in lines)
        assert plan is None

    def test_a_dense_arm_that_does_not_fit_goes_packed_and_is_charged_on_one_matrix(
        self, monkeypatch
    ):
        mixin, lines, _plan, seen = _ml_budget(monkeypatch, seq=512, free=_HARNESS_CAP)
        assert mixin._stream_packed_backward is True
        assert seen["recompute"]["dense_layer_elements"] == _ML_LARGEST
        assert lines[0] == (
            "  peak VRAM    ~4.92 GB at batch 1 x seq 512 (logits 0.23 GB)"
        )
        said = _plain(" ".join(lines))
        assert "packed NF4" in said
        assert "6.98 GB" in said  # what the dense-retaining arm was predicted to need

    def test_the_same_card_at_128_tokens_keeps_the_dense_arm(self, monkeypatch):
        """The decision follows the pre-flight's numbers, not the model's name."""
        mixin, lines, _plan, seen = _ml_budget(monkeypatch, seq=128, free=_HARNESS_CAP)
        assert mixin._stream_packed_backward is False
        assert seen["recompute"]["dense_layer_elements"] == _ML_DENSE
        assert "~5.92 GB" in lines[0]

    @pytest.mark.parametrize("delta,packed", [(0, False), (-1, True)])
    def test_the_boundary_is_the_dense_predictions_own(self, monkeypatch, delta, packed):
        free = _ml_peak(512, _ML_DENSE) + delta
        mixin, _lines, _plan, _seen = _ml_budget(monkeypatch, seq=512, free=free)
        assert mixin._stream_packed_backward is packed

    def test_neither_arm_fitting_refuses_on_the_packed_prediction(self, monkeypatch):
        with pytest.raises(ValueError) as caught:
            _ml_budget(monkeypatch, seq=512, free=4_000_000_000)
        message = _plain(str(caught.value))
        assert "predicted to need 4.92 GB" in message
        assert "packed-state NF4 backward" in message

    def test_a_refusal_without_nf4_is_the_one_it_always_was(self, monkeypatch):
        with pytest.raises(ValueError) as caught:
            _ml_budget(monkeypatch, seq=512, free=4_000_000_000, nf4=False)
        message = _plain(str(caught.value))
        assert "a streaming step is predicted to need" in message
        assert "packed-state" not in message

    def test_a_bf16_base_never_goes_packed(self, monkeypatch):
        with pytest.raises(ValueError):
            _ml_budget(monkeypatch, seq=512, free=1_000_000_000, nf4=False)
        mixin, _lines, _plan, seen = _ml_budget(
            monkeypatch, seq=512, free=10**12, nf4=False
        )
        assert mixin._stream_packed_backward is False
        assert seen["recompute"]["dense_layer_elements"] == 0

    def test_off_cuda_there_is_no_free_figure_to_decide_on(self, monkeypatch):
        mixin, lines, _plan, seen = _ml_budget(
            monkeypatch, seq=512, free=1_000, on_cuda=False
        )
        assert mixin._stream_packed_backward is False
        assert seen["recompute"]["dense_layer_elements"] == _ML_DENSE
        assert "~6.98 GB" in lines[0]

    def test_the_probe_stays_authoritative_and_keeps_the_dense_arm(self, monkeypatch, capsys):
        """Over budget on the formula but with ``stream_vram_probe`` on: today's path.
        The packed arm is not even priced, the prediction, the plan and the notice
        are the dense arm's, and the probe is left to measure the arm that runs today."""
        mixin, lines, plan, seen = _ml_budget(
            monkeypatch, seq=512, free=_HARNESS_CAP, probe=True
        )
        assert mixin._stream_packed_backward is False
        assert [call["dense_layer_elements"] for call in seen["recompute_calls"]] == [_ML_DENSE]
        assert lines[0] == "  peak VRAM    ~6.98 GB at batch 1 x seq 512 (logits 0.23 GB)"
        assert not any("packed" in line for line in lines)
        assert plan.predicted_bytes == _ml_peak(512, _ML_DENSE)
        assert plan.available_bytes == _HARNESS_CAP
        said = _plain(capsys.readouterr().out)
        assert "Predicted over budget (6.98 GB vs 6.71 GB free)" in said
        assert "packed" not in said

    def test_the_same_config_without_the_probe_goes_packed(self, monkeypatch):
        """The contrast that makes the test above mean something."""
        mixin, _lines, plan, _seen = _ml_budget(
            monkeypatch, seq=512, free=_HARNESS_CAP, probe=False
        )
        assert mixin._stream_packed_backward is True
        assert plan is None

    def test_a_far_over_budget_probe_config_is_refused_as_it_always_was(self, monkeypatch):
        with pytest.raises(ValueError) as caught:
            _ml_budget(monkeypatch, seq=512, free=1_000_000_000, probe=True)
        message = _plain(str(caught.value))
        assert "cannot overrule a prediction this far over budget (7.0x)" in message
        assert "packed" not in message

    def test_nothing_to_save_records_no_packed_arm_when_it_refuses(self, monkeypatch):
        """Over budget with no NF4 weight: refused, and no packed arm recorded on the
        way (the setup never runs, but the flag is the pre-flight's own answer)."""
        from soup_cli.trainer.stream_setup import StreamingSetupMixin

        mixin = StreamingSetupMixin()
        with pytest.raises(ValueError, match="predicted to need"):
            _ml_budget(
                monkeypatch, seq=512, free=_ml_peak(512, 0) - 1, nf4=False, mixin=mixin
            )
        assert mixin._stream_packed_backward is False

    def test_the_decision_is_made_afresh_on_every_call(self, monkeypatch):
        mixin, _lines, _plan, _seen = _ml_budget(monkeypatch, seq=512, free=_HARNESS_CAP)
        assert mixin._stream_packed_backward is True
        _ml_budget(monkeypatch, seq=512, free=_CARD_8151_MIB, mixin=mixin)
        assert mixin._stream_packed_backward is False
        assert type(mixin)._stream_packed_backward is False  # the class default


# --------------------------------------------------------------------------
# the runtime: tiny CPU NF4 stream, four layers through two slots
# --------------------------------------------------------------------------
def _stream(tmp_path, *, n_layers=_N_LAYERS, dtype="float32"):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    from tests.test_v07202 import _nf4_stream, _randomise_lora_b

    model, runtime, _weights, _index, _shards = _nf4_stream(
        tmp_path, n_layers=n_layers, buffers=_BUFFERS, dtype=dtype
    )
    # PEFT starts lora_B at zero, which zeroes every lora_A gradient and makes an
    # equality check vacuous for half the adapter.
    _randomise_lora_b(model)
    return model, runtime


def _ids():
    import torch

    return torch.randint(0, 64, (1, _TOKENS), generator=torch.Generator().manual_seed(5))


def _lora_params(model):
    return [param for param in model.parameters() if param.requires_grad]


def _linear4bits(layer):
    import bitsandbytes as bnb

    return [child for child in layer.inner.modules() if isinstance(child, bnb.nn.Linear4bit)]


def _flags(model):
    from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

    out = {}
    for layer in streamed_decoder_layers(model):
        children = _linear4bits(layer)
        assert children, "vacuous: no Linear4bit under a streamed layer"
        out[layer.idx] = {
            bool(getattr(child, "_soup_packed_backward", False)) for child in children
        }
    return out


def _spy_packed_calls(monkeypatch):
    import soup_cli.utils.layer_stream_runtime as rt

    calls = []
    real = rt.packed_dequant_nf4_linear

    def counting(x, packed, quant_state, bias=None):
        calls.append(tuple(quant_state.shape))
        return real(x, packed, quant_state, bias)

    monkeypatch.setattr(rt, "packed_dequant_nf4_linear", counting)
    return calls


class TestSetStreamPackedBackward:
    def test_the_default_is_the_dense_arm_on_every_layer(self, tmp_path):
        model, _runtime = _stream(tmp_path)
        assert _flags(model) == {idx: {False} for idx in range(_N_LAYERS)}

    def test_switches_every_nf4_layer_and_back(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import set_stream_packed_backward

        model, _runtime = _stream(tmp_path)
        assert set_stream_packed_backward(model, True) == _N_LAYERS
        assert _flags(model) == {idx: {True} for idx in range(_N_LAYERS)}
        assert set_stream_packed_backward(model, False) == _N_LAYERS
        assert _flags(model) == {idx: {False} for idx in range(_N_LAYERS)}

    def test_it_survives_the_recompute_switches(self, tmp_path):
        """``set_stream_recompute`` and the 'auto' probe restore a layer through
        ``_set_layer_recompute``; that must not turn the packed arm back off."""
        from soup_cli.utils.layer_stream_runtime import (
            set_stream_packed_backward,
            set_stream_recompute,
        )

        model, _runtime = _stream(tmp_path)
        set_stream_packed_backward(model, True)
        set_stream_recompute(model, False, retain=2)
        assert _flags(model) == {idx: {True} for idx in range(_N_LAYERS)}
        set_stream_recompute(model, True)
        assert _flags(model) == {idx: {True} for idx in range(_N_LAYERS)}
        set_stream_packed_backward(model, False)
        assert _flags(model) == {idx: {False} for idx in range(_N_LAYERS)}

    def test_turning_it_off_leaves_a_retained_layer_packed(self, tmp_path):
        """A layer that skips the recompute is packed for its own reason."""
        from soup_cli.utils.layer_stream_runtime import (
            set_stream_packed_backward,
            set_stream_recompute,
        )

        model, _runtime = _stream(tmp_path)
        set_stream_recompute(model, False, retain=2)
        set_stream_packed_backward(model, True)
        set_stream_packed_backward(model, False)
        assert _flags(model) == {0: {False}, 1: {False}, 2: {True}, 3: {True}}

    @pytest.mark.parametrize("enabled", [1, 0, "yes", None])
    def test_a_non_bool_is_refused(self, tmp_path, enabled):
        from soup_cli.utils.layer_stream_runtime import set_stream_packed_backward

        model, _runtime = _stream(tmp_path, n_layers=2)
        with pytest.raises(TypeError, match="enabled must be a bool"):
            set_stream_packed_backward(model, enabled)

    def test_a_model_without_streamed_layers_switches_nothing(self):
        torch = pytest.importorskip("torch")
        from soup_cli.utils.layer_stream_runtime import set_stream_packed_backward

        assert set_stream_packed_backward(torch.nn.Linear(2, 2), True) == 0

    def test_a_bf16_stream_has_no_nf4_to_route_and_never_imports_bitsandbytes_here(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("torch")
        pytest.importorskip("peft")
        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.utils.layer_shard import shard_checkpoint
        from tests.test_v07202 import _tiny_llama_dir, _tiny_lora

        weights, _resident, _config = _tiny_llama_dir(tmp_path, n_layers=_N_LAYERS)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
        model, _runtime = rt.build_streamed_model(
            model_id=weights,
            shard_dir=shards,
            index=index,
            lora_config=_tiny_lora(),
            device="cpu",
            dtype="float32",
            buffers=_BUFFERS,
            pin=False,
            seed=3,
        )

        def refuse():
            raise AssertionError("bitsandbytes was reached for a layer with no NF4 linear")

        monkeypatch.setattr(rt, "_linear4bit_class", refuse)
        assert rt.set_stream_packed_backward(model, True) == 0
        assert rt.set_stream_packed_backward(model, False) == 0


# Layer 0's q/k/v projections read the frozen embedding output, which needs no
# gradient, so autograd saves nothing for them on either arm: 4 linears of layer 0
# plus all 7 of layer 1 keep something for the backward.
_SAVING_LINEARS_OF_TWO_LAYERS = 4 + 7


class _DenseTap:
    """Every dense weight ``bitsandbytes.functional.dequantize_4bit`` builds.

    Must be installed BEFORE the stream is built: ``install_dequant_forward``
    binds the function when it patches a layer. ``max_live`` is the most dense
    weights alive at once, sampled at each build, i.e. the dense memory a layer
    holds at its worst.
    """

    def __init__(self, monkeypatch):
        import bitsandbytes.functional as bnb_functional

        real = bnb_functional.dequantize_4bit
        self.refs = []
        self.max_live = 0
        self.calls = 0

        def tapped(*args, **kwargs):
            out = real(*args, **kwargs)
            self.calls += 1
            self.refs = [ref for ref in self.refs if ref() is not None]
            self.refs.append(weakref.ref(out))
            self.max_live = max(self.max_live, len(self.refs))
            return out

        monkeypatch.setattr(bnb_functional, "dequantize_4bit", tapped)

    def is_dense_weight(self, tensor):
        ptr = tensor.untyped_storage().data_ptr()
        return any(
            ref() is not None and ref().untyped_storage().data_ptr() == ptr for ref in self.refs
        )


def _observe_saved_tensors(model, tap):
    """Record what each layer's first forward saves for the backward.

    Installed through the layers' ``checkpoint_context_fn`` seam: that context is
    entered INSIDE the checkpoint's own hook, so it sees the tensors a decoder
    layer saves (a ``saved_tensors_hooks`` outside the checkpoint sees nothing).
    The pack hook returns the tensor unchanged, so use it for a forward only.
    """
    import torch

    from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

    record = SimpleNamespace(dense=0, packed_bytes=0, total=0)

    def pack(tensor):
        record.total += 1
        if tap.is_dense_weight(tensor):
            record.dense += 1
        if tensor.dtype == torch.uint8:
            record.packed_bytes += 1
        return tensor

    def context():
        return (
            torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
            contextlib.nullcontext(),
        )

    for layer in streamed_decoder_layers(model):
        layer.checkpoint_context_fn = context
    return record


class TestNoDenseWeightIsSaved:
    """Test 1: what the layers hold for the backward."""

    def test_the_dense_arm_saves_every_dense_weight(self, tmp_path, monkeypatch):
        """The positive control: without it 'saved no dense weight' could mean the
        detector never sees one."""
        tap = _DenseTap(monkeypatch)
        model, _runtime = _stream(tmp_path, n_layers=2)
        record = _observe_saved_tensors(model, tap)
        model(input_ids=_ids(), labels=_ids())
        assert len(_linear4bits(next(iter(_layers(model))))) == 7
        assert tap.calls == 14
        assert record.dense == _SAVING_LINEARS_OF_TWO_LAYERS
        assert record.packed_bytes == 0

    def test_the_packed_arm_saves_no_dense_weight_and_saves_the_packed_bytes(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.layer_stream_runtime import set_stream_packed_backward

        tap = _DenseTap(monkeypatch)
        model, _runtime = _stream(tmp_path, n_layers=2)
        set_stream_packed_backward(model, True)
        record = _observe_saved_tensors(model, tap)
        model(input_ids=_ids(), labels=_ids())
        assert tap.calls == 14, "vacuous: no dense weight was built at all"
        assert record.dense == 0
        # The packed weight and the uint8 absmax of every linear that saves anything
        # went through save_for_backward.
        assert record.packed_bytes == 2 * _SAVING_LINEARS_OF_TWO_LAYERS

    def test_a_layer_holds_at_most_one_dense_weight_through_the_whole_step(
        self, tmp_path, monkeypatch
    ):
        """The step the pre-flight prices: under the real checkpoint the recompute
        rebuilds a layer's dense weights together (7 live at once on the dense
        arm); on the packed arm each is built, used and freed (1 live)."""
        from soup_cli.utils.layer_stream_runtime import set_stream_packed_backward

        tap = _DenseTap(monkeypatch)
        model, _runtime = _stream(tmp_path)
        ids = _ids()
        model(input_ids=ids, labels=ids).loss.backward()
        assert tap.max_live == 7, "the dense arm no longer holds a whole layer"

        tap.max_live = 0
        tap.calls = 0
        set_stream_packed_backward(model, True)
        model.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids).loss.backward()
        assert tap.calls > 0
        assert tap.max_live == 1


def _layers(model):
    from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

    return streamed_decoder_layers(model)


def _step(model, ids):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    return out.loss.detach().clone(), grads


class TestPackedEqualsDense:
    """Test 2: same loss, same gradients, same AdamW update, to the bit."""

    @pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
    def test_loss_every_lora_grad_and_one_adamw_update_are_torch_equal(
        self, tmp_path, monkeypatch, dtype
    ):
        import torch

        from soup_cli.utils.layer_stream_runtime import set_stream_packed_backward

        model, runtime = _stream(tmp_path, dtype=dtype)
        calls = _spy_packed_calls(monkeypatch)
        params = _lora_params(model)
        initial = [param.detach().clone() for param in params]
        ids = _ids()

        def run(packed):
            with torch.no_grad():
                for param, start in zip(params, initial):
                    param.copy_(start)
            set_stream_packed_backward(model, packed)
            optimizer = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0, foreach=False)
            loads = runtime.pool.loads
            loss, grads = _step(model, ids)
            assert runtime.pool.loads - loads >= _N_LAYERS, "no slot was recycled"
            optimizer.step()
            return loss, grads, [param.detach().clone() for param in params]

        dense_loss, dense_grads, dense_params = run(False)
        assert calls == [], "the dense run took the packed Function"
        packed_loss, packed_grads, packed_params = run(True)
        assert len(calls) >= 2 * 7 * _N_LAYERS, "the packed run did not take the packed Function"

        assert torch.equal(dense_loss, packed_loss)
        assert dense_grads.keys() == packed_grads.keys()
        assert dense_grads, "vacuous: nothing is trainable"
        for name in dense_grads:
            assert torch.equal(dense_grads[name], packed_grads[name]), name
        assert all(bool(grad.abs().sum() > 0) for grad in dense_grads.values()), (
            "vacuous: a LoRA gradient is all zero"
        )
        assert len(dense_params) == len(packed_params) == len(params)
        for before, after_dense, after_packed in zip(initial, dense_params, packed_params):
            assert not torch.equal(before, after_dense), "vacuous: AdamW did not move a weight"
            assert torch.equal(after_dense, after_packed)

    def test_a_second_step_after_the_first_stays_equal(self, tmp_path):
        """The slot cache and the substituted views survive a step on either arm."""
        import torch

        from soup_cli.utils.layer_stream_runtime import set_stream_packed_backward

        model, _runtime = _stream(tmp_path)
        ids = _ids()
        reference = [_step(model, ids) for _ in range(2)]
        set_stream_packed_backward(model, True)
        packed = [_step(model, ids) for _ in range(2)]
        for (loss_a, grads_a), (loss_b, grads_b) in zip(reference, packed):
            assert torch.equal(loss_a, loss_b)
            for name in grads_a:
                assert torch.equal(grads_a[name], grads_b[name]), name


class TestTheSlotOwnerCheckStillFires:
    """Test 5: the packed arm keeps the guard against a recycled weight slot."""

    def _count_owner_checks(self, monkeypatch):
        from soup_cli.utils.layer_stream_runtime import _streamed_layer_class

        cls = _streamed_layer_class()
        real = cls._assert_nf4_slot_owner
        seen = []

        def counting(self):
            seen.append(self.idx)
            return real(self)

        monkeypatch.setattr(cls, "_assert_nf4_slot_owner", counting)
        return seen

    def _fresh(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import set_stream_packed_backward

        model, runtime = _stream(tmp_path)
        set_stream_packed_backward(model, True)
        for layer in _layers(model):
            # The weights are built (and the owner callback bound) on the next visit.
            layer._cached_substitution_buffers = None
            layer._cached_substitution_weights = None
        return model, runtime

    def test_every_packed_backward_is_checked(self, tmp_path, monkeypatch):
        seen = self._count_owner_checks(monkeypatch)
        model, _runtime = self._fresh(tmp_path)
        _step(model, _ids())
        # Layer 0's q/k/v have no input gradient to compute (frozen embeddings).
        assert [seen.count(idx) for idx in range(_N_LAYERS)] == [4, 7, 7, 7], seen

    def test_a_slot_recycled_before_the_backward_is_refused(self, tmp_path, monkeypatch):
        """Layer 2's first check finds another layer in its slot: the production
        guard raises, and not through the packed Function's own arithmetic."""
        from soup_cli.utils.layer_stream_runtime import _streamed_layer_class

        cls = _streamed_layer_class()
        real = cls._assert_nf4_slot_owner

        def tampering(self):
            if self.idx == 2:
                self.pool.owner[self.pool.slot_for(2)] = 3
            return real(self)

        monkeypatch.setattr(cls, "_assert_nf4_slot_owner", tampering)
        model, _runtime = self._fresh(tmp_path)
        with pytest.raises(
            RuntimeError,
            match=r"streamed NF4 backward found a recycled weight slot: slot \d holds layer 3, "
            r"expected layer 2",
        ):
            _step(model, _ids())

    def test_the_function_alone_refuses_a_recycled_slot(self):
        torch = pytest.importorskip("torch")
        bnb_functional = pytest.importorskip("bitsandbytes.functional")
        from soup_cli.utils.layer_stream_runtime import packed_dequant_nf4_linear

        torch.manual_seed(31)
        packed, state = bnb_functional.quantize_4bit(torch.randn(16, 16), quant_type="nf4")
        owner = {"current": 0}

        def assert_owner():
            if owner["current"] != 0:
                raise RuntimeError("recycled weight slot")

        packed._soup_stream_owner_check = assert_owner
        x = torch.randn(3, 16, requires_grad=True)
        output = packed_dequant_nf4_linear(x, packed, state)
        owner["current"] = 1
        with pytest.raises(RuntimeError, match="recycled weight slot"):
            output.square().sum().backward()


# --------------------------------------------------------------------------
# the setup applies the decision, from YAML to the runtime (spied), on CPU
# --------------------------------------------------------------------------
def _drive_with_decision(tmp_path, monkeypatch, *, packed, mode=None, extra_yaml=""):
    """The real ``_setup_streaming_transformers`` with the pre-flight's decision forced.

    ``order`` records, in call order, the packed switch and the VRAM probe.
    """
    pytest.importorskip("torch")
    import soup_cli.utils.layer_stream_runtime as rt
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests.test_stream_recompute import _MAX_LENGTH, _drive

    order = []

    def fake_budget(self, *args, **kwargs):
        self._stream_packed_backward = packed

        def probe(wrapper, model, plan):
            order.append("probe")

        # `_drive` installs its own probe spy before the setup runs; this replaces
        # it at the one moment that is still before the probe is called.
        monkeypatch.setattr(SFTTrainerWrapper, "_run_stream_vram_probe", probe)
        plan = (
            SimpleNamespace(rows=1, seq_len=_MAX_LENGTH)
            if "stream_vram_probe" in extra_yaml
            else None
        )
        return (), plan

    def spy(model, enabled):
        order.append(("packed", model, enabled))
        return 3

    monkeypatch.setattr(SFTTrainerWrapper, "_stream_budget_lines", fake_budget)
    monkeypatch.setattr(rt, "set_stream_packed_backward", spy)
    wrapper, model, events, said = _drive(
        tmp_path, monkeypatch, mode=mode, extra_yaml=extra_yaml
    )
    return order, model, events, said


class TestSetupWiring:
    def test_a_dense_decision_calls_nothing(self, tmp_path, monkeypatch):
        order, _model, events, said = _drive_with_decision(tmp_path, monkeypatch, packed=False)
        assert order == []
        assert events == []
        assert "Packed-state NF4 backward" not in said

    def test_a_packed_decision_routes_the_built_model_and_says_so(self, tmp_path, monkeypatch):
        order, model, events, said = _drive_with_decision(tmp_path, monkeypatch, packed=True)
        assert order == [("packed", model, True)]
        assert events == []
        assert "Packed-state NF4 backward: 3 streamed layers" in said
        assert "one matrix at a time" in said

    def test_the_routing_is_done_before_the_probe_runs(self, tmp_path, monkeypatch):
        """Defensive ordering: the real pre-flight never routes under the probe, but
        whatever routes the model must do it before anything is measured."""
        order, model, _events, _said = _drive_with_decision(
            tmp_path, monkeypatch, packed=True, extra_yaml="  stream_vram_probe: true\n"
        )
        assert order == [("packed", model, True), "probe"]

    @pytest.mark.parametrize("mode", ["off", "auto"])
    def test_it_composes_with_the_recompute_modes(self, tmp_path, monkeypatch, mode):
        order, model, events, _said = _drive_with_decision(
            tmp_path, monkeypatch, packed=True, mode=mode
        )
        assert order == [("packed", model, True)]
        assert [event[0] for event in events] == ["switch" if mode == "off" else "plan"]

    def test_a_failing_switch_closes_the_runtime_and_refuses(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from tests.test_stream_recompute import _drive

        def fake_budget(self, *args, **kwargs):
            self._stream_packed_backward = True
            return (), None

        def boom(model, enabled):
            raise RuntimeError("no layer to route")

        monkeypatch.setattr(SFTTrainerWrapper, "_stream_budget_lines", fake_budget)
        monkeypatch.setattr(rt, "set_stream_packed_backward", boom)
        # `plan_error` makes `_drive` expect the setup to raise that error.
        _wrapper, _model, events, _said = _drive(
            tmp_path, monkeypatch, mode=None, plan_error=RuntimeError("no layer to route")
        )
        assert events == [("close",)]


class TestDefaultsAreUntouched:
    """Test 3b: nothing changes for a config whose dense arm fits."""

    def test_a_default_step_never_takes_the_packed_function(self, tmp_path, monkeypatch):
        calls = _spy_packed_calls(monkeypatch)
        tap = _DenseTap(monkeypatch)
        model, _runtime = _stream(tmp_path)
        _step(model, _ids())
        assert calls == []
        assert tap.calls > 0
        assert _flags(model) == {idx: {False} for idx in range(_N_LAYERS)}

    def test_the_layer_attribute_defaults_to_off(self, tmp_path):
        model, _runtime = _stream(tmp_path, n_layers=2)
        assert all(layer.packed_checkpoint_backward is False for layer in _layers(model))

    def test_the_console_text_of_a_fitting_preflight_has_no_packed_line(
        self, monkeypatch
    ):
        _mixin, lines, _plan, _seen = _ml_budget(monkeypatch, seq=512, free=10**12)
        buffer = io.StringIO()
        from rich.console import Console

        Console(file=buffer, width=200).print("\n".join(lines))
        assert "packed" not in buffer.getvalue()
