"""Layer streaming without the checkpoint recompute (``training.stream_recompute``).

A streamed decoder layer normally runs its forward twice: once in the forward
and again as the checkpoint recompute in the backward, which is also where the
layer's weights are re-read into the buffer pool. With the recompute off the
layer keeps its activations instead. Pool-slot tensors are still never saved as
live views (#331): ``_body_retained`` saves them as tokens and, at the first
backward unpack of a layer, does the same ``pool.wait`` + ``prefetcher.advance``
the recompute did (``_backward_visit``). Later unpacks of that layer check the
slot still holds it.

Every model here streams 4 layers through 2 buffers, so slots really are
recycled between a layer's forward and its backward. The config field itself
is in ``tests/test_stream_recompute_config.py``.
"""

from __future__ import annotations

import io
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from soup_cli.utils.layer_stream import plan_retained_layers

_N_LAYERS = 4
_BUFFERS = 2
_HIDDEN = 64  # tests.test_v07202._tiny_llama_dir
_TOKENS = 8
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed console output, safe to match."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _console():
    from rich.console import Console

    buffer = io.StringIO()
    return Console(file=buffer, width=200), buffer


# --------------------------------------------------------------------------
# fixtures: the smallest CPU streamed models the existing suites build
# --------------------------------------------------------------------------
def _stream(tmp_path, quant):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    if quant == "nf4":
        pytest.importorskip("bitsandbytes")
        from tests.test_v07202 import _nf4_stream

        model, runtime, _weights, _index, _shards = _nf4_stream(
            tmp_path, n_layers=_N_LAYERS, buffers=_BUFFERS
        )
    else:
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import build_streamed_model
        from tests.test_v07202 import _tiny_llama_dir, _tiny_lora

        weights, _resident, _config = _tiny_llama_dir(tmp_path, n_layers=_N_LAYERS)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
        model, runtime = build_streamed_model(
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
    from tests.test_v07202 import _randomise_lora_b

    # PEFT starts lora_B at zero, which zeroes every lora_A gradient and makes
    # an equality check vacuous for half the adapter.
    _randomise_lora_b(model)
    return model, runtime


def _ids():
    import torch

    return torch.randint(0, 64, (1, _TOKENS), generator=torch.Generator().manual_seed(5))


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


def _layer_of(name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", name)
    assert match, name
    return int(match.group(1))


def _layer_class():
    from soup_cli.utils.layer_stream_runtime import _streamed_layer_class

    return _streamed_layer_class()


def _count_retained_forwards(monkeypatch):
    """Which layer indices ran ``_body_retained`` (the no-recompute forward)."""
    cls = _layer_class()
    real = cls._body_retained
    seen: list = []

    def counting(self, *args, **kwargs):
        seen.append(self.idx)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(cls, "_body_retained", counting)
    return seen


def _assert_same(run_a, run_b):
    import torch

    loss_a, grads_a = run_a
    loss_b, grads_b = run_b
    assert torch.equal(loss_a, loss_b)
    assert grads_a.keys() == grads_b.keys()
    assert grads_a, "vacuous: nothing is trainable"
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_b[name]), name
    assert all(bool(grad.abs().sum() > 0) for grad in grads_a.values()), (
        "vacuous: a LoRA gradient is all zero"
    )


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------
class TestRecomputeBudget:
    """min(85% of the card, free + reserved now, stream_vram_override) - reserved peak."""

    def _budget(self, **overrides):
        from soup_cli.utils.layer_stream import stream_recompute_budget

        args = dict(
            total_bytes=100_000,
            free_bytes=50_000,
            reserved_bytes=40_000,
            reserved_peak_bytes=30_000,
        )
        args.update(overrides)
        return stream_recompute_budget(**args)

    def test_the_card_cap_binds(self):
        # card int(0.85 * 100_000) = 85_000 < free + reserved 90_000
        budget = self._budget()
        assert (budget.bound, budget.cap_bytes, budget.budget_bytes) == ("card", 85_000, 55_000)

    def test_the_free_cap_binds_on_a_shared_card(self):
        # free + reserved = 20_000 + 40_000 = 60_000 < card 85_000
        budget = self._budget(free_bytes=20_000)
        assert (budget.bound, budget.cap_bytes, budget.budget_bytes) == ("free", 60_000, 30_000)

    def test_the_override_binds_when_it_is_the_smallest(self):
        budget = self._budget(override_bytes=50_000)
        assert (budget.bound, budget.cap_bytes, budget.budget_bytes) == (
            "override",
            50_000,
            20_000,
        )

    def test_an_override_above_the_others_does_not_raise_the_cap(self):
        budget = self._budget(override_bytes=10**9)
        assert (budget.bound, budget.cap_bytes) == ("card", 85_000)

    def test_a_peak_above_the_cap_is_a_negative_budget(self):
        assert self._budget(reserved_peak_bytes=90_000).budget_bytes == -5_000

    def test_the_headroom_scales_the_card_term(self):
        budget = self._budget(headroom=0.5)
        assert (budget.bound, budget.cap_bytes) == ("card", 50_000)

    @pytest.mark.parametrize("headroom", [0.0, -0.1, 1.01, True])
    def test_a_headroom_outside_0_1_is_refused(self, headroom):
        with pytest.raises(ValueError, match="headroom must be a float in"):
            self._budget(headroom=headroom)


class TestRetainedLayerCharge:
    def test_charged_at_the_reserved_to_allocated_ratio(self):
        from soup_cli.utils.layer_stream import retained_layer_charge

        assert retained_layer_charge(10_000, 60_000, 50_000) == 12_000

    def test_rounded_up(self):
        from soup_cli.utils.layer_stream import retained_layer_charge

        # 10_001 * 1.2 = 12_001.2
        assert retained_layer_charge(10_001, 60_000, 50_000) == 12_002

    def test_never_below_the_allocated_bytes(self):
        from soup_cli.utils.layer_stream import retained_layer_charge

        assert retained_layer_charge(10_000, 40_000, 50_000) == 10_000

    def test_no_allocated_peak_charges_the_bytes(self):
        from soup_cli.utils.layer_stream import retained_layer_charge

        assert retained_layer_charge(10_000, 60_000, 0) == 10_000


class TestPlanRetainedLayers:
    def test_exact_floor(self):
        assert plan_retained_layers(15_000, 4_000, 32) == 3

    def test_a_budget_one_byte_short_of_a_layer_floors_down(self):
        assert plan_retained_layers(11_999, 4_000, 32) == 2
        assert plan_retained_layers(12_000, 4_000, 32) == 3

    def test_clamped_at_n_layers(self):
        assert plan_retained_layers(10**12, 1, 32) == 32

    @pytest.mark.parametrize("budget", [0, -1, -(10**9)])
    def test_zero_or_negative_budget_is_zero(self, budget):
        assert plan_retained_layers(budget, 4_000, 32) == 0

    @pytest.mark.parametrize("per_layer", [0, -1])
    def test_a_layer_charge_at_or_below_zero_is_zero(self, per_layer):
        assert plan_retained_layers(100_000, per_layer, 32) == 0


# --------------------------------------------------------------------------
# the switch
# --------------------------------------------------------------------------
class TestSetStreamRecompute:
    def test_returns_the_layer_count_and_switches_every_layer(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import (
            set_stream_recompute,
            streamed_decoder_layers,
        )

        model, _runtime = _stream(tmp_path, "none")
        layers = streamed_decoder_layers(model)
        assert [layer.idx for layer in layers] == list(range(_N_LAYERS))
        assert all(layer.recompute for layer in layers)

        assert set_stream_recompute(model, False) == _N_LAYERS
        assert [layer.recompute for layer in layers] == [False] * _N_LAYERS

    @pytest.mark.parametrize("retain", [0, 1, 2, 3, 4, 9])
    def test_retain_switches_exactly_the_last_k(self, tmp_path, retain):
        from soup_cli.utils.layer_stream_runtime import (
            set_stream_recompute,
            streamed_decoder_layers,
        )

        model, _runtime = _stream(tmp_path, "none")
        assert set_stream_recompute(model, False, retain=retain) == _N_LAYERS
        kept = {layer.idx for layer in streamed_decoder_layers(model) if not layer.recompute}
        assert kept == set(range(max(0, _N_LAYERS - retain), _N_LAYERS))

    def test_enabled_true_restores_every_layer(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import (
            set_stream_recompute,
            streamed_decoder_layers,
        )

        model, _runtime = _stream(tmp_path, "none")
        set_stream_recompute(model, False, retain=3)
        assert set_stream_recompute(model, True) == _N_LAYERS
        assert all(layer.recompute for layer in streamed_decoder_layers(model))

    @pytest.mark.parametrize("retain", [-1, True, 1.5])
    def test_a_bad_retain_is_refused(self, tmp_path, retain):
        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        model, _runtime = _stream(tmp_path, "none")
        with pytest.raises(ValueError, match="retain must be None or an int >= 0"):
            set_stream_recompute(model, False, retain=retain)

    def test_a_model_without_streamed_layers_switches_nothing(self):
        torch = pytest.importorskip("torch")
        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        assert set_stream_recompute(torch.nn.Linear(2, 2), False) == 0

    def test_packed_backward_flags_only_on_switched_nf4_layers(self, tmp_path):
        import bitsandbytes as bnb

        from soup_cli.utils.layer_stream_runtime import (
            set_stream_recompute,
            streamed_decoder_layers,
        )

        model, _runtime = _stream(tmp_path, "nf4")

        def flags():
            out = {}
            for layer in streamed_decoder_layers(model):
                children = [
                    child for child in layer.inner.modules()
                    if isinstance(child, bnb.nn.Linear4bit)
                ]
                assert children, "vacuous: no Linear4bit under a streamed layer"
                out[layer.idx] = {
                    bool(getattr(child, "_soup_packed_backward", False)) for child in children
                }
            return out

        assert flags() == {idx: {False} for idx in range(_N_LAYERS)}
        set_stream_recompute(model, False, retain=2)
        assert flags() == {0: {False}, 1: {False}, 2: {True}, 3: {True}}
        set_stream_recompute(model, True)
        assert flags() == {idx: {False} for idx in range(_N_LAYERS)}

    def test_retain_is_keyword_only(self, tmp_path):
        """``retain`` is keyword-only: a positional count must fail loudly rather
        than be read as how many layers keep their activations."""
        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        model, _runtime = _stream(tmp_path, "none")
        with pytest.raises(TypeError):
            set_stream_recompute(model, False, 2)

    def test_a_stream_without_nf4_never_imports_bitsandbytes_here(self, tmp_path, monkeypatch):
        import soup_cli.utils.layer_stream_runtime as rt

        model, _runtime = _stream(tmp_path, "none")
        assert all(layer.n_dequant_forward == 0 for layer in rt.streamed_decoder_layers(model))

        def refuse():
            raise AssertionError("bitsandbytes was reached for a layer with no NF4 linear")

        monkeypatch.setattr(rt, "_linear4bit_class", refuse)
        assert rt.set_stream_recompute(model, False) == _N_LAYERS
        assert rt.set_stream_recompute(model, True) == _N_LAYERS


# --------------------------------------------------------------------------
# exactness, and the control that proves the harness would see #331
# --------------------------------------------------------------------------
@pytest.mark.parametrize("quant", ["none", "nf4"])
class TestExactness:
    def test_off_is_bit_identical_to_the_recompute(self, tmp_path, monkeypatch, quant):
        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        model, runtime = _stream(tmp_path, quant)
        ids = _ids()
        loads_before = runtime.pool.loads
        shipped = _step(model, ids)
        # 4 layers through 2 slots: layers 2 and 3 are read in the forward and
        # 1 and 0 again in the backward (layer 0 also in a first forward).
        assert runtime.pool.loads - loads_before >= _N_LAYERS

        retained = _count_retained_forwards(monkeypatch)
        assert set_stream_recompute(model, False) == _N_LAYERS
        loads_before = runtime.pool.loads
        no_recompute = _step(model, ids)
        assert sorted(retained) == list(range(_N_LAYERS)), "the no-recompute path did not run"
        # The backward visit re-reads layers 1 and 0 exactly as the recompute did.
        assert runtime.pool.loads - loads_before == _N_LAYERS
        _assert_same(shipped, no_recompute)

    def test_partial_retain_is_bit_identical(self, tmp_path, monkeypatch, quant):
        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        model, _runtime = _stream(tmp_path, quant)
        ids = _ids()
        shipped = _step(model, ids)
        retained = _count_retained_forwards(monkeypatch)
        set_stream_recompute(model, False, retain=2)
        mixed = _step(model, ids)
        assert sorted(retained) == [2, 3]
        _assert_same(shipped, mixed)

    def test_back_on_is_bit_identical(self, tmp_path, monkeypatch, quant):
        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        model, _runtime = _stream(tmp_path, quant)
        ids = _ids()
        shipped = _step(model, ids)
        set_stream_recompute(model, False)
        _step(model, ids)
        set_stream_recompute(model, True)
        retained = _count_retained_forwards(monkeypatch)
        again = _step(model, ids)
        assert retained == []
        _assert_same(shipped, again)


class TestTheHarnessSeesAliasing:
    """#331-class negative controls: with the backward visit gone, the pool slot
    of a lower layer still holds an upper layer when its backward reads it."""

    def test_plain_weights_give_wrong_grads_without_the_backward_visit(
        self, tmp_path, monkeypatch
    ):
        import torch

        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        model, _runtime = _stream(tmp_path, "none")
        ids = _ids()
        loss_on, grads_on = _step(model, ids)
        set_stream_recompute(model, False)
        monkeypatch.setattr(_layer_class(), "_backward_visit", lambda self: None)
        loss_off, grads_off = _step(model, ids)

        assert torch.equal(loss_on, loss_off)  # the forward is unaffected
        differing = {
            _layer_of(name) for name in grads_on if not torch.equal(grads_on[name], grads_off[name])
        }
        # Layers 3 and 2 still own their slots; 1 and 0 read layers 3 and 2.
        assert differing == {0, 1}

    def test_nf4_owner_check_fires_without_the_backward_visit(self, tmp_path, monkeypatch):
        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        model, _runtime = _stream(tmp_path, "nf4")
        set_stream_recompute(model, False)
        monkeypatch.setattr(_layer_class(), "_backward_visit", lambda self: None)
        with pytest.raises(
            RuntimeError, match="streamed NF4 backward found a recycled weight slot"
        ):
            _step(model, _ids())

    def test_a_slot_recycled_mid_backward_is_refused(self, tmp_path, monkeypatch):
        """Layer 3's first backward unpack visits it; a load into its slot right
        after must stop the backward at layer 3's next unpack."""
        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        model, runtime = _stream(tmp_path, "none")
        set_stream_recompute(model, False)
        real_advance = runtime.prefetcher.advance
        state = {"backward": False}

        def advance(idx):
            real_advance(idx)
            if state["backward"] and idx == _N_LAYERS - 1:
                runtime.pool.load_async(1, runtime.source)  # slot 1 = layer 3's slot

        monkeypatch.setattr(runtime.prefetcher, "advance", advance)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=_ids(), labels=_ids())
        state["backward"] = True
        with pytest.raises(
            RuntimeError,
            match=(
                r"streamed backward \(no recompute\) found slot 1 holding layer 1 "
                r"while layer 3 still unpacks from it"
            ),
        ):
            out.loss.backward()


class TestBackwardVisit:
    def _layer(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

        model, runtime = _stream(tmp_path, "none")
        return streamed_decoder_layers(model)[0], runtime

    def test_the_first_visit_waits_and_prefetches_once(self, tmp_path, monkeypatch):
        layer, runtime = self._layer(tmp_path)
        calls = []
        monkeypatch.setattr(runtime.pool, "wait", lambda idx: calls.append(("wait", idx)))
        monkeypatch.setattr(runtime.prefetcher, "advance", lambda idx: calls.append(("adv", idx)))
        runtime.pool.owner[runtime.pool.slot_for(0)] = 0
        layer._bwd_visited = False
        layer._backward_visit()
        layer._backward_visit()
        assert calls == [("wait", 0), ("adv", 0)]
        assert layer._bwd_visited is True

    def test_a_later_unpack_from_a_recycled_slot_raises(self, tmp_path):
        layer, runtime = self._layer(tmp_path)
        layer._bwd_visited = True
        runtime.pool.owner[runtime.pool.slot_for(layer.idx)] = 2
        with pytest.raises(
            RuntimeError,
            match=(
                r"streamed backward \(no recompute\) found slot 0 holding layer 2 "
                r"while layer 0 still unpacks from it"
            ),
        ):
            layer._backward_visit()

    def _visited(self, tmp_path):
        """Layer 0 after its first backward visit, its slot holding it."""
        layer, runtime = self._layer(tmp_path)
        runtime.pool.load_async(0, runtime.source)
        layer._bwd_visited = False
        layer._backward_visit()
        return layer, runtime

    def test_a_later_unpack_from_its_own_slot_passes(self, tmp_path):
        layer, _runtime = self._visited(tmp_path)
        layer._backward_visit()
        layer._backward_visit()

    def test_a_slot_reloaded_with_the_same_layer_is_refused(self, tmp_path):
        """The owner alone cannot see slot 0 refilled with layer 2 and then layer 0
        again between two unpacks; the load generation can."""
        layer, runtime = self._visited(tmp_path)
        before = runtime.pool.generation[0]
        runtime.pool.load_async(2, runtime.source)
        runtime.pool.load_async(0, runtime.source)
        assert runtime.pool.owner[0] == 0
        with pytest.raises(
            RuntimeError,
            match=(
                rf"streamed backward \(no recompute\) found slot 0 reloaded \(load "
                rf"generation {before} -> {before + 2}\) while layer 0 still unpacks from it"
            ),
        ):
            layer._backward_visit()

    def test_every_load_bumps_the_slot_generation(self, tmp_path):
        _layer, runtime = self._layer(tmp_path)
        pool = runtime.pool
        start = list(pool.generation)
        pool.load_async(0, runtime.source)
        pool.load_async(1, runtime.source)
        pool.load_async(2, runtime.source)
        assert pool.generation == [start[0] + 2, start[1] + 1]


class TestExactnessWithOtherFeatures:
    def test_off_with_the_nf4_scale_cache_is_bit_identical(self, tmp_path):
        """The off step runs FIRST, so the scale cache is filled under the
        no-recompute path; the recompute step then reuses it."""
        pytest.importorskip("bitsandbytes")
        from soup_cli.utils.layer_stream_runtime import set_stream_recompute
        from tests.test_stream_nf4_scale_cache import _nf4_stream
        from tests.test_v07202 import _randomise_lora_b

        model, runtime, _index = _nf4_stream(tmp_path, cache=True, n_layers=_N_LAYERS)
        _randomise_lora_b(model)
        ids = _ids()
        set_stream_recompute(model, False)
        no_recompute = _step(model, ids)
        assert len(runtime.scale_cache) > 0, "vacuous: the scale cache was never filled"
        set_stream_recompute(model, True)
        _assert_same(_step(model, ids), no_recompute)

    def test_two_accumulated_micro_steps_are_bit_identical(self, tmp_path):
        import torch

        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        model, _runtime = _stream(tmp_path, "nf4")
        generator = torch.Generator().manual_seed(11)
        batches = [torch.randint(0, 64, (1, _TOKENS), generator=generator) for _ in range(2)]

        def accumulate():
            model.zero_grad(set_to_none=True)
            losses = []
            for ids in batches:
                out = model(input_ids=ids, labels=ids)
                (out.loss / len(batches)).backward()
                losses.append(out.loss.detach().clone())
            grads = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            return torch.stack(losses), grads

        shipped = accumulate()
        set_stream_recompute(model, False)
        _assert_same(shipped, accumulate())

    def test_off_through_the_real_setup_with_fast_lora(self, tmp_path, monkeypatch):
        """``stream_recompute: off`` from the config through the real setup(),
        with the fast LoRA kernels on: every layer is switched, and the LoRA
        gradients equal the recompute's on the same model."""
        import torch

        from soup_cli.utils.layer_stream_runtime import (
            set_stream_recompute,
            streamed_decoder_layers,
        )
        from tests.test_fast_lora_flag import TARGETS
        from tests.test_v07204 import _build_streamed_wrapper

        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path,
            monkeypatch,
            task="sft",
            n_layers=_N_LAYERS,
            device="cpu",
            tie=False,
            stream_buffers=_BUFFERS,
            fast_lora=True,
            stream_recompute="off",
            lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": TARGETS},
        )
        try:
            model = wrapper.model
            assert [layer.recompute for layer in streamed_decoder_layers(model)] == [
                False
            ] * _N_LAYERS
            assert sum("forward" in vars(module) for module in model.modules()) > 0, (
                "vacuous: the fast LoRA kernels are not on"
            )
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "lora_B" in name:
                        torch.manual_seed(abs(hash(name)) % 2**31)
                        param.normal_(std=0.1)
            ids = torch.tensor([[1, 5, 9, 13, 2, 7, 11]])
            no_recompute = _step(model, ids)
            set_stream_recompute(model, True)
            _assert_same(_step(model, ids), no_recompute)
        finally:
            wrapper._close_stream_runtime()


# --------------------------------------------------------------------------
# 'auto': one synthetic step at the configured shape, at setup
# --------------------------------------------------------------------------
_OUTPUT_BYTES = 1 * _TOKENS * _HIDDEN * 4  # the measured layer's fp32 output


def _readings(*values):
    """A fake ``memory_allocated`` returning ``values`` in order, then failing."""
    it = iter(values)
    calls = []

    def allocated():
        calls.append(1)
        return next(it)

    return allocated, calls


class TestRetainedLayerProbe:
    def test_measures_the_last_layer_once(self, tmp_path, monkeypatch):
        from soup_cli.utils.layer_stream_runtime import RetainedLayerProbe, streamed_decoder_layers

        model, _runtime = _stream(tmp_path, "none")
        ids = _ids()
        shipped = _step(model, ids)

        allocated, calls = _readings(1_000, 1_000 + 7_000)
        probe = RetainedLayerProbe(allocated=allocated)
        retained = _count_retained_forwards(monkeypatch)
        assert probe.arm(model) == _N_LAYERS - 1
        measured_step = _step(model, ids)
        assert retained == [_N_LAYERS - 1]
        assert probe.per_layer_bytes == 7_000 - _OUTPUT_BYTES
        assert probe.layer_idx == _N_LAYERS - 1
        assert probe.error is None
        # The measured step trains exactly like the shipped one.
        _assert_same(shipped, measured_step)
        # The next forward is the shipped path again, and nothing is re-read.
        assert all(layer.recompute for layer in streamed_decoder_layers(model))
        _step(model, ids)
        assert retained == [_N_LAYERS - 1]
        assert len(calls) == 2

    def test_bytes_the_scale_cache_fills_are_not_counted_as_retained(self, tmp_path):
        """The first visit of a layer fills its share of stream_nf4_scale_cache
        inside the measured forward; that is persistent VRAM, charged up front by
        the pre-flight, not activations the layer keeps for its backward."""
        pytest.importorskip("bitsandbytes")
        from soup_cli.utils.layer_stream import nf4_scale_cache_bytes
        from soup_cli.utils.layer_stream_runtime import RetainedLayerProbe
        from tests.test_stream_nf4_scale_cache import _nf4_stream

        model, runtime, index = _nf4_stream(tmp_path, cache=True, n_layers=_N_LAYERS)
        filled_by_one_layer = nf4_scale_cache_bytes(index.quant_specs, n_layers=1)
        assert filled_by_one_layer > 0
        allocated, _calls = _readings(1_000, 1_000 + 900_000)
        probe = RetainedLayerProbe(allocated=allocated)
        probe.arm(model)
        _step(model, _ids())  # the first visit of every layer
        assert runtime.scale_cache.nbytes == _N_LAYERS * filled_by_one_layer
        assert probe.per_layer_bytes == 900_000 - filled_by_one_layer - _OUTPUT_BYTES

    def test_a_cpu_pool_without_a_reader_cannot_measure(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import RetainedLayerProbe, streamed_decoder_layers

        model, _runtime = _stream(tmp_path, "none")
        probe = RetainedLayerProbe()
        assert probe.arm(model) is None
        assert probe.error == "no CUDA device to measure the retained bytes on"
        assert all(layer.recompute for layer in streamed_decoder_layers(model))

    def test_a_reader_that_raises_is_recorded_not_raised(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import RetainedLayerProbe

        model, _runtime = _stream(tmp_path, "none")

        def broken():
            raise RuntimeError("boom")

        probe = RetainedLayerProbe(allocated=broken)
        probe.arm(model)
        _step(model, _ids())  # training goes on
        assert probe.per_layer_bytes is None
        assert probe.error == "the measurement raised RuntimeError: boom"

    def test_no_streamed_layers(self):
        torch = pytest.importorskip("torch")
        from soup_cli.utils.layer_stream_runtime import RetainedLayerProbe

        probe = RetainedLayerProbe(allocated=lambda: 0)
        assert probe.arm(torch.nn.Linear(2, 2)) is None
        assert probe.error == "the model has no streamed decoder layers"


def _kept(model):
    from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

    return {layer.idx for layer in streamed_decoder_layers(model) if not layer.recompute}


class _FakeMemory:
    """The allocator figures the planner reads, CPU-side. ``readings`` feed the
    probe's ``allocated()`` in order; None makes it raise."""

    def __init__(
        self,
        *,
        readings=(1_000, 21_000),
        peak_allocated=50_000,
        peak_reserved=60_000,
        reserved=60_000,
        free=30_000,
        total=200_000,
    ):
        self._readings = None if readings is None else iter(readings)
        self._figures = dict(
            peak_allocated=peak_allocated,
            peak_reserved=peak_reserved,
            reserved=reserved,
            free=free,
            total=total,
        )
        self.events = []

    def allocated(self):
        self.events.append("allocated")
        if self._readings is None:
            raise RuntimeError("boom")
        return next(self._readings)

    def reset_peak(self):
        self.events.append("reset_peak")

    def synchronize(self):
        self.events.append("synchronize")

    def release(self):
        self.events.append("release")

    def peak_allocated(self):
        return self._figures["peak_allocated"]

    def peak_reserved(self):
        return self._figures["peak_reserved"]

    def reserved(self):
        return self._figures["reserved"]

    def free(self):
        return self._figures["free"]

    def total(self):
        return self._figures["total"]


#: The configured shape the planner is handed, and its last layer's fp32 output.
_ROWS, _SEQ = 2, 16
_SHAPE_OUTPUT_BYTES = _ROWS * _SEQ * _HIDDEN * 4  # 8_192


def _plan(model, console, memory, **kwargs):
    from soup_cli.utils.layer_stream_runtime import plan_stream_recompute_auto

    kwargs.setdefault("rows", _ROWS)
    kwargs.setdefault("seq_len", _SEQ)
    kwargs.setdefault("vocab_size", 64)
    return plan_stream_recompute_auto(model, notify=console.print, memory=memory, **kwargs)


class TestPlanStreamRecomputeAuto:
    """'auto' decides before training, from one synthetic step at the configured
    shape (not from whatever the first real batch pads to)."""

    def test_plans_k_from_the_synthetic_step(self, tmp_path):
        model, _runtime = _stream(tmp_path, "none")
        console, buffer = _console()
        memory = _FakeMemory()
        # per layer = 20_000 - 8_192 = 11_808; charge = ceil(11_808 x 60_000 / 50_000)
        # = 14_170; cap = min(170_000 card, 30_000 + 60_000 free) = 90_000;
        # budget = 90_000 - 60_000 = 30_000 -> K = 2
        assert _plan(model, console, memory) == 2
        assert _kept(model) == {2, 3}
        assert memory.events[0] == "reset_peak"
        assert memory.events[-1] == "release"
        said = _plain(buffer.getvalue())
        assert (
            "training.stream_recompute='auto': the last 2 of 4 streamed layers keep their "
            "activations, the rest recompute" in said
        )
        assert (
            "measured at batch 2 x seq 16: one layer 11808 bytes retained, charged 14170 "
            "bytes at reserved/allocated 1.20" in said
        )
        assert (
            "budget 30000 bytes = free + reserved VRAM 90000 bytes minus the reserved "
            "peak 60000 bytes" in said
        )

    def test_the_step_runs_at_the_configured_shape(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import _synthetic_step

        model, _runtime = _stream(tmp_path, "none")
        console, buffer = _console()
        shapes = []

        def step(model, *, rows, seq_len, vocab_size, device):
            shapes.append((rows, seq_len, vocab_size, str(device)))
            _synthetic_step(model, rows=rows, seq_len=seq_len, vocab_size=vocab_size, device=device)

        _plan(model, console, _FakeMemory(), rows=3, seq_len=24, step=step)
        assert shapes == [(3, 24, 64, "cpu")]
        # The measured layer's output was that shape's: 20_000 - 3 x 24 x 64 x 4.
        assert "one layer 1568 bytes retained" in _plain(buffer.getvalue())

    def test_the_synthetic_step_leaves_no_gradient(self, tmp_path):
        model, _runtime = _stream(tmp_path, "none")
        console, _buffer = _console()
        _plan(model, console, _FakeMemory())
        assert all(param.grad is None for param in model.parameters())

    def test_the_planned_mode_trains_bit_identically(self, tmp_path):
        model, _runtime = _stream(tmp_path, "none")
        console, _buffer = _console()
        ids = _ids()
        shipped = _step(model, ids)
        assert _plan(model, console, _FakeMemory()) == 2
        _assert_same(shipped, _step(model, ids))

    def test_the_override_caps_the_budget(self, tmp_path):
        model, _runtime = _stream(tmp_path, "none")
        console, buffer = _console()
        # free cap 10**6 + 60_000, card 170_000; override 74_170 -> budget 14_170 -> K = 1
        memory = _FakeMemory(free=10**6)
        assert _plan(model, console, memory, override_bytes=74_170) == 1
        assert _kept(model) == {3}
        assert "training.stream_vram_override 74170 bytes minus the reserved peak" in _plain(
            buffer.getvalue()
        )

    def test_the_card_cap_binds(self, tmp_path):
        model, _runtime = _stream(tmp_path, "none")
        console, buffer = _console()
        # card int(0.85 x 100_000) = 85_000 -> budget 25_000 -> K = 1
        memory = _FakeMemory(free=10**6, total=100_000)
        assert _plan(model, console, memory) == 1
        assert "85% of the card's 100000 bytes minus the reserved peak" in _plain(
            buffer.getvalue()
        )


class TestPlanStreamRecomputeAutoFallbacks:
    """Every path that cannot plan keeps the checkpoint recompute and says why."""

    def _never(self, *_a, **_k):
        raise AssertionError("the synthetic step ran although nothing could be measured")

    def _run(self, tmp_path, memory, **kwargs):
        model, _runtime = _stream(tmp_path, "none")
        console, buffer = _console()
        k = _plan(model, console, memory, **kwargs)
        return k, model, _plain(buffer.getvalue())

    def _assert_kept(self, k, model, said, why):
        from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

        assert k == 0
        assert _kept(model) == set()
        assert all(layer._retained_probe is None for layer in streamed_decoder_layers(model))
        assert (
            "training.stream_recompute='auto' keeps the recompute on all 4 streamed layers:"
            in said
        )
        assert why in said

    def test_no_cuda_device(self, tmp_path):
        k, model, said = self._run(tmp_path, None, step=self._never)
        self._assert_kept(k, model, said, "no CUDA device to measure VRAM on")

    @staticmethod
    def _out_of_memory_errors():
        import torch

        return [
            # the allocator's own
            torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
            # an asynchronous OOM surfacing at a synchronise (#649 / #901)
            RuntimeError("CUDA error: out of memory"),
        ]

    @pytest.mark.parametrize("which", [0, 1])
    def test_an_out_of_memory_step_keeps_the_recompute(self, tmp_path, which):
        error = self._out_of_memory_errors()[which]

        def oom(*_a, **_k):
            raise error

        k, model, said = self._run(tmp_path, _FakeMemory(), step=oom)
        self._assert_kept(
            k,
            model,
            said,
            "the synthetic step at batch 2 x seq 16 ran out of memory, so the configured "
            "shape does not fit with one layer retained; set training.stream_vram_probe: "
            "true to measure whether it fits at all",
        )

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("CUDA error: an illegal memory access was encountered"),
            ValueError("a bug in the step"),
        ],
    )
    def test_any_other_step_error_propagates(self, tmp_path, error):
        """Not a fit answer: the context may be unusable, so the setup must get
        the exception (and close the runtime) rather than train on."""
        from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

        model, _runtime = _stream(tmp_path, "none")
        console, buffer = _console()
        memory = _FakeMemory()

        def broken(*_a, **_k):
            raise error

        with pytest.raises(type(error), match=re.escape(str(error))):
            _plan(model, console, memory, step=broken)
        assert _kept(model) == set()
        assert all(layer._retained_probe is None for layer in streamed_decoder_layers(model))
        assert memory.events[-1] == "release"
        assert "keeps the recompute" not in _plain(buffer.getvalue())

    def test_the_measurement_raised(self, tmp_path):
        k, model, said = self._run(tmp_path, _FakeMemory(readings=None))
        self._assert_kept(k, model, said, "the measurement raised RuntimeError: boom")

    def test_a_layer_measured_at_or_below_zero(self, tmp_path):
        k, model, said = self._run(tmp_path, _FakeMemory(readings=(1_000, 1_000)))
        self._assert_kept(
            k, model, said, f"one layer measured at {-_SHAPE_OUTPUT_BYTES} bytes retained (<= 0)"
        )

    def test_no_vram_headroom(self, tmp_path):
        k, model, said = self._run(tmp_path, _FakeMemory(peak_reserved=95_000))
        self._assert_kept(
            k,
            model,
            said,
            "no VRAM headroom: free + reserved VRAM 90000 bytes minus the reserved peak of "
            "the batch 2 x seq 16 step, 95000 bytes, is -5000 bytes",
        )

    def test_a_budget_smaller_than_one_layer(self, tmp_path):
        # cap 12_000 + 60_000 = 72_000 -> budget 12_000 < one layer's charge 14_170
        k, model, said = self._run(tmp_path, _FakeMemory(free=12_000))
        self._assert_kept(
            k, model, said, "one layer (14170 bytes charged) does not fit the budget of 12000 bytes"
        )

    def test_no_step_shape(self, tmp_path):
        k, model, said = self._run(tmp_path, _FakeMemory(), vocab_size=0, step=self._never)
        self._assert_kept(
            k, model, said, "no step shape to measure (batch 2 x seq 16, vocab_size 0)"
        )

    def test_no_streamed_layers(self):
        torch = pytest.importorskip("torch")
        console, buffer = _console()
        assert _plan(torch.nn.Linear(2, 2), console, _FakeMemory(), step=self._never) == 0
        assert "the model has no streamed decoder layers" in _plain(buffer.getvalue())


# --------------------------------------------------------------------------
# the setup path, from YAML to the runtime (spied), on CPU
# --------------------------------------------------------------------------
_MAX_LENGTH = 96


def _drive(
    tmp_path,
    monkeypatch,
    *,
    mode,
    stream_source="ram",
    device="cuda",
    extra_yaml="",
    plan_error=None,
):
    """Real config and pre-flight, spied ``build_streamed_model``, switch, planner
    and VRAM probe.

    Returns ``(wrapper, model, events, console_text)``; ``events`` records each
    spied call in order, and ``("close",)`` when the runtime is closed. With
    ``plan_error`` the planner raises it, and the setup must raise it too.
    """
    pytest.importorskip("torch")
    from rich.console import Console

    import soup_cli.trainer.stream_setup as ss
    import soup_cli.utils.layer_stream as ls
    import soup_cli.utils.layer_stream_runtime as rt
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests.test_issue623_stream_pin_allocation import (
        _FREE_BYTES,
        _cfg_yaml,
        _tiny_checkpoint,
    )

    weights = _tiny_checkpoint(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        "soup_cli.utils.spectrum_scan.resolve_model_weights", lambda *_a, **_k: weights
    )
    monkeypatch.setattr(ls, "free_ram_bytes", lambda: _FREE_BYTES)
    monkeypatch.setattr(
        ls, "classify_disk_kind", lambda *_a, **_k: ls.DiskClassification("nvme")
    )
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a, **_k: (_FREE_BYTES, _FREE_BYTES))
    monkeypatch.setattr(rt, "expandable_segments_status", lambda *_a, **_k: (True, ""))

    built = MagicMock(name="streamed-model")
    events = []

    def spy_build(**kwargs):
        runtime = MagicMock()
        runtime.close.side_effect = lambda: events.append(("close",))
        runtime.tier = kwargs.get("tier", "ram")
        runtime.stats.return_value = {
            "tier": runtime.tier,
            "store_bytes": 1_000_000,
            "disk_bytes": 1_000_000,
            "pinned": False,
            "read_ahead": kwargs.get("read_ahead"),
            "buffers": 2,
            "buffer_bytes": 4_000,
            "n_layers": 3,
        }
        return built, runtime

    monkeypatch.setattr(rt, "build_streamed_model", spy_build)

    def spy_switch(model, enabled, **kwargs):
        events.append(("switch", model, enabled, kwargs))
        return 3

    def spy_plan(model, **kwargs):
        events.append(("plan", model, kwargs))
        if plan_error is not None:
            raise plan_error
        return 2

    def spy_probe(self, model, plan):
        events.append(("probe", model, plan.rows, plan.seq_len))

    monkeypatch.setattr(rt, "set_stream_recompute", spy_switch)
    monkeypatch.setattr(rt, "plan_stream_recompute_auto", spy_plan)
    monkeypatch.setattr(SFTTrainerWrapper, "_run_stream_vram_probe", spy_probe)
    buffer = io.StringIO()
    monkeypatch.setattr(ss, "console", Console(file=buffer, width=200))

    extra = "" if mode is None else f"  stream_recompute: '{mode}'\n"
    text = _cfg_yaml(
        weights, "", stream_source=stream_source, extra_training_yaml=extra + extra_yaml
    ).replace("  format: alpaca\n", f"  format: alpaca\n  max_length: {_MAX_LENGTH}\n")
    cfg = load_config_from_string(text)
    assert cfg.data.max_length == _MAX_LENGTH
    wrapper = SFTTrainerWrapper(cfg)
    wrapper.device = device
    if plan_error is None:
        wrapper._setup_streaming_transformers(cfg, cfg.training)
    else:
        with pytest.raises(type(plan_error), match=re.escape(str(plan_error))):
            wrapper._setup_streaming_transformers(cfg, cfg.training)
    return wrapper, built, events, _plain(buffer.getvalue())


class _FakeTrainer:
    def __init__(self):
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)


def _callbacks(wrapper):
    """What the trainer gets right before training: the save guard only."""
    wrapper.trainer = _FakeTrainer()
    wrapper._attach_streamed_save_guard()
    return [type(callback).__name__ for callback in wrapper.trainer.callbacks]


class TestSetupWiring:
    @pytest.mark.parametrize("mode", [None, "on"])
    def test_on_changes_nothing(self, tmp_path, monkeypatch, mode):
        wrapper, _model, events, said = _drive(tmp_path, monkeypatch, mode=mode)
        assert events == []
        assert _callbacks(wrapper) == ["StreamedSaveGuardCallback"]
        assert "stream_recompute" not in said

    def test_off_switches_every_layer_and_says_how_many(self, tmp_path, monkeypatch):
        _wrapper, model, events, said = _drive(tmp_path, monkeypatch, mode="off")
        assert events == [("switch", model, False, {})]
        assert (
            "training.stream_recompute='off': 3 streamed layers keep their activations "
            "instead of recomputing them in the backward" in said
        )
        assert "read-bound" not in said

    def test_off_on_the_disk_tier_warns_and_still_switches(self, tmp_path, monkeypatch):
        _wrapper, model, events, said = _drive(
            tmp_path, monkeypatch, mode="off", stream_source="disk"
        )
        assert events == [("switch", model, False, {})]
        assert "training.stream_recompute='off' on the disk tier:" in said
        assert "a disk-tier step is usually read-bound" in said
        assert "the recompute ('on') may be cheaper here" in said

    def test_auto_plans_at_setup_from_the_configured_shape(self, tmp_path, monkeypatch):
        wrapper, model, events, said = _drive(tmp_path, monkeypatch, mode="auto")
        assert events == [
            (
                "plan",
                model,
                {
                    "rows": 1,  # training.batch_size
                    "seq_len": _MAX_LENGTH,  # data.max_length
                    "vocab_size": 64,
                    "notify": events[0][2]["notify"],
                    "override_bytes": None,
                },
            )
        ]
        # Decided before training: nothing is left for a training callback.
        assert _callbacks(wrapper) == ["StreamedSaveGuardCallback"]
        assert (
            f"training.stream_recompute='auto': measuring one step at batch 1 x seq "
            f"{_MAX_LENGTH} to choose how many of the last layers keep their activations "
            "within 85% of the card." in said
        )

    def test_auto_passes_the_vram_override(self, tmp_path, monkeypatch):
        _wrapper, _model, events, _said = _drive(
            tmp_path, monkeypatch, mode="auto", extra_yaml="  stream_vram_override: 5000000000\n"
        )
        assert [event[0] for event in events] == ["plan"]
        assert events[0][2]["override_bytes"] == 5_000_000_000

    def test_a_planner_error_closes_the_runtime_and_refuses(self, tmp_path, monkeypatch):
        """A non-OOM error of the synthetic step propagates out of the planner;
        the setup releases the streaming runtime and re-raises it."""
        error = RuntimeError("CUDA error: an illegal memory access was encountered")
        _wrapper, _model, events, _said = _drive(
            tmp_path, monkeypatch, mode="auto", plan_error=error
        )
        assert [event[0] for event in events] == ["plan", "close"]

    @pytest.mark.parametrize("mode", ["off", "auto"])
    def test_the_vram_probe_measures_the_mode_that_will_train(self, tmp_path, monkeypatch, mode):
        """stream_vram_probe runs after the switch (or the plan), so its peak is
        the retained mode's, not the recompute's."""
        _wrapper, model, events, _said = _drive(
            tmp_path, monkeypatch, mode=mode, extra_yaml="  stream_vram_probe: true\n"
        )
        assert [event[0] for event in events] == [
            "switch" if mode == "off" else "plan",
            "probe",
        ]
        assert events[-1] == ("probe", model, 1, _MAX_LENGTH)

    def test_auto_on_the_disk_tier_keeps_the_recompute(self, tmp_path, monkeypatch):
        wrapper, _model, events, said = _drive(
            tmp_path, monkeypatch, mode="auto", stream_source="disk"
        )
        assert events == []
        assert _callbacks(wrapper) == ["StreamedSaveGuardCallback"]
        assert (
            "training.stream_recompute='auto' keeps the recompute on the disk tier" in said
        )

    def test_auto_without_cuda_keeps_the_recompute(self, tmp_path, monkeypatch):
        _wrapper, _model, events, said = _drive(
            tmp_path, monkeypatch, mode="auto", device="cpu"
        )
        assert events == []
        assert (
            "training.stream_recompute='auto' keeps the recompute: no CUDA device to "
            "measure VRAM on" in said
        )

    @pytest.mark.parametrize("mode", ["off", "auto"])
    def test_the_setup_refuses_a_task_other_than_sft_too(self, mode, monkeypatch):
        """Defence in depth behind the config gate: a caller that reaches the
        setup with another task (a stand-in config) is refused before anything
        is switched."""
        from types import SimpleNamespace

        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.trainer.stream_setup import StreamingSetupMixin

        def never(*_a, **_k):
            raise AssertionError("switched before the task was checked")

        monkeypatch.setattr(rt, "set_stream_recompute", never)
        monkeypatch.setattr(rt, "plan_stream_recompute_auto", never)
        with pytest.raises(
            ValueError,
            match=rf"training.stream_recompute='{mode}' supports task='sft' only; got task='dpo'",
        ):
            StreamingSetupMixin()._configure_stream_recompute(
                MagicMock(),
                SimpleNamespace(stream_recompute=mode),
                tier="ram",
                on_cuda=True,
                task="dpo",
                step_shape=(1, 8, 64),
            )


class TestTheProbeHarness:
    """benchmarks/harness/stream_probe.py replaces the pool's ``load_async`` with
    a timed replica; the replica must bump the slot's load generation too, or
    the no-recompute backward's reload check goes blind under the probe."""

    def _harness(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "stream_probe.py"
        spec = importlib.util.spec_from_file_location("stream_probe_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_replica_load_bumps_the_slot_generation(self, tmp_path):
        harness = self._harness()
        model, runtime = _stream(tmp_path, "none")
        harness.Instruments(runtime, model, "none")
        pool = runtime.pool
        assert "Instruments" in pool.load_async.__qualname__, "the replica is not installed"
        before = list(pool.generation)
        pool.load_async(2, runtime.source)
        assert pool.owner[0] == 2
        assert pool.generation == [before[0] + 1, before[1]]

    def test_off_under_the_replica_is_bit_identical(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import set_stream_recompute

        harness = self._harness()
        model, runtime = _stream(tmp_path, "none")
        harness.Instruments(runtime, model, "none")
        ids = _ids()
        shipped = _step(model, ids)
        set_stream_recompute(model, False)
        _assert_same(shipped, _step(model, ids))


class TestTheDefaultPathAddsNothing:
    """With ``stream_recompute`` 'on' (or a stand-in config that never heard of
    it) the setup must not read anything the feature added, call anything, or
    reach for a runtime method. Regression: the setup evaluated ``cfg.task`` and
    the step shape for every mode, so a stub config without ``task`` raised and
    the cleanup then called ``close()`` on a stub runtime that has none (#2152,
    test_issue803 / test_qwen35_streaming on CI)."""

    @pytest.mark.parametrize("spelled", [False, True])
    def test_a_minimal_stub_setup_makes_no_new_calls(self, monkeypatch, tmp_path, spelled):
        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.trainer.stream_setup import StreamingSetupMixin
        from soup_cli.utils.layer_stream import DEFAULT_STREAM_READ_AHEAD
        from tests.test_issue803_lora_patterns_all_trainers import _config, _tokenizer

        def never(name):
            def refuse(*_a, **_k):
                raise AssertionError(f"{name} was reached on the default path")

            return refuse

        class _Wrapper(StreamingSetupMixin):
            def __init__(self):
                self.device = "cpu"
                self._trust_remote_code = False

            def _stream_budget_lines(self, *_args, **_kwargs):
                return (), None

        for name in ("_configure_stream_recompute", "_stream_step_shape"):
            monkeypatch.setattr(StreamingSetupMixin, name, never(name))
        for name in ("set_stream_recompute", "plan_stream_recompute_auto"):
            monkeypatch.setattr(rt, name, never(name))

        extra = {"stream_recompute": "on"} if spelled else {}
        tcfg = SimpleNamespace(
            quantization="none",
            double_quant_on=True,
            stream_source="auto",
            stream_buffers=2,
            stream_disk_kind=None,
            stream_pin=None,
            seed=7,
            moe_lora=False,
            batch_size=1,
            gradient_accumulation_steps=1,
            stream_vram_probe=False,
            stream_vram_override=None,
            stream_read_ahead=DEFAULT_STREAM_READ_AHEAD,
            lora=_config("sft").training.lora,
            **extra,
        )
        # No `task`, no `data.max_length`: nothing on the default path may need them.
        cfg = SimpleNamespace(base="Qwen/Qwen3-0.6B", data=SimpleNamespace())
        model_cfg = SimpleNamespace(
            model_type="qwen3", hidden_size=64, num_hidden_layers=2, vocab_size=128
        )
        # Only `stats`: no `close`, so any cleanup call would raise here.
        runtime = SimpleNamespace(
            stats=lambda: {
                "tier": "ram",
                "store_bytes": 0,
                "pinned": False,
                "n_layers": 2,
                "buffers": 2,
                "buffer_bytes": 8,
            }
        )
        built = SimpleNamespace()
        fakes = {
            "transformers.AutoTokenizer.from_pretrained": lambda *_a, **_k: _tokenizer(),
            "transformers.AutoConfig.from_pretrained": lambda *_a, **_k: model_cfg,
            "soup_cli.utils.layer_shard.resolve_shard_dir": lambda *_a, **_k: str(tmp_path / "s"),
            "soup_cli.utils.layer_shard.shard_checkpoint": lambda *_a, **_k: SimpleNamespace(
                n_layers=2, total_params=100, quant="none", quant_specs={}
            ),
            "soup_cli.utils.layer_shard.source_weight_bytes": lambda *_a, **_k: 1024,
            "soup_cli.utils.spectrum_scan.resolve_model_weights": (
                lambda *_a, **_k: str(tmp_path / "w")
            ),
            "soup_cli.utils.layer_stream.free_ram_bytes": lambda: 1_000_000,
            "soup_cli.utils.layer_stream_runtime.build_meta_skeleton": (
                lambda *_a, **_k: SimpleNamespace()
            ),
            "soup_cli.utils.layer_stream_runtime.RamSource.layer_specs_from_paths": (
                lambda *_a, **_k: [
                    {"self_attn.q_proj.weight": ((4, 4), "float32")},
                    {"self_attn.q_proj.weight": ((4, 4), "float32")},
                ]
            ),
            "soup_cli.utils.layer_stream_runtime.extras_resident_bytes": lambda *_a, **_k: 0,
            "soup_cli.utils.layer_stream_runtime.build_streamed_model": (
                lambda **_k: (built, runtime)
            ),
            "soup_cli.utils.moe.detect_moe_model": lambda *_a, **_k: False,
            "soup_cli.utils.layer_stream.render_stream_panel": lambda *_a, **_k: "panel",
            "soup_cli.utils.layer_stream_runtime.expandable_segments_status": lambda: (True, ""),
        }
        for target, fake in fakes.items():
            monkeypatch.setattr(target, fake)

        wrapper = _Wrapper()
        wrapper._setup_streaming_transformers(cfg, tcfg)
        assert wrapper.model is built
        assert wrapper._stream_runtime is runtime
