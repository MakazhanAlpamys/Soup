"""Residual-budget QKVO experiments must preserve native auto retention."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _helper():
    directory = Path(__file__).parents[1] / "benchmarks/harness"
    path = directory / "stream_projection_retention.py"
    spec = importlib.util.spec_from_file_location("stream_projection_retention", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _layers(retained=1):
    return [
        SimpleNamespace(
            idx=index,
            recompute=index < 4 - retained,
            inner=SimpleNamespace(
                self_attn=SimpleNamespace(
                    **{
                        name: SimpleNamespace(out_features=width)
                        for name, width in zip(
                            ("q_proj", "k_proj", "v_proj", "o_proj"), (4, 2, 2, 4)
                        )
                    }
                )
            ),
        )
        for index in range(4)
    ]


def _plan(budget=130, retained=1):
    return dict(
        measured=True,
        k=retained,
        budget_bytes=budget,
        full_layer_charge=100,
        peak_reserved=125,
        peak_allocated=100,
    )


@pytest.mark.parametrize("residual,count", [(0, 0), (14, 0), (15, 1), (29, 1), (30, 2)])
def test_residual_budget_boundary_preserves_the_full_retained_suffix(residual, count):
    helper = _helper()
    layers = _layers()
    result = helper.projection_plan(layers, _plan(100 + residual), batch=1, seq=1, element_bytes=1)
    assert result["retained_indices"] == [3]
    assert result["selected_indices"] == list(range(3 - count, 3))
    assert result["charged_bytes"] == count * 15
    assert result["raw_bytes"] == count * 12
    assert result["residual_bytes"] == residual
    assert [layer.recompute for layer in layers] == [True, True, True, False]


def test_gqa_count_uses_actual_four_output_widths():
    helper = _helper()
    layer = _layers()[0]
    for name, width in zip(("q_proj", "k_proj", "v_proj", "o_proj"), (4096, 1024, 1024, 4096)):
        getattr(layer.inner.self_attn, name).out_features = width
    assert helper.projection_bytes(layer, batch=1, seq=2048, element_bytes=2) == 41_943_040
    with pytest.raises(ValueError, match="positive"):
        helper.projection_bytes(layer, batch=1, seq=0, element_bytes=2)


def test_failed_probe_and_fully_retained_model_are_noops():
    helper = _helper()
    result = helper.projection_plan(_layers(), {"measured": False}, batch=1, seq=1, element_bytes=1)
    assert result["selected_indices"] == []
    assert result["reason"] == "native auto probe unavailable"
    result = helper.projection_plan(_layers(4), _plan(1000, 4), batch=1, seq=1, element_bytes=1)
    assert result["selected_indices"] == []
    assert result["reason"] == "all layers already retained"


def test_changed_native_decision_is_refused():
    helper = _helper()
    with pytest.raises(ValueError, match="native"):
        helper.projection_plan(_layers(2), _plan(), batch=1, seq=1, element_bytes=1)


def test_observer_preserves_the_native_plan_and_probe_cleanup(tmp_path):
    from soup_cli.utils.layer_stream_runtime import plan_stream_recompute_auto, set_stream_recompute
    from tests.test_stream_recompute import _FakeMemory, _kept, _stream

    helper = _helper()
    model, runtime = _stream(tmp_path, "none")
    try:
        kwargs = dict(rows=2, seq_len=16, vocab_size=64, notify=lambda message: None)
        assert plan_stream_recompute_auto(model, memory=_FakeMemory(), **kwargs) == 2
        set_stream_recompute(model, True)
        result = helper.observe_auto_plan(model, memory=_FakeMemory(), **kwargs)
        assert result["k"] == 2
        assert result["full_layer_bytes"] == 11808
        assert result["full_layer_charge"] == 14170
        assert result["budget_bytes"] == 30000
        assert _kept(model) == {2, 3}
        assert all(param.grad is None for param in model.parameters())
    finally:
        runtime.close()


def test_subset_refuses_retained_layers_without_patching(tmp_path):
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute
    from tests.test_stream_recompute import _stream

    helper = _helper()
    model, runtime = _stream(tmp_path, "none")
    try:
        set_stream_recompute(model, False, retain=1)
        modules = []
        model.apply(modules.append)
        forwards = [module.forward for module in modules]
        with pytest.raises(ValueError, match="recompute"):
            helper.install_subset(model, [1, 3])
        assert [module.forward for module in modules] == forwards
        assert helper.install_subset(model, []) is None
    finally:
        runtime.close()


def _mistral_stream(tmp_path, quant):
    from peft import LoraConfig, TaskType
    from transformers import MistralConfig, MistralForCausalLM

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )
    from tests.test_v07202 import _randomise_lora_b

    torch = pytest.importorskip("torch")
    torch.manual_seed(73)
    config = MistralConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        sliding_window=32, attention_dropout=0.0,
    )
    resident = MistralForCausalLM(config)
    weights, shards = str(tmp_path / "weights"), str(tmp_path / "shards")
    resident.save_pretrained(weights)
    suffixes = ()
    if quant == "nf4":
        skeleton = build_meta_skeleton(weights, dtype="float32", quant=quant)
        suffixes = quantised_layer_suffixes(skeleton)
        del skeleton
    index = shard_checkpoint(
        weights, shards, dtype="float32", arch="mistral", quant=quant,
        quant_suffixes=suffixes,
        quant_device="cpu" if quant == "nf4" else None, double_quant=True,
    )
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, device="cpu", dtype="float32",
        quant=quant, double_quant=True, buffers=2, pin=False, seed=3,
        lora_config=LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    _randomise_lora_b(model)
    return model, runtime


@pytest.mark.parametrize("family", ["llama", "mistral"])
@pytest.mark.parametrize("quant", ["none", "nf4"])
def test_mixed_modes_match_all_tensors_across_accumulation_and_two_updates(tmp_path, quant, family):
    torch = pytest.importorskip("torch")
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute, streamed_decoder_layers
    from tests.test_stream_recompute import _stream

    helper = _helper()
    records = []
    for mode in ("auto", "auto_qkvo"):
        build = _stream if family == "llama" else _mistral_stream
        model, runtime = build(tmp_path / mode, quant)
        controller = None
        try:
            set_stream_recompute(model, False, retain=1)
            layers = streamed_decoder_layers(model)
            native = [layer.forward for layer in layers]
            if mode == "auto_qkvo":
                controller = helper.install_subset(model, [1, 2])
                assert layers[0].forward == native[0]
                assert layers[3].forward == native[3]
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-4
            )
            rows = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                outputs = []
                for length in (8, 5):
                    ids = torch.randint(
                        0, 64, (1, length), generator=torch.Generator().manual_seed(5)
                    )
                    output = model(input_ids=ids, labels=ids)
                    outputs.extend((output.logits.detach().clone(), output.loss.detach().clone()))
                    if controller:
                        expected = sum(
                            helper.projection_bytes(layers[i], batch=1, seq=length, element_bytes=4)
                            for i in (1, 2)
                        )
                        assert controller.cache_bytes() == expected
                    (output.loss / 2).backward()
                    del output
                gradients = {
                    n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad
                }
                assert gradients and all(bool(g.abs().sum() > 0) for g in gradients.values())
                optimizer.step()
                parameters = {
                    n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
                }
                rows.append((outputs, gradients, parameters))
            records.append(rows)
            if controller:
                controller.remove()
                assert [layer.forward for layer in layers] == native
        finally:
            if controller:
                controller.remove()
            runtime.close()
    for left, right in zip(*records):
        for actual, expected in zip(left[0], right[0]):
            assert torch.equal(actual, expected)
        for actual, expected in zip(left[1:], right[1:]):
            assert actual.keys() == expected.keys()
            for name in actual:
                assert torch.equal(actual[name], expected[name]), name


def test_mutated_projection_cache_is_refused_in_mixed_mode(tmp_path):
    torch = pytest.importorskip("torch")
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute
    from tests.test_stream_recompute import _ids, _stream

    model, runtime = _stream(tmp_path, "none")
    controller = None
    try:
        set_stream_recompute(model, False, retain=1)
        controller = _helper().install_subset(model, [1, 2])
        output = model(input_ids=_ids(), labels=_ids())
        with torch.no_grad():
            controller.cached_tensors()[0].add_(1)
        with pytest.raises(RuntimeError, match="mutat"):
            output.loss.backward()
    finally:
        if controller:
            controller.remove()
        runtime.close()


def test_planner_reconfiguration_after_install_is_refused(tmp_path):
    from soup_cli.utils.layer_stream_runtime import set_stream_recompute
    from tests.test_stream_recompute import _ids, _stream

    model, runtime = _stream(tmp_path, "none")
    controller = None
    try:
        controller = _helper().install_subset(model, [1, 2])
        set_stream_recompute(model, False, retain=2)
        with pytest.raises(RuntimeError, match="mode changed"):
            model(input_ids=_ids(), labels=_ids())
    finally:
        if controller:
            controller.remove()
        runtime.close()


def test_no_full_layer_fit_can_still_have_a_valid_projection_budget():
    helper = _helper()
    result = helper.projection_plan(_layers(0), _plan(29, 0), batch=1, seq=1, element_bytes=1)
    assert result["retained_indices"] == []
    assert result["selected_indices"] == [3]
    assert result["charged_bytes"] == 15


def test_observer_restores_original_functions_after_a_failed_step(tmp_path):
    from soup_cli.utils import layer_stream
    from tests.test_stream_recompute import _FakeMemory, _kept, _stream

    model, runtime = _stream(tmp_path, "none")
    charge, budget = layer_stream.retained_layer_charge, layer_stream.stream_recompute_budget

    def broken(*args, **kwargs):
        raise ValueError("failed native step")

    try:
        with pytest.raises(ValueError, match="failed native step"):
            _helper().observe_auto_plan(
                model, rows=1, seq_len=8, vocab_size=64, memory=_FakeMemory(),
                notify=lambda message: None, step=broken,
            )
        assert layer_stream.retained_layer_charge is charge
        assert layer_stream.stream_recompute_budget is budget
        assert _kept(model) == set()
    finally:
        runtime.close()
