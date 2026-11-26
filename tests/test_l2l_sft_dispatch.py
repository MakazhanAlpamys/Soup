"""``training.l2l`` through the SFT wrapper: the recipe, the dispatch and the closed codec.

``SFTTrainerWrapper.setup`` is driven with the model load replaced by a sentinel, so the
tests see what happens before a model exists and which trainer builder the chain picks.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.trainer.sft import SFTTrainerWrapper, rewind_skip_reason
from soup_cli.trainer.stream_setup import StreamingSetupMixin


def _cfg(**training):
    fields = {
        "batch_size": 1,
        "quantization": "none",
        "stream_layers": True,
        "gradient_accumulation_steps": 2,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
    }
    fields.update(training)
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "task": "sft",
                "data": {"train": "data.jsonl"},
                "training": fields,
            }
        )
    )


class _ReachedError(Exception):
    """The streamed model load, which the tests replace."""


def _wrapper(monkeypatch, **training):
    wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
    wrapper.config = _cfg(**training)
    wrapper.device = "cpu"

    def reached(self, cfg, tcfg):
        raise _ReachedError

    monkeypatch.setattr(SFTTrainerWrapper, "_setup_streaming_transformers", reached)
    return wrapper


def test_a_config_without_l2l_resolves_no_recipe_and_loads_as_before(monkeypatch):
    wrapper = _wrapper(monkeypatch)
    with pytest.raises(_ReachedError):
        wrapper.setup({})
    assert wrapper._l2l_recipe is None


def test_l2l_resolves_the_recipe_before_the_model_load(monkeypatch):
    wrapper = _wrapper(monkeypatch, l2l=True, gradient_accumulation_steps=4)
    with pytest.raises(_ReachedError):
        wrapper.setup({})
    assert (wrapper._l2l_recipe.requested_k, wrapper._l2l_recipe.effective_k) == (4, 4)


def test_fp8_is_refused_before_the_model_load_with_the_exact_message(monkeypatch):
    wrapper = _wrapper(monkeypatch, l2l=True, l2l_activation_codec="fp8_e4m3")
    with pytest.raises(ValueError) as caught:
        wrapper.setup({})
    assert str(caught.value) == (
        "training.l2l_activation_codec='fp8_e4m3' is not available yet: the launch "
        "supports 'none'. Set it to 'none'."
    )


def test_auto_k_is_refused_before_the_model_load(monkeypatch):
    wrapper = _wrapper(monkeypatch, l2l=True, l2l_auto_k=True, stream_expert_layout="per_expert")
    with pytest.raises(ValueError, match="l2l_auto_k plans the window of a MoE model"):
        wrapper.setup({})


def _builder(monkeypatch, codec, calls, k=2):
    from soup_cli.trainer import l2l_trainer

    def guard(*args, **kwargs):
        calls.append("guard")

    monkeypatch.setattr(l2l_trainer, "refuse_unsupported_l2l", guard)
    monkeypatch.setattr(
        l2l_trainer, "make_l2l_trainer_class", lambda base: lambda **kw: calls.append("class")
    )
    monkeypatch.setattr(
        l2l_trainer, "attach_l2l_state", lambda *args, **kwargs: calls.append(kwargs)
    )
    builder = SimpleNamespace(
        config=SimpleNamespace(
            task="sft",
            training=SimpleNamespace(
                l2l_activation_codec=codec, l2l_spill_dir=None, stream_expert_layout="fused"
            ),
        ),
        model=SimpleNamespace(config=SimpleNamespace(model_type="llama")),
        device="cpu",
        _l2l_recipe=None,
        _stream_runtime=object(),
        _stream_weight_dirs=("cache",),
    )
    args = SimpleNamespace(n_gpu=1, fp16=False, gradient_accumulation_steps=k)
    return builder, lambda: StreamingSetupMixin._build_l2l_trainer(builder, object, {}, args)


def test_the_dispatch_never_passes_fp8_to_the_trainer_while_the_gate_is_closed(monkeypatch):
    calls = []
    _owner, build = _builder(monkeypatch, "fp8_e4m3", calls)
    with pytest.raises(ValueError, match="is not available yet: the launch supports 'none'"):
        build()
    assert calls == []


def test_the_builder_passes_the_none_codec_spill_dir_and_k_to_the_trainer(monkeypatch):
    calls = []
    _owner, build = _builder(monkeypatch, "none", calls, k=3)
    build()
    attach = calls[-1]
    assert attach["activation_codec"] == "none"
    assert attach["effective_k"] == 3
    assert attach["spill_dir"] is None
    assert attach["weight_dirs"] == ("cache",)


def test_the_builder_refuses_a_k_that_differs_from_the_recipe(monkeypatch):
    calls = []
    owner, build = _builder(monkeypatch, "none", calls, k=2)
    owner._l2l_recipe = SimpleNamespace(effective_k=4)
    with pytest.raises(ValueError, match="must equal the L2L recipe's effective k"):
        build()


def test_l2l_is_a_named_reason_for_no_rewind_log():
    reason = rewind_skip_reason(
        task="sft",
        pretokenized=False,
        is_raft=False,
        multipack=False,
        vision=False,
        audio=False,
        l2l=True,
    )
    assert reason == "L2L runs the layers outside the loss path the recorder wraps"


def test_the_sft_trainer_chain_dispatches_to_the_l2l_builder_before_the_rewind_trainer():
    source = inspect.getsource(SFTTrainerWrapper._finish_setup)
    assert "elif tcfg.l2l:" in source
    assert source.index("elif tcfg.l2l:") < source.index("self._build_rewind_trainer(")
    assert source.index("elif use_vision and not tcfg.packing:") < source.index("elif tcfg.l2l:")
