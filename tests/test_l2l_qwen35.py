"""CPU L2L parity for the Qwen3.5 hybrid decoder.

Transformers supplies a recurrent mask to linear-attention layers and a causal mask to
full-attention layers. L2L must replay each layer type's arguments to match accumulation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, _micro_batches, hermetic_ram

HIDDEN_SIZE = 64
LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention", "full_attention")


@pytest.fixture()
def qwen35_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Any, Any, Any]]:
    pytest.importorskip("peft")
    import torch
    from peft import LoraConfig, TaskType
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=2 * HIDDEN_SIZE,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=list(LAYER_TYPES),
        tie_word_embeddings=False,
        max_position_embeddings=128,
    )
    weights = str(tmp_path / "weights")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        resident = Qwen3_5ForCausalLM(config).to(torch.float32).eval()
    resident.save_pretrained(weights, safe_serialization=True)

    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch=stream_arch_of(config))
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model, runtime = build_streamed_model(
        model_id=weights,
        shard_dir=shards,
        index=index,
        lora_config=lora,
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=3,
    )
    try:
        hermetic_ram(monkeypatch, tmp_path)
        # Eager attention materializes the causal mask even for rows without padding.
        model.get_base_model().set_attn_implementation("eager")
        yield model, runtime, config
    finally:
        runtime.close()


def _store(k: int) -> Any:
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    return ActivationStore(
        n_layers=len(LAYER_TYPES),
        k=k,
        chunk_shape=(1, SEQ, HIDDEN_SIZE),
        dtype=torch.float32,
        device="cpu",
        pin=False,
    )


def test_fixture_has_both_layer_types(qwen35_streamed: tuple[Any, Any, Any]) -> None:
    _model, _runtime, config = qwen35_streamed
    assert tuple(config.layer_types) == LAYER_TYPES


def test_capture_records_arguments_per_layer_type(qwen35_streamed: tuple[Any, Any, Any]) -> None:
    import torch

    from soup_cli.utils.l2l_step import capture_layer_call, model_parts

    model, _runtime, config = qwen35_streamed
    ids = _micro_batches(1, seed=29)[0]
    call = capture_layer_call(model, ids, draws=False)
    assert set(call.typed) == {"linear_attention", "full_attention"}
    assert call.layer_types == tuple(config.layer_types)
    assert call.for_layer(0)[1]["attention_mask"] is None
    assert torch.is_tensor(call.for_layer(3)[1]["attention_mask"])

    _causal, owner = model_parts(model)
    actual = {}

    def layer_hook(idx: int) -> Any:
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            actual[idx] = (tuple(args[1:]), dict(kwargs))

        return hook

    handles = [
        layer.register_forward_pre_hook(layer_hook(idx), with_kwargs=True)
        for idx, layer in enumerate(owner.layers)
    ]
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for handle in handles:
            handle.remove()

    def assert_same(got: Any, want: Any) -> None:
        if torch.is_tensor(want):
            assert torch.is_tensor(got) and torch.equal(got, want)
        elif isinstance(want, (tuple, list)):
            assert isinstance(got, type(want)) and len(got) == len(want)
            for item, expected in zip(got, want):
                assert_same(item, expected)
        else:
            assert got == want

    assert set(actual) == set(range(len(owner.layers)))
    for idx, (want_args, want_kwargs) in actual.items():
        args, kwargs = call.for_layer(idx)
        want_kwargs.pop("num_items_in_batch", None)
        assert_same(args, want_args)
        assert kwargs.keys() == want_kwargs.keys()
        for key in kwargs:
            assert_same(kwargs[key], want_kwargs[key])


@pytest.mark.parametrize("k", [1, 2, 3])
def test_l2l_step_equals_gradient_accumulation_on_qwen35(
    qwen35_streamed: tuple[Any, Any, Any], k: int
) -> None:
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    harness = _load("l2l_schedule")
    model, runtime, _config = qwen35_streamed
    harness.make_non_vacuous(model)
    batches = _micro_batches(k, seed=29)

    harness.zero_grads(model)
    ga_losses = harness.ga_step(model, batches)
    ga = harness.collect_grads(model)

    step = L2LStep(model, runtime, capture_layer_call(model, batches[0]))
    harness.zero_grads(model)
    store = _store(k)
    try:
        l2l_losses = step.run(batches, store)
    finally:
        store.close()
    l2l = harness.collect_grads(model)

    result = harness.compare(ga, l2l)
    assert result["all_zero"] == []
    assert result["equal"], result
    assert harness.losses_equal(ga_losses, l2l_losses)


def test_reference_pass_replays_qwen35_layer_types(
    qwen35_streamed: tuple[Any, Any, Any],
) -> None:
    import torch

    from soup_cli.utils.l2l_step import L2LStep, ReferencePass, capture_layer_call

    model, runtime, _config = qwen35_streamed
    ids = _micro_batches(1, seed=29)[0]
    with torch.no_grad():
        expected = model(input_ids=ids).logits

    def head(causal: Any, owner: Any, hidden: Any, _mb: int) -> Any:
        return causal.lm_head(owner.norm(hidden))

    reference = ReferencePass(head=head)
    step = L2LStep(model, runtime, capture_layer_call(model, ids, draws=False))
    store = _store(1)
    try:
        step.run([ids], store, reference=reference)
    finally:
        store.close()
    torch.testing.assert_close(reference.outputs[0], expected, rtol=0, atol=0)


def test_trainer_admits_qwen35() -> None:
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    for arch in ("qwen3_5", "qwen3_5_text"):
        refuse_unsupported_l2l(SimpleNamespace(model_type=arch), n_gpu=1)


@pytest.mark.parametrize(
    ("arch", "layer_types", "layers_called", "error_type", "message"),
    [
        (
            "qwen3_5_text",
            ("linear_attention",),
            0,
            ValueError,
            "config.layer_types names 1 layers, the decoder has 2",
        ),
        (
            "qwen3_5",
            ("linear_attention", "full_attention"),
            1,
            RuntimeError,
            "decoder layer(s) [1] never called by the forward",
        ),
        (
            "llama",
            ("full_attention",),
            0,
            RuntimeError,
            "decoder layer 0 was never called by the forward",
        ),
    ],
    ids=["too-few-layer-types", "missing-layer-type", "dense-missing-first-layer"],
)
def test_capture_refuses_invalid_decoder_calls(
    arch: str,
    layer_types: tuple[str, ...],
    layers_called: int,
    error_type: type[Exception],
    message: str,
) -> None:
    import torch
    from torch import nn

    from soup_cli.utils.l2l_step import capture_layer_call

    class CausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type=arch, layer_types=layer_types)
            if arch == "qwen3_5":
                text_config = SimpleNamespace(layer_types=layer_types)
                self.config = SimpleNamespace(model_type=arch, get_text_config=lambda: text_config)
            self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.lm_head = nn.Identity()
            self.loss_function = object()

        def forward(self, input_ids: torch.Tensor) -> None:
            for idx in range(layers_called):
                self.layers[idx](input_ids)

    with pytest.raises(error_type) as caught:
        capture_layer_call(CausalLM(), torch.zeros(1, SEQ, dtype=torch.long), draws=False)
    assert str(caught.value) == message


@pytest.mark.parametrize(
    "text_attrs",
    [
        pytest.param({}, id="missing"),
        pytest.param({"layer_types": None}, id="none"),
        pytest.param({"layer_types": []}, id="empty-list"),
        pytest.param({"layer_types": "full_attention"}, id="non-list"),
        pytest.param({"layer_types": ["linear_attention", 1]}, id="non-string-item"),
    ],
)
def test_capture_refuses_qwen35_without_layer_types(text_attrs: dict[str, Any]) -> None:
    import torch
    from torch import nn

    from soup_cli.utils.l2l_step import capture_layer_call

    forwards: list[Any] = []
    text_config = SimpleNamespace(**text_attrs)

    class CausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_5", get_text_config=lambda: text_config)
            self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.lm_head = nn.Identity()
            self.loss_function = object()

        def forward(self, input_ids: torch.Tensor) -> None:
            forwards.append(input_ids)

    with pytest.raises(ValueError) as caught:
        capture_layer_call(CausalLM(), torch.zeros(1, SEQ, dtype=torch.long), draws=False)
    assert str(caught.value) == (
        "qwen3_5 config.layer_types must be a non-empty list of layer-type strings, "
        "one per decoder layer: the decoder has 2"
    )
    assert forwards == []


def _two_layer_qwen35(*, fail_after_first_layer: bool) -> Any:
    import torch
    from torch import nn

    text_config = SimpleNamespace(layer_types=["linear_attention", "full_attention"])

    class CausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_5", get_text_config=lambda: text_config)
            self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.lm_head = nn.Identity()
            self.loss_function = object()

        def forward(self, input_ids: torch.Tensor) -> None:
            self.layers[0](input_ids)
            if fail_after_first_layer:
                raise RuntimeError("forward failed")
            self.layers[1](input_ids)

    return CausalLM()


def _pending_pre_hooks(model: Any) -> int:
    # Torch keeps forward pre-hooks in a private dict; removing a handle drops its entry.
    return sum(len(layer._forward_pre_hooks) for layer in model.layers)


def test_capture_removes_its_hooks_after_a_normal_capture() -> None:
    import torch

    from soup_cli.utils.l2l_step import capture_layer_call

    model = _two_layer_qwen35(fail_after_first_layer=False)
    call = capture_layer_call(model, torch.zeros(1, SEQ, dtype=torch.long), draws=False)
    assert call.layer_types == ("linear_attention", "full_attention")
    assert _pending_pre_hooks(model) == 0


def test_capture_removes_its_hooks_when_the_forward_raises() -> None:
    import torch

    from soup_cli.utils.l2l_step import capture_layer_call

    model = _two_layer_qwen35(fail_after_first_layer=True)
    with pytest.raises(RuntimeError) as caught:
        capture_layer_call(model, torch.zeros(1, SEQ, dtype=torch.long), draws=False)
    assert str(caught.value) == "forward failed"
    assert _pending_pre_hooks(model) == 0
