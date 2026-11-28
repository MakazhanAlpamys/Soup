"""Split windows use the shipped streaming runtime and one adapter optimizer."""

from contextlib import closing, nullcontext
from types import SimpleNamespace

import pytest

from tests.test_l2l_probe_harness import (  # noqa: F401
    _load,
    _micro_batches,
    build_streamed,
    streamed,
)
from tests.test_l2l_step import _store


def _config(task="sft", **training):
    from soup_cli.config.schema import SoupConfig

    return SoupConfig.model_validate(
        {
            "base": "example/model",
            "task": task,
            "data": {"train": "train.jsonl", "format": "chatml"},
            "training": {
                "stream_layers": True,
                "batch_size": 1,
                "quantization": "none",
                "l2l": True,
                "lora": {"dropout": 0},
                **training,
            },
        }
    )


def test_split_config_defaults_and_bounds():
    from soup_cli.config.schema import TrainingConfig

    assert _config().training.l2l_split_gpus == 1
    assert _config(l2l_split_gpus=2).training.l2l_split_gpus == 2
    assert TrainingConfig(l2l_split_gpus=8).l2l_split_gpus == 8


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0, "Input should be greater than or equal to 1"),
        (9, "Input should be less than or equal to 8"),
    ],
)
def test_split_gpu_count_outside_bounds(value, message):
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        _config(l2l_split_gpus=value)
    error, = exc.value.errors()
    assert error["loc"] == ("training", "l2l_split_gpus")
    assert error["msg"] == message


@pytest.mark.parametrize(
    ("training", "message"),
    [
        (
            {"task": "ipo"},
            "training.l2l_split_gpus requires task='sft'; got task='ipo'.",
        ),
        (
            {"l2l": False},
            "training.l2l_split_gpus is set but training.l2l is false: it changes how "
            "L2L runs and nothing else reads it. Set l2l=true or drop l2l_split_gpus.",
        ),
        (
            {"lora": {"dropout": 0.05}},
            "training.l2l_split_gpus requires training.lora.dropout=0 for SFT.",
        ),
        (
            {"stream_loss_chunk": 16},
            "training.l2l cannot be combined with training.stream_loss_chunk: each builds "
            "its own trainer or replaces how the loss is computed (the L2L loss bypasses "
            "the chunked-loss hook), so L2L would skip it. Remove one of the two.",
        ),
    ],
)
def test_split_config_refusals(training, message):
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        _config(l2l_split_gpus=2, **training)
    assert exc.value.errors()[0]["msg"].removeprefix("Value error, ") == message


def test_split_config_rejects_unavailable_prefix_cache_option():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        _config(
            l2l_split_gpus=2,
            lora_layers={"method": "fixed", "layers": [0], "prefix_cache": True},
        )
    error, = exc.value.errors()
    assert error["loc"] == ("training", "lora_layers", "prefix_cache")
    assert error["msg"] == "Extra inputs are not permitted"


def test_split_setup_refuses_cpu_before_model_work(monkeypatch):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    wrapper = StreamingSetupMixin()
    wrapper.device = "cpu"
    cfg = _config(l2l_split_gpus=2)
    with pytest.raises(ValueError) as exc:
        wrapper._setup_streaming_transformers(cfg, cfg.training)
    assert str(exc.value) == "training.l2l_split_gpus=2 requires CUDA for SFT; got device='cpu'."


def test_split_setup_checks_visible_gpu_indices(monkeypatch):
    import torch

    from soup_cli.utils.l2l_split import split_devices

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert split_devices("cuda", 2) == ("cuda:1",)
    with pytest.raises(ValueError) as exc:
        split_devices("cuda:1", 2)
    assert str(exc.value) == (
        "training.l2l_split_gpus=2 needs 2 CUDA devices starting at cuda:1; "
        "2 CUDA devices are visible. Select the cards with CUDA_VISIBLE_DEVICES."
    )


def test_split_setup_refuses_moe_before_tokenizer_or_weights(monkeypatch):
    import torch
    from transformers import AutoTokenizer

    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    def fail_if_called(*args, **kwargs):
        pytest.fail("the split preflight reached tokenizer loading")

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        "soup_cli.utils.kimi_k2.load_stream_model_config",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="deepseek_v3"),
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fail_if_called)
    wrapper = StreamingSetupMixin()
    wrapper.device = "cuda"
    cfg = _config(l2l_split_gpus=2)
    with pytest.raises(ValueError) as exc:
        wrapper._setup_streaming_transformers(cfg, cfg.training)
    assert str(exc.value) == (
        "training.l2l_split_gpus does not support routed experts (dense SFT only)."
    )


def test_builder_keeps_one_hf_gpu_and_attaches_split_devices(streamed, monkeypatch):  # noqa: F811
    import torch

    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr("soup_cli.utils.layer_stream.resolve_stream_dtype", lambda _: "float32")
    wrapper = StreamingSetupMixin()
    wrapper.device = "cuda"
    wrapper.config = _config(l2l_split_gpus=2)
    wrapper.model, wrapper._stream_runtime = streamed

    class Args(SimpleNamespace):
        @property
        def n_gpu(self):
            return self._n_gpu

    args = Args(_n_gpu=2, fp16=False, gradient_accumulation_steps=3)

    class Base:
        def __init__(self, args):
            assert args._n_gpu == 1
            self.args = args

    trainer = wrapper._build_l2l_trainer(Base, {"args": args}, args)
    state = trainer._soup_l2l
    assert state.split_devices == ("cuda:1",)
    assert state.effective_k == 3
    state.stepper = object()
    state.close()
    assert state.stepper is None


def _make_step(model, runtime, ids, split=()):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    return L2LStep(
        model, runtime, capture_layer_call(model, ids, draws=False), split_devices=split
    )


def _assert_gradients_match(actual, expected, *, rtol=1e-5, atol=1e-7):
    import torch

    assert set(actual) == set(expected) and actual
    assert all(grad.count_nonzero().item() for grad in expected.values())
    for name, grad in actual.items():
        torch.testing.assert_close(grad, expected[name], rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("k", "devices"), [(k, ("cpu",)) for k in (1, 2, 3, 4)] + [(4, ("cpu", "cpu"))]
)
def test_split_window_matches_single_device_loss_gradients_and_reads(
    streamed, k, devices  # noqa: F811
):
    import torch

    model, runtime = streamed
    harness = _load("l2l_schedule")
    harness.make_non_vacuous(model)
    batches = _micro_batches(k)

    def run(devices):
        model.zero_grad(set_to_none=True)
        # Give each step the same pool residency before measuring the reads.
        with torch.no_grad():
            model(batches[0])
        step = _make_step(model, runtime, batches[0], devices)
        with closing(_store(k)) as store:
            losses = step.run(batches, store)
        return torch.stack(losses), harness.collect_grads(model), step.counters.layer_loads

    expected_loss, expected_grads, expected_reads = run(())
    loss, grads, reads = run(devices)
    torch.testing.assert_close(loss, expected_loss)
    _assert_gradients_match(grads, expected_grads)
    assert reads == expected_reads


def test_split_refreshes_after_optimizer_step_and_adds_existing_gradients(
    streamed,  # noqa: F811
):
    model, runtime = streamed
    harness = _load("l2l_schedule")
    harness.make_non_vacuous(model)
    batches = _micro_batches(3)
    step = _make_step(model, runtime, batches[0], ("cpu",))
    single = _make_step(model, runtime, batches[0])
    import torch

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    with closing(_store(3)) as store:
        step.run(batches, store)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        single.run(batches, store)
        expected = harness.collect_grads(model)
        step.run(batches, store)
    _assert_gradients_match(harness.collect_grads(model), {n: 2 * g for n, g in expected.items()})


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("reference", "training.l2l_split_gpus does not support reference passes (SFT only)."),
        (
            "dropout",
            "training.l2l_split_gpus does not support model dropout in SFT. "
            "Set training.lora.dropout and model attention dropout to 0.",
        ),
        ("embedding", "training.l2l_split_gpus requires a frozen embedding for SFT."),
        ("experts", "training.l2l_split_gpus does not support routed experts (dense SFT only)."),
        ("prefix", "training.l2l_split_gpus cannot be combined with a frozen-prefix cache."),
    ],
)
def test_split_step_refuses_unsupported_model_or_window(
    streamed, monkeypatch, kind, message  # noqa: F811
):
    import torch

    from soup_cli.utils.l2l_step import ReferencePass

    model, runtime = streamed
    batches = _micro_batches(2)
    step = _make_step(model, runtime, batches[0], ("cpu",))
    kwargs = {}
    if kind == "reference":
        kwargs["reference"] = ReferencePass(head=lambda *_: None)
    elif kind == "dropout":
        monkeypatch.setattr(step.causal.config, "attention_dropout", 0.1)
    elif kind == "embedding":
        step.owner.embed_tokens.register_parameter("offset", torch.nn.Parameter(torch.zeros(1)))
    elif kind == "experts":
        monkeypatch.setattr("soup_cli.utils.l2l_step._streamed_moe_experts", lambda _: object())
    else:
        monkeypatch.setattr(step.owner, "_soup_prefix_cache", object(), raising=False)
    with closing(_store(2)) as store, pytest.raises(ValueError) as exc:
        step.run(batches, store, **kwargs)
    assert str(exc.value) == message


def test_split_calls_backward_once_per_micro_batch_and_layer(streamed, monkeypatch):  # noqa: F811
    import torch

    model, runtime = streamed
    batches = _micro_batches(4)
    step = _make_step(model, runtime, batches[0], ("cpu",))
    sizes = []
    backward = torch.autograd.backward

    def spy(tensors, *args, **kwargs):
        sizes.append(len(tensors) if isinstance(tensors, (list, tuple)) else 1)
        return backward(tensors, *args, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", spy)
    with closing(_store(4)) as store:
        step.run(batches, store)
    assert sizes == [1] * (4 * (len(step.layers) + 1))


@pytest.mark.gpu
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_two_cuda_split_matches_single_device_after_update(tmp_path, monkeypatch, bf16):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_split import DeviceReplica

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    if bf16 and any(torch.cuda.get_device_capability(i)[0] < 8 for i in range(2)):
        pytest.skip("requires native BF16 on both devices")
    with torch.cuda.device(0):
        model, runtime = build_streamed(tmp_path, device="cuda:0")
        with closing(runtime):
            _check_two_cuda_split(model, runtime, monkeypatch, bf16, ActivationStore, DeviceReplica)


def _check_two_cuda_split(model, runtime, monkeypatch, bf16, store_cls, replica_cls):
    import torch

    harness = _load("l2l_schedule")
    harness.make_non_vacuous(model)
    batches = [ids.to("cuda:0") for ids in _micro_batches(3)]
    single = _make_step(model, runtime, batches[0])
    split = _make_step(model, runtime, batches[0], ("cuda:1",))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    events = []
    backward_devices = []
    weights = replica_cls.weights
    backward = torch.autograd.backward

    def copy_weights(replica, layer, buffers):
        result = weights(replica, layer, buffers)
        events.append(("copy", layer.idx, replica.device.index))
        return result

    monkeypatch.setattr(replica_cls, "weights", copy_weights)

    def watch_backward(*args, **kwargs):
        assert not torch.is_autocast_enabled("cuda")
        backward_devices.append(torch.cuda.current_device())
        return backward(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", watch_backward)

    def watch_layer(idx):
        def watch(_module, args):
            assert torch.is_autocast_enabled("cuda") is bf16
            events.append(("compute", idx, args[0].device.index))
        return watch

    handles = [
        layer.inner.register_forward_pre_hook(watch_layer(idx))
        for idx, layer in enumerate(split.layers)
    ]

    def run(step):
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
        with closing(store_cls(
            n_layers=3, k=3, chunk_shape=(1, 8, 64), dtype=torch.float32, device="cuda:0",
        )) as store, ctx:
            return torch.stack(step.run(batches, store))

    try:
        run(split)
        optimizer.step()
        expected_loss = run(single)
        expected_grads = harness.collect_grads(model)
        events.clear()
        backward_devices.clear()
        actual_loss = run(split)
    finally:
        for handle in handles:
            handle.remove()
    _assert_gradients_match(
        harness.collect_grads(model), expected_grads,
        rtol=0.02 if bf16 else 1e-4, atol=1e-5 if bf16 else 1e-7,
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-5)
    assert events == [
        event
        for idx in (0, 1, 2, 2, 1, 0)
        for event in (
            ("copy", idx, 1), ("compute", idx, 0), ("compute", idx, 1), ("compute", idx, 0)
        )
    ]
    assert backward_devices == [0, 0, 0] + [0, 1, 0] * 3
