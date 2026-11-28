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


@pytest.mark.parametrize("key", ["l2l_split_overlap", "l2l_split_defer_read"])
def test_split_scheduling_config_is_opt_in(key):
    assert getattr(_config().training, key) is False
    assert getattr(_config(l2l_split_gpus=2, **{key: True}).training, key) is True


@pytest.mark.parametrize("key", ["l2l_split_overlap", "l2l_split_defer_read"])
@pytest.mark.parametrize("value", [1, 0, "yes"])
def test_split_scheduling_config_requires_bool(key, value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        _config(l2l_split_gpus=2, **{key: value})
    error, = exc.value.errors()
    assert error["loc"] == ("training", key)
    assert error["msg"] == (
        f"Value error, training.{key} must be true or false, got {type(value).__name__} {value!r}"
    )


@pytest.mark.parametrize("key", ["l2l_split_overlap", "l2l_split_defer_read"])
@pytest.mark.parametrize(
    "training", [{}, {"l2l": False, "l2l_split_gpus": 2}, {"task": "ipo", "l2l_split_gpus": 2}]
)
def test_split_scheduling_config_requires_split_sft(key, training):
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        _config(**{key: True}, **training)
    assert exc.value.errors()[0]["msg"] == (
        f"Value error, training.{key} requires training.l2l=true and "
        "training.l2l_split_gpus>1 for text SFT with backend='transformers'."
    )


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

    from soup_cli.trainer.l2l_trainer import _l2l_step_for
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr("soup_cli.utils.layer_stream.resolve_stream_dtype", lambda _: "float32")
    wrapper = StreamingSetupMixin()
    wrapper.device = "cuda"
    wrapper.config = _config(l2l_split_gpus=2, l2l_split_overlap=True, l2l_split_defer_read=True)
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

    def unexpected_cuda_sync(*_args, **_kwargs):
        pytest.fail("CPU layer-call capture must not synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_cuda_sync)
    stepper, _calls = _l2l_step_for(state, wrapper.model, _micro_batches(2))
    assert state.captures == 1
    assert stepper.split_overlap is True
    assert stepper.split_defer_read is True
    state.close()
    assert state.stepper is None


def _make_step(model, runtime, ids, split=(), **options):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    return L2LStep(
        model, runtime, capture_layer_call(model, ids, draws=False), split_devices=split, **options
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
@pytest.mark.parametrize("defer", [False, True], ids=["early-read", "deferred-read"])
def test_split_window_matches_single_device_loss_gradients_and_reads(
    streamed, k, devices, defer  # noqa: F811
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
        step = _make_step(model, runtime, batches[0], devices, split_defer_read=defer)
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
        # The streaming wrapper enumerates and runs the inner embedding.
        embedding = step.owner.embed_tokens
        offset = torch.nn.Parameter(torch.zeros(1))
        monkeypatch.setattr(embedding.inner, "offset", offset, raising=False)
        embedding.inner.register_forward_hook(lambda module, _args, output: output + module.offset)
        assert any(param is offset and param.requires_grad for param in embedding.parameters())

        def unexpected_prime(*_args):
            pytest.fail("the split step primed the model with a trainable embedding")

        monkeypatch.setattr(step, "_prime", unexpected_prime)
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


@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("hook", [None, "wrapper", "global"])
def test_split_read_order_preserves_hooks(streamed, monkeypatch, defer, hook):  # noqa: F811
    from torch.nn.modules.module import register_module_forward_pre_hook

    model, runtime = streamed
    batches = _micro_batches(2)
    step = _make_step(model, runtime, batches[0], ("cpu",), split_defer_read=defer)
    events = []
    advance = runtime.prefetcher.advance

    def record_advance(idx):
        events.append(("read", idx))
        advance(idx)

    monkeypatch.setattr(runtime.prefetcher, "advance", record_advance)

    def watch(idx):
        def record_compute(_layer, _args):
            events.append(("compute", idx))
        return record_compute

    handles = [layer.inner.register_forward_pre_hook(watch(idx))
               for idx, layer in enumerate(step.layers)]
    def record_wrapper(module, _args):
        if module is step.layers[1]:
            events.append(("wrapper", 1))

    if hook == "wrapper":
        handles.append(step.layers[1].register_forward_pre_hook(record_wrapper))
    elif hook == "global":
        handles.append(register_module_forward_pre_hook(record_wrapper))
    try:
        with closing(_store(2)) as store:
            step.run(batches, store)
    finally:
        for handle in handles:
            handle.remove()

    expected = (
        ["compute", "compute", "read"] if defer and not hook
        else (["wrapper"] if hook else []) + ["read", "compute", "compute"]
    )
    assert [kind for kind, idx in events if idx == 1] == expected * 2


def test_split_overlap_uses_side_streams_only_when_requested(monkeypatch):
    import torch

    from soup_cli.utils.l2l_split import SplitTransfers

    streams = []

    def new_stream(device):
        streams.append(str(device))
        return object()

    monkeypatch.setattr(torch.cuda, "Stream", new_stream)
    assert not SplitTransfers("cuda:0", ["cuda:1"]).enabled
    assert not SplitTransfers("cpu", ["cpu"], overlap=True).enabled
    assert not SplitTransfers("cuda:0", [], overlap=True).enabled
    assert streams == []
    assert SplitTransfers("cuda:0", ["cuda:1"], overlap=True).enabled
    assert streams == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]


def test_split_profile_requires_opt_in_and_cuda(monkeypatch, tmp_path):
    from soup_cli.utils.l2l_split import PROFILE_ENV, SplitProfile

    monkeypatch.delenv(PROFILE_ENV, raising=False)
    assert not SplitProfile(["cuda:0", "cuda:1"]).enabled
    path = tmp_path / "profile.jsonl"
    monkeypatch.setenv(PROFILE_ENV, str(path))
    assert SplitProfile(["cuda:0", "cuda:1"]).enabled
    profile = SplitProfile(["cpu", "cpu"])
    assert not profile.enabled
    profile.start()
    with profile.span("fwd"):
        pass
    assert profile.finish(2) is None
    assert not path.exists()


def test_split_profile_aggregates_phases_and_appends(monkeypatch, tmp_path):
    import json

    import torch

    from soup_cli.utils.l2l_split import PROFILE_ENV, SplitProfile

    class Event:
        def __init__(self, time):
            self.time = time

        def elapsed_time(self, other):
            return other.time - self.time

    clock = {"cuda:0": 0.0, "cuda:1": 0.0}

    def fake_event(_profile, device):
        clock[str(device)] += 10.0
        return Event(clock[str(device)])

    synchronized = []
    path = tmp_path / "profile.jsonl"
    monkeypatch.setenv(PROFILE_ENV, str(path))
    monkeypatch.setattr(SplitProfile, "_event", fake_event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(str(device)))
    profile = SplitProfile(["cuda:0", "cuda:1"])
    records = []
    for k in (3, 2):
        profile.start()
        for _ in range(2):
            with profile.span("fwd", "cuda:1"):
                pass
        with profile.span("load"):
            pass
        records.append(profile.finish(k))

    assert synchronized == ["cuda:0", "cuda:1"] * 2
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == records
    for record, k in zip(records, (3, 2)):
        assert record["devices"] == {
            "cuda:0": {"wall_ms": 30.0, "busy_ms": {"load": 10.0}, "idle_ms": 20.0},
            "cuda:1": {"wall_ms": 70.0, "busy_ms": {"fwd": 20.0, "load": 10.0}, "idle_ms": 40.0},
        }
        assert record["k"] == k
        assert set(record["host_ms"]) == {"fwd", "load"}


@pytest.mark.gpu
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("overlap", [False, True], ids=["plain-copy", "overlapped-copy"])
@pytest.mark.parametrize("defer", [False, True], ids=["early-read", "deferred-read"])
def test_two_cuda_split_matches_single_device_after_update(
    tmp_path, monkeypatch, bf16, overlap, defer
):
    import json

    import torch

    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_split import PROFILE_ENV, DeviceReplica

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    if bf16 and any(torch.cuda.get_device_capability(i)[0] < 8 for i in range(2)):
        pytest.skip("requires native BF16 on both devices")
    profile_path = tmp_path / "split.jsonl"
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    if overlap and defer:
        monkeypatch.setenv(PROFILE_ENV, str(profile_path))
    with torch.cuda.device(0):
        model, runtime = build_streamed(tmp_path, device="cuda:0")
        with closing(runtime):
            _check_two_cuda_split(
                model, runtime, monkeypatch, bf16, ActivationStore, DeviceReplica,
                split_overlap=overlap, split_defer_read=defer,
            )
    if overlap and defer:
        lines = profile_path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert len(records) == 2
        phases = {"load", "copy_ahead", "read", "put_cross", "get_cross"}
        for record in records:
            assert record["k"] == 3
            assert set(record["devices"]) == {"cuda:0", "cuda:1"}
            assert phases <= record["host_ms"].keys()
    else:
        assert not profile_path.exists()


def _check_two_cuda_split(model, runtime, monkeypatch, bf16, store_cls, replica_cls, **options):
    import torch

    harness = _load("l2l_schedule")
    harness.make_non_vacuous(model)
    batches = [ids.to("cuda:0") for ids in _micro_batches(3)]
    single = _make_step(model, runtime, batches[0])
    split = _make_step(model, runtime, batches[0], ("cuda:1",), **options)
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


@pytest.mark.gpu
@pytest.mark.parametrize("overlap", [False, True], ids=["plain-copy", "overlapped-copy"])
@pytest.mark.parametrize("defer", [False, True], ids=["early-read", "deferred-read"])
def test_two_cuda_split_matches_chunked_fp16_with_scaler(tmp_path, overlap, defer):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import label_tokens
    from soup_cli.utils.stream_loss_chunk import chunked_causal_lm_loss

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with torch.cuda.device(0):
        model, runtime = build_streamed(tmp_path, device="cuda:0")
        with closing(runtime), closing(ActivationStore(
            n_layers=3, k=3, chunk_shape=(1, 8, 64), dtype=torch.float32, device="cuda:0",
        )) as store:
            harness = _load("l2l_schedule")
            harness.make_non_vacuous(model)
            batches = [ids.to("cuda:0") for ids in _micro_batches(3)]
            single = _make_step(model, runtime, batches[0])
            split = _make_step(
                model, runtime, batches[0], ("cuda:1",),
                split_overlap=overlap, split_defer_read=defer,
            )
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)

            def run(step):
                optimizer.zero_grad(set_to_none=True)
                scaler = torch.amp.GradScaler("cuda", init_scale=128)

                def head(causal, owner, hidden, mb):
                    # Exercise the shipped chunked primitive through the step's head contract.
                    loss, _ = chunked_causal_lm_loss(
                        owner.norm(hidden), causal.lm_head, batches[mb], 3,
                        num_items_in_batch=label_tokens(batches),
                    )
                    return scaler.scale(loss)

                with torch.autocast("cuda", dtype=torch.float16):
                    losses = step.run(batches, store, head=head)
                scaler.unscale_(optimizer)
                return torch.stack(losses) / scaler.get_scale(), harness.collect_grads(model)

            expected_loss, expected_grads = run(single)
            for _ in range(2):
                loss, grads = run(split)
                torch.testing.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-5)
                _assert_gradients_match(grads, expected_grads, rtol=0.02, atol=1e-5)
