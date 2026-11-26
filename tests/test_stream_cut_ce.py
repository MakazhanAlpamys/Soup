"""Cut Cross-Entropy under layer streaming: the schema gate and the hook that runs it.

Streaming never installs the resident CCE patch (the model is on the meta device); with
``stream_layers`` the kernel runs through the head-and-loss hook of
``utils/stream_loss_chunk.py``. The kernel itself needs a CUDA card, so the hook tests stand
in a ``cut_cross_entropy`` module whose ``linear_cross_entropy`` is the exact loss.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig


def _config(**training) -> SoupConfig:
    return SoupConfig(
        base="local/model", data={"train": "train.jsonl"},
        training={"batch_size": 1, "quantization": "none", **training},
    )


# --- the schema ---------------------------------------------------------------------


def test_streamed_sft_cce_loads():
    cfg = _config(stream_layers=True, use_cut_ce=True)
    assert cfg.training.stream_layers and cfg.training.use_cut_ce


def test_public_yaml_accepts_streamed_cce():
    cfg = load_config_from_string(
        "base: local/model\ndata:\n  train: train.jsonl\ntraining:\n"
        "  batch_size: 1\n  stream_layers: true\n  use_cut_ce: true\n"
    )
    assert cfg.training.use_cut_ce


@pytest.mark.parametrize("task", ["dpo", "kto", "orpo", "simpo"])
def test_streamed_cce_refuses_a_preference_task(task):
    with pytest.raises(
        ValueError, match=r"training\.use_cut_ce with training\.stream_layers supports task='sft'"
    ):
        SoupConfig(
            base="local/model", task=task, data={"train": "train.jsonl"},
            training={"batch_size": 2, "stream_layers": True, "use_cut_ce": True},
        )


def test_streamed_cce_refuses_raft():
    with pytest.raises(ValueError, match="cannot be combined with data.format='raft'"):
        SoupConfig(
            base="local/model", data={"train": "train.jsonl", "format": "raft"},
            training={"batch_size": 1, "stream_layers": True, "use_cut_ce": True},
        )


def test_streamed_cce_refuses_l2l():
    with pytest.raises(ValueError, match=r"use_cut_ce.*cannot be combined with training\.l2l"):
        _config(stream_layers=True, use_cut_ce=True, l2l=True, gradient_accumulation_steps=2)


def test_supported_controls():
    assert _config(stream_layers=True).training.stream_layers
    assert not _config(stream_layers=True, use_cut_ce=False).training.use_cut_ce
    assert _config(use_cut_ce=True).training.use_cut_ce


# --- the loss function --------------------------------------------------------------


@pytest.fixture()
def fake_cce(monkeypatch):
    """A ``cut_cross_entropy`` whose ``linear_cross_entropy`` is the exact fp32 loss."""
    torch = pytest.importorskip("torch")
    calls = []

    def linear_cross_entropy(embeddings, classifier, targets, ignore_index=-100,
                             reduction="mean", **kwargs):
        calls.append({"dtype": embeddings.dtype, "reduction": reduction, "kwargs": kwargs})
        logits = embeddings.float() @ classifier.float().T
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=ignore_index, reduction=reduction,
        )

    module = types.ModuleType("cut_cross_entropy")
    module.linear_cross_entropy = linear_cross_entropy
    monkeypatch.setitem(sys.modules, "cut_cross_entropy", module)
    return calls


def _head_case():
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    head = torch.nn.Linear(16, 50, bias=False)
    hidden = torch.randn(2, 9, 16, requires_grad=True)
    labels = torch.randint(0, 50, (2, 9))
    labels[0, :3] = -100
    return head, hidden, labels


def _reference(head, hidden, labels, num_items=None):
    from transformers.loss.loss_utils import ForCausalLMLoss

    return ForCausalLMLoss(
        head(hidden), labels, vocab_size=50, num_items_in_batch=num_items
    )


@pytest.mark.parametrize("num_items", [None, 11])
def test_cce_loss_matches_the_causal_lm_loss(fake_cce, num_items):
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from soup_cli.utils.stream_loss_chunk import cce_causal_lm_loss

    head, hidden, labels = _head_case()
    items = None if num_items is None else torch.tensor(num_items)
    want = _reference(head, hidden, labels, items)
    got = cce_causal_lm_loss(hidden, head, labels, items)

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert fake_cce[0]["reduction"] == "sum"
    (grad_got,) = torch.autograd.grad(got, hidden)
    (grad_want,) = torch.autograd.grad(want, hidden)
    torch.testing.assert_close(grad_got, grad_want, rtol=1e-4, atol=1e-6)


def test_the_head_weight_of_a_streamed_large_layer_comes_from_the_pool():
    from soup_cli.utils.stream_loss_chunk import head_weight

    class Pool:
        def wait(self, key):
            return f"weight-of-{key}"

    streamed = SimpleNamespace(pool=Pool(), key="lm_head")
    assert head_weight(streamed) == "weight-of-lm_head"
    plain = SimpleNamespace(weight="w")
    assert head_weight(plain) == "w"


# --- the hook -----------------------------------------------------------------------


def _tiny_causal_lm(tie=False):
    from tests.test_stream_loss_chunk import _tiny_causal_lm as build

    return build(tie)


@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("with_items", [False, True])
def test_a_labelled_forward_through_the_cce_hook(fake_cce, tie, with_items):
    torch = pytest.importorskip("torch")
    from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss
    from tests.test_stream_loss_chunk import _close, _step

    kwargs = {"num_items_in_batch": torch.tensor(50)} if with_items else {}
    model = _tiny_causal_lm(tie)
    plain, plain_grads = _step(model, **kwargs)
    hook = ChunkedHeadLoss(model, None, "cce")
    try:
        outputs, grads = _step(model, **kwargs)
    finally:
        hook.remove()

    assert outputs.logits is None and "logits" not in outputs.keys()
    assert _close(outputs.loss.detach(), plain.loss.detach())
    assert grads.keys() == plain_grads.keys() and grads
    assert fake_cce
    assert hook.take_step() is None  # the kernel returns no per-token terms
    assert "forward" not in model.__dict__


def test_the_cce_hook_does_not_demand_a_frozen_head(fake_cce):
    """The kernel's backward reaches the weight, so a trainable head is not refused."""
    from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

    model = _tiny_causal_lm()
    model.get_output_embeddings().weight.requires_grad_(True)
    ChunkedHeadLoss(model, None, "cce").remove()  # no ValueError


def test_the_chunk_hook_still_demands_a_frozen_head():
    from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

    model = _tiny_causal_lm()
    model.get_output_embeddings().weight.requires_grad_(True)
    with pytest.raises(ValueError, match="the output head must be frozen"):
        ChunkedHeadLoss(model, 16)


def test_a_softcapping_model_is_refused_naming_use_cut_ce():
    from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

    model = _tiny_causal_lm()
    model.config.final_logit_softcapping = 30.0
    with pytest.raises(ValueError, match=r"training\.use_cut_ce cannot be used with"):
        ChunkedHeadLoss(model, None, "cce")


def test_an_unknown_impl_is_refused():
    from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

    with pytest.raises(ValueError, match="impl must be 'chunked' or 'cce'"):
        ChunkedHeadLoss(_tiny_causal_lm(), 16, "nope")


def test_the_chunked_impl_still_validates_its_chunk():
    from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss

    with pytest.raises(ValueError, match="chunk must be a positive integer"):
        ChunkedHeadLoss(_tiny_causal_lm(), None)


@pytest.mark.parametrize(
    ("training", "expected"),
    [
        ({}, None),
        ({"stream_layers": True}, None),
        ({"stream_layers": True, "stream_loss_chunk": 64}, "chunked"),
        ({"stream_layers": True, "use_cut_ce": True}, "cce"),
        # the resident CCE patch is not the hook's business
        ({"stream_layers": False, "use_cut_ce": True}, None),
        ({"stream_layers": False, "stream_loss_chunk": 64}, None),
    ],
)
def test_which_impl_a_config_selects(training, expected):
    from soup_cli.utils.stream_loss_chunk import stream_loss_impl

    assert stream_loss_impl(SimpleNamespace(**training)) == expected
    assert stream_loss_impl(None) is None


def test_a_model_wrapper_enters_through_its_forward_under_the_hook(fake_cce):
    """A DistributedDataParallel stand-in: the forward the wrapper owns must run, with the
    labels inside it, so the wrapper's gradient bookkeeping sees the loss."""
    torch = pytest.importorskip("torch")
    from soup_cli.utils.stream_loss_chunk import ChunkedHeadLoss
    from tests.test_stream_loss_chunk import _ids

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.seen = []

        def forward(self, *args, **kwargs):
            self.seen.append(sorted(kwargs))
            return self.module(*args, **kwargs)

    for impl, chunk in (("chunked", 8), ("cce", None)):
        model = _tiny_causal_lm()
        ids, labels = _ids()
        direct = model(input_ids=ids, labels=labels).loss.detach()
        hook = ChunkedHeadLoss(model, chunk, impl)
        wrapped = Wrapper(model)
        try:
            loss = wrapped(input_ids=ids, labels=labels).loss
            loss.backward()
        finally:
            hook.remove()
        assert wrapped.seen and "labels" in wrapped.seen[0]
        torch.testing.assert_close(loss.detach(), direct, rtol=1e-5, atol=1e-6)
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


# --- what the trainer does around it --------------------------------------------------


def test_the_rewind_skip_reason_names_the_kernel():
    from soup_cli.trainer.sft import rewind_skip_reason

    reason = rewind_skip_reason(
        task="sft", pretokenized=False, is_raft=False, multipack=False, vision=False,
        audio=False, cut_ce=True,
    )
    assert "Cut Cross-Entropy" in reason and "per-token losses" in reason


def test_the_setup_installs_the_cce_hook_and_charges_a_chunk(tmp_path, monkeypatch, fake_cce):
    """The hook is on the model before the probe, the trainer is the loss trainer, the rewind
    recorder steps aside, and the pre-flight charges one chunk of logits."""
    from soup_cli.utils import layer_stream
    from soup_cli.utils.stream_loss_chunk import CCE_CHARGE_CHUNK, HOOK_ATTR
    from tests.test_v07204 import _build_streamed_wrapper

    seen = {}
    estimate = layer_stream.estimate_stream_peak_vram

    def recording(**kwargs):
        seen["chunk"] = kwargs["loss_chunk"]
        return estimate(**kwargs)

    monkeypatch.setattr(layer_stream, "estimate_stream_peak_vram", recording)
    wrapper, _, _ = _build_streamed_wrapper(
        tmp_path, monkeypatch, task="sft", n_layers=2, use_cut_ce=True
    )
    try:
        hook = wrapper._stream_loss_chunk_hook
        assert hook is not None and hook.impl == "cce"
        assert getattr(wrapper.trainer, HOOK_ATTR) is hook
        assert seen["chunk"] == CCE_CHARGE_CHUNK
        assert getattr(wrapper, "_rewind_state", None) is None
        assert type(wrapper.trainer).__name__.startswith("ChunkedLoss")
    finally:
        wrapper._close_stream_runtime()
