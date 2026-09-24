"""#1174 part 2 — issue the head's load right after the embedding lookup.

With #1178 in, the embedding is reloaded at the tail of the previous step's
backward, so the head's forward-tail load into the shared slot is what is left of
the large-layer wait. This is the zero-VRAM alternative to a second slot (option (c)
on #1174), measured before it. `StreamPrefetcher.tail_prefetch` fires from `advance()` at
the last decoder layer, before that layer's `functional_call`, so the copy hides
behind one layer's compute. `head_prefetch_layer` moves that call earlier in the
forward: the embedding's bytes in the slot are dead once the lookup has run, and
the lookup precedes layer 0's `advance`, so layer 0 is the earliest point.

`head_prefetch_layer = None` is the original timing, so both timings run in one
process and every test here compares them. The refill is ordered after the
queued embedding lookup by the `wait_stream` in `LargeLayerBufferPool.load_async`;
the GPU test at the bottom pins that, with the embedding lookup as the queued read.

What these tests cannot show is whether the move is faster. That needs the
`wait()` bracket on a GPU, judged by the step's total wait on both the RAM and the
disk tier.
"""

from __future__ import annotations

import pytest

from tests.test_v07204 import (
    _batch_on,
    _build_streamed_wrapper,
    _mps_is_the_accelerator,
    _randomise_lora_b,
)

_NO_MPS = pytest.mark.skipif(
    _mps_is_the_accelerator(),
    reason="MPS is untested for layer streaming (CUDA + CPU only)",
)


# --------------------------------------------------------------------------
# the trigger, in isolation
# --------------------------------------------------------------------------
class _FakePool:
    def __init__(self, n_layers):
        self.owner = [None] * n_layers
        self.loads = []

    def slot_for(self, idx):
        return idx

    def load_async(self, idx, source, stream=None):
        self.loads.append(idx)
        self.owner[idx] = idx


def _walk(prefetcher, n_layers):
    """One forward, then the backward recompute walk, as the streamed layers call it."""
    prefetcher.prime()
    for idx in range(n_layers):
        prefetcher.advance(idx)
    for idx in reversed(range(n_layers)):
        prefetcher.advance(idx)


def _fired_at(head_layer, n_layers=4):
    from soup_cli.utils.layer_stream_runtime import StreamPrefetcher

    fired = []
    prefetcher = StreamPrefetcher(
        _FakePool(n_layers),
        None,
        n_layers,
        None,
        tail_prefetch=lambda: fired.append(("head", prefetcher.prev, prefetcher.direction)),
        backward_tail_prefetch=lambda: fired.append(
            ("embed", prefetcher.prev, prefetcher.direction)
        ),
        head_prefetch_layer=head_layer,
    )
    _walk(prefetcher, n_layers)
    return fired


class TestTheTrigger:
    def test_none_fires_the_head_at_the_last_layer_going_forward(self):
        fired = _fired_at(None)
        assert ("head", 3, 1) in fired
        assert [f for f in fired if f[0] == "head"] == [("head", 3, 1)]

    def test_layer_zero_fires_the_head_at_layer_zero_going_forward_only(self):
        heads = [f for f in _fired_at(0) if f[0] == "head"]
        assert heads == [("head", 0, 1)], heads

    def test_a_middle_layer_fires_once_and_never_in_the_backward_walk(self):
        heads = [f for f in _fired_at(2) if f[0] == "head"]
        assert heads == [("head", 2, 1)], heads

    def test_the_backward_tail_embed_prefetch_is_unchanged_by_the_head_timing(self):
        for head_layer in (None, 0, 2):
            embeds = [f for f in _fired_at(head_layer) if f[0] == "embed"]
            assert embeds == [("embed", 0, -1)], (head_layer, embeds)

    def test_a_second_forward_after_prime_fires_again(self):
        from soup_cli.utils.layer_stream_runtime import StreamPrefetcher

        fired = []
        prefetcher = StreamPrefetcher(
            _FakePool(4), None, 4, None,
            tail_prefetch=lambda: fired.append(prefetcher.prev),
            head_prefetch_layer=0,
        )  # fmt: skip
        _walk(prefetcher, 4)
        _walk(prefetcher, 4)
        assert fired == [0, 0]

    @pytest.mark.parametrize("bad", [-1, 4, 99])
    def test_an_out_of_range_layer_is_refused(self, bad):
        from soup_cli.utils.layer_stream_runtime import StreamPrefetcher

        with pytest.raises(ValueError, match="head_prefetch_layer"):
            StreamPrefetcher(_FakePool(4), None, 4, head_prefetch_layer=bad)


# --------------------------------------------------------------------------
# both timings, end to end on a streamed model
# --------------------------------------------------------------------------
def _recorder():
    from transformers import TrainerCallback

    class Recorder(TrainerCallback):
        def __init__(self):
            self.grads = []

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            self.grads.append(
                [
                    param.grad.detach().clone() if param.grad is not None else None
                    for param in model.parameters()
                    if param.requires_grad
                ]
            )

    return Recorder()


def _run_arm(tmp_path, monkeypatch, *, head_layer, task="sft", tie=False, steps=2, **training):
    """Train `steps` steps with the head prefetched at `head_layer`; `None` is the
    original last-layer timing. Returns the per-step gradients, the final adapter
    weights, the real large-layer loads, and the events the pool saw."""
    import torch

    torch.manual_seed(1234)
    wrapper, _, _ = _build_streamed_wrapper(
        tmp_path, monkeypatch, task=task, n_layers=4, tie=tie, **training
    )
    # PEFT starts `lora_B` at zero, which leaves most gradients zero and lets a
    # comparison pass without checking anything (KTO's, in bf16, were all zero).
    _randomise_lora_b(wrapper.model)
    runtime = wrapper._stream_runtime
    pool, prefetcher = runtime.large_pool, runtime.prefetcher
    events = []
    if pool is not None:
        prefetcher.head_prefetch_layer = head_layer
        original = pool.load_async

        def spy(key, source, stream=None):
            before = pool.loads
            out = original(key, source, stream)
            events.append((key, prefetcher.prev, pool.loads > before))
            return out

        pool.load_async = spy
    recorder = _recorder()
    wrapper.trainer.add_callback(recorder)
    wrapper.trainer.args.max_steps = steps
    try:
        wrapper.trainer.train()
        weights = {
            name: param.detach().clone()
            for name, param in wrapper.model.named_parameters()
            if param.requires_grad
        }
        loads = pool.loads if pool is not None else None
    finally:
        wrapper._close_stream_runtime()
    return recorder.grads, weights, loads, events


def _assert_identical(arm_a, arm_b):
    import torch

    grads_a, weights_a, _, _ = arm_a
    grads_b, weights_b, _, _ = arm_b
    assert grads_a and len(grads_a) == len(grads_b), (len(grads_a), len(grads_b))
    seen_nonzero = False
    for step, (step_a, step_b) in enumerate(zip(grads_a, grads_b)):
        assert len(step_a) == len(step_b)
        for index, (ga, gb) in enumerate(zip(step_a, step_b)):
            assert (ga is None) == (gb is None), (step, index)
            if ga is None:
                continue
            assert torch.equal(ga, gb), f"step {step} gradient {index} differs"
            seen_nonzero = seen_nonzero or bool(ga.abs().max() > 0)
    assert seen_nonzero, "vacuous: every gradient was zero"
    assert weights_a.keys() == weights_b.keys()
    for name in weights_a:
        assert torch.equal(weights_a[name], weights_b[name]), f"final weight {name} differs"


@_NO_MPS
class TestBothTimingsAreBitIdentical:
    """Moving the head's load must change WHEN the copy is issued, never what the
    step computes. The arms differ only in `head_prefetch_layer`."""

    @pytest.mark.parametrize(
        ("task", "training"),
        [
            ("sft", {}),
            ("sft", {"gradient_accumulation_steps": 2}),
            ("orpo", {}),
            ("simpo", {}),
            ("dpo", {}),
            ("kto", {"batch_size": 2}),
        ],
        ids=["sft", "sft-ga2", "orpo", "simpo", "dpo", "kto"],
    )
    def test_untied(self, tmp_path, monkeypatch, task, training):
        arm_a = _run_arm(tmp_path / "a", monkeypatch, head_layer=None, task=task, **training)
        arm_b = _run_arm(tmp_path / "b", monkeypatch, head_layer=0, task=task, **training)
        _assert_identical(arm_a, arm_b)

    def test_the_comparison_is_not_noise(self, tmp_path, monkeypatch):
        """Guard: the same timing twice must already be identical, or every other
        test in this class would be comparing two different runs."""
        first = _run_arm(tmp_path / "a", monkeypatch, head_layer=None)
        second = _run_arm(tmp_path / "b", monkeypatch, head_layer=None)
        _assert_identical(first, second)

    def test_tied_has_no_large_pool_so_the_switch_does_nothing(self, tmp_path, monkeypatch):
        arm_a = _run_arm(tmp_path / "a", monkeypatch, head_layer=None, tie=True)
        arm_b = _run_arm(tmp_path / "b", monkeypatch, head_layer=0, tie=True)
        assert arm_a[2] is None and arm_b[2] is None, "a tied model grew a large pool"
        _assert_identical(arm_a, arm_b)


@_NO_MPS
class TestTheInstalledDefault:
    """The arms above set `head_prefetch_layer` themselves, so nothing there covers
    what `install_streaming` wires when nobody touches it."""

    def test_an_untied_model_defaults_to_the_earliest_point(self, tmp_path, monkeypatch):
        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path, monkeypatch, task="sft", n_layers=4, tie=False
        )
        try:
            assert wrapper._stream_runtime.prefetcher.head_prefetch_layer == 0
        finally:
            wrapper._close_stream_runtime()

    def test_a_tied_model_has_nothing_to_move(self, tmp_path, monkeypatch):
        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path, monkeypatch, task="sft", n_layers=4, tie=True
        )
        try:
            assert wrapper._stream_runtime.large_pool is None
            assert wrapper._stream_runtime.prefetcher.head_prefetch_layer is None
        finally:
            wrapper._close_stream_runtime()


# --------------------------------------------------------------------------
# what the pool sees
# --------------------------------------------------------------------------
def _real(events, needle):
    return [prev for key, prev, real in events if real and needle in key]


@_NO_MPS
class TestTheLoadsMoveButDoNotMultiply:
    def test_the_head_load_comes_at_the_configured_layer(self, tmp_path, monkeypatch):
        _, _, _, events_last = _run_arm(tmp_path / "a", monkeypatch, head_layer=None)
        _, _, _, events_zero = _run_arm(tmp_path / "b", monkeypatch, head_layer=0)
        assert _real(events_last, "lm_head") == [3, 3], events_last
        assert _real(events_zero, "lm_head") == [0, 0], events_zero

    def test_the_real_load_sequence_and_count_are_the_same_in_both_timings(
        self, tmp_path, monkeypatch
    ):
        *_, loads_last, events_last = _run_arm(tmp_path / "a", monkeypatch, head_layer=None)
        *_, loads_zero, events_zero = _run_arm(tmp_path / "b", monkeypatch, head_layer=0)
        keys_last = [key for key, _, real in events_last if real]
        keys_zero = [key for key, _, real in events_zero if real]
        assert keys_last == keys_zero, (keys_last, keys_zero)
        assert loads_last == loads_zero == 5, (loads_last, loads_zero)

    @pytest.mark.parametrize("head_layer", [None, 0])
    def test_an_eval_forward_then_a_training_step(self, tmp_path, monkeypatch, head_layer):
        """A forward that never reaches a backward loads the embedding at its start
        and the head after the lookup; the training step after it must still find
        the slot in a state its own `_prime` can repair with one real load."""
        import torch

        wrapper, _, _ = _build_streamed_wrapper(
            tmp_path, monkeypatch, task="sft", n_layers=4, tie=False
        )
        runtime = wrapper._stream_runtime
        pool, prefetcher = runtime.large_pool, runtime.prefetcher
        prefetcher.head_prefetch_layer = head_layer
        batch = _batch_on(wrapper.model, next(iter(wrapper.trainer.get_train_dataloader())))
        wrapper.model.eval()
        with torch.no_grad():
            wrapper.model(input_ids=batch["input_ids"])
        after_eval = pool.loads
        wrapper.model.train()
        wrapper.trainer.args.max_steps = 1
        try:
            wrapper.trainer.train()
            after_step = pool.loads
        finally:
            wrapper._close_stream_runtime()
        # eval: the embed at `_prime`, then the head. The step: the embed again at
        # `_prime` (the slot holds the head), the head, and the backward-tail embed.
        assert after_eval == 2, after_eval
        assert after_step == 5, after_step


# --------------------------------------------------------------------------
# the GPU-side guard for the new refill point
# --------------------------------------------------------------------------
class _PinnedSource:
    def __init__(self, tensors):
        self.tensors = tensors

    def get(self, idx, key):
        return self.tensors[key]


@pytest.mark.gpu
def test_the_head_refill_waits_for_the_embedding_lookup_still_queued():
    """The refill at layer 0 runs while the embedding lookup can still be queued on
    the compute stream, reading the shared slot. What orders the refill after it is
    the `wait_stream` in `LargeLayerBufferPool.load_async`. Same construction as the
    backward-tail test in test_issue975_backward_tail_embed_prefetch.py, with the
    roles swapped: the queued read is the embedding's, the refill is the head's.
    Remove the wait and `seen` picks up the head's bytes.
    """
    import torch

    from soup_cli.utils.layer_stream_runtime import LargeLayerBufferPool

    shape = (1024, 1024)
    pool = LargeLayerBufferPool(
        {"embed_tokens": (shape, "float32"), "lm_head": (shape, "float32")},
        {"embed_tokens": 0, "lm_head": 1},
        device="cuda",
    )
    embed = torch.full(shape, 1.0).pin_memory()
    head = torch.full(shape, 2.0).pin_memory()
    source = _PinnedSource({"embed_tokens": embed, "lm_head": head})
    copy_stream = torch.cuda.Stream()

    pool.load_async("embed_tokens", source, copy_stream)
    slot = pool.wait("embed_tokens")
    torch.cuda.synchronize()

    torch.cuda._sleep(int(1e9))  # the compute stream lags, like a queued lookup
    seen = slot.clone()  # a read of the slot, queued behind the sleep
    pool.load_async("lm_head", source, copy_stream)  # the head's early refill
    torch.cuda.synchronize()

    assert torch.equal(seen, embed.cuda())
