"""#975 — the embedding's `_prime`-time load pays a full head-sized H2D copy
with nothing to overlap it against, because `_prime()` (the forward-pre-hook)
fires right as the next step's forward starts and the embedding lookup is
almost the first thing that forward does.

`StreamPrefetcher` now also prefetches `embed_tokens` at the tail of the
BACKWARD pass, when the checkpoint recompute for decoder layer 0 confirms
this step's backward walk is done. Two things make refilling the shared slot
there safe. On the CPU side, `lm_head`'s own backward Node has already been
dispatched (it sits between the loss and every decoder layer, so the autograd
engine reaches it first), so the saved-tensor version check has already run.
On the GPU side, the head's backward GEMM can still be queued on the compute
stream, and what orders the refill after it is the `stream.wait_stream(...)`
in `LargeLayerBufferPool.load_async`; the GPU-only test at the bottom of this
file is the one that pins that guard. `_prime()` still issues the same load
unconditionally at the next step's start as a fallback for step 0 and for any
backward that never reaches layer 0; on the hot path that call now finds the
slot already holding what it asked for and is a same-owner no-op.

This only matters for an UNTIED checkpoint: a tied model streams one key for
both `embed_tokens` and `lm_head`, so `LargeLayerBufferPool.owner` never
changes across a whole run and the ping-pong this fix targets does not exist.
`_build_streamed_wrapper` in `test_v07204.py` never passes `tie` through to
`_tiny_llama_dir`, so every fixture built through it is tied — this file
builds its own untied fixture directly.
"""

from __future__ import annotations

import pytest

from tests.conftest import cuda_available
from tests.test_v07204 import (
    _TASK_ROWS,
    _mps_is_the_accelerator,
    _stream_cfg,
    _tiny_llama_dir,
    _wrapper_for,
    _write_tiny_tokenizer,
)


def _build_untied_wrapper(tmp_path, monkeypatch, task="sft", n_layers=2, **training):
    weights, resident, _ = _tiny_llama_dir(tmp_path, n_layers=n_layers, tie=False)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    cfg = _stream_cfg(weights, tmp_path / "out", task=task, **training)
    # The real accelerator when there is one, like `_build_streamed_wrapper`:
    # `TrainingArguments` picks CUDA on its own, so pinning the model to the CPU
    # there sends the batches to `cuda:0` and the embedding lookup fails on a
    # device mismatch. None of these tests compares floats, so nothing needs
    # the exact CPU arithmetic that pinning would buy.
    device = "cuda" if cuda_available() else "cpu"
    wrapper = _wrapper_for(task)(cfg, device=device)
    wrapper.setup({"train": _TASK_ROWS[task](8)})
    return wrapper, resident, weights


class TestBackwardTailEmbedPrefetch:
    """#975 — untied `embed_tokens` / `lm_head` share one `LargeLayerBufferPool`
    slot, and the fix moves the embedding's reload from the next step's
    `_prime()` to this step's backward tail."""

    # With no CUDA, `_build_untied_wrapper` falls back to the CPU, and on a box
    # where MPS is the accelerator that is a device mismatch no user would ever
    # hit, not something this fix is responsible for.
    pytestmark = pytest.mark.skipif(
        _mps_is_the_accelerator(),
        reason="MPS is untested for layer streaming (CUDA + CPU only)",
    )

    def test_untied_fixture_actually_shares_one_slot(self, tmp_path, monkeypatch):
        """Guard: if the fixture ever stopped being untied, the rest of this
        file would pass vacuously (a tied model never re-copies at all)."""
        wrapper, _, _ = _build_untied_wrapper(tmp_path, monkeypatch)
        large_pool = wrapper._stream_runtime.large_pool
        assert large_pool is not None
        assert len(large_pool.specs) > 1, "fixture is tied; this file needs an untied one"
        wrapper._close_stream_runtime()

    def test_one_step_completes_and_reloads_embed_before_prime_needs_it(
        self, tmp_path, monkeypatch
    ):
        """The whole point of the fix: after one full step (forward, backward,
        optimizer), the slot already holds `embed_tokens` again — reloaded
        during backward, not waiting for the next step's forward-pre-hook."""
        wrapper, _, _ = _build_untied_wrapper(tmp_path, monkeypatch)
        runtime = wrapper._stream_runtime
        large_pool = runtime.large_pool
        prefetcher = runtime.prefetcher
        embed_key = next(key for key in large_pool.specs if "embed_tokens" in key)

        wrapper.trainer.args.max_steps = 1
        wrapper.trainer.train()

        assert prefetcher.backward_tail_prefetched, (
            "layer 0's backward recompute never fired the backward-tail "
            "prefetch — did the checkpoint/backward wiring change?"
        )
        assert large_pool.owner == embed_key, (
            "backward tail fired but did not leave the slot holding embed_tokens"
        )
        wrapper._close_stream_runtime()

    def test_two_steps_complete_without_an_autograd_version_error(self, tmp_path, monkeypatch):
        """The regression this guards against would not be a wrong number —
        it would be `RuntimeError: ... modified by an inplace operation`, the
        same failure class #1049 hit for the untied projection. A second step
        is what actually exercises the boundary: the first step's backward
        tail must hand `_prime()` a slot nothing from the first step still
        needs."""
        wrapper, _, _ = _build_untied_wrapper(tmp_path, monkeypatch)
        wrapper.trainer.args.max_steps = 2
        wrapper.trainer.train()
        losses = [entry["loss"] for entry in wrapper.trainer.state.log_history if "loss" in entry]
        assert len(losses) == 2
        assert all(loss == loss for loss in losses)  # not NaN
        wrapper._close_stream_runtime()

    def test_prime_no_op_saves_one_reload_from_the_second_step_on(self, tmp_path, monkeypatch):
        """Counts real copies directly: from the second step on, `_prime()`'s
        load is a same-owner no-op, so a steady-state step costs two real
        large-layer loads (the head at the forward tail, the embedding at the
        backward tail), the same two `main` makes per step."""
        wrapper, _, _ = _build_untied_wrapper(tmp_path, monkeypatch)
        large_pool = wrapper._stream_runtime.large_pool
        wrapper.trainer.args.max_steps = 2
        wrapper.trainer.train()
        # `main` makes 4 real loads over these 2 steps: the cold embed load in
        # step 0's `_prime()` and the head at its forward tail, then the embed
        # in step 1's `_prime()` and the head at its forward tail. This change
        # makes 5. Steady state is 2 loads per step in both trees; the fifth is
        # the LAST backward's prefetch, which no later forward ever consumes,
        # so every `train()` now ends with exactly one extra load.
        assert large_pool.loads == 5, large_pool.loads
        wrapper._close_stream_runtime()


class _PinnedSource:
    def __init__(self, tensors):
        self.tensors = tensors

    def get(self, idx, key):
        return self.tensors[key]


@pytest.mark.gpu
def test_refill_waits_for_a_read_still_queued_on_the_compute_stream():
    """The GPU-side guard for the backward-tail refill, at pool level so it does
    not depend on how fast the host is.

    When the CPU reaches layer 0, the head's backward GEMM can still be queued on
    the compute stream, reading the shared slot. What orders the refill after it
    is `stream.wait_stream(torch.cuda.current_stream())` in
    `LargeLayerBufferPool.load_async`. Remove that wait and the copy stream
    overwrites the slot while the queued read has not run, so `seen` picks up the
    embedding's bytes instead of the head's.

    The compute stream is held back with `torch.cuda._sleep`, about a second, and
    the refill is issued microseconds after the read is queued, so the race is
    always open. A trainer-level version of this needs to assert the race really
    opened (for example `event.query()` on the sleep when the prefetch fires);
    otherwise a slow host makes it pass with the guard deleted.
    """
    import torch

    from soup_cli.utils.layer_stream_runtime import LargeLayerBufferPool

    shape = (1024, 1024)
    pool = LargeLayerBufferPool(
        {"lm_head": (shape, "float32"), "embed_tokens": (shape, "float32")},
        {"lm_head": 0, "embed_tokens": 1},
        device="cuda",
    )
    head = torch.full(shape, 1.0).pin_memory()
    embed = torch.full(shape, 2.0).pin_memory()
    source = _PinnedSource({"lm_head": head, "embed_tokens": embed})
    copy_stream = torch.cuda.Stream()

    pool.load_async("lm_head", source, copy_stream)
    slot = pool.wait("lm_head")
    torch.cuda.synchronize()

    torch.cuda._sleep(int(1e9))  # the compute stream lags, like a queued head backward
    seen = slot.clone()  # a read of the slot, queued behind the sleep
    pool.load_async("embed_tokens", source, copy_stream)  # the #975 refill
    torch.cuda.synchronize()

    assert torch.equal(seen, head.cuda())
