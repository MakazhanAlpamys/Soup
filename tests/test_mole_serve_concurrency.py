"""Tests for concurrent request serialization and lock safety in LoadedMole (#2866)."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any

import pytest
import torch
from torch import nn

from soup_cli.utils.mole_routing import (
    LoadedMole,
    MoleGatingConfig,
    build_gating_kernel,
)


class _MoleOut:
    def __init__(
        self,
        logits: torch.Tensor,
        hidden_states: Any = None,
        past_key_values: Any = None,
    ):
        self.logits = logits
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values


def _make_fake_peft_model(
    hidden: int = 4, vocab: int = 6, supports_cache: bool = False
) -> nn.Module:
    """Mock PEFT model where active adapter is mutable state."""
    if supports_cache:
        class _FakePeftModelWithCache(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = hidden
                self.vocab = vocab
                self._adapter = "task_0"
                self._disabled = False

            @property
            def device(self):
                return torch.device("cpu")

            def disable_adapter(self):
                import contextlib

                @contextlib.contextmanager
                def _cm():
                    self._disabled = True
                    try:
                        yield
                    finally:
                        self._disabled = False

                return _cm()

            def set_adapter(self, name: str):
                self._adapter = name

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_hidden_states=False,
            ):
                b, t = input_ids.shape
                hs = torch.ones(b, t, self.hidden)
                idx = 0 if self._disabled else int(self._adapter.split("_")[1])
                logits = torch.zeros(b, t, self.vocab)
                logits[..., idx % self.vocab] = float(idx + 1)
                cache = ("cached_kv",) if use_cache else None
                return _MoleOut(
                    logits,
                    [hs] if output_hidden_states else None,
                    past_key_values=cache,
                )

        return _FakePeftModelWithCache()

    class _FakePeftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = hidden
            self.vocab = vocab
            self._adapter = "task_0"
            self._disabled = False

        @property
        def device(self):
            return torch.device("cpu")

        def disable_adapter(self):
            import contextlib

            @contextlib.contextmanager
            def _cm():
                self._disabled = True
                try:
                    yield
                finally:
                    self._disabled = False

            return _cm()

        def set_adapter(self, name: str):
            self._adapter = name

        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            b, t = input_ids.shape
            hs = torch.ones(b, t, self.hidden)
            idx = 0 if self._disabled else int(self._adapter.split("_")[1])
            logits = torch.zeros(b, t, self.vocab)
            logits[..., idx % self.vocab] = float(idx + 1)
            return _MoleOut(logits, [hs] if output_hidden_states else None)

    return _FakePeftModel()


def _build_loaded_mole(n_adapters: int = 2, supports_cache: bool = False) -> LoadedMole:
    model = _make_fake_peft_model(hidden=4, vocab=6, supports_cache=supports_cache)
    gate = build_gating_kernel(
        MoleGatingConfig(
            num_task_adapters=n_adapters,
            hidden_dim=4,
            temperature=1.0,
            top_k=n_adapters,
        )
    )
    gate.gate.weight.data.zero_()
    adapter_names = [f"task_{i}" for i in range(n_adapters)]
    return LoadedMole(model, None, gate, adapter_names)


def test_mole_has_lock_initialized():
    """LoadedMole.__init__ initializes an internal threading.Lock."""
    mole = _build_loaded_mole(n_adapters=2)
    assert hasattr(mole, "_lock")
    assert not mole._lock.locked()


def test_mole_concurrent_threads_match_sequential():
    """Concurrent decode requests against shared LoadedMole match sequential outputs."""
    mole = _build_loaded_mole(n_adapters=2)
    n_requests = 10
    inputs = [
        (torch.tensor([[i + 1, i + 2]]), torch.ones((1, 2)), 3 + (i % 3))
        for i in range(n_requests)
    ]

    sequential_outputs = [
        mole.generate(ids, attn, max_new_tokens=m, temperature=0.0).clone()
        for ids, attn, m in inputs
    ]

    concurrent_outputs = [None] * n_requests

    def run_one(idx: int):
        ids, attn, m = inputs[idx]
        concurrent_outputs[idx] = mole.generate(ids, attn, max_new_tokens=m, temperature=0.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as executor:
        futures = [executor.submit(run_one, i) for i in range(n_requests)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    for i in range(n_requests):
        assert torch.equal(concurrent_outputs[i], sequential_outputs[i]), (
            f"Concurrent request {i} differed from sequential baseline"
        )


def test_mole_lock_held_during_generate_loop():
    """The lock is held continuously across the entire generation loop."""
    mole = _build_loaded_mole(n_adapters=2)
    was_locked = []

    orig_blended = mole._blended_last_logits

    def spied_blended(input_ids, attention_mask):
        was_locked.append(mole._lock.locked())
        time.sleep(0.005)
        return orig_blended(input_ids, attention_mask)

    mole._blended_last_logits = spied_blended

    ids = torch.tensor([[1, 2]])
    attn = torch.ones_like(ids)
    mole.generate(ids, attn, max_new_tokens=4, temperature=0.0)

    assert len(was_locked) == 4
    assert all(was_locked)
    assert not mole._lock.locked()


def test_mole_exception_releases_lock():
    """An exception raised inside generate releases the lock for future requests."""
    mole = _build_loaded_mole(n_adapters=2)

    def failing_blended(input_ids, attention_mask):
        raise RuntimeError("simulated failure inside decode loop")

    mole._blended_last_logits = failing_blended

    ids = torch.tensor([[1, 2]])
    attn = torch.ones_like(ids)

    with pytest.raises(RuntimeError, match="simulated failure inside decode loop"):
        mole.generate(ids, attn, max_new_tokens=3)

    assert not mole._lock.locked()

    # Restore healthy blended logits and verify next request succeeds without deadlock
    mole._blended_last_logits = mole.__class__._blended_last_logits.__get__(mole)
    out = mole.generate(ids, attn, max_new_tokens=2, temperature=0.0)
    assert out.shape == (1, 4)
    assert not mole._lock.locked()


def _run_switch_sensitive_race(supports_cache: bool):
    """Run two concurrent requests with a switch-sensitive stand-in (#2866).

    Request 1 starts decoding task_1. In request 1's forward under task_1,
    it records the expected adapter ('task_1') and signals request 2.
    Request 2 calls generate, which immediately attempts to set_adapter('task_0').
    With the lock: request 2 blocks until request 1 finishes; the wait times out
    and request 1's observed adapter matches its expected adapter.
    Without the lock: request 2 interleaves and switches the active adapter to
    'task_0' while request 1 is still inside its forward, causing a mismatch.
    """
    mole = _build_loaded_mole(n_adapters=2, supports_cache=supports_cache)
    mismatches = []
    req1_in_adapter_forward = threading.Event()
    req2_called_set_adapter = threading.Event()

    orig_set_adapter = mole.model.set_adapter
    orig_forward = mole.model.forward

    def spied_set_adapter(name: str):
        orig_set_adapter(name)
        if threading.current_thread().name == "req2_worker":
            req2_called_set_adapter.set()

    def spied_forward(*args, **kwargs):
        if threading.current_thread().name == "req1_worker":
            if not getattr(mole.model, "_disabled", False) and mole.model._adapter == "task_1":
                expected = mole.model._adapter
                req1_in_adapter_forward.set()
                req2_called_set_adapter.wait(timeout=0.05)
                actual = mole.model._adapter
                if actual != expected:
                    mismatches.append((expected, actual))
        return orig_forward(*args, **kwargs)

    mole.model.set_adapter = spied_set_adapter
    mole.model.forward = spied_forward

    ids1 = torch.tensor([[1, 2]])
    attn1 = torch.ones_like(ids1)
    ids2 = torch.tensor([[5, 6]])
    attn2 = torch.ones_like(ids2)

    def worker1():
        mole.generate(ids1, attn1, max_new_tokens=2, temperature=0.0)

    def worker2():
        req1_in_adapter_forward.wait(timeout=0.5)
        mole.generate(ids2, attn2, max_new_tokens=2, temperature=0.0)

    t1 = threading.Thread(target=worker1, name="req1_worker")
    t2 = threading.Thread(target=worker2, name="req2_worker")
    t1.start()
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert not mole._lock.locked()
    assert mismatches == []


def test_mole_switch_sensitive_reforward_path():
    """Switch-sensitive stand-in test on full re-forward path."""
    _run_switch_sensitive_race(supports_cache=False)


def test_mole_switch_sensitive_kv_cached_path():
    """Switch-sensitive stand-in test on KV-cached path."""
    _run_switch_sensitive_race(supports_cache=True)


def test_mole_streamed_branch_generate_text_lock_behavior():
    """Streamed MoLE branch runs generate_text to completion; lock is released (#2866)."""
    mole = _build_loaded_mole(n_adapters=2)

    class _MockTokenizer:
        eos_token_id = 1

        def __call__(self, text, return_tensors=None, **kwargs):
            ids = torch.tensor([[1, 2, 3]])
            attn = torch.ones_like(ids)
            if return_tensors == "pt":
                return {"input_ids": ids, "attention_mask": attn}
            return {"input_ids": ids.tolist(), "attention_mask": attn.tolist()}

        def decode(self, token_ids, skip_special_tokens=True):
            return "sample response text"

    mole.tokenizer = _MockTokenizer()
    messages = [{"role": "user", "content": "hello"}]
    response, prompt_tokens, completion_tokens, finish_reason = mole.generate_text(
        messages, max_tokens=3, temperature=0.0
    )
    assert response == "sample response text"
    assert prompt_tokens == 3
    assert completion_tokens == 1
    assert finish_reason == "stop"
    assert not mole._lock.locked()


@pytest.mark.parametrize(
    ("max_tokens", "expected_tokens"),
    [
        (1, [3, 4, 1]),
        (2, [3, 4, 1, 1]),
        (4, [3, 4, 1, 1, 1, 1]),
    ],
)
def test_mole_positive_control_greedy_pinned_tokens(max_tokens: int, expected_tokens: list[int]):
    """Greedy generation matches exact pinned token sequences on main (#2866)."""
    mole = _build_loaded_mole(n_adapters=2)
    ids = torch.tensor([[3, 4]])
    attn = torch.ones_like(ids)
    out = mole.generate(ids, attn, max_new_tokens=max_tokens, temperature=0.0)
    assert out.tolist()[0] == expected_tokens
    assert not mole._lock.locked()


@pytest.mark.parametrize(
    ("max_tokens", "expected_tokens"),
    [
        (1, [3, 4, 4]),
        (2, [3, 4, 4, 0]),
        (4, [3, 4, 4, 0, 0, 2]),
    ],
)
def test_mole_positive_control_sampled_pinned_tokens(max_tokens: int, expected_tokens: list[int]):
    """Sampled generation under fixed seed matches exact pinned token sequences (#2866)."""
    torch.manual_seed(42)
    mole = _build_loaded_mole(n_adapters=2)
    ids = torch.tensor([[3, 4]])
    attn = torch.ones_like(ids)
    out = mole.generate(ids, attn, max_new_tokens=max_tokens, temperature=0.7)
    assert out.tolist()[0] == expected_tokens
    assert not mole._lock.locked()
