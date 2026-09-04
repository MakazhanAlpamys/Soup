"""#649 — the CUDA batch probe must refuse on a MEASURED peak, not only on a raise.

Under the WDDM driver (native Windows, WSL2) the CUDA allocator does not raise
``OutOfMemoryError`` when dedicated VRAM is exhausted: it spills into host
memory and the step completes. ``make_cuda_probe_fn`` used "the step did not
throw" as its entire fit criterion, so it approved a batch that did not fit and
``pick_batch_size`` wrote that answer to ``~/.soup/batch_cache.json``.

Every test here runs with a fake ``torch`` in ``sys.modules`` — the WDDM case
has to be testable without WDDM, and the whole file must pass with no GPU.
"""

from __future__ import annotations

import hashlib
import sys
import types

import pytest

GB = 1024**3


class _FakeOOMError(Exception):
    """Stand-in for ``torch.cuda.OutOfMemoryError``."""


class _FakeAcceleratorError(RuntimeError):
    """Stand-in for ``torch.AcceleratorError`` (torch >= 2.8 spelling)."""


class _Tensor:
    def clone(self):
        return _Tensor()


class _Loss:
    def backward(self):
        return None


class _Outputs:
    loss = _Loss()


class _Model:
    def __init__(self, step=None):
        self._step = step

    def zero_grad(self, set_to_none=True):
        return None

    def __call__(self, **kwargs):
        if self._step is not None:
            self._step()
        return _Outputs()


class _Tokenizer:
    pad_token_id = 0

    def __len__(self):
        return 32000


def _fake_torch(*, peak_allocated: int, allocated_before: int, free_before: int, total: int):
    """A ``torch`` whose CUDA counters report the given picture around one step."""
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    state = {"peak_reset": False}

    cuda.OutOfMemoryError = _FakeOOMError
    cuda.is_available = lambda: True
    cuda.synchronize = lambda *a, **k: None
    cuda.empty_cache = lambda: None
    cuda.memory_allocated = lambda *a, **k: allocated_before
    cuda.mem_get_info = lambda *a, **k: (free_before, total)

    def reset_peak_memory_stats(*a, **k):
        state["peak_reset"] = True

    def max_memory_allocated(*a, **k):
        # Before a reset the counter would carry a stale, possibly higher, peak
        # from an earlier probe; the gate must reset before it measures.
        assert state["peak_reset"], "probe read max_memory_allocated without resetting the peak"
        return peak_allocated

    cuda.reset_peak_memory_stats = reset_peak_memory_stats
    cuda.max_memory_allocated = max_memory_allocated
    cuda.max_memory_reserved = lambda *a, **k: int(peak_allocated * 1.3)

    class _Props:
        total_memory = total

    cuda.get_device_properties = lambda *a, **k: _Props()

    torch.cuda = cuda
    torch.long = "long"
    torch.full = lambda *a, **k: _Tensor()
    torch.ones_like = lambda *a, **k: _Tensor()
    torch.AcceleratorError = _FakeAcceleratorError
    return torch


def _probe_with(monkeypatch, torch, model=None):
    monkeypatch.setitem(sys.modules, "torch", torch)
    from soup_cli.utils.batch_probe import make_cuda_probe_fn

    probe = make_cuda_probe_fn(model or _Model(), _Tokenizer(), max_length=512, device="cuda")
    assert probe is not None
    return probe


class TestMeasuredPeakGate:
    def test_step_that_completes_but_peaks_over_budget_is_refused(self, monkeypatch):
        """The WDDM case: nothing raised, the peak says it did not fit."""
        torch = _fake_torch(
            peak_allocated=21 * GB,  # spilled: more than the card has
            allocated_before=6 * GB,
            free_before=9 * GB,
            total=16 * GB,
        )
        probe = _probe_with(monkeypatch, torch)
        assert probe(16) is False

    def test_step_under_budget_is_still_approved(self, monkeypatch):
        """Control: a fix that refuses everything is not a fix."""
        torch = _fake_torch(
            peak_allocated=12 * GB,
            allocated_before=6 * GB,
            free_before=9 * GB,
            total=16 * GB,
        )
        probe = _probe_with(monkeypatch, torch)
        assert probe(8) is True

    def test_budget_is_what_this_process_can_reach_not_the_whole_card(self, monkeypatch):
        """Another process holds 5 GB: peak 13 GB on a 16 GB card does NOT fit."""
        torch = _fake_torch(
            peak_allocated=13 * GB,
            allocated_before=6 * GB,
            free_before=5 * GB,  # 6 (ours) + 5 (free) = 11 GB reachable
            total=16 * GB,
        )
        probe = _probe_with(monkeypatch, torch)
        assert probe(8) is False

    def test_peak_exactly_at_budget_fits(self, monkeypatch):
        torch = _fake_torch(
            peak_allocated=15 * GB,
            allocated_before=6 * GB,
            free_before=9 * GB,
            total=16 * GB,
        )
        probe = _probe_with(monkeypatch, torch)
        assert probe(8) is True

    def test_gate_uses_allocated_not_reserved(self, monkeypatch):
        """Reserved runs 1.08-1.41x allocated (layer_stream.decide_measured_fit);
        the fake reports reserved at 1.3x so a gate on reserved would refuse
        this fitting step."""
        torch = _fake_torch(
            peak_allocated=14 * GB,  # reserved fake = 18.2 GB > 15 GB budget
            allocated_before=6 * GB,
            free_before=9 * GB,
            total=16 * GB,
        )
        probe = _probe_with(monkeypatch, torch)
        assert probe(8) is True


class TestSynchronizeOOMSpelling:
    def test_accelerator_error_out_of_memory_counts_as_oom(self, monkeypatch):
        """WDDM's eventual failure is ``AcceleratorError: CUDA error: out of
        memory`` from ``cuStreamSynchronize``, not ``OutOfMemoryError``."""
        torch = _fake_torch(
            peak_allocated=1 * GB, allocated_before=1 * GB, free_before=9 * GB, total=16 * GB
        )

        def step():
            raise _FakeAcceleratorError("CUDA error: out of memory")

        probe = _probe_with(monkeypatch, torch, model=_Model(step))
        assert probe(8) is False

    def test_runtime_error_out_of_memory_counts_as_oom(self, monkeypatch):
        """Older torch spells the same failure as a plain RuntimeError."""
        torch = _fake_torch(
            peak_allocated=1 * GB, allocated_before=1 * GB, free_before=9 * GB, total=16 * GB
        )

        def step():
            raise RuntimeError("CUDA out of memory. Tried to allocate 560.00 MiB")

        probe = _probe_with(monkeypatch, torch, model=_Model(step))
        assert probe(8) is False

    def test_other_accelerator_error_propagates(self, monkeypatch):
        """An illegal memory access is misconfiguration, not a fit answer."""
        torch = _fake_torch(
            peak_allocated=1 * GB, allocated_before=1 * GB, free_before=9 * GB, total=16 * GB
        )

        def step():
            raise _FakeAcceleratorError("CUDA error: an illegal memory access was encountered")

        probe = _probe_with(monkeypatch, torch, model=_Model(step))
        with pytest.raises(_FakeAcceleratorError, match="illegal memory access"):
            probe(8)


class TestCacheKeyInvalidatesOldProbe:
    def test_key_differs_from_the_pre_649_key_for_the_same_tuple(self):
        """Entries written by the exception-only probe must not survive: the
        pre-#649 key was sha256 of the bare tuple, and a poisoned
        ``batch_cache.json`` has no other way to be noticed."""
        from soup_cli.utils.batch_probe import make_cache_key

        args = ("Qwen/Qwen2.5-1.5B-Instruct", 512, "none", 16, "NVIDIA GeForce RTX 5080", 15)
        old_raw = "|".join(str(a) for a in args)
        old_key = hashlib.sha256(old_raw.encode("utf-8")).hexdigest()[:32]
        assert make_cache_key(*args) != old_key

    def test_key_is_still_stable_and_still_32_hex(self):
        from soup_cli.utils.batch_probe import make_cache_key

        a = make_cache_key("m", 2048, "4bit", 64, "gpu", 80)
        b = make_cache_key("m", 2048, "4bit", 64, "gpu", 80)
        assert a == b
        assert len(a) == 32
        int(a, 16)
