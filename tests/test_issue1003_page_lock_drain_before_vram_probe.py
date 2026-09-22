"""GPU-free regression test for #1003: a page-lock failure's stale CUDA error
must not survive into the VRAM probe that runs later in the same ``setup()``
call.

#1003 reported the probe (``measure_step_peak_bytes``, via
``_run_stream_vram_probe``) refusing a run with "does not fit" right after a
RAM-tier page-lock failure, even though the box had gigabytes free. Tracing
the call path at HEAD showed #901's ``recover_from_failed_page_lock`` already
runs inside ``_build_source`` before the probe starts, draining exactly the
stale error #1003 describes, so the defect it reports is unreachable today.

This is a positive control (the harness WOULD catch the failure if it were
still present) plus the real path through ``_build_source(pin=True)``, with
only ``RamSource`` mocked to fail its pinned allocation, proving the drain
that runs inside it leaves the probe clean.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from soup_cli.utils import layer_stream_runtime as lsr


def _get_torch_and_accelerator_error():
    """Import torch and return (torch, AcceleratorError) or skip if AcceleratorError is absent.

    torch.AcceleratorError was introduced in torch 2.8; on older versions (declared
    floor is 2.6) this skips with a version note, not a GPU note.
    """
    torch = pytest.importorskip("torch", reason="torch is not installed in this environment")
    acc = getattr(torch, "AcceleratorError", None)
    if acc is None:
        pytest.skip(
            f"torch {torch.__version__} predates torch.AcceleratorError "
            "(added in 2.8); nothing to classify here"
        )
    return torch, acc


class _FailsOncePinnedRamSource:
    """Stand-in for RamSource: pin=True fails the page-lock, pin=False succeeds.

    Mirrors what a box that cannot page-lock the base actually does: the
    only thing this test replaces in the real _build_source -> RamSource path.
    """

    def __init__(self, shard_dir, n_layers, spec, *, pin=True, shard_paths=None):
        if pin:
            raise RuntimeError("simulated page-lock failure")
        self.pinned = False


class TestPositiveControlUndrainedErrorReadsAsOom:
    """Without the drain, the stale error DOES read as a real OOM: #1003's report."""

    def test_undrained_stale_error_reads_as_oom(self, monkeypatch):
        _, accelerator_error = _get_torch_and_accelerator_error()
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def raise_stale_error(*args, **kwargs):
            raise accelerator_error("CUDA error: out of memory")

        monkeypatch.setattr(torch.cuda, "synchronize", raise_stale_error)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(lsr, "_zero_probe_grads", lambda *_: None)

        peak = lsr.measure_step_peak_bytes(object(), rows=1, seq_len=4, vocab_size=8)

        assert peak is not None
        assert peak.oom is True
        assert peak.failed is False


class TestRealPageLockFailureIsDrainedBeforeTheProbeRuns:
    """The real chain: _build_source(pin=True) -> recover_from_failed_page_lock
    -> drain_stale_cuda_error, then the probe, in that order (#901 before #349).

    drain_stale_cuda_error's default launch does torch.ones(...) then
    torch.cuda.synchronize(...): the FIRST kernel launch after a failed
    page-lock hits the stale error (#901's docstring), the one right after it
    always succeeds. torch.ones is stubbed out entirely (it never touches a
    real device here); only synchronize is made to fail once, on that first
    launch.
    """

    def test_real_page_lock_failure_is_drained_before_the_probe_runs(self, monkeypatch):
        _, accelerator_error = _get_torch_and_accelerator_error()
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch, "ones", lambda *a, **k: None)
        launches = {"n": 0}

        def flaky_synchronize(*args, **kwargs):
            launches["n"] += 1
            if launches["n"] == 1:
                raise accelerator_error("CUDA error: out of memory")

        monkeypatch.setattr(torch.cuda, "synchronize", flaky_synchronize)
        monkeypatch.setattr(lsr, "RamSource", _FailsOncePinnedRamSource)

        source, pinned = lsr._build_source("ignored-shard-dir", 1, {}, True, None, tier="ram")

        assert pinned is False, "the pinned attempt must have failed for the drain to matter"
        assert launches["n"] == 2, (
            "drain_stale_cuda_error should see exactly one failed launch (the "
            "stale error) and one clean one (proof the context recovered)"
        )

        # Same setup() call, right after: nothing here is mocked to fail.
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 123)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: 456)
        monkeypatch.setattr(torch, "randint", lambda *a, **k: object())
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(lsr, "_zero_probe_grads", lambda *_: None)

        peak = lsr.measure_step_peak_bytes(MagicMock(), rows=1, seq_len=4, vocab_size=8)

        assert peak is not None
        assert peak.oom is False
        assert peak.failed is False
