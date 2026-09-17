"""GPU-free tests for AcceleratorError-as-OOM classification (issue #1002).

#989 classified an OOM arriving as AcceleratorError as a result (oom=True) rather
than an instrument failure (failed=True). However, that coverage was gated behind
@requires_cuda in tests/test_issue901_failed_page_lock_recovery.py, causing it to
skip across all CI matrix cells without a GPU runner.

The classifier (_is_out_of_memory, delegating to batch_probe._is_cuda_oom) and
the outcome mapping (_classify_probe_exception) are pure functions of the exception
type and message. These tests exercise the classifier, the three-outcome StepPeak
mapping, and the anti-forking predicate delegation directly without requiring a GPU.
"""

from __future__ import annotations

import pytest

from soup_cli.utils.layer_stream_runtime import (
    StepPeak,
    _classify_probe_exception,
    _is_out_of_memory,
)


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


class TestIsOutOfMemoryClassification:
    """Direct GPU-free classification tests for _is_out_of_memory."""

    def test_torch_cuda_out_of_memory_error_is_oom(self):
        torch = pytest.importorskip("torch", reason="torch is not installed in this environment")
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is None:
            pytest.skip("torch.cuda.OutOfMemoryError is not available in this torch")
        exc = oom_cls("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert _is_out_of_memory(exc) is True

    def test_accelerator_error_out_of_memory_is_oom(self):
        _, accelerator_error = _get_torch_and_accelerator_error()
        exc = accelerator_error("CUDA error: out of memory")
        assert _is_out_of_memory(exc) is True

    def test_accelerator_error_illegal_access_is_not_oom(self):
        _, accelerator_error = _get_torch_and_accelerator_error()
        exc = accelerator_error("CUDA error: an illegal memory access was encountered")
        assert _is_out_of_memory(exc) is False

    def test_bare_runtime_error_out_of_memory_is_oom(self):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; ...)")
        assert _is_out_of_memory(exc) is True

    def test_bare_runtime_error_non_oom_is_not_oom(self):
        assert _is_out_of_memory(RuntimeError("device-side assert triggered")) is False
        assert _is_out_of_memory(RuntimeError("an illegal memory access was encountered")) is False

    def test_non_runtime_error_is_not_oom(self):
        """Exceptions outside RuntimeError/OutOfMemoryError must never classify as OOM."""
        assert _is_out_of_memory(ValueError("CUDA out of memory")) is False
        assert _is_out_of_memory(KeyError("out of memory")) is False
        assert _is_out_of_memory(TypeError("out of memory")) is False
        assert _is_out_of_memory(Exception("out of memory")) is False


class TestClassifyProbeExceptionOutcomes:
    """GPU-free tests for the three-outcome mapping in _classify_probe_exception."""

    def test_oom_accelerator_error_yields_oom_result(self):
        _, accelerator_error = _get_torch_and_accelerator_error()
        exc = accelerator_error("CUDA error: out of memory")
        peak = _classify_probe_exception(exc, rows=2, seq_len=16, seconds=0.42)

        assert isinstance(peak, StepPeak)
        assert peak.oom is True
        assert peak.failed is False
        assert peak.error is None
        assert peak.peak_bytes == 0
        assert peak.reserved_bytes == 0
        assert peak.rows == 2
        assert peak.seq_len == 16
        assert peak.seconds == 0.42

    def test_torch_cuda_oom_error_yields_oom_result(self):
        torch = pytest.importorskip("torch", reason="torch is not installed in this environment")
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is None:
            pytest.skip("torch.cuda.OutOfMemoryError is not available in this torch")
        exc = oom_cls("CUDA out of memory")
        peak = _classify_probe_exception(exc, rows=1, seq_len=8, seconds=0.1)

        assert peak.oom is True
        assert peak.failed is False
        assert peak.error is None
        assert peak.peak_bytes == 0
        assert peak.reserved_bytes == 0
        assert peak.rows == 1
        assert peak.seq_len == 8

    def test_non_oom_accelerator_error_yields_instrument_failure(self):
        _, accelerator_error = _get_torch_and_accelerator_error()
        exc = accelerator_error("CUDA error: an illegal memory access was encountered")
        peak = _classify_probe_exception(exc, rows=2, seq_len=16, seconds=0.25)

        assert isinstance(peak, StepPeak)
        assert peak.oom is False
        assert peak.failed is True
        assert peak.error == "AcceleratorError"
        assert peak.peak_bytes == 0
        assert peak.reserved_bytes == 0
        assert peak.rows == 2
        assert peak.seq_len == 16

    def test_generic_runtime_error_yields_instrument_failure(self):
        exc = RuntimeError("CUDA error: device-side assert triggered")
        peak = _classify_probe_exception(exc, rows=1, seq_len=4, seconds=0.05)

        assert peak.oom is False
        assert peak.failed is True
        assert peak.error == "RuntimeError"
        assert peak.peak_bytes == 0
        assert peak.reserved_bytes == 0


class TestPredicateIdentityAndAntiForking:
    """Pin that _is_out_of_memory delegates to batch_probe._is_cuda_oom."""

    def test_is_out_of_memory_delegates_to_batch_probe_predicate(self, monkeypatch):
        """Pin that _is_out_of_memory calls batch_probe._is_cuda_oom directly.

        A third spelling cannot appear: _is_out_of_memory must delegate to
        batch_probe._is_cuda_oom rather than forking a divergent copy (precedent:
        test_tracker_and_execution_share_one_primitive in
        test_issue424_windows_liveness.py).
        """
        torch = pytest.importorskip("torch", reason="torch is not installed in this environment")

        calls: list[tuple[BaseException, object]] = []
        sentinel = object()

        def spy_is_cuda_oom(exc: BaseException, torch_arg: object) -> object:
            calls.append((exc, torch_arg))
            return sentinel

        monkeypatch.setattr("soup_cli.utils.batch_probe._is_cuda_oom", spy_is_cuda_oom)

        exc = RuntimeError("test exception")
        result = _is_out_of_memory(exc)

        assert result is sentinel, "_is_out_of_memory must return the verdict from _is_cuda_oom"
        assert len(calls) == 1, "_is_out_of_memory must call batch_probe._is_cuda_oom exactly once"
        assert calls[0][0] is exc
        assert calls[0][1] is torch

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("CUDA out of memory. Tried to allocate 1024 bytes"),
            RuntimeError("CUDA error: out of memory"),
            RuntimeError("CUDA error: an illegal memory access was encountered"),
            RuntimeError("device-side assert triggered"),
            ValueError("CUDA out of memory"),
            TypeError("invalid argument"),
        ],
    )
    def test_is_out_of_memory_agrees_with_batch_probe_predicate(self, exc):
        """Verify output identity between _is_out_of_memory and batch_probe._is_cuda_oom."""
        torch = pytest.importorskip("torch", reason="torch is not installed in this environment")
        from soup_cli.utils.batch_probe import _is_cuda_oom

        assert _is_out_of_memory(exc) == _is_cuda_oom(exc, torch)
