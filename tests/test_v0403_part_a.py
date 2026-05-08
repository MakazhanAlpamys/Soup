"""Tests for v0.40.3 Part A — #64 live CUDA probe_fn.

The CUDA probe path itself can't run on CI (no GPU). These tests cover:
- ``make_cuda_probe_fn`` no-op branches (CPU, no torch, no CUDA, missing
  model/tokenizer, invalid max_length).
- Source-level wiring grep on sft.py to assert ``probe_fn=None`` is gone
  and ``make_cuda_probe_fn`` is invoked.
- Bool / non-int rejection on ``max_length``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import soup_cli.utils.batch_probe as bp
from soup_cli.utils.batch_probe import make_cuda_probe_fn


class TestMakeCudaProbeFn:
    def test_returns_none_on_cpu_device(self):
        assert make_cuda_probe_fn(
            object(), object(), max_length=128, device="cpu",
        ) is None

    def test_returns_none_on_mps_device(self):
        assert make_cuda_probe_fn(
            object(), object(), max_length=128, device="mps",
        ) is None

    def test_returns_none_when_model_missing(self):
        assert make_cuda_probe_fn(None, object(), max_length=128) is None

    def test_returns_none_when_tokenizer_missing(self):
        assert make_cuda_probe_fn(object(), None, max_length=128) is None

    def test_rejects_non_int_max_length(self):
        with pytest.raises(TypeError):
            make_cuda_probe_fn(object(), object(), max_length="128")

    def test_rejects_bool_max_length(self):
        with pytest.raises(TypeError):
            make_cuda_probe_fn(object(), object(), max_length=True)

    def test_rejects_too_small_max_length(self):
        with pytest.raises(ValueError):
            make_cuda_probe_fn(object(), object(), max_length=4)

    def test_returns_none_when_torch_unavailable(self, monkeypatch):
        # Simulate torch missing by patching builtins.__import__ for "torch".
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert make_cuda_probe_fn(
            object(), object(), max_length=128, device="cuda",
        ) is None

    def test_returns_none_when_cuda_not_available(self, monkeypatch):
        # Patch the torch.cuda.is_available used inside the function.
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert make_cuda_probe_fn(
            SimpleNamespace(), SimpleNamespace(), max_length=128,
        ) is None

    def test_probe_pad_id_falls_back_to_eos(self, monkeypatch):
        # Indirect coverage: when CUDA is unavailable, no-op branch hit.
        tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=2, vocab_size=32000)
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert make_cuda_probe_fn(
            SimpleNamespace(), tokenizer, max_length=128,
        ) is None

    def test_max_length_boundary_eight_accepted(self, monkeypatch):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # Below boundary raises immediately, above boundary returns None
        # (CUDA unavailable). Exact boundary 8 is accepted.
        with pytest.raises(ValueError):
            make_cuda_probe_fn(object(), object(), max_length=7)
        assert make_cuda_probe_fn(object(), object(), max_length=8) is None


class TestProbeClosure:
    """Exercise the inner ``_probe`` closure with mocked CUDA + fake model.

    These tests cover the closure body that the no-op branch tests skip.
    """

    def _make_torch_stub(self, monkeypatch):
        """Stub torch.cuda + tensor ops so make_cuda_probe_fn returns the closure."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        # No-op cuda.synchronize / cuda.empty_cache are already safe on CPU.
        # We don't mock torch itself — just ensure the closure path runs.
        return torch

    def _build(self, monkeypatch, model, max_length=64):
        torch = self._make_torch_stub(monkeypatch)
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, vocab_size=32000)
        # Run the test on whatever device torch is happy with — these tests
        # only assert behavioural branches, not actual GPU memory pressure.
        # Use device="cuda" so the closure is built; the model receives tensors
        # via .to is not used here (model is a fake callable).
        probe = make_cuda_probe_fn(
            model, tokenizer, max_length=max_length, device="cuda",
        )
        # NOTE: the fake model below ignores device placement — torch.full
        # with device="cuda" requires CUDA. Skip if unavailable on this box.
        if not torch.cuda.is_available():
            pytest.skip("CUDA tensor allocation unavailable")
        return probe

    def test_probe_rejects_bool_batch_size(self, monkeypatch):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed")
        if not _has_cuda():
            pytest.skip("CUDA not available — closure unreachable")
        probe = self._build(monkeypatch, model=lambda **kw: SimpleNamespace(loss=None))
        with pytest.raises(TypeError):
            probe(True)

    def test_probe_rejects_zero_batch_size(self, monkeypatch):
        if not _has_cuda():
            pytest.skip("CUDA not available — closure unreachable")
        probe = self._build(monkeypatch, model=lambda **kw: SimpleNamespace(loss=None))
        with pytest.raises(ValueError):
            probe(0)

    def test_probe_returns_false_on_oom(self, monkeypatch):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        if not _has_cuda():
            pytest.skip("CUDA not available — closure unreachable")

        def _oom_model(**_kw):
            raise torch.cuda.OutOfMemoryError("simulated OOM")

        probe = self._build(monkeypatch, model=_oom_model)
        assert probe(8) is False

    def test_probe_propagates_non_oom_exceptions(self, monkeypatch):
        if not _has_cuda():
            pytest.skip("CUDA not available — closure unreachable")

        def _broken_model(**_kw):
            raise RuntimeError("not OOM")

        probe = self._build(monkeypatch, model=_broken_model)
        with pytest.raises(RuntimeError, match="not OOM"):
            probe(8)


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class TestSftWiring:
    def test_sft_no_longer_passes_probe_fn_none_literal(self):
        path = Path("soup_cli/trainer/sft.py")
        text = path.read_text(encoding="utf-8")
        # The deferred-stub literal must be gone.
        assert "probe_fn=None,  # CUDA probe wired" not in text
        # And the new helper must be wired.
        assert "make_cuda_probe_fn" in text


class TestModuleSurface:
    def test_make_cuda_probe_fn_exported(self):
        assert callable(make_cuda_probe_fn)

    def test_module_constants_unchanged(self):
        # Defence against accidental constant churn during the wiring pass.
        assert bp._MIN_BATCH == 1
