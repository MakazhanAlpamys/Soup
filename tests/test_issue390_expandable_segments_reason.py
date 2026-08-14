"""The streaming panel blamed Windows on a Linux box.

The first real free-tier Colab run of `notebooks/proof-4gb.ipynb` — a T4 on
Ubuntu — printed:

    expandable_segments allocator hint is unavailable on this platform
    (silently ignored on Windows) — not enabled

That sentence is false there twice over: the platform is not Windows, and the
hint is not unavailable on it. The real cause is that the CUDA allocator reads
``PYTORCH_CUDA_ALLOC_CONF`` once, when the context is created, so anything that
touches CUDA before Soup does makes the setting a no-op — which is exactly what
a notebook does when it checks the GPU in an earlier cell.

The message was a single hardcoded string for every False, so it could not be
right anywhere but Windows. ``expandable_segments_status()`` now returns the
reason alongside the verdict. This is the most-read line the feature has after
the panel itself; stating the wrong cause is worse than stating none.
"""

import sys

import pytest


class _FakeCuda:
    def __init__(self, available=True, initialized=True):
        self._available = available
        self._initialized = initialized

    def is_available(self):
        return self._available

    def is_initialized(self):
        return self._initialized


@pytest.fixture()
def fake_cuda(monkeypatch):
    import torch

    def apply(**kwargs):
        fake = _FakeCuda(**kwargs)
        monkeypatch.setattr(torch, "cuda", fake)
        return fake

    return apply


class TestExpandableSegmentsStatus:
    def test_windows_says_windows(self, monkeypatch):
        from soup_cli.utils.layer_stream_runtime import expandable_segments_status

        monkeypatch.setattr(sys, "platform", "win32")
        enabled, why = expandable_segments_status()
        assert enabled is False
        assert "Windows" in why

    def test_linux_with_cuda_already_up_names_the_real_cause(
        self, fake_cuda, monkeypatch
    ):
        """The Colab case. Must NOT mention Windows, and must name the cause."""
        from soup_cli.utils.layer_stream_runtime import expandable_segments_status

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        fake_cuda(available=True, initialized=True)

        enabled, why = expandable_segments_status()
        assert enabled is False
        assert "Windows" not in why, f"blamed Windows on Linux: {why!r}"
        assert "already initialised" in why
        assert "PYTORCH_CUDA_ALLOC_CONF" in why

    def test_linux_before_cuda_init_enables_it(self, fake_cuda, monkeypatch):
        from soup_cli.utils.layer_stream_runtime import expandable_segments_status

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        fake_cuda(available=True, initialized=False)

        enabled, why = expandable_segments_status()
        assert enabled is True
        assert why == ""

    def test_already_set_by_the_operator_counts_as_enabled(
        self, fake_cuda, monkeypatch
    ):
        """CUDA is up, but the env var was exported before the process started —
        which is the one way to actually get this on a notebook."""
        from soup_cli.utils.layer_stream_runtime import expandable_segments_status

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        fake_cuda(available=True, initialized=True)

        assert expandable_segments_status() == (True, "")

    def test_no_cuda_says_no_cuda(self, fake_cuda, monkeypatch):
        from soup_cli.utils.layer_stream_runtime import expandable_segments_status

        monkeypatch.setattr(sys, "platform", "linux")
        fake_cuda(available=False)
        enabled, why = expandable_segments_status()
        assert enabled is False
        assert "CUDA" in why

    def test_probe_still_returns_a_bool(self):
        """The old public name is kept — callers outside the panel use it."""
        from soup_cli.utils.layer_stream_runtime import probe_expandable_segments

        assert isinstance(probe_expandable_segments(), bool)


class TestThePanelPrintsTheReason:
    def test_setup_no_longer_hardcodes_the_windows_sentence(self):
        from pathlib import Path

        import soup_cli.trainer.stream_setup as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "silently ignored on Windows) — not enabled" not in src
        assert "expandable_segments_status" in src
