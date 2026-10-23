"""#1355 — the streamed builders build on the device `TrainingArguments` picks by itself.

Pinned without a Mac or a card: otherwise the MPS branch of the resolver only ever
runs on the macOS cells, and the CUDA branch only on a GPU box.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests import conftest


@pytest.fixture()
def fake_accelerators(monkeypatch):
    """Fake `torch.cuda` and `torch.backends.mps`; both probes are cached, so clear them."""
    import torch

    def apply(*, cuda: bool, mps: bool | None) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        backend = None if mps is None else SimpleNamespace(is_available=lambda: mps)
        monkeypatch.setattr(torch.backends, "mps", backend, raising=False)
        conftest.cuda_available.cache_clear()
        conftest.mps_is_the_accelerator.cache_clear()

    yield apply
    # Runs before monkeypatch restores torch, so the next caller re-probes the real one.
    conftest.cuda_available.cache_clear()
    conftest.mps_is_the_accelerator.cache_clear()


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
        (False, None, "cpu"),  # a torch build with no torch.backends.mps at all
    ],
)
def test_the_builders_device_is_cuda_then_mps_then_cpu(fake_accelerators, cuda, mps, expected):
    fake_accelerators(cuda=cuda, mps=mps)
    assert conftest.accelerator_device() == expected
    assert conftest.mps_is_the_accelerator() is (expected == "mps")
