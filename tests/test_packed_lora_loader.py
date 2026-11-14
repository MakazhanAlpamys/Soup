"""``soup_cli.kernels.packed_lora.loader``: the one door to the Triton half (CPU)."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types

import pytest

from soup_cli.kernels.packed_lora import loader


def test_triton_available_reads_the_import_system_without_importing(monkeypatch):
    seen = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: seen.append(name) or object())
    assert loader.triton_available() is True
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert loader.triton_available() is False
    assert seen == ["triton"]


def test_load_says_why_when_triton_is_missing(monkeypatch):
    monkeypatch.setattr(loader, "triton_available", lambda: False)
    with pytest.raises(loader.KernelsUnavailableError) as excinfo:
        loader.load()
    assert str(excinfo.value) == (
        "the triton package is not installed (it comes with the Linux CUDA build of torch)")


def test_load_names_the_dependency_that_failed_to_import(monkeypatch):
    monkeypatch.setattr(loader, "triton_available", lambda: True)

    def boom(name):
        raise ModuleNotFoundError("No module named 'bitsandbytes'", name="bitsandbytes")

    monkeypatch.setattr(importlib, "import_module", boom)
    with pytest.raises(loader.KernelsUnavailableError) as excinfo:
        loader.load()
    assert str(excinfo.value).startswith("bitsandbytes could not be imported: ")


def test_load_returns_the_runtime_module(monkeypatch):
    fake = types.SimpleNamespace(PackedRun=object)
    monkeypatch.setattr(loader, "triton_available", lambda: True)
    monkeypatch.setattr(importlib, "import_module", lambda name: fake if name.endswith(
        "packed_lora.runtime") else sys.modules[name])
    assert loader.load() is fake
