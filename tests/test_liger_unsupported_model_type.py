"""#1811 — ``use_liger: true`` on a model type Liger has no kernels for reports False.

liger's ``_apply_liger_kernel(model_type)`` logs and returns, without raising,
when the type is not in its ``MODEL_TYPE_TO_APPLY_LIGER_FN`` table, so
``apply_liger_kernel`` returned True for ``gpt_neox``, ``phi`` or ``gpt2``: SFT
printed "Liger Kernel enabled" and set ``use_liger_kernel=True`` with no kernel
applied. The type is now looked up in liger's own table first. A model type that
is known but not in the table is unsupported, full stop: the name-based fallback
is for a model whose config cannot be read, and on ``microsoft/phi-2`` it would
"succeed" by patching Phi-3's classes.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from soup_cli.utils import liger as liger_mod


def _install_liger(monkeypatch, calls: list, *, with_table: bool = True) -> None:
    """A stand-in liger-kernel whose ``_apply_liger_kernel`` has upstream's body."""
    table = {"llama": lambda **kw: calls.append("apply_liger_kernel_to_llama"),
             "phi3": lambda **kw: calls.append("apply_liger_kernel_to_phi3")}

    def _apply_liger_kernel(model_type, **kwargs):
        calls.append(("_apply_liger_kernel", model_type))
        if not model_type or model_type not in table:
            return  # upstream: logger.info(...) and return, no exception
        table[model_type]()

    transformers_mod = types.ModuleType("liger_kernel.transformers")
    transformers_mod._apply_liger_kernel = _apply_liger_kernel
    transformers_mod.apply_liger_kernel_to_phi3 = table["phi3"]
    transformers_mod.apply_liger_kernel_to_llama = table["llama"]
    parent = types.ModuleType("liger_kernel")
    parent.transformers = transformers_mod
    monkeypatch.setitem(sys.modules, "liger_kernel", parent)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", transformers_mod)
    if with_table:
        monkey_patch = types.ModuleType("liger_kernel.transformers.monkey_patch")
        monkey_patch.MODEL_TYPE_TO_APPLY_LIGER_FN = table
        monkey_patch._apply_liger_kernel = _apply_liger_kernel
        transformers_mod.monkey_patch = monkey_patch
        monkeypatch.setitem(sys.modules, "liger_kernel.transformers.monkey_patch", monkey_patch)
    monkeypatch.setattr(liger_mod, "check_liger_available", lambda: True)


def _model_dir(tmp_path: Path, name: str, model_type: str) -> str:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}),
                                           encoding="utf-8")
    return str(directory)


@pytest.mark.parametrize("model_type", ["gpt_neox", "phi", "gpt2"])
def test_a_model_type_liger_has_no_kernels_for_reports_false(tmp_path, monkeypatch, model_type):
    calls: list = []
    _install_liger(monkeypatch, calls)
    path = _model_dir(tmp_path, "checkpoint", model_type)
    assert liger_mod.apply_liger_kernel(path) is False
    assert calls == [], calls  # nothing applied, nothing claimed


def test_a_phi_model_is_not_patched_as_phi3_by_its_name(tmp_path, monkeypatch):
    # model_type "phi" (phi-2) is known and not in liger's table; the name-based
    # fallback would match "phi" and patch Phi-3's classes, which phi-2 never uses.
    calls: list = []
    _install_liger(monkeypatch, calls)
    path = _model_dir(tmp_path, "phi-2", "phi")
    assert liger_mod.apply_liger_kernel(path) is False
    assert "apply_liger_kernel_to_phi3" not in calls


def test_control_llama_is_still_patched(tmp_path, monkeypatch):
    calls: list = []
    _install_liger(monkeypatch, calls)
    path = _model_dir(tmp_path, "checkpoint", "llama")
    assert liger_mod.apply_liger_kernel(path) is True
    assert calls == [("_apply_liger_kernel", "llama"), "apply_liger_kernel_to_llama"]


def test_control_without_liger_s_table_the_previous_behaviour_stands(tmp_path, monkeypatch):
    # An older or reorganised liger with no importable table: the type is handed to
    # _apply_liger_kernel as before (the pre-#1811 path, pinned in test_v07300.py).
    calls: list = []
    _install_liger(monkeypatch, calls, with_table=False)
    path = _model_dir(tmp_path, "checkpoint", "llama")
    assert liger_mod.apply_liger_kernel(path) is True
    assert calls[0] == ("_apply_liger_kernel", "llama")


def test_control_an_unreadable_config_still_uses_the_name(tmp_path, monkeypatch):
    # No config.json: the model type is unknown, so the name-based match stays the
    # fallback, as before.
    calls: list = []
    _install_liger(monkeypatch, calls)
    monkeypatch.setattr(liger_mod, "_detect_model_type", lambda name: "")
    assert liger_mod.apply_liger_kernel("org/Llama-3.1-8B") is True
    assert calls == ["apply_liger_kernel_to_llama"]
