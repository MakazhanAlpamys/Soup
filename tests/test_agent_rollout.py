"""agent_rollout: the ``ruler`` backend must point at a package that exists (#2046)."""

from __future__ import annotations

import pytest

from soup_cli.utils import agent_rollout


def test_ruler_missing_package_message_names_an_installable_package(monkeypatch):
    monkeypatch.setattr(agent_rollout, "_spec_exists", lambda _name: False)
    with pytest.raises(ImportError) as excinfo:
        agent_rollout.launch_rollout("ruler", prompts=["x"])
    message = str(excinfo.value)
    assert "ruler-eval" not in message
    assert "pip install openpipe-art" in message


def test_ruler_probe_is_not_satisfied_by_an_unrelated_ruler_module(monkeypatch):
    # PyPI's `ruler` (a grammar library) installs a top-level `ruler` module.
    monkeypatch.setattr(agent_rollout, "_spec_exists", lambda name: name == "ruler")
    with pytest.raises(ImportError, match="openpipe-art"):
        agent_rollout.launch_rollout("ruler", prompts=["x"])


def test_ruler_with_openpipe_art_installed_reaches_the_beta_gate(monkeypatch):
    monkeypatch.setattr(agent_rollout, "_spec_exists", lambda name: name == "art")
    with pytest.raises(RuntimeError) as excinfo:
        agent_rollout.launch_rollout("ruler", prompts=["x"])
    message = str(excinfo.value)
    assert "openpipe-art" in message
    assert "not yet validated" in message
