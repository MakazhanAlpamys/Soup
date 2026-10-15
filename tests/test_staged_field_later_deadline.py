"""#1814 — ``training.forgetting_threshold`` is staged like its siblings, with its own deadline.

Its description said "Staged, not enforced during training", but unlike
``forgetting_eval_steps`` / ``forgetting_benchmark`` / ``forgetting_stop`` it was
not in ``STAGED_FIELDS``, so it loaded silently at config load. It joins them,
first warned in 0.77, so it gets one minor of notice: v0.78 refuses it. The
existing entries keep their v0.77 date, and when 0.77 flips
``STAGED_FIELD_SEVERITY`` to ``"error"`` a field with a later deadline keeps
warning instead of being refused early.
"""

from __future__ import annotations

import re

import pytest

import soup_cli.config.loader as loader
from soup_cli import __version__
from soup_cli.commands.train import _nondefault_unwired_training_settings
from soup_cli.config.staged_fields import (
    STAGED_FIELD_LATER_DEADLINES,
    STAGED_FIELD_REJECTION_VERSION,
    STAGED_FIELDS,
    find_staged_config_fields,
    format_staged_fields,
)

_VALID = "base: org/m\ntask: sft\ndata:\n  train: ./t.jsonl\n"


def _plain(text: str) -> str:
    return " ".join(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).split())


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:2])


def test_forgetting_threshold_is_staged_with_the_v0_78_deadline():
    assert STAGED_FIELDS[("training", "forgetting_threshold")] == 0.10
    assert STAGED_FIELD_LATER_DEADLINES == {("training", "forgetting_threshold"): "0.78"}
    staged = find_staged_config_fields({"training": {"forgetting_threshold": 0.2}})
    assert [(item.path, item.deadline) for item in staged] == [
        ("training.forgetting_threshold", "0.78")
    ]
    message = format_staged_fields(staged, include_deadline=True)
    assert message == (
        "training.forgetting_threshold is accepted but read by nothing; v0.78 will refuse it"
    )


def test_the_existing_entries_keep_their_v0_77_date():
    staged = find_staged_config_fields({"training": {"forgetting_stop": True}})
    assert [item.deadline for item in staged] == [STAGED_FIELD_REJECTION_VERSION]
    assert STAGED_FIELD_REJECTION_VERSION == "0.77"


def test_the_default_value_stays_silent(capsys, monkeypatch):
    monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")
    loader.load_config_from_string(_VALID + "training:\n  forgetting_threshold: 0.10\n")
    assert "forgetting_threshold" not in _plain(capsys.readouterr().out)


def test_it_warns_at_load_today(capsys, monkeypatch):
    monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")
    loader.load_config_from_string(_VALID + "training:\n  forgetting_threshold: 0.2\n")
    out = _plain(capsys.readouterr().out)
    assert "training.forgetting_threshold is accepted but read by nothing" in out
    assert "v0.78 will refuse it" in out


def test_when_0_77_refuses_it_still_only_warns(capsys, monkeypatch):
    monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
    cfg = loader.load_config_from_string(_VALID + "training:\n  forgetting_threshold: 0.2\n")
    assert cfg.training.forgetting_threshold == 0.2
    assert "v0.78 will refuse it" in _plain(capsys.readouterr().out)


def test_under_error_a_due_field_is_refused_and_a_later_one_still_warned(capsys, monkeypatch):
    monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
    with pytest.raises(ValueError) as excinfo:
        loader.load_config_from_string(
            _VALID + "training:\n  forgetting_threshold: 0.2\n  forgetting_stop: true\n"
        )
    refusal = str(excinfo.value)
    assert "training.forgetting_stop" in refusal
    assert "refused as of v0.77" in refusal
    assert "forgetting_threshold" not in refusal
    assert "training.forgetting_threshold" in _plain(capsys.readouterr().out)


def test_soup_train_no_longer_lists_it_separately():
    # The loader reports it now, with its date; soup train's note would repeat it.
    cfg = loader.load_config_from_string(_VALID + "training:\n  forgetting_threshold: 0.2\n")
    assert _nondefault_unwired_training_settings(cfg.training) == []


def test_a_later_deadline_is_later_and_not_yet_reached():
    # Mirrors #808's deadline check for the per-entry dates: a later deadline must
    # be after the shared one, and the release that reaches it has to refuse it.
    for key, deadline in STAGED_FIELD_LATER_DEADLINES.items():
        assert key in STAGED_FIELDS, key
        assert _version(deadline) > _version(STAGED_FIELD_REJECTION_VERSION), (key, deadline)
        assert _version(__version__) < _version(deadline), (
            f"v{__version__} has reached {key}'s v{deadline} deadline: refuse it "
            "(move it to the shared deadline once STAGED_FIELD_REJECTION_VERSION is "
            f"{deadline}, or wire it)"
        )
