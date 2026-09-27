"""#761 — three documented training settings that nothing reads get the #808 rule.

``training.lr_groups`` and ``training.citation_recall_threshold`` were validated
and read by nothing, and setting them printed nothing. ``early_stop_patience``
got ``soup train``'s "set but not enforced" note, with no refusal date. Ruled on
#761 (2026-09-24): warn in v0.76, refuse in v0.77, with the deadline in #1106's
``STAGED_FIELD_REJECTION_VERSION``.
"""

from __future__ import annotations

import re

import pytest

from soup_cli.config import loader
from soup_cli.config.staged_fields import STAGED_FIELD_REJECTION_VERSION

_BASE = "base: org/m\ntask: sft\ndata:\n  train: ./x.jsonl\n"
# Each field, set as a user would write it, in a config that otherwise loads.
_SET = {
    "training.lr_groups": _BASE + "training:\n  lr_groups:\n    - {pattern: lm_head, lr: 1.0e-5}\n",
    "training.early_stop_patience": _BASE + "training:\n  early_stop_patience: 5\n",
    "training.citation_recall_threshold": (
        "base: org/m\ntask: sft\ndata:\n  train: ./x.jsonl\n  format: raft\n"
        "training:\n  citation_faithful: true\n  citation_recall_threshold: 0.8\n"
    ),
}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


class TestEachFieldWarnsThenRefuses:
    @pytest.mark.parametrize("path", sorted(_SET))
    def test_it_warns_and_names_the_version(self, path, capsys, monkeypatch):
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")

        loader.load_config_from_string(_SET[path])

        out = _plain(capsys.readouterr().out)
        assert f"{path} is accepted but read by nothing" in out, out
        assert f"v{STAGED_FIELD_REJECTION_VERSION} will refuse it" in out

    @pytest.mark.parametrize("path", sorted(_SET))
    def test_it_is_refused_once_the_switch_flips(self, path, monkeypatch):
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")

        with pytest.raises(ValueError, match=re.escape(f"{path} is accepted but read by nothing")):
            loader.load_config_from_string(_SET[path])


class TestTheDefaultsStaySilent:
    @pytest.mark.parametrize("severity", ["warn", "error"])
    @pytest.mark.parametrize(
        "training",
        [
            "  early_stop_patience: 2\n",  # the schema default, written out
            "  lr_groups: []\n",  # parses to None: no groups
            "  lr_groups: {}\n",  # the dict spelling, also None
            "  citation_recall_threshold: null\n",  # the default, written out
            "  lr: 2.0e-5\n",  # none of the three set
        ],
    )
    def test_a_default_prints_nothing(self, training, severity, capsys, monkeypatch):
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", severity)

        loader.load_config_from_string(_BASE + "training:\n" + training)

        assert "read by nothing" not in _plain(capsys.readouterr().out)


class TestOneDiagnosticForEarlyStopPatience:
    """#808 left it out of STAGED_FIELDS to avoid two diagnostics for one field:
    ``soup train`` already listed it as not enforced. It moves, it is not copied."""

    def test_soup_train_no_longer_lists_it(self):
        from soup_cli.commands.train import _nondefault_unwired_training_settings

        cfg = loader.load_config_from_string(
            _BASE + "training:\n  early_stop_on_regression: true\n  early_stop_patience: 5\n"
        )

        assert _nondefault_unwired_training_settings(cfg.training) == ["early_stop_on_regression"]
