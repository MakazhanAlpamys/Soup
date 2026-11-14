"""#1814 -- ``training.warmup_auto`` and ``training.forgetting_threshold`` warn at load.

Neither field reaches a consumer: no trainer reads ``warmup_auto`` (``warmup_ratio``
already resolves over the real optimizer steps, #1507), and nothing reads
``training.forgetting_threshold`` (``soup ship`` has its own,
``eval.ship.forgetting_threshold``). No warning for either ever shipped, so v0.78.0
loads them with one and v0.79.0 refuses them: one minor of notice, as 0.76 -> 0.77
gave the first cohort. The value is not touched; only a value set away from the
default speaks, because Soup's own config dumps write every default.
"""

from __future__ import annotations

import io
import pathlib
import re
import warnings

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

import soup_cli
from soup_cli.cli import app
from soup_cli.config import deprecation, loader
from soup_cli.config.deprecation import (
    UNCONSUMED_FIELD_REJECTION_VERSION,
    SoupConfigDeprecationWarning,
)
from soup_cli.config.schema import TrainingConfig
from tests.conftest import strip_ansi
from tests.test_issue748_config_fields_reach_a_consumer import KNOWN_UNCONSUMED

ROOT = pathlib.Path(__file__).resolve().parents[1]
_BASE = "base: org/m\ntask: sft\ndata:\n  train: ./t.jsonl\n"
_RELEASE = f"v{UNCONSUMED_FIELD_REJECTION_VERSION}"

_WARMUP_LINE = "training.warmup_auto: true has no effect"
_THRESHOLD_LINE = "training.forgetting_threshold: 0.2 has no effect"


def _plain(text: str) -> str:
    """CLI output as one line without colour: Rich splits numbers with escapes."""
    return " ".join(strip_ansi(text).split())


def _major_minor(version: str) -> tuple[int, int]:
    major, minor = re.match(r"(\d+)\.(\d+)", version).groups()
    return int(major), int(minor)


@pytest.fixture
def printed(monkeypatch):
    """What the loader prints, uncoloured and unwrapped."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        loader, "console", Console(file=buffer, width=500, force_terminal=False, color_system=None)
    )
    return buffer


def _training(*lines: str) -> str:
    return _BASE + "training:\n" + "".join(f"  {line}\n" for line in lines)


# ---------------------------------------------------------------------------
# The warning itself
# ---------------------------------------------------------------------------


class TestTheWarning:
    def test_warmup_auto_true_names_the_field_the_effect_the_fix_and_the_release(self):
        with pytest.warns(SoupConfigDeprecationWarning) as record:
            TrainingConfig(warmup_auto=True)
        assert len(record) == 1
        message = str(record[0].message)
        assert message.startswith(_WARMUP_LINE + ":")
        assert "training.warmup_ratio" in message  # what to use instead
        assert "Remove the line." in message
        assert message.endswith(f"Soup {_RELEASE} will refuse it.")

    def test_forgetting_threshold_names_the_field_the_effect_the_fix_and_the_release(self):
        with pytest.warns(SoupConfigDeprecationWarning) as record:
            TrainingConfig(forgetting_threshold=0.2)
        assert len(record) == 1
        message = str(record[0].message)
        assert message.startswith(_THRESHOLD_LINE + ":")
        assert "Remove the line." in message
        # the ship threshold has the same name and IS read: say where it lives
        assert "eval.ship.forgetting_threshold" in message
        assert message.endswith(f"Soup {_RELEASE} will refuse it.")

    def test_it_is_a_future_warning_the_loader_already_knows(self):
        assert issubclass(SoupConfigDeprecationWarning, FutureWarning)

    def test_the_value_is_kept_unchanged(self):
        with pytest.warns(SoupConfigDeprecationWarning):
            config = TrainingConfig(warmup_auto=True, forgetting_threshold=0.25)
        assert config.warmup_auto is True
        assert config.forgetting_threshold == 0.25

    def test_both_fields_give_two_separate_warnings(self):
        with pytest.warns(SoupConfigDeprecationWarning) as record:
            TrainingConfig(warmup_auto=True, forgetting_threshold=0.2)
        messages = [str(item.message) for item in record]
        assert len(messages) == 2
        assert sum(_WARMUP_LINE in text for text in messages) == 1
        assert sum(_THRESHOLD_LINE in text for text in messages) == 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"warmup_auto": False},
            {"forgetting_threshold": 0.10},
            {"warmup_auto": False, "forgetting_threshold": 0.1},
        ],
        ids=["omitted", "warmup-false", "threshold-default", "both-default"],
    )
    def test_the_default_is_silent(self, kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("error", SoupConfigDeprecationWarning)
            TrainingConfig(**kwargs)

    def test_the_refusal_release_is_the_ruled_one(self):
        """The ruling on #1814: warned in 0.78, refused in 0.79. A published promise:
        moving it needs a changelog entry, not a quiet edit of the constant."""
        assert UNCONSUMED_FIELD_REJECTION_VERSION == "0.79.0"

    def test_the_release_is_not_reached_yet(self):
        """The tripwire: the release that reaches v0.79 turns this red and must make
        ``warn_unconsumed_field`` refuse (and move both fields to REFUSED_AT_LOAD in
        the #748 scanner), or the warning keeps promising a refusal that did not come."""
        deadline = _major_minor(UNCONSUMED_FIELD_REJECTION_VERSION)
        assert _major_minor(soup_cli.__version__) < deadline, (
            f"soup_cli is {soup_cli.__version__}: {_RELEASE} "
            "has arrived, so warmup_auto and forgetting_threshold are refused now"
        )

    def test_it_warns_whatever_the_deprecated_value_switch_says(self, monkeypatch):
        """``DEPRECATED_VALUE_SEVERITY`` flipped to ``error`` for the v0.77 cohort; this
        cohort has its own deadline, so that switch must not refuse it early."""
        monkeypatch.setattr(deprecation, "DEPRECATED_VALUE_SEVERITY", "error")
        with pytest.warns(SoupConfigDeprecationWarning):
            TrainingConfig(warmup_auto=True)

    def test_the_other_forgetting_threshold_is_a_different_field_and_stays_quiet(self, printed):
        """``eval.ship.forgetting_threshold`` is read by ``soup ship``; only the
        ``training`` one is warned about."""
        text = _BASE + "eval:\n  ship:\n    forgetting_threshold: 0.2\n"
        config = loader.load_config_from_string(text)
        assert config.eval.ship.forgetting_threshold == 0.2
        assert printed.getvalue() == ""

    def test_a_value_out_of_bounds_is_still_refused_without_the_warning(self, printed):
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(_training("forgetting_threshold: 0.8"))
        assert "less than or equal to 0.5" in str(excinfo.value)
        assert "has no effect" not in printed.getvalue()


# ---------------------------------------------------------------------------
# Every spelling the loader accepts
# ---------------------------------------------------------------------------


class TestSpellings:
    @pytest.mark.parametrize("spelling", ["true", "True", "yes", "on", '"true"', "1"])
    def test_every_truthy_spelling_of_warmup_auto_warns_once(self, printed, spelling):
        config = loader.load_config_from_string(_training(f"warmup_auto: {spelling}"))
        assert config.training.warmup_auto is True
        output = printed.getvalue()
        assert output.count(_WARMUP_LINE) == 1, output
        assert "Warning:" in output

    @pytest.mark.parametrize("spelling", ["false", "False", "no", "off", '"false"', "0"])
    def test_every_false_spelling_of_warmup_auto_is_silent(self, printed, spelling):
        config = loader.load_config_from_string(_training(f"warmup_auto: {spelling}"))
        assert config.training.warmup_auto is False
        assert printed.getvalue() == ""

    @pytest.mark.parametrize("spelling", ["0.2", '"0.2"', "2.0e-1", "2e-1", "0.20", ".2"])
    def test_every_spelling_of_a_non_default_threshold_warns_once(self, printed, spelling):
        config = loader.load_config_from_string(_training(f"forgetting_threshold: {spelling}"))
        assert config.training.forgetting_threshold == 0.2
        output = printed.getvalue()
        assert output.count(_THRESHOLD_LINE) == 1, output

    @pytest.mark.parametrize("bound", ["0.01", "0.5"])
    def test_the_bounds_of_the_range_are_values_too(self, printed, bound):
        loader.load_config_from_string(_training(f"forgetting_threshold: {bound}"))
        shown = f"training.forgetting_threshold: {float(bound)!r} has no effect"
        assert shown in printed.getvalue()

    @pytest.mark.parametrize("spelling", ["0.1", "0.10", "1e-1", ".1", '"0.1"', "1.0e-1"])
    def test_every_spelling_of_the_default_threshold_is_silent(self, printed, spelling):
        config = loader.load_config_from_string(_training(f"forgetting_threshold: {spelling}"))
        assert config.training.forgetting_threshold == 0.1
        assert printed.getvalue() == ""

    def test_unset_is_silent(self, printed):
        loader.load_config_from_string(_training("lr: 1e-4"))
        assert printed.getvalue() == ""


# ---------------------------------------------------------------------------
# Every way a config reaches the loader
# ---------------------------------------------------------------------------


def _from_string(text, tmp_path):
    return loader.load_config_from_string(text)


def _from_file(text, tmp_path):
    path = tmp_path / "soup.yaml"
    path.write_text(text, encoding="utf-8")
    return loader.load_config(path)


def _from_mapping(text, tmp_path):
    return loader.config_from_mapping(yaml.safe_load(text))


def _from_validated_mapping(text, tmp_path):
    return loader.validate_config_mapping(yaml.safe_load(text))


def _from_file_with_override(text, tmp_path):
    """The CLI-flag path: ``training_overrides`` merge into ``training:`` before the build."""
    raw = yaml.safe_load(text)
    raw["training"].pop("warmup_auto", None)
    raw["training"].pop("forgetting_threshold", None)
    path = tmp_path / "soup.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return loader.load_config(
        path, training_overrides={"warmup_auto": True, "forgetting_threshold": 0.2}
    )


_ENTRY_POINTS = [
    pytest.param(_from_string, id="load_config_from_string"),
    pytest.param(_from_file, id="load_config"),
    pytest.param(_from_mapping, id="config_from_mapping"),
    pytest.param(_from_validated_mapping, id="validate_config_mapping"),
    pytest.param(_from_file_with_override, id="training_overrides"),
]


class TestEveryEntryPoint:
    @pytest.mark.parametrize("load", _ENTRY_POINTS)
    def test_each_field_is_printed_once_through_the_rich_console(
        self, printed, tmp_path, capsys, load
    ):
        text = _training("warmup_auto: true", "forgetting_threshold: 0.2")
        with warnings.catch_warnings(record=True) as leaked:
            warnings.simplefilter("always")
            config = load(text, tmp_path)
        output = printed.getvalue()
        assert output.count(_WARMUP_LINE) == 1, output
        assert output.count(_THRESHOLD_LINE) == 1, output
        assert output.count("Warning:") == 2, output
        # printed by the loader's Console, not by print() and not as a Python warning
        captured = capsys.readouterr()
        assert (captured.out, captured.err) == ("", "")
        assert not [w for w in leaked if issubclass(w.category, SoupConfigDeprecationWarning)]
        assert config.training.warmup_auto is True
        assert config.training.forgetting_threshold == 0.2

    @pytest.mark.parametrize("load", _ENTRY_POINTS[:-1])  # the override path always sets both
    def test_a_config_that_sets_neither_prints_nothing(self, printed, tmp_path, load):
        text = _training("warmup_auto: false", "forgetting_threshold: 0.1")
        load(text, tmp_path)
        assert printed.getvalue() == ""

    def test_it_prints_again_on_the_next_load(self, printed):
        text = _training("warmup_auto: true")
        loader.load_config_from_string(text)
        loader.load_config_from_string(text)
        assert printed.getvalue().count(_WARMUP_LINE) == 2

    def test_the_line_is_safe_for_the_terminal(self, printed):
        """The text is built from constants and a validated float, never from the file."""
        loader.load_config_from_string(_training("forgetting_threshold: 0.2"))
        assert "\x1b" not in printed.getvalue()


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


class TestTheCli:
    @pytest.mark.parametrize("color", [False, True])
    def test_soup_cost_shows_both_warnings_once(self, tmp_path, monkeypatch, color):
        """``soup cost --config`` loads the file through ``load_config``; Rich colours the
        version number on a colour terminal, so the output is read plain (FORCE_COLOR=1)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(
            _training("warmup_auto: true", "forgetting_threshold: 0.2"), encoding="utf-8"
        )
        result = CliRunner().invoke(
            app, ["cost", "--config", "soup.yaml"], color=color,
            env={"FORCE_COLOR": "1" if color else "0"},
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert output.count(_WARMUP_LINE) == 1, output
        assert output.count(_THRESHOLD_LINE) == 1, output
        assert output.count(f"Soup {_RELEASE} will refuse it.") == 2, output
        assert output.count("Remove the line.") == 2, output

    def test_soup_cost_is_silent_when_neither_is_set(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_training("lr: 1e-4"), encoding="utf-8")
        result = CliRunner().invoke(app, ["cost", "--config", "soup.yaml"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "has no effect" not in _plain(result.output)


# ---------------------------------------------------------------------------
# One diagnostic per field, in the right place
# ---------------------------------------------------------------------------


class TestOneDiagnosticForForgettingThreshold:
    """``soup train`` used to list ``forgetting_threshold`` as "not enforced during
    training". The loader reports it now, with the refusal release, so it moves."""

    def test_soup_train_no_longer_lists_it(self):
        from soup_cli.commands.train import _nondefault_unwired_training_settings

        with pytest.warns(SoupConfigDeprecationWarning):
            config = TrainingConfig(forgetting_threshold=0.2, forgetting_detection=True)
        assert _nondefault_unwired_training_settings(config) == ["forgetting_detection"]

    def test_the_unsloth_tip_probe_does_not_report_it_a_second_time(self, printed):
        """``_loads_on_unsloth`` validates the dumped config again; the loader already
        reported the field, and a raw Python warning would be the second line."""
        from soup_cli.commands.train import _loads_on_unsloth

        config = loader.load_config_from_string(
            _training("warmup_auto: true", "forgetting_threshold: 0.2")
        )
        assert printed.getvalue().count("has no effect") == 2
        with warnings.catch_warnings(record=True) as leaked:
            warnings.simplefilter("always")
            _loads_on_unsloth(config)
        assert not [w for w in leaked if issubclass(w.category, SoupConfigDeprecationWarning)]


# ---------------------------------------------------------------------------
# The registry and the shipped files
# ---------------------------------------------------------------------------


class TestRegistry:
    @pytest.mark.parametrize("key", ["training.warmup_auto", "training.forgetting_threshold"])
    def test_the_scanner_records_warned_in_0_78_refused_in_0_79(self, key):
        reason = KNOWN_UNCONSUMED[key]
        assert "#1814" in reason
        assert "warns at load from v0.78.0" in reason
        assert f"refused as of {_RELEASE}" in reason

    @pytest.mark.parametrize("name", ["warmup_auto", "forgetting_threshold"])
    def test_the_field_description_says_so_and_names_the_release(self, name):
        description = TrainingConfig.model_fields[name].description
        assert description.startswith("Deprecated, no effect")
        assert f"refused as of {_RELEASE}" in description


class TestNoShippedFileSetsThem:
    """The synthetic-workflow example set ``warmup_auto: true``, so ``soup train --config
    examples/synthetic_workflow.yaml`` would have warned about a line the user never
    wrote (and refused it in v0.79.0). Nothing enumerated ``examples/`` for it."""

    def test_no_example_sets_either_field_under_training(self):
        offenders = []
        for path in sorted((ROOT / "examples").rglob("*.yaml")):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            training = document.get("training") if isinstance(document, dict) else None
            if isinstance(training, dict):
                offenders += [
                    f"{path.relative_to(ROOT).as_posix()}: training.{name}"
                    for name in ("warmup_auto", "forgetting_threshold")
                    if name in training
                ]
        assert offenders == [], offenders

    def test_the_synthetic_workflow_example_loads_without_a_warning(self, printed):
        text = (ROOT / "examples" / "synthetic_workflow.yaml").read_text(encoding="utf-8")
        loader.load_config_from_string(text)
        assert "has no effect" not in printed.getvalue()
