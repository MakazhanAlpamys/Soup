"""``DEPRECATED_VALUE_SEVERITY``: the switch that turns deprecated config values into refusals.

v0.76 loaded three values with a warning that named the release that would refuse
them (``config/deprecation.py``): ``data.remove_unused_columns: true`` (#759),
``training.quantization: 4bit`` / ``8bit`` on a task that never quantises (#795),
and ``soup ab --beta`` / ``MsprtConfig.beta`` (#1418). Staged fields already had
a switch (``STAGED_FIELD_SEVERITY``); deprecated values had none, so the release
that reached the deadline would have kept printing "Soup v0.77 will refuse it."

v0.77.0 reached that deadline and ships ``"error"``: the three values are refused.
These tests pin both settings of the switch -- ``"warn"`` (through the ``warn``
fixture) byte-identical to v0.76.0, ``"error"`` as shipped -- and the two
staged-field refusal fixes that ship with it (every refusal line indented; each
one says to delete the line).
"""

from __future__ import annotations

import io
import json
import re

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli
from soup_cli.config import deprecation, loader
from soup_cli.config.deprecation import (
    DEPRECATED_VALUE_REJECTION_VERSION,
    SoupConfigDeprecationWarning,
)
from soup_cli.config.staged_fields import STAGED_FIELD_REJECTION_VERSION

VERSION = DEPRECATED_VALUE_REJECTION_VERSION
WILL_REFUSE = f"Soup v{VERSION} will refuse it."
REFUSED = f"Refused as of Soup v{VERSION}."

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOX_RE = re.compile(f"[{chr(0x2500)}-{chr(0x257F)}|]")


def _plain(text: str) -> str:
    """ANSI- and table-border-stripped, whitespace-collapsed CLI output."""
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


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


@pytest.fixture
def warn(monkeypatch):
    monkeypatch.setattr(deprecation, "DEPRECATED_VALUE_SEVERITY", "warn")


@pytest.fixture
def refuse(monkeypatch):
    monkeypatch.setattr(deprecation, "DEPRECATED_VALUE_SEVERITY", "error")


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


class TestTheSwitch:
    def test_this_release_refuses(self):
        """v0.76 warned; v0.77.0 reached the deadline and refuses. The flip was a
        deliberate one-line change in that release's commit, beside the version bump."""
        assert _major_minor(soup_cli.__version__) >= _major_minor(VERSION), (
            f"v{soup_cli.__version__} is before the v{VERSION} deadline this release "
            "was cut to reach"
        )
        assert deprecation.DEPRECATED_VALUE_SEVERITY == "error", (
            f"v{soup_cli.__version__} has reached the v{VERSION} deadline: set "
            "DEPRECATED_VALUE_SEVERITY to 'error'"
        )
        assert deprecation.refuses_deprecated_values() is True

    def test_it_has_exactly_two_settings(self):
        assert deprecation.DEPRECATED_VALUE_SEVERITY in ("warn", "error")

    def test_it_flips_with_the_staged_field_switch(self):
        """Both deadlines name the same release, so both switches move together."""
        if STAGED_FIELD_REJECTION_VERSION == VERSION:
            assert deprecation.DEPRECATED_VALUE_SEVERITY == loader.STAGED_FIELD_SEVERITY

    def test_the_deadline_clause_while_warning(self, warn):
        assert deprecation.deadline_clause() == WILL_REFUSE

    def test_the_deadline_clause_once_refusing(self, refuse):
        assert deprecation.deadline_clause() == REFUSED

    def test_warn_deprecated_value_raises_the_refusal_text_once_refusing(self, refuse):
        with pytest.raises(ValueError) as excinfo:
            deprecation.warn_deprecated_value("x: y is ignored.", refusal="x: y is refused.")
        assert str(excinfo.value) == f"x: y is refused. {REFUSED}"

    def test_without_a_refusal_text_the_message_is_refused(self, refuse):
        with pytest.raises(ValueError) as excinfo:
            deprecation.warn_deprecated_value("x: y does nothing. Remove x.")
        assert str(excinfo.value) == f"x: y does nothing. Remove x. {REFUSED}"

    def test_warn_deprecated_value_still_only_warns_by_default(self, warn):
        with pytest.warns(SoupConfigDeprecationWarning) as record:
            deprecation.warn_deprecated_value("x: y is ignored.", refusal="x: y is refused.")
        assert str(record[0].message) == f"x: y is ignored. {WILL_REFUSE}"

    def test_another_refusals_fix_follows_the_deadline(self, refuse):
        """The deadline belongs to the deprecated value; an older refusal's fix that
        the same message names comes after it, not inside it (#795 with #1357)."""
        with pytest.raises(ValueError) as excinfo:
            deprecation.warn_deprecated_value(
                "x: y is ignored.", refusal="x: y is refused.", after_deadline="Also, z."
            )
        assert str(excinfo.value) == f"x: y is refused. {REFUSED} Also, z."

    def test_a_warning_does_not_carry_the_other_refusals_fix(self, warn):
        with pytest.warns(SoupConfigDeprecationWarning) as record:
            deprecation.warn_deprecated_value(
                "x: y is ignored.", refusal="x: y is refused.", after_deadline="Also, z."
            )
        assert str(record[0].message) == f"x: y is ignored. {WILL_REFUSE}"


# ---------------------------------------------------------------------------
# data.remove_unused_columns: true (#759)
# ---------------------------------------------------------------------------

_RUC = "base: org/model\ntask: sft\ndata:\n  train: d.jsonl\n  remove_unused_columns: {}\n"

_RUC_WARNING = (
    "data.remove_unused_columns: true has no effect and is ignored (no trainer reads "
    "it; the trainers that set it pass false). Delete the line. " + WILL_REFUSE
)
_RUC_REFUSAL = (
    "data.remove_unused_columns: true never had an effect (no trainer reads it; the "
    "trainers that set it pass false). Delete the line. " + REFUSED
)


class TestRemoveUnusedColumns:
    def test_warn_loads_and_prints_the_v0760_line(self, warn, printed):
        cfg = loader.load_config_from_string(_RUC.format("true"))
        assert cfg.data.remove_unused_columns is False
        assert printed.getvalue() == f"Warning: {_RUC_WARNING}\n"

    def test_error_refuses_the_string_load(self, refuse, printed):
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(_RUC.format("true"))
        assert str(excinfo.value) == (
            f"data -> remove_unused_columns: Value error, {_RUC_REFUSAL}"
        )
        assert printed.getvalue() == ""

    def test_error_refuses_the_file_load(self, refuse, printed, tmp_path):
        path = tmp_path / "soup.yaml"
        path.write_text(_RUC.format("true"), encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)
        assert excinfo.value.code == 1
        out = _plain(printed.getvalue())
        assert _RUC_REFUSAL in out, out
        assert "Warning:" not in out and "will refuse it" not in out

    def test_error_still_loads_false(self, refuse, printed):
        """Control: only the deprecated value is refused."""
        cfg = loader.load_config_from_string(_RUC.format("false"))
        assert cfg.data.remove_unused_columns is False
        assert printed.getvalue() == ""


# ---------------------------------------------------------------------------
# training.quantization on a task that never quantises (#795)
# ---------------------------------------------------------------------------

_PRM = "base: org/model\ntask: prm\ndata:\n  train: x.jsonl\n"


def _quant_warning(quant: str) -> str:
    return (
        f"training.quantization: {quant} has no effect on task='prm' and is ignored: "
        "its trainer never quantises the base, so the run trains unquantised. Set "
        "quantization: none. " + WILL_REFUSE
    )


def _quant_refusal(quant: str, fix: str = "Delete the line or set quantization: none.") -> str:
    return (
        f"training.quantization: {quant} has no effect on task='prm': its trainer never "
        f"quantises the base, so no run can honour it. {fix} {REFUSED}"
    )


class TestQuantizationOnAnUnquantisedTask:
    @pytest.mark.parametrize("quant", ["4bit", "8bit"])
    def test_warn_loads_and_prints_the_v0760_line(self, warn, printed, quant):
        cfg = loader.load_config_from_string(_PRM + f"training:\n  quantization: {quant}\n")
        assert cfg.training.quantization == "none"
        assert printed.getvalue() == f"Warning: {_quant_warning(quant)}\n"

    @pytest.mark.parametrize("quant", ["4bit", "8bit"])
    def test_error_refuses_the_string_load(self, refuse, printed, quant):
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(_PRM + f"training:\n  quantization: {quant}\n")
        assert str(excinfo.value) == f": Value error, {_quant_refusal(quant)}"
        assert printed.getvalue() == ""

    def test_error_refuses_the_file_load(self, refuse, printed, tmp_path):
        path = tmp_path / "soup.yaml"
        path.write_text(_PRM + "training:\n  quantization: 4bit\n", encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)
        assert excinfo.value.code == 1
        out = _plain(printed.getvalue())
        assert _quant_refusal("4bit") in out, out
        assert "Warning:" not in out and "will refuse it" not in out

    def test_error_names_the_load_in_8bit_alias_that_set_it(self, refuse):
        """``load_in_8bit: true`` rewrites ``quantization`` to ``8bit``: a user told
        to set ``quantization: none`` would be refused again, so the alias is named."""
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(_PRM + "training:\n  load_in_8bit: true\n")
        fix = (
            "Delete training.load_in_8bit: true, which sets quantization to 8bit, and "
            "any training.quantization line."
        )
        assert str(excinfo.value) == f": Value error, {_quant_refusal('8bit', fix)}"

    @pytest.mark.parametrize("alias", ["false", "null"])
    def test_error_does_not_name_an_alias_that_was_not_set(self, refuse, alias):
        """``load_in_8bit: false`` / ``null`` did not set the value, so the refusal
        must not tell the user to delete a ``load_in_8bit: true`` they never wrote."""
        text = _PRM + f"training:\n  load_in_8bit: {alias}\n  quantization: 4bit\n"
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(text)
        message = str(excinfo.value)
        assert "Delete the line or set quantization: none." in message, message
        assert "load_in_8bit" not in message, message

    @pytest.mark.parametrize("training", ["", "training:\n  quantization: none\n"])
    def test_error_still_loads_an_unset_or_none_value(self, refuse, printed, training):
        """Control: the resolved default is not a deprecated value."""
        cfg = loader.load_config_from_string(_PRM + training)
        assert cfg.training.quantization == "none"
        assert printed.getvalue() == ""

    def test_error_leaves_a_quantising_task_alone(self, refuse, printed):
        """Control: sft quantises, so ``4bit`` is a real setting there."""
        cfg = loader.load_config_from_string(
            "base: org/model\ntask: sft\ndata:\n  train: x.jsonl\n"
            "training:\n  quantization: 4bit\n"
        )
        assert cfg.training.quantization == "4bit"
        assert printed.getvalue() == ""


# ---------------------------------------------------------------------------
# soup ab --beta / MsprtConfig.beta (#1418)
# ---------------------------------------------------------------------------

_BETA_HEAD = "beta no longer does anything (#1418): soup ab now accepts H0 once"
_BETA_TAIL = "Remove beta; to make accept_h0 stricter, lower alpha."
_ROWS = [0.80, 0.83, 0.77, 0.81, 0.79, 0.84, 0.78, 0.82, 0.80, 0.76] * 3

runner = CliRunner()


def _run_ab(tmp_path, monkeypatch, *extra):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    rows = [{"arm": arm, "judge_score": x} for arm in ("control", "treatment") for x in _ROWS]
    (tmp_path / "ab.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return runner.invoke(app, ["ab", "--input", "ab.jsonl", "--metric", "judge_score", *extra])


class TestRetiredBeta:
    def test_warn_msprt_config_warns_with_the_v0760_text(self, warn):
        from soup_cli.utils.ab_test import MsprtConfig, retired_beta_message

        with pytest.warns(SoupConfigDeprecationWarning) as record:
            cfg = MsprtConfig(metric="judge_score", beta=0.2)
        assert cfg.beta == 0.2
        message = str(record[0].message)
        assert message == f"{retired_beta_message()} {WILL_REFUSE}"
        assert message.startswith(_BETA_HEAD)
        assert message.endswith(f"{_BETA_TAIL} {WILL_REFUSE}")

    def test_error_msprt_config_refuses_beta(self, refuse):
        from soup_cli.utils.ab_test import MsprtConfig

        with pytest.raises(ValueError) as excinfo:
            MsprtConfig(metric="judge_score", beta=0.2)
        message = str(excinfo.value)
        assert message.startswith(_BETA_HEAD)
        assert message.endswith(f"{_BETA_TAIL} {REFUSED}")

    def test_error_msprt_config_without_beta_builds(self, refuse, recwarn):
        from soup_cli.utils.ab_test import MsprtConfig

        assert MsprtConfig(metric="judge_score").beta is None
        assert not recwarn.list

    def test_warn_cli_prints_the_v0760_warning_and_runs(self, warn, tmp_path, monkeypatch):
        result = _run_ab(tmp_path, monkeypatch, "--beta", "0.95")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert (
            "Warning: --beta no longer does anything (#1418): soup ab now accepts H0 once"
            in out
        ), out
        assert f"Remove --beta; to make accept_h0 stricter, lower --alpha. {WILL_REFUSE}" in out
        assert "decision accept_h0" in out

    def test_error_cli_refuses_beta_with_a_usage_error(self, refuse, tmp_path, monkeypatch):
        # Exit 2, the code soup ab already uses for its other flag errors.
        result = _run_ab(tmp_path, monkeypatch, "--beta", "0.95")
        assert result.exit_code == 2, (
            result.output,
            repr(result.exception),
        )
        out = _plain(result.output)
        assert (
            "Error: --beta no longer does anything (#1418): soup ab now accepts H0 once"
            in out
        ), out
        assert f"Remove --beta; to make accept_h0 stricter, lower --alpha. {REFUSED}" in out
        # Refused before any work: no verdict, and no warning wording.
        assert "decision" not in out
        assert "Warning:" not in out and "will refuse it" not in out

    def test_error_cli_without_beta_runs(self, refuse, tmp_path, monkeypatch):
        result = _run_ab(tmp_path, monkeypatch)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert "decision accept_h0" in out
        assert "no longer does anything" not in out


# ---------------------------------------------------------------------------
# As shipped: no fixture, the switch as the release sets it
# ---------------------------------------------------------------------------


class TestAsShipped:
    """v0.77.0 refuses each of the three values with the switch untouched."""

    def test_remove_unused_columns_true_is_refused(self, printed):
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(_RUC.format("true"))
        assert str(excinfo.value) == (
            f"data -> remove_unused_columns: Value error, {_RUC_REFUSAL}"
        )

    @pytest.mark.parametrize("quant", ["4bit", "8bit"])
    def test_quantization_on_an_unquantised_task_is_refused(self, printed, quant):
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(_PRM + f"training:\n  quantization: {quant}\n")
        assert str(excinfo.value) == f": Value error, {_quant_refusal(quant)}"

    def test_soup_ab_beta_is_a_usage_error(self, tmp_path, monkeypatch):
        result = _run_ab(tmp_path, monkeypatch, "--beta", "0.95")
        assert result.exit_code == 2, (result.output, repr(result.exception))
        out = _plain(result.output)
        assert f"Remove --beta; to make accept_h0 stricter, lower --alpha. {REFUSED}" in out
        assert "decision" not in out


# ---------------------------------------------------------------------------
# Staged-field refusal wording (#808)
# ---------------------------------------------------------------------------

_STAGED = (
    "base: hf/model\ntask: sft\ndata:\n  train: ./t.jsonl\n  format: auto\n"
    "  split_thinking: true\n  video_fps: 24.0\noutput: ./o\n"
)
_STAGED_VERSION = STAGED_FIELD_REJECTION_VERSION


class TestStagedFieldRefusal:
    def test_error_string_load_says_to_delete_each_line(self, monkeypatch, printed):
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(_STAGED)
        assert str(excinfo.value) == (
            f"data.video_fps is accepted but read by nothing; refused as of "
            f"v{_STAGED_VERSION}. Delete the line.\n"
            f"data.split_thinking is accepted but read by nothing; refused as of "
            f"v{_STAGED_VERSION}. Delete the line."
        )

    def test_error_file_load_indents_every_refusal_line(self, monkeypatch, printed, tmp_path):
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
        path = tmp_path / "soup.yaml"
        path.write_text(_STAGED, encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)
        assert excinfo.value.code == 1
        lines = printed.getvalue().splitlines()
        assert lines == [
            "Config validation error:",
            "",
            f"  data.video_fps is accepted but read by nothing; refused as of "
            f"v{_STAGED_VERSION}. Delete the line.",
            f"  data.split_thinking is accepted but read by nothing; refused as of "
            f"v{_STAGED_VERSION}. Delete the line.",
        ], lines

    def test_warn_output_is_the_v0760_output(self, monkeypatch, printed):
        """Under ``"warn"`` nothing changes, the second line's indent included."""
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")
        cfg = loader.load_config_from_string(_STAGED)
        assert cfg.data.split_thinking is True
        assert printed.getvalue() == (
            f"Warning: data.video_fps is accepted but read by nothing; "
            f"v{_STAGED_VERSION} will refuse it\n"
            f"data.split_thinking is accepted but read by nothing; "
            f"v{_STAGED_VERSION} will refuse it\n"
        )
