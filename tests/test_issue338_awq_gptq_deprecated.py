"""#338 -- AWQ and GPTQ export are deprecated, and the commands say so.

Both upstream projects are archived and neither extra installs next to the
train extra, so the two formats go away one release after this notice. Until
then they keep working: ``soup export --format awq / gptq`` prints the notice
first and then does what it did, and ``soup quantize --to awq / gptq`` prints
it next to the command it hands the user.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

import soup_cli.commands.export as export_mod
from soup_cli.cli import app
from soup_cli.commands.export import (
    DEPRECATED_FORMATS,
    REPLACEMENT_FORMATS,
    SUPPORTED_FORMATS,
    deprecated_format_notice,
)

# Rich colours per character on a colour-capable runner and wraps at the
# terminal width, drawing box borders between the fragments of a wrapped help
# line. Strip ANSI escapes and box-drawing characters, then collapse
# whitespace, before any substring match on CLI output.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOX_RE = re.compile(r"[\u2500-\u257f]")

DEPRECATED = ("awq", "gptq")
# The formats the decision on #338 names as the way out.
REPLACEMENTS = ("gguf", "onnx", "tensorrt", "bitnet", "tq1_0")


def _plain(text: str) -> str:
    """ANSI- and border-stripped, whitespace-collapsed CLI output."""
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


class TestTheNotice:
    def test_exactly_awq_and_gptq_are_deprecated(self) -> None:
        assert sorted(DEPRECATED_FORMATS) == sorted(DEPRECATED)

    def test_the_deprecated_formats_still_run(self) -> None:
        """Deprecated is not removed: the formats stay selectable for one more release."""
        assert set(DEPRECATED) <= set(SUPPORTED_FORMATS)

    def test_the_replacements_are_the_ones_the_decision_names(self) -> None:
        assert tuple(REPLACEMENT_FORMATS) == REPLACEMENTS

    def test_every_replacement_is_a_format_that_stays(self) -> None:
        assert set(REPLACEMENT_FORMATS) <= set(SUPPORTED_FORMATS)
        assert not set(REPLACEMENT_FORMATS) & set(DEPRECATED_FORMATS)

    @pytest.mark.parametrize("fmt", DEPRECATED)
    def test_it_names_the_format_the_removal_and_every_replacement(self, fmt: str) -> None:
        notice = deprecated_format_notice(fmt)
        assert notice is not None
        assert f"--format {fmt} is deprecated" in notice
        assert "will be removed in the next release" in notice
        for replacement in REPLACEMENTS:
            assert re.search(rf"\b{re.escape(replacement)}\b", notice), (replacement, notice)

    @pytest.mark.parametrize("fmt", DEPRECATED)
    def test_it_does_not_send_the_user_to_the_other_deprecated_format(self, fmt: str) -> None:
        other = "gptq" if fmt == "awq" else "awq"
        assert other not in deprecated_format_notice(fmt).lower()

    @pytest.mark.parametrize("fmt", DEPRECATED)
    def test_it_says_why(self, fmt: str) -> None:
        notice = deprecated_format_notice(fmt)
        assert "archived upstream" in notice
        assert f"the {fmt} extra cannot be installed next to the train extra" in notice

    @pytest.mark.parametrize("fmt", sorted(set(SUPPORTED_FORMATS) - set(DEPRECATED)))
    def test_a_format_that_stays_has_no_notice(self, fmt: str) -> None:
        assert deprecated_format_notice(fmt) is None

    @pytest.mark.parametrize("fmt", ["", "AWQ", "awq ", "exl2", "nope"])
    def test_an_unknown_spelling_has_no_notice(self, fmt: str) -> None:
        """The command lower-cases nothing: ``--format AWQ`` is refused as unsupported."""
        assert deprecated_format_notice(fmt) is None

    def test_the_notice_carries_no_console_markup(self) -> None:
        """It is printed through Rich: a square bracket in it would be read as a style."""
        for fmt in DEPRECATED:
            assert "[" not in deprecated_format_notice(fmt)


class TestExportPrintsIt:
    @pytest.mark.parametrize("fmt", DEPRECATED)
    def test_the_notice_comes_first_and_the_export_path_still_runs(self, fmt, tmp_path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        result = CliRunner().invoke(app, ["export", "--model", str(model_dir), "--format", fmt])

        output = _plain(result.output)
        assert f"Deprecated: --format {fmt} is deprecated" in output, (
            result.output,
            repr(result.exception),
        )
        # What the command did before, it still does: with no calibration data
        # it stops at the same requirement, with the same exit code.
        requirement = f"{fmt.upper()} export requires --calibration-data"
        assert requirement in output
        assert output.index("Deprecated:") < output.index(requirement)
        assert result.exit_code == 1

    @pytest.mark.parametrize("fmt", DEPRECATED)
    def test_the_notice_leads_when_the_extra_is_missing(self, fmt, tmp_path, monkeypatch) -> None:
        """Most users meet these formats through the "not installed" message."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(
            app,
            [
                "export", "--model", str(model_dir), "--format", fmt,
                "--calibration-data", str(cal_file),
            ],
        )

        output = _plain(result.output)
        assert result.exit_code != 0
        assert output.startswith(f"Deprecated: --format {fmt} is deprecated"), result.output

    @pytest.mark.parametrize("fmt", DEPRECATED)
    def test_the_notice_is_printed_once(self, fmt, tmp_path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        result = CliRunner().invoke(app, ["export", "--model", str(model_dir), "--format", fmt])

        assert _plain(result.output).count("is deprecated") == 1

    def test_a_format_that_stays_prints_no_notice(self, tmp_path, monkeypatch) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        calls = []
        monkeypatch.setattr(export_mod, "_export_onnx", lambda *args, **kwargs: calls.append(args))

        result = CliRunner().invoke(app, ["export", "--model", str(model_dir), "--format", "onnx"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(calls) == 1
        assert "deprecated" not in _plain(result.output).lower()

    def test_the_help_marks_both_formats(self) -> None:
        result = CliRunner().invoke(app, ["export", "--help"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "awq and gptq are deprecated" in _plain(result.output)


class TestQuantizePrintsIt:
    @pytest.mark.parametrize("fmt", DEPRECATED)
    def test_it_still_prints_the_command_and_adds_the_notice(self, fmt: str) -> None:
        result = CliRunner().invoke(app, ["quantize", "./out", "--to", fmt])

        output = _plain(result.output)
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert f"soup export --model ./out --format {fmt} --bits 4" in output
        assert f"Deprecated: --format {fmt} is deprecated" in output

    @pytest.mark.parametrize("fmt", DEPRECATED)
    def test_an_upper_case_target_gets_the_notice_too(self, fmt: str) -> None:
        """``soup quantize`` lower-cases ``--to``, so the notice follows the canonical name."""
        result = CliRunner().invoke(app, ["quantize", "./out", "--to", fmt.upper()])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert f"Deprecated: --format {fmt} is deprecated" in _plain(result.output)

    @pytest.mark.parametrize("fmt", ["gguf", "onnx", "tensorrt"])
    def test_a_target_that_stays_prints_no_notice(self, fmt: str) -> None:
        result = CliRunner().invoke(app, ["quantize", "./out", "--to", fmt])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "deprecated" not in _plain(result.output).lower()

    def test_the_help_marks_both_targets(self) -> None:
        result = CliRunner().invoke(app, ["quantize", "--help"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "awq and gptq are deprecated" in _plain(result.output)
