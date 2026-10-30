"""How ``load_config`` prints a config Pydantic refuses (``Config validation error:``).

Each Pydantic error entry was printed as ``<loc>: <msg>``. A model-level refusal
(a ``model_validator`` raising ``ValueError``) has an empty ``loc``, so it came out
as ``  : Value error, training.quantization: 4bit has no effect...`` -- an empty
location, then Pydantic's ``Value error, `` prefix in front of Soup's own text. In
v0.77 many refusals are model-level, so that was the first line most upgraders saw
(found by the v0.77.0 release smoke).

The printed form is now the message alone for a model-level entry and
``<loc>: <message>`` for a field-level one, without the ``Value error, `` prefix in
either. Only the terminal is changed: ``load_config_from_string`` still raises
``ValueError("<loc>: <msg>")`` with Pydantic's text, because the Web UI, MCP and
tests (``test_deprecated_value_severity``, ``test_issue795_*``, ``test_issue1357_*``)
read that string, and ``test_issue1357_*`` parses the fix out of it.
"""

from __future__ import annotations

import io
import re

import pytest
from pydantic import BaseModel, model_validator
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config import loader
from soup_cli.config.deprecation import DEPRECATED_VALUE_REJECTION_VERSION

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_PRM_4BIT = """\
base: org/model
task: prm
data:
  train: ./x.jsonl
training:
  quantization: 4bit
output: ./o
"""

_SFT_REMOVE_UNUSED_COLUMNS = """\
base: org/model
task: sft
data:
  train: ./x.jsonl
  remove_unused_columns: true
output: ./o
"""

_GRPO_REMOTE_HTTP_JUDGE = """\
base: org/model
task: grpo
data:
  train: ./x.jsonl
training:
  reward_fn: judge_rank
  grpo_judge: http://judge.example.com/m
  grpo_judge_max_calls: 1000
output: ./o
"""

_SFT_EPOCHS_NOT_A_NUMBER = """\
base: org/model
task: sft
data:
  train: ./x.jsonl
training:
  epochs: not-a-number
output: ./o
"""

_QUANT_REFUSAL = (
    "training.quantization: 4bit has no effect on task='prm': its trainer never "
    "quantises the base, so no run can honour it. Delete the line or set "
    f"quantization: none. Refused as of Soup v{DEPRECATED_VALUE_REJECTION_VERSION}."
)

_RUC_REFUSAL = (
    "data.remove_unused_columns: true never had an effect (no trainer reads it; the "
    "trainers that set it pass false). Delete the line. "
    f"Refused as of Soup v{DEPRECATED_VALUE_REJECTION_VERSION}."
)


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _dry_run(tmp_path, monkeypatch, yaml_text: str):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(yaml_text, encoding="utf-8")
    return CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])


@pytest.fixture
def printed(monkeypatch):
    """What the loader prints, uncoloured and unwrapped."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        loader, "console", Console(file=buffer, width=500, force_terminal=False, color_system=None)
    )
    return buffer


class TestThroughSoupTrainDryRun:
    def test_a_model_level_refusal_is_the_message_alone(self, tmp_path, monkeypatch):
        result = _dry_run(tmp_path, monkeypatch, _PRM_4BIT)
        out = _plain(result.output)

        assert result.exit_code == 1, (result.output, repr(result.exception))
        # Contiguous: no empty location (": ") and no "Value error, " between the
        # header and Soup's own text.
        assert (
            "Config validation error: training.quantization: 4bit has no effect on task='prm'"
        ) in out, out
        assert "Value error" not in out, out
        assert ": :" not in out, out
        # It still names the fix.
        assert "Delete the line or set quantization: none." in out, out

    def test_a_field_level_deprecated_value_names_its_field_once(self, tmp_path, monkeypatch):
        """A deprecated-value refusal starts with the field's dotted path (its text
        was the v0.76 warning, which had no location), so ``<loc>: `` in front
        named the field twice (v0.77.0 release smoke). It is the message alone."""
        result = _dry_run(tmp_path, monkeypatch, _SFT_REMOVE_UNUSED_COLUMNS)
        out = _plain(result.output)

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert (
            "Config validation error: data.remove_unused_columns: true never had an effect"
        ) in out, out
        assert "data -> remove_unused_columns" not in out, out
        assert "Value error" not in out, out

    def test_a_field_level_refusal_keeps_its_location(self, tmp_path, monkeypatch):
        """The control: an entry whose message does not repeat the field still reads
        ``<loc>: <message>``, and loses the ``Value error, `` prefix there too."""
        result = _dry_run(tmp_path, monkeypatch, _GRPO_REMOTE_HTTP_JUDGE)
        out = _plain(result.output)

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert (
            "Config validation error: training -> grpo_judge: judge_model "
            "'http://judge.example.com/m' uses unsupported scheme"
        ) in out, out
        assert "Value error" not in out, out

    def test_a_field_level_type_error_is_unchanged(self, tmp_path, monkeypatch):
        """Pydantic's own messages carry no ``Value error, `` prefix and print as
        before."""
        result = _dry_run(tmp_path, monkeypatch, _SFT_EPOCHS_NOT_A_NUMBER)
        out = _plain(result.output)

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "training -> epochs: Input should be a valid integer" in out, out


class TestThePrintedLines:
    """Exact lines, so indentation and every byte of the message are pinned."""

    def test_model_level(self, printed, tmp_path):
        path = tmp_path / "soup.yaml"
        path.write_text(_PRM_4BIT, encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)

        assert excinfo.value.code == 1
        assert printed.getvalue().splitlines() == [
            "Config validation error:",
            "",
            f"  {_QUANT_REFUSAL}",
        ]

    def test_field_level(self, printed, tmp_path):
        path = tmp_path / "soup.yaml"
        path.write_text(_SFT_REMOVE_UNUSED_COLUMNS, encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)

        assert excinfo.value.code == 1
        assert printed.getvalue().splitlines() == [
            "Config validation error:",
            "",
            f"  {_RUC_REFUSAL}",
        ]

    def test_field_level_with_a_location_the_message_does_not_repeat(self, printed, tmp_path):
        path = tmp_path / "soup.yaml"
        path.write_text(_GRPO_REMOTE_HTTP_JUDGE, encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)

        assert excinfo.value.code == 1
        assert printed.getvalue().splitlines() == [
            "Config validation error:",
            "",
            "  training -> grpo_judge: judge_model 'http://judge.example.com/m' uses "
            "unsupported scheme",
        ]


class _QuotesTheFile(BaseModel):
    """A model-level refusal whose text carries an escape byte and Rich markup,
    as one quoting a value from the file could."""

    base: str

    @model_validator(mode="after")
    def _refuse(self):
        raise ValueError(f"refused {self.base}")


class TestTheMessageAloneIsStillTerminalSafe:
    def test_escape_bytes_and_markup_from_the_file_are_neutralised(
        self, printed, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(loader, "_build_config", lambda raw: _QuotesTheFile(**raw))
        path = tmp_path / "soup.yaml"
        path.write_text(_PRM_4BIT, encoding="utf-8")
        hostile = "x" + chr(27) + "[2J[bold]y[/bold]"
        monkeypatch.setattr(
            loader, "_read_config_file", lambda *_args, **_kwargs: {"base": hostile}
        )

        with pytest.raises(SystemExit):
            loader.load_config(path)

        lines = printed.getvalue().splitlines()
        assert chr(27) not in printed.getvalue()
        assert lines[-1] == "  refused x[2J[bold]y[/bold]", lines


class TestTheExceptionTextIsUnchanged:
    """``load_config_from_string`` is the API path: its ``ValueError`` keeps
    Pydantic's text, which callers read and ``test_issue1357_*`` parses."""

    def test_model_level(self):
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(_PRM_4BIT)
        assert str(excinfo.value) == f": Value error, {_QUANT_REFUSAL}"
