"""All file-collision refusals use one terminal-safe rendering contract."""

import os

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import paths
from soup_cli.utils.terminal import strip_control
from tests.conftest import strip_ansi
from tests.test_data_output_aliases import _CASES, _invoke_refused, _setup


def _plain(text):
    return " ".join(strip_ansi(text).split())


@pytest.mark.parametrize("case", [case for case in _CASES if case.command != "recipe"],
                         ids=lambda case: case.command)
def test_commands_share_the_same_refusal_message(case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = _setup(case)
    original_quote = paths.quote_path

    # Real file identity checks, with external text containing terminal controls
    # and Rich markup in the diagnostic. No unsafe filename is needed on Windows.
    def quote(path):
        return original_quote(path) + " [bold]literal[/bold]\x1b\x07\u202e"

    monkeypatch.setattr(paths, "quote_path", quote)
    output = case.input_name
    result = _invoke_refused(monkeypatch, case, case.argv(case.input_name, output))
    given_input = os.path.abspath(case.input_name) if case.absolute_input else case.input_name
    expected = (
        f"{case.output_field} {quote(output)} is the same file as {case.input_field} "
        f"{quote(given_input)}; choose a different {case.output_field}"
    )
    # Rich may fold an absolute path in the middle of a component.
    assert "".join(_plain(result.output).split()) == "".join(
        _plain(strip_control(expected)).split()
    )
    assert "\x07" not in result.output and "\u202e" not in result.output
    for name, payload in before.items():
        assert (tmp_path / name).read_bytes() == payload


@pytest.mark.parametrize("command", ["decontaminate", "toxicity", "langdetect", "pii",
                                     "educational", "ingest"])
def test_score_and_ingest_use_the_same_refusal(command, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = b'{"text":"original input"}\n'
    (tmp_path / "in.jsonl").write_bytes(payload)
    if command == "ingest":
        args = ["ingest", "--source", "langfuse", "--logs", "in.jsonl", "-o", "in.jsonl"]
        input_label = "--logs"
    else:
        args = ["data", command, "--input", "in.jsonl", "-o", "in.jsonl"]
        input_label = "--input"
        if command == "decontaminate":
            args += ["--benchmark-file", "in.jsonl"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert _plain(result.output) == (
        f"--output 'in.jsonl' is the same file as {input_label} 'in.jsonl'; "
        "choose a different --output"
    )
    assert (tmp_path / "in.jsonl").read_bytes() == payload
