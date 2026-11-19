"""Offline ingest displays output paths without changing the files."""

import json
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import ingest
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi


@pytest.mark.parametrize(
    ("suffix", "visible"),
    [
        ("plain", "plain"),
        ("[red]x", "[red]x"),
        ("x\x9b\u202e", "x"),
        (r"\[unclosed", r"\[unclosed"),
        pytest.param(
            ":warning:",
            ":warning:",
            marks=skip_colon_on_windows,
        ),
    ],
)
@pytest.mark.parametrize("outside", [False, True], ids=["written", "refused"])
def test_ingest_output_path(suffix, visible, outside, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ingest, "console", Console(width=240, height=80))
    source = Path("input.jsonl")
    original = b'{"id":"one","input":"hello","output":"world"}\n'
    source.write_bytes(original)
    output = f"out-{suffix}.jsonl"
    if outside:
        output = "../" + output
        expected = f"--output '{Path('../out-' + visible + '.jsonl')}' is outside cwd — refusing"
    else:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        expected = f"Wrote 1 traces from langfuse -> {Path('out-' + visible + '.jsonl').name}"

    result = make_cli_runner().invoke(
        app, ["ingest", "--source", "langfuse", "--logs", source.name, "--output", output]
    )

    assert result.exit_code == int(outside), (result.output, repr(result.exception))
    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    assert expected in strip_ansi(result.stdout).splitlines()
    assert source.read_bytes() == original
    if outside:
        assert not Path(output).exists()
    else:
        row = json.loads(Path(output).read_text(encoding="utf-8"))
        assert row["trace_id"] == "one" and row["prompt"] == "hello" and row["output"] == "world"
