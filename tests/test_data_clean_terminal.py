"""Dataset-cleaning paths and failures remain literal without changing stored data."""

import json
import sys
from pathlib import Path, PureWindowsPath

import pytest
from rich.console import Console

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()
VALUES = ["plain", "x[red]n", "x\\[n", "n\\\\", "x\x9b\u202en"]
NAMES = VALUES + [
    pytest.param(
        "n:warning:",
        marks=pytest.mark.skipif(
            sys.platform == "win32", reason="colon is not valid in a Windows filename"
        ),
    )
]
ROW = {"instruction": "Say hi", "output": "Hello"}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr("soup_cli.commands.data_clean.console", Console(width=500, height=100))
    Path("input.jsonl").write_text(json.dumps(ROW) + "\n", encoding="utf-8")


def plain(result):
    return " ".join(strip_ansi(result.output).split())


@pytest.mark.parametrize("value", NAMES)
def test_missing_input_is_named_as_typed(value):
    path = Path(f"missing-{value}.jsonl")
    result = runner.invoke(app, ["data", "clean", str(path)])
    assert result.exit_code == 1
    assert plain(result) == f"File not found: {str(path).replace(chr(155) + chr(8238), '')}"
    assert not path.exists()


@pytest.mark.parametrize("value", NAMES)
@pytest.mark.parametrize("kind", ["input-outside", "output-outside", "same", "exists", "saved"])
def test_real_paths_preserve_input_and_output_bytes(value, kind):
    source = Path("input.jsonl")
    target = Path(f"out-{value}-rows.jsonl")
    if kind == "input-outside":
        source = Path("..") / target
    elif kind == "output-outside":
        target = Path("..") / target
    elif kind == "same":
        source = target
    assert PureWindowsPath(str(source)).suffix == ".jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "exists":
        target.write_bytes(b"keep output\n")
    original = source.read_bytes()

    result = runner.invoke(app, ["data", "clean", str(source), "-o", str(target)])

    assert source.read_bytes() == original
    shown = str(source if kind == "input-outside" else target).replace("\x9b\u202e", "")
    message = {
        "input-outside": f"Input path must be under the current working directory: {shown}",
        "output-outside": f"Output path must be under the current working directory: {shown}",
        "same": f"Output path cannot be the same as input path: {shown} "
        "(soup data clean never modifies its input file in place)",
        "exists": f"Output file already exists: {shown} (use --force / -f to overwrite)",
        "saved": f"✓ Cleaned dataset saved to: {shown} (1 rows)",
    }[kind]
    assert message in plain(result)
    assert result.exit_code == (0 if kind == "saved" else 1), result.output
    if kind == "exists":
        assert target.read_bytes() == b"keep output\n"
    elif kind == "saved":
        assert json.loads(target.read_text(encoding="utf-8")) == ROW
    elif kind != "same":
        assert not target.exists()


@pytest.mark.parametrize("value", VALUES + ["n:warning:", "[/]n"])
@pytest.mark.parametrize("stage", ["load", "write", "detect"])
def test_dataset_errors_and_json_warnings_are_literal(monkeypatch, value, stage):
    def fail(*args, **kwargs):
        raise ValueError(value)

    function = {"load": "load_raw_data", "write": "atomic_write_text", "detect": "detect_format"}
    monkeypatch.setattr(f"soup_cli.commands.data_clean.{function[stage]}", fail)
    original = Path("input.jsonl").read_bytes()
    args = ["data", "clean", "input.jsonl", "-o", "output.jsonl"]
    if stage == "detect":
        args += ["--json", "--dry-run"]

    result = runner.invoke(app, args)

    assert Path("input.jsonl").read_bytes() == original
    assert not Path("output.jsonl").exists()
    shown = value.replace("\x9b\u202e", "")
    if stage == "detect":
        assert result.exit_code == 0
        assert shown in strip_ansi(result.stderr)
        assert json.loads(result.stdout)["output_rows"] == 1
    else:
        label = "Failed to load dataset:" if stage == "load" else "Failed to write output file:"
        assert result.exit_code == 1
        assert f"{label} {shown}" in plain(result)


def test_receipt_and_refusal_keep_their_label_styles(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.data_clean.console",
        Console(
            width=500, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )
    good = runner.invoke(app, ["data", "clean", "input.jsonl"])
    bad = runner.invoke(app, ["data", "clean", "missing.jsonl"])
    assert good.exit_code == 0 and bad.exit_code == 1
    assert "\x1b[1;32m✓ Cleaned dataset saved to:\x1b[0m" in good.output
    assert "\x1b[31mFile not found:\x1b[0m missing.jsonl" in bad.output
