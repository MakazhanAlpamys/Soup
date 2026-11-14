"""Canary CLI display is literal; secrets, datasets and verdicts stay original."""

import io
import json
import os
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import data_canary as command
from soup_cli.utils.canary import (
    Canary,
    build_canary_report,
    canary_report_to_dict,
    canary_rows,
    generate_canaries,
    interleave_canary_rows,
    write_manifest,
)
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

VALUES = [
    pytest.param("ordinary", "ordinary", id="plain-control"),
    pytest.param(":rocket:", ":rocket:", id="emoji-alias"),
    pytest.param("x[red]name", "x[red]name", id="markup"),
    pytest.param(r"x\[unfinished", r"x\[unfinished", id="backslash"),
    pytest.param("x\x1b\x07\x9b\u202e[end]", "x[end]", id="controls"),
]
FILE_VALUES = [
    pytest.param(*case.values, id=case.id,
                 marks=skip_colon_on_windows if ":" in case.values[0] else ())
    for case in VALUES
]


def _plain(text):
    text = strip_ansi(text)
    assert not any(c in text for c in ("\x1b", "\x07", "\x9b", "\u202e")), repr(text)
    return " ".join(text.split())


def _filename(value):
    return "item-" + value.replace("\x1b", "").replace("\x07", "")


@pytest.fixture
def printed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stream = io.StringIO()
    monkeypatch.setattr(command, "console", Console(
        file=stream, width=700, height=200, force_terminal=True,
        color_system="standard", no_color=False,
    ))
    Path("input.jsonl").write_text('{"instruction":"say hi","output":"hi"}\n')
    return stream


@pytest.mark.parametrize("value,shown", FILE_VALUES)
def test_canary_missing_input_is_literal(printed, value, shown):
    path = Path(_filename(value) + ".jsonl")
    result = CliRunner().invoke(app, [
        "data", "canary", "insert", str(path), "-o", "out.jsonl", "--manifest", "manifest.json",
    ])
    assert result.exit_code == 1
    assert _plain(printed.getvalue()) == f"File not found: {Path(_filename(shown) + '.jsonl')}"
    assert not Path("manifest.json").exists() and not Path("out.jsonl").exists()


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("label", ["Output", "Manifest"])
def test_canary_outside_path_refusal_is_literal_without_writing(
    printed, label, value, shown
):
    outside = f"../outside-{value}.jsonl"
    args = ["data", "canary", "insert", "input.jsonl", "-o", "out.jsonl",
            "--manifest", "manifest.json"]
    args[args.index("-o" if label == "Output" else "--manifest") + 1] = outside
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert _plain(printed.getvalue()) == (
        f"{label} path is outside the working directory: ../outside-{shown}.jsonl"
    )
    assert not Path("manifest.json").exists() and not Path("out.jsonl").exists()


@pytest.mark.parametrize("value,shown", VALUES)
def test_canary_format_refusal_is_literal(printed, value, shown):
    result = CliRunner().invoke(app, [
        "data", "canary", "insert", "input.jsonl", "-o", "out.jsonl",
        "--manifest", "manifest.json", "--format", value,
    ])
    assert result.exit_code == 1
    assert _plain(printed.getvalue()) == (
        f"canary insert cannot write '{shown}' rows. Supported: alpaca, sharegpt, chatml"
    )


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("kind", ["generate", "manifest-write", "dataset-write", "manifest-read",
                                  "controls", "model", "report"])
def test_canary_failures_show_literal_errors_and_preserve_input(
    printed, monkeypatch, value, shown, kind
):
    before = Path("input.jsonl").read_bytes()
    write_manifest([Canary("carrier", "secret")], "manifest.json", "alpaca")
    original_manifest = Path("manifest.json").read_bytes()

    def fail(*args, **kwargs):
        raise ValueError(f"Failure {value}")

    monkeypatch.setattr(command, "_load_pair", lambda *a: (object(), object(), "cpu"))
    monkeypatch.setattr(command, "compute_pair_losses", lambda *a, **k: [2.0] * 5)
    site = {"generate": "generate_canaries", "manifest-write": "write_manifest",
            "dataset-write": "atomic_write_text", "manifest-read": "load_manifest",
            "controls": "generate_controls", "model": "_load_pair",
            "report": "build_canary_report"}[kind]
    monkeypatch.setattr(command, site, fail)
    if kind in {"generate", "manifest-write", "dataset-write"}:
        args = ["data", "canary", "insert", "input.jsonl", "-o", "out.jsonl",
                "--manifest", "new-manifest.json", "--count", "1"]
    else:
        args = ["data", "canary", "check", "--manifest", "manifest.json", "--base", "base",
                "--controls", "4"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, (result.output, result.exception)
    text = _plain(printed.getvalue())
    if kind == "dataset-write":
        assert text == (f"Could not write out.jsonl: Failure {shown} new-manifest.json was "
                        "already written and now describes canaries that were NOT inserted "
                        "— delete it or re-run.")
        assert Path("new-manifest.json").is_file()
        assert "\x1b[33m" in printed.getvalue()
    else:
        assert text == ("Could not load model: " if kind == "model" else "") + f"Failure {shown}"
    assert "\x1b[31m" in printed.getvalue()
    assert Path("input.jsonl").read_bytes() == before
    assert Path("manifest.json").read_bytes() == original_manifest
    assert not Path("out.jsonl").exists()


@pytest.mark.parametrize("value,shown", FILE_VALUES)
def test_canary_insert_receipt_keeps_literal_paths_and_real_artifacts(printed, value, shown):
    output = Path("out-" + _filename(value) + ".jsonl")
    manifest = Path("manifest-" + _filename(value) + ".json")
    for path in [output, manifest]:
        path.parent.mkdir(parents=True, exist_ok=True)
    original = Path("input.jsonl").read_bytes()
    rows = [json.loads(original)]
    result = CliRunner().invoke(app, [
        "data", "canary", "insert", "input.jsonl", "-o", str(output),
        "--manifest", str(manifest), "--count", "1", "--seed", "7",
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    text = _plain(printed.getvalue())
    for label, path in [("Output", output), ("Manifest", manifest)]:
        assert f"{label}: {str(path).replace(_filename(value), _filename(shown))}" in text
    canaries = generate_canaries(count=1, seed=7)
    mixed, positions = interleave_canary_rows(rows, canary_rows(canaries, "alpaca"), seed=7)
    assert [json.loads(line) for line in output.read_text().splitlines()] == mixed
    assert json.loads(manifest.read_text())["canaries"] == [
        {"carrier": canaries[0].carrier, "secret": canaries[0].secret, "row": positions[0]}
    ]
    assert Path("input.jsonl").read_bytes() == original
    if os.name != "nt":
        assert manifest.stat().st_mode & 0o777 == 0o600
    assert "\x1b[32mInserted " in printed.getvalue()


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("memorized", [False, True])
def test_canary_exposure_display_keeps_original_secret_scoring_and_report(
    printed, monkeypatch, value, shown, memorized
):
    canary = Canary(f"carrier {value}", f"secret {value}")
    write_manifest([canary], "manifest.json", "alpaca")
    original = Path("manifest.json").read_bytes()
    calls = []

    def load(base, adapter, device):
        calls.append((base, adapter, device))
        return "model", "tokenizer", "cpu"

    def score(model, tokenizer, pairs, *, device):
        assert (model, tokenizer, device) == ("model", "tokenizer", "cpu")
        assert pairs[0] == (canary.carrier, canary.secret)
        assert all(secret != canary.secret for _, secret in pairs[1:])
        return [0.0 if memorized else 2.0] + [1.0] * 4

    monkeypatch.setattr(command, "_load_pair", load)
    monkeypatch.setattr(command, "compute_pair_losses", score)
    result = CliRunner().invoke(app, [
        "data", "canary", "check", "--manifest", "manifest.json", "--base", value,
        "--adapter", value, "--device", "cpu", "--controls", "4", "-o", "report.json",
    ])
    assert result.exit_code == (2 if memorized else 0), (result.output, result.exception)
    text = _plain(printed.getvalue())
    assert f"secret {shown}" in text
    assert f"Verdict: {'MAJOR' if memorized else 'OK'}" in text
    expected = build_canary_report([0.0 if memorized else 2.0], [1.0] * 4, [canary.secret])
    assert json.loads(Path("report.json").read_text()) == canary_report_to_dict(expected)
    assert calls == [(value, value, "cpu")]
    assert Path("manifest.json").read_bytes() == original
    if os.name != "nt":
        assert Path("report.json").stat().st_mode & 0o777 == 0o600
