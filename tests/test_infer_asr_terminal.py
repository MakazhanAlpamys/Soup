"""ASR warnings and receipts show filenames and diagnostics as literal text."""

import json
import os
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import infer
from soup_cli.utils import asr_metrics
from soup_cli.utils.paths import quote_path
from tests.conftest import strip_ansi

SPELLINGS = [
    pytest.param("plain", "plain", id="plain"),
    pytest.param(":rocket:", ":rocket:", id="emoji"),
    pytest.param("[red]name", "[red]name", id="markup"),
    pytest.param(r"name\[red]tag", r"name\[red]tag", id="backslash"),
    pytest.param("name\x1b\x07\x9b\u202e[end]", "name[end]", id="control-and-bidi"),
]


@pytest.fixture(autouse=True)
def isolated_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        infer, "console",
        Console(width=250, height=40, force_terminal=True, color_system="standard", no_color=False),
    )


def _run(rows, output, *, model="whisper-tiny"):
    Path("rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    return CliRunner().invoke(app, [
        "infer", "--task", "asr", "--model", model, "--device", "cpu",
        "--input", "rows.jsonl", "--output", output,
    ])


def _lines(result, exit_code=0):
    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    plain = strip_ansi(result.output)
    assert not any(char in plain for char in ("\x1b", "\x07", "\x9b", "\u202e")), repr(plain)
    return [" ".join(line.strip("│| ").split()) for line in plain.splitlines()]


def _require_filename(value):
    if os.name == "nt" and any(char in value for char in (":", "\\", "\x1b")):
        pytest.skip("POSIX filename spelling")


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_asr_skip_and_receipt_show_real_paths_literally(monkeypatch, value, shown):
    _require_filename(value)
    Path("good.wav").touch()
    called = []

    def transcribe(path):
        called.append(Path(path).name)
        return "hello"

    monkeypatch.setattr(infer, "_ASR_TRANSCRIBER_OVERRIDE", transcribe)
    bad = f"outside-{value}.wav"
    output = f"output-{value}.jsonl"
    rows = [{"audio": f"../{bad}"}, {"audio": "good.wav", "text": "hello"}]
    lines = _lines(_run(rows, output))

    assert f"Skipped {bad!r}: audio path {bad!r} must stay under the audio dir" in lines
    assert f"Transcribed 1 clip(s) -> output-{shown}.jsonl (1 skipped)" in lines
    assert "Corpus WER: 0.000" in lines
    assert called == ["good.wav"]
    assert json.loads(Path(output).read_text(encoding="utf-8")) == {
        "audio": "good.wav", "transcription": "hello",
        "reference": "hello", "wer": 0.0, "cer": 0.0,
    }


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_asr_decoder_error_is_literal_and_all_skipped_keeps_output(monkeypatch, value, shown):
    def fail(path):
        raise OSError(f"Decoder failed for {value}")

    monkeypatch.setattr(infer, "_ASR_TRANSCRIBER_OVERRIDE", fail)
    output = Path("output.jsonl")
    previous = b'{"audio": "old.wav", "transcription": "old"}\n'
    output.write_bytes(previous)

    lines = _lines(_run([{"audio": "clip.wav"}], str(output)), exit_code=2)

    assert lines == [
        f"Skipped 'clip.wav': Decoder failed for {shown}",
        "No clips transcribed (1 skipped). Check --audio-dir and the input audio paths.",
    ]
    assert output.read_bytes() == previous


@pytest.mark.parametrize("value,_shown", SPELLINGS)
def test_asr_metric_skip_shows_filename_and_keeps_transcript(monkeypatch, value, _shown):
    _require_filename(value)
    bad = f"clip-{value}.wav"
    Path(bad).touch()
    monkeypatch.setattr(infer, "_ASR_TRANSCRIBER_OVERRIDE", lambda path: "hello")
    size = asr_metrics._MAX_RAW_CHARS + 1

    lines = _lines(_run([{"audio": bad, "text": "x" * size}], "output.jsonl"))

    warning = (
        f"Metric skipped for {bad!r}: input too long "
        f"({size} chars > {size - 1}); split the input"
    )
    assert warning in lines
    assert "Transcribed 1 clip(s) -> output.jsonl" in lines
    assert json.loads(Path("output.jsonl").read_text(encoding="utf-8")) == {
        "audio": bad, "transcription": "hello",
    }


def _assert_red_refusal(result, exit_code, message):
    assert _lines(result, exit_code) == [message]
    styled = result.stdout.strip().replace("\x1b[0m\x1b[31m", "")
    assert styled == f"\x1b[31m{message}\x1b[0m"


@pytest.mark.parametrize("value,_shown", SPELLINGS)
def test_asr_refuses_real_outside_output_with_literal_path(tmp_path, monkeypatch, value, _shown):
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    output = f"../outside-{value}.jsonl"
    result = _run([{"audio": "clip.wav"}], output)

    # Path refusals already spell ASCII/C1 controls as quoted escape sequences.
    quoted = quote_path(output).replace("\u202e", "")
    _assert_red_refusal(result, 1, f"--output {quoted} must stay under cwd")
    assert list(tmp_path.iterdir()) == [workdir]


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_asr_dependency_refusal_shows_literal_error(monkeypatch, value, shown):
    def missing_dependency(*args, **kwargs):
        raise ImportError(f"Could not import decoder {value}")

    monkeypatch.setattr(infer, "_build_asr_transcriber", missing_dependency)
    result = _run([{"audio": "clip.wav"}], "output.jsonl")

    _assert_red_refusal(result, 1, f"Could not import decoder {shown}")
    assert not Path("output.jsonl").exists()


@pytest.mark.parametrize("value,_shown", SPELLINGS)
def test_asr_rejects_real_non_whisper_config_with_literal_model_name(value, _shown):
    pytest.importorskip("transformers")
    _require_filename(value)
    model = Path(f"gpt2-{value}")
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")

    result = _run([{"audio": "clip.wav"}], "output.jsonl", model=str(model))

    message = (
        f"task='asr' requires a Whisper base model, but {str(model)!r} has "
        "model_type='gpt2'. Use e.g. openai/whisper-tiny / whisper-base / whisper-large-v3."
    )
    _assert_red_refusal(result, 2, message)
    assert not Path("output.jsonl").exists()
