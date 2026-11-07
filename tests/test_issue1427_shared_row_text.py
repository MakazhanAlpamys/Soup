"""Quality scans must see every supported row shape and report missing text."""

import json
import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import data_score
from tests.conftest import strip_ansi

PAYLOAD = "idiot alice@example.org"
CORPUS = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
SHAPES = ["text", "chat", "alpaca", "dpo", "kto", "sharegpt"]


def _row(shape, text):
    return {
        "text": {"text": text},
        "chat": {"messages": [{"role": "assistant", "content": text}]},
        "alpaca": {"instruction": "", "input": "", "output": text},
        "dpo": {"prompt": "", "chosen": "", "rejected": text},
        "kto": {"prompt": "", "completion": [{"role": "assistant", "content": text}],
                "label": True},
        "sharegpt": {"conversations": [{"from": "gpt", "value": text}]},
    }[shape]


def _file(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _run(command, *extra):
    return CliRunner().invoke(app, ["data", command, "--input", "rows.jsonl", *extra])


def _plain(result):
    return " ".join(strip_ansi(result.output).replace("│", " ").split())


@pytest.fixture(autouse=True)
def local_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Exercise Soup's unchanged regex detector, without optional Presidio models.
    monkeypatch.setattr(data_score, "_presidio_pii", lambda text: None)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("command", ["pii", "toxicity", "score"])
def test_quality_commands_scan_all_formats(tmp_path, shape, command):
    row = _row(shape, PAYLOAD)
    _file(tmp_path / "rows.jsonl", [row])
    result = _run(command)
    assert result.exit_code == 0, (result.output, result.exception)
    if command == "score":
        assert re.search(r"PII flagged\s+1", _plain(result))
        assert re.search(r"Abuse-keyword flagged\s+1", _plain(result))
    else:
        assert re.search(r"Flagged:\s*1", _plain(result))
        output = json.loads((tmp_path / f"{command}.jsonl").read_text(encoding="utf-8"))
        assert all(output[key] == value for key, value in row.items())
        if command == "toxicity":
            assert output["_abuse_keyword_score"] == data_score.score_abuse_keywords(PAYLOAD)
        else:
            assert output["_pii_hits"] == data_score.detect_pii(PAYLOAD)


@pytest.mark.parametrize("command", ["score", "decontaminate"])
@pytest.mark.parametrize("shape", SHAPES + ["prompt"])
def test_benchmark_formats_are_used(tmp_path, command, shape):
    _file(tmp_path / "rows.jsonl", [{"text": CORPUS}, {"text": "unrelated content"}])
    row = {"prompt": CORPUS} if shape == "prompt" else _row(shape, CORPUS)
    _file(tmp_path / "benchmark.jsonl", [row])
    result = _run(command, "--benchmark-file", "benchmark.jsonl")
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Loaded 1 operator-supplied benchmark texts" in _plain(result)
    if command == "score":
        assert re.search(r"Decontaminated\s+1", _plain(result))
    else:
        assert re.search(r"Removed:\s*1", _plain(result))


@pytest.mark.parametrize(
    "command", ["pii", "toxicity", "score", "decontaminate", "langdetect", "educational"],
)
@pytest.mark.parametrize("mixed", [False, True])
def test_missing_text_is_reported_before_clean_results(tmp_path, command, mixed):
    rows = [{"unknown": "unscanned"}, {"text": "  "}]
    if mixed:
        rows.append({"text": PAYLOAD})
    _file(tmp_path / "rows.jsonl", rows)
    _file(tmp_path / "benchmark.jsonl", [{"text": CORPUS}])
    extra = ["--benchmark-file", "benchmark.jsonl"] if command == "decontaminate" else []
    result = _run(command, *extra)
    output = _plain(result)
    assert "no extractable text" in output
    if mixed:
        assert result.exit_code == 0, (result.output, result.exception)
        assert "2 of 3 input rows" in output and "Warning:" in output
    else:
        assert result.exit_code == 1
        assert "2 input rows" in output
        assert "Flagged:" not in output and "Data Scorecard" not in output
        assert not list(tmp_path.glob("pii.jsonl"))


@pytest.mark.parametrize("mixed", [False, True])
def test_decontamination_reports_missing_benchmark_text(tmp_path, mixed):
    _file(tmp_path / "rows.jsonl", [{"text": CORPUS}])
    rows = [{"unknown": "unscanned"}]
    if mixed:
        rows.append({"prompt": CORPUS})
    _file(tmp_path / "benchmark.jsonl", rows)
    result = _run("decontaminate", "--benchmark-file", "benchmark.jsonl")
    assert "no extractable text" in _plain(result)
    assert result.exit_code == (0 if mixed else 1)
    if mixed:
        assert "1 of 2 benchmark rows" in _plain(result)
        assert re.search(r"Removed:\s*1", _plain(result))


@pytest.mark.parametrize("content", ["", "\n\n", "not json\n[1, 2]\n"])
@pytest.mark.parametrize(
    "command", ["pii", "toxicity", "score", "langdetect", "educational", "decontaminate"],
)
def test_input_without_rows_is_refused(tmp_path, command, content):
    (tmp_path / "rows.jsonl").write_text(content, encoding="utf-8")
    _file(tmp_path / "benchmark.jsonl", [{"text": CORPUS}])
    extra = ["--benchmark-file", "benchmark.jsonl"] if command == "decontaminate" else []
    result = _run(command, *extra)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "no extractable text in 0 input rows" in _plain(result)


def test_shared_extraction_retains_all_fields_and_expectations_alias():
    from soup_cli.utils import expectations, row_text

    row = {"instruction": "instruction", "input": "input", "output": "output",
           "prompt": "prompt", "chosen": "chosen", "rejected": "rejected",
           "anchor": "anchor", "positive": ["positive"], "negative": "negative",
           "tool_calls": [{"function": {"arguments": '{"email":"alice@example.org"}'}}]}
    text = row_text.extract_row_text(row)
    assert expectations._extract_row_text is row_text.extract_row_text
    assert data_score.extract_row_text(row) == text
    for word in ("instruction", "input", "output", "prompt", "chosen", "rejected",
                 "anchor", "positive", "negative", "alice@example.org"):
        assert word in text
