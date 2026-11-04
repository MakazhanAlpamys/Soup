"""Tests for count retries and empty round bounds in soup data generate (#2415)."""

import json
from unittest.mock import patch

from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()


def test_generate_retries_until_target_count(tmp_path, monkeypatch):
    """The generation loop should keep requesting batches until --count is satisfied."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.jsonl"
    calls = []
    counter = 0

    def fake_generate_batch(**kwargs):
        nonlocal counter
        calls.append(kwargs["count"])
        res = [
            {"instruction": f"Question {counter + i}", "output": f"Answer {counter + i}"}
            for i in range(2)
        ]
        counter += 2
        return res

    with patch("soup_cli.commands.generate._generate_batch", side_effect=fake_generate_batch):
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "Test prompt",
                "--count",
                "5",
                "--batch-size",
                "5",
                "--requests-per-minute",
                "6000",
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    # 5 requested, 2 per call -> calls were: [5, 3, 1]
    assert calls == [5, 3, 1]

    with open(output_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 5


def test_generate_exits_1_when_provider_repeatedly_empty(tmp_path, monkeypatch):
    """Three consecutive empty rounds must halt generation and exit with code 1."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.jsonl"
    calls = 0

    def fake_generate_batch_empty(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"instruction": "Q1", "output": "A1"}]
        return []

    with patch("soup_cli.commands.generate._generate_batch", side_effect=fake_generate_batch_empty):
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "Test prompt",
                "--count",
                "5",
                "--batch-size",
                "5",
                "--requests-per-minute",
                "6000",
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert calls == 4

    output_text = " ".join(strip_ansi(result.output).split())
    expected_msg = (
        "Generation halted: received no valid, unique examples after 3 consecutive attempts "
        "(1/5 generated)"
    )
    assert expected_msg in output_text

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1


def test_generate_resets_consecutive_empty_rounds_on_success(tmp_path, monkeypatch):
    """An empty round followed by a successful round must reset the consecutive empty counter."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.jsonl"
    calls = 0

    def fake_generate_intermittent(**kwargs):
        nonlocal calls
        calls += 1
        # Calls 1, 3, 6, 7 return an example; calls 2, 4, 5 return empty list
        if calls in (1, 3, 6, 7):
            return [{"instruction": f"Q{calls}", "output": f"A{calls}"}]
        return []

    patch_fn = "soup_cli.commands.generate._generate_batch"
    with patch(patch_fn, side_effect=fake_generate_intermittent):
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "Test prompt",
                "--count",
                "4",
                "--batch-size",
                "2",
                "--requests-per-minute",
                "6000",
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls == 7

    with open(output_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 4


def _run_generate(provider, output_file, count, batch_size):
    with patch("soup_cli.commands.generate._generate_batch", side_effect=provider):
        return runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "Test prompt",
                "--count",
                str(count),
                "--batch-size",
                str(batch_size),
                "--requests-per-minute",
                "6000",
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )


def _rows(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_generate_counts_duplicates_toward_count(tmp_path, monkeypatch):
    """The missing rows are asked for again; repeats count toward --count, exit 0."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.jsonl"
    calls = []

    def provider(**kwargs):
        calls.append(kwargs["count"])
        return [{"instruction": f"Question {i}", "output": "Answer"} for i in range(2)]

    result = _run_generate(provider, output_file, count=5, batch_size=5)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls == [5, 3, 1]
    output_text = " ".join(strip_ansi(result.output).split())
    assert "Generated 2 of 5 requested examples" in output_text
    assert "Duplicates: 3 removed" in output_text
    assert [row["instruction"] for row in _rows(output_file)] == ["Question 0", "Question 1"]


def test_generate_asks_again_for_invalid_rows(tmp_path, monkeypatch):
    """A row that fails validation does not count toward --count: it is asked for again."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.jsonl"
    calls = []

    def provider(**kwargs):
        calls.append(kwargs["count"])
        if len(calls) == 1:
            return [{"instruction": "no output"}, {"instruction": "Q0", "output": "A0"}]
        return [{"instruction": "Q1", "output": "A1"}]

    result = _run_generate(provider, output_file, count=2, batch_size=2)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls == [2, 1]
    assert [row["instruction"] for row in _rows(output_file)] == ["Q0", "Q1"]


def test_generate_duplicates_do_not_hide_an_under_delivery(tmp_path, monkeypatch):
    """Duplicates explain only their own rows: a provider that stops delivering exits 1."""
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "output.jsonl"
    calls = []

    def provider(**kwargs):
        calls.append(kwargs["count"])
        if len(calls) == 1:
            return [{"instruction": "Q0", "output": "A0"}] * 2
        return []

    result = _run_generate(provider, output_file, count=4, batch_size=4)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert calls == [4, 2, 2, 2]
    output_text = " ".join(strip_ansi(result.output).split())
    assert "after 3 consecutive attempts (1/4 generated)" in output_text
    assert [row["instruction"] for row in _rows(output_file)] == ["Q0"]
