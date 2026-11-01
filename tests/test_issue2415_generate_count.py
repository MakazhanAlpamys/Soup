import json
from unittest.mock import patch

from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def test_generate_retries_until_target_count(tmp_path, monkeypatch):
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
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )

    assert result.exit_code == 0, result.output
    # 5 requested, 2 per call -> calls were: [5, 3, 1]
    assert len(calls) == 3
    assert calls == [5, 3, 1]

    with open(output_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 5


def test_generate_exits_1_when_provider_repeatedly_empty(tmp_path, monkeypatch):
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
                "--provider",
                "openai",
                "--api-key",
                "fake-key",
                "--output",
                str(output_file),
            ],
        )

    assert result.exit_code == 1
    assert "Generation halted" in result.output
    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
