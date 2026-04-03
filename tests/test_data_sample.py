"""Tests for soup data sample — intelligent dataset sampling."""

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def _create_jsonl(tmp_path: Path, filename: str, num_rows: int) -> Path:
    """Helper to create a JSONL file with sample data."""
    file_path = tmp_path / filename
    with open(file_path, "w", encoding="utf-8") as fh:
        for idx in range(num_rows):
            row = {
                "instruction": f"Question {idx}",
                "input": f"Context for question {idx}" * (idx + 1),
                "output": f"Answer {idx}" * (idx + 1),
            }
            fh.write(json.dumps(row) + "\n")
    return file_path


class TestSampleCLI:
    """Test soup data sample CLI command."""

    def test_random_sample_n(self, tmp_path, monkeypatch):
        """Sample N rows with random strategy."""
        monkeypatch.chdir(tmp_path)
        input_path = _create_jsonl(tmp_path, "big.jsonl", 100)
        output_path = tmp_path / "small.jsonl"
        result = runner.invoke(app, [
            "data", "sample",
            str(input_path),
            "--output", str(output_path),
            "--n", "10",
        ])
        assert result.exit_code == 0
        with open(output_path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert len(rows) == 10

    def test_random_sample_pct(self, tmp_path, monkeypatch):
        """Sample by percentage."""
        monkeypatch.chdir(tmp_path)
        input_path = _create_jsonl(tmp_path, "big.jsonl", 100)
        output_path = tmp_path / "small.jsonl"
        result = runner.invoke(app, [
            "data", "sample",
            str(input_path),
            "--output", str(output_path),
            "--pct", "10",
        ])
        assert result.exit_code == 0
        with open(output_path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert len(rows) == 10

    def test_sample_n_larger_than_dataset(self, tmp_path, monkeypatch):
        """When n > dataset size, return all rows."""
        monkeypatch.chdir(tmp_path)
        input_path = _create_jsonl(tmp_path, "small.jsonl", 5)
        output_path = tmp_path / "out.jsonl"
        result = runner.invoke(app, [
            "data", "sample",
            str(input_path),
            "--output", str(output_path),
            "--n", "100",
        ])
        assert result.exit_code == 0
        with open(output_path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert len(rows) == 5

    def test_sample_with_seed(self, tmp_path, monkeypatch):
        """Seed produces deterministic output."""
        monkeypatch.chdir(tmp_path)
        input_path = _create_jsonl(tmp_path, "data.jsonl", 50)
        out1 = tmp_path / "out1.jsonl"
        out2 = tmp_path / "out2.jsonl"

        runner.invoke(app, [
            "data", "sample", str(input_path),
            "--output", str(out1), "--n", "10", "--seed", "42",
        ])
        runner.invoke(app, [
            "data", "sample", str(input_path),
            "--output", str(out2), "--n", "10", "--seed", "42",
        ])

        with open(out1, encoding="utf-8") as fh:
            rows1 = fh.readlines()
        with open(out2, encoding="utf-8") as fh:
            rows2 = fh.readlines()
        assert rows1 == rows2

    def test_diverse_strategy(self, tmp_path, monkeypatch):
        """Diverse strategy returns requested number of samples."""
        monkeypatch.chdir(tmp_path)
        input_path = _create_jsonl(tmp_path, "data.jsonl", 50)
        output_path = tmp_path / "diverse.jsonl"
        result = runner.invoke(app, [
            "data", "sample", str(input_path),
            "--output", str(output_path),
            "--n", "10",
            "--strategy", "diverse",
        ])
        assert result.exit_code == 0
        with open(output_path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert len(rows) == 10

    def test_hard_strategy(self, tmp_path, monkeypatch):
        """Hard strategy returns requested number of samples."""
        monkeypatch.chdir(tmp_path)
        input_path = _create_jsonl(tmp_path, "data.jsonl", 50)
        output_path = tmp_path / "hard.jsonl"
        result = runner.invoke(app, [
            "data", "sample", str(input_path),
            "--output", str(output_path),
            "--n", "10",
            "--strategy", "hard",
        ])
        assert result.exit_code == 0
        with open(output_path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert len(rows) == 10

    def test_missing_input_file(self):
        """Should fail for nonexistent input file."""
        result = runner.invoke(app, [
            "data", "sample", "nonexistent.jsonl",
            "--output", "out.jsonl", "--n", "10",
        ])
        assert result.exit_code != 0

    def test_empty_input(self, tmp_path):
        """Should handle empty input file gracefully."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        output_path = tmp_path / "out.jsonl"
        result = runner.invoke(app, [
            "data", "sample", str(empty_file),
            "--output", str(output_path), "--n", "10",
        ])
        assert result.exit_code != 0

    def test_default_output_name(self, tmp_path):
        """Default output should be <input>_sampled.jsonl."""
        input_path = _create_jsonl(tmp_path, "data.jsonl", 20)
        result = runner.invoke(app, [
            "data", "sample", str(input_path), "--n", "5",
        ])
        assert result.exit_code == 0
        expected_output = tmp_path / "data_sampled.jsonl"
        assert expected_output.exists()
        with open(expected_output, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        assert len(rows) == 5

    def test_must_specify_n_or_pct(self, tmp_path):
        """Should fail if neither --n nor --pct specified."""
        input_path = _create_jsonl(tmp_path, "data.jsonl", 20)
        result = runner.invoke(app, [
            "data", "sample", str(input_path),
        ])
        assert result.exit_code != 0

    def test_invalid_strategy(self, tmp_path):
        """Unknown strategy shows error."""
        input_path = _create_jsonl(tmp_path, "data.jsonl", 20)
        result = runner.invoke(app, [
            "data", "sample", str(input_path),
            "--n", "5", "--strategy", "nonexistent",
        ])
        assert result.exit_code != 0


class TestSampleSecurity:
    """Security tests for sample command."""

    def test_sample_output_path_traversal(self, tmp_path):
        """--output with path traversal should be rejected."""
        input_path = _create_jsonl(tmp_path, "data.jsonl", 20)
        result = runner.invoke(app, [
            "data", "sample", str(input_path),
            "--output", "../../evil.jsonl", "--n", "5",
        ])
        assert result.exit_code != 0


class TestSampleStrategies:
    """Test sampling strategy functions directly."""

    def test_random_sample(self):
        from soup_cli.commands.data import _sample_random

        data = [{"text": f"row {idx}"} for idx in range(100)]
        result = _sample_random(data, 10, seed=42)
        assert len(result) == 10
        # All sampled items should be from original data
        for item in result:
            assert item in data

    def test_diverse_sample(self):
        from soup_cli.commands.data import _sample_diverse

        data = [
            {"text": "Python is a programming language" * (idx + 1)}
            for idx in range(50)
        ]
        result = _sample_diverse(data, 10, seed=42)
        assert len(result) == 10

    def test_hard_sample(self):
        from soup_cli.commands.data import _sample_hard

        # Create data with varying lengths (proxy for difficulty)
        data = [
            {"text": "word " * (idx + 1)}
            for idx in range(50)
        ]
        result = _sample_hard(data, 10)
        assert len(result) == 10
        # Hard samples should be longer (more difficult)
        avg_len = sum(len(str(row)) for row in result) / len(result)
        all_avg = sum(len(str(row)) for row in data) / len(data)
        assert avg_len >= all_avg  # hard samples are above average length
