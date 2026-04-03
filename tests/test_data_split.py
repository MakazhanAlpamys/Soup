"""Tests for soup data split — train/val/test splitting."""

import json
from pathlib import Path


def _make_jsonl(path: Path, count: int, field: str = "category"):
    """Create a JSONL file with count rows, optional category field."""
    lines = []
    for idx in range(count):
        row = {"text": f"sample {idx}", field: f"cat_{idx % 3}"}
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_jsonl(path: Path) -> list:
    """Read JSONL file and return list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ─── CLI Tests ────────────────────────────────────────────────────────────


class TestDataSplitCLI:
    """Test soup data split command via CLI."""

    def test_split_in_help(self):
        """Data help should mention split subcommand."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "--help"])
        assert "split" in result.output.lower()

    def test_split_basic_ratio(self, tmp_path):
        """Basic split with --val 20 should produce two files."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "data.jsonl"
        _make_jsonl(data_file, 100)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "split", str(data_file), "--val", "20"],
        )
        assert result.exit_code == 0

        train_file = tmp_path / "data_train.jsonl"
        val_file = tmp_path / "data_val.jsonl"
        assert train_file.exists()
        assert val_file.exists()

        train_rows = _read_jsonl(train_file)
        val_rows = _read_jsonl(val_file)
        assert len(train_rows) + len(val_rows) == 100
        assert len(val_rows) == 20

    def test_split_train_val_test(self, tmp_path):
        """Split with --val 10 --test 10 should produce three files."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "data.jsonl"
        _make_jsonl(data_file, 100)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "split", str(data_file), "--val", "10", "--test", "10"],
        )
        assert result.exit_code == 0

        train_file = tmp_path / "data_train.jsonl"
        val_file = tmp_path / "data_val.jsonl"
        test_file = tmp_path / "data_test.jsonl"
        assert train_file.exists()
        assert val_file.exists()
        assert test_file.exists()

        train_rows = _read_jsonl(train_file)
        val_rows = _read_jsonl(val_file)
        test_rows = _read_jsonl(test_file)
        assert len(train_rows) + len(val_rows) + len(test_rows) == 100
        assert len(val_rows) == 10
        assert len(test_rows) == 10

    def test_split_absolute_counts(self, tmp_path):
        """Split with --absolute should use absolute counts."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "data.jsonl"
        _make_jsonl(data_file, 100)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "split", str(data_file), "--val", "15", "--absolute"],
        )
        assert result.exit_code == 0

        val_file = tmp_path / "data_val.jsonl"
        val_rows = _read_jsonl(val_file)
        assert len(val_rows) == 15

    def test_split_seed_reproducible(self, tmp_path):
        """Same seed should produce the same split."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "data.jsonl"
        _make_jsonl(data_file, 50)

        runner = CliRunner()

        # First run
        result1 = runner.invoke(
            app,
            ["data", "split", str(data_file), "--val", "20", "--seed", "42"],
        )
        assert result1.exit_code == 0
        val1 = _read_jsonl(tmp_path / "data_val.jsonl")

        # Second run (overwrite)
        result2 = runner.invoke(
            app,
            ["data", "split", str(data_file), "--val", "20", "--seed", "42"],
        )
        assert result2.exit_code == 0
        val2 = _read_jsonl(tmp_path / "data_val.jsonl")

        assert val1 == val2

    def test_split_file_not_found(self, tmp_path):
        """Should error if input file doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "split", str(tmp_path / "nope.jsonl"), "--val", "10"],
        )
        assert result.exit_code != 0

    def test_split_no_val_or_test(self, tmp_path):
        """Should error if neither --val nor --test is specified."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "data.jsonl"
        _make_jsonl(data_file, 50)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "split", str(data_file)],
        )
        assert result.exit_code != 0

    def test_split_empty_dataset(self, tmp_path):
        """Should error on empty dataset."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "data.jsonl"
        data_file.write_text("", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "split", str(data_file), "--val", "10"],
        )
        assert result.exit_code != 0


# ─── Edge Case Tests ─────────────────────────────────────────────────────


class TestDataSplitEdgeCases:
    """Test edge cases for data split."""

    def test_split_val_pct_too_large(self, tmp_path):
        """val=90 should work but leave 10% for train."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "data.jsonl"
        _make_jsonl(data_file, 100)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "split", str(data_file), "--val", "90"],
        )
        assert result.exit_code == 0
        train_rows = _read_jsonl(tmp_path / "data_train.jsonl")
        assert len(train_rows) == 10

    def test_split_stratified(self, tmp_path):
        """Stratified split should preserve category distribution."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "data.jsonl"
        _make_jsonl(data_file, 90)  # 30 per category (cat_0, cat_1, cat_2)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data", "split", str(data_file),
                "--val", "30", "--stratify", "category",
            ],
        )
        assert result.exit_code == 0

        val_rows = _read_jsonl(tmp_path / "data_val.jsonl")
        # Each category should have ~10 samples (30% of 30)
        categories = {}
        for row in val_rows:
            cat = row.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        # Each category should be represented
        assert len(categories) == 3
        # Should be roughly equal (~9-10 each from 30, rounding allowed)
        for count in categories.values():
            assert 8 <= count <= 11

    def test_split_absolute_val_exceeds_dataset(self, tmp_path):
        """Should error if absolute val count exceeds dataset size."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "data.jsonl"
        _make_jsonl(data_file, 10)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "split", str(data_file), "--val", "20", "--absolute"],
        )
        assert result.exit_code != 0


# ─── Security Tests ──────────────────────────────────────────────────────


class TestDataSplitSecurity:
    """Security tests for data split."""

    def test_output_files_in_same_directory(self, tmp_path):
        """Output files should be created in the same directory as input."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data_file = tmp_path / "data.jsonl"
        _make_jsonl(data_file, 20)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "split", str(data_file), "--val", "20"],
        )
        assert result.exit_code == 0

        # Output should be in same dir as input
        assert (tmp_path / "data_train.jsonl").exists()
        assert (tmp_path / "data_val.jsonl").exists()
