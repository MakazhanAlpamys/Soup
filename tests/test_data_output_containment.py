"""Tests for output path working directory containment: convert, merge, filter (#2966)."""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


def _plain(output: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output."""
    return " ".join(strip_ansi(output).split())


def _make_sample_dataset(path: Path) -> None:
    rows = [
        {"instruction": "What is AI?", "output": "Artificial Intelligence."},
        {"instruction": "Define ML.", "output": "Machine Learning."},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestDataConvertContainment:
    """Verify soup data convert enforces is_under_cwd on output paths."""

    def test_convert_relative_outside_rejected(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "convert", str(data_file), "--to", "sharegpt", "-o", "../escaped.jsonl"],
        )
        assert result.exit_code == 1
        plain = _plain(result.output)
        assert "Output path is outside the working directory:" in plain
        assert not (tmp_path / "escaped.jsonl").exists()

    def test_convert_absolute_outside_rejected(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)
        outside_file = tmp_path / "outside.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "convert", str(data_file), "--to", "sharegpt", "-o", str(outside_file)],
        )
        assert result.exit_code == 1
        plain = _plain(result.output)
        assert "Output path is outside the working directory:" in plain
        assert not outside_file.exists()

    def test_convert_inside_cwd_allowed(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)
        out_file = workspace / "converted.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "convert", str(data_file), "--to", "sharegpt", "-o", str(out_file)],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
        expected = [
            {"conversations": [
                {"from": "human", "value": "What is AI?"},
                {"from": "gpt", "value": "Artificial Intelligence."},
            ]},
            {"conversations": [
                {"from": "human", "value": "Define ML."},
                {"from": "gpt", "value": "Machine Learning."},
            ]},
        ]
        assert rows == expected

    def test_convert_default_name_allowed(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)
        default_out = workspace / "data_sharegpt.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "convert", str(data_file), "--to", "sharegpt"],
        )
        assert result.exit_code == 0
        assert default_out.exists()
        rows = [json.loads(line) for line in default_out.read_text(encoding="utf-8").splitlines()]
        expected = [
            {"conversations": [
                {"from": "human", "value": "What is AI?"},
                {"from": "gpt", "value": "Artificial Intelligence."},
            ]},
            {"conversations": [
                {"from": "human", "value": "Define ML."},
                {"from": "gpt", "value": "Machine Learning."},
            ]},
        ]
        assert rows == expected


class TestDataMergeContainment:
    """Verify soup data merge enforces is_under_cwd on output paths."""

    def test_merge_relative_outside_rejected(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file1 = workspace / "data1.jsonl"
        data_file2 = workspace / "data2.jsonl"
        _make_sample_dataset(data_file1)
        _make_sample_dataset(data_file2)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "merge", str(data_file1), str(data_file2), "-o", "../escaped_merge.jsonl"],
        )
        assert result.exit_code == 1
        plain = _plain(result.output)
        assert "Output path is outside the working directory:" in plain
        assert not (tmp_path / "escaped_merge.jsonl").exists()

    def test_merge_absolute_outside_rejected(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)
        outside_file = tmp_path / "outside_merged.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "merge", str(data_file), "-o", str(outside_file)],
        )
        assert result.exit_code == 1
        plain = _plain(result.output)
        assert "Output path is outside the working directory:" in plain
        assert not outside_file.exists()

    def test_merge_inside_cwd_allowed(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file1 = workspace / "data1.jsonl"
        data_file2 = workspace / "data2.jsonl"
        _make_sample_dataset(data_file1)
        _make_sample_dataset(data_file2)
        out_file = workspace / "merged_out.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "merge", str(data_file1), str(data_file2), "-o", str(out_file)],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
        expected = [
            {"instruction": "What is AI?", "output": "Artificial Intelligence."},
            {"instruction": "Define ML.", "output": "Machine Learning."},
            {"instruction": "What is AI?", "output": "Artificial Intelligence."},
            {"instruction": "Define ML.", "output": "Machine Learning."},
        ]
        assert rows == expected

    def test_merge_default_name_allowed(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file1 = workspace / "data1.jsonl"
        data_file2 = workspace / "data2.jsonl"
        _make_sample_dataset(data_file1)
        _make_sample_dataset(data_file2)
        default_out = workspace / "merged.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "merge", str(data_file1), str(data_file2)],
        )
        assert result.exit_code == 0
        assert default_out.exists()
        rows = [json.loads(line) for line in default_out.read_text(encoding="utf-8").splitlines()]
        expected = [
            {"instruction": "What is AI?", "output": "Artificial Intelligence."},
            {"instruction": "Define ML.", "output": "Machine Learning."},
            {"instruction": "What is AI?", "output": "Artificial Intelligence."},
            {"instruction": "Define ML.", "output": "Machine Learning."},
        ]
        assert rows == expected


class TestDataFilterContainment:
    """Verify soup data filter enforces is_under_cwd on output paths."""

    def test_filter_relative_outside_rejected(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "filter", str(data_file), "--coherence", "0.5", "-o", "../escaped_f.jsonl"],
        )
        assert result.exit_code == 1
        plain = _plain(result.output)
        assert "Output path is outside the working directory:" in plain
        assert not (tmp_path / "escaped_f.jsonl").exists()

    def test_filter_absolute_outside_rejected(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)
        outside_file = tmp_path / "outside_filter.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "filter", str(data_file), "--coherence", "0.5", "-o", str(outside_file)],
        )
        assert result.exit_code == 1
        plain = _plain(result.output)
        assert "Output path is outside the working directory:" in plain
        assert not outside_file.exists()

    def test_filter_score_only_outside_rejected(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "filter", str(data_file), "--score-only", "-o", "../escaped_scored.jsonl"],
        )
        assert result.exit_code == 1
        plain = _plain(result.output)
        assert "Output path is outside the working directory:" in plain
        assert not (tmp_path / "escaped_scored.jsonl").exists()

    def test_filter_inside_cwd_allowed(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)
        out_file = workspace / "filtered_out.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "filter", str(data_file), "--coherence", "0.1", "-o", str(out_file)],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
        expected = [
            {"instruction": "What is AI?", "output": "Artificial Intelligence."},
            {"instruction": "Define ML.", "output": "Machine Learning."},
        ]
        assert rows == expected

    def test_filter_default_name_allowed(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)
        default_out = workspace / "data_filtered.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "filter", str(data_file), "--coherence", "0.1"],
        )
        assert result.exit_code == 0
        assert default_out.exists()
        rows = [
            json.loads(line) for line in default_out.read_text(encoding="utf-8").splitlines()
        ]
        expected = [
            {"instruction": "What is AI?", "output": "Artificial Intelligence."},
            {"instruction": "Define ML.", "output": "Machine Learning."},
        ]
        assert rows == expected

    def test_filter_score_only_inside_cwd_allowed(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)
        monkeypatch.setattr(
            "soup_cli.utils.quality.compute_perplexity_scores",
            lambda texts, model_name="gpt2": [12.5] * len(texts),
        )

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)
        out_file = workspace / "scored_out.jsonl"

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["data", "filter", str(data_file), "--score-only", "-o", str(out_file)],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
        expected = [
            {
                "instruction": "What is AI?",
                "output": "Artificial Intelligence.",
                "_coherence_score": 0.9067,
                "_perplexity_score": 12.5,
            },
            {
                "instruction": "Define ML.",
                "output": "Machine Learning.",
                "_coherence_score": 0.8933,
                "_perplexity_score": 12.5,
            },
        ]
        assert rows == expected


class TestDataSymlinkContainment:
    """Verify symlinks pointing outside cwd are refused across all three commands."""

    @pytest.mark.requires_symlink
    @pytest.mark.parametrize(
        ("cmd", "extra_args"),
        [
            ("convert", ["--to", "sharegpt"]),
            ("merge", []),
            ("filter", ["--coherence", "0.5"]),
        ],
    )
    def test_symlink_leaving_cwd_rejected(self, cmd, extra_args, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()

        link_dir = workspace / "link_to_outside"
        os.symlink(outside_dir, link_dir, target_is_directory=True)

        data_file = workspace / "data.jsonl"
        _make_sample_dataset(data_file)
        target = link_dir / "escaped_via_symlink.jsonl"

        args = ["data", cmd, str(data_file)] + extra_args + ["-o", str(target)]
        runner = CliRunner()
        result = runner.invoke(app, args)
        assert result.exit_code == 1
        plain = _plain(result.output)
        assert "Output path is outside the working directory:" in plain
        assert list(outside_dir.iterdir()) == []
