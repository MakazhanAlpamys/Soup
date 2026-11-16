"""Tests for CLI boundary validation of -n and --pct in soup data sample (#2929)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.data import (
    _sample_diverse,
    _sample_hard,
    _sample_random,
)
from tests.conftest import strip_ansi

runner = CliRunner()


def _clean(text: str) -> str:
    """Strip ANSI color codes and collapse whitespace for resilient assertions."""
    return " ".join(strip_ansi(text).split())


@pytest.fixture(scope="module")
def sample_test_dir() -> Path:
    """Fixture creating a test directory under cwd to satisfy is_under_cwd."""
    test_dir = Path.cwd() / ".tmp_test_sample_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def sample_jsonl(sample_test_dir: Path) -> Path:
    """Fixture creating a 10-row synthetic JSONL dataset under cwd."""
    data_file = sample_test_dir / "sample_input.jsonl"
    rows = [{"id": i, "text": f"row text {i} " * (i + 1)} for i in range(10)]
    with open(data_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return data_file


@pytest.mark.parametrize("invalid_n", [0, -1, -10])
@pytest.mark.parametrize("strategy", ["random", "diverse", "hard"])
def test_sample_rejects_non_positive_n(sample_jsonl: Path, invalid_n: int, strategy: str):
    """Negative and zero -n values are refused with exit code 1."""
    res = runner.invoke(
        app,
        ["data", "sample", str(sample_jsonl), "-n", str(invalid_n), "--strategy", strategy],
    )
    assert res.exit_code == 1
    assert "--n must be a positive integer." in _clean(res.output)


@pytest.mark.parametrize(
    "invalid_pct",
    ["nan", "+nan", "-nan", "NaN", "NAN", "inf", "+inf", "-inf", "0", "-0", "-50", "100.1", "150"],
)
def test_sample_rejects_invalid_pct(sample_jsonl: Path, invalid_pct: str):
    """Non-finite or out-of-range (not in (0, 100]) --pct values are refused with exit 1."""
    res = runner.invoke(
        app,
        ["data", "sample", str(sample_jsonl), "--pct", invalid_pct],
    )
    assert res.exit_code == 1
    assert "--pct must be a finite number in (0, 100]." in _clean(res.output)


@pytest.mark.parametrize("invalid_num", [0, -1, -5])
def test_sample_helpers_guard_against_non_positive_num(invalid_num: int):
    """Internal helpers safely return empty lists when num <= 0 without crashing."""
    data = [{"id": i, "text": f"hello {i}"} for i in range(5)]
    assert _sample_random(data, invalid_num) == []
    assert _sample_diverse(data, invalid_num) == []
    assert _sample_hard(data, invalid_num) == []


@pytest.mark.parametrize("strategy", ["random", "diverse", "hard"])
@pytest.mark.parametrize("n", [1, 3, 7])
def test_sample_positive_control_n(
    sample_jsonl: Path, sample_test_dir: Path, strategy: str, n: int
):
    """Multi-item positive control fixture: valid -n produces exact sample sizes."""
    out_file = sample_test_dir / f"out_{strategy}_{n}.jsonl"
    res = runner.invoke(
        app,
        [
            "data", "sample", str(sample_jsonl),
            "-n", str(n),
            "--strategy", strategy,
            "--output", str(out_file),
            "--seed", "42",
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == n


@pytest.mark.parametrize("pct,expected_len", [(10.0, 1), (50.0, 5), (100.0, 10)])
def test_sample_positive_control_pct(
    sample_jsonl: Path, sample_test_dir: Path, pct: float, expected_len: int
):
    """Multi-item positive control fixture: valid --pct produces expected sample sizes."""
    out_file = sample_test_dir / f"out_pct_{int(pct)}.jsonl"
    res = runner.invoke(
        app,
        [
            "data", "sample", str(sample_jsonl),
            "--pct", str(pct),
            "--output", str(out_file),
            "--seed", "42",
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == expected_len


@pytest.mark.parametrize("strategy", ["random", "diverse", "hard"])
def test_sample_accepts_n_larger_than_dataset(
    sample_jsonl: Path, sample_test_dir: Path, strategy: str
):
    """Counts larger than dataset are accepted and clamped to full dataset size."""
    out_file = sample_test_dir / f"out_large_{strategy}.jsonl"
    res = runner.invoke(
        app,
        [
            "data", "sample", str(sample_jsonl),
            "-n", "50",
            "--strategy", strategy,
            "--output", str(out_file),
            "--seed", "42",
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 10


def test_sample_neither_n_nor_pct_refused(sample_jsonl: Path):
    """Running without either --n or --pct is refused as a usage error."""
    res = runner.invoke(app, ["data", "sample", str(sample_jsonl)])
    assert res.exit_code == 1
    assert "Specify either --n (count) or --pct (percentage)." in _clean(res.output)


def test_sample_both_n_and_pct_given(sample_jsonl: Path, sample_test_dir: Path):
    """When both --n and --pct are passed, --pct takes precedence as documented."""
    out_file = sample_test_dir / "out_both.jsonl"
    res = runner.invoke(
        app,
        [
            "data", "sample", str(sample_jsonl),
            "-n", "2",
            "--pct", "50",
            "--output", str(out_file),
            "--seed", "42",
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 5
