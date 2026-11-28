"""Tests for CLI boundary validation of -n and --pct in soup data sample (#2931, #2929)."""

from __future__ import annotations

import json
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


@pytest.fixture
def sample_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fixture creating a 10-row synthetic JSONL dataset under temporary cwd."""
    monkeypatch.chdir(tmp_path)
    data_file = tmp_path / "sample_input.jsonl"
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
def test_sample_positive_control_n(sample_jsonl: Path, tmp_path: Path, strategy: str, n: int):
    """Multi-item positive control fixture: valid -n produces exact sample sizes."""
    out_file = tmp_path / f"out_{strategy}_{n}.jsonl"
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
    sample_jsonl: Path, tmp_path: Path, pct: float, expected_len: int
):
    """Multi-item positive control fixture: valid --pct produces expected sample sizes."""
    out_file = tmp_path / f"out_pct_{int(pct)}.jsonl"
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
def test_sample_accepts_n_larger_than_dataset(sample_jsonl: Path, tmp_path: Path, strategy: str):
    """Counts larger than dataset are accepted and clamped to full dataset size."""
    out_file = tmp_path / f"out_large_{strategy}.jsonl"
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


BOTH_REFUSAL = "Cannot use --n (count) and --pct (percentage) together. Pick one."


@pytest.mark.parametrize("strategy", ["random", "diverse", "hard"])
@pytest.mark.parametrize(
    "sizes",
    [["-n", "2", "--pct", "50"], ["--pct", "50", "--n", "2"]],
    ids=["n-then-pct", "pct-then-n"],
)
def test_sample_both_n_and_pct_refused(
    sample_jsonl: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    strategy: str, sizes: list[str],
):
    """#3436: --n with --pct exits 1 before the input is read, in either order."""

    def _must_not_load(*_args, **_kwargs):
        raise AssertionError("the input was read before the refusal")

    monkeypatch.setattr("soup_cli.commands.data.load_raw_data", _must_not_load)
    res = runner.invoke(
        app,
        ["data", "sample", str(sample_jsonl), *sizes, "--strategy", strategy, "--seed", "42"],
    )
    assert res.exit_code == 1, res.output
    assert BOTH_REFUSAL in _clean(res.output)
    assert not (tmp_path / f"sample_input_sampled_{strategy}.jsonl").exists()


def test_sample_both_n_and_pct_leaves_existing_output_unchanged(
    sample_jsonl: Path, tmp_path: Path,
):
    """#3436: the refusal does not touch a file already at the --output path."""
    out_file = tmp_path / "out_both.jsonl"
    out_file.write_text('{"kept": "from an earlier run"}\n', encoding="utf-8")
    before = out_file.read_bytes()
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
    assert res.exit_code == 1, res.output
    assert BOTH_REFUSAL in _clean(res.output)
    assert out_file.read_bytes() == before


def test_sample_out_of_range_pct_with_n_reports_the_range_first(sample_jsonl: Path):
    """The bounds checks still run first: a bad --pct beside --n names the range."""
    res = runner.invoke(
        app,
        ["data", "sample", str(sample_jsonl), "-n", "2", "--pct", "150"],
    )
    assert res.exit_code == 1
    output = _clean(res.output)
    assert "--pct must be a finite number in (0, 100]." in output
    assert BOTH_REFUSAL not in output


def test_sample_invalid_n_with_valid_pct_refused(sample_jsonl: Path):
    """Passing an invalid non-positive -n alongside a valid --pct exits 1 with a refusal."""
    res = runner.invoke(
        app,
        ["data", "sample", str(sample_jsonl), "-n", "0", "--pct", "50"],
    )
    assert res.exit_code == 1
    assert "--n must be a positive integer." in _clean(res.output)
