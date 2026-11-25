"""Requested perplexity scoring must finish before an output can be written."""

import json
import sys
import types
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import data
from soup_cli.utils import quality
from tests.conftest import strip_ansi

_ROWS = [
    {"text": "This is a complete sentence with clear meaning."},
    {"text": "x"},
]
_SENTINEL = b"existing output must stay byte-identical\r\n"


@pytest.fixture(autouse=True)
def _console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "console", Console(width=100))


def _dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bytes:
    monkeypatch.chdir(tmp_path)
    payload = "".join(json.dumps(row) + "\n" for row in _ROWS).encode()
    (tmp_path / "rows.jsonl").write_bytes(payload)
    return payload


def _block_dependencies(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    # The real scorer reaches its lazy imports. Even on a core-only install,
    # the transformers case reaches that import without needing a real torch.
    monkeypatch.setitem(sys.modules, "torch", None if missing != "transformers"
                        else types.ModuleType("torch"))
    if missing != "torch":
        monkeypatch.setitem(sys.modules, "transformers", None)


@pytest.mark.parametrize("missing", ["torch", "transformers", "both"])
@pytest.mark.parametrize("option", ["--perplexity", "--score-only"])
@pytest.mark.parametrize("explicit_output", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_missing_scoring_dependency_refuses_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str, option: str,
    explicit_output: bool, existing: bool,
) -> None:
    source = _dataset(tmp_path, monkeypatch)
    suffix = "scored" if option == "--score-only" else "filtered"
    output = tmp_path / ("output.jsonl" if explicit_output else f"rows_{suffix}.jsonl")
    if existing:
        output.write_bytes(_SENTINEL)
    _block_dependencies(monkeypatch, missing)
    args = ["data", "filter", "rows.jsonl", "--field", "text", option]
    if option == "--perplexity":
        args += ["5", "--coherence", "0.3"]
    if explicit_output:
        args += ["--output", output.name]

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 1, (result.output, result.exception)
    assert isinstance(result.exception, SystemExit)
    message = " ".join(strip_ansi(result.output).split())
    assert "torch" in message and "transformers" in message
    assert 'pip install "soup-cli[train]"' in message
    assert "Skipping perplexity" not in message
    assert "Filter complete" not in message and "Scored 2 rows" not in message
    if existing:
        assert output.read_bytes() == _SENTINEL
    else:
        assert not output.exists()
    assert (tmp_path / "rows.jsonl").read_bytes() == source


def test_coherence_only_works_without_scoring_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _dataset(tmp_path, monkeypatch)
    _block_dependencies(monkeypatch, "both")

    result = CliRunner().invoke(app, [
        "data", "filter", "rows.jsonl", "--coherence", "0.3", "--field", "text",
    ])

    assert result.exit_code == 0, (result.output, result.exception)
    rows = [json.loads(line) for line in (tmp_path / "rows_filtered.jsonl").read_text()
            .splitlines()]
    assert rows == _ROWS[:1]
    assert (tmp_path / "rows.jsonl").read_bytes() == source


@pytest.mark.parametrize("option", ["--perplexity", "--score-only"])
def test_successful_scoring_keeps_filter_and_score_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str,
) -> None:
    source = _dataset(tmp_path, monkeypatch)

    def scores(texts: list[str], model_name: str) -> list[float]:
        assert texts == [row["text"] for row in _ROWS]
        assert model_name == "local-test-model"
        return [2.0, 20.0]

    monkeypatch.setattr(quality, "compute_perplexity_scores", scores)
    args = ["data", "filter", "rows.jsonl", "--field", "text", "--ppl-model",
            "local-test-model", option]
    if option == "--perplexity":
        args += ["5"]
    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, (result.output, result.exception)
    suffix = "scored" if option == "--score-only" else "filtered"
    rows = [json.loads(line) for line in (tmp_path / f"rows_{suffix}.jsonl").read_text()
            .splitlines()]
    if option == "--perplexity":
        assert rows == _ROWS[:1]
    else:
        assert [row["_perplexity_score"] for row in rows] == [2.0, 20.0]
        assert [row["_coherence_score"] for row in rows] == quality.compute_coherence_score(
            [row["text"] for row in _ROWS]
        )
        assert [row["text"] for row in rows] == [row["text"] for row in _ROWS]
    assert (tmp_path / "rows.jsonl").read_bytes() == source
