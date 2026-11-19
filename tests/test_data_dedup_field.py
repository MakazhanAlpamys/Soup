"""A wholly absent selected field must be refused before dedup dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.cli import app
from soup_cli.commands import data as command
from tests.conftest import make_cli_runner, strip_ansi


def _write_rows(path: Path, rows: list[dict]) -> bytes:
    content = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(content)
    return content


@pytest.mark.parametrize("semantic", [False, True], ids=["minhash", "semantic"])
@pytest.mark.parametrize("existing", [False, True], ids=["new-output", "existing-output"])
@pytest.mark.parametrize("field", [
    "text", "[bold]missing[/bold]",
    "this missing field has enough words to wrap the diagnostic",
])
def test_absent_field_refuses_before_backend_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, semantic: bool, existing: bool, field: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "rows.jsonl"
    original = _write_rows(source, [{"output": "one two three"}, {"output": "four five six"}])
    output = tmp_path / "result.jsonl"
    sentinel = b"existing output must survive\n"
    if existing:
        output.write_bytes(sentinel)
    called = []

    def unexpected_backend(*_args: object, **_kwargs: object) -> None:
        called.append(True)
        raise AssertionError("backend or writer reached for a wholly absent field")

    monkeypatch.setattr(command, "_semantic_dedup", unexpected_backend)
    monkeypatch.setattr(command, "_write_jsonl", unexpected_backend)
    args = ["data", "dedup", "rows.jsonl", "--field", field, "-o", "result.jsonl"]
    if semantic:
        args.append("--semantic")
    result = make_cli_runner().invoke(app, args)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    printed = " ".join(strip_ansi(result.output).split())
    assert f"--field {field!r}" in printed
    assert "absent from every row" in printed
    assert called == []
    assert source.read_bytes() == original
    if existing:
        assert output.read_bytes() == sentinel
    else:
        assert not output.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_absent_field_preserves_output_without_backend_doubles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool,
) -> None:
    pytest.importorskip("datasketch")
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "rows.jsonl"
    original = _write_rows(source, [{"output": "one two three"}, {"output": "four five six"}])
    output = tmp_path / "result.jsonl"
    sentinel = b"keep this output\n"
    if existing:
        output.write_bytes(sentinel)
    result = make_cli_runner().invoke(
        app, ["data", "dedup", "rows.jsonl", "--field", "text", "-o", "result.jsonl"],
    )
    assert result.exit_code == 1, (result.output, repr(result.exception))
    printed = " ".join(strip_ansi(result.output).split())
    assert "absent from every row" in printed
    assert source.read_bytes() == original
    assert output.read_bytes() == sentinel if existing else not output.exists()


@pytest.mark.parametrize("count", [2, 20])
@pytest.mark.parametrize("field", [None, "output", ""],
                         ids=["default-fields", "selected-field", "empty-field"])
def test_distinct_rows_survive_real_minhash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, field: str | None,
) -> None:
    pytest.importorskip("datasketch")
    monkeypatch.chdir(tmp_path)
    rows = [{"output": " ".join(f"row{i}word{j}" for j in range(4))} for i in range(count)]
    original = _write_rows(tmp_path / "rows.jsonl", rows)
    args = ["data", "dedup", "rows.jsonl", "-o", "result.jsonl"]
    if field is not None:
        args.extend(["--field", field])
    result = make_cli_runner().invoke(app, args)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    loaded = [json.loads(line) for line in (tmp_path / "result.jsonl").read_text().splitlines()]
    assert loaded == rows
    assert (tmp_path / "rows.jsonl").read_bytes() == original


@pytest.mark.parametrize("rows,expected", [
    ([{"output": ""}, {"output": ""}], [{"output": ""}]),
    ([{"output": "one two three"}, {"output": "four five six"}, {"other": 1}, {"other": 2}],
     [{"output": "one two three"}, {"output": "four five six"}, {"other": 1}]),
])
def test_present_empty_and_partially_missing_keep_the_existing_minhash_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict], expected: list[dict],
) -> None:
    pytest.importorskip("datasketch")
    monkeypatch.chdir(tmp_path)
    _write_rows(tmp_path / "rows.jsonl", rows)
    result = make_cli_runner().invoke(
        app, ["data", "dedup", "rows.jsonl", "--field", "output", "-o", "result.jsonl"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    loaded = [json.loads(line) for line in (tmp_path / "result.jsonl").read_text().splitlines()]
    assert loaded == expected


@pytest.mark.parametrize("field", ["output", ""], ids=["selected-field", "empty-field"])
@pytest.mark.parametrize("rows", [
    [{"output": ""}, {"output": ""}],
    [{"output": "one two three"}, {"other": "four five six"}],
])
def test_present_empty_and_partially_missing_reach_semantic_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict], field: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_rows(tmp_path / "rows.jsonl", rows)
    seen = []

    def record(data: list[dict], **kwargs: object) -> None:
        seen.append((data, kwargs["field"]))

    monkeypatch.setattr(command, "_semantic_dedup", record)
    result = make_cli_runner().invoke(
        app, ["data", "dedup", "rows.jsonl", "--field", field, "--semantic"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert seen == [(rows, field)]
