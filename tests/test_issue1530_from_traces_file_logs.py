"""#1530 — `--format soup-serve` must refuse a file, not harvest zero pairs.

``parse_soup_serve`` returns nothing for a non-directory, so
``soup data from-traces --logs session.jsonl --format soup-serve`` ended in a
green ``Wrote 0 preference pair(s)``, exit 0, and an empty output file a
pipeline would train from. The fix refuses the file at the CLI boundary (the
smaller of the two changes the issue proposed) and, when a source reads zero
traces at all, says so instead of falling through to the same green line.

The zero-pair EXIT CODE is deliberately unchanged: whether an empty harvest
should fail the command is a design question the maintainer has left open.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tests.conftest import strip_ansi


def _runner(tmp_path, monkeypatch) -> CliRunner:
    monkeypatch.chdir(tmp_path)
    return CliRunner()


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _from_traces(runner, logs: str, fmt: str, signal: str = "thumbs_up"):
    from soup_cli.cli import app

    return runner.invoke(app, [
        "data", "from-traces", "--logs", logs, "--format", fmt,
        "--signal", signal, "-o", "prefs.jsonl",
    ])


# --------------------------------------------------------------------------- #
# The refusal: a single file under --format soup-serve
# --------------------------------------------------------------------------- #


def test_soup_serve_refuses_a_single_file_instead_of_writing_an_empty_output(
    tmp_path, monkeypatch,
):
    """The headline: exit non-zero, name --logs, and say what shape it wants."""
    runner = _runner(tmp_path, monkeypatch)
    # Rows a user would swear are readable — the refusal is about the SHAPE
    # (file vs directory), and losing the data to a green 0-pair line is the
    # bug, so the records are valid soup-serve records.
    _write_jsonl(
        tmp_path / "session.jsonl",
        [
            {"id": "1", "prompt": "Q", "response": "Good", "signal": "thumbs_up"},
            {"id": "2", "prompt": "Q", "response": "Bad", "signal": "thumbs_down"},
        ],
    )
    result = _from_traces(runner, "session.jsonl", "soup-serve")
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code != 0, text
    assert "is not a directory" in text, text
    # The option and the expected shape are named, so the mistake is fixable.
    assert "--logs 'session.jsonl'" in text, text
    assert "soup serve --trace-log" in text, text
    # The usable alternative for a single JSONL is named too.
    assert "--format langchain" in text and "openai" in text, text
    # Refused before anything was written.
    assert not (tmp_path / "prefs.jsonl").exists()


def test_a_real_soup_serve_directory_still_harvests(tmp_path, monkeypatch):
    """Control: the refusal is scoped to files; a directory keeps pairing."""
    runner = _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "traces" / "t.jsonl",
        [
            {"id": "1", "prompt": "Q", "response": "Good", "signal": "thumbs_up"},
            {"id": "2", "prompt": "Q", "response": "Bad", "signal": "thumbs_down"},
        ],
    )
    result = _from_traces(runner, "traces", "soup-serve")
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, text
    assert "Wrote 1 preference pair(s)" in text, text


def test_single_files_still_read_for_langchain(tmp_path, monkeypatch):
    """Control: only soup-serve refuses a file; langchain keeps reading one."""
    runner = _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "langchain.jsonl",
        [{
            "id": "1",
            "inputs": {"prompt": "Q"},
            "outputs": {"text": "A"},
            "feedback": [{"key": "thumbs", "score": 1}],
        }],
    )
    result = _from_traces(runner, "langchain.jsonl", "langchain")
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, text
    assert "is not a directory" not in text, text


# --------------------------------------------------------------------------- #
# Zero traces read: said out loud, exit code untouched
# --------------------------------------------------------------------------- #


def test_an_empty_directory_says_it_read_nothing(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    (tmp_path / "traces").mkdir()
    result = _from_traces(runner, "traces", "soup-serve")
    text = " ".join(strip_ansi(result.output).split())
    # The zero-pair exit code is the maintainer's open question — unchanged.
    assert result.exit_code == 0, text
    assert "Read 0 trace(s)" in text, text
    # The path is named, so the reader can see which --logs read nothing.
    assert "--logs 'traces'" in text, text
    assert "no readable *.jsonl trace files" in text, text


def test_a_dir_holding_only_non_jsonl_files_also_says_it_read_nothing(
    tmp_path, monkeypatch,
):
    """The parser walks *.jsonl only; a dir of .txt files is the same trap."""
    runner = _runner(tmp_path, monkeypatch)
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "t.txt").write_text("not a trace\n", encoding="utf-8")
    result = _from_traces(runner, "traces", "soup-serve")
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, text
    assert "Read 0 trace(s)" in text, text


def test_a_langchain_file_no_line_of_which_parses_says_it_read_nothing(
    tmp_path, monkeypatch,
):
    """The file formats share the message, with their own reason."""
    runner = _runner(tmp_path, monkeypatch)
    (tmp_path / "garbage.jsonl").write_text("not json\nalso not json\n", encoding="utf-8")
    result = _from_traces(runner, "garbage.jsonl", "langchain")
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, text
    assert "Read 0 trace(s)" in text, text
    assert "no line parsed as a langchain record" in text, text


def test_a_directory_under_a_file_format_names_the_shape_not_the_content(
    tmp_path, monkeypatch,
):
    """langchain/openai only read files; a directory read nothing at all.

    Saying "no line parsed" there misreports a shape problem (directory where
    a JSONL file belongs) as a content problem, so the reason names the shape.
    """
    runner = _runner(tmp_path, monkeypatch)
    (tmp_path / "traces").mkdir()
    result = _from_traces(runner, "traces", "langchain")
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, text
    assert "Read 0 trace(s)" in text, text
    assert (
        "--logs is a directory; --format langchain reads a single JSONL file"
        in text
    ), text
    assert "no line parsed" not in text, text


def test_the_openai_reason_uses_the_right_article(tmp_path, monkeypatch):
    """Review nit: 'a openai record' read badly; keep the grammar pinned."""
    runner = _runner(tmp_path, monkeypatch)
    (tmp_path / "garbage.jsonl").write_text("not json\n", encoding="utf-8")
    result = _from_traces(runner, "garbage.jsonl", "openai")
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, text
    assert "no line parsed as an openai record" in text, text
