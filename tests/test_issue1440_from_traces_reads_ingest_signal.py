"""#1440 — `soup data from-traces` must read what Soup's own producers write.

The pair builders key on `Trace.signal` (`thumbs_up` / `thumbs_down`,
`regenerated`, `user_edit`) and on `Trace.edited_output`. `parse_soup_serve`
took the signal only from a nested `feedback.rating` field that none of Soup's
producers writes: `soup ingest` emits a **top-level** `signal`, and
`soup serve --trace-log` records no feedback at all. No parser ever set
`user_edit` or `edited_output`. Every one of those cases ended in
`Wrote 0 preference pair(s)`, exit 0, and an empty file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
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


def _read_jsonl(path) -> list:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# The round trip: `soup ingest` -> `soup data from-traces`
# --------------------------------------------------------------------------- #


def test_ingest_thumbs_survive_the_round_trip_into_one_pair(tmp_path, monkeypatch):
    """The headline: Soup's own producer, read back by Soup's own consumer.

    Before the fix every row was read as signal `none`, so this wrote an empty
    file and exited 0.
    """
    from soup_cli.cli import app
    from soup_cli.data.traces import parse_soup_serve

    runner = _runner(tmp_path, monkeypatch)
    # The issue's snippet does this too: `ingest` opens --output for writing
    # without creating parent directories.
    (tmp_path / "traces").mkdir()
    _write_jsonl(
        tmp_path / "export.jsonl",
        [
            {"id": "1", "input": "Q", "output": "Good", "rating": "up"},
            {"id": "2", "input": "Q", "output": "Bad", "rating": "down"},
        ],
    )
    ingest = runner.invoke(
        app,
        [
            "ingest",
            "--source",
            "langfuse",
            "--logs",
            "export.jsonl",
            "--output",
            "traces/traces.jsonl",
        ],
    )
    assert ingest.exit_code == 0, strip_ansi(ingest.output)

    # `soup ingest` writes the canonical signal the parsers used to ignore.
    written = _read_jsonl(tmp_path / "traces" / "traces.jsonl")
    assert [r["signal"] for r in written] == ["thumbs_up", "thumbs_down"]

    assert [t.signal for t in parse_soup_serve("traces")] == [
        "thumbs_up",
        "thumbs_down",
    ]

    result = runner.invoke(
        app,
        [
            "data",
            "from-traces",
            "--logs",
            "traces",
            "--format",
            "soup-serve",
            "--signal",
            "thumbs_up",
            "-o",
            "prefs.jsonl",
        ],
    )
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, text
    assert "Wrote 1 preference pair(s)" in text, text

    pairs = _read_jsonl(tmp_path / "prefs.jsonl")
    assert len(pairs) == 1, pairs
    assert pairs[0]["chosen"] == "Good"
    assert pairs[0]["rejected"] == "Bad"


def test_loop_harvest_reads_the_same_ingest_records(tmp_path, monkeypatch):
    """The pre-wired loop stage is parse_soup_serve + build_pairs(thumbs_up)."""
    from soup_cli.utils.loop_stages import harvest_from_traces
    from soup_cli.utils.loop_state import LoopState

    _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "traces" / "t.jsonl",
        [
            {"id": "1", "prompt": "Q", "output": "Good", "signal": "thumbs_up"},
            {"id": "2", "prompt": "Q", "output": "Bad", "signal": "thumbs_down"},
        ],
    )
    # The resolver honours SOUP_LOOP_TRACE_DIR before any model path.
    monkeypatch.setenv("SOUP_LOOP_TRACE_DIR", str(tmp_path / "traces"))
    state = LoopState(
        served_model="registry://dummy",
        eval_suite="e",
        baseline="b",
    )

    result = harvest_from_traces(state)
    assert result["pairs_harvested"] == 1, result
    assert result["traces_collected"] == 2, result
    pairs = _read_jsonl(result["pairs_path"])
    assert pairs[0]["chosen"] == "Good"
    assert pairs[0]["rejected"] == "Bad"


@pytest.mark.parametrize(
    "extra",
    [
        {"signal": "user_edit", "edited_output": "Polished"},
        {"signal": "user_edit", "edited_response": "Polished"},
        {"signal": "user_edit", "feedback": {"edited_output": "Polished"}},
    ],
    ids=["edited_output", "edited_response", "nested_feedback"],
)
def test_user_edit_pairs_from_its_documented_shapes(tmp_path, monkeypatch, extra):
    from soup_cli.data.traces import build_pairs, parse_soup_serve

    _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "edits" / "t.jsonl",
        [{"id": "3", "prompt": "Q", "response": "Raw", **extra}],
    )
    traces = list(parse_soup_serve("edits"))
    assert traces[0].signal == "user_edit", traces
    assert traces[0].edited_output == "Polished", traces

    pairs = list(build_pairs(traces, signal="user_edit"))
    assert len(pairs) == 1, pairs
    assert pairs[0].chosen == "Polished"
    assert pairs[0].rejected == "Raw"


def test_user_edit_is_not_reported_when_nothing_was_edited(tmp_path, monkeypatch):
    """An edit equal to the original answer is not a preference pair."""
    from soup_cli.data.traces import build_pairs, parse_soup_serve

    _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "edits" / "t.jsonl",
        [
            {
                "id": "4",
                "prompt": "Q",
                "response": "Same",
                "signal": "user_edit",
                "edited_output": "Same",
            }
        ],
    )
    traces = list(parse_soup_serve("edits"))
    assert list(build_pairs(traces, signal="user_edit")) == []


# --------------------------------------------------------------------------- #
# Controls: the older nested shape keeps pairing, and a stale block does not
# downgrade a canonical signal.
# --------------------------------------------------------------------------- #


def test_feedback_rating_still_pairs(tmp_path, monkeypatch):
    from soup_cli.data.traces import build_pairs, parse_soup_serve

    _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "traces" / "t.jsonl",
        [
            {"id": "1", "prompt": "Q", "response": "Good", "feedback": {"rating": "up"}},
            {"id": "2", "prompt": "Q", "response": "Bad", "feedback": {"rating": "down"}},
        ],
    )
    pairs = list(build_pairs(list(parse_soup_serve("traces")), signal="thumbs_up"))
    assert len(pairs) == 1, pairs
    assert pairs[0].chosen == "Good"


def test_top_level_signal_wins_over_a_stale_feedback_block(tmp_path, monkeypatch):
    """A record carrying both is not downgraded to `none` by the old block."""
    from soup_cli.data.traces import parse_soup_serve

    _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "traces" / "t.jsonl",
        [
            {
                "id": "1",
                "prompt": "Q",
                "response": "A",
                "signal": "thumbs_up",
                "feedback": {"note": "no rating here"},
            }
        ],
    )
    assert [t.signal for t in parse_soup_serve("traces")] == ["thumbs_up"]


def test_an_unknown_signal_string_does_not_become_a_trace_signal(tmp_path, monkeypatch):
    from soup_cli.data.traces import parse_soup_serve

    _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "traces" / "t.jsonl",
        [{"id": "1", "prompt": "Q", "response": "A", "signal": "banana"}],
    )
    assert [t.signal for t in parse_soup_serve("traces")] == ["none"]


# --------------------------------------------------------------------------- #
# An empty harvest must not read as a completed one.
# --------------------------------------------------------------------------- #


def test_reading_traces_that_pair_nothing_says_so(tmp_path, monkeypatch):
    from soup_cli.cli import app

    runner = _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "traces" / "t.jsonl",
        [{"id": "1", "prompt": "Q", "response": "A", "signal": "thumbs_up"}],
    )
    result = runner.invoke(
        app,
        [
            "data",
            "from-traces",
            "--logs",
            "traces",
            "--format",
            "soup-serve",
            "--signal",
            "user_edit",
            "-o",
            "prefs.jsonl",
        ],
    )
    text = " ".join(strip_ansi(result.output).split())
    assert "Read 1 trace(s) but built no pairs" in text, text
    assert "--signal user_edit" in text, text
    # The signal that WAS present is named, so the reader can see the mismatch.
    assert "thumbs_up" in text, text


def test_an_empty_directory_is_not_reported_as_a_failed_harvest(tmp_path, monkeypatch):
    from soup_cli.cli import app

    runner = _runner(tmp_path, monkeypatch)
    (tmp_path / "traces").mkdir()
    result = runner.invoke(
        app,
        [
            "data",
            "from-traces",
            "--logs",
            "traces",
            "--format",
            "soup-serve",
            "--signal",
            "thumbs_up",
            "-o",
            "prefs.jsonl",
        ],
    )
    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 0, text
    assert "built no pairs" not in text, text


def test_a_serve_trace_log_with_no_signal_says_no_trace_carried_one(tmp_path, monkeypatch):
    """`soup serve --trace-log` records ts/prompt/response/latency_ms/tokens only."""
    from soup_cli.cli import app

    runner = _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "serve" / "t.jsonl",
        [
            {"ts": 1, "prompt": "Q", "response": "A", "latency_ms": 12, "tokens": 3},
            {"ts": 2, "prompt": "Q2", "response": "B", "latency_ms": 9, "tokens": 2},
        ],
    )
    result = runner.invoke(
        app,
        [
            "data",
            "from-traces",
            "--logs",
            "serve",
            "--format",
            "soup-serve",
            "--signal",
            "thumbs_up",
            "-o",
            "prefs.jsonl",
        ],
    )
    text = " ".join(strip_ansi(result.output).split())
    assert "Read 2 trace(s) but built no pairs for --signal thumbs_up" in text, text
    assert "No trace carried a signal" in text, text
    assert "Signals present" not in text, text


def test_the_present_signals_list_never_names_none(tmp_path, monkeypatch):
    from soup_cli.cli import app

    runner = _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "traces" / "t.jsonl",
        [
            {"id": "1", "prompt": "Q", "response": "A", "signal": "thumbs_up"},
            {"id": "2", "prompt": "Q2", "response": "B"},
        ],
    )
    result = runner.invoke(
        app,
        [
            "data",
            "from-traces",
            "--logs",
            "traces",
            "--format",
            "soup-serve",
            "--signal",
            "user_edit",
            "-o",
            "prefs.jsonl",
        ],
    )
    text = " ".join(strip_ansi(result.output).split())
    assert "Signals present: thumbs_up." in text, text


def test_a_canonical_top_level_signal_beats_a_conflicting_feedback_rating(
    tmp_path,
    monkeypatch,
):
    """Documented precedence: the top-level `signal` wins; feedback.rating is a fallback."""
    from soup_cli.data.traces import parse_soup_serve

    _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "traces" / "t.jsonl",
        [
            {
                "id": "1",
                "prompt": "Q",
                "response": "A",
                "signal": "thumbs_up",
                "feedback": {"rating": "down"},
            }
        ],
    )
    assert [t.signal for t in parse_soup_serve("traces")] == ["thumbs_up"]


def test_a_whitespace_only_edit_is_not_an_edit(tmp_path, monkeypatch):
    from soup_cli.data.traces import build_pairs, parse_soup_serve

    _runner(tmp_path, monkeypatch)
    _write_jsonl(
        tmp_path / "edits" / "t.jsonl",
        [
            {
                "id": "5",
                "prompt": "Q",
                "response": "Raw",
                "signal": "user_edit",
                "edited_output": "   ",
            }
        ],
    )
    traces = list(parse_soup_serve("edits"))
    assert traces[0].edited_output is None, traces
    assert list(build_pairs(traces, signal="user_edit")) == []


def test_the_issue_reproducer_runs_offline_in_a_fresh_cwd():
    """The issue's snippet, run in a temp dir that is left behind."""
    from soup_cli.cli import app

    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            _write_jsonl(
                Path(tmp) / "export.jsonl",
                [
                    {"id": "1", "input": "Q", "output": "Good", "rating": "up"},
                    {"id": "2", "input": "Q", "output": "Bad", "rating": "down"},
                ],
            )
            Path(tmp, "traces").mkdir()
            runner = CliRunner()
            ingest = runner.invoke(
                app,
                [
                    "ingest",
                    "--source",
                    "langfuse",
                    "--logs",
                    "export.jsonl",
                    "--output",
                    "traces/traces.jsonl",
                ],
            )
            assert ingest.exit_code == 0, strip_ansi(ingest.output)
            result = runner.invoke(
                app,
                [
                    "data",
                    "from-traces",
                    "--logs",
                    "traces",
                    "--format",
                    "soup-serve",
                    "--signal",
                    "thumbs_up",
                    "-o",
                    "prefs.jsonl",
                ],
            )
            assert "Wrote 1 preference pair(s)" in " ".join(strip_ansi(result.output).split()), (
                strip_ansi(result.output)
            )
        finally:
            os.chdir(previous)
