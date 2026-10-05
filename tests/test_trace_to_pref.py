"""Tests for Trace-to-Preference harvester (v0.26.0 Part C)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# LangChain trace parser
# ---------------------------------------------------------------------------


class TestLangChainParser:
    def test_parse_event_with_thumbs_up(self):
        from soup_cli.data.traces.parsers import parse_langchain

        event = {
            "id": "run-1",
            "inputs": {"messages": [{"role": "user", "content": "Hi"}]},
            "outputs": {"generations": [[{"text": "Hello!"}]]},
            "feedback": [{"key": "thumbs", "score": 1}],
        }
        traces = list(parse_langchain([event]))
        assert len(traces) == 1
        assert traces[0].prompt == "Hi"
        assert traces[0].output == "Hello!"
        assert traces[0].signal == "thumbs_up"

    def test_parse_event_with_thumbs_down(self):
        from soup_cli.data.traces.parsers import parse_langchain

        event = {
            "id": "run-2",
            "inputs": {"messages": [{"role": "user", "content": "Q"}]},
            "outputs": {"generations": [[{"text": "Bad"}]]},
            "feedback": [{"key": "thumbs", "score": 0}],
        }
        traces = list(parse_langchain([event]))
        assert traces[0].signal == "thumbs_down"

    def test_parse_skips_missing_outputs(self):
        from soup_cli.data.traces.parsers import parse_langchain

        events = [{"id": "x", "inputs": {"messages": []}, "outputs": None}]
        assert list(parse_langchain(events)) == []


# ---------------------------------------------------------------------------
# OpenAI-style trace parser
# ---------------------------------------------------------------------------


class TestOpenAIParser:
    def test_parse_completion_with_feedback(self):
        from soup_cli.data.traces.parsers import parse_openai

        event = {
            "id": "cmpl-1",
            "messages": [{"role": "user", "content": "Explain"}],
            "choices": [
                {"message": {"role": "assistant", "content": "Answer"}},
            ],
            "feedback": {"rating": "up"},
        }
        traces = list(parse_openai([event]))
        assert len(traces) == 1
        assert traces[0].prompt == "Explain"
        assert traces[0].output == "Answer"
        assert traces[0].signal == "thumbs_up"

    def test_parse_regeneration_chain(self):
        """Two completions with the same prompt → first=rejected, last=chosen."""
        from soup_cli.data.traces.parsers import parse_openai

        events = [
            {
                "id": "a",
                "messages": [{"role": "user", "content": "Q"}],
                "choices": [{"message": {"role": "assistant", "content": "First"}}],
                "regenerated_from": None,
            },
            {
                "id": "b",
                "messages": [{"role": "user", "content": "Q"}],
                "choices": [{"message": {"role": "assistant", "content": "Second"}}],
                "regenerated_from": "a",
            },
        ]
        traces = list(parse_openai(events))
        assert len(traces) == 2


# ---------------------------------------------------------------------------
# Soup serve trace parser
# ---------------------------------------------------------------------------


class TestSoupServeParser:
    def test_parse_soup_serve_dir(self, tmp_path):
        from soup_cli.data.traces.parsers import parse_soup_serve

        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        (trace_dir / "req-1.jsonl").write_text(
            json.dumps(
                {
                    "id": "req-1",
                    "prompt": "What?",
                    "response": "That.",
                    "feedback": {"rating": "up"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        traces = list(parse_soup_serve(str(trace_dir)))
        assert len(traces) == 1
        assert traces[0].prompt == "What?"


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------


class TestPairBuilder:
    def test_build_pairs_from_thumbs(self):
        from soup_cli.data.traces.pair_builder import build_pairs
        from soup_cli.data.traces.parsers import Trace

        traces = [
            Trace(trace_id="a", prompt="Q", output="Good", signal="thumbs_up"),
            Trace(trace_id="b", prompt="Q", output="Bad", signal="thumbs_down"),
        ]
        pairs = list(build_pairs(traces, signal="thumbs_up"))
        assert len(pairs) == 1
        assert pairs[0].prompt == "Q"
        assert pairs[0].chosen == "Good"
        assert pairs[0].rejected == "Bad"

    def test_build_pairs_from_regenerations(self):
        from soup_cli.data.traces.pair_builder import build_pairs
        from soup_cli.data.traces.parsers import Trace

        traces = [
            Trace(trace_id="1", prompt="Q", output="First", signal="regenerated", regen_order=0),
            Trace(trace_id="2", prompt="Q", output="Better", signal="regenerated", regen_order=1),
        ]
        pairs = list(build_pairs(traces, signal="regenerations"))
        assert len(pairs) == 1
        assert pairs[0].chosen == "Better"
        assert pairs[0].rejected == "First"

    def test_build_pairs_from_user_edit(self):
        from soup_cli.data.traces.pair_builder import build_pairs
        from soup_cli.data.traces.parsers import Trace

        traces = [
            Trace(
                trace_id="x", prompt="Q", output="Raw", signal="user_edit", edited_output="Polished"
            ),
        ]
        pairs = list(build_pairs(traces, signal="user_edit"))
        assert len(pairs) == 1
        assert pairs[0].chosen == "Polished"
        assert pairs[0].rejected == "Raw"

    def test_build_pairs_skips_missing_counterpart(self):
        from soup_cli.data.traces.pair_builder import build_pairs
        from soup_cli.data.traces.parsers import Trace

        traces = [
            Trace(trace_id="a", prompt="Q", output="Good", signal="thumbs_up"),
        ]
        pairs = list(build_pairs(traces, signal="thumbs_up"))
        # No rejected counterpart → pair dropped
        assert pairs == []


# ---------------------------------------------------------------------------
# CLI: soup data from-traces
# ---------------------------------------------------------------------------


class TestFromTracesCLI:
    def _write_langchain_log(self, path, prompts_and_ratings):
        """Write a JSONL log where each row is a LangChain run event."""
        rows = []
        for idx, (prompt, output, rating) in enumerate(prompts_and_ratings):
            rows.append(
                {
                    "id": f"run-{idx}",
                    "inputs": {"messages": [{"role": "user", "content": prompt}]},
                    "outputs": {"generations": [[{"text": output}]]},
                    "feedback": [{"key": "thumbs", "score": rating}],
                }
            )
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )

    def test_happy_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logs = tmp_path / "traces.jsonl"
        self._write_langchain_log(
            logs,
            [
                ("Hi", "Hello!", 1),
                ("Hi", "Ugh.", 0),
            ],
        )
        output = tmp_path / "prefs.jsonl"

        result = runner.invoke(
            app,
            [
                "data",
                "from-traces",
                "--logs",
                str(logs),
                "--format",
                "langchain",
                "--signal",
                "thumbs_up",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert output.exists()
        pairs = [json.loads(ln) for ln in output.read_text(encoding="utf-8").splitlines()]
        assert len(pairs) == 1
        assert pairs[0]["chosen"] == "Hello!"
        assert pairs[0]["rejected"] == "Ugh."

    def test_invalid_format_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logs = tmp_path / "x.jsonl"
        logs.write_text("{}\n", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "data",
                "from-traces",
                "--logs",
                str(logs),
                "--format",
                "hacker",
                "--signal",
                "thumbs_up",
                "--output",
                "prefs.jsonl",
            ],
        )
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_invalid_signal_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logs = tmp_path / "x.jsonl"
        logs.write_text("{}\n", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "data",
                "from-traces",
                "--logs",
                str(logs),
                "--format",
                "langchain",
                "--signal",
                "bribery",
                "--output",
                "prefs.jsonl",
            ],
        )
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_path_traversal_on_logs_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside_traces.jsonl"
        outside.write_text("{}\n", encoding="utf-8")
        try:
            result = runner.invoke(
                app,
                [
                    "data",
                    "from-traces",
                    "--logs",
                    str(outside),
                    "--format",
                    "langchain",
                    "--signal",
                    "thumbs_up",
                    "--output",
                    "prefs.jsonl",
                ],
            )
            assert result.exit_code != 0, (result.output, repr(result.exception))
            assert "outside" in result.output.lower() or "cwd" in result.output.lower()
        finally:
            outside.unlink(missing_ok=True)

    def test_path_traversal_on_output_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logs = tmp_path / "x.jsonl"
        logs.write_text("{}\n", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "data",
                "from-traces",
                "--logs",
                str(logs),
                "--format",
                "langchain",
                "--signal",
                "thumbs_up",
                "--output",
                str(tmp_path.parent / "escape.jsonl"),
            ],
        )
        assert result.exit_code != 0, (result.output, repr(result.exception))


# ---------------------------------------------------------------------------
# CLI: soup data review
# ---------------------------------------------------------------------------


class TestReviewCLI:
    def test_review_requires_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["data", "review"])
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_review_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["data", "review", "nope.jsonl"])
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_review_shows_sample(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        prefs = tmp_path / "prefs.jsonl"
        prefs.write_text(
            json.dumps(
                {
                    "prompt": "Q",
                    "chosen": "A",
                    "rejected": "B",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "data",
                "review",
                str(prefs),
                "--sample",
                "1",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Q" in result.output or "prompt" in result.output.lower()


# ---------------------------------------------------------------------------
# Signals / format literals
# ---------------------------------------------------------------------------


class TestEmptyAndMalformed:
    def test_empty_log_yields_no_pairs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logs = tmp_path / "empty.jsonl"
        logs.write_text("", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "data",
                "from-traces",
                "--logs",
                str(logs),
                "--format",
                "langchain",
                "--signal",
                "thumbs_up",
                "--output",
                "prefs.jsonl",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "prefs.jsonl").exists()
        assert (tmp_path / "prefs.jsonl").read_text(encoding="utf-8") == ""

    def test_malformed_lines_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logs = tmp_path / "mixed.jsonl"
        logs.write_text(
            "not valid json\n"
            + json.dumps(
                {
                    "id": "ok",
                    "inputs": {"messages": [{"role": "user", "content": "Q"}]},
                    "outputs": {"generations": [[{"text": "A"}]]},
                    "feedback": [{"key": "thumbs", "score": 1}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "data",
                "from-traces",
                "--logs",
                str(logs),
                "--format",
                "langchain",
                "--signal",
                "thumbs_up",
                "--output",
                "prefs.jsonl",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_lone_thumbs_down_drops_pair(self):
        from soup_cli.data.traces.pair_builder import build_pairs
        from soup_cli.data.traces.parsers import Trace

        traces = [
            Trace(trace_id="a", prompt="Q", output="Bad", signal="thumbs_down"),
        ]
        pairs = list(build_pairs(traces, signal="thumbs_up"))
        assert pairs == []

    def test_parse_soup_serve_non_directory_returns_empty(self, tmp_path):
        from soup_cli.data.traces.parsers import parse_soup_serve

        file_not_dir = tmp_path / "file.txt"
        file_not_dir.write_text("hi", encoding="utf-8")
        assert list(parse_soup_serve(str(file_not_dir))) == []


class TestLiterals:
    def test_format_literal_accepts_known(self):
        from soup_cli.data.traces.pair_builder import SUPPORTED_SIGNALS

        # All 3 launch signals should be in the supported set
        for sig in ("thumbs_up", "regenerations", "user_edit"):
            assert sig in SUPPORTED_SIGNALS

    def test_parser_registry(self):
        from soup_cli.data.traces.parsers import SUPPORTED_FORMATS

        assert "langchain" in SUPPORTED_FORMATS
        assert "openai" in SUPPORTED_FORMATS
        assert "soup-serve" in SUPPORTED_FORMATS
