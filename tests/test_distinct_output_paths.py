"""#1422: an output path that aliases an input must be refused before any write.

Four commands (``ingest``, ``prune-prompt``, ``ra-dit`` and ``data best-of-n``)
opened their output for writing without checking that it was not one of their
inputs, so the write destroyed the input and the command still exited 0.

``ensure_distinct_paths`` is the one shared check. It compares real paths
(``normcase(realpath(...))``) and, for files that exist, ``os.path.samefile``,
so a spelling of the input (``./x``, a case variant on Windows) and a hard link
are refused as well as the identical string. It opens and creates nothing.

This file covers the shared helper and soup ingest.
Other commands have dedicated output-safety test modules.
"""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.paths import ensure_distinct_paths
from tests.conftest import strip_ansi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    # Soup refuses paths outside the working directory, so every test runs
    # inside its own temporary directory and never touches the repository.
    monkeypatch.chdir(tmp_path)


def _write(name: str, text: str = "x\n") -> str:
    # newline="" keeps the bytes exactly as given on Windows, so the
    # byte-identical assertions below compare what was really written.
    with open(name, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return name


def _read(name: str) -> str:
    with open(name, encoding="utf-8", newline="") as fh:
        return fh.read()


def _flat(output: str) -> str:
    # Rich wraps long messages at the terminal width; flatten the lines so an
    # assertion on a phrase does not depend on where the wrap fell.
    return " ".join(strip_ansi(output).split())


def _plain(output: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output (the name the suite's ANSI
    scanner recognises; the same as ``_flat``)."""
    return _flat(output)


# ---------------------------------------------------------------------------
# The shared helper
# ---------------------------------------------------------------------------


class TestEnsureDistinctPaths:
    def test_same_string_is_refused_and_names_both_options(self):
        # The plain case from the issue: --logs is also the default --output.
        _write("a.jsonl")
        with pytest.raises(ValueError) as info:
            ensure_distinct_paths([("--logs", "a.jsonl")], [("--output", "a.jsonl")])
        message = str(info.value)
        # The operator must be told which two options collide.
        assert "--logs" in message
        assert "--output" in message

    def test_dot_slash_spelling_is_refused(self):
        # A different spelling of the same file must not slip past a string compare.
        _write("a.jsonl")
        with pytest.raises(ValueError):
            ensure_distinct_paths([("--logs", "./a.jsonl")], [("--output", "a.jsonl")])

    def test_a_missing_file_spelled_the_same_is_refused(self):
        # Nothing exists yet, so samefile cannot help; the real-path compare must.
        with pytest.raises(ValueError):
            ensure_distinct_paths([("--logs", "no.jsonl")], [("--output", "no.jsonl")])

    @pytest.mark.skipif(os.name != "nt", reason="case-insensitive file system")
    def test_case_variant_is_refused_on_windows(self):
        # On Windows Traces.jsonl and traces.jsonl are one file.
        _write("Traces.jsonl")
        with pytest.raises(ValueError):
            ensure_distinct_paths(
                [("--logs", "Traces.jsonl")], [("--output", "traces.jsonl")]
            )

    @pytest.mark.skipif(os.name != "nt", reason="case-insensitive file system")
    def test_case_variant_of_a_path_that_does_not_exist_yet_is_refused_on_windows(self):
        with pytest.raises(ValueError):
            ensure_distinct_paths([("--logs", "Missing.jsonl")], [("--output", "missing.jsonl")])

    @pytest.mark.skipif(os.name != "nt", reason="case-insensitive file system")
    def test_nested_case_variant_of_a_missing_directory_is_refused_on_windows(self):
        with pytest.raises(ValueError):
            ensure_distinct_paths(
                [("--retriever-config output", "out/R1/stage")],
                [("--generator-config output", "out/r1")],
                reject_nested=True,
            )

    @pytest.mark.requires_symlink
    def test_symlinked_directory_alias_of_a_missing_output_is_refused(self):
        os.mkdir("real")
        os.symlink("real", "link", target_is_directory=True)
        with pytest.raises(ValueError):
            ensure_distinct_paths(
                [("--retriever-config output", "real/r1")],
                [("--generator-config output", "link/r1")],
                reject_nested=True,
            )
        with pytest.raises(ValueError):
            ensure_distinct_paths([("--logs", "real/in.jsonl")], [("--output", "link/in.jsonl")])
        with pytest.raises(ValueError):
            ensure_distinct_paths(
                [("--retriever-config output", "link/r1")],
                [("--generator-config output", "real/r1")],
                reject_nested=True,
            )
        with pytest.raises(ValueError):
            ensure_distinct_paths([("--logs", "link/in.jsonl")], [("--output", "real/in.jsonl")])

    @pytest.mark.requires_hardlink
    def test_hard_link_is_refused(self):
        # A hard link has a different path but is the same file on disk, so only
        # os.path.samefile catches it.
        _write("a.jsonl")
        os.link("a.jsonl", "b.jsonl")
        with pytest.raises(ValueError):
            ensure_distinct_paths([("--input", "a.jsonl")], [("--output", "b.jsonl")])

    def test_distinct_paths_pass(self):
        # The control: the check is about aliasing, not about every pair of paths.
        _write("a.jsonl")
        ensure_distinct_paths([("--logs", "a.jsonl")], [("--output", "b.jsonl")])

    def test_the_check_writes_nothing(self):
        # It is called before any open-for-write, so it must not create or
        # change a file itself.
        _write("a.jsonl", "keep\n")
        ensure_distinct_paths([("--logs", "a.jsonl")], [("--output", "b.jsonl")])
        assert _read("a.jsonl") == "keep\n"
        assert not os.path.exists("b.jsonl")


# ---------------------------------------------------------------------------
# soup ingest
# ---------------------------------------------------------------------------


class TestIngestRefusesItsOwnInput:
    def test_default_output_equal_to_logs_is_refused(self):
        # The issue's first reproduction: without --output, the default
        # traces.jsonl is also --logs. On main this left a 0-byte file and exit 0.
        original = '{"id": "t1", "input": "hi", "output": "hello"}\n'
        _write("traces.jsonl", original)
        result = CliRunner().invoke(
            app, ["ingest", "--source", "langfuse", "--logs", "traces.jsonl"]
        )
        assert result.exit_code != 0, result.output
        out = _flat(result.output)
        assert "--logs" in out
        assert "--output" in out
        # The input survives byte for byte.
        assert _read("traces.jsonl") == original

    def test_a_different_output_still_works(self):
        # The control: a normal run is unchanged by the new check.
        _write("in.jsonl", '{"id": "t1", "input": "hi", "output": "hello"}\n')
        result = CliRunner().invoke(
            app,
            ["ingest", "--source", "langfuse", "--logs", "in.jsonl", "-o", "out.jsonl"],
        )
        assert result.exit_code == 0, result.output
        assert os.path.getsize("out.jsonl") > 0

    def test_dot_slash_spelling_of_logs_is_refused(self):
        # The same file spelled ./traces.jsonl must be refused like traces.jsonl.
        original = '{"id": "t1", "input": "hi", "output": "hello"}\n'
        _write("traces.jsonl", original)
        result = CliRunner().invoke(
            app, ["ingest", "--source", "langfuse", "--logs", "./traces.jsonl"]
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert (
            "--output 'traces.jsonl' is the same file as --logs 'traces.jsonl'; "
            "choose a different --output"
        ) in _plain(result.output)
        assert _read("traces.jsonl") == original

    def test_explicit_output_equal_to_logs_is_refused(self):
        # Not only the default output: -o pointing at --logs is the same bug.
        original = '{"id": "t1", "input": "hi", "output": "hello"}\n'
        _write("e.jsonl", original)
        result = CliRunner().invoke(
            app,
            ["ingest", "--source", "langfuse", "--logs", "e.jsonl", "-o", "e.jsonl"],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert (
            "--output 'e.jsonl' is the same file as --logs 'e.jsonl'; choose a different --output"
        ) in _plain(result.output)
        assert _read("e.jsonl") == original

    def test_a_logs_directory_leaves_an_existing_output_untouched(self):
        # A read failure must never leave a truncated output behind: on main this
        # emptied a good earlier traces.jsonl before the read failed.
        good = '{"id": "t1", "input": "hi", "output": "hello"}\n'
        _write("traces.jsonl", good)
        os.mkdir("some_dir")
        result = CliRunner().invoke(
            app, ["ingest", "--source", "langfuse", "--logs", "some_dir"]
        )
        # A named refusal, not an uncaught FileNotFoundError.
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        assert "--logs 'some_dir' is a directory, not an export file" in _plain(result.output)
        assert _read("traces.jsonl") == good

    def test_a_non_utf8_export_leaves_an_existing_output_untouched(self):
        # Same property for an export the reader cannot decode.
        good = '{"id": "t1", "input": "hi", "output": "hello"}\n'
        _write("traces.jsonl", good)
        with open("bad.jsonl", "wb") as fh:
            fh.write(b"\xff\xfe\x00 not utf-8\n")
        result = CliRunner().invoke(
            app, ["ingest", "--source", "langfuse", "--logs", "bad.jsonl"]
        )
        # A named refusal, not an uncaught UnicodeDecodeError.
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        assert "--logs 'bad.jsonl' is not valid UTF-8 (byte 0)" in _plain(result.output)
        assert _read("traces.jsonl") == good
        # The staging file of the atomic write must be cleaned up too.
        assert [n for n in os.listdir(".") if n.startswith(".soup.")] == []
