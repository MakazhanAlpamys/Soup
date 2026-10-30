"""#2225: the soup data scoring commands must not write --output over their input.

``soup data decontaminate``, ``toxicity``, ``langdetect``, ``pii`` and
``educational`` read every row of ``--input`` and then replace ``--output``.
Each default output name (``clean.jsonl``, ``toxicity.jsonl``,
``langdetect.jsonl``, ``pii.jsonl``, ``educational.jsonl``) is also an ordinary
input name, so the plain form of the command replaced the file it was given and
still exited 0. An explicit ``-o`` that spelled the input another way (``./``,
absolute, ``sub/..``, a case variant, a trailing space or dot, ``::$DATA``, a
junction, a hard link) did the same, and ``decontaminate`` could write its
default output over ``--benchmark-file``.

The commands now call the shared ``ensure_distinct_paths`` check (#1422) before
anything is read or written: the run exits 1, names both options and the fix,
and leaves every file byte-identical. A distinct output works as before.
"""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import data_score
from tests.conftest import strip_ansi

# Four rows a scorer accepts, plus a line that is not JSON and a blank line:
# the loader skips both, so a run that rewrote the input would lose them even
# for langdetect and educational, which keep every row.
ROWS = (
    '{"instruction": "Say hello", "input": "", "output": "Hello there"}\n'
    '{"instruction": "Mail the report", "input": "", '
    '"output": "Sent it to alice@example.com today"}\n'
    "not json at all\n"
    "\n"
    '{"text": "Photosynthesis converts light energy into chemical energy in plants."}\n'
    '{"text": "the mitochondria is the powerhouse of the cell and makes energy"}\n'
)
BENCHMARK = '{"text": "the mitochondria is the powerhouse of the cell and makes energy"}\n'
NO_TEXT = '{"id": 1}\n{"id": 2}\n'

DECONTAMINATE_EXTRA = ("--benchmark-file", "bench.jsonl")

# (command, default --output, the arguments a successful run also needs)
COMMANDS = [
    pytest.param("decontaminate", "clean.jsonl", DECONTAMINATE_EXTRA, id="decontaminate"),
    pytest.param("toxicity", "toxicity.jsonl", (), id="toxicity"),
    pytest.param("langdetect", "langdetect.jsonl", (), id="langdetect"),
    pytest.param("pii", "pii.jsonl", (), id="pii"),
    pytest.param("educational", "educational.jsonl", (), id="educational"),
]

ABSOLUTE = "<absolute>"  # replaced by the absolute spelling of the file in the test


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    # Soup refuses paths outside the working directory, so each test runs in
    # its own temporary directory.
    monkeypatch.chdir(tmp_path)
    # The regex detector only: no optional Presidio models.
    monkeypatch.setattr(data_score, "_presidio_pii", lambda text: None)
    _write("bench.jsonl", BENCHMARK)


def _write(name: str, text: str) -> str:
    # Binary, so the bytes on disk are exactly ROWS on every OS.
    with open(name, "wb") as fh:
        fh.write(text.encode("utf-8"))
    return name


def _bytes(name: str) -> bytes:
    with open(name, "rb") as fh:
        return fh.read()


def _listing(directory: str = ".") -> list[str]:
    return sorted(os.listdir(directory))


def _run(command: str, *args: str):
    return CliRunner().invoke(app, ["data", command, *args])


def _plain(output: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output."""
    return " ".join(strip_ansi(output).split())


def _squeezed(text: str) -> str:
    # Rich folds a long path across lines at the console width; dropping all
    # whitespace from both sides compares the message wherever the fold fell.
    return "".join(strip_ansi(text).split())


def _assert_refused(result, output: str, field: str, given: str) -> None:
    assert result.exit_code == 1, (result.output, repr(result.exception))
    expected = (
        f"--output '{output}' is the same file as {field} '{given}'; "
        "choose a different --output"
    )
    assert _squeezed(expected) in _squeezed(result.output), _plain(result.output)
    assert "choose a different --output" in _plain(result.output)


class TestDefaultOutputNamedLikeTheInput:
    """The issue's first reproduction: no -o, and the input has the default name."""

    @pytest.mark.parametrize("command,default,extra", COMMANDS)
    def test_is_refused_and_the_input_is_unchanged(self, command, default, extra):
        _write(default, ROWS)
        before = _listing()

        result = _run(command, "-i", default, *extra)

        _assert_refused(result, default, "--input", default)
        assert _bytes(default) == ROWS.encode("utf-8")
        # Nothing was created either: no output, no staging file.
        assert _listing() == before

    @pytest.mark.parametrize("command,default,extra", COMMANDS)
    def test_is_refused_before_the_input_is_read(self, command, default, extra):
        # A file with no extractable text fails while it is read; the alias
        # refusal must come first, so its message is the one printed.
        _write(default, NO_TEXT)

        result = _run(command, "-i", default, *extra)

        _assert_refused(result, default, "--input", default)
        assert "no extractable text" not in _plain(result.output)
        assert _bytes(default) == NO_TEXT.encode("utf-8")


class TestExplicitOutputThatSpellsTheInput:
    @pytest.mark.parametrize("command,default,extra", COMMANDS)
    @pytest.mark.parametrize(
        "spelling", ["d.jsonl", "./d.jsonl", "sub/../d.jsonl", ABSOLUTE]
    )
    def test_portable_spelling_is_refused(self, command, default, extra, spelling):
        _write("d.jsonl", ROWS)
        os.mkdir("sub")  # so sub/../d.jsonl is a path the write could really use
        if spelling == ABSOLUTE:
            spelling = os.path.join(os.getcwd(), "d.jsonl")
        before = _listing()

        result = _run(command, "-i", "d.jsonl", "-o", spelling, *extra)

        _assert_refused(result, spelling, "--input", "d.jsonl")
        assert _bytes("d.jsonl") == ROWS.encode("utf-8")
        assert _listing() == before

    @pytest.mark.skipif(os.name != "nt", reason="Windows file-name rules")
    @pytest.mark.parametrize("command,default,extra", COMMANDS)
    @pytest.mark.parametrize(
        "spelling", ["D.JSONL", "d.jsonl ", "d.jsonl.", "d.jsonl::$DATA"]
    )
    def test_windows_spelling_is_refused(self, command, default, extra, spelling):
        # A case variant, a trailing space, a trailing dot and the default data
        # stream all open d.jsonl on NTFS.
        _write("d.jsonl", ROWS)
        before = _listing()

        result = _run(command, "-i", "d.jsonl", "-o", spelling, *extra)

        _assert_refused(result, spelling, "--input", "d.jsonl")
        assert _bytes("d.jsonl") == ROWS.encode("utf-8")
        assert _listing() == before

    @pytest.mark.skipif(os.name != "nt", reason="Windows junctions only exist on Windows")
    @pytest.mark.parametrize("command,default,extra", COMMANDS)
    def test_output_through_a_junction_is_refused(self, command, default, extra):
        import _winapi

        os.mkdir("real")
        _write(os.path.join("real", "d.jsonl"), ROWS)
        _winapi.CreateJunction(os.path.abspath("real"), os.path.abspath("jn"))
        before = _listing("real")

        result = _run(command, "-i", "real/d.jsonl", "-o", "jn/d.jsonl", *extra)

        _assert_refused(result, "jn/d.jsonl", "--input", "real/d.jsonl")
        assert _bytes(os.path.join("real", "d.jsonl")) == ROWS.encode("utf-8")
        assert _listing("real") == before

    @pytest.mark.requires_hardlink
    @pytest.mark.parametrize("command,default,extra", COMMANDS)
    def test_output_that_is_a_hard_link_of_the_input_is_refused(
        self, command, default, extra
    ):
        # Two names, one file: only os.path.samefile sees it. The write swaps a
        # new file in under d.jsonl, so both names must be checked.
        _write("d.jsonl", ROWS)
        os.link("d.jsonl", "hl.jsonl")

        result = _run(command, "-i", "hl.jsonl", "-o", "d.jsonl", *extra)

        _assert_refused(result, "d.jsonl", "--input", "hl.jsonl")
        assert _bytes("d.jsonl") == ROWS.encode("utf-8")
        assert _bytes("hl.jsonl") == ROWS.encode("utf-8")
        assert os.path.samefile("d.jsonl", "hl.jsonl")


class TestDecontaminateBenchmarkFile:
    def test_default_output_named_like_the_benchmark_file_is_refused(self):
        # The issue's example: --benchmark-file clean.jsonl and no -o.
        _write("train.jsonl", ROWS)
        _write("clean.jsonl", BENCHMARK)

        result = _run(
            "decontaminate", "-i", "train.jsonl", "--benchmark-file", "clean.jsonl"
        )

        _assert_refused(result, "clean.jsonl", "--benchmark-file", "clean.jsonl")
        assert _bytes("clean.jsonl") == BENCHMARK.encode("utf-8")
        assert _bytes("train.jsonl") == ROWS.encode("utf-8")

    @pytest.mark.parametrize("spelling", ["bench.jsonl", "./bench.jsonl", ABSOLUTE])
    def test_explicit_output_that_spells_the_benchmark_file_is_refused(self, spelling):
        _write("train.jsonl", ROWS)
        if spelling == ABSOLUTE:
            spelling = os.path.join(os.getcwd(), "bench.jsonl")
        before = _listing()

        result = _run(
            "decontaminate", "-i", "train.jsonl",
            "--benchmark-file", "bench.jsonl", "-o", spelling,
        )

        _assert_refused(result, spelling, "--benchmark-file", "bench.jsonl")
        assert _bytes("bench.jsonl") == BENCHMARK.encode("utf-8")
        assert _listing() == before


class TestDistinctOutputStillWorks:
    """The control: the check is about aliasing, not about every pair of paths."""

    @pytest.mark.parametrize("command,default,extra", COMMANDS)
    @pytest.mark.parametrize("output", ["out.jsonl", "d.json"], ids=["other-name", "other-ext"])
    def test_distinct_output_is_written(self, command, default, extra, output):
        _write("d.jsonl", ROWS)

        result = _run(command, "-i", "d.jsonl", "-o", output, *extra)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert os.path.isfile(output)
        assert _bytes("d.jsonl") == ROWS.encode("utf-8")
        assert _bytes("bench.jsonl") == BENCHMARK.encode("utf-8")

    @pytest.mark.parametrize("command,default,extra", COMMANDS)
    def test_default_output_beside_a_differently_named_input_is_written(
        self, command, default, extra
    ):
        _write("d.jsonl", ROWS)

        result = _run(command, "-i", "d.jsonl", *extra)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert os.path.isfile(default)
        assert _bytes("d.jsonl") == ROWS.encode("utf-8")
