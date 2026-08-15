"""Repo-wide ratchet: a `--help` assertion must never read RAW CLI output.

This failure has turned CI red **four times** — v0.71.26, v0.71.32, v0.71.35 and
v0.73.2, the last one on all nine test cells at once. Each time it was written
down (CLAUDE.md, CONTRIBUTING, a memory note) and each time it happened again,
which is the argument for a check that FAILS rather than another paragraph that
asks people to remember.

**The mechanism, precisely.** Typer renders `--help` through a Rich table and
styles each option name, so on a colour-capable stream Rich emits escapes
*inside* the flag::

    --noise-floor  ->  -\\x1b[0m\\x1b[1;36m-noise\\x1b[0m\\x1b[1;36m-floor

`"--noise-floor" in result.output` then cannot match. **Windows passes**, because
Rich auto-disables colour there — so a green local run on a Windows box proves
nothing about the Linux/macOS cells. That asymmetry is the entire trap.

**Scope is deliberately narrow: `--help` output only.** A flag named inside an
ERROR message is fine and is NOT flagged, because `console.print(f"... --model
...")` interpolates plain text that Rich does not style per-token. Measured
before writing this guard: 13 such error-message assertions exist across 10
files and all of them are correct. Flagging those would make this test noise,
and a noisy guard gets deleted.

At the time of writing this scanner finds **zero** offenders — it is a ratchet,
not a cleanup. `TestTheScannerCanActuallyFail` is therefore load-bearing: a
scanner that has nothing to find must still be shown capable of finding
something, or it is indistinguishable from one that is silently broken.
"""

from __future__ import annotations

import ast
import pathlib
import re
import textwrap

TESTS_DIR = pathlib.Path(__file__).parent

# A quoted CLI flag: "--noise-floor", '--gpus'.
_FLAG_RE = re.compile(r"""["']--[a-z][a-z0-9-]*["']""")
# The raw CLI-output expressions an assertion might read.
_RAW_OUTPUT_RE = re.compile(r"\b(result\.output|\.stdout\b|readouterr\(\))")
# The most reliable signal that a line strips ANSI is the escape byte itself,
# written inline as a regex. Measured across the suite: 31 correct assertions in
# 18 files spell it `re.sub(r"\x1b\[[0-9;]*m", "", result.output)` with no named
# helper at all, so a list of blessed helper NAMES flags all of them as
# offenders. Detecting the escape is name-agnostic and does not police style —
# the project's own "scan, don't hand-write a list" rule applied here.
_ESCAPE_LITERAL_RE = re.compile(r"\\x1b|\\033|\\u001b|\\e\[", re.IGNORECASE)

# Named helpers, kept as a second route for files that wrap the regex away.
_NORMALISERS = (
    "_plain(",
    "_strip_ansi(",
    "strip_ansi(",
    "_clean(",
    "_clean_help(",
    "_ANSI_RE",
    "_ANSI_ESCAPE",
    "no_color",
)


def _looks_normalised(line: str) -> bool:
    """True when ``line`` strips ANSI — inline regex or a named helper.

    Whitespace collapse alone is deliberately NOT normalisation: ``" ".join(
    text.split())`` is exactly the insufficient fix that shipped in v0.73.2 and
    turned CI red.
    """
    if _ESCAPE_LITERAL_RE.search(line):
        return True
    return any(token in line for token in _NORMALISERS)


_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")


def find_raw_help_assertions(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, text)`` for every unsafe `--help` flag assertion.

    Scoped per test FUNCTION: a function that invokes ``--help`` and then
    asserts a quoted flag against text that has not had ANSI stripped.

    **Indirection is tracked, and that is the whole point.** The assertion that
    actually broke CI in v0.73.2 never mentioned ``result.output`` at all::

        plain = " ".join(result.output.split())   # collapses WHITESPACE only
        assert "--noise-floor" in plain           # asserts on the variable

    A first version of this scanner looked for ``result.output`` on the assert
    line, was verified against an invented example, passed — and did **not**
    catch the real commit. So a variable derived from raw output is followed,
    and is treated as unsafe unless its derivation strips ANSI. Whitespace
    collapse alone is exactly the insufficient fix that shipped.

    Returns ``[]`` for source that cannot be parsed, so one malformed file never
    fails the whole guard.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover — a broken file fails its own tests
        return []
    lines = source.splitlines()
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test"):
            continue
        end = node.end_lineno or node.lineno
        body = lines[node.lineno - 1 : end]
        joined = "\n".join(body)
        if '"--help"' not in joined and "'--help'" not in joined:
            continue

        # Variables carrying CLI output that was never ANSI-stripped.
        tainted: set[str] = set()
        for offset, line in enumerate(body):
            stripped = line.strip()

            if not stripped.startswith("assert"):
                match = _ASSIGN_RE.match(line)
                if match:
                    name, rhs = match.group(1), match.group(2)
                    derived = _RAW_OUTPUT_RE.search(rhs) or any(
                        f"{var}" in rhs for var in tainted
                    )
                    if derived and not _looks_normalised(rhs):
                        tainted.add(name)
                    elif derived:
                        tainted.discard(name)
                continue

            if _looks_normalised(line):
                continue
            if not _FLAG_RE.search(stripped):
                continue
            reads_raw = _RAW_OUTPUT_RE.search(stripped)
            reads_tainted = any(
                re.search(rf"\b{re.escape(var)}\b", stripped) for var in tainted
            )
            if reads_raw or reads_tainted:
                offenders.append((node.lineno + offset, stripped[:100]))
    return offenders


class TestNoRawHelpAssertionsInTheSuite:
    def test_every_test_file_normalises_help_output(self):
        offenders: list[str] = []
        scanned = 0
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            source = path.read_text(encoding="utf-8", errors="replace")
            if "--help" not in source:
                continue
            scanned += 1
            for lineno, text in find_raw_help_assertions(source):
                offenders.append(f"{path.name}:{lineno}: {text}")
        assert scanned > 0, "the scan found no --help tests at all — scanner broken?"
        assert not offenders, (
            "A `--help` assertion is reading RAW CLI output. Rich splits flag "
            "names with ANSI escapes on Linux/macOS, so this passes on Windows "
            "and turns CI red everywhere else (it has, four times). Route the "
            "output through an ANSI-strip + whitespace-collapse helper — see "
            "tests/test_v07302.py::_plain.\n  " + "\n  ".join(offenders)
        )

    def test_the_scan_actually_covers_the_suite(self):
        """CONTROL for the control: if `--help` tests stopped being discovered,
        the guard above would pass by scanning nothing."""
        scanned = [
            p.name
            for p in TESTS_DIR.glob("test_*.py")
            if "--help" in p.read_text(encoding="utf-8", errors="replace")
        ]
        assert len(scanned) >= 20, f"only {len(scanned)} files scanned: {scanned[:5]}"


class TestTheScannerCanActuallyFail:
    """The scanner currently finds nothing. That is only meaningful if it is
    shown able to find something — otherwise it is indistinguishable from a
    scanner that silently matches nothing at all."""

    BAD = '''
def test_help_mentions_the_flag():
    result = runner.invoke(app, ["ship", "--help"])
    assert "--noise-floor" in result.output
'''

    #: VERBATIM from `15b57f7`, the commit whose CI went red on all nine test
    #: cells. Not paraphrased — the first version of this scanner passed its
    #: invented example and missed this, because the assertion names a local
    #: variable rather than `result.output`.
    REAL_BROKEN = '''
    def test_the_flag_exists_and_is_documented(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["ship", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain = " ".join(result.output.split())
        assert "--noise-floor" in plain
'''

    def test_it_catches_the_real_commit_that_broke_ci(self):
        """The load-bearing test in this file. Whitespace-collapse via a local
        variable is the shape that shipped; a scanner that misses it is
        decorative."""
        found = find_raw_help_assertions(textwrap.dedent(self.REAL_BROKEN))
        assert len(found) == 1, found
        assert "--noise-floor" in found[0][1]

    def test_it_catches_the_direct_shape_too(self):
        found = find_raw_help_assertions(self.BAD)
        assert len(found) == 1
        assert "--noise-floor" in found[0][1]

    def test_the_actual_repaired_version_is_accepted(self):
        """CONTROL. The shipped fix — same indirection, but ANSI-stripped —
        must NOT be flagged, or the guard would demand rewriting correct code."""
        fixed = textwrap.dedent(self.REAL_BROKEN).replace(
            'plain = " ".join(result.output.split())', "plain = _plain(result.output)"
        )
        assert find_raw_help_assertions(fixed) == []

    def test_a_normalised_assertion_is_accepted(self):
        good = self.BAD.replace("in result.output", "in _plain(result.output)")
        assert find_raw_help_assertions(good) == []

    def test_an_error_message_assertion_is_not_flagged(self):
        """The narrow scope, pinned. A flag named in an ERROR message is plain
        interpolated text that Rich does not style per-token — 13 such
        assertions exist in this suite and every one is correct."""
        err = '''
def test_missing_model_is_reported():
    result = runner.invoke(app, ["deploy", "ollama"])
    assert result.exit_code == 1
    assert "--model" in result.output
'''
        assert find_raw_help_assertions(err) == []

    def test_an_assertion_without_a_flag_is_not_flagged(self):
        plain = '''
def test_help_renders():
    result = runner.invoke(app, ["ship", "--help"])
    assert "SHIP" in result.output
'''
        assert find_raw_help_assertions(plain) == []

    def test_unparseable_source_does_not_explode(self):
        assert find_raw_help_assertions("def broken(:\n") == []
