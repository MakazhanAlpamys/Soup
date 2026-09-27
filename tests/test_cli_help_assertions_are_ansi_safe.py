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

**A third class exists and is deliberately NOT scanned (issue #987).** A bare
`assert "\\x1b" not in result.output` on ordinary (non-`--help`,
non-highlighted) CLI output is environment-dependent: the command under test
prints through an unpinned module-level `Console()`, so under
`FORCE_COLOR=1 TERM=xterm-256color` Rich styles the *surrounding* panel markup
and the assertion fails even when the untrusted field itself was sanitised
correctly. The fix is per-test Console pinning::

    monkeypatch.setattr(mod, "console", Console(force_terminal=False))

`no_color=True` alone is NOT sufficient — it suppresses colour but does not pin
tty detection, which is why `test_replay.py` carried that exact bug.

**Why this is not automated.** These assertions split roughly half and half
between ones reading CLI output and ones reading a pure-function return
(`_safe_md_cell(...)`, `_yaml_dq(...)`, `str(exc.value)`, a `cleaned` local, a
`messages[0]`). No exact count is quoted here on purpose: four AST definitions
of "such an assertion" gave four different totals, so any number would be an
artefact of its matcher. The ratio survives every definition, and the ratio is
what the argument needs. A scanner rule for this class must tell those two
kinds apart, and at a roughly even split a rule that cannot do so cleanly
false-positives about half of what it sees — the same "a noisy guard gets
deleted" argument made one paragraph up. The split is not academic:
`test_issue627_unknown_config_keys.py` asserts on a local named `out` that *is*
console output (a `Console(file=...)` StringIO), so name-based and call-based
heuristics both misclassify it. That distinction is left to review and to
per-test pinning, not automated here.

Seven were pinned under #987 (`test_replay`, `test_runs`,
`test_review_fixes_v07133`, `test_v07127`, `test_v07136`, `test_v07139`,
`test_issue367_ship_numerics`). `test_issue627_unknown_config_keys.py::
TestTheReportIsSafeForTheTerminal` was the same defect and was fixed
separately in #984.

CI does not go red on these today: no workflow sets `FORCE_COLOR`/`TERM`, and
`CliRunner`/`StringIO` are non-ttys, so Rich disables colour on every runner.
They are latent-until-something-upstream-changes — exactly the shape of the
four incidents named at the top of this docstring.

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

import pytest

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
    "_ANSI.",
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


# ---------------------------------------------------------------------------
# #633 — the same hazard, on subcommands the `--help` scan never reaches.
# ---------------------------------------------------------------------------

#: Commands whose output is **syntax highlighted**. Rich renders these through
#: Pygments, which emits SGR escapes *between* the tokens of one logical line:
#: ``modality``, ``:`` and ``text`` are three tokens, so ``"modality: text"`` is
#: not a substring of the rendered output and ``yaml.safe_load`` rejects
#: ``\x1b``. Plain ``console.print`` output is not affected the same way -- an
#: unstyled message is wrapped in escapes rather than split by them. Two
#: mechanisms are in play here, not one: ``recipes show`` and ``migrate`` render
#: through ``Syntax(...)`` i.e. Pygments, while ``data mix --apply`` uses plain
#: ``console.print`` and is split by Rich's default ``ReprHighlighter`` instead.
#: The scanner keys on the INVOCATION rather than the mechanism, which is why it
#: covers both; the suite's ~200 other raw-output assertions are left alone.
_HIGHLIGHTED_INVOCATIONS = (
    re.compile(r'"recipes"\s*,\s*"show"'),
    re.compile(r"'recipes'\s*,\s*'show'"),
    re.compile(r'"mix"\s*,\s*"--(apply|optimize)"'),
    re.compile(r"'mix'\s*,\s*'--(apply|optimize)'"),
    # `soup migrate` is the third `Syntax(...)` site in `src/`
    # (commands/migrate.py:121). Latent rather than live: no assertion on its
    # output is multi-token today, so this flags nothing now. Added by the
    # maintainer after review so the guard covers every highlighted command
    # rather than the two this issue happened to surface.
    re.compile(r'"migrate"'),
    re.compile(r"'migrate'"),
)

_PARSES_OUTPUT = re.compile(r"(yaml\.safe_load|json\.loads)\s*\(")

#: Broader than ``_RAW_OUTPUT_RE``, which anchors on the exact name
#: ``result.output`` and therefore cannot match ``show_result.output`` -- ``_``
#: is a word character, so ``\b`` never fires before ``result`` there. Real
#: tests name their results ``show_result`` / ``use_result`` all the time.
_ANY_RAW_OUTPUT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.(output|stdout)\b|readouterr\(\)")
_STRING_LITERAL = re.compile(r'"([^"]{2,})"' + r"|'([^']{2,})'")


def _is_multi_token(literal: str) -> bool:
    """A literal that Pygments would split across tokens.

    The hazard is YAML *structure*. ``modality: text`` is three tokens -- key,
    punctuation, value -- with escapes between them, and ``train:`` is two. A
    bare model id (``Qwen/Qwen3.8-27B``) is one token and survives raw, which is
    why neighbouring assertions in the same file kept passing.

    A plain message such as ``"validation failed"`` is also safe: Rich wraps an
    unstyled string in escapes rather than splitting it, so requiring a colon
    keeps the guard on the assertions that actually break rather than every
    substring containing a space.
    """
    return ":" in literal


def _assert_expression(line: str) -> str:
    """Return the tested expression of an ``assert``, dropping its message.

    ``assert "train:" in compact, result.output`` reads raw output only in the
    *failure message*, which is harmless and in fact good practice. Scanning the
    whole line would flag it, so the top-level comma is found (ignoring commas
    inside brackets or strings) and everything after it discarded.
    """
    depth = 0
    quote = ""
    for index, char in enumerate(line):
        if quote:
            if char == quote and line[index - 1 : index] != "\\":
                quote = ""
            continue
        if char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            return line[:index]
    return line


def _assert_statement_lines(tree: ast.AST) -> set[int]:
    """Every line belonging to an `assert` statement, continuations included.

    The scanner decides `is this an assertion?` from `stripped.startswith(
    "assert")`, which is true of the first physical line only. Once the
    formatter wraps a long assertion, the line that actually reads the output
    is a continuation and was dropped one branch after the literal-skip rule.
    Fixing only the skip rule moved the false negative rather than closing it,
    which is why both are needed.
    """
    covered: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        # `node.test` ONLY, never `node.msg`. Reading raw output inside a
        # failure message is harmless and in fact good practice --
        # `_assert_expression` already drops it on a single-line assert, and
        # taking the whole statement span here would re-introduce exactly that
        # false positive one line lower, on the wrapped form. Measured: it
        # flagged `test_recipes_v031.py:468`, a correct assertion, before this
        # was narrowed.
        test = node.test
        covered.update(range(test.lineno, (test.end_lineno or test.lineno) + 1))
    return covered


def _multiline_literal_lines(tree: ast.AST) -> set[int]:
    """Line numbers occupied by a string constant that spans several lines.

    Those are this guard's own synthetic fixtures: adjacent string literals are
    merged by the parser into ONE `ast.Constant`, so a fixture block written as
    several quoted lines has `end_lineno > lineno` and is skipped whole.

    A real assertion the formatter wrapped -- `assert (` on one line and
    `"modality: text" in result.output` on the next -- carries only a
    SINGLE-line constant, so it is not skipped. The rule this replaced skipped
    any line beginning with a quote and could not tell the two apart, which
    made exactly the shape ruff will eventually produce invisible (#635
    follow-up, closed by the maintainer).
    """
    covered: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        end = node.end_lineno or node.lineno
        if end > node.lineno:
            covered.update(range(node.lineno, end + 1))
    return covered


def find_unsafe_highlighted_assertions(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, text)`` for unnormalised reads of highlighted output.

    Scoped per test FUNCTION, like ``find_raw_help_assertions``: a function that
    invokes a syntax-highlighted command and then either feeds raw output to a
    parser or asserts a multi-token substring against it.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover — a broken file fails its own tests
        return []
    lines = source.splitlines()
    literal_lines = _multiline_literal_lines(tree)
    assert_lines = _assert_statement_lines(tree)
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test"):
            continue
        end = node.end_lineno or node.lineno
        body = lines[node.lineno - 1 : end]
        joined = "\n".join(body)
        if not any(pattern.search(joined) for pattern in _HIGHLIGHTED_INVOCATIONS):
            continue

        tainted: set[str] = set()
        for offset, line in enumerate(body):
            stripped = line.strip()
            if (node.lineno + offset) in literal_lines:
                # Inside a MULTI-LINE string constant -- the synthetic fixtures
                # this guard's own tests are built from. Scanning them would
                # flag the examples written to prove the guard works. A
                # single-line literal is deliberately NOT skipped, so a
                # formatter-wrapped assertion stays visible.
                continue
            if _looks_normalised(stripped):
                # Record that this variable is safe, then move on.
                match = _ASSIGN_RE.match(line)
                if match:
                    tainted.discard(match.group(1))
                continue

            match = _ASSIGN_RE.match(line)
            if match and not stripped.startswith("assert"):
                name, rhs = match.group(1), match.group(2)
                derived = _ANY_RAW_OUTPUT_RE.search(rhs) or any(
                    re.search(rf"\b{re.escape(var)}\b", rhs) for var in tainted
                )
                if derived:
                    tainted.add(name)
                    if _PARSES_OUTPUT.search(rhs):
                        offenders.append((node.lineno + offset, stripped[:100]))
                continue

            reads_raw = _ANY_RAW_OUTPUT_RE.search(stripped)
            reads_tainted = any(
                re.search(rf"\b{re.escape(var)}\b", stripped) for var in tainted
            )
            if not (reads_raw or reads_tainted):
                continue
            if _PARSES_OUTPUT.search(stripped):
                offenders.append((node.lineno + offset, stripped[:100]))
                continue
            is_assert_head = stripped.startswith("assert")
            if not (is_assert_head or (node.lineno + offset) in assert_lines):
                continue
            expression = (
                _assert_expression(stripped) if is_assert_head else stripped
            )
            if not (
                _ANY_RAW_OUTPUT_RE.search(expression)
                or any(re.search(rf"\b{re.escape(var)}\b", expression) for var in tainted)
            ):
                continue
            found = _STRING_LITERAL.search(expression)
            literal = (found.group(1) or found.group(2)) if found else ""
            if literal and _is_multi_token(literal):
                offenders.append((node.lineno + offset, stripped[:100]))
    return offenders


# ---------------------------------------------------------------------------
# #1068 — Rich ReprHighlighter splits percentage / number tokens in raw output
# ---------------------------------------------------------------------------

_IDENTIFIER_EXEMPT_RE = re.compile(
    r"""^(?:
        --?[a-zA-Z0-9_-]+                       # CLI flags: --min-acceptance, -v
        |[A-Za-z0-9._-]+/[A-Za-z0-9._-]+        # Repo/model IDs: meta-llama/Llama-3.1-8B
        |[A-Za-z0-9]+-[0-9.]+[A-Za-z0-9-]*      # Model names: Llama-3.1-8B, Qwen2.5-7B
        # Versions: v0.71.26, 1.2.3
        |(?:v[0-9]+\.[0-9]+(?:\.[0-9]+)?|[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9.-]*)
        |[A-Z][0-9]{2,4}                        # Hardware names: H100, A100
    )$""",
    re.VERBOSE,
)

_PERCENT_LITERAL_RE = re.compile(r"""[0-9]+(?:\.[0-9]+)?%""")
_DIGIT_TEXT_RE = re.compile(
    r"""
    (?:
        \b\d+(?:/\d+|\.\d+)?\s+[a-zA-Z]+        # 5 rows, 128 tokens, 2/2 rows, 1/3 prompts
        |
        \b[a-zA-Z]+\s+\d+(?:/\d+|\.\d+)?\b        # Running 2, between 2, line 2
    )
""",
    re.VERBOSE,
)
_QUOTED_SUBSTR_RE = re.compile(r"""(?<!\w)(?:'[^'\n]+'|"[^"\n]+")(?:(?!\w)|$)""")
_KEY_EQUALS_RE = re.compile(r"""\b[a-zA-Z_][a-zA-Z0-9_]{1,30}=\S+""")
_GROUPED_OR_DECIMAL_RE = re.compile(
    r"""(?<![\w./-])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)(?![\w./-])"""
)


def _is_unsafe_cli_literal(s: str) -> bool:
    """Return True if ``s`` contains tokens split by Rich's ReprHighlighter."""
    s_strip = s.strip()
    if not s_strip or _IDENTIFIER_EXEMPT_RE.match(s_strip):
        return False
    if _PERCENT_LITERAL_RE.search(s_strip):
        return True
    if _DIGIT_TEXT_RE.search(s_strip):
        return True
    if _KEY_EQUALS_RE.search(s_strip):
        return True
    if _GROUPED_OR_DECIMAL_RE.search(s_strip):
        return True
    m = _QUOTED_SUBSTR_RE.search(s_strip)
    if m and m.group(0) != s_strip:
        return True
    return False


#: Inline exemption marker for intentional uncoloured assertions (e.g. non-CLI harnesses,
#: mock subprocesses, or tests with console explicitly pinned to no-color under #987).
_ANSI_OK_RE = re.compile(r"#\s*ansi-ok\s*:\s*(.+)$")


def _find_raw_cli_output_statements(
    source: str,
) -> list[tuple[int, int, str, bool]]:
    """Return (start_line, end_line, first_line_text, has_ansi_ok) for raw output asserts."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover — a broken file fails its own tests
        return []
    lines = source.splitlines()
    literal_lines = _multiline_literal_lines(tree)
    records: list[tuple[int, int, str, bool]] = []

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not fn.name.startswith("test"):
            continue

        tainted: set[str] = set()
        stmts = sorted(
            [
                s
                for s in ast.walk(fn)
                if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Assert))
            ],
            key=lambda s: s.lineno,
        )

        for stmt in stmts:
            if stmt.lineno in literal_lines:
                continue

            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                end = stmt.end_lineno or stmt.lineno
                rhs_text = "\n".join(lines[stmt.lineno - 1 : end])
                derived = _ANY_RAW_OUTPUT_RE.search(rhs_text) or any(
                    re.search(rf"\b{re.escape(var)}\b", rhs_text) for var in tainted
                )
                target_names: list[str] = []
                if isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Name):
                            target_names.append(t.id)
                elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) and isinstance(
                    stmt.target, ast.Name
                ):
                    target_names.append(stmt.target.id)

                for name in target_names:
                    if derived and not _looks_normalised(rhs_text):
                        tainted.add(name)
                    elif derived and _looks_normalised(rhs_text):
                        tainted.discard(name)

            elif isinstance(stmt, ast.Assert):
                test = stmt.test
                test_end = test.end_lineno or test.lineno
                test_text = "\n".join(lines[test.lineno - 1 : test_end])
                if stmt.msg:
                    test_text = _assert_expression(test_text)

                if _looks_normalised(test_text):
                    continue

                reads_raw = _ANY_RAW_OUTPUT_RE.search(test_text)
                reads_tainted = any(
                    re.search(rf"\b{re.escape(var)}\b", test_text) for var in tainted
                )
                if not (reads_raw or reads_tainted):
                    continue

                # Negative assertions ("not in") are unproblematic because missing
                # text cannot be split by ANSI
                if " not in " in test_text:
                    continue

                has_unsafe = False
                for sub in ast.walk(test):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        if _is_unsafe_cli_literal(sub.value):
                            has_unsafe = True
                            break

                if has_unsafe:
                    stmt_start = stmt.lineno
                    stmt_end = stmt.end_lineno or stmt.lineno
                    stmt_context = lines[max(0, stmt_start - 2) : stmt_end]
                    has_ansi_ok = any(_ANSI_OK_RE.search(line) for line in stmt_context)
                    first_line = lines[stmt_start - 1].strip()
                    records.append((stmt_start, stmt_end, first_line[:100], has_ansi_ok))

    return records


def find_raw_cli_output_assertions(
    source: str, ignore_exemptions: bool = False
) -> list[tuple[int, str]]:
    """Return ``(lineno, text)`` for unsafe token assertions on CLI output.

    Rich's default ``ReprHighlighter`` styles numbers (such as ``60.0%``,
    ``128 tokens``, ``2/2 rows valid``, ``1,000``, ``2.0``), quoted substrings,
    and ``key=`` pairs inside console output. On colour-capable terminals, SGR
    escape sequences land inside or between those tokens, causing assertions on
    raw output to fail under ``FORCE_COLOR=1 TERM=xterm-256color`` while passing
    on Windows.
    """
    records = _find_raw_cli_output_statements(source)
    if ignore_exemptions:
        return [(start, text) for start, _, text, _ in records]
    return [(start, text) for start, _, text, has_ok in records if not has_ok]


def find_stale_ansi_ok_markers(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, line_text)`` for any ``# ansi-ok:`` marker that does not
    actively suppress an otherwise-flagged raw CLI output assertion."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    literal_lines = _multiline_literal_lines(tree)
    marker_lines: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, 1):
        if idx in literal_lines:
            continue
        if _ANSI_OK_RE.search(line):
            marker_lines.append((idx, line.strip()))

    if not marker_lines:
        return []

    records = _find_raw_cli_output_statements(source)
    stale: list[tuple[int, str]] = []
    for lineno, text in marker_lines:
        matches_any = any(
            start - 1 <= lineno <= end for start, end, _, _ in records
        )
        if not matches_any:
            stale.append((lineno, text))

    return stale


class TestNoRawCliOutputAssertionsInTheSuite:
    def test_every_test_file_normalises_cli_output_tokens(self):
        offenders: list[str] = []
        scanned = 0
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            source = path.read_text(encoding="utf-8", errors="replace")
            scanned += 1
            for lineno, text in find_raw_cli_output_assertions(source):
                offenders.append(f"{path.name}:{lineno}: {text}")
        assert scanned >= 20, f"only {scanned} test files scanned"
        assert not offenders, (
            "An assertion on CLI output reads numbers/percentages/tokens from raw "
            "output without ANSI stripping. Rich's ReprHighlighter styles numbers "
            "and splits them with ANSI escapes on Linux/macOS, so this passes on "
            "Windows and turns CI red on colour terminals (issue #1068). Route the "
            "output through an ANSI-strip helper (e.g. _plain or strip_ansi):\n  "
            + "\n  ".join(offenders)
        )

    def test_no_stale_ansi_ok_markers_in_the_suite(self):
        """Every # ansi-ok: marker must actively suppress an otherwise-flagged
        assertion. A stale marker on safe code is dead debt and must be deleted."""
        stale: list[str] = []
        total_markers = 0
        this_file = pathlib.Path(__file__).resolve()
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            if path.resolve() == this_file:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            markers = find_stale_ansi_ok_markers(source)
            for lineno, text in markers:
                stale.append(f"{path.name}:{lineno}: {text}")
            tree = ast.parse(source)
            lit_lines = _multiline_literal_lines(tree)
            total_markers += sum(
                1
                for idx, line in enumerate(source.splitlines(), 1)
                if idx not in lit_lines and _ANSI_OK_RE.search(line)
            )

        assert total_markers > 0, "no # ansi-ok: markers found in suite — scanner broken?"
        assert not stale, (
            "Found stale `# ansi-ok:` marker(s) that do not suppress any flagged "
            "assertion:\n  " + "\n  ".join(stale)
        )


class TestTheCliOutputScannerCanActuallyFail:
    """A scanner that finds zero offenders in the active suite is only trustworthy
    if it is proven capable of finding both direct, indirect, wrapped, and token-based
    unsafe assertions."""

    DIRECT_BAD = '''
def test_draft_acceptance():
    result = runner.invoke(app, ["measure", "--target", "org/target"])
    assert "60.0%" in result.output
'''

    INDIRECT_BAD = '''
def test_draft_acceptance_indirect():
    result = runner.invoke(app, ["measure", "--target", "org/target"])
    plain = " ".join(result.output.split())
    assert "60.0%" in plain
'''

    WRAPPED_BAD = '''
def test_draft_acceptance_wrapped():
    result = runner.invoke(app, ["measure", "--target", "org/target"])
    assert (
        "60.0%"
        in result.output
    )
'''

    NON_PCT_BAD = '''
def test_merge_row_count():
    result = runner.invoke(app, ["data", "merge"])
    assert "5 rows" in result.output
'''

    KEY_EQUALS_BAD = '''
def test_param_forwarding():
    result = runner.invoke(app, ["train"])
    assert "iterations=2" in result.output
'''

    REAL_PRE_FIX = '''
def test_below_min_acceptance_exits_two(self, runner, in_tmp_cwd, monkeypatch):
    result = runner.invoke(
        app,
        ["measure", "--target", "org/target", "--draft", "org/tiny",
         "--prompts", prompts, "--min-acceptance", "0.6"],
    )
    assert result.exit_code == 2
    assert "60.0%" in result.output
    assert "below" in result.output.lower()
'''

    INLINE_FIXTURE = '''
def test_inline_fixture_handling():
    fixture = """
    assert "60.0%" in result.output
    """
    assert fixture
'''

    def test_it_catches_direct_raw_output_assertion(self):
        found = find_raw_cli_output_assertions(textwrap.dedent(self.DIRECT_BAD))
        assert len(found) == 1, found
        assert "60.0%" in found[0][1]

    def test_it_catches_indirect_whitespace_collapsed_assertion(self):
        found = find_raw_cli_output_assertions(textwrap.dedent(self.INDIRECT_BAD))
        assert len(found) == 1, found
        assert "60.0%" in found[0][1]

    def test_it_catches_wrapped_raw_output_assertion(self):
        found = find_raw_cli_output_assertions(textwrap.dedent(self.WRAPPED_BAD))
        assert len(found) == 1, found
        assert "assert (" in found[0][1] or "60.0%" in found[0][1]

    def test_it_catches_non_percentage_unsafe_literal(self):
        found = find_raw_cli_output_assertions(textwrap.dedent(self.NON_PCT_BAD))
        assert len(found) == 1, found
        assert "5 rows" in found[0][1]

    def test_it_catches_key_equals_unsafe_literal(self):
        found = find_raw_cli_output_assertions(textwrap.dedent(self.KEY_EQUALS_BAD))
        assert len(found) == 1, found
        assert "iterations=2" in found[0][1]

    def test_it_catches_the_real_pre_fix_pattern_from_issue1068(self):
        found = find_raw_cli_output_assertions(textwrap.dedent(self.REAL_PRE_FIX))
        assert len(found) == 1, found
        assert "60.0%" in found[0][1]

    def test_a_normalised_direct_assertion_is_accepted(self):
        good = self.DIRECT_BAD.replace("in result.output", "in _plain(result.output)")
        assert find_raw_cli_output_assertions(textwrap.dedent(good)) == []

    def test_a_normalised_indirect_assertion_is_accepted(self):
        good = self.INDIRECT_BAD.replace(
            'plain = " ".join(result.output.split())',
            "plain = _plain(result.output)",
        )
        assert find_raw_cli_output_assertions(textwrap.dedent(good)) == []

    def test_plain_non_token_assertion_is_not_flagged(self):
        plain = '''
def test_draft_measure():
    result = runner.invoke(app, ["measure", "--target", "org/target"])
    assert "STRONG" in result.output
'''
        assert find_raw_cli_output_assertions(textwrap.dedent(plain)) == []

    def test_exempt_model_identifier_is_not_flagged(self):
        model = '''
def test_adapters_list():
    result = runner.invoke(app, ["adapters", "list"])
    assert "Llama-3.1-8B" in result.output
'''
        assert find_raw_cli_output_assertions(textwrap.dedent(model)) == []

    def test_failure_message_reading_raw_output_is_not_flagged(self):
        msg_only = '''
def test_draft_acceptance_msg():
    result = runner.invoke(app, ["measure", "--target", "org/target"])
    plain = _plain(result.output)
    assert "60.0%" in plain, result.output
'''
        assert find_raw_cli_output_assertions(textwrap.dedent(msg_only)) == []

    def test_multiline_string_fixture_in_test_is_skipped(self):
        assert find_raw_cli_output_assertions(textwrap.dedent(self.INLINE_FIXTURE)) == []

    def test_unparseable_source_does_not_explode(self):
        assert find_raw_cli_output_assertions("def broken(:\n") == []

    @pytest.mark.parametrize("bad_number", ["1,000", "50,000", "0.7", "3.14"])
    def test_it_catches_grouped_and_decimal_numbers(self, bad_number):
        src = f'''
def test_number_output():
    result = runner.invoke(app, ["measure"])
    assert "{bad_number}" in result.output
'''
        found = find_raw_cli_output_assertions(textwrap.dedent(src))
        assert len(found) == 1, found
        assert bad_number in found[0][1]

    def test_ansi_ok_inline_marker_suppresses_offender(self):
        src = '''
def test_uncoloured_stdout():
    result = runner.invoke(app, ["measure"])
    assert "60.0%" in result.output  # ansi-ok: mock subprocess stdout is uncoloured
'''
        assert find_raw_cli_output_assertions(textwrap.dedent(src)) == []
        found = find_raw_cli_output_assertions(textwrap.dedent(src), ignore_exemptions=True)
        assert len(found) == 1
        assert "60.0%" in found[0][1]

    def test_stale_ansi_ok_marker_is_flagged(self):
        src = '''
def test_safe():
    assert "plain text" in result.output  # ansi-ok: unnecessary exemption
'''
        stale = find_stale_ansi_ok_markers(textwrap.dedent(src))
        assert len(stale) == 1
        assert "unnecessary exemption" in stale[0][1]

