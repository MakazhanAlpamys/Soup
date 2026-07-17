"""v0.71.37 — every printed `pip install soup-cli[extra]` hint must be cmd.exe-safe.

The single-quoted spelling is bash / zsh / PowerShell syntax. Windows
`cmd.exe` has no single-quote quoting: it hands the quotes to pip verbatim and
pip rejects the requirement outright:

    ERROR: Invalid requirement: "'soup-cli[train]'": Expected package name at
    the start of dependency specifier

Measured on Windows before this fix:

    cmd.exe      'soup-cli[train]' -> ERROR   "soup-cli[train]" -> ok   bare -> ok
    PowerShell   'soup-cli[train]' -> ok      "soup-cli[train]" -> ok

Double quotes are the only spelling that works in *every* shell, which is why
the repo already says `pip install -e ".[dev]"`. Bare `soup-cli[train]` is not
an option: zsh globs the bracket and fails with `no matches found`.

Nothing in Soup can rescue the command once it is typed — pip and the shell own
it, and Soup is not installed yet when the README command runs. The only lever
is what we print, so these tests guard that.
"""

import pathlib
import re

import pytest

# `pip install "soup-cli[x]"` and the Rich-escaped `pip install "soup-cli\[x]"`.
# `\\*` (any run of backslashes), NOT `\\?`: the escaped hint carries TWO
# backslash characters in the source, so `\\?` silently misses every Rich hint —
# including the `soup ui` one that started this — and the guard passes vacuously.
SINGLE_QUOTED = re.compile(r"pip install 'soup-cli\\*\[[a-z][a-z0-9-]*\]'")


def _package_root() -> pathlib.Path:
    import soup_cli

    return pathlib.Path(soup_cli.__file__).parent


def _repo_root() -> pathlib.Path | None:
    """src-layout: <repo>/src/soup_cli. None when installed as a plain wheel."""
    root = _package_root().parent.parent
    return root if (root / "pyproject.toml").is_file() else None


class TestHintsAreCmdSafe:
    def test_no_single_quoted_hint_in_package(self):
        """Nothing under src/soup_cli may print the cmd-hostile spelling."""
        root = _package_root()
        offenders = []
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if SINGLE_QUOTED.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert not offenders, (
            "single-quoted soup-cli[extra] hint — cmd.exe passes the quotes to "
            'pip and it errors out. Use \\"soup-cli[extra]\\" (works in every '
            "shell):\n" + "\n".join(offenders)
        )

    def test_no_single_quoted_hint_in_docs_code_blocks(self):
        """No fenced block may hand a reader a command that dies on cmd.exe.

        Scoped to fenced code blocks on purpose: those are what people copy.
        Prose is allowed to *name* the broken spelling — the README note does
        exactly that, so someone who followed an older video recognises their
        error and can search for it.
        """
        repo = _repo_root()
        if repo is None:
            pytest.skip("not an editable/source checkout — docs/ unavailable")
        targets = [repo / "README.md", *sorted((repo / "docs").rglob("*.md"))]
        offenders = []
        for path in targets:
            if not path.is_file():
                continue
            # Plans are a historical record of what was decided, not advice.
            if "superpowers/plans" in path.as_posix():
                continue
            rel = path.relative_to(repo).as_posix()
            in_fence = False
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if line.lstrip().startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence and SINGLE_QUOTED.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert not offenders, (
            "single-quoted soup-cli[extra] in a docs code block — a Windows "
            "cmd.exe reader copies this and gets `Invalid requirement`:\n"
            + "\n".join(offenders)
        )

    def test_regex_actually_matches_the_broken_spelling(self):
        """Guard the guard: a regex that matches nothing would pass vacuously.

        The first cut used `\\\\?` and matched at most ONE backslash, so it saw
        the plain `'soup-cli[fast]'` sites but not a single Rich-escaped one --
        i.e. it would have gone green while `soup ui` still printed a command
        that dies on cmd.exe. Both spellings are pinned here on purpose.
        """
        # Plain (YAML template comments, plain exception text).
        assert SINGLE_QUOTED.search("pip install 'soup-cli[train]'")
        # Rich-escaped, exactly as it sits in ui.py: TWO backslash chars.
        assert SINGLE_QUOTED.search(
            '"Install with: [bold]pip install \'soup-cli\\\\[ui]\'[/]"'
        )
        # The spellings we are migrating *to* must not be flagged.
        assert not SINGLE_QUOTED.search('pip install "soup-cli[train]"')
        assert not SINGLE_QUOTED.search('pip install "soup-cli\\\\[ui]"')
        assert not SINGLE_QUOTED.search("pip install soup-cli")


class TestDoubleQuotesSurviveRich:
    """The bracket still needs escaping — double quotes do not change that."""

    def test_rich_keeps_escaped_bracket_inside_double_quotes(self):
        from io import StringIO

        from rich.console import Console

        def render(markup: str) -> str:
            buf = StringIO()
            Console(file=buf, force_terminal=False, width=100).print(markup)
            return buf.getvalue().strip()

        assert render('[bold]pip install "soup-cli\\[ui]"[/]') == (
            'pip install "soup-cli[ui]"'
        ), "escaped bracket must survive inside double quotes"
        assert render('[bold]pip install "soup-cli[ui]"[/]') == (
            'pip install "soup-cli"'
        ), "unescaped bracket is still eaten — quoting does not fix escaping"
