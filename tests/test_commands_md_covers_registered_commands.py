"""Ratchet: every registered Typer leaf command must appear in docs/commands.md.

`docs/commands.md` opens with "> The full `soup` command list." When new leaf
commands land, Markdown docs auto-merge quietly across concurrent PRs and the
reference drifts (see #822 — 23 leaves were missing). This test walks the live
Typer app the same way the issue reproduced the gap and asserts coverage,
accepting the page's existing `a|b|c` / `a / b` shorthand and the
`soup advise <data>` → `soup advise run` argv rewrite.

Matching rules (mutation-tested):
- Only lines that *start* with ``soup `` count as documentation (prose mentions
  do not).
- Only the command column (pre-description separator) counts for coverage;
  prose descriptions mentioning other commands cannot satisfy their requirement (#1046).
- A command needle must end on a token boundary ``(?![\\w.-])`` so
  ``soup eval gate-install`` or ``soup eval gate.v2`` cannot cover ``soup eval gate``.
- Shorthand alternatives are one ``|``/``/``-connected group after the path
  prefix (not adjacent bare tokens), so ``soup llama …|quantize`` cannot cover
  top-level ``soup quantize``.
"""

from __future__ import annotations

import pathlib
import re
from typing import Iterable

import pytest
from typer.core import TyperGroup
from typer.main import get_command

from soup_cli.cli import app

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMANDS_MD = ROOT / "docs" / "commands.md"

_ARGISH = re.compile(r"^[A-Z][A-Z0-9_-]*$")
_CMDISH = re.compile(r"^[a-z0-9|._-]+$")


def iter_leaf_paths(cmd=None, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Recursively collect leaf command paths from the Typer/Click app."""
    if cmd is None:
        cmd = get_command(app)
    leaves: list[tuple[str, ...]] = []
    if isinstance(cmd, TyperGroup):
        names = list(cmd.list_commands(None) or [])
        if names:
            for name in names:
                sub = cmd.get_command(None, name)
                if sub is not None:
                    leaves.extend(iter_leaf_paths(sub, prefix + (name,)))
            return leaves
    if prefix:
        leaves.append(prefix)
    return leaves


def _entry_rows(doc: str) -> list[str]:
    """Return command-list rows: lines that start with ``soup ``."""
    return [line for line in doc.splitlines() if line.startswith("soup ")]


def _command_column(line: str) -> str:
    """Return the command portion of an entry row (before the description)."""
    parts = re.split(r"\s{2,}", line, maxsplit=1)
    return parts[0].strip()


def _shorthand_alts(line: str, prefix: tuple[str, ...]) -> list[str]:
    """Return alternate leaf names from one shorthand group after ``prefix``.

    A shorthand group is a single argv slot (``a|b|c``) or a ``|`` / ``/``-
    connected run (``a | b | c``, ``a / b``). Adjacent bare command tokens
    without a separator — e.g. ``llama`` then ``cli|…|quantize`` — are *not*
    one group, so top-level ``soup quantize`` cannot ride a ``soup llama`` row.
    """
    if not line.startswith("soup "):
        return []
    toks = line[len("soup ") :].split()
    if list(toks[: len(prefix)]) != list(prefix):
        return []
    rest = toks[len(prefix) :]
    region: list[str] = []
    expecting_name = True
    for t in rest:
        if expecting_name:
            if t.startswith(("-", "<", "[", "./", '"', "'")):
                break
            if _ARGISH.fullmatch(t):
                break
            if any(
                t.endswith(suf)
                for suf in (
                    ".jsonl",
                    ".yaml",
                    ".can",
                    ".gguf",
                    ".json",
                    ".md",
                    ".py",
                )
            ):
                break
            if t in ("|", "/"):
                break
            cleaned = t.rstrip(".")
            if not _CMDISH.fullmatch(cleaned):
                break
            region.append(cleaned)
            expecting_name = False
        else:
            if t in ("|", "/"):
                region.append(t)
                expecting_name = True
                continue
            break
    blob = " ".join(region).replace("...", "")
    alts: list[str] = []
    for piece in re.split(r"\s*[|/]\s*", blob):
        piece = piece.strip()
        if not piece or piece == "soup":
            continue
        alts.append(piece.split()[0])
    return alts


def is_documented(path: tuple[str, ...], doc: str) -> bool:
    """True when ``docs/commands.md`` has a real entry row for ``soup <path>``."""
    if path == ("advise", "run"):
        # cli._rewrite_advise_argv turns documented `soup advise <data>` into
        # `soup advise run <data>`. Anchor on `<` so compare/explain rows cannot
        # satisfy the special case.
        return any(
            _command_column(line).startswith("soup advise <")
            for line in _entry_rows(doc)
        )
    needle = "soup " + " ".join(path)
    boundary = re.compile(r"^" + re.escape(needle) + r"(?![\w.-])")
    leaf = path[-1]
    prefix = path[:-1]
    for line in _entry_rows(doc):
        cmd = _command_column(line)
        if boundary.search(cmd):
            return True
        if leaf in _shorthand_alts(cmd, prefix):
            return True
    return False


def undocumented_commands(
    doc: str, leaves: Iterable[tuple[str, ...]] | None = None
) -> list[str]:
    """Return `soup …` paths missing from ``doc``."""
    if leaves is None:
        leaves = iter_leaf_paths()
    return ["soup " + " ".join(p) for p in leaves if not is_documented(p, doc)]


@pytest.mark.unit
class TestCommandsMdCoversRegisteredCommands:
    """Live Typer leaf paths must stay represented in the command reference."""

    def test_every_registered_leaf_is_mentioned_in_commands_md(self) -> None:
        doc = COMMANDS_MD.read_text(encoding="utf-8")
        missing = undocumented_commands(doc)
        assert not missing, (
            "docs/commands.md claims to be the full command list but omits:\n  - "
            + "\n  - ".join(missing)
        )

    def test_leaf_walk_is_non_trivial(self) -> None:
        leaves = iter_leaf_paths()
        assert len(leaves) >= 200
        assert ("eval", "against") in leaves
        assert ("llama", "quantize") in leaves
        assert ("mcp", "runs", "reconcile") in leaves

    def test_advise_run_special_case_tracks_live_typer_tree(self) -> None:
        leaves = iter_leaf_paths()
        assert ("advise", "run") in leaves, (
            "the live Typer tree no longer registers `soup advise run`, but "
            "is_documented() still special-cases it — delete the special case "
            "with the command"
        )

    def test_every_entry_row_has_two_or_more_space_separator(self) -> None:
        """Entry rows must cleanly separate the command column from description."""
        doc = COMMANDS_MD.read_text(encoding="utf-8")
        allowlist = {
            'soup mcp serve --transport sse --host 127.0.0.1 --port 8765 --auth-token "$TOKEN"',
        }
        violations = [
            line
            for line in _entry_rows(doc)
            if len(re.split(r"\s{2,}", line, maxsplit=1)) == 1
            and line not in allowlist
        ]
        assert not violations, (
            "Entry rows must separate command column from description with 2+ spaces:\n  - "
            + "\n  - ".join(violations)
        )


@pytest.mark.unit
class TestCommandsMdCoverageGuardHasTeeth:
    """CONTROL: the auditor must fail when a known leaf is stripped from the doc."""

    def test_stripped_eval_against_is_reported(self) -> None:
        doc = COMMANDS_MD.read_text(encoding="utf-8")
        scrubbed = doc.replace("soup eval against", "soup eval __against__")
        # Keep other commands intact; only hide the against needle + shorthand.
        missing = undocumented_commands(scrubbed, leaves=[("eval", "against")])
        assert missing == ["soup eval against"]

    def test_llama_quantize_shorthand_counts(self) -> None:
        stub = (
            "soup llama cli|mtmd-cli|gguf-split|server|quantize ... "
            "Proxy to the llama.cpp binaries\n"
        )
        assert undocumented_commands(stub, leaves=[("llama", "quantize")]) == []
        stub_old = (
            "soup llama cli|mtmd-cli|gguf-split|server ... "
            "Proxy to the llama.cpp binaries\n"
        )
        assert undocumented_commands(stub_old, leaves=[("llama", "quantize")]) == [
            "soup llama quantize"
        ]

    def test_stripped_advise_data_row_is_reported(self) -> None:
        doc = COMMANDS_MD.read_text(encoding="utf-8")
        scrubbed = "\n".join(
            line
            for line in doc.splitlines()
            if not line.startswith("soup advise <")
        )
        missing = undocumented_commands(scrubbed, leaves=[("advise", "run")])
        assert missing == ["soup advise run"]

    @pytest.mark.parametrize(
        ("target", "cmd_path"),
        [
            ("soup serve", ("serve",)),
            ("soup export", ("export",)),
            ("soup train", ("train",)),
            ("soup bom emit", ("bom", "emit")),
        ],
    )
    def test_stripped_command_whose_name_appears_in_other_descriptions_is_reported(
        self, target: str, cmd_path: tuple[str, ...]
    ) -> None:
        doc = COMMANDS_MD.read_text(encoding="utf-8")
        scrubbed = "\n".join(
            line for line in doc.splitlines() if not line.startswith(target)
        )
        missing = undocumented_commands(scrubbed, leaves=[cmd_path])
        assert missing == [target]


@pytest.mark.unit
class TestCoverageRequiresARealEntry:
    """Stub-based controls so matcher regressions fail without the live document."""

    def test_longer_command_does_not_document_its_prefix(self) -> None:
        stub = "soup eval gate-install --baseline X  Install gate\n"
        assert undocumented_commands(stub, leaves=[("eval", "gate")]) == [
            "soup eval gate"
        ]

    def test_prose_mention_does_not_count_as_documentation(self) -> None:
        stub = "See `soup ui` in the paragraph below for details.\n"
        assert undocumented_commands(stub, leaves=[("ui",)]) == ["soup ui"]

    def test_shorthand_does_not_leak_across_argv_slots(self) -> None:
        stub = (
            "soup llama cli|mtmd-cli|gguf-split|server|quantize ... "
            "Proxy to the llama.cpp binaries\n"
        )
        assert undocumented_commands(stub, leaves=[("quantize",)]) == [
            "soup quantize"
        ]
        assert undocumented_commands(stub, leaves=[("llama", "quantize")]) == []

    def test_mention_in_another_entry_row_description_does_not_credit_command(
        self,
    ) -> None:
        stub = (
            "soup draft list                               "
            "List local drafts that soup serve --auto-spec will pick up\n"
        )
        assert undocumented_commands(stub, leaves=[("serve",)]) == ["soup serve"]
        assert undocumented_commands(stub, leaves=[("draft", "list")]) == []

    def test_shorthand_in_description_does_not_credit_command(self) -> None:
        stub = (
            "soup steer apply --name <id> --strength <s>  "
            "Preview a stored steering vector; soup steer list lists them\n"
        )
        assert undocumented_commands(stub, leaves=[("steer", "list")]) == [
            "soup steer list"
        ]
        assert undocumented_commands(stub, leaves=[("steer", "apply")]) == []

    def test_dot_lookahead_prevents_prefix_credit(self) -> None:
        stub = "soup eval gate.v2 --baseline X  Install gate v2\n"
        assert undocumented_commands(stub, leaves=[("eval", "gate")]) == [
            "soup eval gate"
        ]

    def test_single_space_row_does_not_credit_contained_command(self) -> None:
        stub = "soup draft list List drafts that soup serve --auto-spec picks up\n"
        assert undocumented_commands(stub, leaves=[("serve",)]) == ["soup serve"]
        assert undocumented_commands(stub, leaves=[("draft", "list")]) == []
