"""Ratchet: every registered Typer leaf command must appear in docs/commands.md.

`docs/commands.md` opens with "> The full `soup` command list." When new leaf
commands land, Markdown docs auto-merge quietly across concurrent PRs and the
reference drifts (see #822 — 23 leaves were missing). This test walks the live
Typer app the same way the issue reproduced the gap and asserts coverage,
accepting the page's existing `a|b|c` / `a / b` shorthand and the
`soup advise <data>` → `soup advise run` argv rewrite.
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


def _shorthand_alts(line: str, prefix: tuple[str, ...]) -> list[str]:
    """Return alternate leaf names from a shorthand region after ``prefix``."""
    alts: list[str] = []
    idx = 0
    while True:
        j = line.find("soup ", idx)
        if j < 0:
            break
        toks = line[j + len("soup ") :].split()
        if list(toks[: len(prefix)]) != list(prefix):
            idx = j + 1
            continue
        region: list[str] = []
        for t in toks[len(prefix) :]:
            if t.startswith(("-", "<", "[", "./", '"', "'")):
                break
            if _ARGISH.fullmatch(t):
                break
            if any(
                t.endswith(suf)
                for suf in (".jsonl", ".yaml", ".can", ".gguf", ".json", ".md", ".py")
            ):
                break
            if t in ("|", "/") or _CMDISH.fullmatch(t):
                region.append(t.rstrip("."))
                continue
            break
        blob = " ".join(region).replace("...", "")
        for piece in re.split(r"\s*[|/]\s*", blob):
            piece = piece.strip()
            if not piece or piece == "soup":
                continue
            alts.append(piece.split()[0])
        idx = j + 1
    return alts


def is_documented(path: tuple[str, ...], doc: str) -> bool:
    """True when ``docs/commands.md`` mentions ``soup <path>`` (or shorthand)."""
    if path == ("advise", "run"):
        # cli._rewrite_advise_argv turns documented `soup advise <data>` into
        # `soup advise run <data>`.
        return "soup advise " in doc
    needle = "soup " + " ".join(path)
    if needle in doc:
        return True
    leaf = path[-1]
    prefix = path[:-1]
    for line in doc.splitlines():
        if "soup " not in line:
            continue
        if leaf in _shorthand_alts(line, prefix):
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
