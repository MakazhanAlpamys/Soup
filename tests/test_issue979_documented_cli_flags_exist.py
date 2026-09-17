"""#979 — docs/commands.md and docs/training.md documented four
``soup train --minillm-*`` flags that were never wired as CLI options; only
``training.minillm_enabled`` and friends exist, as ``soup.yaml`` fields. A
user who copied the documented command got ``No such option`` and exit 2.

Nothing checked documented flags against the live Typer tree, so this guards
the class of bug rather than just the four instances: it walks ``soup
train``'s real options and fails if a doc names one that does not exist.

Scoped to ``soup train`` and to ``docs/commands.md`` / ``docs/training.md``,
the files #979 named — not the whole CLI surface. Building this turned up
several more pre-existing mismatches (ULD strategy, RL mid-epoch checkpoint
flags, echo-trap flags, ``--output``) that are a different bug in different
subsystems; fixing them here would turn a doc fix for MiniLLM into an
unreviewable diff across three unrelated features. They are named in
``_PRE_EXISTING_TRAIN_FLAG_MISMATCHES`` below, tracked in #997, and excluded
from the guard until #997 fixes them one way or the other.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer

from soup_cli.cli import app

DOCS = Path(__file__).parents[1] / "docs"

# soup train --flag documented in docs/*.md that is not a real Typer option
# today, and is not #979's concern. Tracked in #997. If a flag here starts
# passing (someone wires it) remove it -- test_allowlist_entries_are_still_
# actually_missing below catches the alternative silent drift.
_PRE_EXISTING_TRAIN_FLAG_MISMATCHES = {
    "--uld-strategy",
    "--rl-checkpoint-save-every-steps",
    "--rl-checkpoint-keep-last",
    "--rl-checkpoint-include-optimizer",
    "--echo-trap-enabled",
    "--echo-trap-threshold",
    "--echo-trap-halt",
    "--output",
}


def _train_command_options() -> set[str]:
    """The real --flag strings ``soup train`` accepts, from the live Typer tree."""
    cli = typer.main.get_command(app)
    train_cmd = cli.commands["train"]
    return {opt for param in train_cmd.params for opt in param.opts if opt.startswith("--")}


def _documented_train_flags(markdown: str) -> dict[str, list[str]]:
    """Map each --flag documented in a ``soup train`` invocation to the
    (possibly backslash-continued) command line it appeared in.

    ``docs/commands.md`` puts a free-text description after 2+ spaces on the
    same line, and that description can itself name another command's flags
    (e.g. `` `soup bom emit --energy <energy.json>` `` describing what to do
    with `soup train`'s own output) -- so each line is truncated at the first
    such gap before flags are pulled out of it, dropping the description
    rather than mistaking it for part of the invocation.
    """
    lines = markdown.splitlines()
    found: dict[str, list[str]] = {}
    i = 0
    while i < len(lines):
        if re.match(r"^\s*soup train\b", lines[i]):
            block = []
            j = i
            while True:
                raw = lines[j].rstrip()
                continued = raw.endswith("\\")
                if continued:
                    raw = raw[:-1]
                command_part = re.split(r"  +", raw.strip(), maxsplit=1)[0]
                block.append(command_part)
                if not continued:
                    break
                j += 1
            full = " ".join(block)
            for flag in re.findall(r"--[a-zA-Z][a-zA-Z0-9-]*", full):
                found.setdefault(flag, []).append(full.strip())
            i = j
        i += 1
    return found


def test_documented_train_flags_exist_in_the_live_typer_tree():
    live = _train_command_options()
    for doc_name in ("commands.md", "training.md"):
        documented = _documented_train_flags((DOCS / doc_name).read_text(encoding="utf-8"))
        for flag, examples in documented.items():
            if flag in live or flag in _PRE_EXISTING_TRAIN_FLAG_MISMATCHES:
                continue
            raise AssertionError(
                f"{doc_name} documents `soup train {flag}` but the live Typer "
                f"tree has no such option. Example: {examples[0][:200]}"
            )


def test_guard_actually_fails_on_a_reintroduced_minillm_flag():
    """#979's exact regression, reproduced in-memory rather than by editing a
    real doc file: the guard must reject this, not just observe that today's
    docs are already clean."""
    reintroduced = "soup train --config soup.yaml --minillm-enabled\n"
    live = _train_command_options()
    documented = _documented_train_flags(reintroduced)
    offending = [
        flag
        for flag in documented
        if flag not in live and flag not in _PRE_EXISTING_TRAIN_FLAG_MISMATCHES
    ]
    assert offending == ["--minillm-enabled"]


def test_a_real_flag_does_not_trip_the_guard():
    """Control: a doc line naming only flags that really exist must not raise."""
    real = "soup train --config soup.yaml --minillm-on-policy --tensorboard\n"
    live = _train_command_options()
    documented = _documented_train_flags(real)
    offending = [
        flag
        for flag in documented
        if flag not in live and flag not in _PRE_EXISTING_TRAIN_FLAG_MISMATCHES
    ]
    assert offending == []


def test_allowlist_entries_are_still_actually_missing():
    """A grandfathered flag that gets wired without removing it from the
    allowlist would silently stop being checked -- catch that drift too."""
    live = _train_command_options()
    still_missing = _PRE_EXISTING_TRAIN_FLAG_MISMATCHES & live
    assert not still_missing, (
        f"{still_missing} now exist as real options -- remove from "
        "_PRE_EXISTING_TRAIN_FLAG_MISMATCHES so the guard covers them again"
    )
