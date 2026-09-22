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
flags, echo-trap flags, ``--output``). #997 rewrote those sites to use the
real YAML fields, so there are no grandfathered exceptions left: every flag
found by this walker must exist in the live command tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer

from soup_cli.cli import app

DOCS = Path(__file__).parents[1] / "docs"

_ISSUE_997_NONEXISTENT_FLAGS = {
    "--uld-strategy",
    "--rl-checkpoint-save-every-steps",
    "--rl-checkpoint-keep-last",
    "--rl-checkpoint-include-optimizer",
    "--echo-trap-enabled",
    "--echo-trap-threshold",
    "--echo-trap-halt",
    "--output",
}

_ISSUE_997_SCHEMA_REPLACEMENTS = {
    "uld_strategy": ("commands.md", "training.md"),
    "rl_checkpoint_save_every_steps": ("commands.md", "training.md"),
    "rl_checkpoint_keep_last": ("commands.md", "training.md"),
    "rl_checkpoint_include_optimizer": ("commands.md", "training.md"),
    "echo_trap_enabled": ("commands.md", "training.md"),
    "echo_trap_threshold": ("commands.md", "training.md"),
    "echo_trap_halt": ("commands.md", "training.md"),
    "output": ("training.md",),
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
            if flag in live:
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
        if flag not in live
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
        if flag not in live
    ]
    assert offending == []


@pytest.mark.parametrize("flag", sorted(_ISSUE_997_NONEXISTENT_FLAGS))
def test_issue997_nonexistent_flags_would_trip_the_guard(flag):
    """Each old spelling is absent from ``train`` and rejected by the walker.

    This is the mutation control for deleting one of #997's doc edits: the
    main guard sees the same spelling and fails rather than grandfathering it.
    """
    live = _train_command_options()
    documented = _documented_train_flags(f"soup train --config soup.yaml {flag}\n")

    assert flag not in live
    assert [option for option in documented if option not in live] == [flag]


@pytest.mark.parametrize(
    "field,doc_names",
    sorted(_ISSUE_997_SCHEMA_REPLACEMENTS.items()),
)
def test_issue997_replacements_are_real_schema_fields_and_stay_documented(field, doc_names):
    """The fix must replace each fake flag with a real config field, not just delete it."""
    from soup_cli.config.schema import SoupConfig, TrainingConfig

    model = SoupConfig if field == "output" else TrainingConfig
    assert field in model.model_fields
    for doc_name in doc_names:
        assert field in (DOCS / doc_name).read_text(encoding="utf-8")
