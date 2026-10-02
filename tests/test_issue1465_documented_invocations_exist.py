"""#1465: every ``soup ...`` invocation written in the docs resolves in the live Click tree.

#979's guard checked ``soup train`` flags in two files. Eight documented invocations in
other files named a flag or a command path the CLI refuses (``--spec-tokens``,
``--max-iter``, ``soup eval --model``, a global option placed after the command), and
one recipe description named ``--online-dpo-judge``, an option no command has. This
generalises the walk to every invocation line in ``docs/*.md`` and ``README.md`` and to
every recipe description.

What counts as an invocation: a line that starts with ``soup`` (an optional ``$ ``
prompt allowed), continued across backslash line breaks, with the two-space description
tail that ``docs/commands.md`` appends dropped. Quoted strings are blanked first, so a
flag inside ``--eval-command "soup eval ... --model x"`` belongs to the quoted command,
not to the one being checked. Lines whose command path carries documentation notation
(``<cmd>``, ``infer|chat|diff``, ``...``, ``[...]``) are syntax, not a command, and are
skipped; flags after a resolved command are always checked.

Resolution follows Click: global options (``--verbose``, ``--log-level debug``) are
accepted only before the command and consume their value; a group that runs on its own
(``soup ship``, ``soup runs``) is checked against its own options; a group that routes a
bare argument to a default subcommand (``soup bench ./m``, ``soup advise d.jsonl``) is
accepted when one of its subcommands takes every flag used; a pass-through command
(``soup llama gguf-split``, ``ignore_unknown_options``) has no flag check; a bare group
(``soup eval``) prints help and is fine. A boolean flag's ``--no-*`` spelling counts.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest
import typer

from soup_cli.cli import app
from soup_cli.recipes.catalog import RECIPES

_ROOT = Path(__file__).resolve().parents[1]
_DOC_FILES = sorted((_ROOT / "docs").glob("*.md")) + [_ROOT / "README.md"]
_NOTATION = re.compile(r"[<>|\[\]]|\.\.\.")
_QUOTED = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'")
_FLAG = re.compile(r"^--[a-zA-Z][a-zA-Z0-9-]*")
# Groups that route a bare argument to a default subcommand. ``soup bench ./m`` does it
# in its Click class (``parse_args`` override, detected below); ``soup advise d.jsonl``
# does it in the entry point (``cli._rewrite_advise_argv``), which the Click tree cannot
# show, so it is named here and a test pins that the rewrite still exists.
_ARGV_REWRITTEN_GROUPS = {"advise"}


def _cli() -> click.Group:
    return typer.main.get_command(app)


def _options(cmd: click.Command) -> set[str]:
    opts: set[str] = {"--help"}
    for param in cmd.params:
        if isinstance(param, click.Option):
            opts.update(o for o in param.opts if o.startswith("--"))
            opts.update(o for o in param.secondary_opts if o.startswith("--"))
    return opts


def _all_options(cmd: click.Command) -> set[str]:
    """Every --option anywhere in the tree (for recipe descriptions)."""
    found = _options(cmd)
    if isinstance(cmd, click.Group):
        for sub in cmd.commands.values():
            found |= _all_options(sub)
    return found


def invocations(markdown: str) -> list[tuple[int, str]]:
    """(line number, full invocation) for every ``soup ...`` line, continuations joined."""
    lines = markdown.splitlines()
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        if re.match(r"^\s*(\$ )?soup\s", lines[i]):
            parts: list[str] = []
            j = i
            while True:
                raw = lines[j].rstrip()
                continued = raw.endswith("\\")
                if continued:
                    raw = raw[:-1]
                parts.append(re.split(r"  +", raw.strip(), maxsplit=1)[0])
                if not continued or j + 1 >= len(lines):
                    break
                j += 1
            out.append((i + 1, re.sub(r"^\$ ", "", " ".join(parts).strip())))
            i = j
        i += 1
    return out


def _routes_a_default_subcommand(group: click.Group) -> bool:
    overridden = type(group).parse_args is not click.Group.parse_args and (
        type(group).parse_args is not typer.core.TyperGroup.parse_args
    )
    return overridden or group.name in _ARGV_REWRITTEN_GROUPS


def check_invocation(invocation: str, root: click.Group) -> str | None:
    """``None`` when the invocation resolves and every flag exists, else the reason."""
    tokens = _QUOTED.sub('""', invocation).split()
    if not tokens or tokens[0] != "soup":
        return None
    root_params = {o: p for p in root.params if isinstance(p, click.Option) for o in p.opts}
    i = 1
    # global options, only before the command; a valued one consumes its value
    while i < len(tokens) and tokens[i].startswith("-"):
        flag = tokens[i].split("=", 1)[0]
        param = root_params.get(flag)
        if param is None:
            return f"unknown global option {flag}"
        i += 1
        if not param.is_flag and "=" not in tokens[i - 1]:
            i += 1
    cmd: click.Command = root
    while isinstance(cmd, click.Group) and i < len(tokens):
        name = tokens[i]
        if _NOTATION.search(name):
            return None  # documentation notation, not a runnable command path
        sub = cmd.commands.get(name)
        if sub is None:
            break
        cmd = sub
        i += 1
    rest = tokens[i:]
    flags = [m.group(0) for t in rest for m in [_FLAG.match(t.strip("[]"))] if m]
    if isinstance(cmd, click.Group):
        if cmd is root:
            return f"no command {rest[0]!r}" if rest else None
        if not rest:
            return None  # a bare group prints its help
        positional = [t for t in rest if not t.startswith("-")]
        if not positional and all(f in _options(cmd) for f in flags):
            return None  # `soup llama --help`: the group's own options, nothing routed
        if _options(cmd) - {"--help"}:
            candidates = [cmd]  # a group that runs on its own (invoke_without_command)
        elif _routes_a_default_subcommand(cmd):
            candidates = list(cmd.commands.values())
        else:
            return f"{cmd.name}: is a command group; {rest[0]!r} is not one of its subcommands"
        if any(all(f in _options(c) for f in flags) for c in candidates):
            return None
        return f"{cmd.name}: no subcommand takes {flags}"
    settings = cmd.context_settings or {}
    if settings.get("ignore_unknown_options") or settings.get("allow_extra_args"):
        return None
    unknown = [f for f in flags if f not in _options(cmd)]
    return f"{cmd.name}: unknown option(s) {unknown}" if unknown else None


def _problems(markdown: str, root: click.Group) -> list[tuple[int, str, str]]:
    return [
        (lineno, inv, reason)
        for lineno, inv in invocations(markdown)
        for reason in [check_invocation(inv, root)]
        if reason is not None
    ]


@pytest.mark.parametrize("path", _DOC_FILES, ids=lambda p: p.name)
def test_every_documented_invocation_resolves_in_the_live_cli(path: Path) -> None:
    problems = _problems(path.read_text(encoding="utf-8"), _cli())
    assert not problems, "\n".join(
        f"{path.name}:{lineno}: {reason}\n    {inv[:160]}" for lineno, inv, reason in problems
    )


def test_recipe_descriptions_name_only_real_options() -> None:
    live = _all_options(_cli())
    bad = {
        name: [f for f in re.findall(r"--[a-zA-Z][a-zA-Z0-9-]*", meta.description) if f not in live]
        for name, meta in RECIPES.items()
    }
    bad = {name: flags for name, flags in bad.items() if flags}
    assert not bad, bad


# --- the guard must be able to fail: each issue case, in memory -------------------------

@pytest.mark.parametrize(
    "line,fragment",
    [
        ("soup serve --model ./output --speculative-decoding d --spec-tokens 5", "--spec-tokens"),
        ("soup loop watch [--detach] [--max-iter N]", "--max-iter"),
        ("soup train --config soup.yaml --no-telemetry", "--no-telemetry"),
        ("soup --verbose eval --model ./output --benchmarks mmlu", "is a command group"),
        ("soup train --config soup.yaml --minillm-enabled", "--minillm-enabled"),  # #979's case
        ("soup adapters blame ./a --live", "--live"),
    ],
)
def test_the_guard_flags_each_documented_mistake(line: str, fragment: str) -> None:
    reason = check_invocation(line, _cli())
    assert reason is not None and fragment in reason, reason


@pytest.mark.parametrize(
    "line",
    [
        "soup --no-telemetry train --config soup.yaml",
        "soup --log-level debug runs show <id>",
        "soup recipes verify --no-templates",
        "soup merge --adapter ./output --save-format 4bit --no-double-quant",
        "soup ship --evidence ev.json --output v.json",
        'soup advise data.jsonl --goal "make it concise" --probe',
        "soup bench ./output --num-prompts 5",
        "soup llama gguf-split --merge a.gguf b.gguf out.gguf",
        'soup adapters bisect a b --eval-command "soup eval benchmark --model x --tasks y"',
        "soup eval",
        "soup infer|chat|diff ... --device cpu",
        "soup --verbose <command>",
    ],
)
def test_real_or_notational_lines_do_not_trip_the_guard(line: str) -> None:
    assert check_invocation(line, _cli()) is None


def test_the_advise_exception_is_backed_by_the_entry_point_rewrite() -> None:
    from soup_cli import cli

    assert callable(cli._rewrite_advise_argv)
    assert _routes_a_default_subcommand(_cli().commands["advise"])
    assert _routes_a_default_subcommand(_cli().commands["bench"])
    assert not _routes_a_default_subcommand(_cli().commands["eval"])


def test_the_recipe_check_flags_the_issue_option() -> None:
    live = _all_options(_cli())
    assert "--online-dpo-judge" not in live
    assert "--trust-remote-code" in live
