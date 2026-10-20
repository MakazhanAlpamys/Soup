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
_NOTATION = re.compile(r"[<>|\[\]{}\u2026]|\.\.\.")
#: `VAR=value soup ...` lines are invocations too (docs/backends-and-ops.md:715).
_LINE = re.compile(r"^\s*(\$ )?(?:[A-Z_][A-Z0-9_]*=\S+ )*soup\s")
_PREFIX = re.compile(r"^(\$ )?(?:[A-Z_][A-Z0-9_]*=\S+ )*")
#: a backtick span that is a `soup ...` invocation, unless the prose says "not `soup ...`"
_INLINE = re.compile(r"(?<!not )`(soup [^`\n]+)`")
_QUOTED = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'")
#: the whole option token, so `--config_file` is checked as itself, not as `--config`
_FLAG = re.compile(r"^--[A-Za-z0-9_-]+")
#: an aligned column that is a bare word (a subcommand or a positional), then a flag
#: or a path; the lowercase start keeps description tails such as `Implies
#: --allow-mutating` out (review of #1578)
_BARE_THEN_ARG = re.compile(r"[a-z]\S* (?:--?[A-Za-z]|\.{0,2}/)")
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
        if _LINE.match(lines[i]):
            parts: list[str] = []
            j = i
            while True:
                raw = lines[j].rstrip()
                continued = raw.endswith("\\")
                if continued:
                    raw = raw[:-1]
                # two or more spaces end the invocation (a description or comment
                # follows) unless the next column is itself a flag: aligned columns such
                # as `soup chat  --model ./output` keep their flags (review of #1578)
                segments = re.split(r"  +", raw.strip())
                kept = [segments[0]]
                for segment in segments[1:]:
                    if not segment.startswith(("-", "[-")) and not _BARE_THEN_ARG.match(
                        segment
                    ):
                        break
                    kept.append(segment)
                parts.append(" ".join(kept))
                if not continued or j + 1 >= len(lines):
                    break
                j += 1
            out.append((i + 1, _PREFIX.sub("", " ".join(parts).strip())))
            i = j
        i += 1
    return out


def inline_invocations(markdown: str) -> list[tuple[int, str]]:
    """(line number, invocation) for every backtick span that starts with ``soup ``:
    headings, contents entries and prose name flags too (the issue's own ``--live``
    site was a heading plus its contents entry)."""
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(markdown.splitlines(), start=1):
        for match in _INLINE.finditer(line):
            out.append((lineno, match.group(1).strip()))
    return out


def _routes_a_default_subcommand(group: click.Group) -> bool:
    overridden = type(group).parse_args is not click.Group.parse_args and (
        type(group).parse_args is not typer.core.TyperGroup.parse_args
    )
    return overridden or group.name in _ARGV_REWRITTEN_GROUPS


def check_invocation(invocation: str, root: click.Group) -> str | None:
    """``None`` when the invocation resolves and every flag exists, else the reason."""
    tokens = _QUOTED.sub('""', _PREFIX.sub("", invocation.strip())).split()
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
            if cmd is not root and _routes_a_default_subcommand(cmd):
                i += 1  # `soup bench <model> --backend auto`: the placeholder is the routed
                break  # argument, and the flags after it are still checked
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
            # a group that runs on its own (invoke_without_command). None of them
            # declares a positional argument, so the first token that is neither an
            # option nor an option's value must be a subcommand: `soup runs list`
            # exits 2 ("No such command 'list'") while `soup runs` lists the runs.
            first = _first_bare_token(rest, cmd)
            if first is not None:
                sub = cmd.commands.get(first)
                if sub is None:
                    return f"{cmd.name}: no subcommand {first!r}"
                split = rest.index(first)
                before = [
                    m.group(0) for t in rest[:split] for m in [_FLAG.match(t.strip("[]"))] if m
                ]
                unknown = [f for f in before if f not in _options(cmd)]
                if unknown:
                    return f"{cmd.name}: unknown option(s) {unknown}"
                after = rest[split + 1 :]
                sub_flags = [m.group(0) for t in after for m in [_FLAG.match(t.strip("[]"))] if m]
                unknown = [f for f in sub_flags if f not in _options(sub)]
                return f"{sub.name}: unknown option(s) {unknown}" if unknown else None
            candidates = [cmd]
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


def _first_bare_token(rest: list[str], cmd: click.Command) -> str | None:
    """The first token of ``rest`` that is neither one of ``cmd``'s options nor the
    value a valued option consumes (bracketed ``[--output v.json]`` included)."""
    params = {o: p for p in cmd.params if isinstance(p, click.Option) for o in p.opts}
    i = 0
    while i < len(rest):
        token = rest[i].strip("[]")
        if token.startswith("-"):
            param = params.get(token.split("=", 1)[0])
            i += 1
            if param is not None and not param.is_flag and "=" not in token:
                i += 1  # its value
            continue
        return rest[i]
    return None


def _problems(markdown: str, root: click.Group) -> list[tuple[int, str, str]]:
    return [
        (lineno, inv, reason)
        for lineno, inv in invocations(markdown) + inline_invocations(markdown)
        for reason in [check_invocation(inv, root)]
        if reason is not None
    ]


@pytest.mark.parametrize("path", _DOC_FILES, ids=lambda p: p.name)
def test_every_documented_invocation_resolves_in_the_live_cli(path: Path) -> None:
    problems = _problems(path.read_text(encoding="utf-8"), _cli())
    assert not problems, "\n".join(
        f"{path.name}:{lineno}: {reason}\n    {inv[:160]}" for lineno, inv, reason in problems
    )


def test_the_file_check_reads_both_lines_and_inline_spans() -> None:
    """The per-file test only sees what ``_problems`` hands it."""
    text = "soup chat --modell ./output\nSee `soup serve --kv-cache-typo` below.\n"
    reasons = [reason for _, _, reason in _problems(text, _cli())]
    assert len(reasons) == 2, reasons
    assert "--modell" in reasons[0] and "--kv-cache-typo" in reasons[1], reasons


def test_the_guard_cannot_pass_vacuously() -> None:
    """Dropping README.md, a wrong docs path or an empty file list all passed before
    (review of #1578): pin the inputs and a floor on what was seen."""
    assert _ROOT / "README.md" in _DOC_FILES
    assert len(_DOC_FILES) >= 10, [p.name for p in _DOC_FILES]
    texts = [p.read_text(encoding="utf-8") for p in _DOC_FILES]
    lines = sum(len(invocations(t)) for t in texts)
    spans = sum(len(inline_invocations(t)) for t in texts)
    assert lines >= 900, lines  # 944 on 2026-10-04
    assert spans >= 400, spans  # 423 on 2026-10-04


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
        ("soup --verbos train --config soup.yaml", "unknown global option"),
        ("soup trian --config soup.yaml", "no command"),
        ("soup runs shwo abc123", "no subcommand"),
        ("soup runs list", "no subcommand"),
        ("soup bench <model> --backendd auto", "--backendd"),
        ("SOUP_TELEMETRY=1 soup train --confg soup.yaml", "--confg"),
        ("soup train --config_file soup.yaml", "--config_file"),
        ("soup runs --bogus show abc123", "--bogus"),
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
        "soup adapters {merge,pr,bisect}",
        "soup data \u2026",
        "soup runs --limit 5 show abc123",
        "soup recipes search mlx",
        "soup bench <model> --backend auto",
        "SOUP_TELEMETRY=1 soup train --config soup.yaml",
    ],
)
def test_real_or_notational_lines_do_not_trip_the_guard(line: str) -> None:
    assert check_invocation(line, _cli()) is None


@pytest.mark.parametrize(
    "markdown,expected",
    [
        ("soup chat  --model ./output                    # talk to your model",
         ["soup chat --model ./output"]),
        ("soup data pii          --input training.jsonl --output pii_flagged.jsonl",
         ["soup data pii --input training.jsonl --output pii_flagged.jsonl"]),
        ("soup loop watch [--detach] [--max-iter N]   Watch the loop",
         ["soup loop watch [--detach] [--max-iter N]"]),
        ("SOUP_TELEMETRY=1 soup train --config soup.yaml", ["soup train --config soup.yaml"]),
        ("$ soup train --config a.yaml \\\n    --resume", ["soup train --config a.yaml --resume"]),
        ("soup eval   benchmark --model ./output               # evaluate",
         ["soup eval benchmark --model ./output"]),
        ("soup data   inspect ./data/train.jsonl               # dataset stats",
         ["soup data inspect ./data/train.jsonl"]),
        ("soup probe harm  meta-llama/Llama-3-8B --evidence activations.json",
         ["soup probe harm meta-llama/Llama-3-8B --evidence activations.json"]),
        ("soup mcp serve --allow-execute   Implies --allow-mutating; enables x",
         ["soup mcp serve --allow-execute"]),
    ],
)
def test_aligned_columns_prefixes_and_continuations_keep_their_flags(markdown, expected) -> None:
    assert [inv for _, inv in invocations(markdown)] == expected


def test_inline_spans_are_collected_except_after_not() -> None:
    text = (
        "## KV cache (`soup serve --kv-cache-type`)\n"
        "Use the export command (not `soup deploy airgap-bundle`); see `soup chat`.\n"
    )
    assert [inv for _, inv in inline_invocations(text)] == [
        "soup serve --kv-cache-type", "soup chat"
    ]


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
