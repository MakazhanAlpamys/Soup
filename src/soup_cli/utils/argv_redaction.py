"""Mask the values of credential-bearing CLI options before argv is persisted.

The audit log (``utils/audit_log.py``) records every command's arguments. Its
value-pattern layer only recognises a few token shapes (``hf_*``, ``sk-*``,
``Bearer ...``), so a credential passed as ``--tool-auth-token Zq3v...`` or
``--api-key gsk_...`` would be written as typed. This module masks by OPTION
NAME instead: any value-taking option whose long name contains ``token``,
``key``, ``secret``, ``password``, ``passwd`` or ``credential`` has its value
replaced with ``<redacted>``, unless the name is listed in
``NOT_SECRET_OPTIONS``, and any option listed in ``SECRET_OPTIONS``.

The command tree is read by attribute (``commands``, ``params``, ``opts``),
never by class: Typer builds it from whichever click it was installed with,
and that is not always the ``click`` package this module could import. The
tree supplies the options a command declares, short aliases included;
:func:`mask_secret_args` also applies the same name rule to every long option
typed on the command line, so an option the tree did not yield is masked by
its name all the same.

Stdlib only: importing this module must stay cheap for the CLI.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    import click

REDACTED = "<redacted>"

# Applied to the long option name without its leading dashes.
SECRET_NAME_RE = re.compile(r"token|key|secret|password|passwd|credential", re.IGNORECASE)

# Long option names that match SECRET_NAME_RE but never carry a credential.
# When unsure, leave a name OUT of this set: masking a harmless value costs
# nothing, keeping a credential in the audit file does.
NOT_SECRET_OPTIONS: frozenset[str] = frozenset(
    {
        "--max-tokens",  # generation length (an integer)
        "--max-new-tokens",  # generation length (an integer)
        "--min-tokens",  # a length threshold (an integer)
        "--num-assistant-tokens",  # speculative-decoding draft length (an integer)
        "--num-speculative-tokens",  # speculative-decoding draft length (an integer)
        "--tokenizer",  # tokenizer repo id or local path
        "--special-token",  # a vocabulary token string, e.g. "<|im_end|>"
        "--echo-trap-tokenizer-aware",  # a mode switch, not a credential
        "--show-token",  # a display switch, not a credential
        "--public-key",  # path to a PUBLIC verification key
        "--generate-key",  # output path for a newly generated key pair
    }
)


# Long option names that carry a credential although their name does not say
# so: a webhook URL is all that is needed to post to the channel.
SECRET_OPTIONS: frozenset[str] = frozenset({"--slack-url", "--discord-url"})


def is_secret_name(name: str) -> bool:
    """Whether the long option ``name`` (``--like-this``) holds a credential."""
    if name in NOT_SECRET_OPTIONS:
        return False
    return name in SECRET_OPTIONS or bool(SECRET_NAME_RE.search(name[2:]))


def _long_names(param: object) -> list[str]:
    """The ``--name`` strings of a parameter; an argument has none."""
    declared = (
        *(getattr(param, "opts", None) or ()),
        *(getattr(param, "secondary_opts", None) or ()),
    )
    return [o for o in declared if o.startswith("--")]


def _is_secret_option(option: object) -> bool:
    if getattr(option, "is_flag", False) or getattr(option, "count", False):
        return False
    longs = _long_names(option)
    if any(name in NOT_SECRET_OPTIONS for name in longs):
        return False
    return any(is_secret_name(name) for name in longs)


def secret_option_strings(command: click.Command) -> frozenset[str]:
    """Every option string (long and short) of ``command``'s credential options."""
    found: set[str] = set()
    for param in getattr(command, "params", None) or ():
        if _is_secret_option(param):
            found.update(param.opts)
    return frozenset(found)


def subcommands(command: object) -> dict | None:
    """The name -> command mapping of a group, or ``None`` for a leaf command."""
    found = getattr(command, "commands", None)
    return found if isinstance(found, dict) else None


def iter_commands(command: click.Command) -> Iterable[click.Command]:
    """Yield ``command`` and every command below it."""
    yield command
    for sub in (subcommands(command) or {}).values():
        yield from iter_commands(sub)


def all_secret_option_strings(root: click.Command) -> frozenset[str]:
    """Union of :func:`secret_option_strings` over the whole command tree."""
    found: set[str] = set()
    for command in iter_commands(root):
        found.update(secret_option_strings(command))
    return frozenset(found)


def resolve_command(root: click.Command, args: Sequence[str]) -> click.Command:
    """Walk ``args`` down the group tree and return the deepest command named.

    At each group, the first token that does not start with ``-`` is looked
    up as a subcommand; when it is not one, the walk stops at that group.
    """
    current = root
    tokens = list(args)
    while (below := subcommands(current)) is not None:
        index = next(
            (i for i, tok in enumerate(tokens) if not str(tok).startswith("-")),
            None,
        )
        if index is None:
            break
        sub = below.get(str(tokens[index]))
        if sub is None:
            break
        current = sub
        tokens = tokens[index + 1:]
    return current


def _mask_short_bundle(tok: str, short_opts: set[str]) -> tuple[str, bool] | None:
    """Mask a bundled short-option token, or return ``None`` if it holds no secret.

    Click allows short options to be bundled: ``-xtVALUE`` is ``-x`` (a flag)
    followed by ``-t VALUE``, and ``-xt VALUE`` puts the value in the next
    token. Matching only ``tok[:2]`` therefore missed every secret typed behind
    another short flag.

    Returns ``(masked_token, mask_next_token)``. Everything up to and including
    the secret option letter is kept — those are flag names, not credentials,
    and they are what makes the audit line readable.
    """
    for index, char in enumerate(tok[1:], start=1):
        if f"-{char}" not in short_opts:
            continue
        if index == len(tok) - 1:
            # The secret option ends the bundle: its value is the next token.
            return tok, True
        return f"{tok[:index + 1]}{REDACTED}", False
    return None


def all_long_option_strings(root: click.Command) -> frozenset[str]:
    """Every ``--name`` any command of the tree declares, switches included."""
    found: set[str] = set()
    for command in iter_commands(root):
        for param in getattr(command, "params", None) or ():
            found.update(_long_names(param))
    return frozenset(found)


def _starts_an_option(tok: str, known: frozenset[str] | set[str]) -> bool:
    """Whether ``tok``, found where a credential's value belongs, is an option instead.

    Only a long option the command tree declares, or one whose own name is a
    credential name, is read as an option there. Anything else is the value,
    even when it starts with ``--``.
    """
    if not tok.startswith("--"):
        return False
    name = tok.partition("=")[0]
    return name in known or is_secret_name(name)


def mask_secret_args(
    args: Sequence[str],
    secret_opts: frozenset[str],
    known_opts: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Return ``args`` with the value of every credential option masked.

    An option is a credential option when it is in ``secret_opts`` (what the
    command tree declares, short aliases included) or when it is a long
    option whose name :func:`is_secret_name` accepts. The second rule needs
    no tree, so it also covers an option the tree did not yield; a switch
    with such a name hides the token after it, which costs nothing.

    Handles ``--opt value``, ``--opt=value``, ``-o value``, ``-oVALUE`` and
    bundled short options (``-xoVALUE`` / ``-xo VALUE``, where ``-x`` is a flag
    and ``-o`` takes the value). An option with no following value is kept as
    is. The token after a credential option is its value and is masked, unless
    it is a long option in ``known_opts`` (every long option of the command
    tree) or ``secret_opts``, or has a credential name itself: then the option
    before it had no value and the scan continues from the new option, so a
    switch in front of a credential option cannot shift the mask off its value.
    """
    long_opts = {o for o in secret_opts if o.startswith("--")}
    known_longs = long_opts | known_opts
    short_opts = {o for o in secret_opts if not o.startswith("--") and len(o) == 2}
    out: list[str] = []
    mask_next = False
    for raw in args:
        tok = str(raw)
        if mask_next:
            mask_next = False
            if not _starts_an_option(tok, known_longs):
                out.append(REDACTED)
                continue
            # An option, not a value: the option before it had none, and this
            # one is scanned like any other.
        if tok in secret_opts:
            out.append(tok)
            mask_next = True
            continue
        if tok.startswith("--"):
            name, has_value, _ = tok.partition("=")
            if name in long_opts or is_secret_name(name):
                out.append(f"{name}={REDACTED}" if has_value else tok)
                mask_next = not has_value
                continue
        elif tok.startswith("-") and not tok.startswith("--") and len(tok) > 1:
            masked = _mask_short_bundle(tok, short_opts)
            if masked is not None:
                out.append(masked[0])
                mask_next = masked[1]
                continue
        out.append(tok)
    return tuple(out)
