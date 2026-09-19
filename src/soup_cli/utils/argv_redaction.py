"""Mask the values of credential-bearing CLI options before argv is persisted.

The audit log (``utils/audit_log.py``) records every command's arguments. Its
value-pattern layer only recognises a few token shapes (``hf_*``, ``sk-*``,
``Bearer ...``), so a credential passed as ``--tool-auth-token Zq3v...`` or
``--api-key gsk_...`` would be written as typed. This module masks by OPTION
NAME instead: any value-taking option whose long name contains ``token``,
``key``, ``secret``, ``password``, ``passwd`` or ``credential`` has its value
replaced with ``<redacted>``, unless the name is listed in
``NOT_SECRET_OPTIONS``.

Stdlib + click only: importing this module must stay cheap for the CLI.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

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


def _long_names(option: click.Option) -> list[str]:
    return [o for o in (*option.opts, *option.secondary_opts) if o.startswith("--")]


def _is_secret_option(option: click.Option) -> bool:
    if option.is_flag or option.count:
        return False
    longs = _long_names(option)
    if not any(SECRET_NAME_RE.search(name[2:]) for name in longs):
        return False
    return not any(name in NOT_SECRET_OPTIONS for name in longs)


def secret_option_strings(command: click.Command) -> frozenset[str]:
    """Every option string (long and short) of ``command``'s credential options."""
    found: set[str] = set()
    for param in getattr(command, "params", None) or ():
        if isinstance(param, click.Option) and _is_secret_option(param):
            found.update(param.opts)
    return frozenset(found)


def iter_commands(command: click.Command) -> Iterable[click.Command]:
    """Yield ``command`` and every command below it."""
    yield command
    if isinstance(command, click.Group):
        for sub in command.commands.values():
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
    while isinstance(current, click.Group):
        index = next(
            (i for i, tok in enumerate(tokens) if not str(tok).startswith("-")),
            None,
        )
        if index is None:
            break
        sub = current.commands.get(str(tokens[index]))
        if sub is None:
            break
        current = sub
        tokens = tokens[index + 1:]
    return current


def mask_secret_args(args: Sequence[str], secret_opts: frozenset[str]) -> tuple[str, ...]:
    """Return ``args`` with the value of every option in ``secret_opts`` masked.

    Handles ``--opt value``, ``--opt=value``, ``-o value`` and ``-oVALUE``.
    An option with no following value is kept as is.
    """
    long_opts = {o for o in secret_opts if o.startswith("--")}
    short_opts = {o for o in secret_opts if not o.startswith("--") and len(o) == 2}
    out: list[str] = []
    mask_next = False
    for raw in args:
        tok = str(raw)
        if mask_next:
            out.append(REDACTED)
            mask_next = False
            continue
        if tok in secret_opts:
            out.append(tok)
            mask_next = True
            continue
        if tok.startswith("--") and "=" in tok:
            name = tok.split("=", 1)[0]
            if name in long_opts:
                out.append(f"{name}={REDACTED}")
                continue
        elif not tok.startswith("--") and len(tok) > 2 and tok[:2] in short_opts:
            out.append(f"{tok[:2]}{REDACTED}")
            continue
        out.append(tok)
    return tuple(out)
