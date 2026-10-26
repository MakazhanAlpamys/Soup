"""Render the two generated reference pages from the code.

``docs/config-reference.md`` lists every key ``soup.yaml`` accepts, walked out of
``SoupConfig.model_fields``. ``docs/cli-reference.md`` lists every command, option and
argument, walked out of the Click tree behind the Typer app. Neither page is written
by hand: change the schema field or the option, then run this script.

    python scripts/gen_reference_docs.py                    # rewrite the pages that changed
    python scripts/gen_reference_docs.py --check-structure  # what the test suite runs
    python scripts/gen_reference_docs.py --check            # byte-exact, for a release

Two checks, because they answer different questions. ``--check-structure`` parses the
committed tables back into rows and compares what a page *states*: per config key its
type, default and constraints; per command its parameters, each with its names, type,
default and whether it is required. A missing, extra or changed one fails and is named.
Descriptions, help text and command summaries are not compared, so rewording a help
string does not turn a pull request red; the wording is refreshed when a release is cut.
``--check`` compares the text itself and is what a release runs
(``SOUP_DOCS_EXACT=1`` makes the test suite run it too).

For the same reason no line on a page is a total or a list over several items: every
line belongs to one key, one command or one parameter, so two changes that each add a
different option merge as text and the merged page is still right.

``tests/test_gen_reference_docs.py`` runs the check on every interpreter and OS in the
CI matrix, so the same text has to come out everywhere. What keeps it that way:

* Types are spelled here, from ``typing`` introspection, never from ``repr()``:
  ``Optional[int]``, ``Union[int, None]`` and ``int | None`` all read ``int | null``.
* Nothing is taken from ``--help`` output, which carries ANSI colour and wraps at the
  terminal width. Commands and parameters are read off the Click objects; Rich markup
  in help strings is removed by a regex of its own.
* Field and parameter order is the declaration order; command names are sorted.
* A default that only this machine could have produced (a path under the home or the
  working directory, a drive path, a callable, a non-container factory value) is
  printed as ``(set at runtime)``, never as its value. Environment variables are named,
  never read.
* The two completion options Typer adds change shape with shell detection, so their
  rows are fixed text.
* Pages are written as UTF-8 with ``\\n`` newlines; the check compares text, so a CRLF
  checkout (``core.autocrlf``) of an LF blob still passes.

A numeric default computed from the machine (``os.cpu_count()``) cannot be told from a
literal one. There is none today; the cross-OS check is what would catch the first.

Rendering imports the schema and the CLI, never torch: both stay on the light core.
"""

from __future__ import annotations

import argparse
import collections.abc
import enum
import json
import os
import re
import sys
import tempfile
import types
import typing
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Callable, Sequence

import click
from pydantic import BaseModel
from pydantic.fields import FieldInfo

_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
#: Where the pages are read and written. Tests set this to a scratch tree.
_REPO_ROOT = _SCRIPT_REPO_ROOT

CONFIG_PAGE = "docs/config-reference.md"
CLI_PAGE = "docs/cli-reference.md"
RUNTIME_DEFAULT = "(set at runtime)"

_REGENERATE = "python scripts/gen_reference_docs.py"
#: How many differences ``--check-structure`` prints per page before "... and N more".
_MAX_SHOWN = 10
_NONE_TYPE = type(None)
_NOT_LITERAL = object()


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
#: ``&`` first, so the entities written for ``<`` are not escaped again; the backslash
#: before every escape that is itself written with one.
_PROSE_ESCAPES = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    ("\\", "\\\\"),
    ("`", "\\`"),
    ("*", "\\*"),
    ("[", "\\["),
    ("|", "\\|"),
    ("~", "\\~"),
    ("$", "\\$"),
)
#: An underscore inside a word (``snake_case``) never opens emphasis; any other one can.
_LOOSE_UNDERSCORE = re.compile(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])")
#: What would turn a paragraph of its own into a heading, a quote or a list item.
_BLOCK_START = re.compile(r"^(?:#{1,6}(?=\s|$)|>|[+-](?=\s|$))")
_LIST_START = re.compile(r"^(\d+)([.)])(?=\s|$)")
_INTERNAL_PHASE = re.compile(r"\b(v\d+(?:\.\d+){1,2})\s+part\s+[a-z]\b", re.IGNORECASE)
#: Rich's own tag grammar (``rich.markup.RE_TAGS``): a ``[`` then a lowercase letter,
#: ``#``, ``/`` or ``@``, up to the next ``]``, with any backslashes in front of it.
_RICH_TAG = re.compile(r"(\\*)\[([a-z#/@][^\[\]]*)\]")
_RICH_COLOUR = re.compile(
    r"#[0-9a-f]{6}|color\(\d{1,3}\)|rgb\(\d{1,3},\d{1,3},\d{1,3}\)|gr[ae]y\d{0,3}"
    r"|(?:bright_)?(?:black|red|green|yellow|blue|magenta|cyan|white)"
)
_RICH_WORDS = frozenset(
    "bold b dim italic i underline u strike s reverse blink conceal link not on default".split()
)


def _escape_prose(text: str) -> str:
    for raw, safe in _PROSE_ESCAPES:
        text = text.replace(raw, safe)
    return _LOOSE_UNDERSCORE.sub(r"\\_", text)


def escape_cell(text: str) -> str:
    """``text`` as one table cell: a single line, no cell break, no stray markup.

    Code spans are kept as code (only their ``|`` is escaped, which a GitHub table
    undoes inside code); everything else has the characters that Markdown or HTML
    would interpret escaped, so ``<id>`` stays visible and ``*`` stays a star.
    """
    text = " ".join(text.split())
    parts: list[str] = []
    position = 0
    for match in _CODE_SPAN.finditer(text):
        parts.append(_escape_prose(text[position : match.start()]))
        parts.append(match.group(0).replace("|", "\\|"))
        position = match.end()
    parts.append(_escape_prose(text[position:]))
    return "".join(parts)


def _escape_paragraph(text: str) -> str:
    """``escape_cell`` for a paragraph of its own, which must not open a block."""
    escaped = _LIST_START.sub(r"\1\\\2", escape_cell(text))
    return f"\\{escaped}" if _BLOCK_START.match(escaped) else escaped


def _code(text: str) -> str:
    """``text`` as a code span that survives a table cell."""
    text = text.replace("\r", " ").replace("\n", " ")
    fence = "`"
    while fence in text:
        fence += "`"
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text.replace('|', chr(92) + '|')}{pad}{fence}"


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table_head(*titles: str) -> list[str]:
    return [_row(titles), "|" + "---|" * len(titles)]


def strip_internal_phase(text: str) -> str:
    """Drop the planning label from ``v0.42.0 Part B``; public docs keep the version."""
    return _INTERNAL_PHASE.sub(r"\1", text)


def _is_rich_style(tag: str) -> bool:
    body = tag[1:] if tag.startswith("/") else tag
    if not body:
        return True  # ``[/]`` closes whatever is open
    if body.startswith(("link=", "@")):
        return True
    return all(word in _RICH_WORDS or _RICH_COLOUR.fullmatch(word) for word in body.split())


def strip_rich_markup(text: str) -> str:
    """Remove Rich style tags from a help string and undo its ``\\[`` escapes.

    Only tags that are styles are removed. Rich itself also swallows ``[sigstore]`` in
    ``pip install soup-cli[sigstore]`` when the author forgot the backslash; the
    bracketed text is what was meant, so it is kept here.
    """

    def replace(match: re.Match[str]) -> str:
        slashes, tag = match.group(1), match.group(2)
        kept = "\\" * (len(slashes) // 2)
        if len(slashes) % 2 or not _is_rich_style(tag):
            return f"{kept}[{tag}]"
        return kept

    return _RICH_TAG.sub(replace, text)


def summary_line(help_text: str | None) -> str:
    """The first paragraph of a command's help, on one line."""
    text = (help_text or "").split("\f", 1)[0].strip()
    first = re.split(r"\n[ \t]*\n", text, maxsplit=1)[0]
    return " ".join(first.split())


def github_slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lowercase, spaces dashed, and only letters,
    marks, decimal digits, ``_`` and ``-`` kept (so the subscript in ``CO₂`` is dropped)."""
    text = heading.replace("`", "").strip().lower()
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if char == " ":
            kept.append("-")
        elif char == "-" or category[0] in "LM" or category in ("Nd", "Nl", "Pc"):
            kept.append(char)
    return "".join(kept)


class _Slugger:
    """Anchors in document order; GitHub numbers a repeated heading ``-1``, ``-2``."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def take(self, heading: str) -> str:
        base = github_slug(heading)
        count = self._seen.get(base, 0)
        self._seen[base] = count + 1
        return base if not count else f"{base}-{count}"


# --------------------------------------------------------------------------
# Values that must read the same on every machine
# --------------------------------------------------------------------------


def _machine_markers() -> tuple[str, ...]:
    """This machine's home, working and temp directories, normalised for matching.

    Only those at least two directories deep. A checkout at ``/repo`` or ``/data`` (a
    container's working directory) is also an ordinary piece of a relative default such
    as ``./data/train.jsonl``, and must not turn that default into a placeholder there.
    """
    found = set()
    for raw in (os.path.expanduser("~"), os.getcwd(), tempfile.gettempdir()):
        marker = raw.replace("\\", "/").rstrip("/").casefold()
        if marker.count("/") >= 2:  # "/home/u" and "c:/users/u"; not "/repo", "/tmp", "c:"
            found.add(marker)
    return tuple(sorted(found))


def _is_machine_specific(text: str) -> bool:
    """True for a string only this machine could have produced: a Windows drive path, or
    one of this machine's directories as a whole path (never as the tail of another)."""
    normalised = text.replace("\\", "/").casefold()
    if re.match(r"^[a-z]:/", normalised):
        return True
    return any(
        re.search(r"(?<![\w./~-])" + re.escape(marker) + r"(?=/|$)", normalised)
        for marker in _machine_markers()
    )


def _plain_literal(value: Any) -> Any:
    """``value`` as JSON-ready data, or ``_NOT_LITERAL`` when it is not a stable literal."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, enum.Enum):
        return _plain_literal(value.value)
    if isinstance(value, PurePath):
        return _plain_literal(value.as_posix())
    if isinstance(value, str):
        return _NOT_LITERAL if _is_machine_specific(value) else value
    if isinstance(value, (set, frozenset)):
        try:
            value = sorted(value)
        except TypeError:
            return _NOT_LITERAL
    if isinstance(value, (list, tuple)):
        items = [_plain_literal(item) for item in value]
        return _NOT_LITERAL if any(item is _NOT_LITERAL for item in items) else items
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            plain = _plain_literal(item)
            if not isinstance(key, str) or plain is _NOT_LITERAL:
                return _NOT_LITERAL
            out[key] = plain
        return out
    return _NOT_LITERAL


def format_default(value: Any) -> str:
    """A default as a YAML flow value (``null``, ``true``, ``"auto"``, ``["a"]``).

    Anything that is not a plain literal, or that carries this machine's paths, is
    ``RUNTIME_DEFAULT``.
    """
    plain = _plain_literal(value)
    if plain is _NOT_LITERAL:
        return RUNTIME_DEFAULT
    return json.dumps(plain, ensure_ascii=False)


def _default_cell(text: str) -> str:
    return f"*{RUNTIME_DEFAULT}*" if text == RUNTIME_DEFAULT else _code(text)


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

_SCALARS = {str: "str", int: "int", float: "float", bool: "bool", bytes: "bytes"}


def _is_class(annotation: Any) -> bool:
    """A real class, never a parametrised generic. ``isinstance(type[int], type)`` is
    true on Python 3.10 and false from 3.11, so asking only that would spell the same
    annotation two ways."""
    return typing.get_origin(annotation) is None and isinstance(annotation, type)


def _container_name(base: Any) -> str | None:
    """``list`` / ``dict`` / ``set`` / ``tuple`` for a container class or generic origin
    (``typing.List``, ``collections.abc.Sequence`` and ``list`` all read ``list``)."""
    if not isinstance(base, type) or issubclass(base, (str, bytes)):
        return None
    if issubclass(base, tuple):
        return "tuple"
    if issubclass(base, collections.abc.Mapping):
        return "dict"
    if issubclass(base, collections.abc.Set):
        return "set"
    if issubclass(base, collections.abc.Sequence):
        return "list"
    return None


def _is_model(annotation: Any) -> bool:
    return _is_class(annotation) and issubclass(annotation, BaseModel)


def _union_members(annotation: Any) -> list[Any]:
    """The members of a union, flattened, ``Annotated`` unwrapped, ``None`` as its type."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _union_members(typing.get_args(annotation)[0])
    if origin is typing.Union or origin is types.UnionType:
        members: list[Any] = []
        for arg in typing.get_args(annotation):
            members.extend(_union_members(arg))
        return members
    return [_NONE_TYPE if annotation is None else annotation]


def _choice_values(annotation: Any) -> list[Any] | None:
    """The values of a ``Literal`` or an ``Enum`` class, else ``None``."""
    if typing.get_origin(annotation) is typing.Literal:
        return [
            item.value if isinstance(item, enum.Enum) else item
            for item in typing.get_args(annotation)
        ]
    if _is_class(annotation) and issubclass(annotation, enum.Enum):
        return [member.value for member in annotation]
    return None


def _unique(items: Sequence[Any]) -> list[Any]:
    """Order-preserving, and ``1`` is not ``True``."""
    seen = set()
    out = []
    for item in items:
        key = (type(item), item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def literal_choices(annotation: Any) -> list[Any]:
    """The allowed values when the annotation is nothing but a choice (or ``None``).

    A choice mixed with another type (``int | "auto"``) is spelled in the type instead,
    so this is empty for it.
    """
    values: list[Any] = []
    for member in _union_members(annotation):
        if member is _NONE_TYPE:
            continue
        found = _choice_values(member)
        if found is None:
            return []
        values.extend(found)
    return _unique(values)


def _value_type(value: Any) -> str:
    return "null" if value is None else _SCALARS.get(type(value), type(value).__name__)


def _format_member(annotation: Any) -> str:
    if annotation is typing.Any:
        return "any"
    origin = typing.get_origin(annotation)
    container = _container_name(annotation if origin is None else origin)
    if container is not None:
        args = [arg for arg in typing.get_args(annotation) if arg is not Ellipsis]
        inner = ", ".join(format_type(arg) for arg in args)
        return f"{container}[{inner}]" if inner else container
    if not _is_class(annotation):
        return "any"
    if annotation in _SCALARS:
        return _SCALARS[annotation]
    if issubclass(annotation, BaseModel):
        return "section"
    if issubclass(annotation, PurePath):
        return "path"
    return annotation.__name__


def format_type(annotation: Any) -> str:
    """One spelling per type, whatever the source wrote and whatever Python runs this.

    ``null`` goes last. A pure choice reads as the type of its values (the values are
    listed as a constraint); a choice mixed with other types is spelled inline.
    """
    members = _union_members(annotation)
    rest = [member for member in members if member is not _NONE_TYPE]
    pure_choice = bool(rest) and all(_choice_values(member) is not None for member in rest)
    names: list[str] = []
    for member in rest:
        values = _choice_values(member)
        if values is None:
            names.append(_format_member(member))
        elif pure_choice:
            names.extend(_value_type(value) for value in values)
        else:
            names.extend(json.dumps(value, ensure_ascii=False) for value in values)
    if len(rest) != len(members):
        names.append("null")
    return " | ".join(dict.fromkeys(names))


# --------------------------------------------------------------------------
# docs/config-reference.md
# --------------------------------------------------------------------------

_BOUNDS = (("gt", ">"), ("ge", ">="), ("lt", "<"), ("le", "<="))
_CONSTRAINT_ATTRS = (
    "gt",
    "ge",
    "lt",
    "le",
    "multiple_of",
    "min_length",
    "max_length",
    "pattern",
    "allow_inf_nan",
)


@dataclass(frozen=True)
class ConfigRow:
    """One ``soup.yaml`` key, every cell already Markdown."""

    key: str
    type: str
    default: str
    constraints: str
    description: str


def format_constraints(info: FieldInfo) -> str:
    """The single-value rules of a field, as one Markdown cell."""
    found: dict[str, Any] = {}
    for meta in info.metadata:
        for attr in _CONSTRAINT_ATTRS:
            value = getattr(meta, attr, None)
            if value is not None:
                found[attr] = value
    bounds = [_code(f"{sign} {found[attr]!r}") for attr, sign in _BOUNDS if attr in found]
    parts = [", ".join(bounds)] if bounds else []
    if "multiple_of" in found:
        parts.append(f"multiple of {found['multiple_of']!r}")
    if "min_length" in found:
        parts.append(f"min length {found['min_length']}")
    if "max_length" in found:
        parts.append(f"max length {found['max_length']}")
    if "pattern" in found:
        pattern = found["pattern"]
        parts.append("pattern " + _code(getattr(pattern, "pattern", pattern)))
    if found.get("allow_inf_nan") is False:
        parts.append("finite")
    choices = literal_choices(info.annotation)
    if choices:
        parts.append(
            "one of " + ", ".join(_code(json.dumps(item, ensure_ascii=False)) for item in choices)
        )
    return "; ".join(parts)


def _config_default(info: FieldInfo) -> str:
    factory = info.default_factory
    if factory is not None:
        if _is_model(factory):
            return "section defaults"
        try:
            value = factory()  # type: ignore[call-arg]
        except Exception:
            return f"*{RUNTIME_DEFAULT}*"
        # A factory exists to build a fresh container. A scalar out of one was computed.
        if not isinstance(value, (list, tuple, dict, set, frozenset)):
            return f"*{RUNTIME_DEFAULT}*"
        return _default_cell(format_default(value))
    if info.is_required():
        return "**required**"
    return _default_cell(format_default(info.default))


def _nested_models(annotation: Any) -> list[tuple[str, type[BaseModel]]]:
    """``(path suffix, model)`` for every model a field holds: ``""`` for the field
    itself, ``"[]"`` for the items of a sequence, ``".*"`` for the values of a mapping."""
    found: list[tuple[str, type[BaseModel]]] = []
    for member in _union_members(annotation):
        if _is_model(member):
            found.append(("", member))
            continue
        container = _container_name(typing.get_origin(member))
        if container is None:
            continue
        args = [arg for arg in typing.get_args(member) if arg is not Ellipsis]
        if container == "dict":
            args, suffix = args[1:], ".*"
        else:
            suffix = "[]"
        for arg in args:
            found.extend((suffix + deeper, model) for deeper, model in _nested_models(arg))
    return found


def _yaml_key(name: str, info: FieldInfo) -> str:
    for alias in (info.validation_alias, info.alias):
        if isinstance(alias, str):
            return alias
    return name


def staged_notes() -> dict[str, str]:
    """The sentence the row of each key in ``STAGED_FIELDS`` ends with, by YAML path.

    A staged field passes the schema and is refused by the loader
    (``config/staged_fields.py``, #808), so its description does not say so by
    itself. This page says it for every key that list names, read from the list,
    so a field staged later gets the sentence without an edit here."""
    from soup_cli.config.staged_fields import STAGED_FIELD_REJECTION_VERSION, STAGED_FIELDS

    note = (
        "Refused at load when set away from its default "
        f"(v{STAGED_FIELD_REJECTION_VERSION}, #808)."
    )
    return {f"{section}.{name}": note for section, name in STAGED_FIELDS}


def _with_sentence(description: str, sentence: str) -> str:
    """``sentence`` after ``description``, which is first closed with a full stop."""
    if description and not description.endswith((".", "!", "?", ":")):
        description += "."
    return f"{description} {sentence}".strip()


def config_sections(
    model: type[BaseModel],
    prefix: str = "",
    _stack: tuple[type[BaseModel], ...] = (),
    *,
    notes: collections.abc.Mapping[str, str] | None = None,
) -> list[tuple[str, list[ConfigRow]]]:
    """``(YAML path, rows)`` per section: the model's own keys, then every nested model
    depth-first in field order. A model already being rendered is not entered again.
    ``notes`` maps a YAML path to a sentence its row's description ends with."""
    notes = notes or {}
    stack = (*_stack, model)
    rows: list[ConfigRow] = []
    nested: list[tuple[str, type[BaseModel]]] = []
    for name, info in model.model_fields.items():
        key = _yaml_key(name, info)
        path = f"{prefix}.{key}" if prefix else key
        children = [
            (path + suffix, sub)
            for suffix, sub in _nested_models(info.annotation)
            if sub not in stack
        ]
        nested.extend(children)
        description = escape_cell(strip_internal_phase(info.description or ""))
        links = ", ".join(f"[{_code(child)}](#{github_slug(child)})" for child, _ in children)
        if links:
            description = _with_sentence(description, f"See {links}.")
        if path in notes:
            description = _with_sentence(description, escape_cell(notes[path]))
        rows.append(
            ConfigRow(
                key=path,
                type=_code(format_type(info.annotation)),
                default=_config_default(info),
                constraints=format_constraints(info),
                description=description,
            )
        )
    sections = [(prefix, rows)]
    for child_path, sub in nested:
        sections.extend(config_sections(sub, child_path, stack, notes=notes))
    return sections


def _load_config_model() -> type[BaseModel]:
    from soup_cli.config.schema import SoupConfig

    return SoupConfig


def render_config_reference(model: type[BaseModel] | None = None) -> str:
    """The text of ``docs/config-reference.md``."""
    sections = config_sections(
        model if model is not None else _load_config_model(), notes=staged_notes()
    )
    lines = [
        "# Config reference",
        "",
        "[← Back to the docs index](README.md)",
        "",
        "> **Generated file — do not edit by hand.**",
        "> It is rendered from `SoupConfig` in `src/soup_cli/config/schema.py`. After adding or",
        "> removing a field, or changing its type, default or bounds, run",
        f"> `{_REGENERATE}` and commit the result; the test suite fails",
        "> until the page matches. Descriptions are refreshed when a release is cut, so between",
        "> releases the wording here can trail the code.",
        "",
        "Every key `soup.yaml` accepts. A key that is not on this page **refuses the load** — see",
        "[Unknown config keys](backends-and-ops.md#unknown-config-keys) for the error and the",
        "nearest-name hint. A key whose description ends \"Refused at load when set away from",
        "its default\" is declared but read by nothing: leave it out, or at its default.",
        "The guides linked from the [docs index](README.md) explain what the settings do",
        "together; this page is the lookup table.",
        "",
        "How to read a table:",
        "",
        "- **Key** is the full YAML path: `training.lora.r` is `r:` under `lora:` under",
        "  `training:`.",
        "- **Type** uses `str`, `int`, `float`, `bool`, `list[...]` and `dict[...]`; `null` means",
        "  the key may be left out or set to `null`; `section` is a nested block with a table of",
        "  its own. A value in quotes (`\"auto\"`) is the one literal accepted next to the type.",
        "- **Default** is written the way YAML reads it (`null`, `true`, `\"auto\"`, `[\"a\"]`).",
        "  **required** means there is none. *(set at runtime)* marks a default computed on",
        "  the machine, which this page does not print.",
        "- **Constraints** are the rules on the value itself: bounds, lengths, the allowed",
        "  choices. Rules that span keys (a setting valid for one task or backend only) are",
        "  enforced by the schema's validators and named in the description.",
        "",
        "Sections:",
        "",
    ]
    # One line per section and one row per key, never a total: two changes that each
    # add a different key then merge as text and the merged page is still right.
    for path, _rows in sections:
        title = _code(path) if path else "Top-level keys"
        anchor = github_slug(path) if path else "top-level-keys"
        lines.append(f"- [{title}](#{anchor})")
    for path, rows in sections:
        lines += ["", f"## {_code(path)}" if path else "## Top-level keys", ""]
        lines += _table_head("Key", "Type", "Default", "Constraints", "Description")
        lines += [
            _row([_code(row.key), row.type, row.default, row.constraints, row.description])
            for row in rows
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# docs/cli-reference.md
# --------------------------------------------------------------------------

#: Typer builds these two as flags when it can detect the shell and as options that
#: take a shell name when it cannot (no ``shellingham``), with different help text.
_TYPER_COMPLETION_HELP = {
    "install_completion": "Install shell completion for the current shell.",
    "show_completion": "Print the shell completion script, to copy or customise it.",
}


@dataclass(frozen=True)
class CommandEntry:
    """One visible command and the names that lead to it (``("data", "inspect")``)."""

    path: tuple[str, ...]
    command: click.Command


def _visible_params(command: click.Command) -> list[click.Parameter]:
    return [param for param in command.params if not getattr(param, "hidden", False)]


def iter_commands(root: click.Command) -> list[CommandEntry]:
    """Every visible command below ``root``: a group before its subcommands, names sorted.

    A hidden group hides everything under it.
    """
    entries: list[CommandEntry] = []

    def walk(command: click.Command, path: tuple[str, ...]) -> None:
        if path:
            entries.append(CommandEntry(path, command))
        if _is_group(command):
            for name in sorted(command.commands):
                sub = command.commands[name]
                if not getattr(sub, "hidden", False):
                    walk(sub, (*path, name))

    walk(root, ())
    return entries


def _is_typer_completion(param: click.Parameter) -> bool:
    return param.name in _TYPER_COMPLETION_HELP and not param.expose_value


def _range_words(param_type: Any) -> str:
    low, high = param_type.min, param_type.max
    if low is not None and high is not None and not (param_type.min_open or param_type.max_open):
        return f"{low!r} to {high!r}"
    parts = []
    if low is not None:
        parts.append(("above " if param_type.min_open else "at least ") + repr(low))
    if high is not None:
        parts.append(("below " if param_type.max_open else "at most ") + repr(high))
    return ", ".join(parts)


def _is_group(command: object) -> bool:
    return isinstance(getattr(command, "commands", None), collections.abc.Mapping)


def _is_option(parameter: object) -> bool:
    return getattr(parameter, "param_type_name", None) == "option"


def _is_argument(parameter: object) -> bool:
    return getattr(parameter, "param_type_name", None) == "argument"


def _click_type_name(param_type: click.ParamType) -> str:
    # These are parameter capabilities, shared by Click and Typer's own types.
    # The script also accepts plain Click trees without importing Soup's CLI.
    if hasattr(param_type, "choices"):
        values = [str(getattr(choice, "value", choice)) for choice in param_type.choices]
        return "one of " + ", ".join(_code(value) for value in values)
    name = getattr(param_type, "name", "")
    if name in ("integer range", "int range", "float range"):
        kind = "float" if name == "float range" else "integer"
        return ", ".join(part for part in (kind, _range_words(param_type)) if part)
    return {
        "integer": "integer",
        "int": "integer",
        "float": "float",
        "boolean": "boolean",
        "path": "path",
        "filename": "file",
        "datetime": "datetime",
    }.get(name, "text")


def format_click_type(param: click.Parameter) -> str:
    """The Type cell of a parameter, read off the Click object (never its ``repr``)."""
    if _is_typer_completion(param) or getattr(param, "is_flag", False):
        return "flag"
    if getattr(param, "count", False):
        return "counter"
    parts = [_click_type_name(param.type)]
    if param.nargs == -1:
        parts.append("any number")
    elif param.nargs != 1:
        parts.append(f"{param.nargs} values")
    if getattr(param, "multiple", False):
        parts.append("repeatable")
    return ", ".join(parts)


def format_click_default(param: click.Parameter) -> str:
    """The Default cell: the value as typed on the command line, ``on`` / ``off`` for a
    flag, empty when there is none, the placeholder when the machine computed it."""
    value = param.default
    if _is_typer_completion(param) or value is None:
        return ""
    if type(value).__module__.partition(".")[0] == "click":
        return ""  # Click's own "no default" sentinel
    if callable(value) and not isinstance(value, enum.Enum):
        return f"*{RUNTIME_DEFAULT}*"
    if getattr(param, "is_flag", False) and isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (list, tuple, set, frozenset, dict)) and not value:
        return ""
    plain = _plain_literal(value)
    if plain is _NOT_LITERAL:
        return f"*{RUNTIME_DEFAULT}*"
    if isinstance(plain, str):
        return _code(plain) if plain else _code('""')
    return _code(json.dumps(plain, ensure_ascii=False))


def _param_name(param: click.Parameter) -> str:
    if _is_option(param):
        names = ", ".join(_code(opt) for opt in param.opts)
        if param.secondary_opts:
            names += " / " + ", ".join(_code(opt) for opt in param.secondary_opts)
        return names
    return _code(_metavar(param))


def _argument_name(param: click.Parameter) -> str:
    return getattr(param, "metavar", None) or (param.name or "").upper()


def _metavar(param: click.Parameter) -> str:
    name = _argument_name(param)
    return f"{name}..." if param.nargs == -1 else name


def _declared_argument_help(command: click.Command) -> dict[str, str]:
    """``{argument name: help}`` as declared to Typer.

    Click 8.5 gave ``Argument`` a ``help`` of its own and resets the attribute Typer
    0.20 has already set, so ``param.help`` is ``None`` there and the declared string
    on an older Click. The declaration reads the same on both.
    """
    if command.callback is None:
        return {}
    try:
        from typer.models import ArgumentInfo
        from typer.utils import get_params_from_function

        declared = get_params_from_function(command.callback)
    except Exception:
        return {}
    return {
        name: meta.default.help
        for name, meta in declared.items()
        if isinstance(meta.default, ArgumentInfo) and meta.default.help
    }


def _param_help(param: click.Parameter, declared: dict[str, str]) -> str:
    if _is_typer_completion(param):
        return _TYPER_COMPLETION_HELP[param.name or ""]
    raw = getattr(param, "help", None) or declared.get(param.name or "") or ""
    text = escape_cell(strip_internal_phase(strip_rich_markup(raw)))
    envvar = getattr(param, "envvar", None)
    if envvar:
        names = [envvar] if isinstance(envvar, str) else list(envvar)
        if text and not text.endswith((".", "!", "?", ":")):
            text += "."
        text = f"{text} Env: {', '.join(_code(name) for name in names)}.".strip()
    return text


def _param_rows(command: click.Command) -> list[str]:
    declared = _declared_argument_help(command)
    return [
        _row(
            [
                _param_name(param),
                format_click_type(param),
                format_click_default(param),
                "yes" if param.required else "",
                _param_help(param, declared),
            ]
        )
        for param in _visible_params(command)
    ]


def _usage(path: tuple[str, ...], command: click.Command) -> str:
    params = _visible_params(command)
    parts = ["soup", *path]
    if any(_is_option(param) for param in params):
        parts.append("[OPTIONS]")
    if _is_group(command):
        # `<command>`, not Click's `COMMAND`: the docs guard that resolves every
        # documented invocation (#1465) reads a bare capitalised word as a subcommand.
        optional = bool(path) and command.invoke_without_command
        parts.append("[<command>] [ARGS]..." if optional else "<command> [ARGS]...")
    for param in params:
        if _is_argument(param):
            name = _argument_name(param)
            name = name if param.required else f"[{name}]"
            parts.append(f"{name}..." if param.nargs == -1 else name)
    passes_extra = (command.context_settings or {}).get("allow_extra_args")
    variadic = any(_is_argument(param) and param.nargs == -1 for param in params)
    if passes_extra and not variadic and not _is_group(command):
        parts.append("[ARGS]...")
    return " ".join(parts)


def _command_summary(command: click.Command) -> str:
    return _escape_paragraph(strip_internal_phase(strip_rich_markup(summary_line(command.help))))


def _command_section(entry: CommandEntry, anchors: dict[tuple[str, ...], str]) -> list[str]:
    command = entry.command
    level = "#" * min(len(entry.path) + 1, 6)
    blocks = [[f"{level} {_code('soup ' + ' '.join(entry.path))}"]]
    summary = _command_summary(command)
    if summary:
        blocks.append([summary])
    blocks.append([f"Usage: {_code(_usage(entry.path, command))}"])
    if _is_group(command):
        subs = [
            f"- [{_code(name)}](#{anchors[(*entry.path, name)]})"
            for name in sorted(command.commands)
            if (*entry.path, name) in anchors
        ]
        if subs:  # one line per subcommand, so adding one never rewrites a shared line
            blocks.append(["Subcommands:", "", *subs])
    rows = _param_rows(command)
    if rows:
        blocks.append([*_table_head("Name", "Type", "Default", "Required", "Help"), *rows])
    elif not _is_group(command):
        blocks.append(["Takes no options or arguments."])
    return [line for block in blocks for line in ("", *block)]


def _load_cli_root() -> click.Command:
    import typer

    from soup_cli.cli import app

    return typer.main.get_command(app)


def render_cli_reference(root: click.Command | None = None) -> str:
    """The text of ``docs/cli-reference.md``."""
    root = root if root is not None else _load_cli_root()
    entries = iter_commands(root)
    slugger = _Slugger()
    for fixed in ("CLI reference", "Global options", "Commands"):
        slugger.take(fixed)
    anchors = {entry.path: slugger.take("soup " + " ".join(entry.path)) for entry in entries}

    # No totals anywhere on the page: every line below the header belongs to one
    # command or one parameter, so two changes that each add a different option merge
    # as text and the merged page is still right.
    lines = [
        "# CLI reference",
        "",
        "[← Back to the docs index](README.md)",
        "",
        "> **Generated file — do not edit by hand.**",
        "> It is rendered from the live Typer command tree (`soup_cli.cli:app`). After adding or",
        "> removing a command, an option or an argument, or changing its type, its default or",
        f"> whether it is required, run `{_REGENERATE}` and commit the",
        "> result; the test suite fails until the page matches. Help text is refreshed when a",
        "> release is cut, so between releases the wording here can trail the code.",
        "",
        "Every `soup` command with every option and argument. For a one-line-per-task listing",
        "see the [Command reference](commands.md); for the keys of `soup.yaml` see the",
        "[Config reference](config-reference.md).",
        "",
        "How to read a table:",
        "",
        "- **Name** is an option with its short form, or an argument in capitals (`PATH`;",
        "  `PATH...` takes any number of values). `--flag` / `--no-flag` is one switch with",
        "  both spellings.",
        "- **Type**: a `flag` takes no value; `text`, `integer`, `float` and `path` take one;",
        "  a `repeatable` option may be given more than once.",
        "- **Default** is the value used when the option is left out (`on` / `off` for a flag);",
        "  an empty cell means there is none. *(set at runtime)* marks a default computed on",
        "  the machine, which this page does not print.",
        "- **Required** says `yes` when the command refuses to run without it.",
        "",
        "`--help` works on every command and is not listed. Global options go before the",
        "command name: `soup --verbose train --config soup.yaml`.",
        "",
        "## Global options",
        "",
        f"Usage: {_code(_usage((), root))}",
        "",
        *_table_head("Name", "Type", "Default", "Required", "Help"),
        *_param_rows(root),
        "",
        "## Commands",
        "",
        *_table_head("Command", "Summary"),
    ]
    for entry in entries:
        if len(entry.path) == 1:
            link = f"[{_code('soup ' + entry.path[0])}](#{anchors[entry.path]})"
            lines.append(_row([link, _command_summary(entry.command)]))
    for entry in entries:
        lines += _command_section(entry, anchors)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Check / write
# --------------------------------------------------------------------------


def render_pages() -> dict[str, str]:
    """``{repo-relative path: text}`` for both pages."""
    return {
        CONFIG_PAGE: render_config_reference(_load_config_model()),
        CLI_PAGE: render_cli_reference(_load_cli_root()),
    }


def _committed(root: Path, rel: str) -> str | None:
    """The page on disk with LF newlines, or ``None`` when there is no such file."""
    path = root / rel
    if not path.is_file():
        return None
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def stale_pages(root: Path, pages: dict[str, str]) -> list[str]:
    """The pages under ``root`` that are missing or differ from ``pages``."""
    return [rel for rel, text in pages.items() if _committed(root, rel) != text]


def _first_difference(committed: str, fresh: str) -> list[str]:
    old, new = committed.split("\n"), fresh.split("\n")
    index = next(
        (i for i, (left, right) in enumerate(zip(old, new)) if left != right),
        min(len(old), len(new)),
    )

    def shown(lines: list[str]) -> str:
        text = lines[index] if index < len(lines) else "<end of file>"
        return text[:160].encode("ascii", "backslashreplace").decode("ascii")

    return [
        f"  first difference at line {index + 1}:",
        f"    committed: {shown(old)}",
        f"    rendered:  {shown(new)}",
    ]


class PageFormatError(ValueError):
    """A table line on a page that is not a row this generator writes."""


#: GitHub ends a cell at a ``|`` unless a backslash stands right before it, whatever
#: comes before that backslash: ``a\|b`` and ``a\\|b`` are both ONE cell (rendered
#: ``a|b`` and ``a\|b``). Read off GitHub's own renderer, not assumed.
_CELL_END = re.compile(r"(?<!\\)\|")
_HEADER_CELLS = frozenset({"Key", "Name", "Command"})


def split_row(line: str) -> list[str]:
    """The cells of one Markdown table row, split where GitHub splits it."""
    return [cell.strip() for cell in _CELL_END.split(line.strip())[1:-1]]


def _data_cells(line: str, lineno: int, columns: int) -> list[str] | None:
    """The cells of a body row; ``None`` for a line that is no table line, or is a header
    or a rule. A table line that is none of those raises: skipping a row that does not
    parse would take its key out of the comparison without a word."""
    if not line.startswith("|"):
        return None
    cells = split_row(line)
    if set(line) <= set("|-: ") or (cells and cells[0] in _HEADER_CELLS):
        return None
    if len(cells) != columns or not cells[0].startswith("`"):
        raise PageFormatError(
            f"line {lineno} is not a {columns}-column row of this page: {line[:70]}"
        )
    return cells


def parse_config_page(text: str) -> dict[str, dict[str, str]]:
    """``{key: {"type": ..., "default": ..., "constraints": ...}}``, read back from the
    tables of a config page. The description column is left out on purpose."""
    keys: dict[str, dict[str, str]] = {}
    for lineno, line in enumerate(text.split("\n"), start=1):
        cells = _data_cells(line, lineno, 5)
        if cells is not None:
            keys[cells[0].replace("`", "")] = {
                "type": cells[1],
                "default": cells[2],
                "constraints": cells[3],
            }
    return keys


def parse_cli_page(text: str) -> dict[str, dict[str, dict[str, str]]]:
    """``{command: {parameter names: {"type": ..., "default": ..., "required": ...}}}``,
    read back from a CLI page; the global options are under ``"soup"``. Help text, the
    usage line and the command summary are left out on purpose."""
    commands: dict[str, dict[str, dict[str, str]]] = {}
    current: dict[str, dict[str, str]] | None = None
    for lineno, line in enumerate(text.split("\n"), start=1):
        if line.startswith("#"):
            title = line.lstrip("#").strip().replace("`", "")
            if title == "Global options":
                title = "soup"
            is_command = title == "soup" or title.startswith("soup ")
            current = commands.setdefault(title, {}) if is_command else None
            continue
        if current is None:
            continue
        cells = _data_cells(line, lineno, 5)
        if cells is not None:
            current[cells[0].replace("`", "")] = {
                "type": cells[1],
                "default": cells[2],
                "required": cells[3],
            }
    return commands


def _shown(cell: str) -> str:
    return cell.replace("\\|", "|") or "(empty)"


def _stated(cells: dict[str, str], columns: Sequence[str] | None = None) -> str:
    chosen = cells if columns is None else {column: cells[column] for column in columns}
    return ", ".join(f"{column} {_shown(cell)}" for column, cell in chosen.items())


def _item_drift(
    label: str, old: dict[str, str] | None, new: dict[str, str] | None, absent: str
) -> list[str]:
    """One line for a key or a parameter whose row differs, each side named: what the
    committed page states, then what the code states now."""
    if new is None:
        assert old is not None
        return [f"{label}: committed page: {_stated(old)} | code now: {absent}"]
    if old is None:
        return [f"{label}: committed page: (not listed) | code now: {_stated(new)}"]
    changed = [column for column in new if old[column] != new[column]]
    if not changed:
        return []
    return [
        f"{label}: committed page: {_stated(old, changed)} | code now: {_stated(new, changed)}"
    ]


def config_drift(
    old: dict[str, dict[str, str]], new: dict[str, dict[str, str]]
) -> list[str]:
    """One line per key the committed config page states differently from the schema."""
    lines: list[str] = []
    for key in dict.fromkeys([*new, *old]):
        lines += _item_drift(key, old.get(key), new.get(key), "(no such field)")
    return lines


def cli_drift(
    old: dict[str, dict[str, dict[str, str]]], new: dict[str, dict[str, dict[str, str]]]
) -> list[str]:
    """One line per command or parameter the committed CLI page states differently."""
    lines: list[str] = []
    for command in dict.fromkeys([*new, *old]):
        if command not in old:
            lines.append(f"{command}: committed page: (no section) | code now: a command")
        elif command not in new:
            lines.append(f"{command}: committed page: a section | code now: (no such command)")
        else:
            was, now = old[command], new[command]
            for name in dict.fromkeys([*now, *was]):
                lines += _item_drift(
                    f"{command} {name}", was.get(name), now.get(name), "(no such parameter)"
                )
    return lines


def _page_drift(
    parse: Callable[[str], Any],
    compare: Callable[[Any, Any], list[str]],
    committed: str,
    fresh: str,
) -> list[str]:
    new = parse(fresh)  # a render that cannot be read back is a bug in here: let it raise
    try:
        old = parse(committed)
    except PageFormatError as error:
        return [f"the committed page cannot be read back: {error}"]
    return compare(old, new)


def structure_drift(root: Path, pages: dict[str, str]) -> dict[str, list[str]]:
    """``{page: differences}`` for every page under ``root`` whose tables no longer state
    what the code does. Wording is not looked at. A missing page is one difference, and
    so is a committed page with a table line that cannot be read back."""
    found: dict[str, list[str]] = {}
    for rel, fresh in pages.items():
        committed = _committed(root, rel)
        if committed is None:
            found[rel] = ["the page is missing"]
        elif rel == CONFIG_PAGE:
            found[rel] = _page_drift(parse_config_page, config_drift, committed, fresh)
        else:
            found[rel] = _page_drift(parse_cli_page, cli_drift, committed, fresh)
    return {rel: lines for rel, lines in found.items() if lines}


def _report_structure(drift: dict[str, list[str]]) -> None:
    for rel, lines in drift.items():
        subject = "the schema" if rel == CONFIG_PAGE else "the CLI"
        count = f"{len(lines)} difference{'s' if len(lines) != 1 else ''}"
        print(f"{rel} no longer matches {subject} ({count}):")
        for line in lines[:_MAX_SHOWN]:
            print("  " + line.encode("ascii", "backslashreplace").decode("ascii"))
        if len(lines) > _MAX_SHOWN:
            print(f"  ... and {len(lines) - _MAX_SHOWN} more")
    print(f"Run `{_REGENERATE}` and commit the result.")


def _ensure_checkout_on_path() -> None:
    """Import ``soup_cli`` from this checkout when the script is run by path."""
    src = str(_SCRIPT_REPO_ROOT / "src")
    if "soup_cli" not in sys.modules and src not in sys.path:
        sys.path.insert(0, src)


def _foreign_install() -> str | None:
    """Where ``soup_cli`` was imported from, when that is not this checkout's ``src``."""
    import soup_cli

    src = os.path.normcase(os.path.realpath(_SCRIPT_REPO_ROOT / "src"))
    origin = os.path.normcase(os.path.realpath(soup_cli.__file__))
    try:
        inside = os.path.commonpath([src, origin]) == src
    except ValueError:  # different drives
        inside = False
    return None if inside else soup_cli.__file__


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite the pages, or with one of the two checks report and exit 1."""
    parser = argparse.ArgumentParser(
        description="Render docs/config-reference.md and docs/cli-reference.md from the code."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-structure",
        action="store_true",
        help=(
            "write nothing; exit 1 and name each key, command or parameter the pages state "
            "differently from the code (wording is not compared)"
        ),
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 and name each page whose text differs from a fresh render",
    )
    args = parser.parse_args(argv)

    _ensure_checkout_on_path()
    foreign = _foreign_install()
    if foreign is not None:
        print(f"soup_cli is imported from {foreign}, not from {_SCRIPT_REPO_ROOT / 'src'}.")
        print("The pages would describe that copy. Install this checkout (pip install -e .)")
        print("or put its src directory first on PYTHONPATH, then run this again.")
        return 2

    pages = render_pages()
    if args.check_structure:
        drift = structure_drift(_REPO_ROOT, pages)
        if drift:
            _report_structure(drift)
            return 1
        print(f"{CONFIG_PAGE} and {CLI_PAGE} state what the code does (wording not compared).")
        return 0
    stale = stale_pages(_REPO_ROOT, pages)
    if not stale:
        state = "up to date" if args.check else "up to date; nothing to do"
        print(f"{CONFIG_PAGE} and {CLI_PAGE} are {state}.")
        return 0
    if args.check:
        for rel in stale:
            committed = _committed(_REPO_ROOT, rel)
            if committed is None:
                print(f"{rel} is missing.")
                continue
            print(f"{rel} is stale.")
            for line in _first_difference(committed, pages[rel]):
                print(line)
        print(f"Run `{_REGENERATE}` and commit the result.")
        return 1
    for rel in stale:
        target = _REPO_ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pages[rel].encode("utf-8"))
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
