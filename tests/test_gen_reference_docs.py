"""``scripts/gen_reference_docs.py``: the generated config and CLI reference pages.

Four things are held here.

1. **The committed pages state what the code does** (``--check-structure``): the same
   keys with the same type, default and constraints, the same commands with the same
   parameters. Wording is NOT compared, on purpose: a pull request that rewords a help
   string or a description must not go red here. The byte-exact check (``--check``) is
   a release-time one and runs only with ``SOUP_DOCS_EXACT=1``.
2. **The render is the same everywhere.** CI runs this on Python 3.11 to 3.13, on
   Linux, Windows and macOS, so every helper that keeps the output identical across
   them is pinned on its own: the type spelling, the default and constraint rendering,
   the cell escaping, the Rich-markup stripping.
3. **Coverage is complete.** Every ``SoupConfig`` key is exactly one row, every
   registered command one section, every visible option and argument one row. The
   schema and the Click tree are walked HERE, independently of the generator's own
   walkers, so a walker that skips something cannot agree with itself.
4. **Rendering is import-light.** ``conftest.py`` has already imported torch into the
   pytest process, so that part runs in a fresh interpreter, like
   ``tests/test_cli_startup_is_light.py``.

Nothing below pins a value that lives in the schema or in a help string (a default, a
bound, a sentence). Those change in ordinary pull requests, and this file must not be
one more thing to edit then: shapes are pinned on the toy model and the toy apps, and
what is asserted about the real pages is derived from the real schema and Click tree.
"""

from __future__ import annotations

import collections.abc
import difflib
import enum
import json
import os
import re
import subprocess
import sys
import tempfile
import typing
from collections import Counter
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

import click
import pytest
import typer
from pydantic import BaseModel, Field, create_model

from scripts import gen_reference_docs as gen
from soup_cli.utils.typer_compat import is_argument, is_group, is_option

_ROOT = Path(__file__).resolve().parents[1]
_HEAVY = ("torch", "transformers", "accelerate", "peft", "trl", "datasets", "bitsandbytes")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
#: the same shape ``tests/test_issue823_stale_feature_descriptions.py`` bans from CLI help
_INTERNAL_PHASE = re.compile(r"\bv\d+(?:\.\d+){1,2}\s+part\s+[a-z]\b", re.IGNORECASE)
_CONFIG_ROW_KEY = re.compile(r"^\| `([^`]+)` \|", re.MULTILINE)
_COMMAND_HEADING = re.compile(r"^#{2,6} `(soup [^`]+)`$")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _plain(text: str) -> str:
    """ANSI-strip + whitespace-collapse, as in ``tests/test_auto_tuning.py``."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _cells(row: str) -> list[str]:
    """The cells of one Markdown table row (an escaped ``\\|`` stays inside its cell)."""
    return [cell.strip() for cell in _UNESCAPED_PIPE.split(row.strip())[1:-1]]


def _rows(lines: list[str]) -> list[list[str]]:
    """Body rows of the tables in ``lines``: header and separator rows dropped."""
    out = []
    for index, line in enumerate(lines):
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        if index + 1 < len(lines) and set(lines[index + 1]) <= set("|- ") and lines[index + 1]:
            continue  # the header row sits right above its separator
        out.append(_cells(line))
    return out


@pytest.fixture(scope="module")
def pages() -> dict[str, str]:
    return gen.render_pages()


@pytest.fixture(scope="module")
def config_page(pages: dict[str, str]) -> str:
    return pages[gen.CONFIG_PAGE]


@pytest.fixture(scope="module")
def cli_page(pages: dict[str, str]) -> str:
    return pages[gen.CLI_PAGE]


def _mirror(tmp_path: Path, rendered: dict[str, str]) -> None:
    """Write ``rendered`` under ``tmp_path`` the way the generator would."""
    (tmp_path / "docs").mkdir(exist_ok=True)
    for rel, text in rendered.items():
        (tmp_path / rel).write_bytes(text.encode("utf-8"))


# --------------------------------------------------------------------------
# 1. the committed pages: structure always, bytes at release
# --------------------------------------------------------------------------

_EXACT_REASON = (
    "byte-exact freshness is a release-time check: maintainers set SOUP_DOCS_EXACT=1 when "
    "they regenerate the reference pages for a release. Between releases only the structure "
    "is held (test_the_committed_pages_state_what_the_code_does), so that rewording a help "
    "string does not fail a pull request."
)


def test_the_committed_pages_state_what_the_code_does(capsys: pytest.CaptureFixture[str]) -> None:
    """The default freshness test. If it fails, the message names the keys or parameters
    and the command to run; a wording-only change never fails it."""
    code = gen.main(["--check-structure"])
    report = capsys.readouterr().out
    assert code == 0, "\n" + report  # one offending key per line: do not collapse them
    assert "state what the code does" in _plain(report)


@pytest.mark.skipif(os.environ.get("SOUP_DOCS_EXACT") != "1", reason=_EXACT_REASON)
def test_the_committed_pages_are_byte_exact(capsys: pytest.CaptureFixture[str]) -> None:
    code = gen.main(["--check"])
    out = _plain(capsys.readouterr().out)
    assert code == 0, out
    assert "up to date" in out, out


def test_the_exact_test_is_skipped_with_a_reason_a_contributor_can_act_on() -> None:
    marks = [mark for mark in test_the_committed_pages_are_byte_exact.pytestmark]
    assert [mark.name for mark in marks] == ["skipif"]
    assert "SOUP_DOCS_EXACT=1" in marks[0].kwargs["reason"]
    assert "release" in marks[0].kwargs["reason"]


def test_the_two_checks_cannot_be_asked_for_together(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        gen.main(["--check", "--check-structure"])
    assert stopped.value.code == 2
    assert "not allowed with" in _plain(capsys.readouterr().err)


def test_the_generator_renders_exactly_the_two_pages(pages: dict[str, str]) -> None:
    assert sorted(pages) == ["docs/cli-reference.md", "docs/config-reference.md"]
    assert {gen.CONFIG_PAGE, gen.CLI_PAGE} == set(pages)


def test_the_docs_index_links_both_pages() -> None:
    """Nothing else links them: a page missing from the index cannot be found."""
    index = (_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    rows = [line for line in index.splitlines() if line.startswith("| [")]
    assert sum("](config-reference.md)" in row for row in rows) == 1, rows
    assert sum("](cli-reference.md)" in row for row in rows) == 1, rows


def test_check_names_only_the_page_that_lost_a_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pages: dict[str, str],
) -> None:
    _mirror(tmp_path, pages)
    monkeypatch.setattr(gen, "_REPO_ROOT", tmp_path)
    assert gen.main(["--check"]) == 0, capsys.readouterr().out

    page = tmp_path / gen.CONFIG_PAGE
    lines = page.read_bytes().decode("utf-8").splitlines(keepends=True)
    victim = next(i for i, line in enumerate(lines) if line.startswith("| `"))
    key = _cells(lines[victim])[0].strip("`")
    del lines[victim]
    page.write_bytes("".join(lines).encode("utf-8"))
    capsys.readouterr()

    assert gen.main(["--check"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/config-reference.md is stale" in out, out
    assert "docs/cli-reference.md" not in out, out
    assert out.endswith("Run `python scripts/gen_reference_docs.py` and commit the result."), out
    assert f"first difference at line {victim + 1}" in out, out

    assert gen.main(["--check-structure"]) == 1
    out = _plain(capsys.readouterr().out)
    assert f"{key}: committed page: (not listed) | code now: type " in out, out
    assert "docs/cli-reference.md" not in out, out


def test_check_fails_when_the_schema_gains_a_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pages: dict[str, str],
) -> None:
    from soup_cli.config.schema import SoupConfig

    class Grown(SoupConfig):
        brand_new_knob: int = Field(default=7, ge=1, le=9, description="Not on the page yet.")

    _mirror(tmp_path, pages)
    monkeypatch.setattr(gen, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(gen, "_load_config_model", lambda: Grown)

    assert gen.main(["--check"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/config-reference.md is stale" in out, out
    assert "docs/cli-reference.md" not in out, out

    assert gen.main(["--check-structure"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/config-reference.md no longer matches the schema (1 difference)" in out, out
    assert (
        "brand_new_knob: committed page: (not listed) | code now: type `int`, default `7`" in out
    ), out
    assert "docs/cli-reference.md" not in out, out


def test_check_fails_when_a_command_gains_an_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pages: dict[str, str],
) -> None:
    root = _live_root()
    title, command = next(
        (title, cmd)
        for title, cmd in _command_tree(root).items()
        if not is_group(cmd) and _visible(cmd)
    )
    command.params.append(
        click.Option(["--brand-new-flag"], is_flag=True, default=False, help="Not yet.")
    )
    _mirror(tmp_path, pages)
    monkeypatch.setattr(gen, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(gen, "_load_cli_root", lambda: root)

    assert gen.main(["--check"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/cli-reference.md is stale" in out, out
    assert "docs/config-reference.md" not in out, out

    assert gen.main(["--check-structure"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/cli-reference.md no longer matches the CLI (1 difference)" in out, out
    listed = f"{title} --brand-new-flag: committed page: (not listed) | code now: type flag"
    assert listed in out, out
    assert "docs/config-reference.md" not in out, out


def test_check_reports_a_page_that_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gen, "_REPO_ROOT", tmp_path)

    assert gen.main(["--check"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/config-reference.md is missing" in out, out
    assert "docs/cli-reference.md is missing" in out, out

    assert gen.main(["--check-structure"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/config-reference.md no longer matches the schema (1 difference)" in out, out
    assert "docs/cli-reference.md no longer matches the CLI (1 difference)" in out, out
    assert out.count("the page is missing") == 2, out
    assert not (tmp_path / "docs").exists(), "a check must not write anything"


def test_writing_produces_lf_utf8_pages_that_then_pass_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(gen, "_REPO_ROOT", tmp_path)

    assert gen.main([]) == 0
    first = _plain(capsys.readouterr().out)
    assert "wrote docs/config-reference.md" in first, first
    assert "wrote docs/cli-reference.md" in first, first
    for rel in (gen.CONFIG_PAGE, gen.CLI_PAGE):
        raw = (tmp_path / rel).read_bytes()
        assert b"\r" not in raw, rel
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), rel
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{rel} carries a BOM"
        raw.decode("utf-8")

    assert gen.main(["--check"]) == 0
    capsys.readouterr()
    assert gen.main([]) == 0
    assert "nothing to do" in _plain(capsys.readouterr().out)


def test_check_reads_a_crlf_checkout_as_the_same_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pages: dict[str, str]
) -> None:
    """``core.autocrlf=true`` (the Windows runners' default) checks the pages out with
    CRLF; the committed blob is still the LF text the generator renders."""
    (tmp_path / "docs").mkdir()
    for rel, text in pages.items():
        (tmp_path / rel).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    monkeypatch.setattr(gen, "_REPO_ROOT", tmp_path)

    assert gen.stale_pages(tmp_path, pages) == []
    assert gen.structure_drift(tmp_path, pages) == {}
    assert gen.main(["--check"]) == 0
    assert gen.main(["--check-structure"]) == 0


@pytest.mark.parametrize("mode", ["--check", "--check-structure"])
def test_a_shadowing_install_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    """A same-version copy of ``soup_cli`` in site-packages shadows the working tree;
    pages rendered from it would describe other code and still look plausible."""
    import soup_cli

    elsewhere = tmp_path / "site-packages" / "soup_cli" / "__init__.py"
    monkeypatch.setattr(soup_cli, "__file__", str(elsewhere))

    assert gen.main([mode]) == 2
    out = _plain(capsys.readouterr().out)
    assert "soup_cli is imported from" in out, out
    assert "site-packages" in out, out


# --------------------------------------------------------------------------
# 2a. config page coverage
# --------------------------------------------------------------------------


def _is_model(annotation: Any) -> bool:
    return (
        typing.get_origin(annotation) is None
        and isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
    )


def _schema_fields(model: type[BaseModel], prefix: str = "") -> list[tuple[str, Any, bool]]:
    """``(path, FieldInfo, is a nested model)`` for every field, walked here on purpose."""
    fields: list[tuple[str, Any, bool]] = []
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        nested = [m for m in (info.annotation, *typing.get_args(info.annotation)) if _is_model(m)]
        fields.append((path, info, bool(nested)))
        for member in nested:
            fields.extend(_schema_fields(member, f"{path}."))
    return fields


def _schema_paths(model: type[BaseModel]) -> tuple[list[str], list[str]]:
    """(every field path, every path that is a nested model)."""
    fields = _schema_fields(model)
    return [path for path, _, _ in fields], [path for path, _, nested in fields if nested]


def test_every_config_field_is_exactly_one_row(config_page: str) -> None:
    from soup_cli.config.schema import SoupConfig

    paths, sections = _schema_paths(SoupConfig)
    assert len(paths) >= 300, len(paths)
    assert "training.lora" in sections, sections

    rendered = Counter(_CONFIG_ROW_KEY.findall(config_page))
    assert set(rendered) == set(paths), (
        sorted(set(paths) - set(rendered)),
        sorted(set(rendered) - set(paths)),
    )
    assert [key for key, count in rendered.items() if count != 1] == []


def test_every_nested_model_is_a_section_named_by_its_yaml_path(config_page: str) -> None:
    from soup_cli.config.schema import SoupConfig

    _, sections = _schema_paths(SoupConfig)
    headings = [line for line in config_page.splitlines() if line.startswith("## ")]
    assert headings == ["## Top-level keys", *(f"## `{path}`" for path in sections)]
    listed = re.findall(r"^- \[`?([^`\]]+)`?\]\(#", config_page, re.MULTILINE)
    assert listed == ["Top-level keys", *sections], "one line per section in the header list"


def test_every_row_states_its_own_fields_default_and_bounds(config_page: str) -> None:
    """Every scalar default and every bound of the real schema, recomputed here from the
    ``FieldInfo`` with ``json.dumps`` / ``repr`` rather than the generator's formatters.
    Derived, not pinned: changing a default in the schema never needs an edit here."""
    from soup_cli.config.schema import SoupConfig

    stated = gen.parse_config_page(config_page)
    scalars = bounds = required = 0
    for path, info, nested in _schema_fields(SoupConfig):
        cells = stated[path]
        if info.is_required():
            required += 1
            assert cells["default"] == "**required**", path
        elif info.default_factory is None and not nested and cells["default"].startswith("`"):
            scalars += 1  # a code cell: "(set at runtime)", were one ever needed, is not
            literal = json.dumps(info.default, ensure_ascii=False)
            assert cells["default"].replace("\\|", "|") == f"`{literal}`", path
        for meta in info.metadata:
            for attr, sign in (("gt", ">"), ("ge", ">="), ("lt", "<"), ("le", "<=")):
                bound = getattr(meta, attr, None)
                if bound is not None:
                    bounds += 1
                    assert f"`{sign} {bound!r}`" in cells["constraints"], (path, cells)
    assert required >= 1 and scalars >= 250 and bounds >= 100, (required, scalars, bounds)


#: The sentence every staged key's row ends with (#808), as a reader sees it.
_STAGED_NOTE = "Refused at load when set away from its default (v{version}, #808)."


@pytest.mark.parametrize("source", ["rendered", "committed"])
def test_every_staged_key_says_it_is_refused(config_page: str, source: str) -> None:
    """A staged key passes the schema and is then refused by the loader (#808), so a
    field description does not say so by itself. Every key ``STAGED_FIELDS`` lists
    says it, and no other key does. The committed page is read too: the structure
    check leaves the description column out, so only this test holds it there."""
    from soup_cli.config.staged_fields import STAGED_FIELD_REJECTION_VERSION, STAGED_FIELDS

    if source == "rendered":
        page = config_page
    else:
        page = (_ROOT / gen.CONFIG_PAGE).read_text(encoding="utf-8")
    note = _STAGED_NOTE.format(version=STAGED_FIELD_REJECTION_VERSION)
    descriptions = {cells[0]: cells[4] for cells in _rows(page.splitlines()) if len(cells) == 5}
    staged = sorted(f"`{section}.{name}`" for section, name in STAGED_FIELDS)

    assert len(staged) >= 31, staged
    assert [key for key in staged if not descriptions[key].endswith(note)] == []
    assert sorted(key for key, text in descriptions.items() if note in text) == staged


def test_a_field_added_to_the_schema_gets_a_full_row() -> None:
    from soup_cli.config.schema import SoupConfig

    class Grown(SoupConfig):
        brand_new_knob: Optional[int] = Field(default=7, ge=1, le=9, description="New | knob.")

    page = gen.render_config_reference(Grown)
    row = next(line for line in page.splitlines() if line.startswith("| `brand_new_knob` |"))
    assert _cells(row) == [
        "`brand_new_knob`",
        "`int \\| null`",
        "`7`",
        "`>= 1`, `<= 9`",
        "New \\| knob.",
    ]


class _Inner(BaseModel):
    depth: int = Field(default=2, ge=1, le=8, description="How deep.")


class _Toy(BaseModel):
    name: str = Field(description="Required, with a | pipe,\nan <angle> and `a|b` code.")
    mode: Literal["fast", "slow"] = Field(default="fast", description="Speed.")
    ratio: Optional[float] = Field(default=None, gt=0.0, lt=1.0, allow_inf_nan=False)
    tags: List[str] = Field(default_factory=lambda: ["a"], min_length=1, max_length=4)
    stamp: str = Field(default_factory=lambda: "computed")
    workdir: str = Field(default="C:\\Users\\someone\\out")
    code: str = Field(default="x", pattern=r"^[a-z|]+$")
    inner: _Inner = Field(default_factory=_Inner, description="Nested (v0.1.0 Part B).")
    maybe: Optional[_Inner] = None
    renamed: int = Field(default=1, alias="yamlName")


def test_a_toy_model_renders_every_kind_of_row() -> None:
    page = gen.render_config_reference(_Toy)
    rows = {cells[0]: cells[1:] for cells in _rows(page.splitlines()) if len(cells) == 5}

    assert rows["`name`"] == [
        "`str`",
        "**required**",
        "",
        "Required, with a \\| pipe, an &lt;angle> and `a\\|b` code.",
    ]
    assert rows["`mode`"] == ["`str`", '`"fast"`', 'one of `"fast"`, `"slow"`', "Speed."]
    assert rows["`ratio`"] == ["`float \\| null`", "`null`", "`> 0.0`, `< 1.0`; finite", ""]
    assert rows["`tags`"] == ["`list[str]`", '`["a"]`', "min length 1; max length 4", ""]
    assert rows["`stamp`"][1] == "*(set at runtime)*", "a scalar from a factory is computed"
    assert rows["`workdir`"][1] == "*(set at runtime)*"
    assert rows["`code`"][2] == "pattern `^[a-z\\|]+$`"
    assert rows["`inner`"] == [
        "`section`",
        "section defaults",
        "",
        "Nested (v0.1.0). See [`inner`](#inner).",
    ]
    assert rows["`maybe`"] == ["`section \\| null`", "`null`", "", "See [`maybe`](#maybe)."]
    assert rows["`yamlName`"][0] == "`int`", "the YAML key is the alias, not the attribute"
    assert "`renamed`" not in rows
    assert rows["`inner.depth`"] == ["`int`", "`2`", "`>= 1`, `<= 8`", "How deep."]
    assert rows["`maybe.depth`"] == rows["`inner.depth`"]
    assert "someone" not in page


def test_a_self_referencing_model_does_not_recurse_forever() -> None:
    class Node(BaseModel):
        label: str = "x"
        child: Optional["Node"] = None

    Node.model_rebuild()
    sections = gen.config_sections(Node)
    assert [path for path, _rows_ in sections] == [""]
    assert [row.key for row in sections[0][1]] == ["label", "child"]


# --------------------------------------------------------------------------
# 2b. CLI page coverage
# --------------------------------------------------------------------------


def _live_root() -> click.Group:
    from soup_cli.cli import app

    return typer.main.get_command(app)


def _visible(cmd: click.Command) -> list[click.Parameter]:
    return [param for param in cmd.params if not getattr(param, "hidden", False)]


def _command_tree(cmd: click.Command, path: tuple[str, ...] = ()) -> dict[str, click.Command]:
    """``{"soup data inspect": command}`` for every visible command, walked here."""
    found: dict[str, click.Command] = {}
    if path:
        found["soup " + " ".join(path)] = cmd
    if is_group(cmd):
        for name in cmd.list_commands(None) or []:
            sub = cmd.get_command(None, name)
            if sub is not None and not sub.hidden:
                found.update(_command_tree(sub, (*path, name)))
    return found


def _cli_sections(page: str) -> dict[str, list[str]]:
    """``{"soup data inspect": [lines of that section]}``; other headings end a section."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in page.splitlines():
        match = _COMMAND_HEADING.match(line)
        if match:
            assert match.group(1) not in sections, f"two sections for {match.group(1)}"
            current = sections.setdefault(match.group(1), [])
        elif line.startswith("#"):
            current = None
        elif current is not None:
            current.append(line)
    return sections


def _row_names(row: list[str]) -> set[str]:
    return set(re.findall(r"`([^`]+)`", row[0]))


def test_every_registered_command_has_exactly_one_section(cli_page: str) -> None:
    tree = _command_tree(_live_root())
    leaves = [title for title, cmd in tree.items() if not is_group(cmd)]
    assert len(leaves) >= 200, len(leaves)

    sections = _cli_sections(cli_page)
    assert sorted(sections) == sorted(tree), (
        sorted(set(tree) - set(sections)),
        sorted(set(sections) - set(tree)),
    )


def test_every_visible_option_and_argument_is_one_row_of_its_command(cli_page: str) -> None:
    tree = _command_tree(_live_root())
    sections = _cli_sections(cli_page)
    options = 0
    arguments = 0
    for title, cmd in tree.items():
        rows = _rows(sections[title])
        params = _visible(cmd)
        assert len(rows) == len(params), (title, len(rows), len(params))
        for param, row in zip(params, rows):
            if is_option(param):
                options += 1
                assert _row_names(row) == {*param.opts, *param.secondary_opts}, (title, row[0])
            else:
                arguments += 1
                shown = param.metavar or param.name.upper()  # what `--help` prints
                assert row[0] in (f"`{shown}`", f"`{shown}...`"), (title, row[0])
            assert row[3] == ("yes" if param.required else ""), (title, row)
    assert options >= 1000, options
    assert arguments >= 100, arguments


def test_the_global_options_are_listed_before_the_commands(cli_page: str) -> None:
    lines = cli_page.splitlines()
    start = lines.index("## Global options")
    end = lines.index("## Commands")
    assert start < end
    rows = _rows(lines[start:end])
    root_params = _visible(_live_root())
    assert root_params, "the root has options of its own"
    assert [_row_names(row) for row in rows] == [
        {*param.opts, *param.secondary_opts} for param in root_params
    ]


def test_the_commands_index_links_every_top_level_command(cli_page: str) -> None:
    lines = cli_page.splitlines()
    start = lines.index("## Commands")
    stop = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("#"))
    linked = re.findall(r"\[`(soup [^`]+)`\]\(#([^)]+)\)", "\n".join(lines[start:stop]))
    root = _live_root()
    top = sorted(name for name, cmd in root.commands.items() if not cmd.hidden)
    assert [title for title, _ in linked] == [f"soup {name}" for name in top]
    assert all(anchor == title.replace(" ", "-") for title, anchor in linked), linked[:3]


def _declared_argument_help(cmd: click.Command) -> dict[str, str]:
    """Argument help as declared to Typer: Click 8.5 drops it from the Click object."""
    from typer.models import ArgumentInfo
    from typer.utils import get_params_from_function

    if cmd.callback is None:
        return {}
    return {
        name: meta.default.help
        for name, meta in get_params_from_function(cmd.callback).items()
        if isinstance(meta.default, ArgumentInfo) and meta.default.help
    }


def _help_strings(root: click.Group) -> list[tuple[str, str]]:
    """``(where, text)`` for every command help and parameter help of the live CLI."""
    found: list[tuple[str, str]] = []
    for title, cmd in {"soup": root, **_command_tree(root)}.items():
        found.append((title, cmd.help or ""))
        declared = _declared_argument_help(cmd)
        for param in cmd.params:
            text = getattr(param, "help", None) or declared.get(param.name, "")
            found.append((f"{title} {param.name}", text))
    return found


def test_every_command_a_help_string_or_a_description_names_is_registered() -> None:
    """``soup eval against`` told its reader to take the run id "from `soup runs list`",
    a subcommand that does not exist (``soup runs`` lists them). The docs guard (#1465)
    reads the committed pages, whose wording may trail the code, so every `` `soup ...` ``
    span is resolved here at its source: command help, parameter help, field description.
    """
    from soup_cli.config.schema import SoupConfig
    from tests.test_issue1465_documented_invocations_exist import check_invocation

    root = _live_root()
    texts = _help_strings(root) + [
        (path, info.description or "") for path, info, _ in _schema_fields(SoupConfig)
    ]
    spans = [
        (where, span) for where, text in texts for span in re.findall(r"`(soup [^`\n]+)`", text)
    ]
    assert len(spans) >= 20, len(spans)
    assert any(where.count(" ") >= 2 for where, _ in spans), "parameter help is read too"

    problems = [
        (where, span, reason)
        for where, span in spans
        for reason in [check_invocation(span, root)]
        if reason is not None
    ]
    assert problems == []
    assert check_invocation("soup runs list", root) is not None, "the control"


class _Colour(str, enum.Enum):
    red = "red"
    blue = "blue"


def _toy_root() -> click.Command:
    app = typer.Typer(rich_markup_mode="rich")
    oven = typer.Typer(help="Oven tools.")

    @app.callback()
    def _main(verbose: bool = typer.Option(False, "--verbose", "-V", help="Say more.")) -> None:
        """A toy app."""

    @app.command()
    def bake(
        recipe: str = typer.Argument(..., help="Recipe [bold]name[/bold]."),
        extras: Optional[List[str]] = typer.Argument(None, help="Extra toppings."),
        heat: int = typer.Option(3, "--heat", "-t", min=1, max=9, help="Heat | level."),
        ratio: float = typer.Option(0.5, "--ratio", min=0.0),
        colour: _Colour = typer.Option(_Colour.red, "--colour", help="Icing."),
        dry_run: bool = typer.Option(True, "--dry-run/--wet-run", help="Do not light it."),
        token: Optional[str] = typer.Option(
            None, "--token", envvar="BAKE_TOKEN", help='Needs "soup-cli\\[bake]" and [doc-N].'
        ),
        tag: Optional[List[str]] = typer.Option(None, "--tag", help="[dim]Repeat me.[/]"),
        secret: bool = typer.Option(False, "--secret", hidden=True, help="Not shown."),
        workdir: str = typer.Option("C:\\Users\\someone", "--workdir", help="Where."),
        stamp: str = typer.Option("", "--stamp", help="Empty by default."),
        when: str = typer.Option(default_factory=lambda: "now", help="Computed."),
    ) -> None:
        """Bake a [green]thing[/green]
        on two lines (v0.1.0 Part C).

        soup bake --nope is on a later paragraph and must not reach the page.
        """

    @app.command(hidden=True)
    def backdoor() -> None:
        """Hidden."""

    @oven.command("clean")
    def clean() -> None:
        """Scrub it."""

    app.add_typer(oven, name="oven")
    return typer.main.get_command(app)


def test_a_toy_app_renders_every_kind_of_parameter() -> None:
    page = gen.render_cli_reference(_toy_root())
    sections = _cli_sections(page)
    assert sorted(sections) == ["soup bake", "soup oven", "soup oven clean"]
    assert "backdoor" not in page and "--secret" not in page

    bake = sections["soup bake"]
    assert "Bake a thing on two lines (v0.1.0)." in bake
    assert "Usage: `soup bake [OPTIONS] RECIPE [EXTRAS]...`" in bake
    assert "--nope" not in page
    rows = {row[0]: row[1:] for row in _rows(bake)}
    assert rows["`RECIPE`"] == ["text", "", "yes", "Recipe name."]
    assert rows["`EXTRAS...`"] == ["text, any number", "", "", "Extra toppings."]
    assert rows["`--heat`, `-t`"] == ["integer, 1 to 9", "`3`", "", "Heat \\| level."]
    assert rows["`--ratio`"][:2] == ["float, at least 0.0", "`0.5`"]
    assert rows["`--colour`"] == ["one of `red`, `blue`", "`red`", "", "Icing."]
    assert rows["`--dry-run` / `--wet-run`"] == ["flag", "on", "", "Do not light it."]
    assert rows["`--token`"] == [
        "text",
        "",
        "",
        'Needs "soup-cli\\[bake]" and \\[doc-N]. Env: `BAKE_TOKEN`.',
    ]
    assert rows["`--tag`"] == ["text, repeatable", "", "", "Repeat me."]
    assert rows["`--workdir`"][1] == "*(set at runtime)*"
    assert rows["`--stamp`"][1] == '`""`'
    assert rows["`--when`"][1] == "*(set at runtime)*"
    assert "someone" not in page

    oven = sections["soup oven"]
    assert oven[oven.index("Subcommands:") + 2] == "- [`clean`](#soup-oven-clean)", oven
    assert "Takes no options or arguments." in sections["soup oven clean"]
    assert "| `--verbose`, `-V` | flag | off |  | Say more. |" in page


def test_a_plain_click_tree_renders_the_remaining_parameter_shapes() -> None:
    """Shapes the Typer toy does not produce; also proves nothing here needs Typer."""
    passthrough = {"allow_extra_args": True, "ignore_unknown_options": True}
    tool = click.Command(
        "tool",
        help="- not a bullet",
        params=[
            click.Option(["-v", "--loud"], count=True, help="More output"),
            click.Option(["--pair"], nargs=2, type=int),
            click.Option(["--src"], type=click.Path(), envvar=["SRC", "SOURCE"], help="From"),
            click.Option(["--share"], type=click.FloatRange(0, 1, min_open=True), default=0.5),
            click.Option(["--strict"], type=bool, default=None),
            click.Option(["--mode"], type=click.Choice(["a", "b"]), default="a", required=True),
            click.Option(["--keep"], multiple=True, default=("x", "y")),
        ],
    )
    group = click.Group(
        "runs",
        help="1. not a list",
        invoke_without_command=True,
        params=[click.Option(["--limit"], type=int, default=10)],
        commands={
            "show": click.Command("show", help="# not a heading"),
            "more": click.Group(
                "more", commands={"deep": click.Command("deep", help="Three names down.")}
            ),
        },
    )
    root = click.Group(
        "soup",
        commands={
            "tool": tool,
            "runs": group,
            "raw": click.Command("raw", help="> not a quote", context_settings=passthrough),
            "plain": click.Command("plain", help="--flag stays as typed"),
        },
    )

    page = gen.render_cli_reference(root)
    lines = page.splitlines()
    sections = _cli_sections(page)
    assert list(sections) == [
        "soup plain",
        "soup raw",
        "soup runs",
        "soup runs more",
        "soup runs more deep",
        "soup runs show",
        "soup tool",
    ]
    assert "#### `soup runs more deep`" in lines, "one heading level per name, three deep"
    for escaped in ("\\- not a bullet", "1\\. not a list", "\\# not a heading", "\\> not a quote"):
        assert escaped in lines, escaped
    assert "--flag stays as typed" in lines
    assert "Usage: `soup raw [ARGS]...`" in lines
    assert "Usage: `soup runs [OPTIONS] [<command>] [ARGS]...`" in lines
    assert "Usage: `soup <command> [ARGS]...`" in lines, "a root without options of its own"
    assert "### `soup runs show`" in lines

    rows = {row[0]: row[1:] for row in _rows(sections["soup tool"])}
    assert rows["`-v`, `--loud`"] == ["counter", "`0`", "", "More output"]
    assert rows["`--pair`"][0] == "integer, 2 values"
    assert rows["`--src`"] == ["path", "", "", "From. Env: `SRC`, `SOURCE`."]
    assert rows["`--share`"][:2] == ["float, above 0, at most 1", "`0.5`"]
    assert rows["`--strict`"][:2] == ["boolean", ""]
    assert rows["`--mode`"][:3] == ["one of `a`, `b`", "`a`", "yes"]
    assert rows["`--keep`"][:2] == ["text, repeatable", '`["x", "y"]`']


def test_argument_help_reads_the_same_whether_or_not_click_kept_it() -> None:
    """Click 8.5 resets the ``help`` Typer 0.20 sets on an argument; older Click keeps
    it. The page must not change with the Click that happens to be installed."""
    declared = {"recipe": "Recipe [bold]name[/bold].", "extras": "Extra toppings."}
    kept, reset = _toy_root(), _toy_root()
    for root, value in ((kept, declared), (reset, {})):
        for param in root.commands["bake"].params:
            if is_argument(param):
                param.help = value.get(param.name)

    assert gen.render_cli_reference(kept) == gen.render_cli_reference(reset)
    assert "| `RECIPE` | text |  | yes | Recipe name. |" in gen.render_cli_reference(reset)


def test_typer_completion_options_do_not_depend_on_shell_detection(
    monkeypatch: pytest.MonkeyPatch, cli_page: str
) -> None:
    """Without ``shellingham`` (or with detection disabled) Typer swaps the two
    completion flags for options that take a shell name, with other help text."""
    monkeypatch.setenv("_TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION", "1")
    undetected = _live_root()
    install = next(p for p in undetected.params if p.name == "install_completion")
    assert not install.is_flag, "the control: Typer really built the other variant"

    assert gen.render_cli_reference(undetected) == cli_page
    assert "| `--install-completion` | flag |" in cli_page


def test_rendering_ignores_the_terminal(
    monkeypatch: pytest.MonkeyPatch, pages: dict[str, str]
) -> None:
    """Nothing is taken from ``--help`` output, so width and colour cannot leak in."""
    monkeypatch.setenv("COLUMNS", "40")
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    assert gen.render_pages() == pages


# --------------------------------------------------------------------------
# 1c. --check-structure, on a toy schema and a toy CLI
# --------------------------------------------------------------------------

_RUN_THIS = "Run `python scripts/gen_reference_docs.py` and commit the result."


@pytest.fixture
def toy_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch repo whose two pages are the toy model's and the toy app's, with the
    generator pointed at both. Each structure test then changes exactly one thing."""
    _mirror(
        tmp_path,
        {
            gen.CONFIG_PAGE: gen.render_config_reference(_Toy),
            gen.CLI_PAGE: gen.render_cli_reference(_toy_root()),
        },
    )
    monkeypatch.setattr(gen, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(gen, "_load_config_model", lambda: _Toy)
    monkeypatch.setattr(gen, "_load_cli_root", _toy_root)
    return tmp_path


def _toy_with(**fields: Any) -> type[BaseModel]:
    """The toy model with fields added, or redeclared in place."""
    return create_model("ToyVariant", __base__=_Toy, **fields)


class _DeeperInner(BaseModel):
    depth: int = Field(default=3, ge=1, le=8, description="How deep.")


def _param(root: click.Command, command: str, name: str) -> click.Parameter:
    target = root.commands[command] if command else root
    return next(param for param in target.params if param.name == name)


def _toy_root_after(*mutations: Any) -> click.Command:
    root = _toy_root()
    for mutate in mutations:
        mutate(root)
    return root


_SPEED = "Speed."
_CONFIG_CHANGES: dict[str, tuple[Any, str]] = {
    "default": (
        lambda: _toy_with(
            mode=(Literal["fast", "slow"], Field(default="slow", description=_SPEED))
        ),
        'mode: committed page: default `"fast"` | code now: default `"slow"`',
    ),
    "default-removed": (
        lambda: _toy_with(mode=(Literal["fast", "slow"], Field(description=_SPEED))),
        'mode: committed page: default `"fast"` | code now: default **required**',
    ),
    "type": (
        lambda: _toy_with(code=(Optional[str], Field(default="x", pattern=r"^[a-z|]+$"))),
        "code: committed page: type `str` | code now: type `str | null`",
    ),
    "bound": (
        lambda: _toy_with(
            tags=(List[str], Field(default_factory=lambda: ["a"], min_length=1, max_length=8))
        ),
        "tags: committed page: constraints min length 1; max length 4"
        " | code now: constraints min length 1; max length 8",
    ),
    "choice": (
        lambda: _toy_with(
            mode=(Literal["fast", "slow", "turbo"], Field(default="fast", description=_SPEED))
        ),
        'mode: committed page: constraints one of `"fast"`, `"slow"`'
        ' | code now: constraints one of `"fast"`, `"slow"`, `"turbo"`',
    ),
    "nested-default": (
        lambda: _toy_with(inner=(_DeeperInner, Field(default_factory=_DeeperInner))),
        "inner.depth: committed page: default `2` | code now: default `3`",
    ),
    "two-columns-one-line": (
        lambda: _toy_with(mode=(Optional[Literal["fast", "slow"]], Field(default=None))),
        'mode: committed page: type `str`, default `"fast"`'
        " | code now: type `str | null`, default `null`",
    ),
    "new-key": (
        lambda: _toy_with(fresh_knob=(int, Field(default=7, ge=1, description="New."))),
        "fresh_knob: committed page: (not listed)"
        " | code now: type `int`, default `7`, constraints `>= 1`",
    ),
}


@pytest.mark.parametrize("case", sorted(_CONFIG_CHANGES))
def test_a_changed_config_key_fails_structure_and_is_named(
    toy_tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    variant, expected = _CONFIG_CHANGES[case]
    monkeypatch.setattr(gen, "_load_config_model", variant)

    assert gen.main(["--check-structure"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/config-reference.md no longer matches the schema (1 difference):" in out, out
    assert expected in out, out
    assert "docs/cli-reference.md" not in out, out
    assert out.endswith(_RUN_THIS), out
    assert gen.main(["--check"]) == 1, "what structure refuses, the exact check refuses too"


def test_a_key_that_left_the_schema_fails_structure_and_is_named(
    toy_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with_extra = gen.render_config_reference(_toy_with(fresh_knob=(int, Field(default=7))))
    (toy_tree / gen.CONFIG_PAGE).write_bytes(with_extra.encode("utf-8"))

    assert gen.main(["--check-structure"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/config-reference.md no longer matches the schema (1 difference):" in out, out
    assert (
        "fresh_knob: committed page: type `int`, default `7`, constraints (empty)"
        " | code now: (no such field)"
    ) in out, out
    assert out.endswith(_RUN_THIS), out


def test_rewording_a_description_passes_structure_and_fails_exact(
    toy_tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reworded = _toy_with(
        mode=(Literal["fast", "slow"], Field(default="fast", description="How fast the run goes.")),
        inner=(_Inner, Field(default_factory=_Inner, description="A nested block, reworded.")),
    )
    monkeypatch.setattr(gen, "_load_config_model", lambda: reworded)

    assert gen.main(["--check-structure"]) == 0
    assert "state what the code does" in _plain(capsys.readouterr().out)
    assert gen.main(["--check"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/config-reference.md is stale" in out, out
    assert "docs/cli-reference.md" not in out, out


_FRESH_FLAG = click.Option(["--fresh-flag"], is_flag=True, default=False, help="New.")
_CLI_CHANGES: dict[str, tuple[Any, str]] = {
    "default": (
        lambda root: setattr(_param(root, "bake", "heat"), "default", 5),
        "soup bake --heat, -t: committed page: default `3` | code now: default `5`",
    ),
    "required": (
        lambda root: setattr(_param(root, "bake", "tag"), "required", True),
        "soup bake --tag: committed page: required (empty) | code now: required yes",
    ),
    "type": (
        lambda root: setattr(_param(root, "bake", "heat"), "type", click.IntRange(1, 12)),
        "soup bake --heat, -t: committed page: type integer, 1 to 9"
        " | code now: type integer, 1 to 12",
    ),
    "new-option": (
        lambda root: root.commands["bake"].params.append(_FRESH_FLAG),
        "soup bake --fresh-flag: committed page: (not listed)"
        " | code now: type flag, default off, required (empty)",
    ),
    "new-argument": (
        lambda root: root.commands["oven"].commands["clean"].params.append(
            click.Argument(["target"])
        ),
        "soup oven clean TARGET: committed page: (not listed)"
        " | code now: type text, default (empty), required yes",
    ),
    "new-command": (
        lambda root: root.commands.__setitem__("fresh", click.Command("fresh", help="New.")),
        "soup fresh: committed page: (no section) | code now: a command",
    ),
    "global-option": (
        lambda root: setattr(_param(root, "", "verbose"), "default", True),
        "soup --verbose, -V: committed page: default off | code now: default on",
    ),
}


@pytest.mark.parametrize("case", sorted(_CLI_CHANGES))
def test_a_changed_command_or_parameter_fails_structure_and_is_named(
    toy_tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    mutate, expected = _CLI_CHANGES[case]
    monkeypatch.setattr(gen, "_load_cli_root", lambda: _toy_root_after(mutate))

    assert gen.main(["--check-structure"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/cli-reference.md no longer matches the CLI (1 difference):" in out, out
    assert expected in out, out
    assert "docs/config-reference.md" not in out, out
    assert out.endswith(_RUN_THIS), out
    assert gen.main(["--check"]) == 1, "what structure refuses, the exact check refuses too"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "new-option",
            "soup bake --fresh-flag: committed page: type flag, default off, required (empty)"
            " | code now: (no such parameter)",
        ),
        ("new-command", "soup fresh: committed page: a section | code now: (no such command)"),
    ],
)
def test_what_left_the_cli_fails_structure_and_is_named(
    toy_tree: Path, capsys: pytest.CaptureFixture[str], case: str, expected: str
) -> None:
    with_extra = gen.render_cli_reference(_toy_root_after(_CLI_CHANGES[case][0]))
    (toy_tree / gen.CLI_PAGE).write_bytes(with_extra.encode("utf-8"))

    assert gen.main(["--check-structure"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/cli-reference.md no longer matches the CLI (1 difference):" in out, out
    assert expected in out, out
    assert out.endswith(_RUN_THIS), out


def test_rewording_help_passes_structure_and_fails_exact(
    toy_tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def reword(root: click.Command) -> None:
        root.commands["bake"].help = "Bake something, said differently."
        _param(root, "bake", "heat").help = "How hot, reworded."
        _param(root, "", "verbose").help = "Say a lot more."

    monkeypatch.setattr(gen, "_load_cli_root", lambda: _toy_root_after(reword))

    assert gen.main(["--check-structure"]) == 0
    assert "state what the code does" in _plain(capsys.readouterr().out)
    assert gen.main(["--check"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/cli-reference.md is stale" in out, out
    assert "docs/config-reference.md" not in out, out


def test_structure_names_ten_differences_per_page_and_counts_the_rest(
    toy_tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    knobs = {f"knob_{index:02d}": (int, Field(default=index)) for index in range(13)}
    monkeypatch.setattr(gen, "_load_config_model", lambda: _toy_with(**knobs))
    monkeypatch.setattr(gen, "_load_cli_root", lambda: _toy_root_after(_CLI_CHANGES["type"][0]))

    assert gen.main(["--check-structure"]) == 1
    report = capsys.readouterr().out
    out = _plain(report)
    assert "docs/config-reference.md no longer matches the schema (13 differences):" in out, out
    assert out.count(": committed page: (not listed)") == 10, out
    assert "knob_09: committed page: (not listed)" in out and "knob_10" not in out, out
    # one offending key per line, indented under its page
    per_key = [line for line in report.splitlines() if line.startswith("  knob_")]
    assert len(per_key) == len(set(per_key)) == gen._MAX_SHOWN, report
    assert all(line.count("committed page:") == line.count("| code now:") == 1 for line in per_key)
    # the count of what was not shown closes the first page's block
    assert "... and 3 more docs/cli-reference.md no longer matches the CLI (1 difference):" in out
    assert out.count(_RUN_THIS) == 1 and out.endswith(_RUN_THIS), out


def test_split_row_splits_where_github_does() -> None:
    r"""Each row below was rendered through GitHub's Markdown API on 2026-10-12. A pipe
    right after a backslash never ends a cell, however many backslashes come before it,
    and GitHub then drops exactly one of them: `x\\|y` in code shows as `x\|y`."""
    assert gen.split_row("| a | b |") == ["a", "b"]
    assert gen.split_row("|---|---|") == ["---", "---"]
    assert gen.split_row(r"| `x\|y` | p\|q |  | z |") == [r"`x\|y`", r"p\|q", "", "z"]
    assert gen.split_row(r"| `x\\|y` | p\\|q |") == [r"`x\\|y`", r"p\\|q"]
    assert gen.split_row(r"| `x\\\|y` | p\\\|q |") == [r"`x\\\|y`", r"p\\\|q"]
    assert gen.split_row(r"| `^[a-z\\|]+$` | end\\ |") == [r"`^[a-z\\|]+$`", r"end\\"]
    assert gen.split_row(r"| `tail\` | z |") == [r"`tail\`", "z"]


class _Patterns(BaseModel):
    alt: str = Field(default="a", pattern=r"^a\|b$", description=r"A literal \| bar, either side.")
    after: int = Field(default=1, ge=1, description="The row below it.")


def test_a_backslash_before_a_pipe_does_not_take_the_row_out_of_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    r"""A regex such as ``^a\|b$`` is written ``^a\\|b$`` in the cell, which is what makes
    GitHub show it as it is. Split by backslash parity, that row had six cells, did not
    look like a row, and silently left the comparison."""
    page = gen.render_config_reference(_Patterns)
    row = next(line for line in page.splitlines() if line.startswith("| `alt` |"))
    assert row == (
        r"| `alt` | `str` | `" + '"a"' + "` "
        r"| pattern `^a\\|b$` | A literal \\\| bar, either side. |"
    )
    parsed = gen.parse_config_page(page)
    assert list(parsed) == ["alt", "after"]
    assert parsed["alt"]["constraints"] == r"pattern `^a\\|b$`"

    _mirror(tmp_path, {gen.CONFIG_PAGE: page, gen.CLI_PAGE: gen.render_cli_reference(_toy_root())})
    monkeypatch.setattr(gen, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(gen, "_load_cli_root", _toy_root)
    monkeypatch.setattr(gen, "_load_config_model", lambda: _Patterns)
    assert gen.main(["--check-structure"]) == 0, capsys.readouterr().out
    capsys.readouterr()

    changed = create_model(
        "Changed", __base__=_Patterns, alt=(str, Field(default="b", pattern=r"^a\|b$"))
    )
    monkeypatch.setattr(gen, "_load_config_model", lambda: changed)
    assert gen.main(["--check-structure"]) == 1
    out = _plain(capsys.readouterr().out)
    assert 'alt: committed page: default `"a"` | code now: default `"b"`' in out, out


@pytest.mark.parametrize(
    ("good", "bad"),
    [
        ("| `mode` | `str` |", "| `mode` | `str` | one cell too many |"),
        ("| `mode` | `str` |", "| `mode` | `str | int` |"),  # a pipe written without its backslash
        ("| `mode` | `str` |", "| mode | `str` |"),  # the key is not a code span
    ],
)
def test_a_table_line_that_cannot_be_read_back_fails_the_check(
    toy_tree: Path, capsys: pytest.CaptureFixture[str], good: str, bad: str
) -> None:
    """Never a skip: a row the parser cannot read would drop its key from the check."""
    page = toy_tree / gen.CONFIG_PAGE
    text = page.read_bytes().decode("utf-8")
    assert text.count(good) == 1
    broken = text.replace(good, bad)
    page.write_bytes(broken.encode("utf-8"))

    with pytest.raises(gen.PageFormatError, match="is not a 5-column row of this page"):
        gen.parse_config_page(broken)
    assert gen.main(["--check-structure"]) == 1
    out = _plain(capsys.readouterr().out)
    assert "docs/config-reference.md no longer matches the schema (1 difference):" in out, out
    assert "the committed page cannot be read back: line " in out, out
    assert "mode" in out and out.endswith(_RUN_THIS), out


def test_the_parsers_read_back_what_the_renderers_stated() -> None:
    config = gen.parse_config_page(gen.render_config_reference(_Toy))
    assert list(config) == [
        "name",
        "mode",
        "ratio",
        "tags",
        "stamp",
        "workdir",
        "code",
        "inner",
        "maybe",
        "yamlName",
        "inner.depth",
        "maybe.depth",
    ]
    assert config["code"] == {
        "type": "`str`",
        "default": '`"x"`',
        "constraints": "pattern `^[a-z\\|]+$`",
    }
    assert config["ratio"]["type"] == "`float \\| null`"
    assert config["name"] == {"type": "`str`", "default": "**required**", "constraints": ""}

    cli = gen.parse_cli_page(gen.render_cli_reference(_toy_root()))
    assert list(cli) == ["soup", "soup bake", "soup oven", "soup oven clean"]
    assert list(cli["soup"]) == ["--verbose, -V", "--install-completion", "--show-completion"]
    assert cli["soup"]["--verbose, -V"] == {"type": "flag", "default": "off", "required": ""}
    assert list(cli["soup bake"])[:3] == ["RECIPE", "EXTRAS...", "--heat, -t"]
    assert cli["soup bake"]["RECIPE"] == {"type": "text", "default": "", "required": "yes"}
    assert cli["soup bake"]["--dry-run / --wet-run"]["default"] == "on"
    assert cli["soup oven"] == {} and cli["soup oven clean"] == {}


def test_the_structure_parsers_read_every_row_of_the_real_pages(
    config_page: str, cli_page: str
) -> None:
    """A parser that silently read nothing would let the structure check pass for ever."""
    from soup_cli.config.schema import SoupConfig

    paths, _ = _schema_paths(SoupConfig)
    assert sorted(gen.parse_config_page(config_page)) == sorted(paths)

    root = _live_root()
    tree = _command_tree(root)
    parsed = gen.parse_cli_page(cli_page)
    assert set(parsed) == {"soup", *tree}
    assert len(parsed["soup"]) == len(_visible(root))
    for title, cmd in tree.items():
        assert len(parsed[title]) == len(_visible(cmd)), title


# --------------------------------------------------------------------------
# 1d. one line per item: concurrent changes merge as text and stay right
# --------------------------------------------------------------------------


def _insertions(base: str, changed: str) -> dict[int, list[str]]:
    """``{line index in base: lines inserted there}`` for a change that only adds lines;
    fails when the change removes or rewrites a line that was already on the page."""
    old, new = base.split("\n"), changed.split("\n")
    found: dict[int, list[str]] = {}
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, old_start, old_stop, new_start, new_stop in matcher.get_opcodes():
        if tag == "equal":
            continue
        assert tag == "insert", (tag, old[old_start:old_stop], new[new_start:new_stop])
        found[old_start] = new[new_start:new_stop]
    return found


def _merged(base: str, *changes: str) -> str:
    """The three-way text merge of changes that insert at different places."""
    lines = base.split("\n")
    inserted: dict[int, list[str]] = {}
    for change in changes:
        for index, added in _insertions(base, change).items():
            assert index not in inserted, f"two changes insert at line {index + 1}"
            inserted[index] = added
    for index in sorted(inserted, reverse=True):
        lines[index:index] = inserted[index]
    return "\n".join(lines)


def test_two_changes_that_each_add_a_key_merge_as_text_and_stay_right() -> None:
    def page(*names: str) -> str:
        fields = {name: (int, Field(default=len(name), description=name)) for name in names}
        return gen.render_config_reference(create_model("Merged", **fields))

    base = page("alpha", "beta", "gamma")
    first = page("alpha", "one", "beta", "gamma")
    second = page("alpha", "beta", "two", "gamma")
    assert list(_insertions(base, first).values()) == [["| `one` | `int` | `3` |  | one |"]]
    assert _merged(base, first, second) == page("alpha", "one", "beta", "two", "gamma")


def test_changes_that_add_an_option_and_commands_merge_as_text_and_stay_right() -> None:
    def add_option(root: click.Command) -> None:
        root.commands["bake"].params.insert(3, click.Option(["--extra"], help="More."))

    def add_subcommand(root: click.Command) -> None:
        root.commands["oven"].commands["wipe"] = click.Command("wipe", help="Wipe it.")

    def add_command(root: click.Command) -> None:
        root.commands["apple"] = click.Command("apple", help="Core it.")

    def page(*mutations: Any) -> str:
        return gen.render_cli_reference(_toy_root_after(*mutations))

    base = page()
    assert list(_insertions(base, page(add_option)).values()) == [
        ["| `--extra` | text |  |  | More. |"]
    ]
    assert "- [`wipe`](#soup-oven-wipe)" in sum(
        _insertions(base, page(add_subcommand)).values(), []
    )
    merged = _merged(base, page(add_option), page(add_subcommand), page(add_command))
    assert merged == page(add_option, add_subcommand, add_command)


# --------------------------------------------------------------------------
# 2c. what the pages must never carry
# --------------------------------------------------------------------------


def _leaked_paths(page: str, directories: typing.Iterable[PurePath]) -> list[str]:
    """The directories that appear in ``page`` as a whole path.

    A plain substring test is wrong here: a checkout at ``/repo`` or ``/data`` is also
    ordinary page text (``./data/train.jsonl``). So a directory fewer than two levels
    deep is not looked for at all, and a deeper one only counts where a path starts, not
    as the tail of another path or the head of a longer name."""
    leaked = set()
    for directory in directories:
        if len(directory.parts) < 3:  # the anchor and two levels
            continue
        for spelled in (str(directory), directory.as_posix()):
            if re.search(r"(?<![\w./\\~-])" + re.escape(spelled) + r"(?![\w-])", page):
                leaked.add(spelled)
    return sorted(leaked)


@pytest.mark.parametrize("rel", ["docs/config-reference.md", "docs/cli-reference.md"])
def test_a_page_carries_nothing_from_this_machine(pages: dict[str, str], rel: str) -> None:
    page = pages[rel]
    assert _leaked_paths(page, (Path.home(), _ROOT, Path.cwd())) == [], rel
    for needle in ("\r", "\t", "\x1b"):
        assert needle not in page, (rel, needle)
    assert not re.search(r"\b[A-Za-z]:\\", page), "a Windows drive path"
    assert [line for line in page.splitlines() if line != line.rstrip()] == []
    assert page.endswith("\n") and not page.endswith("\n\n")


def test_the_leak_check_reads_whole_paths_only() -> None:
    text = "default `./data/train.jsonl`; /home/someone/.soup; C:\\Users\\someone\\x; /repository"
    shallow = [PurePosixPath("/data"), PurePosixPath("/repo"), PurePosixPath("/")]
    assert _leaked_paths(text, shallow) == []
    assert _leaked_paths("mounted at /data, cloned to /repo", shallow) == []
    assert _leaked_paths(text, [PurePosixPath("/home/someone")]) == ["/home/someone"]
    assert _leaked_paths(text, [PureWindowsPath("C:\\Users\\someone")]) == ["C:\\Users\\someone"]
    assert _leaked_paths("see ./home/someone/x", [PurePosixPath("/home/someone")]) == []
    assert _leaked_paths("see /home/someone-else/x", [PurePosixPath("/home/someone")]) == []
    assert _leaked_paths("/srv/ci/soup", [PurePosixPath("/srv/ci/soup")]) == ["/srv/ci/soup"]


@pytest.mark.parametrize("rel", ["docs/config-reference.md", "docs/cli-reference.md"])
def test_a_page_says_it_is_generated_and_how_to_regenerate_it(
    pages: dict[str, str], rel: str
) -> None:
    lines = pages[rel].split("\n## ", 1)[0].splitlines()
    head = " ".join(" ".join(line.lstrip("> ") for line in lines).split())
    assert "Generated file" in head and "do not edit by hand" in head, head
    assert "`python scripts/gen_reference_docs.py` and commit the result" in head, head
    assert "refreshed when a release is cut" in head, head


@pytest.mark.parametrize("rel", ["docs/config-reference.md", "docs/cli-reference.md"])
def test_a_page_header_carries_no_number_at_all(pages: dict[str, str], rel: str) -> None:
    """No total, count, date or version above the first table: a line that sums over
    the page changes in every pull request that adds a row, so two of them conflict,
    or worse merge cleanly into a number that is wrong."""
    head = pages[rel].split("\n## ", 1)[0]
    assert re.findall(r"\d+", head) == [], rel


@pytest.mark.parametrize("rel", ["docs/config-reference.md", "docs/cli-reference.md"])
def test_a_page_carries_no_internal_release_phase_label(pages: dict[str, str], rel: str) -> None:
    """"vX.Y.Z Part B" is planning shorthand; public docs carry the version only."""
    assert _INTERNAL_PHASE.findall(pages[rel]) == []


def _slug(heading: str) -> str:
    """GitHub's heading anchor, written here independently of the generator's."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading.replace("`", "")).strip().lower()
    return "".join("-" if char == " " else char for char in text if char.isalnum() or char in "_- ")


def _anchors(markdown: str) -> set[str]:
    seen: Counter[str] = Counter()
    anchors = set()
    for line in markdown.splitlines():
        match = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if match:
            base = _slug(match.group(1))
            anchors.add(base if not seen[base] else f"{base}-{seen[base]}")
            seen[base] += 1
    return anchors


def _internal_targets(page: str) -> list[str]:
    """The link targets this page is answerable for: an in-page ``#anchor`` or a relative
    ``*.md`` file. A URL with a scheme is somebody else's page, and a description is free
    to carry one (``See [the guide](https://example.com/lr)``)."""
    targets = re.findall(r"\]\(([^)\s]+)\)", page)
    return [
        target
        for target in targets
        if not re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target)
        and (target.startswith("#") or target.partition("#")[0].endswith(".md"))
    ]


def _broken_links(page: str) -> list[str]:
    own = _anchors(page)
    broken = []
    for target in _internal_targets(page):
        file_part, _, fragment = target.partition("#")
        if not file_part:
            known = own
        else:
            other = _ROOT / "docs" / file_part
            if not other.is_file():
                broken.append(target)
                continue
            known = _anchors(other.read_text(encoding="utf-8")) if fragment else set()
        if fragment and fragment not in known:
            broken.append(target)
    return broken


@pytest.mark.parametrize("rel", ["docs/config-reference.md", "docs/cli-reference.md"])
def test_every_link_in_a_page_resolves(pages: dict[str, str], rel: str) -> None:
    assert len(_internal_targets(pages[rel])) >= 8
    assert _broken_links(pages[rel]) == []


def test_an_external_link_in_a_description_is_not_a_broken_file() -> None:
    class Linked(BaseModel):
        rate: float = Field(default=0.1, description="See [the guide](https://example.com/lr).")
        mail: str = Field(default="", description="Ask [us](mailto:team@example.com) or (x)[y].")
        page: str = Field(default="", description="Raw: [md](https://example.com/guide.md#lr).")
        logo: str = Field(default="", description="A picture, ![logo](assets/none.png).")

    page = gen.render_config_reference(Linked)
    assert "https://example.com/lr" in page and "assets/none.png" in page
    assert "#top-level-keys" in _internal_targets(page)
    assert [target for target in _internal_targets(page) if ":" in target] == []
    assert _broken_links(page) == []
    # the control: an internal target that goes nowhere is still caught
    assert _broken_links(page + "\n[a](#nowhere) [b](missing-page.md) [c](README.md#nope)\n") == [
        "#nowhere",
        "missing-page.md",
        "README.md#nope",
    ]


def test_the_config_page_links_the_unknown_key_section(config_page: str) -> None:
    assert "refuses the load" in " ".join(config_page.split())
    assert "](backends-and-ops.md#unknown-config-keys)" in config_page


# --------------------------------------------------------------------------
# 1b. the determinism helpers, one by one
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (str, "str"),
        (bool, "bool"),
        (Optional[int], "int | null"),
        (Union[int, None], "int | null"),
        (Union[None, int], "int | null"),
        (int | None, "int | null"),
        (None | int, "int | null"),
        (Union[str, List[str]], "str | list[str]"),
        (str | list[str], "str | list[str]"),
        (List[str], "list[str]"),
        (list[str], "list[str]"),
        (list, "list"),
        (List, "list"),
        (Dict[str, int], "dict[str, int]"),
        (dict[str, int], "dict[str, int]"),
        (Optional[dict[str, float]], "dict[str, float] | null"),
        (Dict, "dict"),
        (dict, "dict"),
        (Union[str, Dict, None], "str | dict | null"),
        (Optional[List[Dict[str, Union[str, float]]]], "list[dict[str, str | float]] | null"),
        (Literal["a", "b"], "str"),
        (Optional[Literal["a", "b"]], "str | null"),
        (Literal[1, 2], "int"),
        (Literal["a", 1], "str | int"),
        (Union[int, Literal["auto"]], 'int | "auto"'),
        (Union[bool, Literal["fp8", "quest"]], 'bool | "fp8" | "quest"'),
        (Union[Literal["auto"], List[str], None], '"auto" | list[str] | null'),
        (Annotated[int, Field(ge=1)], "int"),
        (Optional[Annotated[int, Field(ge=1)]], "int | null"),
        (_Inner, "section"),
        (Optional[_Inner], "section | null"),
        (List[_Inner], "list[section]"),
        (_Colour, "str"),
        (Optional[_Colour], "str | null"),
        (tuple[int, ...], "tuple[int]"),
        (frozenset[str], "set[str]"),
        (collections.abc.Sequence[str], "list[str]"),
        (typing.Mapping[str, int], "dict[str, int]"),
        (Path, "path"),
        (Any, "any"),
        # isinstance(type[int], type) is True on 3.10 and False from 3.11: the generic
        # must not fall through to the class branch on one of them only
        (type[int], "any"),
        (typing.Callable[[int], str], "any"),
        (type(None), "null"),
    ],
)
def test_types_are_spelled_one_way_on_every_python(annotation: Any, expected: str) -> None:
    assert gen.format_type(annotation) == expected


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (Literal["a", "b"], ["a", "b"]),
        (Optional[Literal["a", "b"]], ["a", "b"]),
        (Literal[1, True], [1, True]),
        (_Colour, ["red", "blue"]),
        (Union[int, Literal["auto"]], []),  # spelled inline in the type instead
        (Optional[int], []),
        (str, []),
    ],
)
def test_choices_are_listed_only_for_a_pure_choice(annotation: Any, expected: list) -> None:
    found = gen.literal_choices(annotation)
    assert found == expected and [type(v) for v in found] == [type(v) for v in expected]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "null"),
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (0.0002, "0.0002"),
        (1e-05, "1e-05"),
        (1.0, "1.0"),
        ("auto", '"auto"'),
        ("", '""'),
        ("./output", '"./output"'),
        ("caf\u00e9", '"caf\u00e9"'),
        (["info_rm"], '["info_rm"]'),
        (("a", 1), '["a", 1]'),
        ({"r": 8, "alpha": 16}, '{"r": 8, "alpha": 16}'),
        ({"b", "a"}, '["a", "b"]'),
        (frozenset({2, 1}), "[1, 2]"),
        (_Colour.blue, '"blue"'),
        (Path("out") / "run", '"out/run"'),
    ],
)
def test_literal_defaults_are_written_as_yaml_flow_values(value: Any, expected: str) -> None:
    assert gen.format_default(value) == expected


@pytest.fixture
def somewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generator believes its home is ``/home/someone`` and its checkout
    ``/srv/checkout/soup``, wherever the suite really runs."""
    monkeypatch.setattr(gen, "_machine_markers", lambda: ("/home/someone", "/srv/checkout/soup"))


@pytest.mark.parametrize(
    "value",
    [
        object(),
        lambda: 1,
        os.cpu_count,
        "/home/someone",
        "/home/someone/.soup/cache",
        "/srv/checkout/soup/output",
        "/SRV/CHECKOUT/SOUP/OUT",
        "\\srv\\checkout\\soup\\out",
        "sqlite:/home/someone/runs.db",
        "C:\\Users\\someone\\models",
        "d:/models",
        ["fine", object()],
        {"nested": {"deeper": object()}},
        {("tuple", "key"): 1},
        {1, "a"},
        PurePosixPath("/home/someone/models"),
    ],
)
def test_a_default_this_machine_computed_becomes_the_placeholder(
    somewhere: None, value: Any
) -> None:
    assert gen.format_default(value) == gen.RUNTIME_DEFAULT == "(set at runtime)"


@pytest.mark.parametrize(
    "value",
    [
        "ollama://model",
        "https://example.com/v1",
        "/v1/chat/completions",
        "module.path:function",
        "./home/someone/x",  # the tail of a relative path
        "data/srv/checkout/soup",
        "/home/someone-else/x",  # the head of a longer name
        "/home/someoneelse",
    ],
)
def test_a_constant_that_only_looks_like_a_path_stays_a_literal(
    somewhere: None, value: str
) -> None:
    assert gen.format_default(value) == f'"{value}"'


def test_a_shallow_working_directory_marks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """In a container the checkout is often ``/repo``, ``/work`` or ``/data``. As a marker
    that would turn the real default ``./data/train.jsonl`` into a placeholder on that
    machine only, and its pages would stop matching everybody else's."""
    monkeypatch.setattr(os.path, "expanduser", lambda path: "/root")
    monkeypatch.setattr(os, "getcwd", lambda: "/data")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/tmp")
    assert gen._machine_markers() == ()
    assert gen.format_default("./data/train.jsonl") == '"./data/train.jsonl"'
    assert gen.format_default("/data/train.jsonl") == '"/data/train.jsonl"'

    monkeypatch.setattr(os.path, "expanduser", lambda path: "/home/someone")
    monkeypatch.setattr(os, "getcwd", lambda: "C:\\Work\\Soup\\")
    assert gen._machine_markers() == ("/home/someone", "c:/work/soup")
    assert gen.format_default("/home/someone/.cache") == gen.RUNTIME_DEFAULT


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a | b", "a \\| b"),
        ("line one\nline two\r\n  three", "line one line two three"),
        ("math | code | json_schema", "math \\| code \\| json_schema"),
        ("registry://<id> | 'previous'", "registry://&lt;id> \\| 'previous'"),
        ("`a|b` stays code, <c> does not", "`a\\|b` stays code, &lt;c> does not"),
        ("``x <y>`` is code too", "``x <y>`` is code too"),
        ("ratio * teacher + (1 - ratio) * x", "ratio \\* teacher + (1 - ratio) \\* x"),
        ("snake_case_name stays", "snake_case_name stays"),
        ("_private and __dunder__", "\\_private and \\_\\_dunder\\_\\_"),
        ("bounded in ~[0,1] (a probability)", "bounded in \\~\\[0,1] (a probability)"),
        ("a $x$ sum and R&D &lt;", "a \\$x\\$ sum and R&amp;D &amp;lt;"),
        ("a lone ` backtick and a \\ slash", "a lone \\` backtick and a \\\\ slash"),
        ("", ""),
    ],
)
def test_cell_text_cannot_break_the_table(text: str, expected: str) -> None:
    escaped = gen.escape_cell(text)
    assert escaped == expected
    assert "\n" not in escaped and len(_cells(f"| {escaped} |")) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[bold]Train[/bold] a model", "Train a model"),
        ("[dim]quiet[/]", "quiet"),
        ("[bold red on white]loud[/bold red on white]", "loud"),
        ("[link=https://example.com]docs[/link]", "docs"),
        ("[#ff8800]orange[/#ff8800]", "orange"),
        ('pip install "soup-cli\\[train]"', 'pip install "soup-cli[train]"'),
        ("a literal \\\\[bold]x", "a literal \\x"),
        ("cite as [doc-N] or [doc-id]", "cite as [doc-N] or [doc-id]"),
        ("pip install soup-cli[sigstore]", "pip install soup-cli[sigstore]"),
        ("range [0, 1] and [OPTIONS]", "range [0, 1] and [OPTIONS]"),
        ("[/path/to/file]", "[/path/to/file]"),
    ],
)
def test_rich_markup_is_stripped_and_bracketed_text_is_kept(text: str, expected: str) -> None:
    assert gen.strip_rich_markup(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("streaming. (v0.42.0 Part B)", "streaming. (v0.42.0)"),
        ("(v0.42.0 Part E; #759)", "(v0.42.0; #759)"),
        ("the v0.62.0 Part A RAFT recipe", "the v0.62.0 RAFT recipe"),
        ("v0.38.0 part g \u2014 storage", "v0.38.0 \u2014 storage"),
        ("Part of the v0.42.0 work", "Part of the v0.42.0 work"),
        ("spare Part B", "spare Part B"),
    ],
)
def test_the_internal_phase_label_is_dropped_and_the_version_kept(text: str, expected: str) -> None:
    assert gen.strip_internal_phase(text) == expected


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("`training.lora`", "traininglora"),
        ("`training.eval_gate`", "trainingeval_gate"),
        ("Top-level keys", "top-level-keys"),
        ("`soup data inspect`", "soup-data-inspect"),
        ("`soup eval gate-install`", "soup-eval-gate-install"),
        ("Telemetry (opt-in)", "telemetry-opt-in"),
    ],
)
def test_heading_anchors_follow_githubs_slug(heading: str, expected: str) -> None:
    assert gen.github_slug(heading) == expected == _slug(heading)


def test_the_slug_drops_what_github_drops() -> None:
    """Read off github.com on 2026-10-12: the heading ``Attaching energy + CO₂
    (`--energy`)`` in ``docs/adapters-and-governance.md`` has this anchor. The subscript
    is a number GitHub does not keep, which ``str.isalnum`` would."""
    heading = "Attaching energy + CO₂ (`--energy`)"
    assert gen.github_slug(heading) == "attaching-energy--co---energy"
    assert gen.github_slug("Café déjà vu — v0.71.20") == "café-déjà-vu--v07120"


def test_first_paragraph_only_is_the_command_summary() -> None:
    assert gen.summary_line("One line.\n\nMore.\n\nsoup x --y") == "One line."
    assert gen.summary_line("Wrapped\n    summary.\n  \n  Later.") == "Wrapped summary."
    assert gen.summary_line("Before the form feed.\fhidden") == "Before the form feed."
    assert gen.summary_line(None) == ""


# --------------------------------------------------------------------------
# 3. import-light, and runnable as a script
# --------------------------------------------------------------------------

_PROBE = (
    "import sys\n"
    "{prelude}"
    "from scripts import gen_reference_docs as gen\n"
    "pages = gen.render_pages()\n"
    "assert sorted(pages) == ['docs/cli-reference.md', 'docs/config-reference.md']\n"
    "assert all(len(text) > 20000 for text in pages.values())\n"
    "print(','.join(sorted(m for m in {heavy!r} if m in sys.modules)))\n"
)


def _heavy_after_rendering(prelude: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PROBE.format(prelude=prelude, heavy=_HEAVY)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_ROOT,
    )


def test_rendering_both_pages_does_not_import_torch() -> None:
    proc = _heavy_after_rendering()
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    leaked = [name for name in proc.stdout.strip().split(",") if name]
    assert leaked == [], f"rendering the reference pages imported {leaked}"


def test_the_torch_probe_would_catch_a_regression() -> None:
    proc = _heavy_after_rendering(prelude="import torch\n")
    if proc.returncode != 0:
        pytest.skip("torch not installed (light-core-only environment)")
    assert "torch" in proc.stdout.strip().split(","), (proc.stdout, proc.stderr)


def test_the_script_runs_from_any_directory(tmp_path: Path) -> None:
    """``python scripts/gen_reference_docs.py`` is what a contributor types: no
    ``scripts`` package on ``sys.path``, and the pages found from the script's path."""
    script = str(_ROOT / "scripts" / "gen_reference_docs.py")
    proc = subprocess.run(
        [sys.executable, script, "--check-structure"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "state what the code does" in _plain(proc.stdout), (proc.stdout, proc.stderr)
    assert list(tmp_path.iterdir()) == []
