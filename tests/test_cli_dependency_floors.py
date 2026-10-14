"""The declared Typer and Click floors are versions the CLI really starts on (#1802).

``pyproject.toml`` declared ``typer>=0.9.0`` while the source needed Typer 0.12.4 to
build its command tree at all (``RuntimeError: Type not yet supported: int | None``, on
every command, ``soup --help`` and ``soup version`` included) and 0.15.4 to be safe next
to any Click it can be installed with. Nothing noticed, because every CI cell resolves
the NEWEST Typer the range allows: the floor was declared and never run.

So the floor is not held here as a number to compare against. It is installed, and the
CLI is started on it -- the rule ``trainer/_trl_compat.py`` already follows for trl: a
bound read off source or release notes is a hypothesis, constructing the object settles
it. Three links, each held by something different:

1. *declared floor == pin.* ``.github/constraints/transformers-floor.txt`` pins ``typer``
   and ``click`` to exactly the floors ``[project] dependencies`` declares. Hermetic, so
   it fails on every cell: ``TestTheFloorJobInstallsTheDeclaredFloors``.
2. *pin == installed.* The ``transformers-floor`` job's "Assert installed versions match
   floor constraints" step refuses a cell where pip installed anything else;
   ``tests/test_transformers_floor_compat.py`` executes that step's script.
3. *installed starts the CLI.* That job runs this file, ``test_click_argument_help.py``
   and ``test_cli_startup_is_light.py``. ``TestTheCliStartsOnTheInstalledTyperAndClick``
   builds the command tree, renders the help page of every command and group, and runs
   ``soup version``. On every other cell the same tests run on the newest Typer and
   Click, the other end of the range.

Lowering the declaration alone fails link 1 by name. Lowering the declaration and the
pin together passes link 1 and turns the floor job red at link 3, with the error a user
would have met.

Measured 2026-10-13 on ``main`` (b43c82a07): the core dependencies only, one scratch
virtual environment per cell, Python 3.12.14::

    Typer    Click 8.0.0   8.1.8   8.2.1     8.4.2
    0.9.0    .             TREE    .         .
    0.12.3   .             TREE    .         .
    0.12.4   ok            ok      HELP      TREE
    0.15.3   ok            ok      HELP      HELP
    0.15.4   ok            ok      refused   refused
    0.16.0   ok            ok      ok        ok
    0.20.1   ok            ok      ok        ok

``ok``: the tree builds, all 254 help pages render, and ``soup --help``, five subcommand
``--help`` pages (argument and option help present) and ``soup version`` exit 0 as real
processes. ``TREE``: the command tree does not build, so no command runs (``Type not yet
supported: int | None`` below 0.12.4; ``Secondary flag is not valid for non-boolean
flag`` for 0.12.4 next to Click 8.4.2). ``HELP``: the tree builds and all 254 help pages
raise ``Parameter.make_metavar() missing 1 required positional argument: 'ctx'``.
``refused``: the pair cannot be installed, Typer 0.15.4 itself requires ``click<8.2``.
``.``: not run.

0.16.1, 0.17.5, 0.18.0, 0.19.2 and 0.20.0 are ``ok`` with Click 8.0.0 and with the
newest Click each accepts. On Python 3.10.21 and 3.11.16 the 0.15.4 row and the two ends
of the 0.20.1 row (Click 8.0.0 and 8.4.2) gave the same verdicts. Hence
``typer>=0.15.4``: 0.12.4 - 0.15.3 start the CLI only next to Click 8.1 or older and do
not ask for it, so a resolver pairs them with a Click they break on (8.4.2 for 0.15.3 on
all three Pythons). ``click>=8.0.0`` is true as declared.
"""

from __future__ import annotations

import importlib.metadata as md
import pathlib
import re
import shlex

import click
import pytest
import typer
import yaml
from click.testing import CliRunner as ClickRunner
from packaging.utils import canonicalize_name
from packaging.version import Version
from typer.testing import CliRunner

from soup_cli import __version__
from soup_cli.cli import app
from tests.test_click_argument_help import _core_requirements, _plain

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / ".github" / "constraints" / "transformers-floor.txt"
CI = ROOT / ".github" / "workflows" / "ci.yml"
FLOOR_JOB = "transformers-floor"
#: The core dependencies whose declared floor the floor job installs.
FLOORED = ("typer", "click")
#: A help page is a Rich table as wide as the terminal. In a narrow one the description
#: column shrinks to a few characters and words are cut mid-word, which no whitespace
#: collapse can undo, so the pages read for their text are rendered wide. Rich takes
#: the width from COLUMNS when the console is created, once per help page.
WIDE = {"COLUMNS": "200"}
#: What the floor job has to run for a pinned floor to mean "the CLI starts on it".
CLI_TESTS = (
    "tests/test_cli_dependency_floors.py",
    "tests/test_click_argument_help.py",
    "tests/test_cli_startup_is_light.py",
)


def _declared_floor(name: str) -> Version:
    """The one inclusive lower bound ``[project] dependencies`` declares for *name*."""
    declared = [req for req in _core_requirements() if canonicalize_name(req.name) == name]
    assert len(declared) == 1, (
        f"pyproject.toml must declare {name} once in [project] dependencies; "
        f"found {[str(req) for req in declared]}"
    )
    floors = [spec.version for spec in declared[0].specifier if spec.operator == ">="]
    assert len(floors) == 1, (
        f"{declared[0]} must carry exactly one `>=` floor: it is the version the "
        f"{FLOOR_JOB} job installs (#1802)"
    )
    return Version(floors[0])


def _pins() -> dict[str, str]:
    """The exact ``name==version`` pins of the floor job's constraints file."""
    text = CONSTRAINTS.read_text(encoding="utf-8")
    pairs = re.findall(r"^([A-Za-z0-9_.-]+)==([^\s#]+)\s*$", text, re.M)
    return {canonicalize_name(name): version for name, version in pairs}


def _floor_job_pytest_arguments() -> list[str]:
    """What follows ``pytest`` on every pytest command line the floor job runs."""
    steps = yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"][FLOOR_JOB]["steps"]
    arguments: list[str] = []
    for step in steps:
        run = str(step.get("run", "")).strip()
        # The other steps are pip, or a Python heredoc that shlex cannot read.
        if run.startswith(("pytest ", "python -m pytest ")):
            words = shlex.split(run, comments=True)
            arguments.extend(words[words.index("pytest") + 1 :])
    return arguments


class TestTheFloorJobInstallsTheDeclaredFloors:
    """Links 1 and 3 of the chain, read off the repository: hermetic, on every cell."""

    @pytest.mark.parametrize("name", FLOORED)
    def test_the_pin_is_the_declared_floor(self, name):
        declared = _declared_floor(name)
        pin = _pins().get(name)
        assert pin is not None, (
            f".github/constraints/transformers-floor.txt has no exact `{name}==` pin, so "
            f"the {FLOOR_JOB} job installs the newest {name} and the declared floor "
            f"{name}>={declared} is never run (#1802)"
        )
        assert Version(pin) == declared, (
            f"pyproject.toml declares {name}>={declared} but the {FLOOR_JOB} job installs "
            f"{name}=={pin}. A floor no CI cell starts the CLI on is how typer>=0.9.0 "
            f"shipped while every command needed 0.12.4 (#1802). Move the declaration "
            f"and the pin together, to a version that job's CLI tests pass on."
        )

    def test_the_floor_job_starts_the_cli_under_those_pins(self):
        arguments = _floor_job_pytest_arguments()
        assert arguments, f"the {FLOOR_JOB} job runs no pytest command"
        missing = [path for path in CLI_TESTS if path not in arguments]
        assert not missing, (
            f"the {FLOOR_JOB} job pins typer and click to their declared floors but does "
            f"not run {missing}: installing a floor proves nothing until the CLI is "
            f"started on it (#1802)"
        )
        narrowed = [arg for arg in arguments if arg in ("-k", "--deselect", "-m")]
        assert not narrowed, f"the {FLOOR_JOB} job deselects tests with {narrowed}"

    def test_doctor_reports_the_declared_typer_floor(self):
        # doctor.py keeps a literal copy of the floor (it cannot read pyproject.toml from
        # an installed wheel). Mutating either side alone fails here by name.
        from soup_cli.commands.doctor import DEPS

        rows = [row for row in DEPS if row[1] == "typer"]
        assert len(rows) == 1, f"expected exactly one typer row in doctor's DEPS: {rows}"
        _, _, floor, required = rows[0]
        assert required is True
        assert Version(floor) == _declared_floor("typer"), (
            f"doctor.py checks typer>={floor} but pyproject.toml declares "
            f"typer>={_declared_floor('typer')}: doctor must report the declared floor "
            f"(#1802)"
        )


def _installed() -> str:
    return f"typer {md.version('typer')} and click {md.version('click')}"


def _command_paths(root: click.Command) -> list[list[str]]:
    """The path of *root* and of every command and group under it, root first."""
    paths: list[list[str]] = []

    def visit(command: click.Command, ctx: click.Context, path: list[str]) -> None:
        paths.append(path)
        if not hasattr(command, "list_commands"):
            return
        for name in command.list_commands(ctx):
            child = command.get_command(ctx, name)
            assert child is not None, f"`{' '.join([*path, name])}` is listed but missing"
            visit(child, click.Context(child, info_name=name, parent=ctx), [*path, name])

    visit(root, click.Context(root, info_name="soup"), [])
    return paths


def _render_every_help_page(root: click.Command) -> tuple[dict[str, str], dict[str, str]]:
    """Run ``--help`` on *root* and on everything under it, as the user would.

    Returns ``(pages, failures)`` keyed by the command line: the plain text of each page
    that rendered, and what went wrong with each one that did not.
    """
    runner = ClickRunner()
    pages: dict[str, str] = {}
    failures: dict[str, str] = {}
    for path in _command_paths(root):
        line = " ".join(["soup", *path])
        result = runner.invoke(root, [*path, "--help"], prog_name="soup")
        text = _plain(result.output)
        if result.exit_code != 0 or "Usage:" not in text:
            failures[line] = f"exit {result.exit_code}: {result.exception!r} {text[-200:]}"
        else:
            pages[line] = text
    return pages, failures


class TestTheCliStartsOnTheInstalledTyperAndClick:
    """Link 3. On the floor job the installed pair is the declared floors; on every other
    cell it is the newest the range allows. The assertions are the same on both."""

    def test_the_walker_reports_a_page_that_does_not_render(self):
        """CONTROL: a walker that stopped at the root, or swallowed the error, would
        pass every test of this class on a Typer that renders no help at all."""

        class Unrenderable(click.Command):
            def get_help(self, ctx):
                raise TypeError("Parameter.make_metavar() missing 1 required argument")

        nested = click.Group("nested", commands={"bad": Unrenderable("bad")})
        root = click.Group("soup", commands={"good": click.Command("good"), "nested": nested})

        pages, failures = _render_every_help_page(root)

        assert set(pages) == {"soup", "soup good", "soup nested"}
        assert list(failures) == ["soup nested bad"]
        assert "make_metavar" in failures["soup nested bad"]

    def test_the_command_tree_builds_and_every_help_page_renders(self):
        # Typer 0.9.0 - 0.12.3 raise here, before any command can run:
        # `RuntimeError: Type not yet supported: int | None`.
        root = typer.main.get_command(app)

        pages, failures = _render_every_help_page(root)

        assert not failures, (
            f"{len(failures)} of {len(pages) + len(failures)} help pages do not render "
            f"on {_installed()}; first: {sorted(failures.items())[:2]}"
        )
        # 254 on 2026-10-13. A count far below that is a walker that lost the tree.
        assert len(pages) > 150, (len(pages), sorted(pages)[:5])
        assert {"soup", "soup train", "soup data clean", "soup adapters scan"} <= set(pages)

    @pytest.mark.parametrize(
        ("argv", "argument_help", "option_help"),
        [
            (
                ["adapters", "scan", "--help"],
                "Path to adapter directory",
                "Output format: text | json",
            ),
            (
                ["migrate", "--help"],
                "Path to the source config file (.yaml or .ipynb)",
                "Output path for generated soup.yaml",
            ),
            (
                ["data", "clean", "--help"],
                "Path to input dataset file (JSONL)",
                "Overwrite output file if it already exists",
            ),
        ],
    )
    def test_argument_and_option_help_both_reach_the_page(
        self, argv, argument_help, option_help
    ):
        result = CliRunner().invoke(app, argv, env=WIDE)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        page = _plain(result.output)
        for expected in (argument_help, option_help):
            assert expected in page, (
                f"`soup {' '.join(argv)}` does not show {expected!r} on {_installed()}",
                page,
            )

    @pytest.mark.parametrize(
        ("argv", "option_help"),
        [
            # The four PEP 604 annotations Typer converts only from 0.12.4 (#1802).
            (["train", "--help"], "Shared coordinator port for multiple nodes"),
            (["train", "--help"], "Rank 0 hostname or IP reachable by all nodes"),
            (["doctor", "--help"], "Also check a soup.yaml"),
        ],
    )
    def test_an_option_annotated_with_a_pep_604_union_is_on_its_help_page(
        self, argv, option_help
    ):
        result = CliRunner().invoke(app, argv, env=WIDE)

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert option_help in _plain(result.output), (
            f"`soup {' '.join(argv)}` does not show {option_help!r} on {_installed()}"
        )

    def test_soup_version_prints_the_version(self):
        result = CliRunner().invoke(app, ["version"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert __version__ in _plain(result.output)
