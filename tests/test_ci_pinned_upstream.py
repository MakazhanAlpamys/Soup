"""The upstream stack that pull-request CI installs is pinned, and the pins are sound.

A new release of torch, transformers or one of their companions fails for a reason no
contributor can fix from their branch, and it used to fail on every open pull request at
once, because every CI cell resolved the newest release on every run. ``pyproject.toml``
keeps one such day on record: ``datasets`` 5.1.0 dropped a dependency that ``trl`` imports
without declaring, and from that day a fresh install could not import two trainers.

So the runs a contributor sees -- every pull-request run and every push to ``main`` --
install under ``.github/constraints/pinned-upstream.txt``, and the runs that exist to meet
a new release -- the nightly full run, a manual dispatch, a push to ``release/**`` and the
weekly ``dependency drift`` job -- still resolve the newest. Which run gets which is the
``plan`` job's decision; ``tests/test_ci_full_matrix_on_approval.py`` pins that, event by
event, through the step's own shell. This file pins the constraints file itself:

* one line per package, an exact ``name==version`` and nothing else, no package twice --
  a range or a bare name is a pin that pins nothing, and a second line for one package is
  at best noise and at worst two pins ``pip`` cannot satisfy together;
* every pinned version inside every bound ``pyproject.toml`` declares for that package,
  in its core dependencies and in the extras a pinned job installs (read off the
  workflows) -- outside one, ``pip`` cannot resolve the cell at all, so a pull request
  that moves a bound moves the pin with it. An extra no pinned job installs is not read.

``pyproject.toml`` is read with regular expressions, not ``tomllib``, which is 3.11+ while
this repository supports 3.10: the convention of ``tests/test_requires_python_bound.py``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from tests.test_ci_full_matrix_on_approval import pinned_job_extras

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / ".github" / "constraints" / "pinned-upstream.txt"
PYPROJECT = ROOT / "pyproject.toml"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"

#: What must not float on a pull request: torch, transformers and the four libraries the
#: `[train]` extra declares beside them, plus the three that transformers resolves with
#: (tokenizers, huggingface-hub and safetensors, which it bounds itself). A pin may be
#: added; dropping one of these lets that package float on pull requests again.
HEAVY_STACK = frozenset(
    {
        "torch",
        "transformers",
        "trl",
        "peft",
        "accelerate",
        "datasets",
        "tokenizers",
        "huggingface-hub",
        "safetensors",
    }
)
#: The ones ``pyproject.toml`` bounds directly. If the reader below stops finding one of
#: these, the bound check has gone vacuous, not green.
DECLARED_IN_PYPROJECT = frozenset(
    {"torch", "transformers", "trl", "peft", "accelerate", "datasets", "huggingface-hub"}
)


# --- Reading the constraints file ---------------------------------------------


def _pins(text: str) -> dict[str, str]:
    """Every pin in *text*, as ``{name: version}``; anything that is not one fails.

    A line is blank, a comment that starts the line, or exactly ``name==version`` with
    the name in its canonical spelling. Inline comments, ranges, wildcards, extras,
    markers, URLs and pip options are all refused: each of them is a way for a line to
    look like a pin and constrain nothing, or to change what every other line means.
    """
    pins: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{PINNED.name}:{number}: {raw!r}"
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            raise AssertionError(f"{where} is not a requirement pip can pin: {exc}") from exc
        clauses = list(requirement.specifier)
        assert (
            len(clauses) == 1
            and clauses[0].operator == "=="
            and not clauses[0].version.endswith(".*")
        ), f"{where} is not an exact pin: write one `name==version`, no range and no wildcard"
        assert (
            not requirement.extras and requirement.marker is None and requirement.url is None
        ), f"{where} carries an extra, a marker or a URL: a pin here is `name==version` only"
        version = clauses[0].version
        try:
            parsed = Version(version)
        except InvalidVersion as exc:
            raise AssertionError(f"{where} pins a version that does not parse: {exc}") from exc
        assert parsed.local is None, f"{where} pins a local version, which PyPI never serves"
        name = canonicalize_name(requirement.name)
        assert line == f"{name}=={version}", (
            f"{where} must be spelled `{name}=={version}`: the canonical name, no spaces and "
            "no trailing comment, so that one grep finds the pin"
        )
        assert name not in pins, (
            f"{where} pins {name} a second time (already =={pins[name]}): pip refuses "
            "two different pins and a reader cannot tell which one is meant"
        )
        pins[name] = version
    return pins


def _committed_pins() -> dict[str, str]:
    assert PINNED.is_file(), f"{PINNED.relative_to(ROOT).as_posix()} is missing"
    return _pins(PINNED.read_text(encoding="utf-8"))


# --- Reading pyproject.toml -----------------------------------------------------


#: The project's own name: an extra that lists ``soup-cli[train]`` installs ``train`` too.
PROJECT = "soup-cli"


def _requirements(block: str) -> list[Requirement]:
    """Every quoted requirement in a block of pyproject.toml, comment lines left out."""
    code = "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))
    return [Requirement(raw) for raw in re.findall(r'"([^"\n]+)"', code)]


def _extras(text: str) -> dict[str, list[Requirement]]:
    """The ``[project.optional-dependencies]`` table: each extra with its requirements."""
    table = re.search(r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[)", text, re.M | re.S)
    assert table is not None, "pyproject.toml has no [project.optional-dependencies] table"
    code = "\n".join(
        line for line in table.group(1).splitlines() if not line.lstrip().startswith("#")
    )
    # An array ends at a `]` that ends its line; the `]` of `soup-cli[train]` does not.
    arrays = re.findall(r"^([A-Za-z0-9_-]+)\s*=\s*\[(.*?)\]\s*$", code, re.M | re.S)
    return {canonicalize_name(name): _requirements(body) for name, body in arrays}


def _declared(text: str, installed: Iterable[str]) -> dict[str, list[Requirement]]:
    """What a cell that installs the *installed* extras is bound by, by canonical name.

    The core dependencies, and each of those extras followed through self-references such
    as ``soup-cli[train,mcp,data]``. An extra that is not installed is not read.
    """
    core = re.search(r"^dependencies\s*=\s*\[\n(.*?)^\]", text, re.M | re.S)
    assert core is not None, "pyproject.toml has no [project] dependencies array"
    requirements = _requirements(core.group(1))
    table = _extras(text)
    pending = [canonicalize_name(extra) for extra in installed]
    seen: set[str] = set()
    while pending:
        extra = pending.pop()
        if extra in seen:
            continue
        seen.add(extra)
        assert extra in table, f"pyproject.toml declares no extra `{extra}`: {sorted(table)}"
        for requirement in table[extra]:
            requirements.append(requirement)
            if canonicalize_name(requirement.name) == PROJECT:
                pending.extend(canonicalize_name(inner) for inner in requirement.extras)
    declared: dict[str, list[Requirement]] = {}
    for requirement in requirements:
        declared.setdefault(canonicalize_name(requirement.name), []).append(requirement)
    return declared


def _bounds() -> dict[str, list[Requirement]]:
    """The bounds on the committed tree: core plus the extras the pinned jobs install."""
    return _declared(PYPROJECT.read_text(encoding="utf-8"), pinned_job_extras())


def _out_of_bounds(pins: dict[str, str], declared: dict[str, list[Requirement]]) -> list[str]:
    """One message per pin that a declared bound does not admit."""
    problems = []
    for name, version in sorted(pins.items()):
        for requirement in declared.get(name, []):
            if not requirement.specifier.contains(Version(version), prereleases=True):
                problems.append(
                    f"{name}=={version} is outside pyproject.toml's `{requirement}`"
                )
    return problems


# --- The file is a list of exact pins ------------------------------------------


class TestTheReaderHasTeeth:
    """CONTROL. The checks on the committed file are only as good as this reader."""

    def test_comments_and_blank_lines_are_skipped_and_pins_are_read(self):
        text = "# a header\n\ntorch==2.14.1\n  # indented comment\nhuggingface-hub==2.2.0\n"
        assert _pins(text) == {"torch": "2.14.1", "huggingface-hub": "2.2.0"}

    @pytest.mark.parametrize(
        "line",
        [
            "numpy",  # a bare name constrains nothing
            "numpy>=2.0",
            "numpy<3",
            "torch~=2.14",
            "torch!=2.14.0",
            "torch==2.14.*",
            "torch===2.14.1",
            "torch==2.14.1,<3",
            "torch>=2.14.1,==2.14.1",
        ],
    )
    def test_a_line_that_does_not_pin_one_exact_version_is_refused(self, line):
        with pytest.raises(AssertionError, match="not an exact pin"):
            _pins(f"transformers==5.19.0\n{line}\n")

    @pytest.mark.parametrize(
        "line",
        [
            "torch[opt-einsum]==2.14.1",
            "torch==2.14.1; python_version < '3.12'",
            "torch==2.14.1 ; sys_platform == 'linux'",
        ],
    )
    def test_an_extra_or_a_marker_is_refused(self, line):
        with pytest.raises(AssertionError, match="an extra, a marker or a URL"):
            _pins(f"{line}\n")

    @pytest.mark.parametrize(
        "line",
        [
            "-r other.txt",
            "-c other.txt",
            "--extra-index-url https://example.invalid/simple",
            "-e .",
            "torch @ https://example.invalid/torch-2.14.1-py3-none-any.whl",
            "torch==2.14.1 --hash=sha256:" + "0" * 64,
            "torch==2.14.1  # the newest",
            "torch == 2.14.1",
            "Torch==2.14.1",
            "huggingface_hub==2.2.0",
            "torch==2.14.1+cpu",
            "torch==",
            "==2.14.1",
        ],
    )
    def test_an_option_a_url_or_another_spelling_is_refused(self, line):
        with pytest.raises(AssertionError):
            _pins(f"{line}\n")

    @pytest.mark.parametrize(
        "second",
        ["torch==2.14.1", "torch==2.13.0"],
        ids=["same-version", "another-version"],
    )
    def test_a_package_listed_twice_is_refused(self, second):
        with pytest.raises(AssertionError, match="a second time"):
            _pins(f"torch==2.14.1\ntransformers==5.19.0\n{second}\n")


class TestTheCommittedFile:
    def test_every_line_is_one_exact_pin_and_no_package_is_listed_twice(self):
        pins = _committed_pins()
        assert pins, "the constraints file pins nothing"

    def test_the_heavy_stack_is_pinned(self):
        missing = HEAVY_STACK - set(_committed_pins())
        assert not missing, (
            f"{sorted(missing)} float on pull requests again: the next release of one of "
            "them fails every open pull request at once"
        )

    def test_it_says_when_and_how_to_bump_it(self):
        """The header is the procedure: a pinned file nobody knows how to move goes stale,
        and stale pins are how a pull request stays green on versions no user installs."""
        header = []
        for line in PINNED.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                break
            header.append(line.lstrip("#").strip())
        text = " ".join(" ".join(header).split())
        assert "nightly" in text, "say which run has to be green on the newer versions"
        assert "pyproject.toml" in text, "say that a bound that moves takes its pin with it"
        assert 'python -m pip install --dry-run --ignore-installed -e ".[dev]"' in text, (
            "give the command that prints the versions to copy"
        )
        assert "tests/test_ci_pinned_upstream.py" in text, "name the test that checks a bump"


# --- Every pin lies inside pyproject's bound -------------------------------------


class TestPinsLieInsideTheDeclaredBounds:
    def test_the_bounds_of_the_declared_stack_were_found(self):
        """CONTROL: an empty reading of pyproject.toml would pass the check below."""
        declared = _bounds()
        missing = DECLARED_IN_PYPROJECT - set(declared)
        assert not missing, f"no declaration found in pyproject.toml for {sorted(missing)}"
        unbounded = sorted(
            name
            for name in DECLARED_IN_PYPROJECT
            if not any(len(requirement.specifier) for requirement in declared[name])
        )
        assert not unbounded, f"{unbounded} are declared with no version bound at all"

    def test_every_pin_is_inside_every_bound_pyproject_declares_for_it(self):
        problems = _out_of_bounds(_committed_pins(), _bounds())
        assert not problems, (
            "; ".join(problems)
            + ". pip cannot resolve a pull-request cell under such a pin. Move the pin "
            f"inside the bound: the header of {PINNED.relative_to(ROOT).as_posix()} says how"
        )

    @pytest.mark.parametrize(
        ("pin", "bound"),
        [
            ({"transformers": "6.0.0"}, "transformers>=5.16.1,<6.0.0"),
            ({"transformers": "5.16.0"}, "transformers>=5.16.1,<6.0.0"),
            ({"trl": "1.15.0"}, "trl>=0.29.0,<1.0.0"),
            ({"torch": "2.5.1"}, "torch>=2.6.0"),
            ({"huggingface-hub": "0.15.0"}, "huggingface_hub>=0.16.0"),
        ],
    )
    def test_a_pin_outside_a_bound_is_reported(self, pin, bound):
        """CONTROL, over both ends of a range and over a name spelled two ways."""
        requirement = Requirement(bound)
        declared = {canonicalize_name(requirement.name): [requirement]}
        (name, version), = pin.items()
        assert _out_of_bounds(pin, declared) == [
            f"{name}=={version} is outside pyproject.toml's `{requirement}`"
        ]

    PYPROJECT_LIKE = (
        "[project]\n"
        "dependencies = [\n"
        '    "huggingface-hub>=0.16.0",\n'
        "]\n\n"
        "[project.optional-dependencies]\n"
        "train = [\n"
        "    # a comment with a \"quoted\" word\n"
        '    "transformers>=5.16.1,<6.0.0",\n'
        "]\n"
        'dev = ["soup-cli[train,data]", "pytest>=7.0"]\n'
        'data = ["datasketch>=1.6.0"]\n'
        'mlx = ["mlx-lm>=0.31.3", "transformers>=5.20.0,<6.0.0"]\n'
        'legacy = ["transformers<5.0.0", "optimum[onnxruntime]>=1.16.0"]\n'
        "\n[project.scripts]\n"
    )

    def test_a_pin_is_checked_against_every_installed_extra_that_declares_the_package(self):
        """Through the self-reference too: `dev` installs `train`, so `train` binds."""
        declared = _declared(self.PYPROJECT_LIKE, {"dev", "mlx"})
        assert sorted(declared) == [
            "datasketch",
            "huggingface-hub",
            "mlx-lm",
            "pytest",
            "soup-cli",
            "transformers",
        ]
        assert len(declared["transformers"]) == 2
        narrower = Requirement("transformers>=5.20.0,<6.0.0")
        assert _out_of_bounds({"transformers": "5.19.0"}, declared) == [
            f"transformers==5.19.0 is outside pyproject.toml's `{narrower}`"
        ]
        assert _out_of_bounds({"transformers": "5.20.1", "tokenizers": "0.23.3"}, declared) == []

    def test_an_extra_no_pinned_job_installs_cannot_fail_the_check(self):
        """`legacy` wants transformers below 5. No cell installs it, so it binds no cell."""
        assert sorted(_extras(self.PYPROJECT_LIKE)) == ["data", "dev", "legacy", "mlx", "train"]
        only_dev = _declared(self.PYPROJECT_LIKE, {"dev"})
        assert _out_of_bounds({"transformers": "5.19.0"}, only_dev) == []
        with_legacy = _declared(self.PYPROJECT_LIKE, {"dev", "legacy"})
        assert len(_out_of_bounds({"transformers": "5.19.0"}, with_legacy)) == 1
        with pytest.raises(AssertionError, match="declares no extra `nope`"):
            _declared(self.PYPROJECT_LIKE, {"nope"})

    def test_the_real_extras_are_read_and_followed(self):
        """CONTROL on the committed pyproject.toml: `dev` reaches torch through `train`,
        `mlx` alone does not, and both bound transformers."""
        text = PYPROJECT.read_text(encoding="utf-8")
        assert {"train", "dev", "mlx", "qat"} <= set(_extras(text))
        assert "torch" in _declared(text, {"dev"})
        assert "torch" not in _declared(text, {"mlx"})
        assert "transformers" in _declared(text, {"mlx"})
        assert set(_bounds()) >= DECLARED_IN_PYPROJECT


# --- Contributors are told ------------------------------------------------------


def test_contributing_says_pull_requests_install_a_pinned_stack():
    text = " ".join(CONTRIBUTING.read_text(encoding="utf-8").split())
    assert "`.github/constraints/pinned-upstream.txt`" in text
    assert "Pull-request CI installs a pinned upstream stack" in text
