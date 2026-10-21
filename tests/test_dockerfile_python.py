"""The Docker image runs a Python soup-cli supports and installs exactly the tagged release.

The image used to be ``nvidia/cuda:12.1.0-runtime-ubuntu22.04`` with the distro ``python3``
(3.10) and ``pip install "soup-cli[...]"``, no version. Nothing in that line ties the image to
the release being published: when ``requires-python`` moved to ``>=3.11`` pip did not fail on
Python 3.10, it quietly resolved the newest soup-cli that still admits 3.10, and
``docker.yml`` would have pushed an image tagged with the new version that held the previous
release.

Four things keep that from happening, and each is checked here without a Docker daemon (a
real ``docker build`` is the only thing that proves the image runs):

- the base image's own Python is inside ``requires-python``, which is read from
  ``pyproject.toml`` and not copied, so a bump of the range fails here until the base follows;
- the Dockerfile refuses a Python outside the range before it downloads a wheel, and its
  guard's floor and ceiling are ``requires-python``'s own, both read from ``pyproject.toml``
  (a guard that is looser or tighter than the range is the same bug from the other side);
- the install is ``soup-cli[...]==${SOUP_VERSION}`` from a build arg with no default, followed
  by a check that the package that landed is that version;
- ``docker.yml`` derives ``SOUP_VERSION`` from the git tag and stops on any other ref.

Reading ``pyproject.toml`` with a regex and not ``tomllib`` is deliberate: the repository
still runs this suite on Python 3.10 (same reason as ``test_requires_python_bound.py``).
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
WORKFLOW = ROOT / ".github" / "workflows" / "docker.yml"
PYPROJECT = ROOT / "pyproject.toml"

#: The extras the image has always installed. Changing them changes what the GPU image can do,
#: so a deliberate change edits this set.
IMAGE_EXTRAS = frozenset({"train", "serve", "data", "eval"})

#: The python3 each Ubuntu LTS ships. A base that is not listed fails the test with the reason,
#: so a new base is a decision and not an accident.
UBUNTU_PYTHON = {"20.04": (3, 8), "22.04": (3, 10), "24.04": (3, 12)}


# --- parsing ------------------------------------------------------------------------------


def _instructions(text: str) -> list[tuple[str, str]]:
    """The Dockerfile as ``(KEYWORD, arguments)`` pairs: continuations joined, comments gone."""
    found: list[tuple[str, str]] = []
    pending = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") or (not line and not pending):
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        full = (pending + line).strip()
        pending = ""
        keyword, _, arguments = full.partition(" ")
        found.append((keyword.upper(), arguments.strip()))
    assert not pending, "the Dockerfile ends inside a line continuation"
    return found


def _dockerfile() -> list[tuple[str, str]]:
    return _instructions(DOCKERFILE.read_text(encoding="utf-8"))


def _runs() -> list[tuple[int, str]]:
    """``(position among all instructions, command)`` of every RUN."""
    return [(i, args) for i, (kw, args) in enumerate(_dockerfile()) if kw == "RUN"]


def _position(predicate: Any, what: str) -> int:
    for index, command in _runs():
        if predicate(command):
            return index
    pytest.fail(f"no RUN instruction in the Dockerfile {what}")


def _pip_install_position() -> int:
    return _position(lambda c: "pip install" in c and "soup-cli" in c, "that pip-installs soup-cli")


def _project_table() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    project = re.search(r"^\[project\]\s*$(.*?)^\[", text, re.M | re.S)
    assert project, "pyproject.toml has no [project] table"
    return project.group(1)


def _requires_python() -> SpecifierSet:
    match = re.search(r'^requires-python\s*=\s*"([^"]+)"', _project_table(), re.M)
    assert match, "no requires-python line in the [project] table of pyproject.toml"
    return SpecifierSet(match.group(1))


def _optional_dependency_names() -> set[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    table = re.search(r"^\[project\.optional-dependencies\]\s*$(.*?)(?=^\[)", text, re.M | re.S)
    assert table, "pyproject.toml has no [project.optional-dependencies] table"
    return set(re.findall(r"^([A-Za-z][\w-]*)\s*=", table.group(1), re.M))


def _bounds(specifier: SpecifierSet) -> tuple[tuple[int, int], tuple[int, int]]:
    """``(lowest, first excluded)`` minor of a ``>=X.Y,<X.Z`` range."""
    lower = upper = None
    for item in specifier:
        major, minor = (int(part) for part in item.version.split(".")[:2])
        if item.operator == ">=":
            lower = (major, minor)
        elif item.operator == "<":
            upper = (major, minor)
        else:
            pytest.fail(f"requires-python uses {item}; teach _bounds() about that operator")
    assert lower and upper, f"requires-python {specifier} needs both a >= floor and a < ceiling"
    return lower, upper


def _base_image() -> str:
    froms = [args for kw, args in _dockerfile() if kw == "FROM"]
    assert len(froms) == 1, f"expected one FROM (one stage), found {froms}"
    return froms[0].split()[0]


def _base_python(image: str) -> tuple[int, int]:
    """The Python minor the base image's ``python3`` is, or a failure that says why not."""
    ubuntu = re.fullmatch(r"nvidia/cuda:\d+\.\d+\.\d+(?:-cudnn)?-\w+-ubuntu(\d+\.\d+)", image)
    if ubuntu:
        release = ubuntu.group(1)
        if release not in UBUNTU_PYTHON:
            pytest.fail(f"Ubuntu {release} is not in UBUNTU_PYTHON: add the python3 it ships")
        return UBUNTU_PYTHON[release]
    official = re.fullmatch(r"python:(\d+)\.(\d+)(?:\.\d+)?(?:-[\w.]+)?", image)
    if official:
        return int(official.group(1)), int(official.group(2))
    pytest.fail(f"cannot tell which Python {image} ships: teach _base_python() about it")


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _steps() -> list[dict[str, Any]]:
    return _workflow()["jobs"]["build-and-push-image"]["steps"]


def _step_index(predicate: Any, what: str) -> int:
    for index, step in enumerate(_steps()):
        if predicate(step):
            return index
    pytest.fail(f"docker.yml has no step {what}")


# --- the base image ---------------------------------------------------------------------


class TestBaseImage:
    def test_base_is_an_exact_tag_not_a_floating_one(self) -> None:
        image = _base_image()
        assert ":" in image, f"{image} has no tag, so it floats to :latest"
        tag = image.split(":", 1)[1]
        assert tag != "latest", f"{image} floats"
        assert re.match(r"\d+\.\d+", tag), f"{image}: pin a full version tag, not '{tag}'"

    def test_base_python_is_inside_requires_python(self) -> None:
        image = _base_image()
        major, minor = _base_python(image)
        specifier = _requires_python()
        for patch in (0, 999):
            version = f"{major}.{minor}.{patch}"
            assert specifier.contains(version), (
                f"{image} ships Python {major}.{minor}, outside requires-python "
                f"'{specifier}': pip would not fail, it would install an older soup-cli "
                "than the release being published"
            )

    def test_python_guard_matches_the_supported_range(self) -> None:
        guard = _position(lambda c: "sys.version_info" in c, "that checks sys.version_info")
        command = dict(_runs())[guard]
        match = re.search(
            r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<=\s*sys\.version_info\[:2\]\s*<\s*"
            r"\(\s*(\d+)\s*,\s*(\d+)\s*\)",
            command,
        )
        assert match, f"cannot read the range off the Python guard: {command}"
        low = (int(match.group(1)), int(match.group(2)))
        high = (int(match.group(3)), int(match.group(4)))
        required_low, required_high = _bounds(_requires_python())
        assert low == required_low, (
            f"the guard's floor is Python {low[0]}.{low[1]} but requires-python "
            f"'{_requires_python()}' starts at {required_low[0]}.{required_low[1]}: "
            "a guard below it admits a Python pip would not install for, one above it refuses "
            "a Python the package supports; move the guard with the range"
        )
        assert high == required_high, (
            f"the guard's ceiling is Python {high[0]}.{high[1]} (excluded) but requires-python "
            f"'{_requires_python()}' stops below {required_high[0]}.{required_high[1]}: "
            "a guard above it admits a Python soup-cli does not support and pip would quietly "
            "install an older release; move the guard with the range"
        )
        assert low <= _base_python(_base_image()) < high, (
            "the guard rejects the Python the base image ships, so every build would fail"
        )

    def test_python_guard_runs_in_the_venv_before_anything_is_installed(self) -> None:
        guard = _position(lambda c: "sys.version_info" in c, "that checks sys.version_info")
        venv = _position(lambda c: "-m venv" in c, "that creates the venv")
        command = dict(_runs())[guard]
        assert venv <= guard, "the guard must run after the venv exists"
        assert re.search(r"(?<![\w.-])python -c ", command), (
            "the guard must call the venv's `python`, not `python3`"
        )
        assert guard < _pip_install_position(), "the guard must run before pip downloads wheels"


class TestVenv:
    def test_pip_installs_into_the_venv_not_the_externally_managed_system_python(self) -> None:
        env: dict[str, str] = {}
        for keyword, arguments in _dockerfile():
            if keyword == "ENV":
                for pair in shlex.split(arguments):
                    key, _, value = pair.partition("=")
                    env[key] = value
        assert "VIRTUAL_ENV" in env, "ENV VIRTUAL_ENV is not set"
        venv = env["VIRTUAL_ENV"]
        assert env.get("PATH", "").startswith(f"{venv}/bin:"), (
            f"PATH must start with {venv}/bin so `pip` and `python` are the venv's "
            f"(PEP 668 refuses the system pip on Ubuntu 24.04); PATH={env.get('PATH')!r}"
        )
        command = dict(_runs())[_position(lambda c: "-m venv" in c, "that creates the venv")]
        target = re.search(r'-m venv "?([^\s"]+)', command)
        assert target, f"cannot read the venv directory off: {command}"
        assert target.group(1) in (venv, "${VIRTUAL_ENV}", "$VIRTUAL_ENV"), (
            f"the venv is created in {target.group(1)}, not in VIRTUAL_ENV={venv}"
        )


# --- the pinned install -------------------------------------------------------------------


class TestPinnedInstall:
    def test_soup_version_is_a_build_arg_with_no_default(self) -> None:
        args = [a for kw, a in _dockerfile() if kw == "ARG"]
        assert "SOUP_VERSION" in args, (
            f"ARG SOUP_VERSION must be declared with no default (found {args}): a default "
            "is an unpinned install again"
        )
        arg_at = next(
            i for i, (kw, a) in enumerate(_dockerfile()) if kw == "ARG" and a == "SOUP_VERSION"
        )
        assert arg_at < _pip_install_position(), "ARG SOUP_VERSION must precede the install"

    def test_a_missing_or_malformed_version_stops_the_build_before_the_install(self) -> None:
        check = _position(
            lambda c: "SOUP_VERSION" in c and "grep -Eq" in c and "exit 1" in c,
            "that rejects a missing or malformed SOUP_VERSION",
        )
        assert check < _pip_install_position()
        command = dict(_runs())[check]
        pattern = re.search(r"grep -Eq '([^']+)'", command)
        assert pattern, command
        shape = re.compile(pattern.group(1))
        for good in ("0.77.0", "0.76.1", "10.2.13"):
            assert shape.search(good), f"{good} should be accepted"
        for bad in (
            "",
            "latest",
            "0.77",
            "v0.77.0",
            "0.77.0rc1",
            "0.77.0,<0.78",
            "0.77.0 --extra-index-url http://example.invalid",
        ):
            assert not shape.search(bad), f"{bad!r} must be refused"
        assert "SOUP_VERSION" in command.split("grep", 1)[0], (
            "the check must read SOUP_VERSION, not another value"
        )

    def test_the_install_pins_the_requested_release_and_keeps_the_extras(self) -> None:
        installs = [c for _, c in _runs() if "pip install" in c]
        assert installs, "the Dockerfile no longer pip-installs anything"
        specs: list[str] = []
        for command in installs:
            specs += re.findall(r"soup-cli\S*", command)
        assert len(specs) == 1, f"expected one soup-cli requirement, found {specs}"
        match = re.fullmatch(r'soup-cli\[([\w,-]+)\]==\$\{SOUP_VERSION\}"?', specs[0])
        assert match, (
            f"{specs[0]!r} does not pin the build arg: it must read "
            'soup-cli[extras]==${SOUP_VERSION}'
        )
        extras = set(match.group(1).split(","))
        assert extras == IMAGE_EXTRAS, f"the image installs {sorted(extras)}"
        assert extras <= _optional_dependency_names(), (
            f"extras {sorted(extras - _optional_dependency_names())} are not in pyproject.toml"
        )

    def test_the_build_fails_unless_the_installed_version_is_the_requested_one(self) -> None:
        check = _position(
            lambda c: "importlib.metadata" in c and "soup-cli" in c and "SOUP_VERSION" in c,
            "that compares the installed soup-cli with SOUP_VERSION",
        )
        command = dict(_runs())[check]
        assert check > _pip_install_position(), "the version check must run after the install"
        assert re.search(r"assert\s+\w+\s*==\s*\w+", command), (
            f"the check does not assert the two versions are equal: {command}"
        )
        assert "command -v soup" in command, "the console script is not checked to be on PATH"


class TestImageContract:
    def test_entrypoint_command_and_workdir_are_unchanged(self) -> None:
        instructions = _dockerfile()
        assert ("ENTRYPOINT", '["soup"]') in instructions
        assert ("CMD", '["--help"]') in instructions
        assert ("WORKDIR", "/workspace") in instructions


# --- docker.yml and docker-compose.yml ----------------------------------------------------


class TestWorkflow:
    def _version_step(self) -> dict[str, Any]:
        return _steps()[_step_index(lambda s: s.get("id") == "version", "with id 'version'")]

    def test_the_version_step_runs_before_login_and_before_the_build(self) -> None:
        version = _step_index(lambda s: s.get("id") == "version", "with id 'version'")
        login = _step_index(lambda s: "docker/login-action" in s.get("uses", ""), "that logs in")
        build = _step_index(
            lambda s: "docker/build-push-action" in s.get("uses", ""), "that builds and pushes"
        )
        assert version < login < build, (
            "resolve the version first: a run on a branch must stop before it can log in or push"
        )

    def test_the_build_receives_the_version_from_the_step_output(self) -> None:
        build = _steps()[_step_index(
            lambda s: "docker/build-push-action" in s.get("uses", ""), "that builds and pushes"
        )]
        build_args = str(build.get("with", {}).get("build-args", ""))
        assert "SOUP_VERSION=${{ steps.version.outputs.version }}" in build_args.splitlines(), (
            f"build-args do not pass the tag's version to the Dockerfile: {build_args!r}"
        )

    def test_the_version_step_stops_unless_the_run_is_on_a_release_tag(self) -> None:
        step = self._version_step()
        script = step["run"]
        env = step.get("env", {})
        assert env.get("REF_TYPE") == "${{ github.ref_type }}"
        assert env.get("REF_NAME") == "${{ github.ref_name }}"
        assert '"${REF_TYPE}" != "tag"' in script, "a branch run is not refused"
        assert "exit 1" in script
        assert 'echo "version=${REF_NAME#v}" >> "${GITHUB_OUTPUT}"' in script
        pattern = re.search(r"grep -Eq '([^']+)'", script)
        assert pattern, script
        shape = re.compile(pattern.group(1))
        for tag in ("v0.77.0", "v0.76.1", "v1.20.300"):
            assert shape.search(tag), f"{tag} is a release tag and must pass"
        for ref in ("main", "v0.77", "0.77.0", "v0.77.0rc1", "v0.77.0-rc.1", "release/v0.77.0", ""):
            assert not shape.search(ref), f"{ref!r} is not a release tag and must be refused"

    def test_no_expression_is_interpolated_into_the_script(self) -> None:
        # A ref name is attacker-chosen text: it reaches the shell through `env`, never `${{ }}`.
        assert "${{" not in self._version_step()["run"]


class TestCompose:
    def test_a_local_build_gets_the_version_from_the_environment(self) -> None:
        data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        args = data["services"]["soup"]["build"].get("args")
        names = set(args) if isinstance(args, list) else set(args or {})
        assert "SOUP_VERSION" in names, (
            "docker compose build would not pass SOUP_VERSION, and the Dockerfile has no default"
        )
