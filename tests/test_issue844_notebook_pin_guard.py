"""Guard for Issue #844: `notebooks/proof-4gb.ipynb` going stale again.

The notebook was found installing from `git+...@main` while its prose called an
older release "published" and cited a closed issue as open -- because nothing
checked the pin against reality. This test reads the notebook JSON directly (no
kernel, no network, no GPU) and asserts the three things that made it stale:

- the install cell pins an exact `soup-cli[train]==X.Y.Z`, never a `git+` URL;
- that `X.Y.Z` has a `## [X.Y.Z]` heading in CHANGELOG.md, i.e. it is a
  released version and not whatever `main` happened to be that day;
- every `from soup_cli.<module> import <name>` in a code cell still resolves
  against the current source tree, so a rename on `main` fails here instead of
  surfacing only when someone bumps the pin.
"""

import ast
import importlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "proof-4gb.ipynb"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"

PIN_RE = re.compile(r'soup-cli\[train\]\s*==\s*(?P<version>\d+\.\d+\.\d+)')
GIT_INSTALL_RE = re.compile(r'soup-cli\[train\]\s*@\s*git\+')


def _load_notebook():
    with NOTEBOOK_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _code_cell_sources():
    nb = _load_notebook()
    return [
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    ]


def _install_cell_source():
    for src in _code_cell_sources():
        if "%pip install" in src and "soup-cli" in src:
            return src
    pytest.fail("no code cell installs soup-cli")


class TestTheInstallCellIsPinnedToARelease:
    def test_it_does_not_install_from_git(self):
        src = _install_cell_source()
        assert not GIT_INSTALL_RE.search(src), (
            "the install cell pulls soup-cli from a git ref, which is not "
            "reinstallable later -- pin it to a released version instead"
        )

    def test_it_pins_an_exact_version(self):
        src = _install_cell_source()
        match = PIN_RE.search(src)
        assert match, (
            'expected `soup-cli[train]==X.Y.Z` in the install cell, found '
            f'none in:\n{src}'
        )

    def test_the_pinned_version_is_actually_released(self):
        src = _install_cell_source()
        match = PIN_RE.search(src)
        assert match, "install cell has no version pin to check"
        version = match.group("version")
        changelog = CHANGELOG_PATH.read_text(encoding="utf-8")
        heading = f"## [{version}]"
        assert heading in changelog, (
            f"notebook pins soup-cli=={version}, but {CHANGELOG_PATH.name} has "
            f"no '{heading}' heading -- that version was never released, or "
            "the changelog entry was removed"
        )


class TestEveryImportInTheNotebookStillResolves:
    """A rename on `main` should fail here, not on the next pin bump."""

    def _soup_cli_imports(self):
        imports = []
        for src in _code_cell_sources():
            try:
                tree = ast.parse(src)
            except SyntaxError:
                # Cells containing %pip magics or bare shell lines aren't
                # valid Python on their own; skip those, they carry no
                # `from soup_cli...` imports anyway.
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "soup_cli" or node.module.startswith("soup_cli."):
                        for alias in node.names:
                            imports.append((node.module, alias.name))
        return imports

    def test_notebook_actually_imports_something_from_soup_cli(self):
        assert self._soup_cli_imports(), (
            "expected at least one `from soup_cli...` import in the "
            "notebook; if that's no longer true this guard needs updating"
        )

    def test_each_imported_name_resolves_against_the_current_tree(self):
        failures = []
        for module_name, attr_name in self._soup_cli_imports():
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                failures.append(f"{module_name}: module import failed ({exc})")
                continue
            if not hasattr(module, attr_name):
                failures.append(f"{module_name}.{attr_name}: no such attribute")
        assert not failures, (
            "notebook imports names that no longer exist -- update the "
            "notebook (or the pinned version) before merging:\n"
            + "\n".join(failures)
        )
