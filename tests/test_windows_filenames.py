"""One Windows colon-filename skip, without censoring other Windows tests (#2492)."""

from __future__ import annotations

import ast
import builtins
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TESTS = Path(__file__).resolve().parent
_HELPER = "_windows_filenames.py"


@pytest.fixture
def filenames():
    # The repository-copy guard can run on main before this helper is added.
    from tests import _windows_filenames

    return _windows_filenames


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    return ""


def _colon_skip_builders(source: str) -> list[int]:
    """Find pytest skip constructors for this filename restriction, not all win32 skips."""
    tree = ast.parse(source)
    aliases = {"pytest": "pytest"}
    strings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name == "pytest":
                    aliases[name.asname or name.name] = name.name
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for name in node.names:
                aliases[name.asname or name.name] = f"pytest.{name.name}"
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        strings[target.id] = node.value.value

    builders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts = _dotted_name(node.func).split(".")
        name = ".".join([aliases.get(parts[0], parts[0]), *parts[1:]])
        if name not in {"pytest.mark.skipif", "pytest.skip"}:
            continue
        reason = next((k.value for k in node.keywords if k.arg == "reason"), None)
        if reason is None and name == "pytest.skip" and node.args:
            reason = node.args[0]
        text = ""
        if isinstance(reason, ast.Constant) and isinstance(reason.value, str):
            text = reason.value
        elif isinstance(reason, ast.Name):
            text = strings.get(reason.id, "")
        lower = text.casefold()
        colon = "colon" in lower or re.search(r"(['\"]):\1", text)
        filename = any(word in lower for word in ("filename", "file name", "windows name"))
        if colon and filename and "windows" in lower:
            builders.append(node.lineno)
    return builders


def test_only_the_shared_helper_constructs_the_colon_filename_skip():
    builders = []
    for path in sorted(_TESTS.rglob("*.py")):
        for line in _colon_skip_builders(path.read_text(encoding="utf-8")):
            builders.append((str(path.relative_to(_TESTS)), line))

    assert [path for path, _ in builders] == [_HELPER], (
        "The Windows colon-filename skip must be built once in "
        f"{_HELPER}, not copied locally: {builders}"
    )


def test_the_platform_predicate_is_defined_only_in_the_helper():
    definers = []
    for path in sorted(_TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "_windows_colon_filename":
                    definers.append(str(path.relative_to(_TESTS)))
    assert definers == [_HELPER], f"Do not copy the shared Windows predicate: {definers}"


@pytest.mark.parametrize(("platform", "expected"), [("win32", True), ("linux", False),
                                                  ("darwin", False)])
@pytest.mark.parametrize("ci", [None, "true"])
def test_filename_restriction_applies_to_windows_locally_and_in_ci(
    platform, expected, ci, monkeypatch,
):
    if ci is None:
        monkeypatch.delenv("CI", raising=False)
    else:
        monkeypatch.setenv("CI", ci)

    def helper_import(name, *args, **kwargs):
        if name == "sys":
            return SimpleNamespace(platform=platform)
        return builtins.__import__(name, *args, **kwargs)

    # Evaluate the actual marker too: a correct predicate wired to skipif(False)
    # would otherwise pass on a non-Windows host. Only this isolated module's
    # import resolves to fake sys; the real sys.platform is never changed.
    path = _TESTS / _HELPER
    namespace = {"__builtins__": {**vars(builtins), "__import__": helper_import}}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    marker = namespace["skip_colon_on_windows"]
    assert marker.args == (expected,)


def test_marker_keeps_the_original_condition_and_reason(filenames):
    marker = filenames.skip_colon_on_windows
    assert marker.args == (sys.platform == "win32",)
    assert marker.kwargs == {"reason": "':' is not valid in a Windows name"}


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\n_NO_COLON_ON_WINDOWS = pytest.mark.skipif(\n"
        "    sys.platform == 'win32', reason=\"':' is not valid in a Windows name\")",
        "import pytest as pt\npt.mark.skipif(os.name == 'nt', "
        "reason='Colon is not a Windows filename')",
        "from pytest import mark as marks\n"
        "marks.skipif(True, reason='Windows filenames cannot contain colons')",
        "from pytest import skip as skip_case\n"
        "if sys.platform == 'win32':\n"
        "    skip_case('colon is not a valid Windows filename character')",
        "import pytest\nWHY = 'colon filename on Windows'\n"
        "pytest.mark.skipif(is_windows, reason=WHY)",
    ],
)
def test_guard_catches_named_inline_runtime_and_aliased_copies(source):
    assert len(_colon_skip_builders(source)) == 1


@pytest.mark.parametrize(
    "source",
    [
        "pytest.mark.skipif(sys.platform == 'win32', reason='symlinks need privileges')",
        "pytest.mark.skipif(sys.platform == 'win32', "
        "reason='a backslash separates directories on Windows')",
        "pytest.mark.skipif(sys.platform == 'win32', reason='Windows native kernel crash')",
        "pytest.skip('this spelling is routed to hub prefetch')",
        "pytest.mark.skipif(True, reason='Windows: colon-separated environment variables')",
        "console.print('colon is not valid in a Windows filename')",
    ],
)
def test_guard_leaves_unrelated_platform_skips_and_text_alone(source):
    assert _colon_skip_builders(source) == []
