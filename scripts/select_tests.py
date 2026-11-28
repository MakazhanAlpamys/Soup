"""Print a focused local pytest command using the changed files and static imports."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).absolute().parents[1]
SCAN_ROOTS = ("src/soup_cli/", "scripts/", "tests/")
MAX_FOCUSED_FILES = 300
SHARED_FILES = {
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "setup.py",
    "tox.ini",
    "src/soup_cli/cli.py",
    "src/soup_cli/__main__.py",
}
SHARED_PREFIXES = (".github/", "src/soup_cli/config/", "tests/fixtures/")
DYNAMIC_IMPORTS = {"__import__", "import_module"}
GENERIC_TEST_STEMS = {
    "cli",
    "data",
    "train",
    "model",
    "eval",
    "serve",
    "config",
    "utils",
    "base",
    "common",
    "helpers",
}


class SelectionError(ValueError):
    """Local evidence is insufficient to suggest a focused run."""


@dataclass(frozen=True)
class TestPlan:
    paths: tuple[str, ...]
    reason: str | None = None
    scope: str | None = None
    unknown_dynamic_imports: int = 0


@dataclass(frozen=True)
class ImportGraph:
    modules: dict[str, str]
    reverse: dict[str, set[str]]
    unknown_dynamic_imports: int


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            stderr=subprocess.DEVNULL,
            timeout=15,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SelectionError(f"Git could not run {args[0]}") from exc


def _git_paths(root: Path, *args: str) -> set[str]:
    output = _git(root, *args)
    if not output:
        return set()
    if not output.endswith(b"\0"):
        raise SelectionError("Git returned an incomplete path list")
    try:
        paths = output[:-1].decode("utf-8").split("\0")
    except UnicodeDecodeError as exc:
        raise SelectionError("Git returned a path that is not UTF-8") from exc
    if not all(_portable_relative_path(path) for path in paths):
        raise SelectionError("Git returned a path that is not portable across shells")
    return set(paths)


def changed_paths(root: Path, base: str = "HEAD") -> set[str]:
    """Compare the working tree with a commit and include non-ignored untracked files."""
    revision = _git(root, "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}").strip()
    if not re.fullmatch(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
        raise SelectionError("Git returned an invalid comparison commit")
    changed = _git_paths(
        root,
        "diff",
        "--name-only",
        "--no-renames",
        "--no-ext-diff",
        "-z",
        revision.decode("ascii"),
        "--",
    )
    return changed | _git_paths(root, "ls-files", "--others", "--exclude-standard", "-z")


def _candidate_paths(root: Path) -> set[str]:
    return _git_paths(
        root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *SCAN_ROOTS
    )


def _portable_relative_path(path: str) -> bool:
    if (
        "\\" in path
        or ":" in path
        or any(
            ord(char) < 32 or 127 <= ord(char) <= 159 or 0xD800 <= ord(char) <= 0xDFFF
            for char in path
        )
    ):
        return False
    return all(part and part.rstrip(" .") == part for part in path.split("/"))


def _contains(root: str, path: str) -> bool:
    try:
        return os.path.commonpath((root, path)) == root
    except ValueError:
        return False


def _canonical_change(root: str, path: str) -> str:
    candidate = os.path.abspath(os.path.join(root, path))
    if not _contains(root, candidate):
        raise SelectionError("an input path is outside the checkout")
    relative = Path(os.path.relpath(candidate, root)).as_posix()
    if not _portable_relative_path(relative):
        raise SelectionError("an input path is not a portable repository file path")
    return relative


def _file_in_checkout(root: str, relative: str) -> Path:
    try:
        path = os.path.realpath(os.path.join(root, relative))
        if not _contains(root, path):
            raise SelectionError(f"file is outside the checkout: {relative}")
        if os.path.isfile(path):
            return Path(path)
    except OSError as exc:
        raise SelectionError(f"could not inspect {relative}") from exc
    raise SelectionError(f"file is missing or is not a regular file: {relative}")


def _test_stem(path: str) -> str | None:
    if not path.startswith("tests/") or not path.endswith(".py"):
        return None
    stem = PurePosixPath(path).stem
    if stem.startswith("test_"):
        return stem.removeprefix("test_")
    if stem.endswith("_test"):
        return stem.removesuffix("_test")
    return None


def _module_name(path: str) -> str:
    parts = path.removeprefix("src/").removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _resolve_import(name: str, package: str) -> str:
    try:
        return resolve_name(name, package)
    except ImportError:
        return ""


def _literal_soup_module(value: str, modules: dict[str, str]) -> str:
    module, separator, attribute = value.partition(":")
    if not module.startswith("soup_cli."):
        return ""
    if not all(part.isidentifier() for part in module.split(".")):
        return ""
    if separator and not all(part.isidentifier() for part in attribute.split(".")):
        return ""
    while module:
        if module in modules:
            return module
        module = module.rpartition(".")[0]
    return ""


def _imported_modules(
    tree: ast.AST, module: str, is_package: bool, modules: dict[str, str]
) -> tuple[set[str], int]:
    """Find static and string references and count unresolved import calls."""
    package = module if is_package else module.rpartition(".")[0]
    names = set()
    unknown = 0
    loader_names = set(DYNAMIC_IMPORTS)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import("." * node.level + (node.module or ""), package)
            if base:
                names.add(base)
                names.update(f"{base}.{alias.name}" for alias in node.names)
            if base in {"builtins", "importlib"}:
                loader_names.update(
                    alias.asname
                    for alias in node.names
                    if alias.asname and alias.name in DYNAMIC_IMPORTS
                )
        elif isinstance(node, ast.Call):
            calls.append(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("soup_cli."):
                names.add(_literal_soup_module(node.value, modules))

    for node in calls:
        callee = node.func
        kind = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
        if kind not in loader_names and kind != "importorskip":
            continue
        first = (
            node.args[0]
            if node.args
            else next((kw.value for kw in node.keywords if kw.arg == "name"), None)
        )
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.add(_resolve_import(first.value, package))
        elif kind in loader_names:
            unknown += 1
    return names, unknown


def _build_graph(root: Path, *, candidates: Iterable[str] | None = None) -> ImportGraph:
    real_root = os.path.realpath(root)
    modules = {}
    paths = _candidate_paths(root) if candidates is None else candidates
    for path in sorted(paths):
        if not path.endswith(".py"):
            continue
        name = _module_name(path)
        if name in modules:
            raise SelectionError("multiple files have the same Python module name")
        modules[name] = path

    reverse = defaultdict(set)
    unknown_dynamic_imports = 0
    for name, relative in modules.items():
        path = _file_in_checkout(real_root, relative)
        try:
            tree = ast.parse(path.read_bytes(), filename=relative)
        except (OSError, SyntaxError, ValueError, RecursionError) as exc:
            raise SelectionError(f"could not parse {relative}") from exc
        dependencies, unknown = _imported_modules(
            tree, name, relative.endswith("/__init__.py"), modules
        )
        unknown_dynamic_imports += unknown
        for dependency in dependencies:
            if dependency in modules and dependency != name:
                reverse[dependency].add(name)
    return ImportGraph(modules, reverse, unknown_dynamic_imports)


def _affected_modules(start: str, graph: ImportGraph, *, direct: bool = False) -> set[str]:
    pending = [start]
    visited = set()
    while pending:
        current = pending.pop()
        if current not in visited:
            visited.add(current)
            consumers = graph.reverse.get(current, ())
            if direct:
                # The narrower tier follows every test/helper edge but stops at
                # intermediary source modules such as the shared CLI dispatcher.
                consumers = (name for name in consumers if graph.modules[name].startswith("tests/"))
            pending.extend(consumers)
    return visited


def _name_matches(path: str, test_stems: dict[str, str]) -> set[str]:
    stem = PurePosixPath(path).stem.strip("_").replace("-", "_")
    if stem in GENERIC_TEST_STEMS:
        return set()
    return {name for name, test_stem in test_stems.items() if f"_{stem}_" in f"_{test_stem}_"}


def _selection_plan(
    root: Path, changes: Iterable[str], max_files: int, graph: ImportGraph | None
) -> TestPlan:
    real_root = os.path.realpath(root)
    selected_changes = sorted({_canonical_change(real_root, path) for path in changes})
    if not selected_changes:
        raise SelectionError("no changed files were found")
    for path in selected_changes:
        basename = PurePosixPath(path).name
        if (
            path in SHARED_FILES
            or path.startswith(SHARED_PREFIXES)
            or basename in {"conftest.py", "__init__.py"}
        ):
            raise SelectionError(f"shared configuration, fixtures or entry point changed: {path}")
        if not path.startswith(SCAN_ROOTS) or not path.endswith(".py"):
            raise SelectionError(f"no mapping for {path}")
        _file_in_checkout(real_root, path)

    if graph is None:
        graph = _build_graph(root)
    test_stems = {
        name: stem for name, path in graph.modules.items() if (stem := _test_stem(path)) is not None
    }
    unknown = graph.unknown_dynamic_imports
    transitive = set()
    named = set()
    imported = []
    for path in selected_changes:
        module = _module_name(path)
        if graph.modules.get(module) != path:
            return _full_plan(f"no mapping for {path} in the checkout", unknown)
        affected = _affected_modules(module, graph) & test_stems.keys()
        if affected:
            imported.append(module)
            transitive.update(affected)
        else:
            matches = _name_matches(path, test_stems)
            if not matches:
                return _full_plan(f"no related tests found for {path}", unknown)
            named.update(matches)

    selected = transitive | named
    scope = "transitive"
    if len(selected) > max_files:
        selected = named.copy()
        for module in imported:
            affected = _affected_modules(module, graph, direct=True) & test_stems.keys()
            if not affected:
                return _full_plan(
                    f"more than {max_files} transitive related test files and no direct tests "
                    f"for {graph.modules[module]}",
                    unknown,
                )
            selected.update(affected)
        if len(selected) > max_files:
            return _full_plan(f"more than {max_files} related test files", unknown)
        scope = "direct"

    paths = tuple(sorted(graph.modules[name] for name in selected))
    return TestPlan(paths, scope=scope, unknown_dynamic_imports=unknown)


def _full_plan(reason: str, unknown_dynamic_imports: int = 0) -> TestPlan:
    return TestPlan(("tests/",), reason=reason, unknown_dynamic_imports=unknown_dynamic_imports)


def plan_tests(
    root: Path,
    changes: Iterable[str],
    *,
    max_files: int = MAX_FOCUSED_FILES,
    graph: ImportGraph | None = None,
) -> TestPlan:
    """Select test files or explain why to use the full local suite."""
    try:
        return _selection_plan(root, changes, max_files, graph)
    except (SelectionError, OSError) as exc:
        return _full_plan(
            str(exc) if isinstance(exc, SelectionError) else "could not read checkout"
        )


def _powershell_quote(arg: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", arg):
        return arg
    return "'" + arg.replace("'", "''") + "'"


def pytest_command(plan: TestPlan, *, shell: str, cwd_prefix: Path | None = None) -> str:
    args = ["python", "-m", "pytest"]
    if plan.reason is None:
        args.append("--no-cov")
    args.extend((cwd_prefix / path).as_posix() if cwd_prefix else path for path in plan.paths)
    return shlex.join(args) if shell == "posix" else " ".join(map(_powershell_quote, args))


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Exit code 0 prints a focused or full-suite command on stdout. "
            "The SOUP_TEST_SELECTION JSON record is on stderr. "
            "Invalid arguments use exit code 2 and print no command."
        ),
    )
    parser.add_argument("paths", nargs="*", help="explicit paths relative to the repository root")
    parser.add_argument("--base", help="compare the working tree with this commit instead of HEAD")
    parser.add_argument(
        "--shell",
        choices=("posix", "powershell"),
        default="powershell" if sys.platform == "win32" else "posix",
    )
    args = parser.parse_args(argv)
    if args.base is not None and args.paths:
        parser.error("--base cannot be combined with explicit paths")

    try:
        base = args.base if args.base is not None else "HEAD"
        changes = args.paths or changed_paths(REPO_ROOT, base)
    except SelectionError as exc:
        plan = _full_plan(str(exc))
    else:
        plan = plan_tests(REPO_ROOT, changes)
    err = Console(stderr=True)
    if plan.reason is not None:
        mode = "full"
        status = f"Full suite: {plan.reason}."
    else:
        mode = "focused"
        count = len(plan.paths)
        noun = "file" if count == 1 else "files"
        status = f"Selected {count} test {noun}."
        if plan.scope == "direct":
            status += " Scope: direct import and test-helper links. Transitive scope was too broad."
    if plan.unknown_dynamic_imports:
        status += f" Found {plan.unknown_dynamic_imports} unresolved dynamic imports."
    err.print(status, markup=False, soft_wrap=True)
    record = json.dumps(
        {
            "mode": mode,
            "reason": plan.reason,
            "scope": plan.scope,
            "unknown_dynamic_imports": plan.unknown_dynamic_imports,
        },
        separators=(",", ":"),
    )
    err.print(f"SOUP_TEST_SELECTION {record}", markup=False, highlight=False, soft_wrap=True)
    cwd_prefix = None
    if os.path.realpath(Path.cwd()) != os.path.realpath(REPO_ROOT):
        cwd_prefix = REPO_ROOT
    Console().print(
        pytest_command(plan, shell=args.shell, cwd_prefix=cwd_prefix),
        markup=False,
        highlight=False,
        soft_wrap=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
