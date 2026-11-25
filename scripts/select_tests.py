"""Print a focused local pytest command using the changed files and static imports."""

from __future__ import annotations

import argparse
import ast
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
MAX_FOCUSED_FILES = 100
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
LITERAL_IMPORTS = {"__import__", "import_module", "importorskip"}


class SelectionError(ValueError):
    """Local evidence is insufficient to suggest a focused run."""


@dataclass(frozen=True)
class TestPlan:
    paths: tuple[str, ...]
    reason: str | None = None


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


def _imported_modules(tree: ast.AST, module: str, is_package: bool) -> set[str]:
    """Find static imports and the literal form of common lazy import calls."""
    package = module if is_package else module.rpartition(".")[0]
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import("." * node.level + (node.module or ""), package)
            if base:
                names.add(base)
                names.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            callee = node.func
            kind = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
            first = node.args[0]
            if kind in LITERAL_IMPORTS and isinstance(first, ast.Constant):
                if isinstance(first.value, str):
                    names.add(_resolve_import(first.value, package))
    return names


def _reverse_imports(root: str, modules: dict[str, str]) -> dict[str, set[str]]:
    reverse = defaultdict(set)
    for name, relative in modules.items():
        path = _file_in_checkout(root, relative)
        try:
            tree = ast.parse(path.read_bytes(), filename=relative)
        except (OSError, SyntaxError, ValueError, RecursionError) as exc:
            raise SelectionError(f"could not parse {relative}") from exc
        for dependency in _imported_modules(tree, name, relative.endswith("/__init__.py")):
            if dependency in modules and dependency != name:
                reverse[dependency].add(name)
    return reverse


def _affected_modules(start: str, reverse: dict[str, set[str]]) -> set[str]:
    pending = [start]
    visited = set()
    while pending:
        current = pending.pop()
        if current not in visited:
            visited.add(current)
            pending.extend(reverse.get(current, ()))
    return visited


def _focused_paths(root: Path, changes: Iterable[str], max_files: int) -> tuple[str, ...]:
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

    modules = {}
    for path in sorted(_candidate_paths(root)):
        if not path.endswith(".py"):
            continue
        name = _module_name(path)
        if name in modules:
            raise SelectionError("multiple files have the same Python module name")
        modules[name] = path
    reverse = _reverse_imports(real_root, modules)
    test_stems = {
        name: stem for name, path in modules.items() if (stem := _test_stem(path)) is not None
    }
    selected = set()
    for path in selected_changes:
        module = _module_name(path)
        if modules.get(module) != path:
            raise SelectionError(f"no mapping for {path} in the checkout")
        consumers = _affected_modules(module, reverse)
        if any(modules[name].endswith("/conftest.py") for name in consumers):
            raise SelectionError(f"conftest.py depends on {path}")
        affected = consumers & test_stems.keys()
        if module not in test_stems:
            stem = PurePosixPath(path).stem.strip("_").replace("-", "_")
            affected.update(
                name for name, test_stem in test_stems.items() if f"_{stem}_" in f"_{test_stem}_"
            )
        if not affected:
            raise SelectionError(f"no related tests found for {path}")
        selected.update(modules[name] for name in affected)
        if len(selected) > max_files:
            raise SelectionError(f"more than {max_files} related test files")
    return tuple(sorted(selected))


def _full_plan(reason: str) -> TestPlan:
    return TestPlan(("tests/",), reason=reason)


def plan_tests(
    root: Path, changes: Iterable[str], *, max_files: int = MAX_FOCUSED_FILES
) -> TestPlan:
    """Select test files or explain why to use the full local suite."""
    try:
        return TestPlan(_focused_paths(root, changes, max_files))
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

    parser = argparse.ArgumentParser(description=__doc__)
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
    if plan.reason:
        err.print(f"Full suite: {plan.reason}.", markup=False, soft_wrap=True)
    else:
        err.print(f"Selected {len(plan.paths)} test files.", markup=False)
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
