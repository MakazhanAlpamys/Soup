"""Soup starts every external program by an absolute path, never by a bare name.

The runtime tests in ``tests/test_external_tool_paths.py`` pin this site by site. This
file is the source-wide first line behind them: it walks ``src/soup_cli`` and reports a
call that would start a program by its bare name or look one up on PATH by itself,
so a newly added call fails here before any per-site test exists for it. Like every
AST guard it sees only the spellings below, which is why the per-site tests stay.

Reported:

* a process start (``subprocess.run`` / ``Popen`` / ``call`` / ``check_call`` /
  ``check_output`` / ``getoutput`` / ``getstatusoutput``, ``asyncio``'s
  ``create_subprocess_exec``) whose program is a relative string constant: given as
  the argument itself, as the first item of a list or tuple literal (also one
  concatenated with more items), or through a local name assigned such a value;
* a PATH lookup of its own: ``shutil.which`` / ``find_executable``;
* an ``os`` start that searches PATH or runs a shell: ``os.exec*p*``,
  ``os.spawn*p*``, ``os.posix_spawnp``, ``os.system``, ``os.popen``.

A program is found with ``soup_cli.utils.tool_paths.find_tool`` / ``require_tool``
instead. Code that a generated script runs elsewhere is a string literal in this tree
and is not a call, so it is not scanned.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

from tests._source_scan import memoized_scan, parse_source

SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"

_PROCESS_STARTS = frozenset(
    {
        "run",
        "Popen",
        "call",
        "check_call",
        "check_output",
        "getoutput",
        "getstatusoutput",
        "create_subprocess_exec",
    }
)
_PATH_LOOKUPS = frozenset({"which", "find_executable"})
_OS_PATH_STARTS = re.compile(r"^(?:(?:exec|spawn)[lv]pe?|posix_spawnp)$")
_OS_SHELL_STARTS = frozenset({"system", "popen"})

#: (file under src/soup_cli, looked-up name) pairs a PATH lookup may use: each one
#: only displays what PATH holds and starts nothing.
ALLOWED_PATH_LOOKUPS: frozenset[tuple[str, str]] = frozenset(
    {
        # soup doctor names a ``python`` on PATH that is not the one running soup.
        ("commands/doctor.py", "python"),
        ("commands/doctor.py", "python3"),
    }
)


def _is_relative_program(value: object) -> bool:
    return isinstance(value, str) and not (value.startswith("/") or os.path.isabs(value))


class _Scan(ast.NodeVisitor):
    # visit_<NodeType> names are the ast.NodeVisitor callbacks, hence the N802 waivers.

    def __init__(self) -> None:
        self.findings: list[tuple[int, str]] = []
        self.os_names: set[str] = set()
        self.os_functions: dict[str, str] = {}
        self.scopes: list[dict[str, ast.AST | None]] = [{}]

    # -- what names mean ---------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name == "os":
                self.os_names.add(alias.asname or "os")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module == "os":
            for alias in node.names:
                self.os_functions[alias.asname or alias.name] = alias.name

    def _scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        # A parameter is computed by the caller: it hides an outer name of its own name.
        arguments = node.args
        names = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        names += [arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None]
        self.scopes.append({arg.arg: None for arg in names})
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._scoped(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self._scoped(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.scopes[-1][target.id] = node.value

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        self.generic_visit(node)
        if node.value is not None and isinstance(node.target, ast.Name):
            self.scopes[-1][node.target.id] = node.value

    def _value_of(self, name: str) -> ast.AST | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _program(self, node: ast.AST | None, depth: int = 0) -> str | None:
        """The relative program name ``node`` starts with, when it is a constant."""
        if node is None or depth > 4:
            return None
        if isinstance(node, ast.Constant):
            return node.value if _is_relative_program(node.value) else None
        if isinstance(node, ast.Name):
            return self._program(self._value_of(node.id), depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._program(node.left, depth + 1)
        if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
            first = node.elts[0]
            if isinstance(first, ast.Starred):
                return None
            return self._program(first, depth + 1)
        return None

    # -- the calls ---------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
            owner = func.value.id if isinstance(func.value, ast.Name) else None
            from_os = owner in self.os_names
        elif isinstance(func, ast.Name):
            name = self.os_functions.get(func.id, func.id)
            from_os = func.id in self.os_functions
        else:
            return
        if name in _PROCESS_STARTS:
            argument = node.args[0] if node.args else next(
                (kw.value for kw in node.keywords if kw.arg in ("args", "program")), None
            )
            program = self._program(argument)
            if program is not None:
                self.findings.append((node.lineno, f"{name}() starts {program!r} by name"))
        elif name in _PATH_LOOKUPS:
            looked_up = (
                node.args[0].value
                if node.args and isinstance(node.args[0], ast.Constant)
                else None
            )
            self.findings.append((node.lineno, f"{name}({looked_up!r}) searches PATH itself"))
        elif from_os and (_OS_PATH_STARTS.match(name) or name in _OS_SHELL_STARTS):
            self.findings.append((node.lineno, f"os.{name}() searches PATH or runs a shell"))


@memoized_scan
def bare_program_starts(source: str) -> list[tuple[int, str]]:
    """``(line, what)`` for each call in ``source`` that starts a program by name."""
    scan = _Scan()
    scan.visit(parse_source(source))
    return scan.findings


def _allowed(relative: str, what: str) -> bool:
    return any(
        relative == path and what.startswith(f"which({name!r})")
        for path, name in ALLOWED_PATH_LOOKUPS
    )


def _source_findings() -> list[str]:
    findings = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        for line, what in bare_program_starts(path.read_text(encoding="utf-8")):
            if not _allowed(relative, what):
                findings.append(f"src/soup_cli/{relative}:{line}: {what}")
    return findings


def test_no_source_file_starts_a_program_by_its_bare_name():
    findings = _source_findings()

    assert not findings, (
        "Start external programs by the absolute path that "
        "soup_cli.utils.tool_paths.find_tool / require_tool returns:\n" + "\n".join(findings)
    )


def test_every_allowed_lookup_is_still_in_the_source():
    """An allow-list entry whose call is gone is removed, not kept to excuse a new one."""
    for relative, name in sorted(ALLOWED_PATH_LOOKUPS):
        found = bare_program_starts((SRC / relative).read_text(encoding="utf-8"))
        assert any(what.startswith(f"which({name!r})") for _line, what in found), (
            relative,
            name,
        )


# ---------------------------------------------------------------------------
# The scanner itself: each spelling it must report, and what it must leave alone.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\nsubprocess.run(['git', 'status'])\n",
        "import subprocess\nsubprocess.run(('git', 'status'), check=True)\n",
        "import subprocess\nsubprocess.Popen(['git'] + extra)\n",
        "import subprocess\nsubprocess.check_output(args=['git', 'rev-parse'])\n",
        "import subprocess\nsubprocess.call('git status', shell=True)\n",
        "import subprocess\nsubprocess.getoutput('git status')\n",
        "import subprocess\ndef f():\n    cmd = ['git', 'status']\n    subprocess.run(cmd)\n",
        "import subprocess\ndef f():\n    cmd: list = ['git']\n    subprocess.check_call(cmd)\n",
        "import subprocess\ndef f():\n    tool = 'git'\n    subprocess.run([tool, 'status'])\n",
        "from subprocess import run\nrun(['git', 'status'])\n",
        "def f(run):\n    run(['gh', 'pr', 'view'])\n",
        "import asyncio\nasyncio.create_subprocess_exec('git', 'status')\n",
        "import subprocess\nsubprocess.run(['bin/tool'])\n",
        "import shutil\nshutil.which('git')\n",
        "import shutil as _shutil\n_shutil.which(name)\n",
        "from shutil import which\nwhich('docker')\n",
        "from distutils.spawn import find_executable\nfind_executable('git')\n",
        "import os\nos.execvp('accelerate', args)\n",
        "import os\nos.execlp('git', 'git')\n",
        "import os\nos.execvpe('git', args, env)\n",
        "import os\nos.spawnvp(os.P_WAIT, 'git', args)\n",
        "import os\nos.posix_spawnp('git', args, env)\n",
        "import os\nos.system('git status')\n",
        "import os as _os\n_os.popen('git status')\n",
        "from os import execvp\nexecvp('git', args)\n",
        "from os import system as sh\nsh('git status')\n",
    ],
)
def test_each_way_of_starting_a_program_by_name_is_reported(source):
    assert len(bare_program_starts(source)) == 1, source


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\nsubprocess.run(['/usr/sbin/sysctl', '-n', 'hw.memsize'])\n",
        "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip'])\n",
        "import subprocess\ndef f(git):\n    subprocess.run([git, 'status'])\n",
        "import subprocess\ndef f(argv):\n    subprocess.run(argv)\n",
        "import subprocess\ndef f(tail):\n    subprocess.run([*tail])\n",
        "import subprocess\ngit = 'git'\ndef f(git):\n    subprocess.run([git, 'status'])\n",
        "import os\nos.execv(path, args)\n",
        "import platform\nplatform.system()\n",
        "import asyncio\nasyncio.run(main())\n",
        "x = 'subprocess.run([\"git\", \"status\"])'\n",
    ],
)
def test_an_absolute_or_computed_program_is_not_reported(source):
    assert bare_program_starts(source) == []


def test_an_allowed_lookup_is_allowed_only_in_its_own_file():
    found = bare_program_starts("import shutil\nshutil.which('python')\n")

    assert _allowed("commands/doctor.py", found[0][1])
    assert not _allowed("utils/other.py", found[0][1])
    assert not _allowed(
        "commands/doctor.py", bare_program_starts("import shutil\nshutil.which('git')\n")[0][1]
    )
