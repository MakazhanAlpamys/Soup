"""No module but ``utils/terminal.py`` may define its own control-strip table.

v0.75.0 added ``src/soup_cli/utils/terminal.py`` (``for_terminal``: strip C0/DEL
control bytes, then escape Rich markup) for the config loader, whose
unknown-key report printed YAML key names unescaped. Six command modules
(``commands/adapters.py``, ``data_canary.py``, ``data_doctor.py``, ``draft.py``,
``infer.py``, ``shrink.py``) plus ``utils/ship_verdict.py`` and
``mcp_server/registry.py`` carried a byte-identical private copy of the same
``_CONTROL_STRIP_TABLE`` under other names -- a seventh (and eighth) private
copy is exactly how the config loader got missed for a release (#896).

This is a repo-wide ratchet, not a one-time cleanup: it fails CI if an eighth
copy is ever reintroduced, mirroring ``test_no_foreign_license_headers.py``'s
approach of using ``git ls-files`` + AST inspection rather than trusting
review to catch a re-added duplicate.

**Name-based detection alone is not enough.** The ninth copy found by hand
while fixing #896 (``commands/serve.py``'s ``--auto-spec`` pairing message)
was a *function-local* variable named ``_ctrl``, not a module-scope
``_CONTROL_STRIP_TABLE`` -- a name-only ratchet locks the door that copy
walked through and leaves open the one it climbed in by (code review on #907).
So this file also matches the table's *shape*, at any scope, under any name:
a dict or dict-comprehension whose iterator is ``range(0x20)``. That pattern
is unique enough in this codebase to have exactly one match -- the real
definer -- confirmed by ``grep -rn "range(0x20)" src/soup_cli``.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "soup_cli"

#: The one file allowed to define the strip table.
ALLOWED_DEFINER = SRC_ROOT / "utils" / "terminal.py"

#: Names that would recreate the duplicated sanitizer if reintroduced as a
#: module-level assignment anywhere except ``ALLOWED_DEFINER``. Kept alongside
#: the shape check below as defence in depth (e.g. a table built via
#: ``dict(...)`` rather than a comprehension would skip the shape check but
#: still trip this).
BANNED_NAMES = frozenset({"_CONTROL_STRIP_TABLE"})


def _tracked_python_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "src/soup_cli"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        REPO_ROOT / line
        for line in out.stdout.splitlines()
        if line.endswith(".py")
    ]


def _module_level_assigned_names(text: str) -> set[str]:
    """Names bound by a top-level assignment (module scope only)."""
    tree = ast.parse(text)
    names: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                # e.g. ``_CONTROL_STRIP_TABLE[0x7F] = None`` -- still a
                # reintroduction of the table under that name.
                names.add(target.value.id)
    return names


def _is_range_0x20_call(node: ast.expr) -> bool:
    """``range(0x20)`` (or ``range(32)``) -- the shape's tell."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and len(node.args) >= 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 0x20
    )


def _control_strip_shape_lines(text: str) -> list[int]:
    """Line numbers of any dict/dict-comp, at ANY scope, shaped like the
    control-strip table: a dict comprehension iterating ``range(0x20)``.

    Unlike :func:`_module_level_assigned_names`, this walks the *entire*
    tree (``ast.walk``, not just ``tree.body``) precisely so a function-local
    copy -- the shape the real ninth copy in ``commands/serve.py`` took --
    is not exempt just because it never reaches module scope.
    """
    tree = ast.parse(text)
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.DictComp) and any(
            _is_range_0x20_call(gen.iter) for gen in node.generators
        ):
            lines.append(node.lineno)
    return lines


def _offenders() -> list[str]:
    offenders: list[str] = []
    for path in _tracked_python_files():
        if path.resolve() == ALLOWED_DEFINER.resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            banned_names = _module_level_assigned_names(text) & BANNED_NAMES
            shape_lines = _control_strip_shape_lines(text)
        except SyntaxError:  # pragma: no cover - not expected in this tree
            continue
        if not banned_names and not shape_lines:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        detail = []
        if banned_names:
            detail.append(f"defines {sorted(banned_names)}")
        if shape_lines:
            detail.append(
                "a range(0x20) dict-comp (the control-strip table's shape) "
                f"at line(s) {shape_lines}"
            )
        offenders.append(f"{rel}: " + "; ".join(detail))
    return offenders


class TestNoDuplicateTerminalSanitizer:
    def test_no_module_besides_terminal_defines_the_strip_table(self):
        offenders = _offenders()
        assert not offenders, (
            "These modules define their own copy of the control-strip table "
            "instead of importing soup_cli.utils.terminal.for_terminal / "
            "strip_control (#896):\n  " + "\n  ".join(offenders)
        )

    def test_the_scan_actually_covers_the_source_tree(self):
        """A scanner that silently stopped reading files would pass vacuously."""
        scanned = _tracked_python_files()
        assert len(scanned) > 300, (
            f"only {len(scanned)} files matched the scan; the tracked-file "
            "listing has broken"
        )
        names = {p.name for p in scanned}
        assert "terminal.py" in names
        assert "ship_verdict.py" in names
        assert "registry.py" in names


class TestTheScannerCanActuallyFail:
    """The repo is clean, so this is the only proof the scanner works."""

    def test_it_flags_a_reintroduced_module_scope_copy(self, tmp_path):
        offending = tmp_path / "some_command.py"
        offending.write_text(
            "_CONTROL_STRIP_TABLE = {i: None for i in range(0x20)}\n"
            "_CONTROL_STRIP_TABLE[0x7F] = None\n"
            "\n"
            "def _for_terminal(text):\n"
            "    return text.translate(_CONTROL_STRIP_TABLE)\n",
            encoding="utf-8",
        )
        text = offending.read_text(encoding="utf-8")
        assert _module_level_assigned_names(text) & BANNED_NAMES == BANNED_NAMES
        assert _control_strip_shape_lines(text)

    def test_it_flags_a_function_local_copy_under_a_different_name(self, tmp_path):
        """The exact shape the real ninth copy (``commands/serve.py``) took,
        and the gap code review on #907 asked this ratchet to close: a
        function-local variable, not module scope, named something other
        than ``_CONTROL_STRIP_TABLE``.
        """
        offending = tmp_path / "recipes.py"
        offending.write_text(
            "def _probe_local_copy(text: str) -> str:\n"
            "    _ctrl = {i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)}\n"
            "    return str(text).translate(_ctrl)\n",
            encoding="utf-8",
        )
        text = offending.read_text(encoding="utf-8")
        # The name-based check alone does not see this -- it is function-local
        # and not spelled `_CONTROL_STRIP_TABLE`, which is exactly the gap.
        assert _module_level_assigned_names(text) & BANNED_NAMES == set()
        # The shape check does.
        assert _control_strip_shape_lines(text) == [2]

    @pytest.mark.parametrize("name", sorted(BANNED_NAMES))
    def test_the_real_shared_definer_uses_the_banned_name_by_design(self, name):
        """Documents *why* ``ALLOWED_DEFINER`` is excluded, not just that it is."""
        text = ALLOWED_DEFINER.read_text(encoding="utf-8")
        assert name in _module_level_assigned_names(text)

    def test_the_real_shared_definer_has_the_banned_shape_by_design(self):
        text = ALLOWED_DEFINER.read_text(encoding="utf-8")
        assert _control_strip_shape_lines(text)
