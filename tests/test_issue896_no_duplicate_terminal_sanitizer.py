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
#: module-level assignment anywhere except ``ALLOWED_DEFINER``.
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


def _offenders() -> list[str]:
    offenders: list[str] = []
    for path in _tracked_python_files():
        if path.resolve() == ALLOWED_DEFINER.resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            found = _module_level_assigned_names(text) & BANNED_NAMES
        except SyntaxError:  # pragma: no cover - not expected in this tree
            continue
        if found:
            rel = path.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{rel} defines {sorted(found)}")
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
        assert len(scanned) > 50, (
            f"only {len(scanned)} files matched the scan; the tracked-file "
            "listing has broken"
        )
        names = {p.name for p in scanned}
        assert "terminal.py" in names
        assert "ship_verdict.py" in names
        assert "registry.py" in names


class TestTheScannerCanActuallyFail:
    """The repo is clean, so this is the only proof the scanner works."""

    def test_it_flags_a_reintroduced_copy(self, tmp_path):
        offending = tmp_path / "some_command.py"
        offending.write_text(
            "_CONTROL_STRIP_TABLE = {i: None for i in range(0x20)}\n"
            "_CONTROL_STRIP_TABLE[0x7F] = None\n"
            "\n"
            "def _for_terminal(text):\n"
            "    return text.translate(_CONTROL_STRIP_TABLE)\n",
            encoding="utf-8",
        )
        found = _module_level_assigned_names(offending.read_text(encoding="utf-8"))
        assert found & BANNED_NAMES == BANNED_NAMES

    @pytest.mark.parametrize("name", sorted(BANNED_NAMES))
    def test_the_real_shared_definer_uses_the_banned_name_by_design(self, name):
        """Documents *why* ``ALLOWED_DEFINER`` is excluded, not just that it is."""
        text = ALLOWED_DEFINER.read_text(encoding="utf-8")
        assert name in _module_level_assigned_names(text)
