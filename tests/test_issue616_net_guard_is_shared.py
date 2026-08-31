"""#616 — the SSRF loopback/private-host predicate must have ONE definition.

``_is_private_or_link_local`` was copied into ``hf.py`` and ``hubs.py``
(``_LOOPBACK_HOSTS`` into those two plus ``loop_stages.py`` and ``qr_url.py``),
the same shape that already cost this repo a fix landing in one copy and not
the others three times (#372, #392, #424). Both now live once in
``utils/net_guard.py``; the four call sites import them under their original
private names so no caller changes.

Mirrors the shape of ``tests/test_issue382_windows_ci_guard_is_shared.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "utils"
_HOME = _SRC_DIR / "net_guard.py"

_CONSUMERS = ("hf.py", "hubs.py", "loop_stages.py", "qr_url.py")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _assigns_name(node: ast.AST, name: str) -> bool:
    """True if ``node`` is a plain or annotated assignment to ``name``.

    Covers both ``NAME = ...`` (``ast.Assign``) and ``NAME: T = ...``
    (``ast.AnnAssign``) — a duplicate definition doesn't care which form it
    takes.
    """
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    return False


def _imports_net_guard(path: Path) -> bool:
    """True if ``path`` has a real ``from soup_cli.utils.net_guard import ...``.

    Parses the AST rather than substring-matching the source, so a comment
    or string literal mentioning the module name can't fake a pass.
    """
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.ImportFrom) and node.module == "soup_cli.utils.net_guard":
            return True
    return False


class TestThereIsExactlyOneDefinition:
    def test_the_predicate_is_defined_only_in_the_shared_module(self):
        """A second ``def is_private_or_link_local`` anywhere is the drift."""
        definers = []
        for path in sorted(_SRC_DIR.rglob("*.py")):
            for node in ast.walk(_parse(path)):
                if isinstance(node, ast.FunctionDef) and node.name == "is_private_or_link_local":
                    definers.append(path.name)
        assert definers == [_HOME.name], (
            f"is_private_or_link_local must be defined only in {_HOME.name}; found in {definers}"
        )

    def test_the_loopback_set_is_built_only_in_the_shared_module(self):
        builders = []
        for path in sorted(_SRC_DIR.rglob("*.py")):
            for node in ast.walk(_parse(path)):
                if _assigns_name(node, "LOOPBACK_HOSTS"):
                    builders.append(path.name)
        assert builders == [_HOME.name], (
            f"LOOPBACK_HOSTS must be built only in {_HOME.name}; found in {builders}"
        )

    def test_all_known_call_sites_import_rather_than_redeclare(self):
        for name in _CONSUMERS:
            assert _imports_net_guard(_SRC_DIR / name), f"{name} must import the shared guard"


class TestNoCallerRedeclaresTheOldPrivateNames:
    """A caller re-adding its own ``_LOOPBACK_HOSTS = frozenset(...)`` line
    would silently reintroduce a fifth copy without tripping the ast-walk
    above (it targets the new public names, not the old private aliases)."""

    def test_no_consumer_assigns_its_own_loopback_frozenset(self):
        for name in _CONSUMERS:
            for node in ast.walk(_parse(_SRC_DIR / name)):
                if _assigns_name(node, "_LOOPBACK_HOSTS"):
                    raise AssertionError(
                        f"{name} must import _LOOPBACK_HOSTS from net_guard, not assign it"
                    )
