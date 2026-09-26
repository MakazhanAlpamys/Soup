"""#616 — the SSRF loopback/private-host predicate must have ONE definition.

The original count was wrong: this repo had **three** copies of the predicate
(``hf.py``, ``hubs.py``, ``webhooks.py`` — the last as
``_is_private_or_link_local`` too) and **six** of the loopback set
(those three plus ``loop_stages.py``, ``qr_url.py``, and ``tracing.py``,
which named its copy of the predicate ``_is_private_ip`` instead). All are
now imports of ``utils/net_guard.py``'s ``is_private_or_link_local`` /
``LOOPBACK_HOSTS`` under each call site's original private name.

A first version of this guard matched only the exact names
``is_private_or_link_local`` / ``LOOPBACK_HOSTS`` — which never fires on a
duplicate written the way every original copy actually was: with a leading
underscore. Every symbol check below therefore matches by **suffix**, so
``_is_private_or_link_local``, ``is_private_or_link_local``, and (in
principle) some future ``new_is_private_or_link_local`` are all caught.
``tracing.py``'s ``_is_private_ip`` is a different name, not a duplicate
definition — it isn't a substring of ``is_private_or_link_local`` and is
correctly ignored by the suffix check; it's covered instead by the
import-not-redeclare check, keyed off the six known call sites.

Mirrors the shape of ``tests/test_issue382_windows_ci_guard_is_shared.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "utils"
_SRC_ROOT = _UTILS_DIR.parent
_HOME = _UTILS_DIR / "net_guard.py"

_FUNC_SUFFIX = "is_private_or_link_local"
_SET_SUFFIX = "LOOPBACK_HOSTS"

_CONSUMERS = ("hf.py", "hubs.py", "loop_stages.py", "qr_url.py", "webhooks.py", "tracing.py")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _assigned_names(node: ast.AST) -> list[str]:
    """Names ``node`` assigns to, covering both plain and annotated forms.

    ``NAME = ...`` (``ast.Assign``) and ``NAME: T = ...`` (``ast.AnnAssign``)
    are equally valid ways to reintroduce a duplicate constant.
    """
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _imports_net_guard(path: Path) -> bool:
    """True if ``path`` has a real ``from soup_cli.utils.net_guard import ...``.

    Parses the AST rather than substring-matching the source, so a comment
    or string literal mentioning the module name can't fake a pass.
    """
    return any(
        isinstance(node, ast.ImportFrom) and node.module == "soup_cli.utils.net_guard"
        for node in ast.walk(_parse(path))
    )


def _normalised(name: str) -> str:
    """Underscore- and case-insensitive form of an identifier.

    `_LOOPBACK_HOSTS`, `LOOPBACK_HOSTS` and `_LOOPBACKHOSTS` are the same
    constant wearing three spellings, and a suffix match on the literal name
    caught only the first two. Dropping one underscore is not a disguise
    anyone adopts on purpose -- it is what a rename produces.
    """
    return name.replace("_", "").upper()


def _definers(path: Path) -> list[str]:
    """Every function in ``path`` whose name ends in the predicate suffix.

    THE matcher. Both the real whole-tree scan and the regression self-tests
    below call this, so a future edit that narrows the match (an exact-name
    comparison, say -- the shape that let a leading-underscore duplicate
    through before #625) fails both at once. The first version of those
    self-tests reimplemented this walk inline, which meant they kept passing
    on a guard that had gone blind: the same defect this repository already
    recorded once, under BUNDLED_SCORER_FINGERPRINT.
    """
    found: list[str] = []
    for node in ast.walk(_parse(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.endswith(_FUNC_SUFFIX):
                found.append(node.name)
            continue
        # A `def` is not the only way to define a predicate. `NAME = lambda h:
        # ...` binds the same name and was invisible to a FunctionDef-only
        # walk -- and an assigned lambda is precisely what someone reaches for
        # when moving a small function between modules.
        for name in _assigned_names(node):
            if name.endswith(_FUNC_SUFFIX):
                found.append(name)
    return found


def _builders(path: Path) -> list[str]:
    """Every name assigned in ``path`` ending in the host-set suffix."""
    return [
        name
        for node in ast.walk(_parse(path))
        for name in _assigned_names(node)
        if _normalised(name).endswith(_normalised(_SET_SUFFIX))
    ]


class TestThereIsExactlyOneDefinition:
    """Suffix match, not exact match: a duplicate is a duplicate whether or
    not it carries the private-helper leading underscore — every one of the
    six original copies did, so matching only the bare name misses the
    shape a fifth copy would actually take."""

    def test_the_predicate_is_defined_only_in_the_shared_module(self):
        definers = [
            f"{path.relative_to(_SRC_ROOT)}::{name}"
            for path in sorted(_SRC_ROOT.rglob("*.py"))
            for name in _definers(path)
        ]
        assert definers == [f"{_HOME.relative_to(_SRC_ROOT)}::{_FUNC_SUFFIX}"], (
            f"a function ending in {_FUNC_SUFFIX!r} must be defined only in "
            f"{_HOME.name}; found: {definers}"
        )

    def test_the_loopback_set_is_built_only_in_the_shared_module(self):
        builders = [
            f"{path.relative_to(_SRC_ROOT)}::{name}"
            for path in sorted(_SRC_ROOT.rglob("*.py"))
            for name in _builders(path)
        ]
        assert builders == [f"{_HOME.relative_to(_SRC_ROOT)}::{_SET_SUFFIX}"], (
            f"a name ending in {_SET_SUFFIX!r} must be built only in "
            f"{_HOME.name}; found: {builders}"
        )

    def test_all_known_call_sites_import_rather_than_redeclare(self):
        for name in _CONSUMERS:
            assert _imports_net_guard(_UTILS_DIR / name), f"{name} must import the shared guard"


class TestShapesTheFirstGuardCouldNotSee:
    """Closed by the maintainer after #625 merged, having left them open once."""

    def test_an_assigned_lambda_is_a_definition(self, tmp_path) -> None:
        rogue = tmp_path / "rogue_lambda.py"
        rogue.write_text(
            "_is_private_or_link_local = lambda host: False\n", encoding="utf-8"
        )

        assert _definers(rogue) == ["_is_private_or_link_local"]

    def test_a_de_underscored_constant_is_a_builder(self, tmp_path) -> None:
        rogue = tmp_path / "rogue_renamed.py"
        rogue.write_text(
            '_LOOPBACKHOSTS = frozenset({"localhost"})\n', encoding="utf-8"
        )

        assert _builders(rogue) == ["_LOOPBACKHOSTS"]

    def test_the_widening_does_not_fire_on_the_real_tree(self) -> None:
        """The control. A guard that flags correct code is one people delete."""
        definers = [
            f"{path.relative_to(_SRC_ROOT)}::{name}"
            for path in sorted(_SRC_ROOT.rglob("*.py"))
            for name in _definers(path)
        ]
        builders = [
            f"{path.relative_to(_SRC_ROOT)}::{name}"
            for path in sorted(_SRC_ROOT.rglob("*.py"))
            for name in _builders(path)
        ]

        assert definers == [f"{_HOME.relative_to(_SRC_ROOT)}::{_FUNC_SUFFIX}"]
        assert builders == [f"{_HOME.relative_to(_SRC_ROOT)}::{_SET_SUFFIX}"]


class TestMutationsThatUsedToSurvive:
    """Demonstrates, not just asserts: these three specific shapes passed
    the pre-fix version of this guard. Reproduced here (against a throwaway
    module, not the real tree) so a future edit to the matching logic that
    reopens one of them fails loudly instead of silently."""

    def test_underscore_prefixed_function_is_still_caught(self, tmp_path):
        rogue = tmp_path / "rogue_predicate.py"
        rogue.write_text("def _is_private_or_link_local(host):\n    return False\n")
        found = _definers(rogue)
        assert found == ["_is_private_or_link_local"]

    def test_underscore_prefixed_constant_is_still_caught(self, tmp_path):
        rogue = tmp_path / "rogue_constant.py"
        rogue.write_text('_LOOPBACK_HOSTS = frozenset({"localhost"})\n')
        found = _builders(rogue)
        assert found == ["_LOOPBACK_HOSTS"]

    def test_annotated_constant_is_still_caught(self, tmp_path):
        rogue = tmp_path / "rogue_annotated.py"
        rogue.write_text('LOOPBACK_HOSTS: frozenset = frozenset({"localhost"})\n')
        found = _builders(rogue)
        assert found == ["LOOPBACK_HOSTS"]
