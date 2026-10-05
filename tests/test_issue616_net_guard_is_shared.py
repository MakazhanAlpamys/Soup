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

import pytest

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
        rogue.write_text("_is_private_or_link_local = lambda host: False\n", encoding="utf-8")

        assert _definers(rogue) == ["_is_private_or_link_local"]

    def test_a_de_underscored_constant_is_a_builder(self, tmp_path) -> None:
        rogue = tmp_path / "rogue_renamed.py"
        rogue.write_text('_LOOPBACKHOSTS = frozenset({"localhost"})\n', encoding="utf-8")

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


# ---------------------------------------------------------------------------
# Outbound endpoint checks refuse non-public IP literals through ONE helper.
# ---------------------------------------------------------------------------

_REFUSAL = "refuse_private_ip_literal"

# Every check in front of an outbound request whose URL comes from a flag, a
# config file or a request body. Each must call the shared refusal, so deleting
# the call fails here as well as in tests/test_outbound_endpoint_ip_literals.py.
_OUTBOUND_GATES = (
    ("data/providers/vllm.py", "validate_vllm_url"),
    ("commands/generate.py", "_generate_openai"),
    ("commands/generate.py", "_generate_server"),
    ("eval/judge.py", "validate_judge_api_base"),
    ("eval/gate.py", "_valid_judge_url"),
    ("commands/ship.py", "_validate_judge_model_url"),
    ("ui/app.py", "chat_send"),
    ("config/schema.py", "_validate_online_dpo_judge_field"),
)

# Every OTHER function that reads a URL's hostname next to a loopback allowlist
# -- the shape every copy of the http-only check had -- and does not call the
# shared refusal, each with the reason. A NEW function of that shape fails
# test_every_hostname_check_is_accounted_for until it calls the refusal or is
# declared here. Calling is_private_or_link_local does not qualify a function
# on its own: a call made only on the plain-HTTP branch looks the same to an
# AST walk, so each entry is a statement that someone read the function.
_HOSTNAME_CHECKS_WITHOUT_THE_REFUSAL = {
    ("data/providers/ollama.py", "validate_ollama_url"): (
        "loopback only: any other host is refused outright"
    ),
    ("eval/gate.py", "_parse_judge_url"): (
        "a classifier; JudgeEvaluator enforces the policy via validate_judge_api_base"
    ),
    ("utils/loop_stages.py", "_endpoint_is_local"): (
        "the deploy-canary policy REQUIRES a local or LAN endpoint, on purpose"
    ),
    ("utils/qr_url.py", "build_phone_url"): "builds a URL to display; makes no request",
    ("utils/webhooks.py", "validate_webhook_url"): (
        "applies is_private_or_link_local itself, on every scheme"
    ),
    ("utils/tracing.py", "validate_otlp_endpoint"): (
        "applies is_private_or_link_local itself, on every scheme"
    ),
    ("utils/trackers.py", "_telemetry_endpoint_is_safe"): (
        "https only; applies is_private_or_link_local itself, before and after DNS"
    ),
    ("utils/hf.py", "resolve_endpoint"): (
        "HF_ENDPOINT is read only from the operator's own environment"
    ),
    ("utils/hubs.py", "validate_hub_endpoint"): (
        "hub endpoints are read only from the operator's own environment"
    ),
}


def _functions(path: Path) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(_parse(path))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _own_nodes(func: ast.AST) -> list[ast.AST]:
    """Nodes of ``func``'s own body, excluding nested defs, classes and lambdas."""
    found: list[ast.AST] = []
    stack = list(getattr(func, "body", []))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _called_names(func: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in _own_nodes(func):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _is_hostname_check(func: ast.AST) -> bool:
    """Reads ``<url>.hostname`` and holds a loopback allowlist in its own body."""
    nodes = _own_nodes(func)
    reads_hostname = any(isinstance(n, ast.Attribute) and n.attr == "hostname" for n in nodes)
    has_allowlist = any(
        (isinstance(n, ast.Constant) and n.value == "localhost")
        or (isinstance(n, ast.Name) and _normalised(n.id).endswith(_normalised(_SET_SUFFIX)))
        for n in nodes
    )
    return reads_hostname and has_allowlist


def _unguarded_hostname_checks(path: Path) -> list[str]:
    """THE discovery matcher, shared by the real-tree scan and its self-tests."""
    return [
        func.name
        for func in _functions(path)
        if _is_hostname_check(func) and _REFUSAL not in _called_names(func)
    ]


def _imports_the_refusal(path: Path) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "soup_cli.utils.net_guard"
        and any(alias.name == _REFUSAL for alias in node.names)
        for node in ast.walk(_parse(path))
    )


class TestOutboundGatesShareTheRefusal:
    def test_the_refusal_is_defined_only_in_the_shared_module(self) -> None:
        definers = [
            f"{path.relative_to(_SRC_ROOT).as_posix()}::{func.name}"
            for path in sorted(_SRC_ROOT.rglob("*.py"))
            for func in _functions(path)
            if func.name.endswith(_REFUSAL)
        ]
        assert definers == [f"{_HOME.relative_to(_SRC_ROOT).as_posix()}::{_REFUSAL}"]

    @pytest.mark.parametrize(("relpath", "func_name"), _OUTBOUND_GATES)
    def test_each_gate_calls_it(self, relpath: str, func_name: str) -> None:
        path = _SRC_ROOT / relpath
        funcs = [func for func in _functions(path) if func.name == func_name]
        assert len(funcs) == 1, f"{relpath}::{func_name} not found exactly once"
        assert _REFUSAL in _called_names(funcs[0]), (
            f"{relpath}::{func_name} must call net_guard.{_REFUSAL}"
        )
        assert _imports_the_refusal(path), f"{relpath} must import {_REFUSAL} from net_guard"

    def test_every_hostname_check_is_accounted_for(self) -> None:
        unaccounted = [
            f"{rel}::{name}"
            for path in sorted(_SRC_ROOT.rglob("*.py"))
            for rel in [path.relative_to(_SRC_ROOT).as_posix()]
            for name in _unguarded_hostname_checks(path)
            if (rel, name) not in _HOSTNAME_CHECKS_WITHOUT_THE_REFUSAL
        ]
        assert unaccounted == [], (
            "these read a URL's hostname next to a loopback allowlist without calling "
            f"net_guard.{_REFUSAL}; call it, or declare them in "
            f"_HOSTNAME_CHECKS_WITHOUT_THE_REFUSAL with the reason: {unaccounted}"
        )

    @pytest.mark.parametrize(("relpath", "func_name"), sorted(_HOSTNAME_CHECKS_WITHOUT_THE_REFUSAL))
    def test_every_declared_exemption_is_still_needed(self, relpath: str, func_name: str) -> None:
        """A renamed, rewritten or since-fixed entry must leave the table, not rot in it."""
        funcs = [func for func in _functions(_SRC_ROOT / relpath) if func.name == func_name]
        assert len(funcs) == 1 and _is_hostname_check(funcs[0]), (
            f"{relpath}::{func_name} is no longer a hostname check; drop its entry"
        )
        assert _REFUSAL not in _called_names(funcs[0]), (
            f"{relpath}::{func_name} calls {_REFUSAL} now; drop its entry"
        )


class TestTheDiscoveryCatchesANewCopy:
    """Demonstrates, not just asserts: both spellings of the old check are caught."""

    _BODY = (
        "from urllib.parse import urlparse\n"
        "{imports}"
        "\n\ndef validate_new_url(url):\n"
        "    parsed = urlparse(url)\n"
        "    if parsed.hostname not in {allowlist} and parsed.scheme != 'https':\n"
        "        raise ValueError('HTTPS for remote')\n"
        "{extra}"
    )

    def _write(self, tmp_path: Path, **parts: str) -> Path:
        values = {"imports": "", "allowlist": "('localhost', '127.0.0.1')", "extra": ""}
        values.update(parts)
        rogue = tmp_path / "rogue_gate.py"
        rogue.write_text(self._BODY.format(**values), encoding="utf-8")
        return rogue

    def test_a_literal_allowlist_copy_is_caught(self, tmp_path: Path) -> None:
        assert _unguarded_hostname_checks(self._write(tmp_path)) == ["validate_new_url"]

    def test_a_shared_set_copy_is_caught(self, tmp_path: Path) -> None:
        rogue = self._write(
            tmp_path,
            imports="from soup_cli.utils.net_guard import LOOPBACK_HOSTS\n",
            allowlist="LOOPBACK_HOSTS",
        )
        assert _unguarded_hostname_checks(rogue) == ["validate_new_url"]

    def test_a_copy_that_calls_the_refusal_is_not(self, tmp_path: Path) -> None:
        rogue = self._write(
            tmp_path,
            imports="from soup_cli.utils.net_guard import refuse_private_ip_literal\n",
            extra="    refuse_private_ip_literal(parsed.hostname, label='x')\n",
        )
        assert _unguarded_hostname_checks(rogue) == []

    def test_a_predicate_call_on_the_http_branch_alone_is_still_caught(
        self, tmp_path: Path
    ) -> None:
        """Why a call to the predicate does not exempt a function by itself."""
        rogue = tmp_path / "rogue_http_branch.py"
        rogue.write_text(
            "from urllib.parse import urlparse\n"
            "from soup_cli.utils.net_guard import LOOPBACK_HOSTS, is_private_or_link_local\n"
            "\n\ndef validate_new_url(url):\n"
            "    parsed = urlparse(url)\n"
            "    if parsed.scheme == 'http' and parsed.hostname not in LOOPBACK_HOSTS:\n"
            "        if is_private_or_link_local(parsed.hostname):\n"
            "            raise ValueError('private hosts require HTTPS')\n"
            "        raise ValueError('HTTPS for remote')\n",
            encoding="utf-8",
        )
        assert _unguarded_hostname_checks(rogue) == ["validate_new_url"]
