"""#1548 — one loopback set for every outbound gate.

``http://[::1]:8000`` was accepted by the judge API-base and vLLM checks but
refused by ``GateTask.judge_model`` (``eval/gate.py``), ``soup ship
--judge-model`` (``commands/ship.py``) and the online-DPO trainer's
``_parse_judge_url`` — each kept its own ``("localhost", "127.0.0.1")``
literal that predates the ``::1`` entry of the shared set. ``vllm.py`` and
``eval/judge.py`` kept a third spelling of the same literal.

Every gate below now imports ``net_guard.LOOPBACK_HOSTS`` instead of
rebuilding it, so ``::1`` is decided once. The structural tests pin that
with an AST walk (a comment or message string mentioning ``localhost``
can't trip them — only a real collection literal of host strings can),
and the behaviour tests pin ``http://[::1]:<port>`` acceptance per gate.

Mirrors the shape of ``tests/test_issue616_net_guard_is_shared.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "soup_cli"

# Files that gate an outbound plain-HTTP URL on loopback membership.
_GATE_FILES = (
    "eval/gate.py",
    "commands/ship.py",
    "data/providers/vllm.py",
    "eval/judge.py",
)

# Uses the shared ``_parse_judge_url`` from eval.gate — must not grow its own.
_TRAINER_FILE = "trainer/online_dpo.py"

_LOOPBACK_STRINGS = frozenset({"localhost", "127.0.0.1", "::1"})


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _loopback_literals(path: Path) -> list[str]:
    """Locations of collection literals holding 2+ loopback host strings.

    Catches ``("localhost", "127.0.0.1")``, ``{"localhost", "127.0.0.1",
    "::1"}``, ``frozenset([...])`` and friends — any shape a rebuilt
    loopback set actually takes. Message strings such as ``"use ollama://,
    https://, or http://localhost"`` hold a single host and are ignored.
    """
    found: list[str] = []
    for node in ast.walk(_parse(path)):
        elts: list[ast.expr] | None = None
        if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
            elts = node.elts
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("frozenset", "tuple", "set", "list")
            and node.args
            and isinstance(node.args[0], (ast.Tuple, ast.Set, ast.List))
        ):
            elts = node.args[0].elts
        if elts is None:
            continue
        hosts = {
            elt.value
            for elt in elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
        if len(hosts & _LOOPBACK_STRINGS) >= 2:
            found.append(f"{path.relative_to(_SRC_ROOT.parent)}:{node.lineno}")
    return found


def _imports_loopback_hosts(path: Path) -> bool:
    """True if ``path`` really imports LOOPBACK_HOSTS from net_guard."""
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "soup_cli.utils.net_guard"
        and any(alias.name == "LOOPBACK_HOSTS" for alias in node.names)
        for node in ast.walk(_parse(path))
    )


class TestNoLoopbackLiteralsInGates:
    @pytest.mark.parametrize("rel", _GATE_FILES)
    def test_no_rebuilt_loopback_set(self, rel: str):
        path = _SRC_ROOT / rel
        assert _loopback_literals(path) == [], (
            f"{rel} rebuilds the loopback set instead of importing "
            f"net_guard.LOOPBACK_HOSTS: {_loopback_literals(path)}"
        )

    @pytest.mark.parametrize("rel", _GATE_FILES)
    def test_imports_shared_loopback_hosts(self, rel: str):
        path = _SRC_ROOT / rel
        assert _imports_loopback_hosts(path), (
            f"{rel} must import LOOPBACK_HOSTS from soup_cli.utils.net_guard"
        )

    def test_trainer_uses_shared_parse_judge_url(self):
        """online_dpo must not define its own _parse_judge_url or literal."""
        path = _SRC_ROOT / _TRAINER_FILE
        tree = _parse(path)
        defined = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_parse_judge_url"
        ]
        assert defined == [], f"{_TRAINER_FILE} defines its own _parse_judge_url"
        imported = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "soup_cli.eval.gate"
            and any(alias.name == "_parse_judge_url" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert imported, f"{_TRAINER_FILE} must use eval.gate._parse_judge_url"
        assert _loopback_literals(path) == []


def _gate_task(judge_model: str):
    from soup_cli.eval.gate import GateTask

    return GateTask(name="t", type="judge", threshold=0.5, judge_model=judge_model)


class TestIpv6LoopbackAcceptedEverywhere:
    """``http://[::1]:<port>`` behaves the same in all gates — accepted."""

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
    def test_gate_task_validator(self, host: str):
        task = _gate_task(f"http://{host}:8000/Qwen2.5")
        assert task.judge_model == f"http://{host}:8000/Qwen2.5"

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
    def test_parse_judge_url(self, host: str):
        from soup_cli.eval.gate import _parse_judge_url

        assert _parse_judge_url(f"http://{host}:8000/Qwen2.5") == (
            "server",
            "Qwen2.5",
            f"http://{host}:8000",
        )

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
    def test_ship_judge_model_url(self, host: str):
        from soup_cli.commands.ship import _validate_judge_model_url

        assert _validate_judge_model_url(f"http://{host}:8000/Qwen2.5") is None

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
    def test_vllm_url(self, host: str):
        from soup_cli.data.providers.vllm import validate_vllm_url

        assert validate_vllm_url(f"http://{host}:8000") is None

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
    def test_judge_api_base(self, host: str):
        from soup_cli.eval.judge import validate_judge_api_base

        assert validate_judge_api_base(f"http://{host}:8000") is None


class TestGuardsStillRefuse:
    """Unifying the set must not loosen the SSRF guards."""

    def test_remote_http_still_refused(self):
        from soup_cli.eval.gate import _parse_judge_url

        with pytest.raises(ValueError, match="unsupported scheme"):
            _parse_judge_url("http://example.com/model")

    def test_prefix_bypass_still_refused(self):
        import typer

        from soup_cli.commands.ship import _validate_judge_model_url

        with pytest.raises(typer.Exit):
            _validate_judge_model_url("http://localhost.attacker.com/model")

    def test_private_ip_literal_still_refused(self):
        from soup_cli.eval.judge import validate_judge_api_base

        with pytest.raises(ValueError, match="not allowed"):
            validate_judge_api_base("https://10.0.0.5:8000")
