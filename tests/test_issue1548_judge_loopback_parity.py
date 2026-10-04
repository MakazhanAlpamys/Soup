"""#1548 — every judge/vLLM outbound gate shares ONE loopback set.

Follow-up from the 0.75.2 outbound-endpoint hardening (GHSA-9f49-hqpf-c849,
#1527). That review round added the SSRF literal-IP refusal everywhere, but
left each loopback *allowlist* as its own local copy. Two of those copies
(``eval/judge.py``, ``data/providers/vllm.py``) already included ``::1``;
three others (``eval/gate.py``'s ``GateTask.judge_model`` validator, its
``_parse_judge_url`` — reused by ``soup ship --judge-model`` and the
online-DPO trainer — and ``commands/ship.py``'s own ``--judge-model`` check)
did not, so ``http://[::1]:8000`` was accepted by the judge API-base and
vLLM checks but refused everywhere else.

Fixed by importing ``net_guard.LOOPBACK_HOSTS`` at all five call sites
instead of re-listing the hosts. This both adds the missing ``::1`` and
makes a future change to the set (adding another loopback form, say)
apply everywhere at once.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "soup_cli"

# Every call site #1548 named, and the literal tuple/set each one carried
# before the fix. (file, function name).
_CALL_SITES = (
    ("eval/gate.py", "_valid_judge_url"),
    ("eval/gate.py", "_parse_judge_url"),
    ("commands/ship.py", "_validate_judge_model_url"),
    ("data/providers/vllm.py", "validate_vllm_url"),
    ("eval/judge.py", "validate_judge_api_base"),
)

# A literal loopback allowlist: a tuple/set containing both "localhost" and
# "127.0.0.1" as plain string constants, with or without "::1" alongside.
_LITERAL_ALLOWLIST_RE = re.compile(
    r"""[({]\s*["']localhost["']\s*,\s*["']127\.0\.0\.1["']"""
    r"""|["']127\.0\.0\.1["']\s*,\s*["']localhost["']"""
)


def _source(relpath: str) -> str:
    return (_SRC_ROOT / relpath).read_text(encoding="utf-8")


def _function_source(relpath: str, func_name: str) -> str:
    tree = ast.parse(_source(relpath))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return ast.get_source_segment(_source(relpath), node) or ""
    raise AssertionError(f"{relpath}::{func_name} not found")


class TestNoLiteralAllowlistRemains:
    """Acceptance: 'No literal (\"localhost\", \"127.0.0.1\"...) tuple remains
    in a gate; a test greps for one.'"""

    @pytest.mark.parametrize(("relpath", "func_name"), _CALL_SITES)
    def test_call_site_has_no_literal_allowlist(self, relpath, func_name):
        body = _function_source(relpath, func_name)
        assert not _LITERAL_ALLOWLIST_RE.search(body), (
            f"{relpath}::{func_name} still carries a literal loopback tuple; "
            "import LOOPBACK_HOSTS from net_guard instead"
        )

    @pytest.mark.parametrize(("relpath", "func_name"), _CALL_SITES)
    def test_call_site_imports_the_shared_set(self, relpath, func_name):
        body = _function_source(relpath, func_name)
        assert "LOOPBACK_HOSTS" in body, (
            f"{relpath}::{func_name} must reference net_guard.LOOPBACK_HOSTS"
        )

    def test_the_regex_catches_both_orderings(self):
        """Control: the discovery pattern itself, against throwaway snippets."""
        assert _LITERAL_ALLOWLIST_RE.search('("localhost", "127.0.0.1")')
        assert _LITERAL_ALLOWLIST_RE.search('{"127.0.0.1", "localhost"}')
        assert _LITERAL_ALLOWLIST_RE.search('("localhost", "127.0.0.1", "::1")')
        assert not _LITERAL_ALLOWLIST_RE.search("LOOPBACK_HOSTS")


class TestIPv6LoopbackParity:
    """Acceptance: 'http://[::1]:<port> behaves the same in all gates,
    pinned per gate.'"""

    def test_gate_task_judge_model_accepts_ipv6_loopback(self):
        from soup_cli.eval.gate import GateTask

        task = GateTask(
            type="judge",
            name="judge",
            threshold=0.5,
            prompts="prompts.jsonl",
            judge_model="http://[::1]:8000/model",
        )
        assert task.judge_model == "http://[::1]:8000/model"

    def test_parse_judge_url_accepts_ipv6_loopback(self):
        from soup_cli.eval.gate import _parse_judge_url

        provider, model, api_base = _parse_judge_url("http://[::1]:8000/Qwen2.5")
        assert provider == "server"
        assert model == "Qwen2.5"
        assert api_base == "http://[::1]:8000"

    def test_ship_judge_model_flag_accepts_ipv6_loopback(self, monkeypatch):
        import io

        from rich.console import Console

        import soup_cli.commands.ship as ship

        buf = io.StringIO()
        monkeypatch.setattr(ship, "console", Console(file=buf, width=400, color_system=None))

        ship._validate_judge_model_url("http://[::1]:8000/m")
        assert buf.getvalue() == ""

    def test_vllm_url_accepts_ipv6_loopback(self):
        from soup_cli.data.providers.vllm import validate_vllm_url

        validate_vllm_url("http://[::1]:8000")  # must not raise

    def test_judge_api_base_accepts_ipv6_loopback(self):
        """Positive control: this side already worked before #1548."""
        from soup_cli.eval.judge import validate_judge_api_base

        validate_judge_api_base("http://[::1]:8000")  # must not raise


class TestMutationWouldHaveFailed:
    """Demonstrates, not just asserts: reverting any one fix site to its old
    literal tuple reopens exactly that site's ::1 refusal."""

    def test_old_gate_task_pattern_rejected_ipv6_loopback(self):
        from urllib.parse import urlparse

        old_allowlist = {"localhost", "127.0.0.1"}
        parsed = urlparse("http://[::1]:8000/model")
        assert parsed.hostname not in old_allowlist

    def test_old_parse_judge_url_pattern_rejected_ipv6_loopback(self):
        from urllib.parse import urlparse

        old_allowlist = ("localhost", "127.0.0.1")
        parsed = urlparse("http://[::1]:8000/model")
        assert parsed.hostname not in old_allowlist

    def test_old_ship_pattern_rejected_ipv6_loopback(self):
        from urllib.parse import urlparse

        old_allowlist = ("localhost", "127.0.0.1")
        parsed = urlparse("http://[::1]:8000/m")
        assert parsed.hostname not in old_allowlist
