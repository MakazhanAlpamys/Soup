"""#1548: every outbound gate that allows plain HTTP to loopback uses one set.

``http://[::1]:<port>`` was accepted by the judge API-base and vLLM checks but
refused by ``GateTask.judge_model``, ``soup ship --judge-model`` and
``_parse_judge_url``; two modules also kept literal copies of
``net_guard.LOOPBACK_HOSTS``. The gates now import the one set and treat
``::1`` alike.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"

# The gates named in #1548: each allowed plain HTTP to a hand-written loopback set.
GATE_FILES = [
    SRC / "eval" / "gate.py",
    SRC / "commands" / "ship.py",
    SRC / "data" / "providers" / "vllm.py",
    SRC / "eval" / "judge.py",
]

# A tuple/set/list literal that spells out a loopback host, e.g.
# ("localhost", "127.0.0.1") or {"localhost", "127.0.0.1", "::1"}.
_LITERAL_LOOPBACK = re.compile(r'[(\[{]\s*"(?:localhost|127\.0\.0\.1|::1)"\s*,')

V6 = "http://[::1]:8000"
V6_JUDGE = "http://[::1]:8000/Qwen2.5"  # judge_model URLs carry the model id last
REMOTE = "http://example.com:8000"


@pytest.mark.parametrize("path", GATE_FILES, ids=lambda p: p.name)
def test_no_literal_loopback_set_in_gate(path: Path) -> None:
    """No gate keeps its own copy of the loopback hosts."""
    hits = [
        f"{path.name}:{n}: {line.strip()}"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _LITERAL_LOOPBACK.search(line)
    ]
    msg = "literal loopback set in a gate (use net_guard.LOOPBACK_HOSTS):\n"
    assert not hits, msg + "\n".join(hits)


@pytest.mark.parametrize("path", GATE_FILES, ids=lambda p: p.name)
def test_gate_imports_the_shared_set(path: Path) -> None:
    """Each gate takes the set from ``net_guard``."""
    text = path.read_text(encoding="utf-8")
    assert "LOOPBACK_HOSTS" in text and "soup_cli.utils.net_guard" in text, path


def test_gate_task_judge_model_accepts_ipv6_loopback() -> None:
    """``GateTask.judge_model`` takes ``http://[::1]`` like ``http://127.0.0.1``."""
    from soup_cli.eval.gate import GateTask

    task = GateTask(name="t", type="judge", threshold=0.5, judge_model=V6_JUDGE)
    assert task.judge_model == V6_JUDGE


def test_parse_judge_url_accepts_ipv6_loopback() -> None:
    """``_parse_judge_url`` resolves ``http://[::1]`` to the server provider."""
    from soup_cli.eval.gate import _parse_judge_url

    provider, model, api_base = _parse_judge_url(V6_JUDGE)
    assert (provider, model, api_base) == ("server", "Qwen2.5", V6)


def test_ship_judge_model_accepts_ipv6_loopback() -> None:
    """``soup ship --judge-model http://[::1]`` passes the URL check."""
    from soup_cli.commands.ship import _validate_judge_model_url

    _validate_judge_model_url(V6)  # must not exit


def test_vllm_url_accepts_ipv6_loopback() -> None:
    """vLLM keeps accepting ``http://[::1]``, now through the shared set."""
    from soup_cli.data.providers.vllm import validate_vllm_url

    validate_vllm_url(V6)


def test_judge_api_base_accepts_ipv6_loopback() -> None:
    """The judge API-base keeps accepting ``http://[::1]``, through the shared set."""
    from soup_cli.eval.judge import validate_judge_api_base

    validate_judge_api_base(V6)


def _gate_task(url: str) -> None:
    from soup_cli.eval.gate import GateTask

    GateTask(name="t", type="judge", threshold=0.5, judge_model=url)


def _parse_judge(url: str) -> None:
    from soup_cli.eval.gate import _parse_judge_url

    _parse_judge_url(url)


def _vllm(url: str) -> None:
    from soup_cli.data.providers.vllm import validate_vllm_url

    validate_vllm_url(url)


def _judge_api_base(url: str) -> None:
    from soup_cli.eval.judge import validate_judge_api_base

    validate_judge_api_base(url)


@pytest.mark.parametrize(
    "check",
    [_gate_task, _parse_judge, _vllm, _judge_api_base],
    ids=["GateTask", "_parse_judge_url", "vllm", "judge_api_base"],
)
def test_plain_http_to_remote_host_still_refused(check) -> None:
    """Sharing the set does not widen the gate: a remote plain-HTTP host is still refused."""
    with pytest.raises(ValueError):
        check(REMOTE + "/m")


def test_ship_judge_model_still_refuses_remote_plain_http() -> None:
    """``soup ship`` keeps refusing a remote plain-HTTP judge (it exits, not raises)."""
    import click

    from soup_cli.commands.ship import _validate_judge_model_url

    with pytest.raises((SystemExit, click.exceptions.Exit)):
        _validate_judge_model_url(REMOTE + "/m")
