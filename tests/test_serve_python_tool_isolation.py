"""soup serve: /v1/tools/python needs OS-level isolation, as /v1/tools/bash does.

Where the code sandbox has only its in-process guards to offer (the
``best-effort`` isolation strategy: Windows, or a Linux interpreter without
``os.unshare``), the python route answers 501 and does not start the
sandbox, the same answer the bash route gives there.

Where the strategy is ``namespaces`` but the kernel or container refuses to
create the namespaces at run time, the route gives the same 501 and the code
does not start, as the bash route does. The code sandbox used by training
(``code_exec_reward``) and by the recipe ``code`` node keeps running the code
without the namespaces there, as before.
"""

from __future__ import annotations

import errno
import io
import os
import subprocess
import sys
import types
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

PYTHON_DETAIL = "python sandbox requires OS-level isolation"
BASH_DETAIL = "bash sandbox requires OS-level isolation"
TOKEN = "s3cret-token"


def _client(**kwargs) -> TestClient:
    # /v1/tools/python and /v1/tools/bash answer 401 without a token on every bind
    # (tests/test_serve_code_tool_token.py), so the app has one and the client sends it.
    from soup_cli.commands.serve import _create_app

    base = dict(
        model_obj=MagicMock(),
        tokenizer=MagicMock(),
        device="cpu",
        model_name="test-model",
        max_tokens_default=128,
        auth_token=TOKEN,
    )
    base.update(kwargs)
    return TestClient(
        _create_app(**base),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


@pytest.fixture
def sandbox_calls(monkeypatch) -> list[str]:
    """Replace the code sandbox with a recorder that returns fixed stdout."""
    from soup_cli.trainer.rewards import SandboxProcessResult

    calls: list[str] = []

    def _fake_sandbox(code: str, strict_namespaces: bool = False) -> SandboxProcessResult:
        assert strict_namespaces is True
        calls.append(code)
        return SandboxProcessResult(returncode=0, stdout="ran", stderr="")

    monkeypatch.setattr("soup_cli.trainer.rewards._run_code_sandbox_process", _fake_sandbox)
    monkeypatch.setattr("soup_cli.trainer.rewards._run_code_sandbox", lambda *a, **k: "ran")
    return calls


def _set_strategy(monkeypatch, strategy: str) -> None:
    monkeypatch.setattr("soup_cli.trainer.rewards._get_isolation_strategy", lambda: strategy)


def test_python_tool_refused_without_os_isolation(monkeypatch, sandbox_calls):
    _set_strategy(monkeypatch, "best-effort")
    resp = _client().post("/v1/tools/python", json={"code": "print(1)"})
    assert resp.status_code == 501, resp.text
    assert resp.json()["detail"] == PYTHON_DETAIL
    assert sandbox_calls == []


@pytest.mark.parametrize("strategy", ["namespaces", "sandbox-exec"])
def test_python_tool_runs_with_os_isolation(monkeypatch, sandbox_calls, strategy):
    _set_strategy(monkeypatch, strategy)
    resp = _client().post("/v1/tools/python", json={"code": "print(1)"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"stdout": "ran", "stderr": "", "exit_code": 0, "timed_out": False}
    assert sandbox_calls == ["print(1)"]


@pytest.mark.parametrize(
    ("path", "body", "detail"),
    [
        ("/v1/tools/python", {"code": "print(1)"}, PYTHON_DETAIL),
        ("/v1/tools/bash", {"command": "echo 1"}, BASH_DETAIL),
    ],
)
def test_code_tools_give_the_same_answer_without_os_isolation(
    monkeypatch, sandbox_calls, path, body, detail
):
    _set_strategy(monkeypatch, "best-effort")
    resp = _client().post(path, json=body)
    assert resp.status_code == 501, (path, resp.text)
    assert resp.json()["detail"] == detail


def test_invalid_request_is_still_400_without_os_isolation(monkeypatch, sandbox_calls):
    _set_strategy(monkeypatch, "best-effort")
    resp = _client().post("/v1/tools/python", json={})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "Invalid request"


def test_token_is_checked_before_isolation(monkeypatch, sandbox_calls):
    _set_strategy(monkeypatch, "best-effort")
    client = _client(auth_token=TOKEN)
    resp = client.post(
        "/v1/tools/python",
        json={"code": "print(1)"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "Invalid or missing bearer token"
    assert sandbox_calls == []


def test_correct_token_does_not_lift_the_isolation_requirement(monkeypatch, sandbox_calls):
    _set_strategy(monkeypatch, "best-effort")
    client = _client(auth_token=TOKEN)
    resp = client.post(
        "/v1/tools/python",
        json={"code": "print(1)"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code == 501, resp.text
    assert resp.json()["detail"] == PYTHON_DETAIL
    assert sandbox_calls == []


# ---------------------------------------------------------------------------
# The ``namespaces`` strategy, with the namespaces refused at run time.
# ---------------------------------------------------------------------------


@dataclass
class _Child:
    """What the stand-in for the forked sandbox child saw."""

    unshare_error: Exception | None
    unshare_flags: list[int] = field(default_factory=list)
    rlimits: list[tuple] = field(default_factory=list)
    started: list[str] = field(default_factory=list)


@pytest.fixture
def child(monkeypatch) -> _Child:
    """Run the sandbox's pre-exec hook in this process instead of a forked child.

    The sandbox sets resource limits and unshares the Linux namespaces in a
    ``preexec_fn``, which runs in the forked child before the code starts. When
    the hook raises, the code never starts and CPython's ``Popen`` raises
    ``SubprocessError("Exception occurred in preexec_fn.")``. The stand-in
    ``Popen`` below does the same here, so every process-wide call the hook
    makes is replaced first: ``resource`` by a recorder, and ``os.unshare`` by
    a recorder that raises ``child.unshare_error`` (EPERM by default, the answer
    of a kernel or container that refuses unprivileged user namespaces). The
    sandbox module sees a Linux host with the ``namespaces`` strategy on every
    platform the suite runs on.
    """
    from soup_cli.trainer import rewards

    state = _Child(unshare_error=PermissionError(errno.EPERM, "Operation not permitted"))

    def _setrlimit(which, limits) -> None:
        state.rlimits.append((which, limits))

    fake_resource = types.SimpleNamespace(
        setrlimit=_setrlimit,
        RLIMIT_AS=9,
        RLIMIT_CPU=0,
        RLIMIT_FSIZE=1,
        RLIMIT_CORE=4,
        RLIMIT_NOFILE=7,
        RLIMIT_NPROC=6,
    )

    def _unshare(flags: int) -> None:
        state.unshare_flags.append(flags)
        if state.unshare_error is not None:
            raise state.unshare_error

    class _Popen:
        def __init__(self, argv, *, preexec_fn=None, **_kwargs) -> None:
            if preexec_fn is not None:
                # The hook must never reach the real limits or namespaces of
                # the test process.
                assert sys.modules["resource"] is fake_resource
                assert os.unshare is _unshare
                try:
                    preexec_fn()
                except Exception as exc:
                    raise subprocess.SubprocessError(
                        "Exception occurred in preexec_fn."
                    ) from exc
            state.started.append(argv[-1])
            self.stdout = io.BytesIO(b"ran\n")
            self.stderr = io.BytesIO(b"")
            self.returncode = 0

        def wait(self, timeout=None) -> int:
            return self.returncode

        def kill(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setattr(os, "unshare", _unshare, raising=False)
    monkeypatch.setattr(
        rewards, "sys", types.SimpleNamespace(platform="linux", executable=sys.executable)
    )
    monkeypatch.setattr(
        rewards,
        "subprocess",
        types.SimpleNamespace(
            Popen=_Popen,
            PIPE=subprocess.PIPE,
            SubprocessError=subprocess.SubprocessError,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    _set_strategy(monkeypatch, "namespaces")
    return state


def _namespace_flags() -> int:
    from soup_cli.trainer import rewards

    return rewards._CLONE_NEWUSER | rewards._CLONE_NEWNET | rewards._CLONE_NEWPID


def test_python_tool_refused_when_the_namespaces_cannot_be_created(child):
    resp = _client().post("/v1/tools/python", json={"code": "print(1)"})
    assert resp.status_code == 501, resp.text
    assert resp.json()["detail"] == PYTHON_DETAIL
    assert child.unshare_flags == [_namespace_flags()]
    assert child.started == []


def test_python_and_bash_tools_agree_when_the_namespaces_cannot_be_created(child):
    python_resp = _client().post("/v1/tools/python", json={"code": "print(1)"})
    bash_resp = _client().post("/v1/tools/bash", json={"command": "echo 1"})
    assert python_resp.status_code == 501, python_resp.text
    assert bash_resp.status_code == 501, bash_resp.text
    assert child.unshare_flags == [_namespace_flags()] * 2
    assert child.started == []


def test_python_tool_runs_when_the_namespaces_are_created(child):
    child.unshare_error = None
    resp = _client().post("/v1/tools/python", json={"code": "print(1)"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"stdout": "ran", "stderr": "", "exit_code": 0, "timed_out": False}
    assert child.unshare_flags == [_namespace_flags()]
    assert len(child.started) == 1
    assert child.started[0].endswith("\nprint(1)")
    assert child.rlimits, "the resource limits are still set before the code starts"


def test_strict_code_sandbox_raises_when_the_namespaces_cannot_be_created(child):
    from soup_cli.trainer import rewards

    with pytest.raises(rewards.SandboxIsolationError):
        rewards._run_code_sandbox("print(1)", strict_namespaces=True)
    assert child.unshare_flags == [_namespace_flags()]
    assert child.started == []


def test_reward_sandbox_still_runs_when_the_namespaces_cannot_be_created(child):
    """Training's code_exec_reward keeps its fallback: no namespaces, the code still runs."""
    from soup_cli.trainer import rewards

    assert rewards._run_code_sandbox("print(1)") == "ran"
    scores = rewards.code_exec_reward(
        [[{"role": "assistant", "content": "print(1)"}]], expected=["ran"]
    )
    assert scores == [1.0]
    assert child.unshare_flags == [_namespace_flags()] * 2
    assert len(child.started) == 2


def test_recipe_code_node_still_runs_when_the_namespaces_cannot_be_created(child):
    from soup_cli.utils.recipe_dag import RecipeNode
    from soup_cli.utils.recipe_run import _node_code

    node = RecipeNode(name="c", kind="code", config={"code": "print(1)"})
    assert _node_code(node, (({"x": 1},),)) == [{"x": 1, "c": "ran"}]
    assert child.unshare_flags == [_namespace_flags()]
    assert len(child.started) == 1
