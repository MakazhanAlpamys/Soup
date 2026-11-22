"""What ``/v1/tools/python`` and ``/v1/tools/bash`` answer, and the line numbers python reports.

* Both routes map a finished run to ``exit_code`` the same way: 124 for a timeout, 1 for
  an output overflow or a missing return code, otherwise the process's own code.
* When the sandbox cannot start, each route answers 501 with one fixed sentence:
  ``python sandbox requires OS-level isolation`` or ``bash sandbox requires OS-level
  isolation``. The text of the start-up error (a path, an errno string) stays in the
  server's debug log and is not sent to the caller.
* ``/v1/tools/python`` compiles the caller's code under the file name ``<code>``, so a
  traceback gives the caller's own line numbers. ``code_exec_reward`` and the recipe
  ``code`` node keep running their text as before.

The sandbox is faked, or replaced by a plain subprocess, so these run on every OS. The
live runs inside the Linux namespaces are in ``tests/test_serve_code_tool_namespaces.py``.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from soup_cli.trainer import rewards  # noqa: E402
from soup_cli.trainer.rewards import SandboxProcessResult  # noqa: E402

TOKEN = "s3cret-tool-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PYTHON_DETAIL = "python sandbox requires OS-level isolation"
BASH_DETAIL = "bash sandbox requires OS-level isolation"


def _client() -> TestClient:
    from soup_cli.commands.serve import _create_app

    app = _create_app(
        model_obj=MagicMock(),
        tokenizer=MagicMock(),
        device="cpu",
        model_name="test-model",
        max_tokens_default=16,
        auth_token=TOKEN,
    )
    return TestClient(app, base_url="http://127.0.0.1:8000", headers=AUTH)


@pytest.fixture
def isolated(monkeypatch):
    """Report OS-level isolation, so neither route stops at the strategy check."""
    monkeypatch.setattr(rewards, "_get_isolation_strategy", lambda: "namespaces")


def _fake_route_result(monkeypatch, route: str, result: SandboxProcessResult) -> None:
    if route == "python":
        monkeypatch.setattr(rewards, "_run_code_sandbox_process", lambda *a, **k: result)
    else:
        monkeypatch.setattr(rewards, "_run_bash_sandbox", lambda *a, **k: result)


def _post(client: TestClient, route: str):
    if route == "python":
        return client.post("/v1/tools/python", json={"code": "print(1)"})
    return client.post("/v1/tools/bash", json={"command": "echo 1"})


ROUTES = ["python", "bash"]


class TestExitCode:
    @pytest.mark.parametrize("route", ROUTES)
    @pytest.mark.parametrize(
        ("result", "exit_code", "timed_out"),
        [
            (SandboxProcessResult(None, "", "", timed_out=True), 124, True),
            (SandboxProcessResult(0, "", "", timed_out=True), 124, True),
            (
                SandboxProcessResult(None, "", "over", timed_out=True, output_exceeded=True),
                124,
                True,
            ),
            (SandboxProcessResult(0, "", "over", output_exceeded=True), 1, False),
            (SandboxProcessResult(-9, "", "over", output_exceeded=True), 1, False),
            (SandboxProcessResult(None, "", ""), 1, False),
            (SandboxProcessResult(3, "out", "err"), 3, False),
            (SandboxProcessResult(0, "out", ""), 0, False),
        ],
        ids=[
            "timeout",
            "timeout-wins-over-a-code",
            "timeout-wins-over-an-overflow",
            "overflow",
            "overflow-wins-over-a-code",
            "no-code",
            "own-code",
            "success",
        ],
    )
    def test_both_routes_map_a_run_the_same_way(
        self, monkeypatch, isolated, route, result, exit_code, timed_out
    ):
        _fake_route_result(monkeypatch, route, result)

        resp = _post(_client(), route)

        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": exit_code,
            "timed_out": timed_out,
        }


class TestStartUpFailure:
    @pytest.mark.parametrize(
        ("route", "detail"), [("python", PYTHON_DETAIL), ("bash", BASH_DETAIL)]
    )
    def test_a_result_that_never_started_is_a_fixed_501(
        self, monkeypatch, isolated, route, detail
    ):
        result = SandboxProcessResult(
            None, "", "[Errno 13] denied: '/home/u/x'", launch_failed=True
        )
        _fake_route_result(monkeypatch, route, result)

        resp = _post(_client(), route)

        assert resp.status_code == 501, resp.text
        assert resp.json() == {"detail": detail}
        assert "/home/u" not in resp.text

    @pytest.mark.parametrize(
        "exc",
        [
            PermissionError(13, "Permission denied", "/home/u/x"),
            NotImplementedError("bash sandbox not supported on /home/u/x"),
            subprocess.SubprocessError("Exception occurred in preexec_fn: /home/u/x"),
        ],
        ids=["permission", "not-implemented", "subprocess"],
    )
    def test_bash_start_up_errors_answer_the_fixed_sentence(self, monkeypatch, isolated, exc):
        def _raise(*_args, **_kwargs):
            raise exc

        monkeypatch.setattr(rewards, "_run_bash_sandbox", _raise)

        resp = _post(_client(), "bash")

        assert resp.status_code == 501, resp.text
        assert resp.json() == {"detail": BASH_DETAIL}
        assert "/home/u" not in resp.text
        assert "denied" not in resp.text.lower()

    def test_python_start_up_error_answers_the_fixed_sentence(self, monkeypatch, isolated):
        def _raise(*_args, **_kwargs):
            raise rewards.SandboxIsolationError("[Errno 1] Operation not permitted: '/home/u'")

        monkeypatch.setattr(rewards, "_run_code_sandbox_process", _raise)

        resp = _post(_client(), "python")

        assert resp.status_code == 501, resp.text
        assert resp.json() == {"detail": PYTHON_DETAIL}
        assert "/home/u" not in resp.text


# ---------------------------------------------------------------------------
# Line numbers: the real interpreter, without the OS sandbox
# ---------------------------------------------------------------------------


@pytest.fixture
def unsandboxed(monkeypatch) -> list[list[str]]:
    """Run the sandbox's argv with a plain subprocess: no limits, no namespaces.

    What is under test is the program the sandbox starts, not the sandbox, so this runs
    on every OS. The child gets the sandbox's own scrubbed environment, as in a real run
    (a FORCE_COLOR from the test's shell would colour a 3.13+ traceback). Returns the
    argv of each run.
    """
    seen: list[list[str]] = []

    def _run(argv, preexec_fn=None, **kwargs):
        seen.append(list(argv))
        env = kwargs.get("env") or rewards._get_safe_sandbox_env()
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=60, check=False, env=env
        )
        return SandboxProcessResult(done.returncode, done.stdout.strip(), done.stderr.strip())

    monkeypatch.setattr(rewards, "_run_sandboxed_subprocess", _run)
    monkeypatch.setattr(rewards, "_show_code_exec_warning_once", lambda: None)
    return seen


class TestPythonLineNumbers:
    @pytest.mark.parametrize(
        ("code", "line", "error"),
        [
            ("print(1)\n1/0", 2, "ZeroDivisionError"),
            ("1/0", 1, "ZeroDivisionError"),
            ("x = 1\n\n\nraise ValueError('third')", 4, "ValueError: third"),
            ("x = 1\ndef foo(", 2, "SyntaxError"),
            ("def foo():\n    return 1/0\nfoo()", 2, "ZeroDivisionError"),
        ],
        ids=["second-line", "first-line", "after-blank-lines", "syntax-error", "in-a-function"],
    )
    def test_a_traceback_names_the_callers_own_line(self, unsandboxed, code, line, error):
        result = rewards._run_code_sandbox_process(code, strict_namespaces=True)

        assert result.returncode == 1, result
        assert f'File "<code>", line {line}' in result.stderr, result.stderr
        assert error in result.stderr, result.stderr

    def test_the_route_reports_the_callers_own_line(self, unsandboxed, isolated):
        resp = _client().post("/v1/tools/python", json={"code": "print(1)\n1/0"})

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["stdout"] == "1"
        assert data["exit_code"] == 1
        assert 'File "<code>", line 2' in data["stderr"], data["stderr"]

    @pytest.mark.parametrize(
        ("code", "stdout"),
        [
            ("import sys\nprint(sys.argv)", "['-c']"),
            ("print(__name__)", "__main__"),
            ("x = 41\ndef f():\n    return x + 1\nprint(f())", "42"),
            (
                "import sys\nclass A:\n    pass\n"
                "print(A.__module__, hasattr(sys.modules['__main__'], 'A'))",
                "__main__ True",
            ),
        ],
        ids=["argv", "name", "module-globals", "main-module"],
    )
    def test_the_code_runs_as_the_main_module_as_before(self, unsandboxed, code, stdout):
        result = rewards._run_code_sandbox_process(code)

        assert result.returncode == 0, result.stderr
        assert result.stdout == stdout

    def test_the_network_guard_still_runs_first(self, unsandboxed):
        result = rewards._run_code_sandbox_process(
            "import socket\ntry:\n    socket.socket()\nexcept OSError as exc:\n    print(exc)"
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout == "network disabled in sandbox"

    def test_the_code_is_its_own_argument(self, unsandboxed):
        rewards._run_code_sandbox_process("print(1)")

        (argv,) = unsandboxed
        assert argv[:4] == [sys.executable, "-I", "-S", "-c"]
        assert argv[-1] == "print(1)"
        assert rewards.SANDBOX_NETWORK_GUARD in argv[4]

    def test_the_reward_sandbox_keeps_its_program(self, unsandboxed):
        """``code_exec_reward`` and the recipe node run the guard and the code as one text."""
        assert rewards._run_code_sandbox("print(1)") == "1"

        (argv,) = unsandboxed
        assert argv == [
            sys.executable, "-I", "-S", "-c", rewards.SANDBOX_NETWORK_GUARD + "\n" + "print(1)"
        ]
