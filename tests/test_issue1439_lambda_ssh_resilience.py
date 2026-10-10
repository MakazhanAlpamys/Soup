"""Tests for Lambda Cloud controller SSH and transport resilience (#1439).

Verifies that the rendered Lambda controller:
1. Configures SSH keepalive options (ServerAliveInterval and ServerAliveCountMax).
2. Retries transient SSH transport drops (exit 255 and TimeoutExpired) rather than
   immediately terminating the remote instance.
3. Distinguishes connection drops from remote training failures (exit != 0).
4. Retries checkpoint copy (scp) before terminating the instance.
5. Polls when the exit status file is not yet present on the instance.
6. Always terminates the instance in its finally block.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from soup_cli.cloud.lambda_labs import render_lambda_stub

_SOUP_YAML = (
    "base: hf-internal-testing/tiny-random-gpt2\n"
    "task: sft\n"
    "data:\n  train: data.jsonl\n  format: chatml\n"
    "output: ./out\n"
)


def _setup_controller_env(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    key = tmp_path / "lambda-key"
    key.write_text("test-only", encoding="utf-8")
    monkeypatch.setenv("LAMBDA_API_KEY", "secret")
    monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "my-key")
    monkeypatch.setenv("LAMBDA_SSH_PRIVATE_KEY", str(key))
    monkeypatch.setenv("LAMBDA_REGION", "us-tx-1")
    stub = render_lambda_stub(
        _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.75.1"
    )
    return stub, key


def _exec_controller(
    stub: str, fake_run_fn, captured_prints: list[str] | None = None
) -> tuple[int, list]:
    namespace: dict[str, object] = {"__name__": "controller"}
    exec(stub, namespace)

    terminations: list[tuple] = []
    clock = {"now": 0.0}

    def fake_request(api_key, path, *, method="GET", payload=None):
        if path.endswith("/launch"):
            return {"data": {"instance_ids": ["instance-1"]}}
        if path.endswith("/terminate"):
            terminations.append((api_key, path, payload))
            return {"data": {}}
        return {"data": {}}

    def fake_sleep(seconds: float) -> None:
        clock["now"] += seconds

    namespace["_request_json"] = fake_request
    namespace["_wait_for_ip"] = lambda *_: "192.0.2.1"
    namespace["_wait_for_termination"] = lambda *_: None
    namespace["time"] = types.SimpleNamespace(
        sleep=fake_sleep,
        monotonic=lambda: clock["now"],
    )

    if captured_prints is not None:
        namespace["print"] = lambda *args, **kw: captured_prints.append(
            " ".join(str(a) for a in args)
        )

    namespace["subprocess"] = types.SimpleNamespace(
        run=fake_run_fn,
        TimeoutExpired=subprocess.TimeoutExpired,
    )

    rc = namespace["main"]()
    return rc, terminations


def _run_controller(tmp_path: Path, monkeypatch, ssh_reply, scp_reply=None):
    """Exec controller with a fake clock; ssh_reply(call_no, now) -> (rc, out, err)."""
    key = tmp_path / "lambda-key"
    key.write_text("test-only", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAMBDA_API_KEY", "secret")
    monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "my-key")
    monkeypatch.setenv("LAMBDA_SSH_PRIVATE_KEY", str(key))
    stub = render_lambda_stub(
        "base: gpt2\ndata:\n  train: data.jsonl\noutput: ./out\n",
        gpu="a100",
        output_dir="./out",
        soup_version="0.75.1",
    )
    ns: dict[str, object] = {"__name__": "controller"}
    exec(stub, ns)
    clock = {"now": 0.0}
    seen = {"ssh": 0, "scp": 0, "terminate": 0, "printed": []}

    def fake_request(api_key, path, *, method="GET", payload=None):
        if path.endswith("/launch"):
            return {"data": {"instance_ids": ["instance-1"]}}
        if path.endswith("/terminate"):
            seen["terminate"] += 1
        return {"data": {}}

    def fake_run(argv, **kw):
        if argv[0] == "scp":
            seen["scp"] += 1
            rc = 0 if scp_reply is None else scp_reply(seen["scp"], clock)
            return subprocess.CompletedProcess(argv, rc, "", "")
        seen["ssh"] += 1
        assert seen["ssh"] < 100_000, "controller is polling without bound"
        return subprocess.CompletedProcess(argv, *ssh_reply(seen["ssh"], clock["now"]))

    def fake_sleep(seconds: float) -> None:
        clock["now"] += seconds

    ns["_request_json"] = fake_request
    ns["_wait_for_ip"] = lambda *_: "192.0.2.1"
    ns["_wait_for_termination"] = lambda *_: None
    ns["time"] = types.SimpleNamespace(sleep=fake_sleep, monotonic=lambda: clock["now"])
    ns["subprocess"] = types.SimpleNamespace(
        run=fake_run, TimeoutExpired=subprocess.TimeoutExpired
    )
    ns["print"] = lambda *a, **k: seen["printed"].append(" ".join(map(str, a)))
    rc = ns["main"]()
    return rc, seen, clock["now"]


class TestLambdaSshResilience:
    def test_keepalive_options_present_in_stub(self) -> None:
        stub = render_lambda_stub(
            _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.75.1"
        )
        assert "ServerAliveInterval" in stub
        assert "ServerAliveCountMax" in stub
        assert "-o" in stub

    def test_ssh_returns_255_twice_then_succeeds(self, tmp_path: Path, monkeypatch) -> None:
        stub, _ = _setup_controller_env(tmp_path, monkeypatch)
        calls: list[str] = []
        ssh_attempts = 0

        def fake_run(argv, **kw):
            nonlocal ssh_attempts
            cmd = argv[0]
            calls.append(cmd)
            if cmd == "ssh":
                ssh_attempts += 1
                if ssh_attempts <= 2:
                    return subprocess.CompletedProcess(
                        argv, 255, "", "Connection to 192.0.2.1 closed by remote host."
                    )
                return subprocess.CompletedProcess(argv, 0, "0\n", "")
            if cmd == "scp":
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec_controller(stub, fake_run)
        assert rc == 0
        assert ssh_attempts == 3
        assert "scp" in calls
        assert len(terminations) == 1
        assert terminations[0][2] == {"instance_ids": ["instance-1"]}

    def test_ssh_timeout_expired_retried_and_succeeds(self, tmp_path: Path, monkeypatch) -> None:
        stub, _ = _setup_controller_env(tmp_path, monkeypatch)
        ssh_attempts = 0

        def fake_run(argv, **kw):
            nonlocal ssh_attempts
            cmd = argv[0]
            if cmd == "ssh":
                ssh_attempts += 1
                if ssh_attempts == 1:
                    raise subprocess.TimeoutExpired(argv, 30)
                return subprocess.CompletedProcess(argv, 0, "0\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec_controller(stub, fake_run)
        assert rc == 0
        assert ssh_attempts == 2
        assert len(terminations) == 1

    def test_ssh_persistent_255_terminates_with_lost_contact_message(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub, _ = _setup_controller_env(tmp_path, monkeypatch)
        prints: list[str] = []
        ssh_attempts = 0

        def fake_run(argv, **kw):
            nonlocal ssh_attempts
            cmd = argv[0]
            if cmd == "ssh":
                ssh_attempts += 1
                return subprocess.CompletedProcess(
                    argv, 255, "", "Connection to 192.0.2.1 closed by remote host."
                )
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec_controller(stub, fake_run, captured_prints=prints)
        assert rc == 1
        assert ssh_attempts > 1
        assert len(terminations) == 1
        failed_lines = [p for p in prints if "Lambda training failed" in p]
        assert len(failed_lines) == 1
        assert "lost contact with the instance for 15 minutes" in failed_lines[0]
        assert "remote training status failed" not in failed_lines[0]

    def test_scp_fails_once_then_succeeds(self, tmp_path: Path, monkeypatch) -> None:
        stub, _ = _setup_controller_env(tmp_path, monkeypatch)
        scp_attempts = 0

        def fake_run(argv, **kw):
            nonlocal scp_attempts
            cmd = argv[0]
            if cmd == "ssh":
                return subprocess.CompletedProcess(argv, 0, "0\n", "")
            if cmd == "scp":
                scp_attempts += 1
                if scp_attempts == 1:
                    return subprocess.CompletedProcess(argv, 1, "", "transient scp error")
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec_controller(stub, fake_run)
        assert rc == 0
        assert scp_attempts == 2
        assert len(terminations) == 1

    def test_scp_fails_on_all_attempts_terminates_with_copy_failure_message(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub, _ = _setup_controller_env(tmp_path, monkeypatch)
        prints: list[str] = []
        scp_attempts = 0

        def fake_run(argv, **kw):
            nonlocal scp_attempts
            cmd = argv[0]
            if cmd == "ssh":
                return subprocess.CompletedProcess(argv, 0, "0\n", "")
            if cmd == "scp":
                scp_attempts += 1
                return subprocess.CompletedProcess(argv, 1, "", "connection lost during scp")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec_controller(stub, fake_run, captured_prints=prints)
        assert rc == 1
        assert scp_attempts > 1
        assert len(terminations) == 1
        failed_lines = [p for p in prints if "Lambda training failed" in p]
        assert len(failed_lines) == 1
        assert "checkpoint copy failed after 15 minutes" in failed_lines[0]

    def test_training_failure_exit_1_terminates_immediately_without_retries(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub, _ = _setup_controller_env(tmp_path, monkeypatch)
        prints: list[str] = []
        ssh_attempts = 0

        def fake_run(argv, **kw):
            nonlocal ssh_attempts
            cmd = argv[0]
            if cmd == "ssh":
                ssh_attempts += 1
                return subprocess.CompletedProcess(argv, 0, "status: error\n1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec_controller(stub, fake_run, captured_prints=prints)
        assert rc == 1
        # 1 poll check + 1 log tail fetch; does not retry polling
        assert ssh_attempts == 2
        assert len(terminations) == 1
        failed_lines = [p for p in prints if "Lambda training failed" in p]
        assert len(failed_lines) == 1
        assert "remote soup train exited 1" in failed_lines[0]

    def test_polling_when_exit_status_absent_until_finished(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        stub, _ = _setup_controller_env(tmp_path, monkeypatch)
        ssh_attempts = 0

        def fake_run(argv, **kw):
            nonlocal ssh_attempts
            cmd = argv[0]
            if cmd == "ssh":
                ssh_attempts += 1
                if ssh_attempts < 3:
                    # Cloud-init still running
                    return subprocess.CompletedProcess(argv, 0, "RUNNING\n", "")
                # Finished on 3rd attempt
                return subprocess.CompletedProcess(argv, 0, "0\n", "")
            if cmd == "scp":
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        rc, terminations = _exec_controller(stub, fake_run)
        assert rc == 0
        assert ssh_attempts == 3
        assert len(terminations) == 1

    def test_a_ten_minute_network_outage_does_not_end_the_run(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        unreachable = (
            255, "", "ssh: connect to host 192.0.2.1 port 22: Network is unreachable"
        )

        def reply(call_no, now):
            return unreachable if now < 600 else (0, "0\n", "")

        rc, seen, _ = _run_controller(tmp_path, monkeypatch, reply)
        assert rc == 0, seen["printed"]
        assert seen["terminate"] == 1

    def test_a_run_that_never_reports_is_terminated_at_the_deadline(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        rc, seen, elapsed = _run_controller(
            tmp_path, monkeypatch, lambda n, now: (0, "RUNNING\n", "")
        )
        assert rc == 1
        assert seen["terminate"] == 1
        assert elapsed <= 90_000 + 60
        assert any("timed out" in line for line in seen["printed"]), seen["printed"]

    def test_poll_reports_finished_no_status_terminates_after_one_poll(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        rc, seen, _ = _run_controller(
            tmp_path, monkeypatch, lambda n, now: (0, "MISSING\n", "")
        )
        assert rc == 1
        assert seen["ssh"] == 1
        assert seen["terminate"] == 1
        assert any(
            "did not publish an exit status" in line for line in seen["printed"]
        ), seen["printed"]

    def test_transport_failures_reset_on_successful_poll(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        unreachable = (
            255, "", "ssh: connect to host 192.0.2.1 port 22: Network is unreachable"
        )

        def reply(call_no, now):
            if call_no <= 4:
                return unreachable
            if call_no == 5:
                return (0, "RUNNING\n", "")
            if call_no <= 9:
                return unreachable
            return (0, "0\n", "")

        rc, seen, _ = _run_controller(tmp_path, monkeypatch, reply)
        assert rc == 0, seen["printed"]
        assert seen["terminate"] == 1

    def test_two_ten_minute_outages_separated_by_a_good_poll_do_not_end_the_run(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        unreachable = (
            255, "", "ssh: connect to host 192.0.2.1 port 22: Network is unreachable"
        )
        state = {"good_at": None}

        def reply(call_no, now):
            if now < 600:
                return unreachable
            if state["good_at"] is None:
                state["good_at"] = now
                return (0, "RUNNING\n", "")
            if now < state["good_at"] + 605:
                return unreachable
            return (0, "0\n", "")

        rc, seen, _ = _run_controller(tmp_path, monkeypatch, reply)
        assert rc == 0, seen["printed"]
        assert seen["terminate"] == 1

    def test_a_long_copy_that_fails_once_is_retried(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        def scp(attempt, clock):
            if attempt == 1:
                clock["now"] += 1200  # a 20-minute copy that drops near the end
                return 1
            return 0

        rc, seen, _ = _run_controller(
            tmp_path, monkeypatch, lambda n, now: (0, "0\n", ""), scp_reply=scp
        )
        assert rc == 0, seen["printed"]
        assert seen["scp"] == 2


def _rendered_poll_command(tmp_path: Path, monkeypatch) -> str:
    """The exact remote command the rendered controller sends over ssh."""
    commands = []
    key = tmp_path / "lambda-key"
    key.write_text("test-only", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAMBDA_API_KEY", "secret")
    monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "my-key")
    monkeypatch.setenv("LAMBDA_SSH_PRIVATE_KEY", str(key))
    stub = render_lambda_stub(
        "base: gpt2\ndata:\n  train: data.jsonl\noutput: ./out\n",
        gpu="a100",
        output_dir="./out",
        soup_version="0.75.1",
    )
    ns = {"__name__": "controller"}
    exec(stub, ns)

    def fake_run(argv, **kw):
        if argv[0] == "ssh":
            commands.append(argv[-1])
            return subprocess.CompletedProcess(argv, 0, "0\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    ns["_request_json"] = lambda *a, **k: {"data": {"instance_ids": ["instance-1"]}}
    ns["_wait_for_ip"] = lambda *_: "192.0.2.1"
    ns["_wait_for_termination"] = lambda *_: None
    ns["time"] = types.SimpleNamespace(sleep=lambda s: None, monotonic=lambda: 0.0)
    ns["subprocess"] = types.SimpleNamespace(
        run=fake_run, TimeoutExpired=subprocess.TimeoutExpired
    )
    ns["print"] = lambda *a, **k: None
    assert ns["main"]() == 0
    return commands[0]


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None, reason="needs a POSIX sh"
)
@pytest.mark.parametrize(
    ("cloud_init", "exit_file", "expected"),
    [
        ("status: running", None, "RUNNING"),
        ("status: running", "0\n", "RUNNING"),
        ("status: not started", None, "RUNNING"),
        ("status: done", None, "MISSING"),
        ("status: done", "", "MISSING"),
        ("status: done", "0\n", "0"),
        ("status: error", "1\n", "1"),
    ],
)
def test_the_remote_poll_command_reports_cloud_init_state(
    tmp_path: Path, monkeypatch, cloud_init: str, exit_file: str | None, expected: str
) -> None:
    command = _rendered_poll_command(tmp_path, monkeypatch)
    exit_path = tmp_path / "soup.exit"
    assert "/home/ubuntu/soup.exit" in command
    command = command.replace("/home/ubuntu/soup.exit", exit_path.as_posix())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "cloud-init"
    fake.write_text(f"#!/bin/sh\necho '{cloud_init}'\n", encoding="utf-8", newline="\n")
    fake.chmod(0o755)
    if exit_file is not None:
        exit_path.write_text(exit_file, encoding="utf-8", newline="\n")
    env = dict(os.environ, PATH=f"{fake_bin.as_posix()}{os.pathsep}{os.environ['PATH']}")
    out = subprocess.run(
        ["sh", "-c", command], capture_output=True, text=True, env=env, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == expected
