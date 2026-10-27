"""Retain useful remote failures before deleting a RunPod, without using SSH or Pods."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tests.test_runpod_cloud import FAKE_TOOL_DIR, _controller


@pytest.fixture
def controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    key = tmp_path / "id_test"
    key.write_text("not-a-private-key", encoding="utf-8")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAATEST fixture", encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-provider-secret")
    monkeypatch.setenv("RUNPOD_SSH_PRIVATE_KEY", str(key))
    ns = _controller()
    events = []

    def request(_key, path, *, method="GET", **_kwargs):
        events.append((method, path))
        return {"id": "pod-1"} if method == "POST" else None

    ns["_request_json"] = request
    ns["_wait_for_ssh"] = lambda *_args: ("8.8.8.8", 2222)
    ns["_notice"] = lambda text: events.append(("notice", text))
    return ns, events


def _run_controller(controller, *, stderr=None, returncode=1, timeout_step=None):
    ns, events = controller
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        step = "run" if "input" in kwargs else "readiness"
        if step == timeout_step:
            raise subprocess.TimeoutExpired("private-command", kwargs["timeout"], stderr=b"secret")
        if argv[0] == FAKE_TOOL_DIR + "scp":
            (Path(argv[-1]) / "out").mkdir()
        return subprocess.CompletedProcess(
            argv, returncode if step == "run" else 0, "", stderr if step == "run" else ""
        )

    ns["subprocess"] = types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired)
    return ns["main"](), events, calls


def _tail(events):
    notice = next(text for kind, text in events if kind == "notice" and "stderr tail" in text)
    return notice.partition("\n")[2]


def test_failed_run_prints_literal_sanitized_stderr_before_deleting(controller):
    raw = "pip failure\x00\x1b]0;title\x07\r\n[bold]Fatal error[/bold]\t\x7f\x9b\u202e\u200b"
    rc, events, calls = _run_controller(controller, stderr=raw)
    assert rc == 1
    tail = _tail(events)
    assert "pip failure" in tail
    assert "[bold]Fatal error[/bold]" in tail
    assert not any(ord(char) < 0x20 and char != "\n" for char in tail)
    assert not any(0x7F <= ord(char) <= 0x9F for char in tail)
    assert "\u202e" not in tail and "\u200b" not in tail
    tail_index = next(index for index, event in enumerate(events) if "stderr tail" in event[1])
    assert tail_index < events.index(("DELETE", "/pods/pod-1"))
    assert ("GET", "/pods/pod-1") in events
    assert ("notice", "RunPod termination verified.") in events
    assert len(calls) == 2 and all(argv[0] != FAKE_TOOL_DIR + "scp" for argv, _ in calls)
    run_options = calls[1][1]
    assert run_options["encoding"] == "utf-8" and run_options["errors"] == "replace"


def test_stderr_tail_keeps_only_the_last_fifty_lines(controller):
    raw = "\n".join(f"line-{index:03}" for index in range(90))
    rc, events, _ = _run_controller(controller, stderr=raw)
    assert rc == 1
    assert _tail(events).splitlines() == [f"line-{index:03}" for index in range(40, 90)]


def test_stderr_tail_caps_utf8_bytes_without_losing_the_last_error(controller):
    marker = "Fatal: remote model load failed"
    raw = "discard-this-prefix\n" + "€" * 20_000 + marker
    rc, events, _ = _run_controller(controller, stderr=raw)
    assert rc == 1
    tail = _tail(events)
    assert tail.endswith(marker)
    assert "discard-this-prefix" not in tail
    assert 49_900 <= len(tail.encode("utf-8")) <= 50_000


@pytest.mark.parametrize("stderr", [None, ""])
def test_no_empty_tail_notice_is_added_to_a_failure(controller, stderr):
    rc, events, _ = _run_controller(controller, stderr=stderr)
    assert rc == 1
    assert not any("stderr tail" in value for _, value in events)
    assert ("notice", "RunPod termination verified.") in events


def test_success_does_not_print_stderr_as_a_failure(controller):
    rc, events, calls = _run_controller(controller, stderr="benign pip warning", returncode=0)
    assert rc == 0
    assert not any("stderr tail" in value for _, value in events)
    assert any(argv[0] == FAKE_TOOL_DIR + "scp" for argv, _ in calls)
    assert ("notice", "RunPod termination verified.") in events


@pytest.mark.parametrize(
    ("step", "message", "training_calls"),
    [
        ("readiness", "RunPod SSH readiness probe timed out after 30 seconds", 0),
        ("run", "RunPod remote training command timed out after 25 hours", 1),
    ],
)
def test_timeout_names_its_step_and_still_confirms_deletion(
    controller, step, message, training_calls
):
    rc, events, calls = _run_controller(controller, timeout_step=step)
    assert rc == 1
    assert ("notice", message) in events
    assert ("notice", "RunPod training/controller failed; cleaning up the Pod.") not in events
    assert ("notice", "RunPod termination verified.") in events
    assert ("DELETE", "/pods/pod-1") in events and ("GET", "/pods/pod-1") in events
    assert sum("input" in kwargs for _, kwargs in calls) == training_calls
    assert not any(argv[0] == FAKE_TOOL_DIR + "scp" for argv, _ in calls)
    assert "private-command" not in str(events) and "secret" not in str(events)


def _remote_script() -> str:
    ns = _controller()
    scripts = []

    def run(argv, **kwargs):
        if "input" in kwargs:
            scripts.append(kwargs["input"])
        return subprocess.CompletedProcess(argv, int("input" in kwargs), "", "")

    ns["subprocess"] = types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired)
    with pytest.raises(ns["_ControllerError"], match="remote training failed"):
        ns["_train_and_copy"]("8.8.8.8", 2222, "fake-key")
    return scripts[0]


@pytest.mark.parametrize(
    ("returncode", "log"),
    [
        (7, b"\n".join(f"line-{index:03}".encode() for index in range(100))),
        (7, b"prefix-" + b"z" * 60_000 + b"FatalTrainingError"),
        (0, b"not a failed run"),
    ],
    ids=["fifty-lines", "fifty-thousand-bytes", "success-control"],
)
def test_remote_script_emits_a_bounded_train_log_on_failure(tmp_path, returncode, log):
    script = _remote_script()
    harness = tmp_path / "remote-harness.py"
    remote_root = tmp_path / "remote-workspace"
    harness.write_text(
        "import base64, pathlib, subprocess, sys, types\n"
        f"remote_root = pathlib.Path({str(remote_root)!r})\n"
        "real_path = pathlib.Path\n"
        "def fake_run(argv, **kwargs):\n"
        "    if argv[0] == sys.executable:\n"
        "        assert argv[1:4] == ['-m', 'pip', 'install']\n"
        "        sys.stderr.buffer.write(b'pip output\\n')\n"
        "        return types.SimpleNamespace(returncode=0)\n"
        "    assert argv == ['soup', 'train', '--config', 'soup.yaml', '--yes']\n"
        f"    kwargs['stdout'].buffer.write(base64.b64decode({base64.b64encode(log)!r}))\n"
        f"    return types.SimpleNamespace(returncode={returncode})\n"
        "sys.modules['pathlib'] = types.SimpleNamespace(Path=lambda path: "
        "remote_root if str(path) == '/workspace/soup' else real_path(path))\n"
        "sys.modules['subprocess'] = types.SimpleNamespace("
        "run=fake_run, STDOUT=subprocess.STDOUT)\n"
        f"exec({json.dumps(script)}, {{'__name__': '__main__'}})\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-I", str(harness)], capture_output=True, check=False, timeout=30
    )
    assert completed.returncode == returncode, completed.stderr
    stderr = completed.stderr
    assert stderr.startswith(b"pip output\n")
    tail = stderr.removeprefix(b"pip output\n")
    expected = b"\n".join(log[-50_000:].splitlines()[-50:]) if returncode else b""
    assert tail == expected
    assert len(tail) <= 50_000
    assert len(tail.splitlines()) <= 50
