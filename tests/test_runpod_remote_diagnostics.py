"""Retain useful remote failures before deleting a RunPod, without using SSH or Pods."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import types
import unicodedata
from pathlib import Path

import pytest

from soup_cli.utils.terminal import strip_control
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
    assert not any(ord(char) < 0x20 and char not in "\t\n\r" for char in tail)
    assert "\t" in tail
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


def test_stderr_tail_uses_the_shared_terminal_character_policy(controller):
    # Include every Unicode format character, C0/C1, and the line separators;
    # a category-wide filter also strips legitimate characters the policy keeps.
    chars = "".join(
        chr(code)
        for code in range(sys.maxunicode + 1)
        if code < 0xA0 or unicodedata.category(chr(code)) in {"Cf", "Zl", "Zp"}
    )
    raw = "start" + chars + "\nArabic joiners: \u200c\u200d\tend"
    rc, events, _ = _run_controller(controller, stderr=raw)
    assert rc == 1
    assert _tail(events) == "\n".join(strip_control(raw).splitlines()[-50:])


def test_rendered_controller_carries_updates_to_the_shared_terminal_policy(monkeypatch):
    from soup_cli.utils import terminal

    monkeypatch.setitem(terminal._CONTROL_STRIP_TABLE, ord("\u2603"), None)
    ns = _controller()
    notices = []
    ns["_notice"] = notices.append
    ns["_print_remote_stderr_tail"]("before\u2603after")
    assert notices == ["RunPod remote stderr tail (last 50 lines):\nbeforeafter"]


@pytest.mark.parametrize(
    ("url", "safe_url"),
    [
        (
            "https://alice:private-password@packages.example/simple",
            "https://<redacted>@packages.example/simple",
        ),
        (
            "https://private-token@packages.example/simple",
            "https://<redacted>@packages.example/simple",
        ),
        (
            "HTTPS://alice:p%40ss%3Aword@packages.example:8443/simple",
            "HTTPS://<redacted>@packages.example:8443/simple",
        ),
        (
            "https://alice:p@ss@packages.example/simple",
            "https://<redacted>@packages.example/simple",
        ),
        (
            "https://alice:pass@[2001:db8::1]:8443/simple",
            "https://<redacted>@[2001:db8::1]:8443/simple",
        ),
        ("https://packages.example/simple", "https://packages.example/simple"),
        ("https://packages.example/files/a@b.whl", "https://packages.example/files/a@b.whl"),
    ],
    ids=[
        "password",
        "token-user",
        "encoded-password",
        "at-in-password",
        "ipv6",
        "public-url",
        "at-in-path",
    ],
)
def test_stderr_tail_redacts_url_credentials_before_deletion(controller, url, safe_url):
    rc, events, _ = _run_controller(
        controller, stderr=f"pip install failed: '{url}'\nRetry exhausted"
    )
    assert rc == 1
    assert _tail(events) == f"pip install failed: '{safe_url}'\nRetry exhausted"
    assert ("notice", "RunPod termination verified.") in events


@pytest.mark.parametrize(
    "secret",
    ["hf_" + "a" * 25, "sk-" + "b" * 25, "api_key=" + "c" * 25, "Bearer " + "d" * 25],
)
def test_stderr_tail_redacts_common_token_shapes(controller, secret):
    rc, events, _ = _run_controller(controller, stderr=f"Authentication failed: {secret}\nFatal")
    assert rc == 1
    assert _tail(events) == "Authentication failed: <redacted>\nFatal"


def test_controls_inside_a_token_are_removed_before_it_is_matched(controller):
    token = "hf_" + "a" * 10 + "\x1b" + "a" * 15
    url = "ht\x1btps://user:private-password@packages.example/simple"
    rc, events, _ = _run_controller(controller, stderr=f"{token}\n{url}\nFatal")
    assert rc == 1
    assert _tail(events) == "<redacted>\nhttps://<redacted>@packages.example/simple\nFatal"


def test_the_controller_redacts_the_same_token_shapes_as_the_crash_bundle(controller):
    from soup_cli.utils import crash

    ns, _ = controller
    assert [(pattern.pattern, pattern.flags) for pattern in ns["_SECRET_PATTERNS"]] == [
        (pattern.pattern, pattern.flags) for pattern in crash._SECRET_PATTERNS
    ]


def test_stderr_tail_redacts_before_byte_truncation_and_after_control_removal(controller):
    raw = "https://user:" + "private-password" * 4_000 + "@packages.example/simple\nFatal"
    raw += "\nhttps://user:private\x1b-password@packages.example/simple"
    rc, events, _ = _run_controller(controller, stderr=raw)
    assert rc == 1
    tail = _tail(events)
    assert "private" not in tail and "password" not in tail
    assert tail.count("https://<redacted>@packages.example/simple") == 2
    assert "Fatal" in tail


@pytest.mark.parametrize("stderr", [None, "", "\x00\x1b\u200b"])
def test_no_empty_tail_notice_is_added_to_a_failure(controller, stderr):
    rc, events, _ = _run_controller(controller, stderr=stderr)
    assert rc == 1
    assert not any("stderr tail" in value for _, value in events)
    assert (
        "notice",
        "RunPod remote training failed; no successful artifact copy claimed",
    ) in events
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
        (7, b"prefix-" + b"z" * 60_000 + b"\nFatalTrainingError"),
        (7, b"https://user:" + b"private-password" * 4_000 + b"@index.example/simple\nFatal"),
        (7, b"z" * 49_994 + b"\nFatal"),
        (7, b"discard\n" + b"z" * 49_994 + b"\nFatal"),
        (7, b"z" * 60_000),
        (0, b"not a failed run"),
    ],
    ids=[
        "fifty-lines",
        "fifty-thousand-bytes",
        "cut-credential",
        "exact-byte-cap",
        "complete-line-at-byte-cap",
        "only-partial-line",
        "success-control",
    ],
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
    complete_tail = log[-50_000:]
    if len(log) > 50_000 and log[-50_001:-50_000] != b"\n":
        complete_tail = complete_tail.partition(b"\n")[2]
    expected = b"\n".join(complete_tail.splitlines()[-50:]) if returncode else b""
    assert tail == expected
    assert b"private-password" not in tail
    assert len(tail) <= 50_000
    assert len(tail.splitlines()) <= 50
