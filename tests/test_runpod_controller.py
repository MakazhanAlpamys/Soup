"""Lifecycle boundaries exercised by running rendered controller functions."""

import io
import subprocess
import types
import urllib.error
from pathlib import Path

import pytest

from tests.test_runpod_cloud import CONFIG, _backend, _controller


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key = tmp_path / "id_test"
    key.write_text("not-a-real-private-key", encoding="utf-8")
    key.with_suffix(".pub").write_text("ssh-ed25519 AAAATEST fixture", encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-provider-secret")
    monkeypatch.setenv("RUNPOD_SSH_PRIVATE_KEY", str(key))
    return key


def _clock(ns, step=1):
    ticks = iter(range(0, 100000, step))
    sleeps = []
    ns["time"] = types.SimpleNamespace(monotonic=lambda: next(ticks), sleep=sleeps.append)
    return sleeps


def test_delete_polls_past_stopped_state_until_actual_absence():
    ns = _controller()
    reads = iter([{"desiredStatus": "EXITED"}, None])
    calls = []
    sleeps = _clock(ns)

    def request(*args, method="GET", **kwargs):
        calls.append(method)
        return None if method == "DELETE" else next(reads)

    ns["_request_json"] = request
    ns["_terminate"]("secret", "pod-1")
    assert calls == ["DELETE", "GET", "DELETE", "GET"]
    assert sleeps == [5]


def test_deletion_timeout_is_a_failure():
    ns = _controller()
    _clock(ns, 100)
    ns["_request_json"] = lambda *args, **kwargs: {"desiredStatus": "EXITED"}
    with pytest.raises(RuntimeError, match="not confirmed"):
        ns["_terminate"]("secret", "pod-1")


@pytest.mark.parametrize(
    "body",
    [b"", b"null", b"bad", b"x" * (1024 * 1024 + 1)],
    ids=["empty", "null", "malformed", "oversized"],
)
def test_only_http_404_can_establish_absence(monkeypatch, body):
    ns = _controller()

    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(body)

    monkeypatch.setattr(ns["urllib"].request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(RuntimeError):
        ns["_request_json"]("secret", "/pods/pod-1")


def test_rest_404_get_and_empty_delete_are_accepted(monkeypatch):
    ns = _controller()

    class Opener:
        def open(self, request, timeout):
            if request.method == "DELETE":
                return io.BytesIO(b"")
            raise urllib.error.HTTPError(request.full_url, 404, "absent", {}, None)

    monkeypatch.setattr(ns["urllib"].request, "build_opener", lambda *handlers: Opener())
    assert ns["_request_json"]("secret", "/pods/pod-1") is None
    assert ns["_request_json"]("secret", "/pods/pod-1", method="DELETE") is None
    with pytest.raises(RuntimeError, match="HTTP 404"):
        ns["_request_json"]("secret", "/pods", method="POST", payload={})


@pytest.mark.parametrize(
    "record",
    [
        None,
        {"desiredStatus": "EXITED"},
        {"publicIp": "127.0.0.1", "portMappings": {"22": 1234}},
        {"publicIp": "169.254.169.254", "portMappings": {"22": 1234}},
    ],
)
def test_ssh_refuses_missing_stopped_or_private_addresses(record):
    ns = _controller()
    ns["_request_json"] = lambda *args: record
    with pytest.raises(RuntimeError):
        ns["_wait_for_ssh"]("secret", "pod-1")


def test_ssh_waits_for_real_port_and_has_a_deadline():
    ns = _controller()
    reads = iter([{}, {"publicIp": "8.8.8.8", "portMappings": {"22": 2222}}])
    ns["_request_json"] = lambda *args: next(reads)
    sleeps = _clock(ns)
    assert ns["_wait_for_ssh"]("secret", "pod-1") == ("8.8.8.8", 2222)
    assert sleeps == [5]
    _clock(ns, 1000)
    with pytest.raises(RuntimeError, match="15 minutes"):
        ns["_wait_for_ssh"]("secret", "pod-1")


def test_training_nonzero_is_not_copied_or_retried(monkeypatch, credentials):
    ns = _controller()
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1 if "input" in kwargs else 0)

    monkeypatch.setattr(ns["subprocess"], "run", run)
    with pytest.raises(RuntimeError, match="training failed"):
        ns["_train_and_copy"]("192.0.2.1", 2222, str(credentials))
    assert [argv[0] for argv in calls] == ["ssh", "ssh"]


def test_submit_keeps_one_controller_through_repeated_interrupts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(CONFIG, encoding="utf-8")
    plan = _backend().plan_runpod_run(
        "soup.yaml", gpu="a100", output_dir="out", soup_version="0.76.0"
    )
    waits = [KeyboardInterrupt(), SystemExit(2), 0]
    spawns = []

    class Process:
        def wait(self):
            value = waits.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: spawns.append((a, kw)) or Process())
    assert (
        _backend().submit_runpod_run(
            plan, env={"RUNPOD_API_KEY": "test", "RUNPOD_SSH_PRIVATE_KEY": "key"}
        )
        == 0
    )
    assert len(spawns) == 1
    assert waits == []


def test_missing_ssh_or_existing_output_never_launches(credentials):
    ns = _controller()
    calls = []
    ns["_request_json"] = lambda *a, **kw: calls.append((a, kw))
    ns["shutil"].which = lambda command: None
    assert ns["main"]() == 2
    ns["shutil"].which = lambda command: command
    Path("out").mkdir()
    assert ns["main"]() == 2
    assert not calls


@pytest.mark.requires_symlink
def test_local_copy_destination_refuses_symlinks(credentials, tmp_path):
    ns = _controller()
    (tmp_path / "real").mkdir()
    (tmp_path / "out").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinks"):
        ns["_local_destination"]()
