"""Lifecycle boundaries exercised by running rendered controller functions."""

import io
import os
import subprocess
import types
import urllib.error
from pathlib import Path

import pytest

from tests.test_runpod_cloud import CONFIG, FAKE_TOOL_DIR, _backend, _controller


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
    assert [argv[0] for argv in calls] == [FAKE_TOOL_DIR + "ssh", FAKE_TOOL_DIR + "ssh"]


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
    ns["_find_tool"] = lambda command: None
    assert ns["main"]() == 2
    ns["_find_tool"] = lambda command: FAKE_TOOL_DIR + command
    Path("out").mkdir()
    assert ns["main"]() == 2
    assert not calls


def _program(directory: Path, name: str) -> Path:
    """An executable file the PATH search can find, under this OS's naming."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + (".EXE" if os.name == "nt" else ""))
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class TestExternalToolsRunByAbsolutePath:
    """ssh and scp are found on PATH only and run by absolute path."""

    def test_the_path_entry_wins_over_a_program_in_the_current_directory(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _program(tmp_path, "ssh")
        _program(tmp_path / "rel", "ssh")
        installed = _program(tmp_path / "bin", "ssh")
        monkeypatch.setenv("PATH", os.pathsep.join([".", "rel", "", str(tmp_path / "bin")]))
        ns = _controller(real_tools=True)
        assert ns["_find_tool"]("ssh") == os.path.abspath(installed)

    def test_a_program_only_in_the_current_directory_is_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _program(tmp_path, "ssh")
        (tmp_path / "bin").mkdir()
        monkeypatch.setenv("PATH", os.pathsep.join([".", str(tmp_path / "bin")]))
        ns = _controller(real_tools=True)
        assert ns["_find_tool"]("ssh") is None

    def test_ssh_and_scp_get_the_environment_without_the_api_key(
        self, credentials, monkeypatch
    ):
        ns = _controller()
        envs = []

        def run(argv, **kwargs):
            envs.append(kwargs["env"])
            if argv[0] == FAKE_TOOL_DIR + "scp":
                (Path(argv[-1]) / "out").mkdir()
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(ns["subprocess"], "run", run)
        ns["_train_and_copy"]("192.0.2.1", 2222, str(credentials))
        assert len(envs) == 3
        for env in envs:
            assert "RUNPOD_API_KEY" not in {name.upper() for name in env}
            assert env.get("RUNPOD_SSH_PRIVATE_KEY") == str(credentials)
            if os.name == "nt":
                assert env["NoDefaultCurrentDirectoryInExePath"] == "1"


class TestResultsAreFetchedIntoADedicatedDirectory:
    """scp writes into a directory created for the copy; only the expected entry
    is then moved to the output path."""

    def _staging(self, tmp_path) -> Path:
        staging = tmp_path / ".soup-runpod-copy-test"
        (staging / "out" / "adapter").mkdir(parents=True)
        (staging / "out" / "adapter" / "weights.bin").write_bytes(b"w")
        return staging

    def test_the_expected_entry_becomes_the_output(self, credentials, tmp_path):
        ns = _controller()
        staging = self._staging(tmp_path)
        ns["_move_copied_output"](staging, Path("out"))
        assert (tmp_path / "out" / "adapter" / "weights.bin").read_bytes() == b"w"
        assert not staging.exists()

    def test_an_extra_entry_moves_nothing(self, credentials, tmp_path):
        ns = _controller()
        staging = self._staging(tmp_path)
        (staging / "soup.yaml").write_text("x", encoding="utf-8")
        with pytest.raises(RuntimeError) as excinfo:
            ns["_move_copied_output"](staging, Path("out"))
        assert str(excinfo.value) == (
            "RunPod artifact copy held unexpected entries; nothing was moved to the output path"
        )
        assert not (tmp_path / "out").exists()

    @pytest.mark.requires_symlink
    def test_a_link_inside_moves_nothing(self, credentials, tmp_path):
        ns = _controller()
        staging = self._staging(tmp_path)
        (staging / "out" / "link").symlink_to(tmp_path / "id_test")
        with pytest.raises(RuntimeError) as excinfo:
            ns["_move_copied_output"](staging, Path("out"))
        assert str(excinfo.value) == (
            "RunPod artifact copy held a link or special file; nothing was moved to the "
            "output path"
        )
        assert not (tmp_path / "out").exists()

    def test_a_directory_junction_inside_moves_nothing(self, credentials, tmp_path, monkeypatch):
        # A Windows directory junction reads as a directory whose attributes carry
        # FILE_ATTRIBUTE_REPARSE_POINT (0x400); the attribute is set by hand here so
        # the check runs on every OS.
        ns = _controller()
        staging = self._staging(tmp_path)
        junction = os.path.normcase(os.path.abspath(staging / "out" / "adapter"))
        real_lstat = os.lstat

        def lstat(path, *args, **kwargs):
            info = real_lstat(path, *args, **kwargs)
            if os.path.normcase(os.path.abspath(path)) == junction:
                return types.SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info

        monkeypatch.setattr(ns["os"], "lstat", lstat)
        with pytest.raises(RuntimeError) as excinfo:
            ns["_move_copied_output"](staging, Path("out"))
        assert str(excinfo.value) == (
            "RunPod artifact copy held a link or special file; nothing was moved to the "
            "output path"
        )
        assert not (tmp_path / "out").exists()
        assert (staging / "out" / "adapter" / "weights.bin").read_bytes() == b"w"

    @staticmethod
    def _failing_copy(ns, monkeypatch, how, write):
        """scp fails (``how`` = "exit") or times out (``how`` = "timeout") after
        ``write`` ran on the directory it was given; returns that directory."""
        destinations = []

        def run(argv, **kwargs):
            if argv[0] == FAKE_TOOL_DIR + "scp":
                destinations.append(Path(argv[-1]))
                write(Path(argv[-1]))
                if how == "timeout":
                    raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
                return subprocess.CompletedProcess(argv, 1)
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(ns["subprocess"], "run", run)
        return destinations

    _REASONS = {"exit": "RunPod artifact copy failed", "timeout": "RunPod artifact copy timed out"}

    @pytest.mark.parametrize("how", ["exit", "timeout"])
    def test_a_failed_copy_that_wrote_nothing_removes_its_directory(
        self, credentials, tmp_path, monkeypatch, how
    ):
        ns = _controller()
        destinations = self._failing_copy(ns, monkeypatch, how, lambda directory: None)
        with pytest.raises(RuntimeError) as excinfo:
            ns["_train_and_copy"]("192.0.2.1", 2222, str(credentials))
        assert str(excinfo.value) == self._REASONS[how]
        assert len(destinations) == 1
        assert not destinations[0].exists()
        assert sorted(path.name for path in tmp_path.iterdir()) == ["id_test", "id_test.pub"]

    @pytest.mark.parametrize("how", ["exit", "timeout"])
    def test_a_failed_copy_that_wrote_a_part_keeps_it_and_names_its_directory(
        self, credentials, tmp_path, monkeypatch, how
    ):
        def write(directory):
            (directory / "out").mkdir()
            (directory / "out" / "part.bin").write_bytes(b"p")

        ns = _controller()
        destinations = self._failing_copy(ns, monkeypatch, how, write)
        with pytest.raises(RuntimeError) as excinfo:
            ns["_train_and_copy"]("192.0.2.1", 2222, str(credentials))
        staging = destinations[0]
        assert str(excinfo.value) == (
            f"{self._REASONS[how]}; the copy directory is left in place: {staging}"
        )
        assert (staging / "out" / "part.bin").read_bytes() == b"p"
        assert not (tmp_path / "out").exists()

    def test_a_failed_copy_whose_directory_is_gone_names_no_directory(
        self, credentials, monkeypatch
    ):
        ns = _controller()
        self._failing_copy(ns, monkeypatch, "exit", os.rmdir)
        with pytest.raises(RuntimeError) as excinfo:
            ns["_train_and_copy"]("192.0.2.1", 2222, str(credentials))
        assert str(excinfo.value) == "RunPod artifact copy failed"

    def test_a_timed_out_copy_is_reported_by_name_and_the_pod_is_deleted(
        self, credentials, monkeypatch
    ):
        ns = _controller()
        notices, deletes = [], []

        def request(*args, method="GET", **kwargs):
            if method == "POST":
                return {"id": "pod-1"}
            if method == "DELETE":
                deletes.append(args[1])
            return None

        ns["_request_json"] = request
        ns["_wait_for_ssh"] = lambda *args: ("8.8.8.8", 2222)
        ns["_notice"] = notices.append
        self._failing_copy(ns, monkeypatch, "timeout", lambda directory: None)
        assert ns["main"]() == 1
        assert "RunPod artifact copy timed out" in notices
        assert deletes == ["/pods/pod-1"]
        assert "RunPod termination verified." in notices

    def test_scp_writes_into_a_fresh_directory_not_the_working_directory(
        self, credentials, tmp_path, monkeypatch
    ):
        ns = _controller()
        destinations = []

        def run(argv, **kwargs):
            if argv[0] == FAKE_TOOL_DIR + "scp":
                destinations.append(Path(argv[-1]))
                assert list(Path(argv[-1]).iterdir()) == []
                (Path(argv[-1]) / "out").mkdir()
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(ns["subprocess"], "run", run)
        ns["_train_and_copy"]("192.0.2.1", 2222, str(credentials))
        assert len(destinations) == 1
        assert destinations[0].resolve() != tmp_path.resolve()
        assert destinations[0].parent.resolve() == tmp_path.resolve()
        assert (tmp_path / "out").is_dir()


@pytest.mark.requires_symlink
def test_local_copy_destination_refuses_symlinks(credentials, tmp_path):
    ns = _controller()
    (tmp_path / "real").mkdir()
    (tmp_path / "out").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinks"):
        ns["_local_destination"]()
