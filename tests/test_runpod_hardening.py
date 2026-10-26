"""Run rendered RunPod controllers through failure and process boundaries."""

import http.client
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import urllib.error
from pathlib import Path

import pytest

from tests.test_runpod_cloud import CONFIG, FAKE_TOOL_DIR, _backend, _controller
from tests.test_runpod_controller import _clock


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key = tmp_path / "id_test"
    key.write_text("not-a-real-private-key", encoding="utf-8")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAATEST fixture", encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-provider-secret")
    monkeypatch.setenv("RUNPOD_SSH_PRIVATE_KEY", str(key))
    return key


@pytest.mark.parametrize("method", ["DELETE", "GET"])
@pytest.mark.parametrize("error", [
    http.client.IncompleteRead(b"private-response"),
    http.client.BadStatusLine("private-response"),
    http.client.LineTooLong("private-response"),
    ConnectionResetError("private-response"),
])
def test_cleanup_retries_transient_http_failures(monkeypatch, method, error):
    ns = _controller()
    sleeps = _clock(ns)
    calls = []
    failed = False

    class Opener:
        def open(self, request, timeout):
            nonlocal failed
            calls.append(request.method)
            if request.method == method and not failed:
                failed = True
                raise error
            if request.method == "GET":
                raise urllib.error.HTTPError(request.full_url, 404, "absent", {}, None)
            return io.BytesIO(b"")

    monkeypatch.setattr(ns["urllib"].request, "build_opener", lambda *handlers: Opener())
    ns["_terminate"]("provider-secret", "pod-1")
    expected = ["DELETE", "DELETE", "GET"] if method == "DELETE" else [
        "DELETE", "GET", "DELETE", "GET",
    ]
    assert calls == expected
    assert sleeps == [5]


@pytest.mark.parametrize("status", [400, 401, 403, 422, 429])
def test_definite_create_rejection_has_reason_and_no_false_cleanup(
    credentials, monkeypatch, status,
):
    ns = _controller()
    _clock(ns, 100)
    calls, notices = [], []

    class Opener:
        def open(self, request, timeout):
            calls.append(request.method)
            raise urllib.error.HTTPError(
                request.full_url, status, "private-response", {}, io.BytesIO(b"provider-secret"),
            )

    monkeypatch.setattr(ns["urllib"].request, "build_opener", lambda *handlers: Opener())
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    assert calls == ["POST"]
    assert any(f"RunPod API POST failed: HTTP {status}" in line for line in notices)
    assert "CRITICAL" not in str(notices)
    assert "termination verified" not in str(notices)
    assert "private-response" not in str(notices)
    assert "provider-secret" not in str(notices)


@pytest.mark.parametrize("failure", [500, 503, 408, "timeout", "incomplete", "json"])
def test_ambiguous_create_still_recovers_only_owned_pod(credentials, monkeypatch, failure):
    ns = _controller()
    _clock(ns)
    calls, notices = [], []
    run_name = None

    class Opener:
        def open(self, request, timeout):
            nonlocal run_name
            path = request.full_url.removeprefix(ns["_API_BASE"])
            calls.append((request.method, path))
            if request.method == "POST":
                run_name = json.loads(request.data)["name"]
                if isinstance(failure, int):
                    raise urllib.error.HTTPError(request.full_url, failure, "secret", {}, None)
                if failure == "timeout":
                    raise TimeoutError("provider-secret")
                if failure == "incomplete":
                    raise http.client.IncompleteRead(b"provider-secret")
                return io.BytesIO(b"invalid json: provider-secret")
            if path == "/pods":
                return io.BytesIO(json.dumps([
                    {"id": "owned", "name": run_name}, {"id": "other", "name": "other"},
                ]).encode())
            if request.method == "DELETE":
                return io.BytesIO(b"")
            raise urllib.error.HTTPError(request.full_url, 404, "absent", {}, None)

    monkeypatch.setattr(ns["urllib"].request, "build_opener", lambda *handlers: Opener())
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    assert calls == [("POST", "/pods"), ("GET", "/pods"),
                     ("DELETE", "/pods/owned"), ("GET", "/pods/owned")]
    assert "RunPod termination verified." in notices
    assert "provider-secret" not in str(notices)


@pytest.mark.parametrize("stage,reason", [
    ("ssh", "did not expose SSH within 15 minutes"),
    ("training", "remote training failed"),
    ("copy", "artifact copy failed"),
])
def test_main_reports_safe_controller_failure_reason(credentials, monkeypatch, stage, reason):
    ns = _controller()
    _clock(ns)
    notices, deletes = [], []

    def request(*args, method="GET", **kwargs):
        if method == "POST":
            return {"id": "pod-1"}
        if method == "DELETE":
            deletes.append(args[1])
        return {} if stage == "ssh" and not deletes else None

    def run(argv, **kwargs):
        failed = (stage == "training" and "input" in kwargs) or (
            stage == "copy" and argv[0] == FAKE_TOOL_DIR + "scp"
        )
        return subprocess.CompletedProcess(argv, int(failed))

    ns["_request_json"] = request
    ns["_notice"] = notices.append
    if stage != "ssh":
        ns["_wait_for_ssh"] = lambda *args: ("8.8.8.8", 2222)
    monkeypatch.setattr(ns["subprocess"], "run", run)
    assert ns["main"]() == 1
    assert any(reason in line for line in notices)
    assert deletes == ["/pods/pod-1"]
    assert "RunPod termination verified." in notices


@pytest.mark.parametrize("error", [RuntimeError("private-error"), ValueError("private-error")])
def test_main_does_not_print_untrusted_exception_text(credentials, error):
    ns = _controller()
    notices = []
    ns["_request_json"] = lambda *a, method="GET", **kw: {"id": "pod-1"} if (
        method == "POST"
    ) else None
    ns["_wait_for_ssh"] = lambda *args: ("8.8.8.8", 2222)

    def train(*args):
        raise error

    ns["_train_and_copy"] = train
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    assert "private-error" not in str(notices)
    assert "RunPod termination verified." in notices


def test_output_appearing_during_training_prevents_scp(credentials, monkeypatch):
    ns = _controller()
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "input" in kwargs:
            Path("out").mkdir()
            Path("out/keep").write_text("original", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ns["subprocess"], "run", run)
    with pytest.raises(RuntimeError, match="already exists"):
        ns["_train_and_copy"]("8.8.8.8", 2222, str(credentials))
    assert [argv[0] for argv in calls] == [FAKE_TOOL_DIR + "ssh", FAKE_TOOL_DIR + "ssh"]
    assert Path("out/keep").read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cleanup_masks_ctrl_c_and_restores_previous_handler(credentials, cleanup_fails):
    ns = _controller()
    previous = object()
    handlers = [previous]
    calls = []

    def install(sig, handler):
        assert sig == "INT"
        old = handlers[-1]
        handlers.append(handler)
        return old

    def terminate(*args):
        assert handlers[-1] == "IGNORE"
        calls.append(args)
        if cleanup_fails:
            raise RuntimeError("private-error")

    def interrupted(*args):
        raise KeyboardInterrupt

    ns["signal"] = types.SimpleNamespace(SIGINT="INT", SIG_IGN="IGNORE", signal=install)
    ns["_request_json"] = lambda *a, **kw: {"id": "pod-1"}
    ns["_wait_for_ssh"] = lambda *args: ("8.8.8.8", 2222)
    ns["_train_and_copy"] = interrupted
    ns["_terminate"] = terminate
    assert ns["main"]() == 1
    assert calls == [("test-provider-secret", "pod-1")]
    assert handlers == [previous, "IGNORE", previous]


def test_readiness_retries_before_training_once(credentials, monkeypatch):
    ns = _controller()
    sleeps = _clock(ns)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == FAKE_TOOL_DIR + "scp":
            (Path(argv[-1]) / "out").mkdir()
        return subprocess.CompletedProcess(argv, 255 if len(calls) == 1 else 0)

    monkeypatch.setattr(ns["subprocess"], "run", run)
    ns["_train_and_copy"]("8.8.8.8", 2222, str(credentials))
    ssh, scp = FAKE_TOOL_DIR + "ssh", FAKE_TOOL_DIR + "scp"
    assert [argv[0] for argv, _ in calls] == [ssh, ssh, ssh, scp]
    assert calls[0][0][-1] == calls[1][0][-1] == "true"
    assert sum("input" in kwargs for _, kwargs in calls) == 1
    assert sleeps == [5]


@pytest.mark.parametrize("pod_id", ["../x", "x" * 65, 17, None, "", True])
def test_invalid_pod_id_never_reaches_delete(pod_id):
    ns = _controller()
    calls = []
    ns["_request_json"] = lambda *a, **kw: calls.append((a, kw))
    with pytest.raises(RuntimeError, match="invalid Pod ID"):
        ns["_terminate"]("secret", pod_id)
    assert calls == []


@pytest.mark.parametrize("key", [
    "ecdsa-sha2-nistp256 AAAATEST", "ssh-ed25519 AAAA\nssh-rsa BBBB", "garbage",
])
def test_invalid_public_key_never_allocates(credentials, key):
    Path(str(credentials) + ".pub").write_text(key, encoding="utf-8")
    ns = _controller()
    calls = []
    ns["_request_json"] = lambda *a, **kw: calls.append((a, kw))
    assert ns["main"]() == 2
    assert calls == []


@pytest.mark.parametrize("kind", ["ed25519", "rsa"])
def test_supported_public_keys_launch_secure_cloud(credentials, kind):
    Path(str(credentials) + ".pub").write_text(f"ssh-{kind} AAAATEST fixture", encoding="utf-8")
    ns = _controller()
    payloads = []

    def request(*args, method="GET", payload=None):
        if method == "POST":
            payloads.append(payload)
            return {"id": "pod-1"}

    ns["_request_json"] = request
    ns["_wait_for_ssh"] = lambda *args: ("8.8.8.8", 2222)
    ns["_train_and_copy"] = lambda *args: None
    assert ns["main"]() == 0
    assert len(payloads) == 1
    assert payloads[0]["cloudType"] == "SECURE"
    assert payloads[0]["env"] == {"PUBLIC_KEY": f"ssh-{kind} AAAATEST fixture"}


def test_realpath_escape_is_refused_before_allocation(credentials, monkeypatch, tmp_path):
    ns = _controller()
    realpath = os.path.realpath
    monkeypatch.setattr(ns["os"].path, "realpath", lambda path, **kw: (
        str(tmp_path.parent / "escaped") if str(path) == "out" else realpath(path, **kw)
    ))
    with pytest.raises(RuntimeError, match="stay under cwd"):
        ns["_local_destination"]()


@pytest.mark.parametrize("output", ["-out", "a/-out", "a/-b/out"])
def test_option_like_output_component_refused_when_rendering(output):
    with pytest.raises(ValueError, match="output_dir"):
        _backend().render_runpod_stub(CONFIG, gpu="a100", output_dir=output, soup_version="0.76.0")


@pytest.mark.parametrize("output", ["-out", "a/-out", "a/-b/out"])
def test_option_like_output_component_refused_by_controller(credentials, output):
    ns = _controller()
    ns["_OUTPUT"] = output
    with pytest.raises(RuntimeError, match="output"):
        ns["_local_destination"]()


def _plan(tmp_path):
    (tmp_path / "soup.yaml").write_text(CONFIG, encoding="utf-8")
    return _backend().plan_runpod_run(
        "soup.yaml", gpu="a100", output_dir="out", soup_version="0.76.0",
    )


def test_submit_isolates_python_and_filters_environment(credentials, monkeypatch, tmp_path):
    plan = _plan(tmp_path)
    environment = {name: f"value-{name}" for name in (
        "RUNPOD_API_KEY", "RUNPOD_SSH_PRIVATE_KEY", "PATH", "HOME", "USERPROFILE", "SYSTEMROOT",
    )}
    untrusted = {name: "private-value" for name in (
        "PYTHONPATH", "PYTHONHOME", "HF_TOKEN", "WANDB_API_KEY", "LD_PRELOAD",
    )}
    spawns = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: (
        spawns.append((argv, kw)) or types.SimpleNamespace(wait=lambda: 7)
    ))
    assert _backend().submit_runpod_run(plan, env={**environment, **untrusted}) == 7
    assert spawns == [([sys.executable, "-I", plan.stub_path], {"env": environment})]
    assert plan.run_command == "python -I soup_runpod_app.py"


@pytest.mark.parametrize("name", [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "TEMP", "TMP", "TMPDIR", "SYSTEMDRIVE", "WINDIR", "SSH_AUTH_SOCK",
    "PATHEXT", "NoDefaultCurrentDirectoryInExePath",
])
def test_controller_forwards_network_temp_and_agent_settings(
    credentials, monkeypatch, tmp_path, name,
):
    plan = _plan(tmp_path)
    supplied = {
        "RUNPOD_API_KEY": "fixture-key", "RUNPOD_SSH_PRIVATE_KEY": str(credentials),
        name: "setting-value", "HF_TOKEN": "not-for-this-controller",
    }
    expected = {key: value for key, value in supplied.items() if key != "HF_TOKEN"}
    # Check exact POSIX spellings too: on Windows a differently cased proxy name
    # can otherwise conceal a missing allow-list entry after normalisation.
    assert _backend()._controller_environment(supplied, case_insensitive=False) == expected
    launches = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: (
        launches.append((argv, kw)) or types.SimpleNamespace(wait=lambda: 0)
    ))
    assert _backend().submit_runpod_run(plan, env=supplied) == 0
    assert launches == [([sys.executable, "-I", plan.stub_path], {"env": expected})]


@pytest.mark.parametrize("case_insensitive", [False, True, None])
def test_controller_environment_matches_windows_names_only_on_windows(case_insensitive):
    supplied = {"Path": "bin", "SystemRoot": "windows", "Https_Proxy": "proxy", "OTHER": "no"}
    expected = {key: value for key, value in supplied.items() if key != "OTHER"}
    windows = os.name == "nt" if case_insensitive is None else case_insensitive
    assert _backend()._controller_environment(
        supplied, case_insensitive=case_insensitive,
    ) == (expected if windows else {})


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("mode", ["ascii", "native"])
def test_isolated_controller_replaces_unencodable_output(tmp_path, monkeypatch, stream, mode):
    monkeypatch.chdir(tmp_path)
    plan = _plan(tmp_path)
    _backend().write_stub(plan)
    # Run the rendered controller under -I, not a replica of its bootstrap.
    # The narrow mode models Windows redirected output on every platform.
    harness = tmp_path / "encoding_harness.py"
    harness.write_text(
        "import sys\n"
        "stream, mode, stub = sys.argv[1:]\n"
        "if mode == 'ascii':\n"
        "    for handle in (sys.stdout, sys.stderr):\n"
        "        handle.reconfigure(encoding='ascii', errors='strict')\n"
        "namespace = {'__name__': 'controller_under_test'}\n"
        "with open(stub, encoding='utf-8') as source:\n"
        "    exec(compile(source.read(), stub, 'exec'), namespace)\n"
        "getattr(sys, stream).write('controller line: \\u2588 \\u00e9 \\u65e5\\n')\n"
        "getattr(sys, stream).flush()\n",
        encoding="utf-8",
    )
    environment = {name: value for name, value in os.environ.items() if name.upper() in {
        "PATH", "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "TEMP", "TMP", "TMPDIR",
    }}
    result = subprocess.run(
        [sys.executable, "-I", str(harness), stream, mode, plan.stub_path],
        env=environment, cwd=tmp_path, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = getattr(result, stream).decode("ascii" if mode == "ascii" else "utf-8", "replace")
    assert "controller line:" in output
    if mode == "ascii":
        assert "controller line: ? ? ?" in output
    assert "UnicodeEncodeError" not in result.stderr.decode("utf-8", "replace")


def test_actual_child_cannot_import_parent_pythonpath_or_read_unrelated_secret(
    credentials, tmp_path,
):
    plan = _plan(tmp_path)
    _backend().write_stub(plan)
    # Execute an inert probe in the same process-launch path; no provider traffic.
    Path(plan.stub_path).write_text(
        "import importlib.util, json, os, sys\n"
        "from pathlib import Path\n"
        "Path('child.json').write_text(json.dumps({"
        "'isolated': sys.flags.isolated, 'secret': os.environ.get('HF_TOKEN'), "
        "'injected': importlib.util.find_spec('parent_injection') is not None, "
        "'key': os.environ.get('RUNPOD_API_KEY')}))\n", encoding="utf-8",
    )
    (tmp_path / "parent_injection.py").write_text(
        "raise AssertionError('injected')", encoding="utf-8",
    )
    environment = {name: os.environ[name] for name in ("PATH", "SYSTEMROOT") if name in os.environ}
    environment.update(RUNPOD_API_KEY="fixture-key", RUNPOD_SSH_PRIVATE_KEY=str(credentials),
                       PYTHONPATH=str(tmp_path), HF_TOKEN="private-parent-secret")
    assert _backend().submit_runpod_run(plan, env=environment) == 0
    assert json.loads(Path("child.json").read_text(encoding="utf-8")) == {
        "isolated": 1, "secret": None, "injected": False, "key": "fixture-key",
    }


@pytest.mark.parametrize("failure", [None, "training", "copy", "interrupt"])
def test_ssh_uses_per_run_hosts_and_only_selected_identity(
    credentials, monkeypatch, tmp_path, failure,
):
    ns = _controller()
    scratch = tmp_path / "ssh scratch"
    scratch.mkdir()
    user_hosts = tmp_path / ".ssh/known_hosts"
    user_hosts.parent.mkdir()
    user_hosts.write_text("keep user hosts", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    ns["tempfile"] = types.SimpleNamespace(TemporaryDirectory=lambda **kw: (
        tempfile.TemporaryDirectory(dir=scratch, **kw)
    ), mkdtemp=tempfile.mkdtemp)
    all_hosts = []
    for _ in range(2):
        hosts, calls = [], []

        def run(argv, **kwargs):
            calls.append(argv)
            assert "IdentitiesOnly=yes" in argv
            options = [arg for arg in argv if arg.startswith("UserKnownHostsFile=")]
            assert len(options) == 1
            host_file = Path(options[0].split("=", 1)[1].strip('"'))
            assert host_file.parent.is_dir()
            assert host_file != user_hosts
            hosts.append(host_file)
            host_file.touch()
            if "input" in kwargs and failure == "interrupt":
                raise KeyboardInterrupt
            failed = ("input" in kwargs and failure == "training") or (
                argv[0] == FAKE_TOOL_DIR + "scp" and failure == "copy"
            )
            if argv[0] == FAKE_TOOL_DIR + "scp":
                assert "--" in argv
                delimiter = argv.index("--")
                assert argv[delimiter + 1] == "root@8.8.8.8:/workspace/soup/out"
                assert len(argv) == delimiter + 3
                staging = Path(argv[-1])
                assert staging.name.startswith(".soup-runpod-copy-"), staging
                assert staging.parent.resolve() == tmp_path.resolve()
                assert list(staging.iterdir()) == []
                if not failed:
                    (staging / "out").mkdir()
            return subprocess.CompletedProcess(argv, int(failed))

        monkeypatch.setattr(ns["subprocess"], "run", run)
        if failure:
            with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
                ns["_train_and_copy"]("8.8.8.8", 2222, str(credentials))
        else:
            ns["_train_and_copy"]("8.8.8.8", 2222, str(credentials))
        assert len(set(hosts)) == 1
        assert not hosts[0].parent.exists()
        ssh, scp = FAKE_TOOL_DIR + "ssh", FAKE_TOOL_DIR + "scp"
        assert [argv[0] for argv in calls] == (
            [ssh, ssh] if failure in {"training", "interrupt"} else [ssh, ssh, scp]
        )
        if failure is None:
            assert Path("out").is_dir()
            shutil.rmtree("out")
        all_hosts.append(hosts[0])
    assert len(set(all_hosts)) == 2
    assert user_hosts.read_text(encoding="utf-8") == "keep user hosts"
