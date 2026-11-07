"""RunPod planning and generated controller behavior, without provider traffic."""

import base64
import io
import json
import subprocess
import types
import urllib.error
from pathlib import Path

import pytest
from typer.testing import CliRunner

CONFIG = "base: public/model\ndata: {train: public/dataset}\noutput: out\n"


def _backend():
    from soup_cli.cloud import runpod

    return runpod


#: Where the controllers built by ``_controller()`` find ssh and scp. A test of the
#: real PATH search uses ``_controller(real_tools=True)``.
FAKE_TOOL_DIR = "/fake-bin/"


def _controller(*, real_tools: bool = False):
    text = _backend().render_runpod_stub(
        CONFIG, gpu="a100", output_dir="out", soup_version="0.76.0"
    )
    namespace = {"__name__": "test_controller"}
    exec(compile(text, "soup_runpod_app.py", "exec"), namespace)
    namespace["shutil"] = types.SimpleNamespace(which=lambda command: command)
    if not real_tools:
        namespace["_find_tool"] = lambda name: FAKE_TOOL_DIR + name
    return namespace


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    private = tmp_path / "id_test"
    private.write_text("not-a-real-private-key", encoding="utf-8")
    Path(str(private) + ".pub").write_text("ssh-ed25519 AAAATEST fixture", encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-provider-secret")
    monkeypatch.setenv("RUNPOD_SSH_PRIVATE_KEY", str(private))
    return private


@pytest.mark.parametrize("value", [True, 1, "", "tpu", "a100\x00", "x" * 100])
def test_closed_gpu_allowlist(value):
    with pytest.raises(ValueError, match="gpu"):
        _backend().validate_gpu(value)


def test_render_is_data_only_and_uses_documented_rest_payload(credentials):
    ns = _controller()
    assert base64.b64decode(ns["_CONFIG_B64"]).decode() == CONFIG
    payload = ns["_launch_payload"]("ssh-ed25519 AAAATEST fixture", "soup-test")
    assert payload["gpuTypeIds"] == ["NVIDIA A100-SXM4-80GB"]
    assert payload["gpuCount"] == 1
    assert payload["ports"] == ["22/tcp"]
    assert payload["env"] == {"PUBLIC_KEY": "ssh-ed25519 AAAATEST fixture"}
    assert "test-provider-secret" not in json.dumps(payload)
    assert "not-a-real-private-key" not in json.dumps(payload)
    assert payload["dockerEntrypoint"] == ["/bin/bash", "-lc"]
    assert "sshd" in payload["dockerStartCmd"][0]


@pytest.mark.parametrize(
    "output", ["/tmp/out", "../out", "a/../../out", ".", "out space", "a;exit"]
)
def test_unsafe_remote_outputs_are_refused(output):
    with pytest.raises(ValueError, match="output_dir"):
        _backend().render_runpod_stub(CONFIG, gpu="a100", output_dir=output, soup_version="0.76.0")


@pytest.mark.parametrize(
    "config", ["hf_token: secret\n", "x: {api_key: secret}\n", "# test-provider-secret\n"]
)
def test_inline_secrets_are_refused(config, credentials):
    with pytest.raises(ValueError, match="credential"):
        _backend().render_runpod_stub(config, gpu="a100", output_dir="out", soup_version="0.76.0")


def test_malformed_config_is_refused_without_echoing_its_content():
    with pytest.raises(ValueError, match="valid YAML") as error:
        _backend().render_runpod_stub(
            "private-value: [", gpu="a100", output_dir="out", soup_version="0.76.0",
        )
    assert "private-value" not in str(error.value)


def test_modest_alias_reuse_is_preserved_as_config_data():
    config = CONFIG + "settings: &settings {epochs: 1}\ncopy: *settings\n"
    stub = _backend().render_runpod_stub(
        config, gpu="a100", output_dir="out", soup_version="0.76.0",
    )
    namespace = {"__name__": "test_controller"}
    exec(compile(stub, "soup_runpod_app.py", "exec"), namespace)
    assert base64.b64decode(namespace["_CONFIG_B64"]).decode() == config


@pytest.mark.parametrize("version", ["x; echo bad", "", "x\n", True])
def test_bad_package_version_is_refused(version):
    with pytest.raises(ValueError, match="soup_version"):
        _backend().render_runpod_stub(CONFIG, gpu="a100", output_dir="out", soup_version=version)


def test_plan_rejects_path_escape_and_writes_atomically(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(CONFIG, encoding="utf-8")
    backend = _backend()
    for output in ("out", "../outside"):
        if output == "out":
            plan = backend.plan_runpod_run(
                "soup.yaml", gpu="a100", output_dir=output, soup_version="0.76.0"
            )
            backend.write_stub(plan)
            assert (tmp_path / plan.stub_path).read_text(encoding="utf-8") == plan.stub_text
        else:
            with pytest.raises(ValueError, match="output_dir"):
                backend.plan_runpod_run(
                    "soup.yaml", gpu="a100", output_dir=output, soup_version="0.76.0"
                )


@pytest.mark.requires_symlink
def test_config_and_output_symlinks_are_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "real.yaml"
    target.write_text(CONFIG, encoding="utf-8")
    (tmp_path / "soup.yaml").symlink_to(target)
    with pytest.raises(ValueError):
        _backend().plan_runpod_run("soup.yaml", gpu="a100", output_dir="out", soup_version="0.76.0")
    (tmp_path / "soup.yaml").unlink()
    (tmp_path / "soup.yaml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "actual").mkdir()
    (tmp_path / "out").symlink_to(tmp_path / "actual", target_is_directory=True)
    with pytest.raises(ValueError):
        _backend().plan_runpod_run("soup.yaml", gpu="a100", output_dir="out", soup_version="0.76.0")


@pytest.mark.parametrize(
    "failure",
    [
        None,
        RuntimeError("secret from training"),
        KeyboardInterrupt(),
        subprocess.TimeoutExpired("ssh", 1),
        SystemExit(2),
    ],
)
def test_controller_terminates_and_confirms_even_after_failure(credentials, failure):
    ns = _controller()
    calls = []

    def request(key, path, *, method="GET", payload=None):
        calls.append((method, path))
        if method == "POST":
            return {"id": "pod-1"}
        if method == "GET":
            return None

    def training(*args):
        if failure is not None:
            raise failure

    ns["_request_json"] = request
    ns["_wait_for_ssh"] = lambda *args: ("192.0.2.1", 2222)
    ns["_train_and_copy"] = training
    ns["_notice"] = lambda message: calls.append(("notice", message))
    assert ns["main"]() == (0 if failure is None else 1)
    assert ("DELETE", "/pods/pod-1") in calls
    assert ("GET", "/pods/pod-1") in calls
    assert "secret from training" not in str(calls)


def test_cleanup_failure_cannot_report_success(credentials):
    ns = _controller()
    ns["_request_json"] = lambda *a, **kw: {"id": "pod-1"}
    ns["_wait_for_ssh"] = lambda *args: ("192.0.2.1", 2222)
    ns["_train_and_copy"] = lambda *args: None
    ns["_terminate"] = lambda *args: (_ for _ in ()).throw(RuntimeError("secret"))
    assert ns["main"]() == 1


def test_uncertain_launch_is_reconciled_by_unique_name(credentials):
    ns = _controller()
    calls = []
    name = []

    def request(key, path, *, method="GET", payload=None):
        calls.append((method, path))
        if method == "POST":
            name.append(payload["name"])
            raise RuntimeError("lost create response")
        if path == "/pods":
            return [{"id": "pod-1", "name": name[0]}, {"id": "other", "name": "unrelated"}]
        return None

    ns["_request_json"] = request
    assert ns["main"]() == 1
    assert ("DELETE", "/pods/pod-1") in calls
    assert ("DELETE", "/pods/other") not in calls


def test_rest_auth_uses_header_and_refuses_redirects(monkeypatch):
    ns = _controller()
    seen = []

    class Response(io.BytesIO):
        status = 200

    class Opener:
        def open(self, request, timeout):
            seen.append((request, timeout))
            return Response(b'{"id":"pod-1"}')

    def build_opener(*handlers):
        assert len(handlers) == 1
        assert isinstance(handlers[0], ns["_NoRedirect"])
        assert handlers[0].redirect_request(
            None, None, None, None, None, "https://evil.invalid",
        ) is None
        return Opener()

    monkeypatch.setattr(ns["urllib"].request, "build_opener", build_opener)
    assert ns["_request_json"]("provider-secret", "/pods/pod-1") == {"id": "pod-1"}
    request, timeout = seen[0]
    assert request.full_url == "https://rest.runpod.io/v1/pods/pod-1"
    assert request.get_header("Authorization") == "Bearer provider-secret"
    assert "provider-secret" not in request.full_url
    assert timeout == 60
    assert (
        ns["_NoRedirect"]().redirect_request(None, None, None, None, None, "https://evil.invalid")
        is None
    )


@pytest.mark.parametrize("status", [401, 500, 302])
def test_rest_error_details_never_expose_credentials(monkeypatch, status):
    ns = _controller()

    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, status, "secret", {}, io.BytesIO(b"secret")
            )

    monkeypatch.setattr(ns["urllib"].request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        ns["_request_json"]("secret", "/pods/pod-1")
    assert "secret" not in str(error.value)


def test_training_failure_or_copy_failure_is_not_success(credentials, monkeypatch, tmp_path):
    ns = _controller()
    monkeypatch.chdir(tmp_path)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0 if argv[0] == FAKE_TOOL_DIR + "ssh" else 1)

    monkeypatch.setattr(ns["subprocess"], "run", run)
    with pytest.raises(RuntimeError, match="copy"):
        ns["_train_and_copy"]("192.0.2.1", 2222, str(credentials))
    assert [item[0][0] for item in calls] == [
        FAKE_TOOL_DIR + "ssh", FAKE_TOOL_DIR + "ssh", FAKE_TOOL_DIR + "scp"
    ]
    assert "-P" in calls[2][0] and "2222" in calls[2][0]
    script = calls[1][1]["input"]
    assert "test-provider-secret" not in script
    assert "soup-cli[train]==0.76.0" in script


@pytest.mark.parametrize("submit,expected", [(False, 0), (True, 7)])
def test_real_cli_plan_and_submit_seam(tmp_path, monkeypatch, submit, expected):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(CONFIG, encoding="utf-8")
    plans = []
    monkeypatch.setattr(_backend(), "_RUNPOD_SUBMIT_OVERRIDE", lambda plan: plans.append(plan) or 7)
    args = ["train", "--cloud", "runpod", "--gpu", "a100", "--config", "soup.yaml"]
    result = CliRunner().invoke(app, args + (["--cloud-submit"] if submit else []))
    assert result.exit_code == expected, (result.output, repr(result.exception))
    assert bool(plans) == submit
    assert (tmp_path / "soup_runpod_app.py").is_file()


def test_submit_requires_credentials_before_spawning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(CONFIG, encoding="utf-8")
    plan = _backend().plan_runpod_run(
        "soup.yaml", gpu="a100", output_dir="out", soup_version="0.76.0"
    )
    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
        _backend().submit_runpod_run(plan, env={})
