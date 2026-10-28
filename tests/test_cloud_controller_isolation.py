"""The local Lambda controller starts in an isolated interpreter with a minimal environment.

``submit_lambda_run`` starts the generated controller as
``[sys.executable, "-I", <stub>]`` and hands it only the variables it reads
(the Lambda settings), the ones ``ssh`` / ``scp`` need (``PATH``, the home and
temp directories, ``SYSTEMROOT`` on Windows) and the proxy / certificate / agent
variables the network calls honour. Everything else in the parent environment
stays in the parent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_LAMBDA_ENV = {
    "LAMBDA_API_KEY": "test-key",
    "LAMBDA_SSH_KEY_NAME": "test-ssh-key",
    "LAMBDA_SSH_PRIVATE_KEY": "/tmp/test-private-key",
}


def _plan(stub_path: str = "x.py"):
    from soup_cli.cloud._common import CloudPlan

    return CloudPlan(
        cloud="lambda",
        gpu="a100",
        output_dir="./out",
        stub_path=stub_path,
        stub_text="",
        run_command=f"python {stub_path}",
    )


class _RecordingPopen:
    """Stands in for ``subprocess.Popen`` and records how the controller is started."""

    launches: list[_RecordingPopen] = []

    def __init__(self, argv, **kwargs):
        self.argv = list(argv)
        self.kwargs = kwargs
        self.env = kwargs.get("env")
        _RecordingPopen.launches.append(self)

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def launched(monkeypatch):
    """Run ``submit_lambda_run`` against a recording Popen and return the launch."""
    _RecordingPopen.launches = []
    monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)

    def run(env=None):
        from soup_cli.cloud.lambda_labs import submit_lambda_run

        if env is None:
            assert submit_lambda_run(_plan()) == 0
        else:
            assert submit_lambda_run(_plan(), env=env) == 0
        assert len(_RecordingPopen.launches) == 1
        return _RecordingPopen.launches[0]

    return run


def _set_lambda_env(monkeypatch) -> None:
    for name, value in _LAMBDA_ENV.items():
        monkeypatch.setenv(name, value)


def test_controller_argv_starts_with_the_isolated_interpreter_flag(launched, monkeypatch):
    _set_lambda_env(monkeypatch)

    launch = launched()

    assert launch.argv[0] == sys.executable
    assert launch.argv[1] == "-I"
    assert launch.argv == [sys.executable, "-I", "x.py"]


def test_printed_manual_run_command_uses_the_isolated_interpreter_flag(tmp_path, monkeypatch):
    from soup_cli.cloud.lambda_labs import plan_lambda_run

    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text("base: x\n", encoding="utf-8")

    plan = plan_lambda_run(
        "soup.yaml", gpu="a100", output_dir="./out", soup_version="0.76.0", stub_path="ctl.py"
    )

    # The command a user is told to run starts the controller the way
    # ``--cloud-submit`` does: ``python -I <stub>``.
    assert plan.run_command == "python -I ctl.py"


def test_controller_gets_the_lambda_and_os_variables_it_reads(launched, monkeypatch):
    _set_lambda_env(monkeypatch)
    monkeypatch.setenv("LAMBDA_REGION", "us-west-1")
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setenv("HOME", "/home/someone")
    monkeypatch.setenv("USERPROFILE", "C:\\Users\\someone")
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    monkeypatch.setenv("TEMP", "/tmp/t")
    monkeypatch.setenv("TMP", "/tmp/t")

    env = launched().env

    assert env["LAMBDA_API_KEY"] == "test-key"
    assert env["LAMBDA_SSH_KEY_NAME"] == "test-ssh-key"
    assert env["LAMBDA_SSH_PRIVATE_KEY"] == "/tmp/test-private-key"
    assert env["LAMBDA_REGION"] == "us-west-1"
    assert env["PATH"] == os.environ["PATH"]
    for name in ("HOME", "USERPROFILE", "SYSTEMROOT", "TEMP", "TMP"):
        assert name in env, f"{name} must reach ssh / scp and the interpreter"


def test_controller_env_leaves_an_unrelated_parent_variable_behind(launched, monkeypatch):
    _set_lambda_env(monkeypatch)
    monkeypatch.setenv("SOUP_TEST_UNRELATED", "1")

    env = launched().env

    assert "SOUP_TEST_UNRELATED" not in env
    assert env["LAMBDA_API_KEY"] == "test-key"


def test_controller_env_is_built_from_a_supplied_mapping_too(launched):
    supplied = dict(_LAMBDA_ENV, SOUP_TEST_UNRELATED="1", PATH=os.environ["PATH"])

    env = launched(env=supplied).env

    assert "SOUP_TEST_UNRELATED" not in env
    assert env["PATH"] == os.environ["PATH"]
    assert env["LAMBDA_API_KEY"] == "test-key"


def test_controller_env_is_a_new_mapping_not_the_parent_environment(launched, monkeypatch):
    _set_lambda_env(monkeypatch)

    env = launched().env

    assert env is not os.environ
    assert isinstance(env, dict)
    assert set(env) <= set(os.environ)
    # Whatever else the parent has, the controller sees no more than the allowlist.
    from soup_cli.cloud.lambda_labs import _controller_environment

    assert set(env) == set(_controller_environment(os.environ))


@pytest.mark.parametrize(
    "name",
    ["HF_TOKEN", "WANDB_API_KEY", "MODAL_TOKEN_SECRET", "PYTHONPATH", "PYTHONSTARTUP"],
)
def test_other_tool_settings_in_the_parent_are_not_forwarded(launched, monkeypatch, name):
    _set_lambda_env(monkeypatch)
    monkeypatch.setenv(name, "value")

    assert name not in launched().env


@pytest.mark.parametrize(
    "name",
    [
        "SSH_AUTH_SOCK",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTPS_PROXY",
        "https_proxy",
        "NO_PROXY",
        "no_proxy",
        "LANG",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
    ],
)
def test_network_agent_locale_and_library_settings_are_forwarded(launched, monkeypatch, name):
    _set_lambda_env(monkeypatch)
    monkeypatch.setenv(name, "value")

    env = launched().env

    # os.environ upper-cases names on Windows, so ``https_proxy`` arrives there
    # as ``HTTPS_PROXY``; everywhere else the name is kept as it was set.
    assert env[name.upper() if os.name == "nt" else name] == "value"


class TestControllerEnvironmentNameMatching:
    def test_windows_names_are_matched_without_regard_to_case(self):
        from soup_cli.cloud.lambda_labs import _controller_environment

        env = _controller_environment(
            {"SystemRoot": "C:\\Windows", "Path": "C:\\bin", "Other": "1"},
            case_insensitive=True,
        )

        assert env == {"SystemRoot": "C:\\Windows", "Path": "C:\\bin"}

    def test_other_platforms_match_names_exactly(self):
        from soup_cli.cloud.lambda_labs import _controller_environment

        env = _controller_environment(
            {"SystemRoot": "x", "Path": "y", "PATH": "z", "HOME": "h"},
            case_insensitive=False,
        )

        assert env == {"PATH": "z", "HOME": "h"}

    def test_default_follows_the_platform(self):
        from soup_cli.cloud.lambda_labs import _controller_environment

        env = _controller_environment({"path": "p", "PATH": "P"})

        if os.name == "nt":
            assert env == {"path": "p", "PATH": "P"}
        else:
            assert env == {"PATH": "P"}


# The generated controller must start under ``-I`` with the minimal environment:
# it imports the standard library only and reads nothing it was not given. These
# runs stop at the controller's own argument checks, so no network call is made.
def _render_controller(tmp_path: Path) -> Path:
    from soup_cli.cloud.lambda_labs import render_lambda_stub

    stub = tmp_path / "soup_lambda_app.py"
    stub.write_text(
        render_lambda_stub(
            "base: x\n", gpu="a100", output_dir="./out", soup_version="0.76.0"
        ),
        encoding="utf-8",
    )
    return stub


def _parent_env(**extra: str) -> dict[str, str]:
    """The test process environment without any Lambda setting, plus ``extra``.

    A developer machine may have the real ``LAMBDA_*`` variables set; the
    controller runs below must never reach the network because of them.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("LAMBDA_") and key.upper() != "PYTHONSAFEPATH"
    }
    env.update(extra)
    return env


def _run_controller(stub: Path, parent_env: dict[str, str]) -> subprocess.CompletedProcess:
    from soup_cli.cloud.lambda_labs import _controller_environment

    return subprocess.run(
        [sys.executable, "-I", str(stub)],
        env=_controller_environment(parent_env),
        cwd=stub.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_rendered_controller_starts_under_isolated_mode(tmp_path):
    stub = _render_controller(tmp_path)

    result = _run_controller(stub, _parent_env())

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "Set LAMBDA_API_KEY, LAMBDA_SSH_KEY_NAME, and LAMBDA_SSH_PRIVATE_KEY." in result.stdout


def test_rendered_controller_reads_its_settings_from_the_minimal_environment(tmp_path):
    stub = _render_controller(tmp_path)
    parent_env = _parent_env(
        SOUP_TEST_UNRELATED="1",
        LAMBDA_API_KEY="k",
        LAMBDA_SSH_KEY_NAME="n",
        LAMBDA_SSH_PRIVATE_KEY=str(tmp_path / "missing-key"),
    )

    result = _run_controller(stub, parent_env)

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "LAMBDA_SSH_PRIVATE_KEY does not name a readable file." in result.stdout


def test_rendered_controller_validates_the_region_it_is_given(tmp_path):
    stub = _render_controller(tmp_path)
    parent_env = _parent_env(
        LAMBDA_API_KEY="k",
        LAMBDA_SSH_KEY_NAME="n",
        LAMBDA_SSH_PRIVATE_KEY=str(tmp_path / "missing-key"),
        LAMBDA_REGION="Bad Region",
    )

    result = _run_controller(stub, parent_env)

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert (
        "LAMBDA_REGION must contain only lowercase letters, digits, and hyphens."
        in result.stdout
    )


def test_rendered_controller_does_not_import_modules_from_its_own_directory(tmp_path):
    stub = _render_controller(tmp_path)
    # A module that shares a standard-library name and sits next to the stub.
    # Under -I the directory of the script is not on sys.path, so it is never
    # imported; without -I it would be the ``json`` the controller picks up.
    (tmp_path / "json.py").write_text(
        "raise RuntimeError('imported from the controller directory')\n",
        encoding="utf-8",
    )

    isolated = _run_controller(stub, _parent_env())
    plain = subprocess.run(
        [sys.executable, str(stub)],
        env=_parent_env(),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert isolated.returncode == 2, (isolated.stdout, isolated.stderr)
    assert "imported from the controller directory" not in isolated.stderr
    # The control run proves the sibling module is really on the path without -I.
    assert "imported from the controller directory" in plain.stderr


# ``-I`` also ignores PYTHONUTF8 and PYTHONIOENCODING, so the controller's output goes
# through the platform default encoding (the ANSI code page on a Windows pipe). The
# rendered file reconfigures stdout and stderr to replace what that encoding cannot
# show, so a log line with such a character is printed instead of raising.
#
# The harness runs the real controller code under ``-I``: ``narrow`` first puts both
# streams on an ASCII pipe encoding that raises on anything else (the same failure a
# Windows code page gives, on every platform); ``native`` keeps whatever the platform
# default is. ``stdout`` then drives ``main()``'s own print path with a failing Lambda
# call; ``stderr`` writes one line after the controller has been loaded.
_ENCODING_HARNESS = r'''
import sys

stream, mode, stub = sys.argv[1:4]
if mode == "narrow":
    for handle in (sys.stdout, sys.stderr):
        handle.reconfigure(encoding="ascii", errors="strict")
namespace = {"__name__": "controller_under_test"}
with open(stub, encoding="utf-8") as fh:
    source = fh.read()
exec(compile(source, stub, "exec"), namespace)
# ssh and scp count as installed wherever this runs.
namespace["_find_tool"] = lambda name: "/usr/bin/" + name
text = "remote said: \u2588 \u00e9 \u65e5"
if stream == "stdout":
    def _fail(*args, **kwargs):
        raise RuntimeError(text)
    namespace["_request_json"] = _fail
    sys.exit(namespace["main"]())
sys.stderr.write("stderr line: " + text + "\n")
sys.stderr.flush()
'''


def _run_encoding_harness(tmp_path: Path, stream: str, mode: str):
    from soup_cli.cloud.lambda_labs import _controller_environment

    stub = _render_controller(tmp_path)
    harness = tmp_path / "encoding_harness.py"
    harness.write_text(_ENCODING_HARNESS, encoding="utf-8")
    key = tmp_path / "key"
    key.write_text("not a real key\n", encoding="utf-8")
    parent_env = _parent_env(
        LAMBDA_API_KEY="k",
        LAMBDA_SSH_KEY_NAME="n",
        LAMBDA_SSH_PRIVATE_KEY=str(key),
    )
    # No PYTHONIOENCODING / PYTHONUTF8: the controller environment never carries them.
    return subprocess.run(
        [sys.executable, "-I", str(harness), stream, mode, str(stub)],
        env=_controller_environment(parent_env),
        cwd=tmp_path,
        capture_output=True,
        timeout=120,
    )


def test_stdout_prints_a_line_its_encoding_cannot_show(tmp_path):
    result = _run_encoding_harness(tmp_path, "stdout", "narrow")

    stdout = result.stdout.decode("ascii")
    stderr = result.stderr.decode("utf-8", errors="replace")
    assert "UnicodeEncodeError" not in stderr, stderr
    assert result.returncode == 1, (stdout, stderr)
    assert "Lambda training failed: remote said: ? ? ?" in stdout


def test_stderr_writes_a_line_its_encoding_cannot_show(tmp_path):
    result = _run_encoding_harness(tmp_path, "stderr", "narrow")

    stderr = result.stderr.decode("ascii")
    assert "UnicodeEncodeError" not in stderr, stderr
    assert result.returncode == 0, (result.stdout, stderr)
    assert "stderr line: remote said: ? ? ?" in stderr


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_output_with_the_platform_default_pipe_encoding_does_not_raise(tmp_path, stream):
    # Windows pipes use the ANSI code page, which cannot show U+2588 or U+65E5; on
    # other platforms the default is usually UTF-8 and this is a smoke run.
    result = _run_encoding_harness(tmp_path, stream, "native")

    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    assert "UnicodeEncodeError" not in stderr, stderr
    if stream == "stdout":
        assert result.returncode == 1, (stdout, stderr)
        assert "Lambda training failed: remote said:" in stdout
    else:
        assert result.returncode == 0, (stdout, stderr)
        assert "stderr line: remote said:" in stderr
