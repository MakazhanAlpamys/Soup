"""The generated remote job owns its Python environment and executable paths."""

import subprocess
import sys
import types
from pathlib import Path

import pytest

from tests.test_runpod_cloud import CONFIG
from tests.test_runpod_remote_diagnostics import _remote_script


def _execute(tmp_path, monkeypatch, *, fail_stage=None, training_code=0):
    script = _remote_script()
    root = tmp_path / "remote"
    calls = []

    def run(argv, **kwargs):
        stage = (
            "venv" if argv[1:3] == ["-m", "venv"]
            else "install" if argv[1:3] == ["-m", "pip"] else "train"
        )
        calls.append((stage, argv, kwargs))
        if stage == fail_stage:
            assert kwargs["check"] is True
            raise subprocess.CalledProcessError(13, argv)
        if stage == "train":
            kwargs["stdout"].write("training result\n")
        else:
            assert kwargs["stdout"] is sys.stderr
        return types.SimpleNamespace(returncode=training_code if stage == "train" else 0)

    with monkeypatch.context() as patch:
        patch.chdir(tmp_path)
        patch.setenv("PATH", "")
        patch.setitem(sys.modules, "pathlib", types.SimpleNamespace(
            Path=lambda path: root if str(path) == "/workspace/soup" else Path(path)
        ))
        patch.setitem(sys.modules, "subprocess", types.SimpleNamespace(
            run=run, STDOUT=subprocess.STDOUT
        ))
        try:
            exec(compile(script, "remote.py", "exec"), {"__name__": "__main__"})
        except BaseException as exc:
            outcome = exc
        else:
            pytest.fail("remote script did not return an exit status")
    return outcome, calls, root


def test_remote_job_uses_system_site_packages_and_absolute_venv_commands(tmp_path, monkeypatch):
    outcome, calls, root = _execute(tmp_path, monkeypatch)
    assert isinstance(outcome, SystemExit) and outcome.code == 0
    assert [stage for stage, *_ in calls] == ["venv", "install", "train"]
    assert calls[0][1] == [
        sys.executable, "-m", "venv", "--system-site-packages", "/workspace/venv"
    ]
    assert calls[1][1] == [
        "/workspace/venv/bin/python", "-m", "pip", "install", "soup-cli[train]==0.76.0"
    ]
    assert calls[2][1] == [
        "/workspace/venv/bin/soup", "train", "--config", "soup.yaml", "--yes"
    ]
    assert all(options["check"] is True for _, _, options in calls[:2])
    assert calls[2][2]["stderr"] == subprocess.STDOUT
    assert (root / "soup.yaml").read_bytes() == CONFIG.encode()
    assert (root / "train.log").read_text() == "training result\n"


@pytest.mark.parametrize("stage", ["venv", "install"])
def test_failed_setup_stops_before_training(tmp_path, monkeypatch, stage):
    outcome, calls, root = _execute(tmp_path, monkeypatch, fail_stage=stage)
    assert isinstance(outcome, subprocess.CalledProcessError)
    assert outcome.returncode == 13
    assert [name for name, *_ in calls] == (
        ["venv"] if stage == "venv" else ["venv", "install"]
    )
    assert not (root / "train.log").exists()


def test_training_failure_still_returns_its_status_and_log_tail(tmp_path, monkeypatch, capsys):
    outcome, calls, _ = _execute(tmp_path, monkeypatch, training_code=7)
    assert isinstance(outcome, SystemExit) and outcome.code == 7
    assert [stage for stage, *_ in calls] == ["venv", "install", "train"]
    assert capsys.readouterr().err == "training result"
