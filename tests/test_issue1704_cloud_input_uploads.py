import subprocess

import pytest

from soup_cli.cloud._local_inputs import LocalTrainInput
from soup_cli.cloud.lambda_labs import render_lambda_stub
from soup_cli.cloud.modal import render_modal_stub


def _manifest(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text('{"text": "hello"}\n', encoding="utf-8")
    info = source.stat()

    return (
        LocalTrainInput(
            local_path=str(source.resolve()),
            remote_path="inputs/train-0.jsonl",
            size_bytes=info.st_size,
            mtime_ns=info.st_mtime_ns,
            device=info.st_dev,
            inode=info.st_ino,
        ),
    )


def _exec_stub(source):
    import sys
    import types
    from unittest.mock import MagicMock, patch

    class ModalApp:
        def function(self, *args, **kwargs):
            return lambda fn: fn

        def local_entrypoint(self, *args, **kwargs):
            return lambda fn: fn

    class ModalStub(types.ModuleType):
        def __getattr__(self, name):
            if name == "App":
                return lambda *args, **kwargs: ModalApp()
            return MagicMock(name=f"modal.{name}")

    namespace = {}
    with patch.dict(sys.modules, {"modal": ModalStub("modal")}):
        exec(compile(source, "<generated-controller>", "exec"), namespace)

    return namespace


def test_modal_upload_failure_prevents_training(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    manifest = _manifest(tmp_path)
    source = render_modal_stub(
        "data:\n  train: inputs/train-0.jsonl\n",
        gpu="a100",
        output_dir="outputs",
        soup_version="0.1.0",
        run_name="test-run",
        local_train_inputs=manifest,
    )
    namespace = _exec_stub(source)

    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
    namespace["_fetch_outputs"] = lambda: None

    train_remote = MagicMock()
    namespace["train"].remote = train_remote

    with pytest.raises(subprocess.CalledProcessError):
        namespace["main"]()

    train_remote.assert_not_called()

def test_modal_upload_uses_expected_remote_path(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    source = render_modal_stub(
        "data:\n  train: inputs/train-0.jsonl\n",
        gpu="a100",
        output_dir="outputs",
        soup_version="0.1.0",
        run_name="test-run",
        local_train_inputs=manifest,
    )
    namespace = _exec_stub(source)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
    namespace["_upload_local_train_inputs"]()

    assert len(calls) == 1
    assert calls[0][-1] == "/test-run/inputs/train-0.jsonl"



def test_lambda_upload_failure_does_not_mark_inputs_ready(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    source = render_lambda_stub(
        "data:\n  train: inputs/train-0.jsonl\n",
        gpu="a100",
        output_dir="outputs",
        soup_version="0.1.0",
        local_train_inputs=manifest,
    )
    namespace = _exec_stub(source)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "scp":
            return subprocess.CompletedProcess(command, 1, "", "upload failed")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)

    clock = iter([10_000, 10_000_000])
    monkeypatch.setattr(
        namespace["time"], "monotonic", lambda: next(clock, 10_000_000)
    )
    monkeypatch.setattr(namespace["time"], "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="uploading local training input"):
        namespace["_upload_local_train_inputs"]("127.0.0.1", [])

    assert any(command[0] == "scp" for command in commands)
    assert not any(
        command[0] == "ssh"
        and "touch /home/ubuntu/soup/inputs-ready" in command
        for command in commands
    )


def test_lambda_upload_retries_until_ssh_is_available(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    source = render_lambda_stub(
        "data:\n  train: inputs/train-0.jsonl\n",
        gpu="a100",
        output_dir="outputs",
        soup_version="0.1.0",
        local_train_inputs=manifest,
    )
    namespace = _exec_stub(source)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ssh" and len(
            [item for item in calls if item[0] == "ssh"]
        ) < 3:
            return subprocess.CompletedProcess(command, 255, "", "SSH unavailable")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
    monkeypatch.setattr(namespace["time"], "monotonic", lambda: 0)
    monkeypatch.setattr(namespace["time"], "sleep", lambda _: None)

    namespace["_upload_local_train_inputs"]("127.0.0.1", [])

    assert len([command for command in calls if command[0] == "ssh"]) >= 3
    assert any(command[0] == "scp" for command in calls)
    assert any(
        command[0] == "ssh"
        and "touch /home/ubuntu/soup/inputs-ready" in command
        for command in calls
    )


def test_lambda_upload_retry_exhaustion_fails_without_ready_marker(
    tmp_path, monkeypatch
):
    manifest = _manifest(tmp_path)
    source = render_lambda_stub(
        "data:\n  train: inputs/train-0.jsonl\n",
        gpu="a100",
        output_dir="outputs",
        soup_version="0.1.0",
        local_train_inputs=manifest,
    )
    namespace = _exec_stub(source)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 255, "", "SSH unavailable")

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)

    clock = iter([0, 901])
    monkeypatch.setattr(
        namespace["time"], "monotonic", lambda: next(clock, 901)
    )
    monkeypatch.setattr(namespace["time"], "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="preparing remote input directory"):
        namespace["_upload_local_train_inputs"]("127.0.0.1", [])

    assert calls
    assert not any(
        command[0] == "ssh"
        and "touch /home/ubuntu/soup/inputs-ready" in command
        for command in calls
    )

def test_prepare_local_train_inputs_preserves_none():
    from types import SimpleNamespace

    from soup_cli.cloud._local_inputs import prepare_local_train_inputs

    cfg = SimpleNamespace(data=SimpleNamespace(train=None))

    manifest = prepare_local_train_inputs(cfg)

    assert manifest == ()
    assert cfg.data.train is None
