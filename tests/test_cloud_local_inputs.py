import subprocess
from dataclasses import replace

import pytest

from soup_cli.cloud._local_inputs import LocalTrainInput
from soup_cli.cloud.lambda_labs import render_lambda_stub
from soup_cli.cloud.modal import render_modal_stub
from soup_cli.config.loader import load_config
from soup_cli.data.loader import load_dataset


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


def test_lambda_user_data_waits_for_inputs_before_training(tmp_path):
    manifest = _manifest(tmp_path)
    source = render_lambda_stub(
        "data:\n  train: inputs/train-0.jsonl\n",
        gpu="a100",
        output_dir="outputs",
        soup_version="0.1.0",
        local_train_inputs=manifest,
    )
    namespace = _exec_stub(source)
    user_data = namespace["_USER_DATA"]

    train_command = "soup train --config soup.yaml --yes"

    assert user_data.count(train_command) == 1
    assert "chown ubuntu:ubuntu /home/ubuntu/soup\n" in user_data
    assert user_data.index("inputs-ready") < user_data.index(train_command)
    assert "input_deadline=$((SECONDS + 1200))" in user_data
    assert 'if [ "$SECONDS" -ge "$input_deadline" ]; then' in user_data
    assert "Timed out waiting for local training inputs" in user_data
    assert "    exit 1\n" in user_data
    assert "  sleep 2\n" in user_data


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

    assert calls == [
        [
            "modal",
            "volume",
            "put",
            "--force",
            "soup-outputs",
            manifest[0].local_path,
            "/test-run/inputs/train-0.jsonl",
        ]
    ]


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

    clock = iter([10_000, 10_000, 10_000_000])
    monkeypatch.setattr(namespace["time"], "monotonic", lambda: next(clock, 10_000_000))
    monkeypatch.setattr(namespace["time"], "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="uploading local training input"):
        namespace["_upload_local_train_inputs"](
            "127.0.0.1",
            [],
            "ssh",
            "scp",
            {},
        )

    assert any(command[0] == "scp" for command in commands)
    assert not any(
        command[0] == "ssh" and "touch /home/ubuntu/soup/inputs-ready" in command
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
        if command[0] == "ssh" and len([item for item in calls if item[0] == "ssh"]) < 3:
            return subprocess.CompletedProcess(command, 255, "", "SSH unavailable")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
    monkeypatch.setattr(namespace["time"], "monotonic", lambda: 0)
    monkeypatch.setattr(namespace["time"], "sleep", lambda _: None)

    namespace["_upload_local_train_inputs"](
        "127.0.0.1",
        [],
        "ssh",
        "scp",
        {},
    )

    assert len([command for command in calls if command[0] == "ssh"]) >= 3
    assert any(command[0] == "scp" for command in calls)
    assert any(
        command[0] == "ssh" and "touch /home/ubuntu/soup/inputs-ready" in command
        for command in calls
    )


def test_lambda_upload_retry_exhaustion_fails_without_ready_marker(tmp_path, monkeypatch):
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
    monkeypatch.setattr(namespace["time"], "monotonic", lambda: next(clock, 901))
    monkeypatch.setattr(namespace["time"], "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="preparing remote input directory"):
        namespace["_upload_local_train_inputs"](
            "127.0.0.1",
            [],
            "ssh",
            "scp",
            {},
        )

    assert calls
    assert not any(
        command[0] == "ssh" and "touch /home/ubuntu/soup/inputs-ready" in command
        for command in calls
    )


def test_prepare_local_train_inputs_preserves_none():
    from types import SimpleNamespace

    from soup_cli.cloud._local_inputs import prepare_local_train_inputs

    cfg = SimpleNamespace(data=SimpleNamespace(train=None))

    manifest = prepare_local_train_inputs(cfg)

    assert manifest == ()
    assert cfg.data.train is None


@pytest.mark.parametrize("backend", ["lambda", "modal"])
def test_local_train_input_changed_after_planning_is_rejected(tmp_path, backend):
    manifest = _manifest(tmp_path)

    if backend == "lambda":
        source = render_lambda_stub(
            "data:\n  train: inputs/train-0.jsonl\n",
            gpu="a100",
            output_dir="outputs",
            soup_version="0.1.0",
            local_train_inputs=manifest,
        )
    else:
        source = render_modal_stub(
            "data:\n  train: inputs/train-0.jsonl\n",
            gpu="a100",
            output_dir="outputs",
            soup_version="0.1.0",
            run_name="test-run",
            local_train_inputs=manifest,
        )

    namespace = _exec_stub(source)
    with open(manifest[0].local_path, "a", encoding="utf-8") as file:
        file.write('{"text": "changed"}\n')

    with pytest.raises(RuntimeError, match="changed after planning"):
        namespace["_validate_local_train_inputs"]()


@pytest.mark.parametrize("backend", ["lambda", "modal"])
def test_invalid_generated_remote_input_path_is_rejected(tmp_path, backend, monkeypatch):
    manifest = (replace(_manifest(tmp_path)[0], remote_path="inputs/../x"),)

    if backend == "lambda":
        source = render_lambda_stub(
            "data:\n  train: inputs/../x\n",
            gpu="a100",
            output_dir="outputs",
            soup_version="0.1.0",
            local_train_inputs=manifest,
        )
    else:
        source = render_modal_stub(
            "data:\n  train: inputs/../x\n",
            gpu="a100",
            output_dir="outputs",
            soup_version="0.1.0",
            run_name="test-run",
            local_train_inputs=manifest,
        )

    namespace = _exec_stub(source)

    monkeypatch.setattr(
        namespace["subprocess"],
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError, match="invalid generated remote input path"):
        if backend == "lambda":
            namespace["_upload_local_train_inputs"](
                "127.0.0.1",
                [],
                "ssh",
                "scp",
                {},
            )
        else:
            namespace["_upload_local_train_inputs"]()


def test_scratch_directory_loads_config_and_uploaded_train_input(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True)

    (run_dir / "soup.yaml").write_text(
        "base: x\ndata:\n  train: inputs/train-0.jsonl\n  format: auto\n  val_split: 0.0\n",
        encoding="utf-8",
    )
    (inputs_dir / "train-0.jsonl").write_text(
        '{"text": "hello"}\n',
        encoding="utf-8",
    )

    monkeypatch.chdir(run_dir)

    cfg = load_config("soup.yaml")
    dataset = load_dataset(cfg.data)

    assert dataset["train"] == [{"text": "hello"}]
