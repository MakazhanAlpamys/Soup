"""Contract tests for the opt-in one-step CPU Hub QA fixture."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "hf_hub_cpu_fixture.py"


def _load_harness():
    name = "_hf_hub_cpu_fixture_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class _FakeHubApi:
    def __init__(self, *, private=True, branches=("checkpoint-1",), files=(), model_error=None):
        self.private = private
        self.branches = branches
        self.files = list(files)
        self.model_error = model_error

    def model_info(self, repo_id):
        del repo_id
        if self.model_error is not None:
            raise self.model_error
        return SimpleNamespace(private=self.private)

    def list_repo_refs(self, repo_id):
        del repo_id
        return SimpleNamespace(branches=[SimpleNamespace(name=name) for name in self.branches])

    def list_repo_files(self, repo_id, revision=None):
        del repo_id, revision
        return self.files


def _use_fake_hub(monkeypatch, harness, api, token="hf_test_token"):
    monkeypatch.syspath_prepend(str(harness._SOURCE_ROOT))
    import soup_cli.utils.hf as hf

    monkeypatch.setattr(hf, "resolve_token", lambda: token)
    monkeypatch.setattr(hf, "resolve_endpoint", lambda: "https://example.invalid")
    monkeypatch.setattr(hf, "get_hf_api", lambda **kwargs: api)


def test_preparation_writes_one_row_one_step_fixture_under_named_directory(tmp_path):
    harness = _load_harness()
    output_dir = tmp_path / "fresh-fixture"

    paths = harness.prepare_fixture(output_dir)

    assert paths.root == output_dir.absolute()
    assert json.loads(paths.data.read_text(encoding="utf-8"))["messages"][1]["content"] == "soup"
    config = paths.config.read_text(encoding="utf-8")
    assert "base: hf-internal-testing/tiny-random-gpt2" in config
    assert "epochs: 1" in config
    assert "batch_size: 1" in config
    assert "save_steps: 1" in config
    assert "--push-as" not in paths.command.read_text(encoding="utf-8")
    assert not paths.output.exists()


def test_preparation_refuses_to_overwrite_an_existing_directory(tmp_path):
    harness = _load_harness()
    output_dir = tmp_path / "existing-fixture"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        harness.prepare_fixture(output_dir)

    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "do not replace"


def test_train_command_is_shell_free_and_push_is_explicit(tmp_path):
    harness = _load_harness()
    paths = harness.fixture_paths(tmp_path / "fixture")

    local = harness.build_train_command(paths)
    live = harness.build_train_command(paths, "tester/private-fixture")

    assert local[:4] == [sys.executable, "-m", "soup_cli.cli", "train"]
    assert "--push-as" not in local
    assert live[-2:] == ["--push-as", "tester/private-fixture"]


def test_local_cpu_child_environment_hides_accelerators_and_strips_hf_tokens(monkeypatch):
    harness = _load_harness()
    monkeypatch.setenv("HF_TOKEN", "not-for-a-report")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "also-not-for-a-report")

    environment = harness.cpu_environment()

    assert environment["CUDA_VISIBLE_DEVICES"] == "-1"
    assert environment["HIP_VISIBLE_DEVICES"] == ""
    assert environment["ROCR_VISIBLE_DEVICES"] == ""
    assert environment["ACCELERATE_USE_CPU"] == "true"
    assert environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert environment["PYTHONPATH"].split(__import__("os").pathsep)[0] == str(harness._SOURCE_ROOT)
    assert "HF_TOKEN" not in environment
    assert "HUGGINGFACE_HUB_TOKEN" not in environment


def test_live_cpu_child_environment_keeps_hf_token_for_explicit_push(monkeypatch):
    harness = _load_harness()
    monkeypatch.setenv("HF_TOKEN", "only-for-explicit-push")

    environment = harness.cpu_environment("tester/private-fixture")

    assert environment["HF_TOKEN"] == "only-for-explicit-push"


def test_parent_side_hub_checks_prefer_this_checkout_source(monkeypatch):
    harness = _load_harness()
    monkeypatch.setattr(harness.sys, "path", ["unrelated-site-packages"])

    harness._prefer_local_source()

    assert harness.sys.path[0] == str(harness._SOURCE_ROOT)


def test_fake_hub_helper_restores_its_temporary_source_path(monkeypatch):
    harness = _load_harness()
    original_path = list(sys.path)

    with monkeypatch.context() as scoped_monkeypatch:
        _use_fake_hub(scoped_monkeypatch, harness, _FakeHubApi())
        assert sys.path[0] == str(harness._SOURCE_ROOT)

    assert sys.path == original_path


def test_cpu_training_preflight_requires_torch_to_report_cuda_hidden(monkeypatch, tmp_path):
    harness = _load_harness()
    paths = harness.fixture_paths(tmp_path / "fixture")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["-c", "import torch; print(torch.cuda.is_available())"]:
            return SimpleNamespace(returncode=0, stdout="False\n", stderr="")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    monkeypatch.setattr(harness, "verify_local_artifacts", lambda _: {"verified": True})

    assert harness.run_training(paths, None) == {"verified": True}
    assert calls[0][0] == [
        sys.executable,
        "-c",
        "import torch; print(torch.cuda.is_available())",
    ]
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert calls[1][0] == harness.build_train_command(paths)


def test_cpu_training_refuses_to_continue_when_torch_still_sees_cuda(monkeypatch, tmp_path):
    harness = _load_harness()
    paths = harness.fixture_paths(tmp_path / "fixture")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="True\n", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="CUDA remained visible"):
        harness.run_training(paths, None)

    assert len(calls) == 1


def test_cpu_training_reports_the_last_failed_probe_stderr_line(monkeypatch, tmp_path):
    harness = _load_harness()
    paths = harness.fixture_paths(tmp_path / "fixture")

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="first detail\nModuleNotFoundError: No module named 'torch'\n",
        ),
    )

    with pytest.raises(RuntimeError, match="Last stderr line: ModuleNotFoundError"):
        harness.run_training(paths, None)


def test_cpu_training_refuses_an_empty_cuda_probe_result(monkeypatch, tmp_path):
    harness = _load_harness()
    paths = harness.fixture_paths(tmp_path / "fixture")

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError, match="no CUDA availability result"):
        harness.run_training(paths, None)


def test_local_artifact_verification_requires_adapter_and_checkpoint_state(tmp_path):
    harness = _load_harness()
    paths = harness.fixture_paths(tmp_path / "fixture")
    paths.output.mkdir(parents=True)
    (paths.output / "adapter_config.json").write_text("{}", encoding="utf-8")
    (paths.output / "adapter_model.safetensors").write_bytes(b"fixture")
    checkpoint = paths.output / harness.CHECKPOINT_NAME
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    result = harness.verify_local_artifacts(paths)

    assert result["checkpoint"] == "checkpoint-1"
    assert result["checkpoint_has_trainer_state"] is True


def test_hub_push_is_rejected_without_explicit_remote_verification(tmp_path, capsys):
    harness = _load_harness()

    with pytest.raises(SystemExit) as exc_info:
        harness.main(
            [
                "--output-dir",
                str(tmp_path / "fixture"),
                "--train",
                "--push-as",
                "tester/private",
            ]
        )

    assert exc_info.value.code == 2
    assert "--push-as requires --verify-hub" in capsys.readouterr().err


def test_hub_verification_is_rejected_without_an_explicit_push_target(tmp_path, capsys):
    harness = _load_harness()

    with pytest.raises(SystemExit) as exc_info:
        harness.main(["--output-dir", str(tmp_path / "fixture"), "--verify-hub"])

    assert exc_info.value.code == 2
    assert "--verify-hub requires --push-as" in capsys.readouterr().err


def test_private_destination_requires_a_token(monkeypatch):
    harness = _load_harness()
    _use_fake_hub(monkeypatch, harness, _FakeHubApi(), token=None)

    with pytest.raises(RuntimeError, match="requires an HF write token"):
        harness.verify_private_destination("tester/private")


def test_private_destination_requires_an_existing_private_repository(monkeypatch):
    harness = _load_harness()

    _use_fake_hub(monkeypatch, harness, _FakeHubApi(model_error=RuntimeError("not found")))
    with pytest.raises(RuntimeError, match="pre-created as a private"):
        harness.verify_private_destination("tester/private")

    _use_fake_hub(monkeypatch, harness, _FakeHubApi(private=False))
    with pytest.raises(RuntimeError, match="non-private"):
        harness.verify_private_destination("tester/private")


def test_private_destination_accepts_an_existing_private_repository(monkeypatch):
    harness = _load_harness()
    _use_fake_hub(monkeypatch, harness, _FakeHubApi(private=True))

    assert harness.verify_private_destination("tester/private") is None


def test_remote_checkpoint_requires_a_token_and_checkpoint_branch(monkeypatch):
    harness = _load_harness()
    _use_fake_hub(monkeypatch, harness, _FakeHubApi(), token=None)
    with pytest.raises(RuntimeError, match="token vanished"):
        harness.verify_remote_checkpoint("tester/private")

    _use_fake_hub(monkeypatch, harness, _FakeHubApi(branches=("main",)))
    with pytest.raises(RuntimeError, match="no 'checkpoint-1' branch"):
        harness.verify_remote_checkpoint("tester/private")


def test_remote_checkpoint_requires_adapter_config_state_and_adapter_weights(monkeypatch):
    harness = _load_harness()
    only_non_adapter_checkpoint_files = [
        "adapter_config.json",
        "trainer_state.json",
        "training_args.bin",
        "optimizer.pt",
    ]
    _use_fake_hub(monkeypatch, harness, _FakeHubApi(files=only_non_adapter_checkpoint_files))

    with pytest.raises(RuntimeError, match="adapter weights"):
        harness.verify_remote_checkpoint("tester/private")

    _use_fake_hub(monkeypatch, harness, _FakeHubApi(files=["adapter_model.safetensors"]))
    with pytest.raises(RuntimeError, match="adapter_config.json, trainer_state.json"):
        harness.verify_remote_checkpoint("tester/private")


def test_remote_checkpoint_returns_branch_and_file_count_for_a_complete_adapter(monkeypatch):
    harness = _load_harness()
    files = ["adapter_config.json", "adapter_model.bin", "trainer_state.json"]
    _use_fake_hub(monkeypatch, harness, _FakeHubApi(files=files))

    assert harness.verify_remote_checkpoint("tester/private") == {
        "branch": "checkpoint-1",
        "file_count": 3,
    }


def test_hub_failure_writes_a_negative_report_and_disables_rich_markup(monkeypatch, tmp_path):
    harness = _load_harness()
    output_dir = tmp_path / "fixture"
    printed = []
    monkeypatch.setattr(harness, "verify_private_destination", lambda _: None)
    monkeypatch.setattr(harness, "run_training", lambda paths, push_as: {"local": True})
    monkeypatch.setattr(
        harness,
        "verify_remote_checkpoint",
        lambda _: (_ for _ in ()).throw(RuntimeError("remote [checkpoint] missing")),
    )
    monkeypatch.setattr(
        harness.console,
        "print",
        lambda *args, **kwargs: printed.append((args, kwargs)),
    )

    status = harness.main(
        [
            "--output-dir",
            str(output_dir),
            "--train",
            "--push-as",
            "tester/private",
            "--verify-hub",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "cpu-fixture-report.json").read_text(encoding="utf-8"))
    assert report["hub_verified"] is False
    assert report["error"] == "remote [checkpoint] missing"
    assert printed[-1][1]["markup"] is False
