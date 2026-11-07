"""Contract tests for the opt-in one-step CPU Hub QA fixture."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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

    environment = harness.cpu_environment()

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HIP_VISIBLE_DEVICES"] == ""
    assert environment["ROCR_VISIBLE_DEVICES"] == ""
    assert environment["ACCELERATE_USE_CPU"] == "true"
    assert environment["PYTHONPATH"].split(__import__("os").pathsep)[0] == str(harness._SOURCE_ROOT)
    assert "HF_TOKEN" not in environment


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
        harness.main(["--output-dir", str(tmp_path / "fixture"), "--push-as", "tester/private"])

    assert exc_info.value.code == 2
    assert "--verify-hub" in capsys.readouterr().err
