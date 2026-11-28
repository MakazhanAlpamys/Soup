"""Contract tests for the local train preflight receipt."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer

from soup_cli.cli import app
from soup_cli.commands import train as train_command
from tests.conftest import make_cli_runner, strip_ansi

_REPO = Path(__file__).resolve().parents[1]
_PRIVATE = "private-preflight-sentinel"
_DISTRIBUTED_ENV = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "ACCELERATE_MIXED_PRECISION",
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_FSDP",
)
_OFFLINE_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "SOUP_NO_AUDIT_LOG": "1",
    "SOUP_TELEMETRY": "0",
}


@pytest.fixture(autouse=True)
def local_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in _DISTRIBUTED_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in _OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def config(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"caf\u00e9 {_PRIVATE} prompt {index}"},
                        {"role": "assistant", "content": f"Synthetic answer {index}."},
                    ]
                },
                ensure_ascii=False,
            )
            + "\n"
            for index in range(5)
        )
        + f"broken {_PRIVATE}\n",
        encoding="utf-8",
    )
    path = tmp_path / "soup.yaml"
    path.write_text(
        "base: example/tiny\n"
        "task: sft\n"
        "backend: transformers\n"
        "data:\n"
        "  train: rows.jsonl\n"
        "  format: chatml\n"
        "  val_split: 0.2\n"
        "  max_length: 512\n"
        "training:\n"
        "  quantization: none\n"
        "  batch_size: 2\n"
        "  gradient_accumulation_steps: 4\n"
        "  epochs: 2\n"
        "  lora:\n"
        "    r: 8\n"
        "output: out\n",
        encoding="utf-8",
    )
    return path


def _invoke(*args, input=None):
    return make_cli_runner().invoke(app, ["train", *args], input=input)


def _plain(text):
    return " ".join(strip_ansi(text).split())


def _receipt(stdout, exit_code, *, status, reason=None):
    assert stdout.endswith(b"\n") and stdout.count(b"\n") == 1, stdout
    assert len(stdout) <= 262_144
    result = json.loads(stdout.decode("utf-8"))
    assert result["schema"] == "soup.command_result"
    assert result["schema_version"] == 1
    assert result["soup_version"]
    assert result["command"] == "train"
    assert result["phase"] == "preflight"
    assert result["status"] == status
    assert result["exit_code"] == exit_code
    assert result["started_at"].endswith(("Z", "+00:00"))
    assert math.isfinite(result["duration_s"]) and result["duration_s"] >= 0
    assert result["run_id"] is None
    assert result["child"] is None
    assert result["artifacts"] == []
    assert _PRIVATE.encode() not in stdout
    if reason is None:
        assert result["reasons"] == []
    else:
        assert result["reasons"][0]["code"] == reason, result
    return result


def _check_result(result, code, *, status, reason=None):
    assert result.exit_code == code, (result.output, result.exception)
    return _receipt(result.stdout_bytes, code, status=status, reason=reason)


def test_result_json_requires_explicit_dry_run(config, monkeypatch):
    load_config = Mock(side_effect=AssertionError("preflight must refuse before config load"))
    monkeypatch.setattr(train_command, "load_config", load_config)

    result = _invoke("--config", str(config), "--result-json")

    _check_result(result, 2, status="refused", reason="soup.train.preflight_unsupported")
    assert "--dry-run" in _plain(result.stderr)
    load_config.assert_not_called()


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(train_command, "detect_device", lambda **_: ("cpu", "Test CPU"))
    monkeypatch.setattr(
        train_command,
        "get_gpu_info",
        lambda **_: {"memory_total": "N/A", "memory_total_bytes": 0, "gpu_count": 0},
    )


@pytest.fixture
def launch_tripwires(monkeypatch):
    targets = (
        "soup_cli.commands.train._run_live_lr_sweep",
        "soup_cli.utils.launcher.run_launcher",
        "soup_cli.experiment.tracker.ExperimentTracker",
        "soup_cli.trainer.dispatch.build_trainer",
    )
    spies = []
    for target in targets:
        spy = Mock(side_effect=AssertionError(f"unexpected preflight side effect: {target}"))
        monkeypatch.setattr(target, spy)
        spies.append(spy)
    yield
    for spy in spies:
        spy.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(("--cloud", "modal"), id="cloud-plan"),
        pytest.param(("--cloud-submit",), id="cloud-submit"),
        pytest.param(("--gpu", "a100"), id="cloud-gpu-default-explicit"),
        pytest.param(("--gpus", "2"), id="gpus"),
        pytest.param(("--nodes", "2"), id="nodes"),
        pytest.param(("--node-rank", "0"), id="rank-zero-explicit"),
        pytest.param(("--master-addr", "127.0.0.1"), id="master-addr"),
        pytest.param(("--master-port", "29500"), id="master-port"),
        pytest.param(("--no-reexec",), id="no-reexec"),
        pytest.param(("--deepspeed", "zero2"), id="deepspeed"),
        pytest.param(("--fsdp", "full_shard"), id="fsdp"),
        pytest.param(("--find-lr",), id="lr"),
        pytest.param(("--find-lr-start", "1e-7"), id="lr-start-default-explicit"),
        pytest.param(("--find-lr-end", "0.1"), id="lr-end-default-explicit"),
        pytest.param(("--find-lr-steps", "100"), id="lr-steps-default-explicit"),
        pytest.param(("--find-lr-output", "lr_finder.json"), id="lr-output-default-explicit"),
        pytest.param(("--hf-resume",), id="hf-resume"),
        pytest.param(("--resume", "latest"), id="local-resume"),
        pytest.param(("--push-as", "example/tiny"), id="push-as"),
        pytest.param(("--push-public",), id="push-public"),
        pytest.param(("--wandb",), id="wandb"),
        pytest.param(("--tensorboard",), id="tensorboard"),
        pytest.param(("--tracker", "mlflow"), id="tracker"),
        pytest.param(("--profile",), id="profile"),
        pytest.param(("--capture-activations", "model.layers.0"), id="activation-capture"),
        pytest.param(("--capture-prompts", "prompts.jsonl"), id="capture-prompts"),
        pytest.param(("--annex-xi", "annex.pdf"), id="annex-xi"),
        pytest.param(("--repro-receipt", "receipt.json"), id="repro-receipt"),
        pytest.param(("--track-energy",), id="track-energy"),
        pytest.param(("--energy-country", "USA"), id="energy-country-default-explicit"),
        pytest.param(("--energy-out", "energy.json"), id="energy-out"),
        pytest.param(("--diagnose-gate", "evidence.json"), id="post-training-diagnose-gate"),
        pytest.param(("--diagnose-gate-require-all-modes",), id="diagnose-gate-all-modes"),
        pytest.param(("--trust-remote-code",), id="trust-remote-code"),
    ],
)
def test_execution_options_refuse_before_config_load(
    config, monkeypatch, launch_tripwires, options,
):
    loader = Mock(side_effect=AssertionError("unsupported options must refuse before config load"))
    monkeypatch.setattr(train_command, "load_config", loader)

    result = _invoke("--config", str(config), "--dry-run", "--result-json", *options)

    receipt = _check_result(result, 2, status="refused", reason="soup.train.preflight_unsupported")
    assert receipt["data"] is None
    assert "Ready to train" not in _plain(result.stderr)
    loader.assert_not_called()


@pytest.mark.parametrize(
    "environment",
    [
        pytest.param({"RANK": "0", "WORLD_SIZE": "1"}, id="torchrun-single-rank"),
        pytest.param({"RANK": "0", "WORLD_SIZE": "2"}, id="torchrun-multi-rank"),
        pytest.param({"WORLD_SIZE": "2"}, id="world-size-only"),
        pytest.param({"LOCAL_RANK": "0"}, id="local-rank-only"),
        pytest.param({"ACCELERATE_MIXED_PRECISION": "no"}, id="accelerate-mixed-precision"),
        pytest.param({"ACCELERATE_USE_DEEPSPEED": "false"}, id="accelerate-deepspeed"),
        pytest.param({"ACCELERATE_USE_FSDP": "false"}, id="accelerate-fsdp"),
    ],
)
def test_launcher_environment_refuses_before_config(config, monkeypatch, environment):
    loader = Mock()
    monkeypatch.setattr(train_command, "load_config", loader)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    result = _invoke("--config", str(config), "--dry-run", "--result-json")

    _check_result(result, 2, status="refused", reason="soup.train.preflight_unsupported")
    loader.assert_not_called()


@pytest.mark.parametrize(
    "training_setting",
    [
        pytest.param("parallelism: pipeline", id="pipeline"),
        pytest.param("use_fsdp2_compile: true", id="fsdp2-compile"),
        pytest.param("hub: modelscope", id="non-hf-hub"),
        pytest.param("ra_dit_stage: generator", id="implicit-ra-dit-retriever"),
    ],
)
def test_incompatible_config_refuses_before_hardware(config, monkeypatch, training_setting):
    source = config.read_text(encoding="utf-8")
    config.write_text(
        source.replace("training:\n", f"training:\n  {training_setting}\n"),
        encoding="utf-8",
    )
    hardware = Mock(side_effect=AssertionError("incompatible config reached hardware detection"))
    autolink = Mock(side_effect=AssertionError("incompatible config tried to contact the registry"))
    monkeypatch.setattr(train_command, "detect_device", hardware)
    monkeypatch.setattr("soup_cli.utils.ra_dit_run.autolink_generator_retriever", autolink)

    result = _invoke("--config", str(config), "--dry-run", "--result-json")

    _check_result(result, 2, status="refused", reason="soup.train.preflight_unsupported")
    hardware.assert_not_called()
    autolink.assert_not_called()


def test_missing_config_has_safe_reason(tmp_path):
    missing = tmp_path / f"{_PRIVATE}.yaml"
    result = _invoke("--config", str(missing), "--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason="soup.config.missing")
    assert receipt["data"] is None
    assert "Config not found" in _plain(result.stderr)
    assert "soup init" in _plain(result.stderr)


@pytest.mark.parametrize(
    ("contents", "category", "diagnostic"),
    [
        pytest.param("base: [\n", "yaml_parse", "is not valid YAML:", id="malformed"),
        pytest.param("- entry\n", "wrong_root", "Config must be a YAML mapping", id="wrong-root"),
        pytest.param("", "wrong_root", "Config file is empty", id="empty"),
    ],
)
def test_non_schema_config_errors_carry_an_empty_bounded_list(
    tmp_path, contents, category, diagnostic,
):
    path = tmp_path / f"{_PRIVATE}.yaml"
    path.write_text(contents, encoding="utf-8")

    result = _invoke("--config", str(path), "--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason="soup.config.invalid")
    assert receipt["data"] is None
    assert receipt["reasons"][0]["details"] == {"category": category, "errors": []}
    assert diagnostic in _plain(result.stderr)


def test_invalid_typed_config_keeps_field_and_type_without_value(config):
    yaml = config.read_text(encoding="utf-8").replace("batch_size: 2", f"batch_size: {_PRIVATE}")
    config.write_text(yaml, encoding="utf-8")

    result = _invoke("--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason="soup.config.invalid")
    errors = receipt["reasons"][0]["details"]["errors"]
    assert any(
        error["loc"][:2] == ["training", "batch_size"] and error["type"] == "int_parsing"
        for error in errors
    ), errors
    assert "Config validation error" in _plain(result.stderr)
    assert "training -> batch_size" in _plain(result.stderr)


def test_unknown_config_key_does_not_enter_receipt(config):
    with config.open("a", encoding="utf-8") as stream:
        stream.write(f"{_PRIVATE}: true\n")

    result = _invoke("--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason="soup.config.invalid")
    errors = receipt["reasons"][0]["details"]["errors"]
    assert errors and all(error["loc"] and error["type"] for error in errors)
    assert "Config validation error" in _plain(result.stderr)


@pytest.mark.parametrize(
    ("contents", "reason", "expected_rows"),
    [
        pytest.param(None, "soup.data.invalid", None, id="missing"),
        pytest.param("", "soup.data.empty_train", 0, id="empty"),
    ],
)
def test_local_data_refusals(config, cpu_only, launch_tripwires, contents, reason, expected_rows):
    data_path = config.with_name("rows.jsonl")
    if contents is None:
        data_path.unlink()
    else:
        data_path.write_text(contents, encoding="utf-8")

    result = _invoke("--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason=reason)
    assert receipt["data"]["training_rows"] == expected_rows
    assert "Ready to train" not in _plain(result.stderr)
    assert not config.with_name("out").exists()


@pytest.mark.parametrize(
    ("error", "code", "status", "reason"),
    [
        pytest.param(
            RuntimeError(_PRIVATE), 1, "failed", "soup.cli.execution_failed", id="unexpected",
        ),
        pytest.param(
            KeyboardInterrupt(), 130, "cancelled", "soup.train.interrupted", id="interrupt",
        ),
        pytest.param(
            typer.Exit(0), 0, "failed", "soup.train.preflight_incomplete", id="early-zero-exit",
        ),
    ],
)
def test_incomplete_and_failed_preflights_keep_actual_exit(
    config, cpu_only, launch_tripwires, monkeypatch, error, code, status, reason,
):
    loader = Mock(side_effect=error)
    monkeypatch.setattr(train_command, "load_dataset", loader)

    result = _invoke("--dry-run", "--result-json")

    receipt = _check_result(result, code, status=status, reason=reason)
    assert receipt["data"]["training_rows"] is None
    assert "Ready to train" not in _plain(result.stderr)
    loader.assert_called_once()


def test_cpu_preflight_reports_resolved_values_without_fabricated_estimates(
    config, cpu_only, launch_tripwires, monkeypatch,
):
    confirm = Mock(side_effect=AssertionError("dry-run should not prompt"))
    monkeypatch.setattr(train_command, "_confirm_start_training", confirm)

    result = _invoke("--dry-run", "--result-json")

    receipt = _check_result(result, 0, status="ok")
    assert "Config valid. Ready to train!" in _plain(result.stderr)
    data = receipt["data"]
    assert data["dry_run"] is True
    assert (data["training_rows"], data["validation_rows"]) == (4, 1)
    assert data["resolved"] == {
        "base_model": "example/tiny",
        "task": "sft",
        "backend": "transformers",
        "device_name": "Test CPU",
        "max_steps": None,
        "epochs": 2,
        "batch_size_per_device": 2,
        "effective_batch_size": 8,
        "context_length": 512,
        "adapter_rank": 8,
    }
    assert data["vram"] == {
        "estimated_peak_gb": None,
        "available_gb": None,
        "margin_fraction": None,
        "required_gb": None,
    }
    assert data["disk"] == {"estimated_output_gb": None, "estimated_checkpoints_gb": None}
    assert data["estimated_wall_time_s"] is None
    assert data["estimate_basis"] == dict.fromkeys(("vram", "disk", "wall_time"), "unavailable")
    confirm.assert_not_called()
    assert not config.with_name("out").exists()


def _subprocess_env():
    python_path = [str(_REPO / "src")]
    if os.environ.get("PYTHONPATH"):
        python_path.append(os.environ["PYTHONPATH"])
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(python_path),
        "PYTHONUTF8": "1",
        "FORCE_COLOR": "1",
    }


def test_real_entry_point_keeps_chatml_warning_off_binary_stdout(config):
    result = subprocess.run(
        [
            sys.executable, "-m", "soup_cli", "train", "--dry-run", "--result-json",
            "--allow-oom-attempt",
        ],
        cwd=config.parent,
        env=_subprocess_env(),
        capture_output=True,
        check=False,
        timeout=90,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    data = _receipt(result.stdout, 0, status="ok")["data"]
    assert (data["training_rows"], data["validation_rows"]) == (4, 1)
    stderr = _plain(result.stderr.decode("utf-8"))
    assert "Warning: invalid JSON on line 6" in stderr
    assert "Config valid. Ready to train!" in stderr
    assert str(config.parent) not in result.stdout.decode("utf-8")
    assert not config.with_name("out").exists()


def test_legacy_dry_run_and_nontty_confirmation_remain_human(
    config, cpu_only, launch_tripwires, monkeypatch,
):
    monkeypatch.setattr(train_command, "_stdin_isatty", lambda: False)

    dry_run = _invoke("--dry-run")
    confirmation = _invoke()

    assert dry_run.exit_code == 0, (dry_run.output, dry_run.exception)
    assert "Config valid. Ready to train!" in _plain(dry_run.stdout)
    assert dry_run.stderr == ""
    assert confirmation.exit_code == 3, (confirmation.output, confirmation.exception)
    assert "stdin is not a TTY" in _plain(confirmation.stdout)
    assert "--yes" in _plain(confirmation.stdout)
    assert "soup.command_result" not in dry_run.output + confirmation.output
    assert not config.with_name("out").exists()


def test_invalid_typed_cli_option_fails_before_dispatch_without_receipt():
    result = _invoke("--dry-run", "--result-json", "--nodes", "not-an-integer")

    assert result.exit_code == 2
    assert "--nodes" in _plain(result.stderr)
    assert result.stdout_bytes == b""



def test_config_error_details_cap_records_and_report_omissions(config):
    invalid_modules = ", ".join(["7"] * 33)
    yaml = config.read_text(encoding="utf-8").replace(
        "    r: 8", f"    r: 8\n    target_modules: [{invalid_modules}]",
    )
    config.write_text(yaml, encoding="utf-8")

    result = _invoke("--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason="soup.config.invalid")
    errors = receipt["reasons"][0]["details"]["errors"]
    assert len(errors) == 32
    assert errors[0]["loc"][:3] == ["training", "lora", "target_modules"]
    assert receipt["truncated"]["/reasons/0/details/errors"] > 0


def test_invalid_json_data_is_a_known_input_refusal(config, cpu_only, launch_tripwires):
    source = config.with_name(f"{_PRIVATE}.json")
    source.write_text(f"[\n{_PRIVATE}\n", encoding="utf-8")
    config.write_text(
        config.read_text(encoding="utf-8").replace("rows.jsonl", source.name), encoding="utf-8",
    )

    result = _invoke("--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason="soup.data.invalid")
    assert receipt["data"]["training_rows"] is None
    assert "Ready to train" not in _plain(result.stderr)


def test_unresolved_batch_and_full_tuning_rank_remain_null(config, cpu_only, launch_tripwires):
    yaml = config.read_text(encoding="utf-8").replace("batch_size: 2", "batch_size: auto")
    config.write_text(yaml.replace("    r: 8", "    r: 0"), encoding="utf-8")

    result = _invoke("--dry-run", "--result-json")

    resolved = _check_result(result, 0, status="ok")["data"]["resolved"]
    assert resolved["batch_size_per_device"] is None
    assert resolved["effective_batch_size"] is None
    assert resolved["adapter_rank"] is None
    assert resolved["max_steps"] is None


def test_data_digest_uses_exact_source_bytes(config, cpu_only, launch_tripwires):
    source = config.with_name("rows.jsonl")
    original = source.read_bytes()
    first = _check_result(_invoke("--dry-run", "--result-json"), 0, status="ok")["data"]
    source.write_bytes(original + b"\n")
    second = _check_result(_invoke("--dry-run", "--result-json"), 0, status="ok")["data"]

    assert first["input_digests"] == {
        "config_sha256": None, "data_sha256": hashlib.sha256(original).hexdigest(),
    }
    assert second["input_digests"] == {
        "config_sha256": None, "data_sha256": hashlib.sha256(original + b"\n").hexdigest(),
    }
    assert (first["training_rows"], first["validation_rows"]) == (4, 1)
    assert (second["training_rows"], second["validation_rows"]) == (4, 1)


def test_native_preflight_diagnostics_go_to_stderr(config):
    script = r"""
import os
import runpy
import sys

from soup_cli.commands import train

load = train.load_dataset


def noisy_load(*args, **kwargs):
    os.write(1, b"native preflight diagnostic\n")
    return load(*args, **kwargs)


train.load_dataset = noisy_load
sys.argv = ["soup", "train", "--dry-run", "--result-json", "--allow-oom-attempt"]
runpy.run_module("soup_cli", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=config.parent,
        env=_subprocess_env(),
        capture_output=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert _receipt(result.stdout, 0, status="ok")["data"]["training_rows"] == 4
    assert b"native preflight diagnostic" in result.stderr


def test_source_change_during_load_leaves_digest_unknown(
    config, cpu_only, launch_tripwires, monkeypatch,
):
    source = config.with_name("rows.jsonl")
    original = source.read_bytes()
    load = train_command.load_dataset

    def changed_load(*args, **kwargs):
        source.write_bytes(original + b"\n")
        return load(*args, **kwargs)

    monkeypatch.setattr(train_command, "load_dataset", changed_load)
    result = _invoke("--dry-run", "--result-json")

    data = _check_result(result, 0, status="ok")["data"]
    assert data["training_rows"] == 4
    assert data["input_digests"]["data_sha256"] is None



@pytest.mark.parametrize(
    ("task", "data_format", "row", "reason", "diagnostic"),
    [
        pytest.param(
            "classifier", "chatml", {"text": "example"}, "soup.data.invalid",
            "missing required 'label' field", id="classification",
        ),
        pytest.param(
            "tts", "audio", {"audio": "clip.wav", "messages": []}, "soup.data.invalid",
            "CSM row needs exactly one transcript", id="csm",
        ),
        pytest.param(
            "kto", "chatml", {"messages": []}, "soup.data.invalid",
            "lacks prompt, completion or label", id="kto",
        ),
        pytest.param(
            "dpo", "chatml", {"messages": []}, "soup.data.invalid",
            "lacks prompt, chosen or rejected", id="preference-pair",
        ),
        pytest.param(
            "dpo", "dpo", None, "soup.data.empty_train",
            "No training rows: preference data loaded as format", id="preference-empty",
        ),
    ],
)
def test_review_dataset_refusal_has_specific_receipt(
    config, cpu_only, launch_tripwires, monkeypatch, task, data_format, row, reason, diagnostic,
):
    from soup_cli.data import loader as data_loader

    settings = {"quantization": "none"}
    if task == "classifier":
        settings["num_labels"] = 2
    if task == "tts":
        settings["tts_family"] = "sesame_csm"
    config.write_text(
        json.dumps({
            "base": "example/tiny",
            "task": task,
            "modality": "audio_out" if task == "tts" else "text",
            "data": {"train": "rows.jsonl", "format": data_format, "val_split": 0},
            "training": settings,
            "output": "out",
        }),
        encoding="utf-8",
    )
    rows = [] if row is None else [row]
    first_drop = (data_format, 1, "missing preference pair", None) if row is None else None
    load = Mock(return_value={"train": rows, "val": []})
    monkeypatch.setattr(train_command, "load_dataset", load)
    monkeypatch.setattr(
        data_loader, "last_load_outcome",
        lambda: data_loader.LoadOutcome(fmt=data_format, first_drop=first_drop),
    )

    result = _invoke("--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason=reason)
    assert len(receipt["reasons"]) == 1
    assert receipt["data"]["training_rows"] == len(rows)
    stderr = _plain(result.stderr)
    assert diagnostic in stderr
    assert "Ready to train" not in stderr
    assert not config.with_name("out").exists()
    load.assert_called_once()


@pytest.mark.parametrize(
    ("settings", "diagnostic"),
    [
        pytest.param(
            "  reward_fn: judge_rank\n"
            "  grpo_judge: http://localhost:8000/judge-model\n"
            "  grpo_judge_max_calls: 1\n",
            "Judge budget:", id="judge-budget",
        ),
        pytest.param("  quantization_aware: fp8\n", "QAT error:", id="qat"),
        pytest.param("  use_galore: true\n", "GaLore error:", id="galore"),
    ],
)
def test_review_uncoded_refusal_uses_fallback(
    config, cpu_only, launch_tripwires, monkeypatch, settings, diagnostic,
):
    is_judge_budget = diagnostic == "Judge budget:"
    source = config.read_text(encoding="utf-8")
    if is_judge_budget:
        source = source.replace("task: sft\n", "task: grpo\n")
    elif diagnostic == "QAT error:":
        monkeypatch.setattr("soup_cli.utils.fp8.is_torchao_float8_available", lambda: False)
        monkeypatch.setattr("soup_cli.utils.fp8.fp8_training_supported", lambda *_: (True, ""))
    else:
        gate = Mock(return_value=["GaLore validation failed in this preflight environment."])
        monkeypatch.setattr("soup_cli.utils.galore.validate_galore_config", gate)
    config.write_text(
        source.replace("training:\n", "training:\n" + settings), encoding="utf-8",
    )
    load = Mock(return_value={"train": [{"prompt": "Example question"}], "val": []})
    monkeypatch.setattr(train_command, "load_dataset", load)

    result = _invoke("--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason="soup.train.preflight_refused")
    assert len(receipt["reasons"]) == 1
    assert receipt["data"]["training_rows"] == (1 if is_judge_budget else None)
    stderr = _plain(result.stderr)
    assert diagnostic in stderr
    assert "Ready to train" not in stderr
    assert not config.with_name("out").exists()
    if is_judge_budget:
        assert "grpo_judge_max_calls=1" in stderr
        load.assert_called_once()
    else:
        load.assert_not_called()


@pytest.fixture
def review_preload_tripwires(monkeypatch, launch_tripwires):
    spies = {}
    for name in ("detect_device", "load_dataset"):
        spy = Mock(side_effect=AssertionError(f"invalid config reached {name}"))
        monkeypatch.setattr(train_command, name, spy)
        spies[name] = spy
    yield
    for spy in spies.values():
        spy.assert_not_called()


def test_review_unknown_chat_template_is_validation(config, review_preload_tripwires):
    source = config.read_text(encoding="utf-8")
    config.write_text(
        source.replace("data:\n", f"data:\n  chat_template: {_PRIVATE}\n"),
        encoding="utf-8",
    )

    result = _invoke("--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason="soup.config.invalid")
    assert receipt["reasons"][0]["details"]["category"] == "validation"
    assert "Invalid data.chat_template" in _plain(result.stderr)
    assert not config.with_name("out").exists()


def test_review_invalid_replay_override_is_validation(config, review_preload_tripwires):
    result = _invoke("--dry-run", "--result-json", "--replay-ratio", "0.2")

    receipt = _check_result(result, 2, status="refused", reason="soup.config.invalid")
    assert receipt["reasons"][0]["details"]["category"] == "validation"
    assert "require data.replay" in _plain(result.stderr)
    assert receipt["data"] is None
    assert not config.with_name("out").exists()


@pytest.mark.parametrize(
    ("fixture_kind", "category", "diagnostic"),
    [
        pytest.param("directory", "unreadable", "is a directory", id="directory"),
        pytest.param("non-utf8", "unreadable", "is not valid UTF-8", id="non-utf8"),
        pytest.param("io-error", "unreadable", "cannot be read: review read denied", id="io-error"),
        pytest.param(
            "oversize", "yaml_limit", "is larger than {max_bytes} bytes; refusing", id="oversize",
        ),
    ],
)
def test_review_config_read_failure_categories(
    tmp_path, monkeypatch, review_preload_tripwires, fixture_kind, category, diagnostic,
):
    from soup_cli.utils.yaml_limits import MAX_CONFIG_BYTES

    path = tmp_path / f"{_PRIVATE}.yaml"
    if fixture_kind == "directory":
        path.mkdir()
    elif fixture_kind == "non-utf8":
        path.write_bytes(b"base: example/tiny\n# \xff\n")
    elif fixture_kind == "io-error":
        path.write_text("base: example/tiny\n", encoding="utf-8")
        monkeypatch.setattr(
            "soup_cli.config.loader.load_yaml_file_limited",
            Mock(side_effect=PermissionError(13, "review read denied")),
        )
    else:
        path.write_bytes(b"#" + b"x" * MAX_CONFIG_BYTES)

    result = _invoke("--config", str(path), "--dry-run", "--result-json")

    receipt = _check_result(result, 1, status="refused", reason="soup.config.invalid")
    assert receipt["reasons"][0]["details"] == {"category": category, "errors": []}
    assert receipt["data"] is None
    assert diagnostic.format(max_bytes=MAX_CONFIG_BYTES) in _plain(result.stderr)
    assert not (tmp_path / "out").exists()
