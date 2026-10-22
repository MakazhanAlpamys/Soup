"""Portable native-distillation configuration and public routing contracts."""

import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.mlx_routing import resolve_trainer


def recipe(**training):
    return {
        "base": "org/student",
        "backend": "mlx",
        "task": "distill",
        "modality": "text",
        "data": {"train": "./train.jsonl", "max_length": 64},
        "training": {
            "teacher_model": "org/teacher",
            "quantization": "none",
            "lora": {"r": 4, "alpha": 8},
            **training,
        },
    }


@pytest.mark.parametrize("divergence", [None, "forward_kl", "kl", "reverse_kl", "js"])
def test_supported_token_configuration_routes_to_native_trainer(divergence):
    raw = recipe(**({"distill_divergence": divergence} if divergence is not None else {}))
    # The Transformers control isolates the current missing-MLX-support refusal.
    control = SoupConfig.model_validate({**raw, "backend": "transformers"})
    assert control.task == "distill"
    cfg = SoupConfig.model_validate(raw)
    assert cfg.training.distill_mode == "token"
    cls, kwargs = resolve_trainer(cfg)
    assert cls.__name__ == "MLXDistillTrainerWrapper"
    assert cls.__module__ == "soup_cli.trainer.mlx_distill"
    assert kwargs == {}


@pytest.mark.parametrize("field,value,diagnostic", [
    ("distill_mode", "sequence", "distill_mode"),
    ("quantization", "4bit", "quantization"),
    ("optimizer", "lion", "optimizer"),
    ("scheduler", "polynomial", "scheduler"),
    ("gradient_checkpointing", "full", "gradient_checkpointing"),
    ("neftune_alpha", 0.2, "neftune_alpha"),
    ("use_liger", True, "use_liger"),
    ("lora", {"r": 0}, "lora.r"),
    ("lora", {"r": 4, "use_dora": True}, "[Dd][Oo][Rr][Aa]|use_dora"),
    ("lora", {"r": 4, "rank_pattern": {"q_proj": 2}}, "rank_pattern"),
])
def test_unsupported_native_modifiers_refuse_by_name_with_valid_control(field, value, diagnostic):
    assert SoupConfig.model_validate(recipe()).task == "distill"
    with pytest.raises(ValueError, match=diagnostic):
        SoupConfig.model_validate(recipe(**{field: value}))


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), 0.049, 100.01])
def test_bad_temperatures_never_coerce_to_supported_runtime(value):
    with pytest.raises((ValueError, TypeError), match="temperature"):
        SoupConfig.model_validate(recipe(distill_temperature=value))


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_bad_chunk_sizes_refuse_before_loading(value):
    with pytest.raises((ValueError, TypeError), match="chunk_size"):
        SoupConfig.model_validate(recipe(distill_chunk_size=value))


def test_public_offline_dry_run_with_real_data_does_not_require_mlx(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    data = tmp_path / "train.jsonl"
    data.write_text('{"text": "alpha beta gamma"}\n' * 4)
    raw = recipe()
    raw["data"].update(train=str(data), val_split=0)
    cfg = SoupConfig.model_validate(raw)
    path = tmp_path / "soup.json"
    path.write_text(cfg.model_dump_json())
    outcome = CliRunner().invoke(app, ["train", "--config", str(path), "--dry-run"])
    assert outcome.exit_code == 0, (outcome.output, repr(outcome.exception))
    assert "Config valid" in outcome.output


@pytest.mark.parametrize("option", ["deepspeed_config", "fsdp_config", "trust_remote_code"])
def test_native_setup_refuses_distributed_or_remote_code_before_backend_import(option):
    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    cfg = SoupConfig.model_validate(recipe())
    trainer = MLXDistillTrainerWrapper(cfg, **{option: True})
    with pytest.raises(ValueError, match="distributed or remote-code"):
        trainer.setup({"train": [{"text": "alpha beta"}]})


def test_missing_native_dependency_has_actionable_error(monkeypatch):
    import builtins

    from soup_cli.trainer.mlx_distill import MLXDistillTrainerWrapper

    original = builtins.__import__

    def without_mlx(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx."):
            raise ImportError("unavailable backend boundary")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mlx)
    trainer = MLXDistillTrainerWrapper(SoupConfig.model_validate(recipe()))
    with pytest.raises(ImportError, match=r'soup-cli\[mlx\]'):
        trainer.setup({"train": [{"text": "alpha beta"}]})
