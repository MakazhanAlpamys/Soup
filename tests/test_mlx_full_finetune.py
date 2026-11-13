"""#1992: public configuration contract for dense native MLX text SFT."""

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string


def _config(**training):
    return load_config_from_string(yaml.safe_dump({
        "base": "./local-llama", "backend": "mlx", "task": "sft",
        "data": {"train": "./train.jsonl"},
        "training": {"quantization": "none", "lora": {"r": 0}, **training},
    }))


@pytest.mark.parametrize("freezing", [{}, {"freeze_layers": 1}, {"freeze_ratio": 0.5}])
def test_public_dense_config_and_dispatch(freezing):
    from soup_cli.trainer.mlx_routing import resolve_trainer
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    config = _config(**freezing)
    assert config.training.lora.r == 0
    assert resolve_trainer(config)[0] is MLXSFTTrainerWrapper
    assert load_config_from_string(yaml.safe_dump(config.model_dump(mode="json"))) == config


@pytest.mark.parametrize("field,value", [("freeze_layers", 1), ("freeze_ratio", 0.5)])
def test_freezing_with_lora_stays_refused(field, value):
    with pytest.raises(ValueError, match=field):
        _config(lora={"r": 4}, **{field: value})


@pytest.mark.parametrize("training,match", [
    ({"quantization": "4bit"}, "quantization"),
    ({"lisa_enabled": True}, "lisa_enabled"),
    ({"unfrozen_parameters": ["model.layers.0"]}, "unfrozen_parameters"),
    ({"lora": {"r": 0, "use_dora": True}}, "use_dora"),
    ({"lora": {"r": 0, "use_rslora": True}}, "use_rslora"),
    ({"lora": {"r": 0, "rank_pattern": {"q_proj": 4}}}, "rank_pattern"),
    ({"lora": {"r": 0, "alpha": 32}}, "alpha"),
    ({"lora": {"r": 0, "dropout": 0.2}}, "dropout"),
    ({"lora": {"r": 0, "target_modules": ["q_proj"]}}, "target_modules"),
    ({"freeze_layers": 0}, "freeze_layers"),
    ({"freeze_ratio": 1}, "freeze_ratio"),
])
def test_named_refusals(training, match):
    with pytest.raises(ValueError, match=match):
        _config(**training)


def test_neighboring_lora_still_loads():
    assert _config(lora={"r": 4}).training.lora.r == 4


@pytest.mark.parametrize("field", ["add_new_tokens", "new_special_tokens"])
def test_dense_vocab_expansion_stays_refused(field):
    raw = _config().model_dump(mode="json")
    raw["data"][field] = ["<new>"]
    with pytest.raises(ValueError, match=field):
        load_config_from_string(yaml.safe_dump(raw))


@pytest.mark.parametrize("flag", ["--resume", "--hf-resume"])
def test_public_command_refuses_resume_before_dataset_or_model_load(tmp_path, flag):
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from tests.conftest import strip_ansi

    path = tmp_path / "soup.yaml"
    path.write_text(yaml.safe_dump(_config().model_dump(mode="json")))
    options = ["train", "--config", str(path), "--yes", flag]
    if flag == "--resume":
        options.append("auto")
    else:
        options.extend(["--push-as", "org/unused"])
    result = CliRunner().invoke(app, options)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"does not support {flag}: optimizer/RNG-state" in strip_ansi(result.output)
