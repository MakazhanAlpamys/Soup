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
    with pytest.raises(ValueError) as exc:
        _config(lora={"r": 4}, **{field: value})
    assert str(exc.value) == (
        f": Value error, training.{field} is not applied by task='sft' with backend='mlx': "
        "layers are frozen only on the text path with backend: transformers and no layer "
        "streaming, so this run would train every layer. Use backend: transformers, "
        "or remove the key."
    )


@pytest.mark.parametrize("training,message", [
    (
        {"quantization": "4bit"},
        ": Value error, training.lora.r=0 (full fine-tuning) requires quantization='none' "
        "(got '4bit'); quantized weights cannot be trained directly. For a quantized run "
        "use LoRA (QLoRA) with lora.r >= 1.",
    ),
    (
        {"lisa_enabled": True},
        ": Value error, training.lisa_enabled requires backend='transformers'; got backend='mlx'",
    ),
    (
        {"unfrozen_parameters": ["model.layers.0"]},
        ": Value error, training.unfrozen_parameters requires backend='transformers'; "
        "got backend='mlx'",
    ),
    ({"freeze_layers": 0}, "training -> freeze_layers: Input should be greater than or equal to 1"),
    ({"freeze_ratio": 1}, "training -> freeze_ratio: Input should be less than 1"),
])
def test_named_refusals(training, message):
    with pytest.raises(ValueError) as exc:
        _config(**training)
    assert str(exc.value) == message


@pytest.mark.parametrize("field,value", [
    ("use_dora", True), ("use_rslora", True), ("rank_pattern", {"q_proj": 4}),
    ("alpha_pattern", {"q_proj": 4}), ("alpha", 32), ("dropout", 0.2),
    ("target_modules", ["q_proj"]),
])
def test_dense_lora_controls_are_refused_by_exact_name(field, value):
    with pytest.raises(ValueError) as exc:
        _config(lora={"r": 0, field: value})
    assert str(exc.value) == (
        ": Value error, training.lora.r=0 means full fine-tuning (no adapter), so it "
        f"is mutually exclusive with LoRA features: lora.{field}. Set lora.r >= 1 to use them."
    )


def test_neighboring_lora_still_loads():
    assert _config(lora={"r": 4}).training.lora.r == 4


@pytest.mark.parametrize("field", ["add_new_tokens", "new_special_tokens"])
def test_dense_vocab_expansion_stays_refused(field):
    raw = _config().model_dump(mode="json")
    raw["data"][field] = ["<new>"]
    with pytest.raises(ValueError) as exc:
        load_config_from_string(yaml.safe_dump(raw))
    assert str(exc.value) == (
        f": Value error, MLX full fine-tuning does not support data.{field}; "
        "vocabulary expansion currently requires LoRA."
    )


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
