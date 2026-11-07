"""Multimodal SFT must not silently drop a requested Unsloth backend."""

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from tests.conftest import strip_ansi


def _config(modality, backend="unsloth"):
    return {
        "base": "org/model", "task": "sft", "backend": backend,
        "modality": modality, "data": {"train": "./train.jsonl"},
    }


@pytest.mark.parametrize("modality", ["vision", "audio"])
@pytest.mark.parametrize("loader", [SoupConfig.model_validate, lambda raw:
                         load_config_from_string(yaml.safe_dump(raw))])
def test_unsloth_multimodal_sft_is_refused_at_load(modality, loader):
    with pytest.raises(ValueError) as caught:
        loader(_config(modality))
    message = str(caught.value)
    assert "task='sft' with backend='unsloth' requires modality='text'" in message
    assert f"got modality={modality!r}" in message
    assert "Use backend: transformers" in message


@pytest.mark.parametrize("modality", ["vision", "audio", "text"])
def test_transformers_sft_still_loads(modality):
    cfg = load_config_from_string(yaml.safe_dump(_config(modality, "transformers")))
    assert cfg.backend == "transformers" and cfg.modality == modality


def test_unsloth_text_sft_still_loads():
    cfg = SoupConfig.model_validate(_config("text"))
    assert cfg.backend == "unsloth" and cfg.modality == "text"


@pytest.mark.parametrize("modality", ["vision", "audio"])
@pytest.mark.parametrize("inline", [False, True])
def test_shipped_multimodal_templates_use_supported_backend(modality, inline):
    from soup_cli.config.schema import TEMPLATES
    from soup_cli.templates import load_template

    text = TEMPLATES[modality] if inline else load_template(modality)
    cfg = load_config_from_string(text)
    assert cfg.backend == "transformers" and cfg.modality == modality
    assert "backend: unsloth" not in text


@pytest.mark.parametrize("modality", ["vision", "audio"])
@pytest.mark.parametrize("color", [False, True])
def test_doctor_reports_ignored_backend(tmp_path, modality, color):
    path = tmp_path / "soup.yaml"
    path.write_text(yaml.safe_dump(_config(modality)), encoding="utf-8")
    result = CliRunner().invoke(
        app, ["doctor", "--config", str(path)], color=color,
        env={"FORCE_COLOR": "1" if color else "0"},
    )
    assert result.exit_code == 2, (result.output, result.exception)
    output = " ".join(strip_ansi(result.output).split())
    assert "task='sft' with backend='unsloth' requires modality='text'" in output
    assert "Use backend: transformers" in output
