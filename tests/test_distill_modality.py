"""Distillation must not accept modalities its text-only trainer never reads."""

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from tests.conftest import strip_ansi


def _config(modality, mode="token", backend="transformers"):
    return {
        "base": "org/student", "task": "distill", "backend": backend,
        "modality": modality, "data": {"train": "./train.jsonl"},
        "training": {"teacher_model": "org/teacher", "distill_mode": mode},
    }


@pytest.mark.parametrize("modality", ["vision", "audio"])
@pytest.mark.parametrize("mode", ["token", "sequence"])
@pytest.mark.parametrize("loader", [SoupConfig.model_validate, lambda raw:
                         load_config_from_string(yaml.safe_dump(raw))])
def test_nontext_distillation_is_refused_at_load(modality, mode, loader):
    with pytest.raises(ValueError) as caught:
        loader(_config(modality, mode))
    message = str(caught.value)
    assert "task='distill' requires modality='text'" in message
    assert f"got modality={modality!r}" in message
    assert "text-only" in message


@pytest.mark.parametrize("mode", ["token", "sequence"])
def test_text_distillation_still_loads(mode):
    cfg = load_config_from_string(yaml.safe_dump(_config("text", mode)))
    assert cfg.modality == "text" and cfg.training.distill_mode == mode


@pytest.mark.parametrize("backend", ["mlx", "unsloth"])
def test_existing_backend_refusal_keeps_its_reason(backend):
    with pytest.raises(ValueError, match=f"not supported on backend={backend}"):
        SoupConfig.model_validate(_config("text", backend=backend))


@pytest.mark.parametrize("modality", ["vision", "audio"])
@pytest.mark.parametrize("color", [False, True])
def test_doctor_reports_unsupported_modality(tmp_path, modality, color):
    path = tmp_path / "soup.yaml"
    path.write_text(yaml.safe_dump(_config(modality)), encoding="utf-8")
    result = CliRunner().invoke(
        app, ["doctor", "--config", str(path)], color=color,
        env={"FORCE_COLOR": "1" if color else "0"},
    )
    assert result.exit_code == 2, (result.output, result.exception)
    output = " ".join(strip_ansi(result.output).split())
    assert "task='distill' requires modality='text'" in output
    assert f"got modality={modality!r}" in output
