"""#1309: refuse GDPO at config load instead of silently using TRL's loss."""

from types import SimpleNamespace

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TrainingConfig
from soup_cli.utils.ebft_gdpo import GDPO_VARIANTS, attach_gdpo_compute_loss, get_gdpo_spec
from tests.conftest import strip_ansi


def _yaml(value: str, *, task: str = "dpo", backend: str = "transformers") -> str:
    return (
        f"base: some-model\ntask: {task}\nbackend: {backend}\n"
        "data: {train: ./data.jsonl}\n"
        f"training: {{gdpo_variant: {value}}}\n"
    )


@pytest.mark.parametrize(
    "value", ["standard", "length_normalized", "margin", "bogus", "''", "true", "1", "[]", "{}"]
)
def test_loader_refuses_every_non_null_value(value):
    with pytest.raises(ValueError, match=r"training.gdpo_variant is refused \(#1309\)"):
        load_config_from_string(_yaml(value))


@pytest.mark.parametrize("value", [*sorted(GDPO_VARIANTS), "bogus", "", True, 1, [], {}])
def test_direct_construction_refuses_every_non_null_value(value):
    with pytest.raises(ValidationError, match="#1309") as excinfo:
        TrainingConfig(gdpo_variant=value)
    assert [error["loc"] for error in excinfo.value.errors()] == [("gdpo_variant",)]


@pytest.mark.parametrize(
    "task,backend", [("sft", "transformers"), ("dpo", "mlx"), ("dpo", "unsloth")]
)
def test_refusal_precedes_task_and_backend_gates(task, backend):
    with pytest.raises(ValueError, match="#1309"):
        load_config_from_string(_yaml("standard", task=task, backend=backend))


@pytest.mark.parametrize("loss", ["dpo", "simpo", "orpo", "ipo", "bco"])
def test_preference_loss_cannot_silently_ignore_gdpo(loss):
    text = _yaml("margin", task="preference").replace(
        "gdpo_variant: margin", f"gdpo_variant: margin, preference_loss: {loss}"
    )
    with pytest.raises(ValueError, match="#1309"):
        load_config_from_string(text)


def test_error_explains_alternatives_without_claiming_equivalence():
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("length_normalized"))
    message = str(excinfo.value)
    for part in [
        "dpo_loss",
        "Remove gdpo_variant",
        "task: dpo",
        "task: simpo",
        "not identical",
        "margin has no equivalent",
    ]:
        assert part in message


@pytest.mark.parametrize("task", ["dpo", "preference", "simpo", "sft"])
@pytest.mark.parametrize("value", ["null", "Null", "NULL", "~", ""])
def test_null_keeps_plain_training_loadable(task, value):
    text = _yaml(value, task=task)
    if task == "preference":
        text = text.replace("gdpo_variant:", "preference_loss: dpo, gdpo_variant:")
    assert load_config_from_string(text).training.gdpo_variant is None


def test_unset_keeps_plain_dpo_loadable_and_hook_inactive():
    cfg = load_config_from_string("base: x\ntask: dpo\ndata: {train: ./data.jsonl}\n")
    assert cfg.training.gdpo_variant is None
    assert attach_gdpo_compute_loss(object(), cfg.training) is False
    assert TrainingConfig().gdpo_variant is None
    assert TrainingConfig(gdpo_variant=None).gdpo_variant is None


def test_metadata_does_not_advertise_live_wiring():
    assert all(not get_gdpo_spec(variant).live_wired for variant in GDPO_VARIANTS)


def test_legacy_hook_cannot_silently_ignore_missing_surface():
    with pytest.raises(ValueError, match="dpo_loss.*#1309"):
        attach_gdpo_compute_loss(object(), SimpleNamespace(gdpo_variant="standard"))


def _write_config(tmp_path):
    path = tmp_path / "soup.yaml"
    path.write_text(_yaml("standard"), encoding="utf-8")
    return path


@pytest.mark.parametrize("extra_args", [[], ["--dry-run"]], ids=["train", "dry-run"])
def test_train_stops_before_model_setup(tmp_path, monkeypatch, extra_args):
    from soup_cli.cli import app

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    path = _write_config(tmp_path)
    result = CliRunner().invoke(app, ["train", "--config", str(path), *extra_args])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "training -> gdpo_variant" in output
    assert "#1309" in output
    for later_line in ("Training Setup", "LoRA applied", "Loading model"):
        assert later_line not in output


def test_doctor_reports_config_as_unloadable(tmp_path, capsys):
    from soup_cli.commands.doctor import doctor

    path = _write_config(tmp_path)
    with pytest.raises(typer.Exit) as excinfo:
        doctor(nccl=False, disk=False, config=str(path))
    assert excinfo.value.exit_code == 2
    output = " ".join(strip_ansi(capsys.readouterr().out).split())
    assert "training -> gdpo_variant" in output
    assert "#1309" in output
