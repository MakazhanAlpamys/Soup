"""``soup plan`` and ``soup apply`` refuse the configs ``soup train`` refuses.

#909 gave both commands the loader's unknown-key check "so the pre-flight and the run
agree about which configs are legal", but neither built the config. A staged field set
off its default, a refused deprecated value or any other schema refusal planned cleanly
(exit 0, state file written), and ``soup apply`` / ``apply --dry-run`` accepted it, one
command before ``soup train`` refused the same file. Both commands now run the loader's
own validation, so they exit 1 with the lines ``soup train`` prints, write no plan and
leave an existing state file as it was.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.terraform_plan import TrainingState, build_plan, read_state, write_state
from tests._tiny_checkpoint import write_tiny_model
from tests.conftest import strip_ansi

_HEADER = "Config validation error:"
_STATE = "soup.tfstate"

#: case -> (training extras, data extras, what the refusal names, the fix it gives)
_REFUSED = {
    "staged field": (
        {"early_stop_patience": 5},
        {},
        "training.early_stop_patience",
        "Delete the line.",
    ),
    "deprecated value": (
        {},
        {"remove_unused_columns": True},
        "data.remove_unused_columns: true",
        "Delete the line.",
    ),
    "value out of range": (
        {"lr": -1},
        {},
        "training -> lr",
        "greater than 0",
    ),
}


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "tiny")
    (tmp_path / "train.jsonl").write_text(
        '{"instruction": "hi", "output": "there"}\n', encoding="utf-8"
    )
    return tmp_path


def _config(workdir: Path, *, training: dict | None = None, data: dict | None = None) -> dict:
    return {
        "base": str(workdir / "tiny"),
        "task": "sft",
        "data": {"train": "train.jsonl", "max_length": 512, **(data or {})},
        "training": {"batch_size": 2, "quantization": "none", **(training or {})},
    }


def _write(config: dict) -> None:
    Path("soup.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _train_refusal() -> str:
    """What ``soup train`` prints for ``soup.yaml``, from the refusal header on."""
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    out = _plain(result.output)
    assert _HEADER in out, out
    return out[out.index(_HEADER):]


@pytest.mark.parametrize("case", sorted(_REFUSED))
def test_soup_plan_refuses_with_train_s_message_and_writes_no_plan(workdir, case):
    training, data, names, fix = _REFUSED[case]
    _write(_config(workdir, training=training, data=data))
    expected = _train_refusal()

    result = CliRunner().invoke(app, ["plan", "--config", "soup.yaml", "--state", _STATE])

    assert result.exit_code == 1, (result.output, repr(result.exception))
    out = _plain(result.output)
    assert out == expected, out
    assert names in out, out
    assert fix in out, out
    assert "Plan written" not in out, out
    assert not Path(_STATE).exists()


@pytest.mark.parametrize("dry_run", [False, True], ids=["apply", "apply-dry-run"])
@pytest.mark.parametrize("case", sorted(_REFUSED))
def test_soup_apply_refuses_with_train_s_message_and_leaves_the_state_alone(
    workdir, case, dry_run
):
    training, data, names, fix = _REFUSED[case]
    config = _config(workdir, training=training, data=data)
    _write(config)
    # The state ``soup plan`` used to write for this very file: no drift, so ``apply``
    # locked it (exit 0) and ``apply --dry-run`` said it would proceed.
    write_state(
        TrainingState(plan=build_plan(config), applied=False, applied_at=None, run_id=None),
        _STATE,
    )
    before = Path(_STATE).read_bytes()
    expected = _train_refusal()

    argv = ["apply", "--config", "soup.yaml", "--state", _STATE]
    result = CliRunner().invoke(app, argv + (["--dry-run"] if dry_run else []))

    assert result.exit_code == 1, (result.output, repr(result.exception))
    out = _plain(result.output)
    assert out == expected, out
    assert names in out, out
    assert fix in out, out
    assert "would proceed" not in out, out
    assert "Plan locked" not in out, out
    assert Path(_STATE).read_bytes() == before
    assert read_state(_STATE).applied is False


def test_a_valid_config_still_plans_and_applies(workdir):
    config = _config(workdir)
    _write(config)
    runner = CliRunner()

    planned = runner.invoke(app, ["plan", "--config", "soup.yaml", "--state", _STATE])
    assert planned.exit_code == 0, (planned.output, repr(planned.exception))
    assert f"Plan written to {_STATE}." in _plain(planned.output)
    assert _HEADER not in _plain(planned.output)
    state = read_state(_STATE)
    assert state.applied is False
    # The plan is still computed from the file as written: same SHA, same numbers.
    assert state.plan == build_plan(config)
    assert state.plan.peak_vram_gb is not None, state.plan.peak_vram_reason

    checked = runner.invoke(
        app, ["apply", "--config", "soup.yaml", "--state", _STATE, "--dry-run"]
    )
    assert checked.exit_code == 0, (checked.output, repr(checked.exception))
    assert "No drift detected; would proceed." in _plain(checked.output)
    assert read_state(_STATE).applied is False

    applied = runner.invoke(app, ["apply", "--config", "soup.yaml", "--state", _STATE])
    assert applied.exit_code == 0, (applied.output, repr(applied.exception))
    assert f"Plan locked in {_STATE}." in _plain(applied.output)
    assert read_state(_STATE).applied is True
