"""Acceptance tests for training.curriculum_dynamic telemetry contract (Refs #808).

Ensures that:
1. `curriculum_dynamic` and its tuning fields document their contract
   as online uncertainty telemetry tracking and curve inspection (`soup runs curriculum-curve`),
   written only to `curriculum_history.jsonl` without modifying which samples a batch contains.
2. Callback docstrings and accessors accurately state telemetry intent.
3. Dynamic curriculum callback correctly logs valid non-uniform per-bucket weights to
   `curriculum_history.jsonl` and updates `current_weights` when fed through `on_log`.
4. `soup runs curriculum-curve` visualises the generated history cleanly end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.config.schema import TrainingConfig
from soup_cli.monitoring import curriculum_callback as cb_mod
from soup_cli.monitoring.curriculum_callback import DynamicCurriculumCallback
from soup_cli.utils import curriculum_dynamic as utils_curriculum_dynamic
from soup_cli.utils.curriculum_dynamic import (
    DynamicCurriculumPolicy,
    parse_history_jsonl,
    render_curve,
)
from tests.conftest import strip_ansi

runner = CliRunner()


@pytest.mark.parametrize(
    "source_text",
    [
        TrainingConfig.model_fields["curriculum_dynamic"].description or "",
        TrainingConfig.model_fields[
            "curriculum_dynamic_recompute_steps"
        ].description
        or "",
        TrainingConfig.model_fields["curriculum_dynamic_floor"].description
        or "",
        TrainingConfig.model_fields[
            "curriculum_dynamic_temperature"
        ].description
        or "",
        utils_curriculum_dynamic.__doc__ or "",
        DynamicCurriculumPolicy.__doc__ or "",
        utils_curriculum_dynamic.compute_bucket_weights.__doc__ or "",
        utils_curriculum_dynamic.validate_distributed_curriculum.__doc__ or "",
        cb_mod.__doc__ or "",
        DynamicCurriculumCallback.__doc__ or "",
    ],
    ids=[
        "curriculum_dynamic",
        "curriculum_dynamic_recompute_steps",
        "curriculum_dynamic_floor",
        "curriculum_dynamic_temperature",
        "utils_curriculum_dynamic_doc",
        "dynamic_curriculum_policy_doc",
        "compute_bucket_weights_doc",
        "validate_distributed_curriculum_doc",
        "callback_module_doc",
        "dynamic_curriculum_callback_doc",
    ],
)
def test_telemetry_descriptions_avoid_steering_and_sampler_terms(
    source_text: str,
):
    """Schema fields and curriculum docstrings avoid claims of steering the sampler."""
    lower = source_text.lower()
    assert "sampler" not in lower
    assert "starv" not in lower
    assert "re-weight" not in lower


def test_training_guide_documents_telemetry_without_changing_batch_samples() -> None:
    """The curriculum guide must retain the user-visible telemetry limitation."""
    guide = Path(__file__).resolve().parents[1] / "docs" / "training.md"
    section = guide.read_text(encoding="utf-8").split(
        "## Curriculum-Aware Training (BETA)\n", 1,
    )[1].split("\n## ", 1)[0]
    text = " ".join(section.split())
    assert "they do not change which samples a batch contains" in text
    assert "re-weights the sampler" not in text


def test_dynamic_curriculum_validation_explains_the_telemetry_requirement() -> None:
    """The refusal must describe telemetry rather than dynamic sampling."""
    with pytest.raises(ValueError) as caught:
        TrainingConfig(curriculum_dynamic=True, curriculum=False)
    message = str(caught.value)
    assert "curriculum_dynamic requires curriculum=true" in message
    assert "dynamic curriculum telemetry needs the static bucketer" in message
    assert "re-weighting" not in message


def test_schema_curriculum_dynamic_telemetry_contract():
    """curriculum_dynamic description specifies telemetry without modifying sampling."""
    desc = TrainingConfig.model_fields["curriculum_dynamic"].description or ""
    assert "BETA:" in desc
    assert "telemetry and curve inspection" in desc
    assert "soup runs curriculum-curve" in desc
    assert (
        "The weights are only written to curriculum_history.jsonl; "
        "they do not change which samples a batch contains"
    ) in desc
    assert "#808" in desc


def test_schema_curriculum_dynamic_recompute_steps_telemetry_contract():
    """curriculum_dynamic_recompute_steps specifies telemetry tracking and cites #808."""
    desc = (
        TrainingConfig.model_fields[
            "curriculum_dynamic_recompute_steps"
        ].description
        or ""
    )
    assert "telemetry tracking" in desc
    assert "#808" in desc


def test_schema_curriculum_dynamic_floor_and_temperature_telemetry_contract():
    """floor and temperature specify telemetry distribution constraints."""
    floor_desc = (
        TrainingConfig.model_fields["curriculum_dynamic_floor"].description
        or ""
    )
    assert "logged telemetry weights" in floor_desc

    temp_desc = (
        TrainingConfig.model_fields[
            "curriculum_dynamic_temperature"
        ].description
        or ""
    )
    assert "logged bucket weights" in temp_desc
    assert "flattens the logged distribution" in temp_desc
    assert "sharpens the logged distribution" in temp_desc


def test_callback_docstrings_specify_telemetry():
    """Callback docstrings and accessors document telemetry contract."""
    assert "#808" in cb_mod.__doc__
    assert "tracked bucket weights" in cb_mod.__doc__

    cb_doc = " ".join((DynamicCurriculumCallback.__doc__ or "").split())
    assert "#808" in cb_doc
    assert (
        "The weights are only written to curriculum_history.jsonl; "
        "they do not change which samples a batch contains"
    ) in cb_doc

    prop_doc = DynamicCurriculumCallback.current_weights.__doc__ or ""
    assert "#808" in prop_doc
    assert "telemetry weights" in prop_doc


class DummyArgs:
    pass


class DummyState:
    def __init__(self, step: int):
        self.global_step = step


class DummyControl:
    pass


def _setup_trained_callback(output_dir: Path) -> DynamicCurriculumCallback:
    """Helper creating and exercising a callback with varied losses/grad norms."""
    policy = DynamicCurriculumPolicy(
        num_buckets=3,
        recompute_every_n_steps=3,
        floor=0.05,
        temperature=1.0,
    )
    callback = DynamicCurriculumCallback(
        policy=policy,
        curriculum_metric="loss",
        output_dir=str(output_dir),
    )

    # The first sample uses round-robin warm-up; later samples are assigned by
    # their loss percentile against the previously observed signal window.
    # global_step=0 -> bucket 0 (warm-up)
    callback.on_log(
        DummyArgs(),
        DummyState(0),
        DummyControl(),
        logs={"loss": 1.0, "grad_norm": 0.2},
    )
    # loss=10 exceeds the previous signal -> bucket 2 (hardest)
    callback.on_log(
        DummyArgs(),
        DummyState(1),
        DummyControl(),
        logs={"loss": 10.0, "grad_norm": 8.0},
    )
    # loss=2.5 lies between the previous signals -> bucket 1 (medium)
    callback.on_log(
        DummyArgs(),
        DummyState(2),
        DummyControl(),
        logs={"loss": 2.5, "grad_norm": 1.0},
    )

    # Step 3 triggers recompute_every_n_steps=3 boundary
    callback.on_step_end(
        DummyArgs(),
        DummyState(3),
        DummyControl(),
    )
    return callback


def test_callback_logs_history_and_updates_telemetry_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Callback logs non-uniform weights with hardest bucket largest when fed through on_log."""
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    policy = DynamicCurriculumPolicy(
        num_buckets=3,
        recompute_every_n_steps=3,
        floor=0.05,
        temperature=1.0,
    )
    callback = DynamicCurriculumCallback(
        policy=policy,
        curriculum_metric="loss",
        output_dir=str(out_dir),
    )

    # Initial weights are uniform
    assert len(callback.current_weights) == 3
    for w in callback.current_weights:
        assert abs(w - 1.0 / 3) < 1e-5

    # Feed distinct losses and grad norms per bucket
    callback.on_log(
        DummyArgs(),
        DummyState(0),
        DummyControl(),
        logs={"loss": 1.0, "grad_norm": 0.2},
    )
    callback.on_log(
        DummyArgs(),
        DummyState(1),
        DummyControl(),
        logs={"loss": 10.0, "grad_norm": 8.0},
    )
    callback.on_log(
        DummyArgs(),
        DummyState(2),
        DummyControl(),
        logs={"loss": 2.5, "grad_norm": 1.0},
    )

    # Step 2: not a boundary yet, history not written
    callback.on_step_end(DummyArgs(), DummyState(2), DummyControl())
    assert not Path(callback.history_path).exists()

    # Step 3: boundary triggers recompute
    callback.on_step_end(DummyArgs(), DummyState(3), DummyControl())
    assert Path(callback.history_path).is_file()

    raw_rows = [
        json.loads(line)
        for line in Path(callback.history_path).read_text().splitlines()
        if line
    ]
    rows = parse_history_jsonl(raw_rows)
    assert len(rows) == 1
    assert rows[0]["step"] == 3
    weights = rows[0]["weights"]
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-4

    # Weights must be non-uniform
    assert weights[0] != weights[1]
    assert weights[1] != weights[2]

    # The hardest bucket (bucket 2 with highest loss and grad norm) has the largest weight
    assert weights[2] == max(weights)
    assert weights[2] > weights[1] > weights[0]

    # Verify current_weights matches latest history row
    assert callback.current_weights == weights

    # Verify render_curve handles the output
    rendered = render_curve(rows, num_buckets=3)
    assert "step" in rendered
    assert "B0" in rendered
    assert "B1" in rendered
    assert "B2" in rendered


def test_curriculum_curve_cli_with_callback_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """soup runs curriculum-curve renders history end-to-end from DynamicCurriculumCallback."""
    import soup_cli.experiment.tracker as tracker_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")

    # Generate genuine callback history via helper
    callback = _setup_trained_callback(tmp_path)
    history_file = Path(callback.history_path)
    assert history_file.is_file()

    class DummyTracker:
        def get_run(self, run_id: str):
            return {"run_id": run_id, "output_dir": str(tmp_path)}

    monkeypatch.setattr(tracker_mod, "ExperimentTracker", DummyTracker)

    result = runner.invoke(
        soup_app,
        ["runs", "curriculum-curve", "run-123", "--history", str(history_file)],
    )
    plain = strip_ansi(result.output)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "B0" in plain
    assert "B1" in plain
    assert "B2" in plain
