from pathlib import Path

from soup_cli.commands.train import _format_training_complete_loss
from soup_cli.trainer.loss_summary import summarize_training_loss


def test_short_run_uses_train_mean_without_a_fake_delta():
    summary = summarize_training_loss([{"train_loss": 2.383, "epoch": 1.0}])

    assert summary == {
        "initial_loss": 2.383,
        "final_loss": 2.383,
        "loss_summary_kind": "mean",
    }
    rendered = _format_training_complete_loss(summary)
    assert rendered == "Loss: [bold]2.3830[/]"
    assert "->" not in rendered


def test_normal_run_keeps_the_measured_per_step_delta():
    summary = summarize_training_loss(
        [{"loss": 3.2}, {"loss": 2.1}, {"loss": 1.1}, {"train_loss": 2.0}]
    )

    assert summary == {
        "initial_loss": 3.2,
        "final_loss": 1.1,
        "loss_summary_kind": "delta",
    }
    assert _format_training_complete_loss(summary) == "Loss: [bold]3.2000 -> 1.1000[/]"


def test_one_per_step_measurement_is_single_value_not_a_delta():
    summary = summarize_training_loss([{"loss": 1.75}, {"train_loss": 1.5}])

    assert summary["loss_summary_kind"] == "single"
    assert _format_training_complete_loss(summary) == "Loss: [bold]1.7500[/]"


def test_missing_loss_data_does_not_invent_a_zero_delta():
    summary = summarize_training_loss([{"train_runtime": 4.0}])

    assert summary["loss_summary_kind"] == "unavailable"
    assert _format_training_complete_loss(summary) == "Loss: [bold]unavailable[/]"


def test_final_metrics_can_supply_the_run_mean():
    summary = summarize_training_loss([], {"train_loss": "0.625"})

    assert summary["final_loss"] == 0.625
    assert summary["loss_summary_kind"] == "mean"
    assert "->" not in _format_training_complete_loss(summary)


def test_trainers_do_not_keep_private_loss_history_extractors():
    trainer_dir = Path(__file__).parents[1] / "src" / "soup_cli" / "trainer"
    copied_pattern = 'train_losses = [entry["loss"] for entry in logs if "loss" in entry]'
    offenders = [
        path.name
        for path in trainer_dir.glob("*.py")
        if copied_pattern in path.read_text()
    ]

    assert offenders == []
