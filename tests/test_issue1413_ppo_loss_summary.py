import ast
from pathlib import Path
from unittest.mock import MagicMock

from soup_cli.trainer.loss_summary import summarize_training_loss


def _ppo_history():
    """Entries shaped like trl 0.29 PPOTrainer's ``state.log_history``.

    Per-step losses live under ``loss/policy_avg``; there is no ``loss``
    key and no final ``train_loss`` mean, which is exactly why a completed
    PPO run ended with "Loss: unavailable" (#1413).
    """
    return [
        {
            "loss/policy_avg": 1.42,
            "loss/policy": 1.42,
            "loss/kl": 0.013,
            "loss/clipfrac": 0.02,
            "objective/kl": 12.1,
            "ppo/learning_rate": 1e-05,
            "step": 10,
        },
        {
            "loss/policy_avg": 0.87,
            "loss/policy": 0.87,
            "loss/kl": 0.009,
            "loss/clipfrac": 0.01,
            "objective/kl": 9.4,
            "ppo/learning_rate": 1e-05,
            "step": 20,
        },
    ]


def test_ppo_policy_avg_history_yields_a_delta_not_unavailable():
    summary = summarize_training_loss(_ppo_history(), loss_key="loss/policy_avg")

    assert summary["initial_loss"] == 1.42
    assert summary["final_loss"] == 0.87
    assert summary["loss_summary_kind"] == "delta"
    assert summary["loss_key"] == "loss/policy_avg"


def test_default_key_still_reports_unavailable_for_ppo_history():
    # Without the declared key the PPO-shaped history has no usable loss,
    # which is the bug: the key must come from the trainer, not be guessed.
    summary = summarize_training_loss(_ppo_history())

    assert summary["loss_summary_kind"] == "unavailable"


def test_single_ppo_step_is_tagged_with_its_key():
    summary = summarize_training_loss([_ppo_history()[0]], loss_key="loss/policy_avg")

    assert summary["loss_summary_kind"] == "single"
    assert summary["initial_loss"] == summary["final_loss"] == 1.42
    assert summary["loss_key"] == "loss/policy_avg"


def test_default_key_is_not_recorded_for_sft_style_summaries():
    summary = summarize_training_loss([{"loss": 2.0}, {"loss": 1.0}])

    assert summary["loss_summary_kind"] == "delta"
    assert "loss_key" not in summary


def test_ppo_trainer_declares_the_policy_avg_key():
    # Guard: the trl-path call in ppo.py must keep declaring the key instead
    # of the summary guessing it (the manual loop normalizes to "loss").
    source = (Path(__file__).parents[1] / "src" / "soup_cli" / "trainer" / "ppo.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    declaring_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "summarize_training_loss"
        and any(
            kw.arg == "loss_key"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == "loss/policy_avg"
            for kw in node.keywords
        )
    ]
    assert len(declaring_calls) == 1


def _trl_029_ppo_entry(step, policy_loss):
    """One ``state.log_history`` entry exactly as trl 0.29.1's PPOTrainer writes it.

    Keys read from ``ppo_trainer.py:883-901``: the per-step policy loss lives
    under ``loss/policy_avg`` and there is no ``loss`` key.
    """
    return {
        "eps": 3,
        "objective/kl": 9.4,
        "objective/entropy": 41.0,
        "objective/non_score_reward": -0.5,
        "objective/rlhf_reward": 0.2,
        "objective/scores": 0.7,
        "policy/approxkl_avg": 0.001,
        "policy/clipfrac_avg": 0.0,
        "loss/policy_avg": policy_loss,
        "loss/value_avg": 0.3,
        "val/clipfrac_avg": 0.0,
        "policy/entropy_avg": 1.9,
        "val/ratio": 1.0,
        "val/ratio_var": 0.0,
        "val/num_eos_tokens": 2,
        "lr": 1e-05,
        "episode": step * 4,
        "epoch": 0.1,
        "step": step,
    }


def test_train_builtin_reports_the_ppo_policy_loss_not_unavailable():
    """The trl path must feed its real ``log_history`` under the declared key.

    The AST guard above counts a declaring call anywhere in ``ppo.py``; this
    drives ``_train_builtin`` on a mock trl trainer so a wiring slip (key on
    the wrong call, wrong history fed to the summary) fails here instead.
    Passes on the fix, fails on ``main`` (``assert 'unavailable' == 'delta'``).
    """
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.ppo import PPOTrainerWrapper

    cfg = SoupConfig(base="test-model", task="ppo", data={"train": "./data.jsonl"})
    wrapper = PPOTrainerWrapper(cfg, device="cpu")
    wrapper._output_dir = "/tmp/test"
    wrapper._dataset_in_constructor = True
    wrapper._train_ds = MagicMock()

    trainer = MagicMock()
    # a real callable with no params, like trl's PPOTrainer.train
    trainer.train = lambda: None
    trainer.state.log_history = [_trl_029_ppo_entry(1, 1.42), _trl_029_ppo_entry(2, 0.87)]
    trainer.state.global_step = 2
    wrapper.trainer = trainer
    wrapper.tokenizer = MagicMock()

    result = wrapper._train_builtin(
        display=None, tracker=None, run_id="", resume_from_checkpoint=None
    )

    assert result["loss_summary_kind"] == "delta"
    assert result["initial_loss"] == 1.42
    assert result["final_loss"] == 0.87
    assert result["loss_key"] == "loss/policy_avg"


def test_ppo_panel_names_the_declared_loss_key():
    """The completion panel says which quantity the number is.

    A PPO policy loss is not comparable to an SFT cross-entropy, so the panel
    labels the value instead of printing a bare ``Loss: a -> b``.
    """
    from soup_cli.commands.train import _format_training_complete_loss

    rendered = _format_training_complete_loss(
        {
            "initial_loss": 1.42,
            "final_loss": 0.87,
            "loss_summary_kind": "delta",
            "loss_key": "loss/policy_avg",
        }
    )
    assert rendered == "Loss: [bold]1.4200 -> 0.8700[/] [dim](loss/policy_avg)[/]"


def test_ppo_panel_labels_the_single_step_value_too():
    from soup_cli.commands.train import _format_training_complete_loss

    rendered = _format_training_complete_loss(
        {
            "initial_loss": 1.42,
            "final_loss": 1.42,
            "loss_summary_kind": "single",
            "loss_key": "loss/policy_avg",
        }
    )
    assert rendered == "Loss: [bold]1.4200[/] [dim](loss/policy_avg)[/]"


def test_unavailable_panel_stays_bare():
    from soup_cli.commands.train import _format_training_complete_loss

    rendered = _format_training_complete_loss(
        {
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "loss_summary_kind": "unavailable",
            "loss_key": "loss/policy_avg",
        }
    )
    assert rendered == "Loss: [bold]unavailable[/]"
