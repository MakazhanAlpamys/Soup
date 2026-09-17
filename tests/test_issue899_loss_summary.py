import ast
import math
import sys
import types
from pathlib import Path

from soup_cli.commands.train import _format_training_complete_loss
from soup_cli.trainer.loss_summary import summarize_training_loss

_MIGRATED_TRAINERS = (
    "asr.py",
    "bco.py",
    "classifier.py",
    "distill.py",
    "dpo.py",
    "embedding.py",
    "grpo.py",
    "ipo.py",
    "kto.py",
    "mlx_sft.py",
    "mole_routing.py",
    "online_dpo.py",
    "orpo.py",
    "ppo.py",
    "pretrain.py",
    "prm.py",
    "reward_model.py",
    "sft.py",
    "simpo.py",
    "unlearn.py",
)


def _is_string_key(node: ast.expr | None, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _is_loss_summary_name(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "loss_summary"


def _is_loss_summary_initial_subscript(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and _is_loss_summary_name(node.value)
        and _is_string_key(node.slice, "initial_loss")
    )


def _dict_uses_shared_loss_summary(node: ast.Dict) -> bool:
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None and _is_loss_summary_name(value):
            return True
        if _is_string_key(key, "initial_loss") and _is_loss_summary_initial_subscript(value):
            return True
    return False


def _private_loss_result_lines(source: str) -> list[int]:
    tree = ast.parse(source)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(
            _is_string_key(key, "initial_loss")
            and not _is_loss_summary_initial_subscript(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    ]


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


def test_nan_tail_is_not_replaced_by_the_last_finite_loss():
    summary = summarize_training_loss([{"loss": 2.1}, {"loss": math.nan}])

    assert summary["initial_loss"] == 2.1
    assert math.isnan(summary["final_loss"])
    assert not summary["final_loss"] < float("inf")
    assert summary["loss_summary_kind"] == "delta"
    assert (
        _format_training_complete_loss(summary) == "Loss: [bold]2.1000 -> nan[/]"
    )


def test_infinite_tail_is_not_replaced_by_the_last_finite_loss():
    summary = summarize_training_loss([{"loss": 2.1}, {"loss": math.inf}])

    assert summary["initial_loss"] == 2.1
    assert math.isinf(summary["final_loss"])
    assert summary["loss_summary_kind"] == "delta"
    assert (
        _format_training_complete_loss(summary) == "Loss: [bold]2.1000 -> inf[/]"
    )


def test_all_nan_per_step_losses_are_not_reported_as_unavailable():
    summary = summarize_training_loss([{"loss": math.nan}, {"loss": math.nan}])

    assert math.isnan(summary["initial_loss"])
    assert math.isnan(summary["final_loss"])
    assert summary["loss_summary_kind"] == "delta"
    assert _format_training_complete_loss(summary) == "Loss: [bold]nan -> nan[/]"


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


def test_non_finite_train_mean_falls_through_to_unavailable():
    summary = summarize_training_loss([{"train_loss": math.nan}], {"train_loss": math.inf})

    assert summary["loss_summary_kind"] == "unavailable"
    assert summary["initial_loss"] == 0.0
    assert summary["final_loss"] == 0.0


def test_legacy_equal_loss_without_summary_kind_does_not_claim_a_delta():
    rendered = _format_training_complete_loss({"initial_loss": 2.383, "final_loss": 2.383})

    assert rendered == "Loss: [bold]2.3830[/]"
    assert "->" not in rendered


def test_trainers_do_not_keep_private_loss_history_extractors():
    trainer_dir = Path(__file__).parents[1] / "src" / "soup_cli" / "trainer"
    copied_pattern = 'train_losses = [entry["loss"] for entry in logs if "loss" in entry]'
    literal_offenders = [
        path.name
        for path in (trainer_dir / name for name in _MIGRATED_TRAINERS)
        if copied_pattern in path.read_text(encoding="utf-8")
    ]
    assert literal_offenders == []

    ast_offenders = []
    for name in _MIGRATED_TRAINERS:
        path = trainer_dir / name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert any(
            isinstance(node, ast.Dict) and _dict_uses_shared_loss_summary(node)
            for node in ast.walk(tree)
        ), f"{name} must source its training result from loss_summary"
        ast_offenders.extend((name, line) for line in _private_loss_result_lines(source))

    assert ast_offenders == []


def test_private_loss_history_guard_is_spelling_independent():
    alternate_spelling = """\
def result(logs):
    return {
        'initial_loss': ([entry['loss'] for entry in logs if 'loss' in entry] or [0])[0],
        'final_loss': ([entry['loss'] for entry in logs if 'loss' in entry] or [0])[-1],
    }
"""
    shared_subscript = """\
def result(loss_summary):
    return {
        'initial_loss': loss_summary['initial_loss'],
        'final_loss': loss_summary['final_loss'],
}
"""
    overrides_shared_spread = """\
def result(logs, loss_summary):
    return {
        **loss_summary,
        'initial_loss': ([entry['loss'] for entry in logs if 'loss' in entry] or [0])[0],
    }
"""

    assert _private_loss_result_lines(alternate_spelling) == [2]
    assert _private_loss_result_lines(shared_subscript) == []
    assert _private_loss_result_lines(overrides_shared_spread) == [2]


class _StubOptimizer:
    def zero_grad(self, *, set_to_none=False):
        pass

    def step(self):
        pass


class _StubSavedObject:
    def save_pretrained(self, output_dir):
        pass


class _StubLoss:
    def __init__(self, value):
        self.value = value

    def backward(self):
        pass

    def item(self):
        return self.value


def _run_unlearn_loss_result(tmp_path, monkeypatch, losses):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

    monkeypatch.chdir(tmp_path)
    cfg = SoupConfig(
        base="test-model",
        task="unlearn",
        data={"train": "train.jsonl", "forget_set": "forget.jsonl"},
        training={"unlearn_method": "npo", "epochs": 1},
        output="out",
    )
    wrapper = UnlearnTrainerWrapper(cfg)
    wrapper._setup_called = True
    wrapper.model = _StubSavedObject()
    wrapper.tokenizer = _StubSavedObject()
    wrapper._dev = "cpu"
    wrapper._optimizer = _StubOptimizer()
    wrapper._forget = [("prompt", "target")] * max(1, len(losses))
    wrapper._retain = []
    remaining = iter(losses)
    monkeypatch.setattr(
        wrapper,
        "_step_preference",
        lambda *args, **kwargs: _StubLoss(next(remaining)) if losses else None,
    )
    return wrapper.train()


def test_unlearn_real_train_reports_unavailable_without_measured_loss(tmp_path, monkeypatch):
    result = _run_unlearn_loss_result(tmp_path, monkeypatch, [])

    assert result["loss_summary_kind"] == "unavailable"
    assert result["initial_loss"] == result["final_loss"] == 0.0


def test_unlearn_real_train_reports_single_for_one_step(tmp_path, monkeypatch):
    result = _run_unlearn_loss_result(tmp_path, monkeypatch, [2.1])

    assert result["loss_summary_kind"] == "single"
    assert result["initial_loss"] == result["final_loss"] == 2.1


def test_unlearn_real_train_reports_delta_for_multiple_steps(tmp_path, monkeypatch):
    result = _run_unlearn_loss_result(tmp_path, monkeypatch, [2.1, 1.4, 0.9])

    assert result["loss_summary_kind"] == "delta"
    assert result["initial_loss"] == 2.1
    assert result["final_loss"] == 0.9


def _run_mlx_loss_result(tmp_path, monkeypatch, losses):
    from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx

    _install_fake_mlx(monkeypatch)

    def _train_with_losses(**kwargs):
        callback = kwargs["training_callback"]
        for loss in losses:
            callback.on_train_loss_report({"train_loss": loss})

    monkeypatch.setattr(sys.modules["mlx_lm.tuner.trainer"], "train", _train_with_losses)

    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    cfg = SoupConfig(
        base="mlx-community/Llama-3.1-8B-Instruct-4bit",
        task="sft",
        backend="mlx",
        data=DataConfig(
            train="./data/train.jsonl",
            format="chatml",
            train_on_responses_only=False,
        ),
        training=TrainingConfig(epochs=1, batch_size=1),
        output=str(tmp_path / "mlx-out"),
    )
    wrapper = MLXSFTTrainerWrapper(cfg)
    wrapper.model = _FakeMlxModel()
    wrapper.tokenizer = object()
    wrapper._dataset = {"train": [{"text": "hello"}], "val": []}
    return wrapper.train()


def test_mlx_real_result_path_reports_unavailable_without_measured_loss(tmp_path, monkeypatch):
    result = _run_mlx_loss_result(tmp_path, monkeypatch, [])

    assert result["loss_summary_kind"] == "unavailable"
    assert result["initial_loss"] == result["final_loss"] == 0.0


def test_mlx_real_result_path_reports_single_for_one_loss(tmp_path, monkeypatch):
    result = _run_mlx_loss_result(tmp_path, monkeypatch, [1.75])

    assert result["loss_summary_kind"] == "single"
    assert result["initial_loss"] == result["final_loss"] == 1.75


def test_mlx_real_result_path_reports_delta_for_multiple_losses(tmp_path, monkeypatch):
    result = _run_mlx_loss_result(tmp_path, monkeypatch, [2.1, 0.9])

    assert result["loss_summary_kind"] == "delta"
    assert result["initial_loss"] == 2.1
    assert result["final_loss"] == 0.9
