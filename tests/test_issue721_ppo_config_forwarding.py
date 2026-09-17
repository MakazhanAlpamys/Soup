"""Regression coverage for PPOConfig schedule forwarding (#721)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.trainer._trl_compat import kl_penalty_kwargs
from soup_cli.trainer.ppo import _effective_ppo_setting, _set_ppo_training_kwargs


def _config_accepting(*fields: str) -> type:
    """A config class whose constructor signature accepts exactly ``fields``."""

    class Config:
        def __init__(self, **kwargs: object) -> None:
            pass

    Config.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None) for name in fields]
    )
    return Config


@pytest.fixture
def training_config() -> SimpleNamespace:
    return SimpleNamespace(epochs=7, ppo_epochs=2, ppo_kl_penalty=0.7)


def test_current_trl_parameter_names_receive_non_defaults(training_config) -> None:
    kwargs: dict[str, object] = {}
    fields = _set_ppo_training_kwargs(
        kwargs,
        _config_accepting("num_train_epochs", "num_ppo_epochs", "kl_coef"),
        training_config,
    )

    assert kwargs == {
        "num_train_epochs": 7,
        "num_ppo_epochs": 2,
        "kl_coef": 0.7,
    }
    assert fields == {
        "train_epochs": "num_train_epochs",
        "ppo_epochs": "num_ppo_epochs",
        "kl_coef": "kl_coef",
    }


def test_legacy_trl_parameter_names_remain_supported(training_config) -> None:
    kwargs: dict[str, object] = {}
    fields = _set_ppo_training_kwargs(
        kwargs,
        _config_accepting("ppo_epochs", "init_kl_coef"),
        training_config,
    )

    assert kwargs == {"ppo_epochs": 2, "init_kl_coef": 0.7}
    assert fields == {
        "ppo_epochs": "ppo_epochs",
        "kl_coef": "init_kl_coef",
    }


def test_current_names_win_when_a_signature_carries_both(training_config) -> None:
    # Gap spotted by @dchaudhari7177 in #741: the two tests above each present
    # one spelling, so preferring the legacy name would pass both of them.
    kwargs: dict[str, object] = {}
    fields = _set_ppo_training_kwargs(
        kwargs,
        _config_accepting(
            "num_train_epochs", "num_ppo_epochs", "ppo_epochs", "kl_coef", "init_kl_coef"
        ),
        training_config,
    )

    assert kwargs == {
        "num_train_epochs": 7,
        "num_ppo_epochs": 2,
        "kl_coef": 0.7,
    }
    assert fields == {
        "train_epochs": "num_train_epochs",
        "ppo_epochs": "num_ppo_epochs",
        "kl_coef": "kl_coef",
    }


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (("kl_coef",), {"kl_coef": 0.7}),
        (("init_kl_coef",), {"init_kl_coef": 0.7}),
        (("init_kl_coef", "kl_coef"), {"kl_coef": 0.7}),
        (("cliprange",), {}),
    ],
)
def test_kl_penalty_kwargs_follows_the_rename(fields, expected) -> None:
    assert kl_penalty_kwargs(_config_accepting(*fields), 0.7) == expected


def test_a_config_without_a_kl_field_leaves_it_unforwarded(training_config) -> None:
    kwargs: dict[str, object] = {}
    fields = _set_ppo_training_kwargs(
        kwargs, _config_accepting("num_train_epochs", "num_ppo_epochs"), training_config
    )

    assert "kl_coef" not in fields
    assert kwargs == {"num_train_epochs": 7, "num_ppo_epochs": 2}


def test_an_unforwarded_setting_is_marked_rather_than_echoed() -> None:
    # When the installed trl exposes no field for a setting, the summary must
    # not print the requested value as if it had reached the trainer.
    shown = _effective_ppo_setting(SimpleNamespace(), {}, None, 7)

    assert shown == "7 (not forwarded)"


def test_a_forwarded_setting_reports_the_constructed_value() -> None:
    config = SimpleNamespace(num_train_epochs=3)

    assert _effective_ppo_setting(config, {"num_train_epochs": 7}, "num_train_epochs", 7) == 3


def test_installed_trl_exposes_and_receives_current_names(training_config) -> None:
    pytest.importorskip("trl.experimental.ppo")
    from trl.experimental.ppo import PPOConfig

    kwargs: dict[str, object] = {}
    _set_ppo_training_kwargs(kwargs, PPOConfig, training_config)

    assert kwargs["num_train_epochs"] == 7
    assert kwargs["num_ppo_epochs"] == 2
    assert kwargs["kl_coef"] == 0.7


def test_setup_passes_current_schedule_to_trainer(tmp_path, capsys) -> None:
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.ppo import PPOTrainerWrapper

    class CurrentPPOConfig:
        def __init__(
            self,
            output_dir,
            per_device_train_batch_size,
            gradient_accumulation_steps,
            learning_rate,
            seed,
            data_seed,
            num_train_epochs=3.0,
            num_ppo_epochs=4,
            kl_coef=0.05,
            cliprange=0.2,
            use_cpu=False,
        ):
            self.num_train_epochs = num_train_epochs
            self.num_ppo_epochs = num_ppo_epochs
            self.kl_coef = kl_coef

    class CurrentPPOTrainer:
        def __init__(self, *, args, **kwargs):
            self.args = args

        def add_callback(self, callback) -> None:
            pass

    config = SoupConfig(
        base="test-model",
        task="ppo",
        output=str(tmp_path),
        data={"train": "./data.jsonl"},
        training={
            "epochs": 7,
            "ppo_epochs": 2,
            "ppo_kl_penalty": 0.7,
            "reward_model": "test-reward-model",
        },
    )
    wrapper = PPOTrainerWrapper(config, device="cpu")
    wrapper.model = MagicMock()
    wrapper.model.get_nb_trainable_parameters.return_value = (100, 1000)
    wrapper.tokenizer = MagicMock()
    wrapper.tokenizer.side_effect = lambda texts, **kwargs: {
        "input_ids": [[1, 2]] * len(texts),
        "attention_mask": [[1, 1]] * len(texts),
    }

    with (
        patch.object(wrapper, "_setup_reward"),
        patch.object(wrapper, "_setup_transformers"),
        patch(
            "soup_cli.trainer.ppo._import_ppo_classes",
            return_value=(CurrentPPOTrainer, CurrentPPOConfig, True),
        ),
        patch.object(wrapper, "_get_or_create_reward_model", return_value=MagicMock()),
        patch.object(wrapper, "_create_value_model", return_value=MagicMock()),
    ):
        wrapper.setup({"train": [{"prompt": "Q?", "answer": "A"}]})

    assert wrapper.trainer.args.num_train_epochs == 7
    assert wrapper.trainer.args.num_ppo_epochs == 2
    assert wrapper.trainer.args.kl_coef == 0.7
    schedule = capsys.readouterr().out
    assert "train epochs=7" in schedule
    assert "PPO epochs=2" in schedule
    assert "KL coefficient=0.7" in schedule
