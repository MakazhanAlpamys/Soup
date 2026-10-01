"""#1497 — ``freeze_layers`` / ``freeze_ratio`` are refused on the tasks that ignore them.

Both fields loaded on every task, but ``freeze_model_layers`` has one call site,
``SFTTrainerWrapper._setup_transformers``, which ``tts`` inherits. On every other
task the run trained at full depth with no message. They are now refused at load,
naming the field, the task and the fix; ``sft`` and ``tts`` are unchanged.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import typing

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import FREEZE_LAYER_TASKS, SoupConfig
from tests.test_issue1357_unsloth_refused_without_setup import _yaml

_TASKS = sorted(typing.get_args(SoupConfig.model_fields["task"].annotation))
_REFUSED = [task for task in _TASKS if task not in FREEZE_LAYER_TASKS]
_FIELDS = {"freeze_layers": 2, "freeze_ratio": 0.5}

# The wrapper each task builds (trainer/dispatch.py), as in #1264's ratchet.
_WRAPPERS = {
    "sft": "sft:SFTTrainerWrapper", "dpo": "dpo:DPOTrainerWrapper",
    "grpo": "grpo:GRPOTrainerWrapper", "ppo": "ppo:PPOTrainerWrapper",
    "reward_model": "reward_model:RewardModelTrainerWrapper",
    "kto": "kto:KTOTrainerWrapper", "orpo": "orpo:ORPOTrainerWrapper",
    "simpo": "simpo:SimPOTrainerWrapper", "ipo": "ipo:IPOTrainerWrapper",
    "bco": "bco:BCOTrainerWrapper", "preference": "preference:PreferenceTrainerWrapper",
    "pretrain": "pretrain:PretrainTrainerWrapper",
    "embedding": "embedding:EmbeddingTrainerWrapper", "prm": "prm:PRMTrainerWrapper",
    "tts": "tts:TTSTrainerWrapper", "classifier": "classifier:ClassifierTrainerWrapper",
    "reranker": "classifier:ClassifierTrainerWrapper",
    "cross_encoder": "classifier:ClassifierTrainerWrapper",
    "distill": "distill:DistillTrainerWrapper", "unlearn": "unlearn:UnlearnTrainerWrapper",
    "moe_lora_routing": "mole_routing:MoleRoutingTrainerWrapper",
    "online_dpo": "online_dpo:OnlineDPOTrainerWrapper", "asr": "asr:AsrTrainerWrapper",
}
# preference builds one of these per preference_loss (trainer/preference.py).
_PREFERENCE_DELEGATES = ("dpo", "simpo", "orpo", "ipo", "bco")


def _refusal(text):
    with pytest.raises(ValueError) as info:
        load_config_from_string(text)
    return " ".join(str(info.value).split())


class TestRefusedWhereNotApplied:
    @pytest.mark.parametrize("field", sorted(_FIELDS))
    @pytest.mark.parametrize("task", _REFUSED)
    def test_names_the_field_the_task_and_the_fix(self, task, field):
        value = _FIELDS[field]
        message = _refusal(_yaml(task, "transformers", **{field: value}))

        assert f"training.{field}={value!r} is not applied by task={task!r}" in message
        assert "Use task: sft" in message
        assert f"remove training.{field}" in message

    @pytest.mark.parametrize("task", _REFUSED)
    def test_without_the_fields_the_task_still_loads(self, task):
        assert load_config_from_string(_yaml(task, "transformers")).task == task

    @pytest.mark.parametrize("field", sorted(_FIELDS))
    def test_the_advice_works(self, field):
        """``task: sft`` with the same field loads."""
        cfg = load_config_from_string(_yaml("sft", "transformers", **{field: _FIELDS[field]}))

        assert getattr(cfg.training, field) == _FIELDS[field]


class TestAppliedTasksAreUnchanged:
    @pytest.mark.parametrize("field", sorted(_FIELDS))
    @pytest.mark.parametrize("task", sorted(FREEZE_LAYER_TASKS))
    def test_still_loads(self, task, field):
        cfg = load_config_from_string(_yaml(task, "transformers", **{field: _FIELDS[field]}))

        assert getattr(cfg.training, field) == _FIELDS[field]

    def test_the_applied_set_is_sft_and_tts(self):
        assert FREEZE_LAYER_TASKS == {"sft", "tts"}


class TestTheListFollowsTheTrainers:
    """A wrapper that starts or stops calling ``freeze_model_layers`` must move in
    ``FREEZE_LAYER_TASKS``, or the fields are accepted and not applied again."""

    def test_every_task_is_classified(self):
        assert set(_TASKS) == set(_WRAPPERS)

    @staticmethod
    def _applies(task):
        if task == "preference":
            return all(TestTheListFollowsTheTrainers._applies(t) for t in _PREFERENCE_DELEGATES)
        module, name = _WRAPPERS[task].split(":")
        wrapper = getattr(importlib.import_module(f"soup_cli.trainer.{module}"), name)
        return any(
            "freeze_model_layers(" in inspect.getsource(cls)
            for cls in wrapper.__mro__
            if cls.__module__.startswith("soup_cli.trainer.")
        )

    @pytest.mark.parametrize("task", _TASKS)
    def test_the_schema_list_matches_the_wrapper(self, task):
        assert (task in FREEZE_LAYER_TASKS) == self._applies(task), task

    def test_the_only_caller_is_the_sft_wrapper(self):
        """A new call site outside trainer/sft.py has to be classified here first."""
        root = pathlib.Path(importlib.import_module("soup_cli").__file__).parent
        callers = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*.py")
            if "freeze_model_layers(" in path.read_text(encoding="utf-8")
            and path.name != "freeze.py"
        )

        assert callers == ["trainer/sft.py"], callers


class TestOrder:
    def test_it_holds_on_every_backend_before_the_unsloth_refusal(self):
        """reward_model has no unsloth setup: the freeze refusal still comes first."""
        on_unsloth = _refusal(_yaml("reward_model", "unsloth", freeze_layers=2))

        assert "no unsloth setup" not in on_unsloth, on_unsloth
        assert on_unsloth == _refusal(_yaml("reward_model", "transformers", freeze_layers=2))

    def test_the_resolver_and_the_unsloth_refusal_keep_their_places(self):
        validators = SoupConfig.__pydantic_decorators__.model_validators
        after = [name for name, dec in validators.items() if dec.info.mode == "after"]

        assert after[0] == "_resolve_quantization_for_unhonouring_tasks"
        assert after[-1] == "_validate_unsloth_has_a_setup"
        assert "_validate_freeze_reaches_a_trainer" in after[1:-1]


class TestSoupDoctorReportsIt:
    def test_doctor_exits_2_naming_the_field(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from tests.conftest import strip_ansi

        path = tmp_path / "soup.yaml"
        raw = yaml.safe_load(_yaml("dpo", "transformers", freeze_layers=2))
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")

        result = CliRunner().invoke(app, ["doctor", "--config", str(path)])

        assert result.exit_code == 2, result.output
        out = " ".join(strip_ansi(result.output).split())
        assert "training.freeze_layers=2 is not applied by task='dpo'" in out
