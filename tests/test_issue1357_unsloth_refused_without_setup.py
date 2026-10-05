"""#1357 — ``backend: unsloth`` is refused at load on the tasks with no unsloth setup.

``reward_model``, ``prm``, ``classifier`` / ``reranker`` / ``cross_encoder``,
``unlearn`` and ``moe_lora_routing`` loaded with ``backend: unsloth``, their
trainers never read it, and the run trained on plain transformers while
``soup train`` printed "unsloth (fast mode)". ``distill``, ``asr`` and
``online_dpo`` already refused it, each with its own reason. The split is
#1323's ``UNSLOTH_SETUP_TASKS``, ratcheted against the wrappers there.
"""

from __future__ import annotations

import typing

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import UNSLOTH_SETUP_TASKS, SoupConfig

_TOP = {"tts": {"modality": "audio_out"}}
_DATA = {
    "dpo": {"format": "dpo"},
    "kto": {"format": "kto"},
    "pretrain": {"format": "plaintext"},
    "unlearn": {"forget_set": "./f.jsonl"},
}
_TRAINING = {
    "preference": {"preference_loss": "dpo"},
    "tts": {"tts_family": "orpheus"},
    "classifier": {"num_labels": 2},
    "reranker": {"num_labels": 2},
    "cross_encoder": {"num_labels": 2},
    "unlearn": {"unlearn_method": "npo"},
    "moe_lora_routing": {"mole_task_adapters": ["./a", "./b"]},
    "distill": {"teacher_model": "org/t"},
    "online_dpo": {"online_dpo_judge": "pairrm"},
}
_NEWLY_REFUSED = (
    "reward_model",
    "prm",
    "classifier",
    "reranker",
    "cross_encoder",
    "unlearn",
    "moe_lora_routing",
)
# Refused on unsloth before this issue, each with its own reason; kept.
_ALREADY_REFUSED = {
    "distill": "task='distill' is not supported on backend=unsloth",
    "asr": "task='asr' requires backend='transformers'",
    "online_dpo": "task='online_dpo' requires backend='transformers'",
}


def _yaml(task, backend, **training):
    raw = {
        "base": "org/m",
        "task": task,
        "backend": backend,
        **_TOP.get(task, {}),
        "data": {"train": "./x.jsonl", **_DATA.get(task, {})},
        "training": {**_TRAINING.get(task, {}), **training},
    }
    return yaml.safe_dump(raw)


def _refusal(text):
    with pytest.raises(ValueError) as info:
        load_config_from_string(text)
    return " ".join(str(info.value).split())


class TestTheSevenAreRefused:
    @pytest.mark.parametrize("task", _NEWLY_REFUSED)
    def test_unsloth_is_refused_naming_the_task(self, task):
        message = _refusal(_yaml(task, "unsloth"))

        assert f"backend='unsloth' is not applied by task={task!r}" in message, message
        assert "Use backend: transformers" in message

    @pytest.mark.parametrize("task", _NEWLY_REFUSED)
    def test_the_transformers_control_still_loads(self, task):
        cfg = load_config_from_string(_yaml(task, "transformers"))

        assert cfg.backend == "transformers"


class TestTheOtherTasksAreUnchanged:
    @pytest.mark.parametrize("task", sorted(UNSLOTH_SETUP_TASKS))
    def test_a_task_with_an_unsloth_setup_still_loads_on_unsloth(self, task):
        cfg = load_config_from_string(_yaml(task, "unsloth"))

        assert cfg.backend == "unsloth"

    @pytest.mark.parametrize("task", sorted(_ALREADY_REFUSED))
    def test_the_existing_refusals_keep_their_own_reason(self, task):
        message = _refusal(_yaml(task, "unsloth"))

        assert _ALREADY_REFUSED[task] in message, message

    def test_every_task_is_covered(self):
        tasks = set(typing.get_args(SoupConfig.model_fields["task"].annotation))

        assert tasks == set(UNSLOTH_SETUP_TASKS) | set(_NEWLY_REFUSED) | set(_ALREADY_REFUSED)


class TestOneRefusalWithMoeLora:
    """#1323's moe_lora message already says unsloth is not applied on these tasks;
    with both set the user gets one refusal, and transformers then loads."""

    def test_moe_lora_on_unsloth_is_one_refusal(self):
        message = _refusal(_yaml("reward_model", "unsloth", moe_lora=True))

        assert message.count("no unsloth setup") == 1, message
        # #1323's wording, not this check's: only then is the order pinned here too.
        assert "training.moe_lora is refused with backend='unsloth'" in message, message
        load_config_from_string(_yaml("reward_model", "transformers", moe_lora=True))


class TestSoupDoctorReportsIt:
    def test_doctor_exits_2_instead_of_reviewing_zero_settings(self, tmp_path):
        """On ``main`` it exited 0 with "None of the 0 setting(s) known to be unread
        on task=reward_model backend=unsloth"."""
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from tests.conftest import strip_ansi

        path = tmp_path / "soup.yaml"
        path.write_text(_yaml("reward_model", "unsloth"), encoding="utf-8")

        result = CliRunner().invoke(app, ["doctor", "--config", str(path)])

        assert result.exit_code == 2, result.output
        out = " ".join(strip_ansi(result.output).split())
        assert "backend='unsloth' is not applied by task='reward_model'" in out


class TestUnslothBnb4bitNamesTheTask:
    """``unsloth_bnb_4bit`` was checked before this refusal, and on six of the seven
    tasks the #795 resolver has already turned ``quantization: 4bit`` into ``none``,
    so that check blamed a value the user never wrote."""

    @pytest.mark.parametrize("task", _NEWLY_REFUSED)
    def test_one_refusal_names_the_task_and_the_whole_way_out(self, task):
        message = _refusal(_yaml(task, "unsloth", quantization="4bit", unsloth_bnb_4bit=True))

        assert f"backend='unsloth' is not applied by task={task!r}" in message, message
        assert "Use backend: transformers" in message
        # the flag has to go too, or backend: transformers meets a second refusal
        assert "unsloth_bnb_4bit" in message

    @pytest.mark.parametrize("task", _NEWLY_REFUSED)
    def test_following_that_advice_loads(self, task):
        cfg = load_config_from_string(_yaml(task, "transformers", quantization="4bit"))

        assert cfg.backend == "transformers"


class TestTheStepAsideStaysNarrow:
    """The ``unsloth_bnb_4bit`` step-aside is for ``backend: unsloth`` only, the flag is
    named only when it is set, and refusals that hold on every backend still come first."""

    @pytest.mark.parametrize("task", _NEWLY_REFUSED)
    def test_transformers_still_refuses_unsloth_bnb_4bit(self, task):
        message = _refusal(_yaml(task, "transformers", unsloth_bnb_4bit=True))

        assert "unsloth_bnb_4bit=true requires backend='unsloth'" in message, message

    @pytest.mark.parametrize("task", _NEWLY_REFUSED)
    def test_the_plain_refusal_does_not_name_the_flag(self, task):
        message = _refusal(_yaml(task, "unsloth"))

        assert "unsloth_bnb_4bit" not in message, message

    @pytest.mark.parametrize(
        ("task", "section", "key"),
        [("unlearn", "data", "forget_set"), ("moe_lora_routing", "training", "mole_task_adapters")],
    )
    def test_a_refusal_on_every_backend_comes_first(self, task, section, key):
        def refusal(backend):
            raw = yaml.safe_load(_yaml(task, backend))
            del raw[section][key]
            return _refusal(yaml.safe_dump(raw))

        on_unsloth = refusal("unsloth")

        assert "no unsloth setup" not in on_unsloth, on_unsloth
        assert on_unsloth == refusal("transformers")

    def test_the_backend_refusal_is_the_last_after_validator(self):
        """Structural, for validators not yet written (#1313's ``eval_steps`` check is
        one): anything that refuses on every backend must run before this one.
        ``model_validators`` is in definition order."""
        validators = SoupConfig.__pydantic_decorators__.model_validators
        after = [name for name, dec in validators.items() if dec.info.mode == "after"]

        assert after[-1] == "_validate_unsloth_has_a_setup", after[-3:]
