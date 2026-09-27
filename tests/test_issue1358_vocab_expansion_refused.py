"""#1358 — ``data.add_new_tokens`` / ``data.new_special_tokens`` load only where a
trainer adds the tokens.

``apply_vocab_expansion`` is the one helper that adds them to the tokenizer and
resizes the embeddings. It runs only in transformers setups: SFT's text / vision /
audio paths and the DPO/GRPO family (``preference`` through the trainers it
delegates to, ``tts`` through SFT's text path). Everywhere else the fields loaded
and were dropped: every ``backend: unsloth`` setup, ``backend: mlx``, SFT's layer
streaming, and ten tasks whose trainers never call it. They are now refused at
load, naming what does not apply them.
"""

from __future__ import annotations

import ast
import importlib
import importlib.resources
import inspect
import textwrap
import typing

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig

_TOP = {"tts": {"modality": "audio_out"}}
_DATA = {
    "dpo": {"format": "dpo"}, "kto": {"format": "kto"}, "pretrain": {"format": "plaintext"},
    "unlearn": {"forget_set": "./f.jsonl"},
}
_TRAINING = {
    "preference": {"preference_loss": "dpo"},
    "tts": {"tts_family": "orpheus"},
    "classifier": {"num_labels": 2}, "reranker": {"num_labels": 2},
    "cross_encoder": {"num_labels": 2},
    "unlearn": {"unlearn_method": "npo"},
    "moe_lora_routing": {"mole_task_adapters": ["./a", "./b"]},
    "distill": {"teacher_model": "org/t"},
    "online_dpo": {"online_dpo_judge": "pairrm"},
}
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
_PREFERENCE_DELEGATES = ("dpo", "simpo", "orpo", "ipo", "bco")

_APPLIES = (
    "sft", "dpo", "kto", "orpo", "ipo", "simpo", "bco", "grpo", "online_dpo",
    "preference", "tts",
)
_NEVER = (
    "pretrain", "embedding", "ppo", "reward_model", "prm", "classifier", "reranker",
    "cross_encoder", "unlearn", "moe_lora_routing", "distill", "asr",
)
_FIELDS = ("add_new_tokens", "new_special_tokens")


def _yaml(task, backend="transformers", field="add_new_tokens", tokens=("<x>",), **extra):
    top = dict(_TOP.get(task, {}))
    data = {"train": "./x.jsonl", **_DATA.get(task, {}), field: list(tokens)}
    training = dict(_TRAINING.get(task, {}))
    training.update(extra.pop("training", {}))
    data.update(extra.pop("data", {}))
    raw = {"base": "org/m", "task": task, "backend": backend, **top, **extra,
           "data": data, "training": training}
    return yaml.safe_dump(raw)


def _refusal(text):
    with pytest.raises(ValueError) as info:
        load_config_from_string(text)
    return " ".join(str(info.value).split())


class TestWhereTheyApplyTheyStillLoad:
    @pytest.mark.parametrize("field", _FIELDS)
    @pytest.mark.parametrize("task", _APPLIES)
    def test_a_task_that_adds_the_tokens_loads(self, task, field):
        cfg = load_config_from_string(_yaml(task, field=field))

        assert getattr(cfg.data, field) == ["<x>"]

    @pytest.mark.parametrize("task", _NEVER + ("sft",))
    def test_an_empty_list_asks_for_nothing_and_loads_anywhere(self, task):
        load_config_from_string(_yaml(task, backend="transformers", tokens=()))


class TestGap2TasksThatNeverAddThem:
    @pytest.mark.parametrize("field", _FIELDS)
    @pytest.mark.parametrize("task", _NEVER)
    def test_refused_naming_the_task(self, task, field):
        message = _refusal(_yaml(task, field=field))

        assert f"data.{field} is not applied by task={task!r}" in message, message


class TestGap1Unsloth:
    @pytest.mark.parametrize("field", _FIELDS)
    @pytest.mark.parametrize("task", ["sft", "dpo", "tts"])
    def test_refused_naming_the_backend(self, task, field):
        message = _refusal(_yaml(task, backend="unsloth", field=field))

        assert f"data.{field} is not applied on backend='unsloth'" in message, message
        assert "Use backend: transformers" in message

    def test_a_task_that_never_adds_them_gets_its_task_refusal_on_both_backends(self):
        """The task refusal holds on every backend, so it comes first: switching to
        transformers must not surface a second, different refusal (#1323's lesson)."""
        on_unsloth = _refusal(_yaml("reward_model", backend="unsloth"))

        assert "not applied by task='reward_model'" in on_unsloth, on_unsloth
        assert on_unsloth == _refusal(_yaml("reward_model"))


class TestTheOtherPathsThatDropThem:
    """Found while mapping the call sites; not in the issue's list."""

    def test_mlx_sft(self):
        message = _refusal(_yaml("sft", backend="mlx", data={"format": "chatml"}))

        assert "data.add_new_tokens is not applied on backend='mlx'" in message, message

    def test_sft_layer_streaming(self):
        message = _refusal(_yaml(
            "sft",
            training={"stream_layers": True, "batch_size": 1, "quantization": "none",
                      "lora": {"r": 8}},
        ))

        assert "data.add_new_tokens is not applied with training.stream_layers" in message, message


# --- the ratchet: the schema's task list must follow the trainers ------------------


def _calls_helper(func) -> bool:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "apply_vocab_expansion"
        for node in ast.walk(tree)
    )


def _calls_super(func, name) -> bool:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return any(
        isinstance(node, ast.Call) and getattr(node.func, "attr", None) == name
        and isinstance(node.func.value, ast.Call)
        and getattr(node.func.value.func, "id", None) == "super"
        for node in ast.walk(tree)
    )


def _method_applies(cls, name) -> bool:
    """Whether ``cls.<name>`` reaches ``apply_vocab_expansion``, following
    ``super().<name>`` up the MRO (``tts`` extends SFT's text setup that way)."""
    for klass in cls.__mro__:
        func = klass.__dict__.get(name)
        if func is None:
            continue
        if _calls_helper(func):
            return True
        if not _calls_super(func, name):
            return False
    return False


def _wrapper(task):
    module, name = _WRAPPERS[task].split(":")
    return getattr(importlib.import_module(f"soup_cli.trainer.{module}"), name)


class TestTheSplitMatchesTheTrainers:
    def test_every_task_is_classified(self):
        tasks = set(typing.get_args(SoupConfig.model_fields["task"].annotation))

        assert tasks == set(_WRAPPERS) == set(_APPLIES) | set(_NEVER)

    @pytest.mark.parametrize("task", sorted(_WRAPPERS))
    def test_the_schema_list_matches_the_transformers_setup(self, task):
        pytest.importorskip("transformers")
        from soup_cli.config.schema import VOCAB_EXPANSION_TASKS

        if task == "preference":
            applies = all(
                _method_applies(_wrapper(t), "_setup_transformers") for t in _PREFERENCE_DELEGATES
            )
        else:
            applies = _method_applies(_wrapper(task), "_setup_transformers")

        assert (task in VOCAB_EXPANSION_TASKS) == applies

    @pytest.mark.parametrize("task", sorted(_WRAPPERS))
    def test_no_unsloth_setup_applies_them_yet(self, task):
        """If one is wired, this fails: lift that task's unsloth refusal."""
        pytest.importorskip("transformers")
        cls = _wrapper(task)

        assert not (hasattr(cls, "_setup_unsloth") and _method_applies(cls, "_setup_unsloth"))

    def test_neither_streaming_nor_mlx_applies_them_yet(self):
        pytest.importorskip("transformers")
        from soup_cli.trainer.sft import SFTTrainerWrapper

        assert not _method_applies(SFTTrainerWrapper, "_setup_streaming_transformers")
        for path in importlib.resources.files("soup_cli.trainer").iterdir():
            if path.name.startswith("mlx_") and path.name.endswith(".py"):
                assert "apply_vocab_expansion" not in path.read_text(encoding="utf-8"), path.name
