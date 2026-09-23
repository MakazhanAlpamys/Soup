from __future__ import annotations

import math
import sys
import types

import pytest

from soup_cli.config.schema import SoupConfig

_LAYA = {
    "checkpoint": "convaiinnovations/laya",
    "question": {
        "id": "urgency",
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical"],
    },
}


def _config(**training):
    return SoupConfig(
        base="model",
        task=training.pop("task", "grpo"),
        data={"train": "data.jsonl"},
        training={"laya_reward": _LAYA, **training},
    )


def test_laya_reward_parses_complete_score_config():
    config = _config()
    reward = config.training.laya_reward
    assert reward is not None
    assert reward.checkpoint == "convaiinnovations/laya"
    assert reward.question.id == "urgency"
    assert reward.question.type == "score"
    assert reward.question.criteria == ["not urgent", "soon", "critical"]


def test_laya_reward_parses_noul_without_criteria():
    config = _config(
        laya_reward={
            "checkpoint": "convaiinnovations/laya",
            "question": {
                "id": "risk",
                "type": "noul",
                "instructions": "How risky is this?",
            },
        }
    )
    assert config.training.laya_reward.question.criteria is None


def test_laya_reward_rejects_choice_and_invalid_score_criteria():
    with pytest.raises(ValueError, match="score.*noul|Input should be 'score' or 'noul'"):
        _config(
            laya_reward={
                "checkpoint": "model",
                "question": {"id": "x", "type": "choice", "instructions": "x"},
            }
        )
    with pytest.raises(ValueError, match="non-empty list of criteria"):
        _config(
            laya_reward={
                "checkpoint": "model",
                "question": {
                    "id": "x",
                    "type": "score",
                    "instructions": "x",
                    "criteria": [],
                },
            }
        )


@pytest.mark.parametrize("field", ["prm_reward", "reward_model"])
def test_laya_reward_rejects_other_reward_sources(field):
    with pytest.raises(ValueError, match="laya_reward.*(prm_reward|reward_model)"):
        _config(**{field: "other"})


def test_laya_reward_distinguishes_omitted_and_explicit_reward_fn():
    assert _config().training.reward_fn == "accuracy"
    with pytest.raises(ValueError, match="explicitly configured reward_fn"):
        _config(reward_fn="accuracy")
    assert _config(reward_fn=None).training.reward_fn is None


def test_laya_reward_rejects_unsupported_task():
    with pytest.raises(ValueError, match="requires task='grpo' or task='ppo'"):
        _config(task="sft")


def test_laya_reward_adapter_extracts_scores_and_normalizes_state(monkeypatch):
    calls = []

    class Agent:
        def predict(self, state, questions):
            calls.append((state, questions))
            return {"answers": {"urgency": {"score": 1.25}}}

    module = types.SimpleNamespace(load=lambda checkpoint, **kwargs: Agent())
    monkeypatch.setitem(sys.modules, "laya", module)
    from soup_cli.trainer.laya_reward import build_laya_reward_fn

    reward = build_laya_reward_fn(_config().training, "cuda:0")
    values = reward(
        [[{"role": "assistant", "content": "answer"}]],
        prompts=[[{"role": "user", "content": "question"}]],
    )
    assert values == [1.25]
    assert calls[0][0] == {"prompt": "question", "completion": "answer"}
    assert calls[0][1]["urgency"]["criteria"] == ["not urgent", "soon", "critical"]


def test_laya_reward_falls_back_to_completion_and_loads_once(monkeypatch):
    load_calls = []

    class Agent:
        def predict(self, state, questions):
            return {"answers": {"urgency": {"score": 2}}}

    def load(checkpoint, **kwargs):
        load_calls.append((checkpoint, kwargs))
        return Agent()

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=load))
    from soup_cli.trainer.laya_reward import build_laya_reward_fn

    reward = build_laya_reward_fn(_config().training, "cpu")
    assert reward(["one", "two"]) == [2.0, 2.0]
    assert len(load_calls) == 1
    assert load_calls[0][1]["device"] == "cpu"


def test_laya_reward_rejects_local_checkpoint_outside_cwd(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    outside = tmp_path / "outside-checkpoint"
    workdir.mkdir()
    outside.mkdir()
    monkeypatch.chdir(workdir)

    config = _config(
        laya_reward={
            "checkpoint": str(outside),
            "question": _LAYA["question"],
        }
    )
    from soup_cli.trainer.laya_reward import build_laya_reward_fn

    with pytest.raises(ValueError, match="checkpoint path must stay under"):
        build_laya_reward_fn(config.training, "cpu")


def test_laya_reward_passes_subfolder_to_loader(monkeypatch):
    load_calls = []

    class Agent:
        def predict(self, state, questions):
            return {"answers": {"urgency": {"score": 1}}}

    def load(checkpoint, **kwargs):
        load_calls.append((checkpoint, kwargs))
        return Agent()

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=load))
    from soup_cli.trainer.laya_reward import build_laya_reward_fn

    config = _config(
        laya_reward={
            "checkpoint": "convaiinnovations/laya",
            "subfolder": "typed-decisions",
            "question": _LAYA["question"],
        }
    )
    build_laya_reward_fn(config.training, "cuda:0")

    assert load_calls == [
        (
            "convaiinnovations/laya",
            {"device": "cuda:0", "subfolder": "typed-decisions"},
        )
    ]


def test_laya_reward_rejects_malformed_or_nonfinite_answers(monkeypatch):
    class Agent:
        def predict(self, state, questions):
            return {"answers": {"urgency": {"score": math.nan}}}

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=lambda *a, **k: Agent()))
    from soup_cli.trainer.laya_reward import build_laya_reward_fn

    reward = build_laya_reward_fn(_config().training, "cpu")
    with pytest.raises(ValueError, match="non-finite"):
        reward(["one"])


def test_grpo_selects_laya_reward(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "soup_cli.trainer.laya_reward.build_laya_reward_fn",
        lambda tcfg, device: sentinel,
    )
    from soup_cli.trainer.grpo import _select_reward_fn

    assert _select_reward_fn(_config().training, "cpu", False) is sentinel
