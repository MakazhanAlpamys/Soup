"""#1849 — the DPO row layout every bundled example uses reaches TRL in a shape it accepts.

``examples/data/dpo_sample.jsonl`` and ``chat_preferences.jsonl`` (and so
``dpo_example.yaml``, ``dpo_chat.yaml``, ``soup data demo dpo_demo`` and
``sharegpt_demo``) write ``prompt`` as a string and ``chosen`` / ``rejected``
as message lists that repeat the user turn. TRL 0.29 decides text vs.
conversational from ``prompt``, so it took the rows as text and its add-EOS
step called ``.endswith`` on a list:
``AttributeError: 'list' object has no attribute 'endswith'``.

``_convert_dpo`` now turns that layout into TRL's explicit conversational one:
the shared leading turns become the ``prompt`` message list and what follows
them the completions. The conversation is what trains, so it wins over the
redundant string.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from soup_cli.data.formats import _convert_dpo

REPO = pathlib.Path(__file__).resolve().parents[1]
U = {"role": "user", "content": "What is 2+2?"}
GOOD = {"role": "assistant", "content": "4"}
BAD = {"role": "assistant", "content": "5"}


class TestConvertDpo:
    def test_implicit_prompt_conversations_become_explicit(self):
        row = _convert_dpo({"prompt": "What is 2+2?", "chosen": [U, GOOD], "rejected": [U, BAD]})
        assert row == {"prompt": [U], "chosen": [GOOD], "rejected": [BAD]}

    def test_multi_turn_shared_history_is_all_prompt(self):
        a1 = {"role": "assistant", "content": "Hi"}
        u2 = {"role": "user", "content": "And 2+2?"}
        row = _convert_dpo(
            {"prompt": "And 2+2?", "chosen": [U, a1, u2, GOOD], "rejected": [U, a1, u2, BAD]}
        )
        assert row["prompt"] == [U, a1, u2]
        assert (row["chosen"], row["rejected"]) == ([GOOD], [BAD])

    def test_string_prompt_with_assistant_only_completions(self):
        row = _convert_dpo({"prompt": "What is 2+2?", "chosen": [GOOD], "rejected": [BAD]})
        assert row == {"prompt": [U], "chosen": [GOOD], "rejected": [BAD]}

    def test_a_tie_keeps_its_last_turn_as_the_completion(self):
        """chosen == rejected is real data (UltraFeedback has ties). TRL trains a
        tie at a constant loss; an empty completion would not tokenize at all."""
        row = _convert_dpo({"prompt": "q", "chosen": [U, GOOD], "rejected": [U, GOOD]})
        assert row == {"prompt": [U], "chosen": [GOOD], "rejected": [GOOD]}

    def test_text_rows_are_untouched(self):
        row = {"prompt": "Q:", "chosen": " yes", "rejected": " no"}
        assert _convert_dpo(dict(row)) == row

    def test_explicit_conversational_rows_are_untouched(self):
        row = {"prompt": [U], "chosen": [GOOD], "rejected": [BAD]}
        assert _convert_dpo(dict(row)) == row

    def test_mixed_completion_types_are_refused(self):
        with pytest.raises(TypeError, match="chosen and rejected"):
            _convert_dpo({"prompt": "q", "chosen": [U, GOOD], "rejected": "5"})

    def test_null_still_refused(self):
        with pytest.raises(TypeError, match="dpo.rejected must not be null"):
            _convert_dpo({"prompt": "q", "chosen": "a", "rejected": None})


@pytest.mark.parametrize("sample", ["dpo_sample.jsonl", "chat_preferences.jsonl"])
def test_every_bundled_row_converts_to_the_explicit_layout(sample):
    for line in (REPO / "examples" / "data" / sample).read_text(encoding="utf-8").splitlines():
        row = _convert_dpo(json.loads(line))
        assert isinstance(row["prompt"], list) and row["prompt"][-1]["role"] == "user"
        assert row["chosen"] and row["chosen"][0]["role"] == "assistant"
        assert row["rejected"] and row["rejected"][0]["role"] == "assistant"


def test_reward_model_no_longer_repeats_the_prompt():
    """RewardTrainer gets flattened text: prompt + completion. The string prompt
    used to be glued onto conversations that already began with it."""
    from soup_cli.trainer.reward_model import _prepare_reward_dataset

    row = _convert_dpo({"prompt": "What is 2+2?", "chosen": [U, GOOD], "rejected": [U, BAD]})
    (prepared,) = _prepare_reward_dataset([row])
    assert prepared == {"chosen": "What is 2+2? 4", "rejected": "What is 2+2? 5"}


# --------------------------------------------------------------------------
# The bundled examples through the real loader and a real TRL trainer
# --------------------------------------------------------------------------

_DPO_FORMAT_TASKS = ("dpo", "ipo", "orpo", "simpo", "bco")


@pytest.mark.parametrize("task", _DPO_FORMAT_TASKS)
@pytest.mark.parametrize("sample", ["dpo_sample.jsonl", "chat_preferences.jsonl"])
def test_bundled_sample_reaches_a_live_trl_trainer(tmp_path, monkeypatch, task, sample):
    """``setup()`` is called, not mocked: TRL tokenizes the converted rows."""
    import importlib

    from soup_cli.config.schema import DataConfig
    from soup_cli.data.loader import load_dataset
    from tests.test_trl_preference_config_contract import (
        _WRAPPERS,
        _cfg,
        _tiny_llama_dir,
        _write_tiny_tokenizer,
    )

    dataset = load_dataset(
        DataConfig(train=str(REPO / "examples" / "data" / sample), format="dpo", val_split=0)
    )
    weights = _tiny_llama_dir(tmp_path)
    _write_tiny_tokenizer(weights)
    monkeypatch.chdir(tmp_path)
    module, cls_name, _ = _WRAPPERS[task]
    wrapper = getattr(importlib.import_module(module), cls_name)(
        _cfg(weights, tmp_path / "out", task), device="cpu"
    )
    wrapper.setup({"train": dataset["train"]})

    assert wrapper.trainer is not None
    # BCO unpairs each row into a chosen (label True) and a rejected (False) row.
    per_row = 2 if task == "bco" else 1
    assert len(wrapper.trainer.train_dataset) == per_row * len(dataset["train"])
