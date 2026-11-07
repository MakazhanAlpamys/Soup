"""The preference trainers on the row layouts the data layer hands them (#1849).

``tests/test_formats_preference_layouts.py`` pins what the converters emit. This
file pins that it is what the trainers need, on the real objects:

- trl decides "conversational or text" by asking one field of the row, taken
  from a set, so the answer has to be the same for every field. That is asked
  of ``trl.data_utils.is_conversational`` itself, one field at a time, which
  holds under any ``PYTHONHASHSEED`` instead of sampling two of them.
- ``setup()`` is CALLED for every wrapper that reads these rows, on the tiny
  local Llama and on CPU, with the bundled example files as they ship. Before
  the fix dpo, ipo and bco raised inside trl, after the model had loaded, and
  so did kto on the same pairs written as unpaired rows. orpo and simpo did
  not (trl splits the layout for them), and they must prepare the same tokens
  as before.
- the tasks that prepare the rows themselves (``reward_model``, ``bco``,
  ``online_dpo``, ``grpo``, ``ppo``) see the prompt once.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from soup_cli.data.formats import format_to_messages

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "examples" / "data"
_BUNDLED = ("dpo_sample.jsonl", "chat_preferences.jsonl")


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


def _raw(name: str) -> list[dict]:
    path = _EXAMPLES / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _dpo_rows(name: str) -> list[dict]:
    """The bundled file as ``soup train`` hands it to a trainer."""
    rows = [format_to_messages(row, "dpo") for row in _raw(name)]
    assert all(row is not None for row in rows)
    return rows


def _kto_rows(name: str) -> list[dict]:
    """The same pairs written the way the file writes them, as unpaired rows:
    a string prompt beside a conversation that repeats it."""
    rows = []
    for row in _raw(name):
        for answer, label in ((row["chosen"], True), (row["rejected"], False)):
            rows.append(
                format_to_messages(
                    {"prompt": row["prompt"], "completion": answer, "label": label}, "kto"
                )
            )
    assert all(row is not None for row in rows)
    return rows


# --------------------------------------------------------------------------
# whichever field trl asks, the answer is the same
# --------------------------------------------------------------------------
class TestTrlSeesOneLayoutWhicheverFieldItAsks:
    @pytest.mark.parametrize("name", _BUNDLED)
    def test_bundled_dpo_rows(self, name):
        _requires_train_extra()
        from trl.data_utils import is_conversational

        for index, row in enumerate(_dpo_rows(name)):
            answers = {field: is_conversational({field: row[field]}) for field in row}
            assert answers == {"prompt": True, "chosen": True, "rejected": True}, (index, answers)
            assert is_conversational(row) is True

    @pytest.mark.parametrize("name", _BUNDLED)
    def test_the_same_pairs_as_kto_rows(self, name):
        _requires_train_extra()
        from trl.data_utils import is_conversational

        for index, row in enumerate(_kto_rows(name)):
            answers = {
                field: is_conversational({field: row[field]}) for field in ("prompt", "completion")
            }
            assert answers == {"prompt": True, "completion": True}, (index, answers)

    def test_text_rows_are_text_for_every_field(self):
        _requires_train_extra()
        from trl.data_utils import is_conversational

        row = format_to_messages({"prompt": "Q", "chosen": "good", "rejected": "bad"}, "dpo")
        assert {is_conversational({field: row[field]}) for field in row} == {False}

    @pytest.mark.parametrize("name", _BUNDLED)
    def test_the_file_as_written_is_what_trl_could_not_decide(self, name):
        """The defect itself, on the installed trl: the raw row answers both ways."""
        _requires_train_extra()
        from trl.data_utils import is_conversational

        row = _raw(name)[0]
        asked = {field: is_conversational({field: row[field]}) for field in row}
        assert asked == {"prompt": False, "chosen": True, "rejected": True}


# --------------------------------------------------------------------------
# the split is the one trl makes itself, where trl makes one
# --------------------------------------------------------------------------
_U = {"role": "user", "content": "What is 2+2?"}
_S = {"role": "system", "content": "Be brief."}
_A = {"role": "assistant", "content": "4"}
_B = {"role": "assistant", "content": "Fish"}
_U2 = {"role": "user", "content": "And 3+3?"}
_A2 = {"role": "assistant", "content": "6"}
_B2 = {"role": "assistant", "content": "Cat"}

_IMPLICIT_ROWS = {
    "a string prompt the answers repeat": {
        "prompt": "What is 2+2?", "chosen": [_U, _A], "rejected": [_U, _B]},
    "a shared system turn": {
        "prompt": "What is 2+2?", "chosen": [_S, _U, _A], "rejected": [_S, _U, _B]},
    "a shared history": {
        "prompt": "And 3+3?", "chosen": [_U, _A, _U2, _A2], "rejected": [_U, _A, _U2, _B2]},
    "a string the shared turns do not say": {
        "prompt": "Something else", "chosen": [_U, _A], "rejected": [_U, _B]},
    "no prompt column": {"chosen": [_U, _A], "rejected": [_U, _B]},
    "identical conversations": {
        "prompt": "What is 2+2?", "chosen": [_U, _A], "rejected": [_U, _A]},
    "one stops after an answer the other continues": {
        "prompt": "What is 2+2?", "chosen": [_U, _A, _U2, _A2], "rejected": [_U, _A]},
}


class TestTheSplitIsTheOneTrlMakes:
    """``trl.data_utils.maybe_extract_prompt`` already splits this layout for
    the trainers that hand it paired rows (ORPO, and CPO for SimPO), so those
    two trained on the bundled files all along. trl's ``DPOTrainer`` does not
    split a row that has a ``prompt`` key (dpo, ipo), and the BCO wrapper
    unpairs the rows itself before trl sees a pair to split. The converter
    makes the same split for all of them, so ORPO and SimPO train on the rows
    they always had."""

    @pytest.mark.parametrize("name", _BUNDLED)
    def test_on_the_bundled_files(self, name):
        _requires_train_extra()
        from trl.data_utils import maybe_extract_prompt

        for index, row in enumerate(_raw(name)):
            assert format_to_messages(row, "dpo") == maybe_extract_prompt(dict(row)), index

    @pytest.mark.parametrize("name", list(_IMPLICIT_ROWS))
    def test_on_each_implicit_spelling(self, name):
        _requires_train_extra()
        from trl.data_utils import maybe_extract_prompt

        row = _IMPLICIT_ROWS[name]
        assert format_to_messages(row, "dpo") == maybe_extract_prompt(dict(row))

    @pytest.mark.parametrize("task", ["dpo", "ipo", "bco"])
    def test_the_file_as_written_fails_where_nothing_splits_it(self, tmp_path, monkeypatch, task):
        """The premise, on the installed trl: handed the file as written, these
        three fail in ``setup()``. Which error it is depends on the hash seed,
        so only the failure is asserted. If trl starts splitting this layout
        there too, this test fails and says so."""
        _requires_train_extra()
        module, cls_name, _, _ = _WRAPPERS[task]
        monkeypatch.chdir(tmp_path)
        wrapper = getattr(importlib.import_module(module), cls_name)(
            _config(tmp_path, task), device="cpu"
        )
        with pytest.raises(Exception) as caught:  # noqa: B017, PT011 - see the docstring
            wrapper.setup({"train": [dict(row) for row in _raw("chat_preferences.jsonl")]})
        assert caught.type.__name__ in {"AttributeError", "TypeError", "UndefinedError"}, caught

    def test_the_same_pairs_written_as_unpaired_rows_fail_in_kto(self, tmp_path, monkeypatch):
        """kto has its own converter and the same premise: a string prompt
        beside a conversation, which trl has no pair to split."""
        _requires_train_extra()
        from soup_cli.trainer.kto import KTOTrainerWrapper

        monkeypatch.chdir(tmp_path)
        rows = [
            {"prompt": row["prompt"], "completion": row[side], "label": side == "chosen"}
            for row in _raw("chat_preferences.jsonl")
            for side in ("chosen", "rejected")
        ]
        wrapper = KTOTrainerWrapper(_config(tmp_path, "kto"), device="cpu")
        with pytest.raises(Exception) as caught:  # noqa: B017, PT011 - as above
            wrapper.setup({"train": rows})
        assert caught.type.__name__ in {"AttributeError", "TypeError", "UndefinedError"}, caught

    @pytest.mark.parametrize("task", ["orpo", "simpo"])
    def test_orpo_and_simpo_always_split_it(self, tmp_path, monkeypatch, task):
        """And the other half: these two set up on the file as written."""
        _requires_train_extra()
        module, cls_name, _, _ = _WRAPPERS[task]
        monkeypatch.chdir(tmp_path)
        wrapper = getattr(importlib.import_module(module), cls_name)(
            _config(tmp_path, task), device="cpu"
        )
        rows = [dict(row) for row in _raw("chat_preferences.jsonl")]
        wrapper.setup({"train": rows})
        assert len(wrapper.trainer.train_dataset) == len(rows)

    @pytest.mark.parametrize("task", ["orpo", "simpo"])
    def test_orpo_and_simpo_prepare_the_same_tokens_either_way(self, tmp_path, monkeypatch, task):
        """No change for the two that worked: the converted rows tokenize to
        exactly what trl made of the file as written."""
        _requires_train_extra()
        module, cls_name, _, _ = _WRAPPERS[task]
        monkeypatch.chdir(tmp_path)
        prepared = []
        for rows in (
            [dict(row) for row in _raw("chat_preferences.jsonl")],
            _dpo_rows("chat_preferences.jsonl"),
        ):
            wrapper = getattr(importlib.import_module(module), cls_name)(
                _config(tmp_path, task), device="cpu"
            )
            wrapper.setup({"train": rows})
            dataset = wrapper.trainer.train_dataset
            columns = sorted(name for name in dataset.column_names if name.endswith("input_ids"))
            assert columns, dataset.column_names
            prepared.append({name: dataset[name] for name in columns})
        assert prepared[0] == prepared[1]


# --------------------------------------------------------------------------
# setup() on every wrapper that reads these rows
# --------------------------------------------------------------------------
#: task -> (wrapper module, class, rows for a bundled file, lines under training:)
_WRAPPERS = {
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper", _dpo_rows, ""),
    "ipo": ("soup_cli.trainer.ipo", "IPOTrainerWrapper", _dpo_rows, ""),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper", _dpo_rows, ""),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper", _dpo_rows, ""),
    "bco": ("soup_cli.trainer.bco", "BCOTrainerWrapper", _dpo_rows, ""),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper", _kto_rows, ""),
    "preference": (
        "soup_cli.trainer.preference",
        "PreferenceTrainerWrapper",
        _dpo_rows,
        "  preference_loss: dpo\n",
    ),
}

#: KTO refuses a batch of one (its KL term is degenerate there).
_BATCH = {"kto": 2}


def _config(tmp_path, task: str, *, batch_size: int | None = None, fmt: str | None = None):
    from soup_cli.config.loader import load_config_from_string
    from tests._tiny_hf_models import tiny_model_dir

    training = _WRAPPERS[task][3] if task in _WRAPPERS else ""
    lora = (
        ""
        if task == "reward_model"
        else "  lora:\n    r: 4\n    alpha: 8\n    target_modules: [q_proj, v_proj]\n"
    )
    fmt = fmt or ("kto" if task == "kto" else "dpo")
    return load_config_from_string(
        f"base: {Path(tiny_model_dir('llama')).as_posix()}\n"
        f"task: {task}\n"
        "backend: transformers\n"
        "data:\n  train: x.jsonl\n"
        f"  format: {fmt}\n"
        # Long enough that no bundled row is truncated: the layout is what is
        # under test, not the caps (tests/test_trl_preference_config_contract.py).
        "  max_length: 2048\n  val_split: 0\n  chat_template: chatml\n"
        "training:\n  epochs: 1\n"
        f"  batch_size: {batch_size or _BATCH.get(task, 1)}\n"
        "  gradient_accumulation_steps: 1\n  quantization: none\n  lr: 1e-3\n"
        "  logging_steps: 1\n  save_steps: 1000\n"
        f"{lora}{training}"
        f"output: {(tmp_path / 'out').as_posix()}\n"
    )


def _setup(tmp_path, monkeypatch, task: str, name: str):
    _requires_train_extra()
    module, cls_name, rows_for, _ = _WRAPPERS[task]
    monkeypatch.chdir(tmp_path)
    wrapper = getattr(importlib.import_module(module), cls_name)(
        _config(tmp_path, task), device="cpu"
    )
    rows = rows_for(name)
    wrapper.setup({"train": rows})
    return wrapper, rows


class TestEveryPreferenceTrainerSetsUpOnTheBundledFiles:
    @pytest.mark.parametrize("name", _BUNDLED)
    @pytest.mark.parametrize("task", list(_WRAPPERS))
    def test_setup_prepares_every_row(self, tmp_path, monkeypatch, task, name):
        wrapper, rows = _setup(tmp_path, monkeypatch, task, name)
        trainer = wrapper.trainer
        assert trainer is not None, f"{task}: setup() left no trainer"
        # bco writes each pair as two unpaired rows
        expected = len(rows) * (2 if task == "bco" else 1)
        assert len(trainer.train_dataset) == expected

    @pytest.mark.parametrize("name", _BUNDLED)
    @pytest.mark.parametrize("task", ["dpo", "ipo"])
    def test_the_question_is_tokenized_once_and_the_answers_follow_it(
        self, tmp_path, monkeypatch, task, name
    ):
        """What trl's DPOTrainer prepared, decoded: the chat template's user
        turn with the question in it once, then each answer once."""
        wrapper, rows = _setup(tmp_path, monkeypatch, task, name)
        source = _raw(name)
        prepared = wrapper.trainer.train_dataset
        tokenizer = wrapper.tokenizer
        for index, row in enumerate(prepared):
            question = source[index]["prompt"]
            prompt = tokenizer.decode(row["prompt_ids"])
            assert prompt == f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            for side in ("chosen", "rejected"):
                answer = tokenizer.decode(row[f"{side}_ids"])
                assert answer == source[index][side][-1]["content"] + "<|im_end|>\n", (index, side)
                assert question not in answer

    @pytest.mark.parametrize("name", _BUNDLED)
    def test_dpo_trains_on_the_bundled_file(self, tmp_path, monkeypatch, name):
        wrapper, rows = _setup(tmp_path, monkeypatch, "dpo", name)
        result = wrapper.train()
        assert result["total_steps"] == len(rows)
        assert result["final_loss"] == result["final_loss"], "the loss is NaN"

    def test_simpo_trains_on_the_bundled_file(self, tmp_path, monkeypatch):
        """One wrapper of the experimental family (trl's CPOTrainer), end to end."""
        wrapper, rows = _setup(tmp_path, monkeypatch, "simpo", "chat_preferences.jsonl")
        result = wrapper.train()
        assert result["total_steps"] == len(rows)


# --------------------------------------------------------------------------
# the tasks that prepare the rows themselves
# --------------------------------------------------------------------------
class TestRewardModel:
    @pytest.mark.parametrize("name", _BUNDLED)
    def test_the_question_is_in_each_text_once(self, name):
        """The reported duplication: the prompt was prepended to a ``chosen``
        that already began with the user turn."""
        from soup_cli.trainer.reward_model import _prepare_reward_dataset

        source = _raw(name)
        prepared = _prepare_reward_dataset(_dpo_rows(name))
        assert len(prepared) == len(source)
        for row, pair in zip(source, prepared):
            question = row["prompt"]
            for side in ("chosen", "rejected"):
                assert pair[side] == f"{question} {row[side][-1]['content']}"
                assert pair[side].count(question) == 1

    def test_setup_builds_a_trainer_on_the_bundled_file(self, tmp_path, monkeypatch):
        _requires_train_extra()
        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        monkeypatch.chdir(tmp_path)
        wrapper = RewardModelTrainerWrapper(_config(tmp_path, "reward_model"), device="cpu")
        rows = _dpo_rows("chat_preferences.jsonl")
        wrapper.setup({"train": rows})
        assert len(wrapper.trainer.train_dataset) == len(rows)


class TestBco:
    @pytest.mark.parametrize("name", _BUNDLED)
    def test_the_unpaired_rows_keep_one_layout(self, name):
        from soup_cli.trainer.bco import _split_dpo_rows_to_bco

        rows = _dpo_rows(name)
        unpaired = _split_dpo_rows_to_bco(rows)
        assert [row["label"] for row in unpaired] == [True, False] * len(rows)
        for row in unpaired:
            assert isinstance(row["prompt"], list) and isinstance(row["completion"], list)
            assert [turn["role"] for turn in row["prompt"]] == ["user"]
            assert [turn["role"] for turn in row["completion"]] == ["assistant"]


class TestOnlineDpo:
    """The task generates its own answers and reads only the prompt. A prompt
    that is a message list was not read at all, so the bundled files, which now
    convert to that layout, would have left it with no rows."""

    @staticmethod
    def _prompt_rows(rows):
        from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

        return OnlineDPOTrainerWrapper._to_prompt_rows(rows)

    @pytest.mark.parametrize("name", _BUNDLED)
    def test_the_bundled_rows_give_one_user_turn_each(self, name):
        source = _raw(name)
        assert self._prompt_rows(_dpo_rows(name)) == [
            {"prompt": [{"role": "user", "content": row["prompt"]}]} for row in source
        ]

    def test_a_message_list_prompt_keeps_its_system_turn(self):
        system = {"role": "system", "content": "be nice"}
        user = {"role": "user", "content": "hi"}
        rows = [{"prompt": [system, user], "chosen": [], "rejected": []}]
        assert self._prompt_rows(rows) == [{"prompt": [system, user]}]

    def test_a_message_list_prompt_ends_on_its_last_user_turn(self):
        turns = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        rows = [{"prompt": [*turns, {"role": "assistant", "content": "a2"}]}]
        assert self._prompt_rows(rows) == [{"prompt": turns}]

    def test_a_message_list_prompt_without_a_user_turn_is_skipped(self):
        assert self._prompt_rows([{"prompt": [{"role": "assistant", "content": "x"}]}]) == []

    def test_a_string_prompt_and_messages_read_as_before(self):
        user = {"role": "user", "content": "hi"}
        rows = [{"prompt": "hi"}, {"messages": [user, {"role": "assistant", "content": "yo"}]}]
        assert self._prompt_rows(rows) == [{"prompt": [user]}, {"prompt": [user]}]

    def test_a_row_with_a_string_prompt_and_messages_gives_one_prompt(self):
        """One row, one prompt: the string, as before. (UltraFeedback rows carry
        both keys, and so does a row whose source columns were kept.)"""
        rows = [{"prompt": "x", "messages": [{"role": "user", "content": "y"}]}]
        assert self._prompt_rows(rows) == [{"prompt": [{"role": "user", "content": "x"}]}]

    def test_a_row_with_a_message_list_prompt_and_messages_gives_one_prompt(self):
        prompt = [{"role": "user", "content": "x"}]
        rows = [{"prompt": prompt, "messages": [{"role": "user", "content": "y"}]}]
        assert self._prompt_rows(rows) == [{"prompt": prompt}]


class TestGrpoAndPpoReadTheSamePrompt:
    """Both accept dpo rows and read the prompt only; a bundled row gave one
    user turn before, and still does."""

    @pytest.mark.parametrize("name", _BUNDLED)
    def test_grpo(self, name):
        from soup_cli.trainer.grpo import _prepare_grpo_dataset

        source = _raw(name)
        prepared = _prepare_grpo_dataset(_dpo_rows(name))
        assert [row["prompt"] for row in prepared] == [
            [{"role": "user", "content": row["prompt"]}] for row in source
        ]

    @pytest.mark.parametrize("name", _BUNDLED)
    def test_ppo_without_a_chat_template(self, name):
        from soup_cli.trainer.ppo import _prepare_ppo_dataset

        source = _raw(name)
        prepared = _prepare_ppo_dataset(_dpo_rows(name))
        assert [row["prompt_text"] for row in prepared] == [row["prompt"] for row in source]


class TestTheMlxBackendGetsTheExplicitLayout:
    """``backend: mlx`` receives the same loaded rows. On main its DPO wrapper
    stores them untouched; the native loop in #1846 parses them itself and
    takes trl's explicit layout (a message-list prompt, answers that do not
    repeat it) on its first branch. That is the shape pinned here."""

    @pytest.mark.parametrize("name", _BUNDLED)
    def test_the_prompt_is_a_message_list_the_answers_do_not_repeat(self, name):
        for row in _dpo_rows(name):
            assert all(isinstance(row[field], list) for field in ("prompt", "chosen", "rejected"))
            size = len(row["prompt"])
            assert row["chosen"][:size] != row["prompt"]
            assert row["rejected"][:size] != row["prompt"]
            assert row["prompt"][-1]["role"] == "user"
            assert row["chosen"][0]["role"] == row["rejected"][0]["role"] == "assistant"
