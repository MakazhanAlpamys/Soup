"""The live generator says whether an answer stopped at its new-token cap (``GeneratedText``).

``soup ship`` keeps this beside the judge's verdict (``base_truncated`` / ``tuned_truncated`` in
the judge rows), so a reader can tell a tie between two cut-off answers from a tie between two
finished ones. The flag comes from the generation result (how many new tokens came back and what
the last one was), never from the length of the text. A model here is a scripted object whose
``generate`` returns the token ids the test chose, so no weights, no GPU and no network are used.
"""

from __future__ import annotations

import copy
import json
import pickle

import pytest

from soup_cli.utils import live_eval
from soup_cli.utils.live_eval import GeneratedText

EOS = 2
HELLO, WORLD = 3, 4


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {
        word: index
        for index, word in enumerate(["<unk>", "<bos>", "<eos>", "hello", "world", "other"])
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<eos>",
        unk_token="<unk>",
    )


class ScriptedModel:
    """``generate`` returns the prompt followed by the new token ids the test scripted."""

    def __init__(self, *new_ids_per_row, eos_token_id=None):
        self.new_ids_per_row = new_ids_per_row
        self.caps: list = []
        if eos_token_id is not None:
            self.generation_config = type("Config", (), {"eos_token_id": eos_token_id})()

    def generate(self, input_ids, **kwargs):
        import torch

        self.caps.append(kwargs["max_new_tokens"])
        width = max(len(row) for row in self.new_ids_per_row)
        padded = [row + [EOS] * (width - len(row)) for row in self.new_ids_per_row]
        new = torch.tensor(padded)
        return torch.cat([input_ids, new], dim=1)


def _generate(tokenizer, new_ids, *, cap, eos_token_id=None):
    model = ScriptedModel(list(new_ids), eos_token_id=eos_token_id)
    generator = live_eval.make_generator(
        "unused", loaded=(model, tokenizer, "cpu"), max_new_tokens=cap
    )
    return generator("hello"), model


def test_generated_text_is_a_string_that_also_carries_the_flag():
    capped = GeneratedText("an answer", truncated=True)
    finished = GeneratedText("an answer", truncated=False)
    assert capped == finished == "an answer" and hash(capped) == hash("an answer")
    assert isinstance(capped, str)
    assert capped.truncated is True and finished.truncated is False
    assert f"<{capped}>" == "<an answer>" and capped.upper() == "AN ANSWER"
    assert json.dumps({"text": capped}) == '{"text": "an answer"}'
    assert GeneratedText("x", truncated=1).truncated is True  # always a real bool


@pytest.mark.parametrize("flag", [True, False])
def test_the_flag_survives_a_copy_and_a_pickle(flag):
    text = GeneratedText("an answer", truncated=flag)
    for clone in (
        copy.copy(text),
        copy.deepcopy(text),
        pickle.loads(pickle.dumps(text)),
    ):
        assert type(clone) is GeneratedText and clone == "an answer"
        assert clone.truncated is flag


@pytest.mark.parametrize(
    "new_ids,cap,truncated",
    [
        ([HELLO, WORLD, HELLO, WORLD], 4, True),  # used the whole budget, never stopped
        ([HELLO, WORLD, HELLO], 4, False),  # stopped before the cap
        ([HELLO, WORLD, HELLO, EOS], 4, False),  # the last token IS the stop: it finished
        ([EOS], 1, False),
        ([HELLO], 1, True),
    ],
)
def test_the_flag_follows_the_new_tokens_the_model_returned(tokenizer, new_ids, cap, truncated):
    text, model = _generate(tokenizer, new_ids, cap=cap)
    assert isinstance(text, GeneratedText)
    assert text.truncated is truncated
    assert model.caps == [cap]  # the model was asked for what the generator was built with


def test_the_flag_does_not_depend_on_how_long_the_text_is(tokenizer):
    # one token, cut off at a cap of one: a five-letter answer that is truncated ...
    short, _ = _generate(tokenizer, [HELLO], cap=1)
    # ... and a long answer that is not
    long, _ = _generate(tokenizer, [HELLO, WORLD] * 20 + [EOS], cap=64)
    assert short == "hello" and short.truncated is True
    assert len(long) > 200 and long.truncated is False


@pytest.mark.parametrize("stop", [WORLD, [99, WORLD]], ids=["int", "list"])
def test_a_stop_token_of_the_models_generation_config_is_a_finished_answer(tokenizer, stop):
    # the model stops on id 4 (not the tokenizer's eos): ending on it at the cap is not a cut-off
    text, _ = _generate(tokenizer, [HELLO, HELLO, HELLO, WORLD], cap=4, eos_token_id=stop)
    assert text.truncated is False
    other, _ = _generate(tokenizer, [HELLO, HELLO, HELLO, HELLO], cap=4, eos_token_id=stop)
    assert other.truncated is True


def test_a_model_with_no_generation_config_still_reports(tokenizer):
    model = ScriptedModel([HELLO, WORLD])
    assert not hasattr(model, "generation_config")
    generator = live_eval.make_generator(
        "unused", loaded=(model, tokenizer, "cpu"), max_new_tokens=2
    )
    assert generator("hello").truncated is True


def test_the_batch_path_reports_each_row_by_its_own_tokens(tokenizer):
    # row 0 uses the whole budget; row 1 stops after one token and is padded with the stop id
    model = ScriptedModel([HELLO, WORLD, HELLO, WORLD], [HELLO, EOS])
    generator = live_eval.make_generator(
        "unused", loaded=(model, tokenizer, "cpu"), max_new_tokens=4
    )
    texts = generator.batch(["hello", "world"])
    assert [type(text) for text in texts] == [GeneratedText, GeneratedText]
    assert [text.truncated for text in texts] == [True, False]
    assert texts[0] == "hello world hello world" and texts[1] == "hello"


def test_the_text_is_what_the_generator_returned_before_the_flag_existed(tokenizer):
    text, _ = _generate(tokenizer, [HELLO, WORLD, EOS], cap=8)
    assert str(text) == "hello world"
    assert type(str(text)) is str
