"""The live generator says whether an answer stopped at its new-token cap (``GeneratedText``).

``soup ship`` records this next to the verdict, so a reader can tell a tie between two cut-off
answers from a tie between two finished ones. The flag is ``True`` (decoding used its whole
``max_new_tokens`` budget and the last token is not a stop token), ``False`` (it stopped on its
own, including a stop token that landed exactly on the last allowed position) or ``None`` (this
generator could not tell). It comes from the token ids the model returned, never from the length
of the text. A model here is a scripted object whose ``generate`` returns the token ids the test
chose, so no weights, no GPU and no network are used.
"""

from __future__ import annotations

import copy
import json
import pickle
import subprocess
import sys

import pytest

from soup_cli.eval.custom import EvalTask, run_eval
from soup_cli.utils import live_eval
from soup_cli.utils.live_eval import GeneratedText, truncation_of

EOS, HELLO, WORLD, PAD = 2, 3, 4, 6
_ABSENT = object()


def _tokenizer(*, eos="<eos>", pad="<eos>"):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<bos>", "<eos>", "hello", "world", "other", "<pad>"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    special = {name: token for name, token in (("eos_token", eos), ("pad_token", pad)) if token}
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", unk_token="<unk>", **special
    )


@pytest.fixture
def tokenizer():
    return _tokenizer()  # pad == eos, the common case


class ScriptedModel:
    """``generate`` appends the scripted new ids to each prompt row, as transformers does.

    ``rows`` is one list of new ids per prompt, consumed in call order. A row that stops early is
    filled with the ``pad_token_id`` it was called with, like a finished row of a real batch.
    """

    def __init__(self, *rows, eos_token_id=_ABSENT):
        self.rows = [list(row) for row in rows]
        self.caps: list = []
        if eos_token_id is not _ABSENT:
            self.generation_config = type("Config", (), {"eos_token_id": eos_token_id})()

    def generate(self, input_ids, **kwargs):
        import torch

        taken, self.rows = self.rows[: len(input_ids)], self.rows[len(input_ids):]
        assert len(taken) == len(input_ids), "the test scripted fewer rows than prompts"
        self.caps.append(kwargs["max_new_tokens"])
        width = max(len(row) for row in taken)
        pad = kwargs.get("pad_token_id")
        new = torch.tensor([row + [pad] * (width - len(row)) for row in taken])
        return torch.cat([input_ids, new], dim=1)


def _generator(tokenizer, *rows, cap=_ABSENT, eos_token_id=_ABSENT, batch_size=8):
    model = ScriptedModel(*rows, eos_token_id=eos_token_id)
    options = {} if cap is _ABSENT else {"max_new_tokens": cap}
    generator = live_eval.make_generator(
        "unused", loaded=(model, tokenizer, "cpu"), batch_size=batch_size, **options
    )
    return generator, model


def _answer(tokenizer, path, new_ids, **options):
    """One answer through the scalar closure or through ``.batch`` with a single prompt."""
    generator, model = _generator(tokenizer, new_ids, **options)
    text = generator("hello") if path == "scalar" else generator.batch(["hello"])[0]
    return text, model


def test_generated_text_behaves_as_the_str_it_wraps():
    capped = GeneratedText("an answer", truncated=True)
    finished = GeneratedText("an answer", truncated=False)
    assert capped == finished == "an answer" and hash(capped) == hash("an answer")
    assert isinstance(capped, str)
    assert f"<{capped}>" == "<an answer>" and capped.upper() == "AN ANSWER"
    assert capped[:2] == "an" and capped + "!" == "an answer!" and "swer" in capped
    assert {capped: 1}["an answer"] == 1 and {"an answer"} == {capped, finished}
    assert json.dumps({"text": capped}) == '{"text": "an answer"}'
    assert json.loads(json.dumps([capped])) == ["an answer"]
    assert str(capped) == "an answer" and type(str(capped)) is str


def test_the_flag_is_true_false_or_none_and_never_invented():
    assert GeneratedText("x", truncated=True).truncated is True
    assert GeneratedText("x", truncated=False).truncated is False
    assert GeneratedText("x", truncated=None).truncated is None  # not measured stays unmeasured
    assert GeneratedText("x", truncated=1).truncated is True  # anything else is a real bool
    with pytest.raises(TypeError):
        GeneratedText("x")  # no default: a caller says True, False or None on purpose


@pytest.mark.parametrize("flag", [True, False, None])
def test_the_flag_survives_a_copy_and_a_pickle(flag):
    text = GeneratedText("an answer", truncated=flag)
    for clone in (copy.copy(text), copy.deepcopy(text), pickle.loads(pickle.dumps(text))):
        assert type(clone) is GeneratedText and clone == "an answer"
        assert clone.truncated is flag


def test_string_operations_lose_the_flag_and_the_reader_says_not_measured():
    text = GeneratedText("an answer", truncated=True)
    assert truncation_of(text) is True
    assert truncation_of(GeneratedText("an answer", truncated=False)) is False
    assert truncation_of(GeneratedText("an answer", truncated=None)) is None
    for derived in (text + "!", text[:2], text.strip(), str(text), "plain"):
        assert type(derived) is str
        assert truncation_of(derived) is None  # a plain str is never reported as False
    assert truncation_of(None) is None and truncation_of(3) is None


@pytest.mark.parametrize("path", ["scalar", "batch"])
@pytest.mark.parametrize(
    "new_ids,cap,truncated",
    [
        ([HELLO, WORLD, HELLO, WORLD], 4, True),  # used the whole budget, never stopped
        ([HELLO, WORLD, HELLO], 4, False),  # stopped before the cap
        ([HELLO, WORLD, HELLO, EOS], 4, False),  # the last allowed token IS the stop: finished
        ([EOS], 1, False),
        ([HELLO], 1, True),
        ([HELLO, WORLD, EOS], 8, False),
    ],
)
def test_the_flag_follows_the_new_tokens_the_model_returned(
    tokenizer, path, new_ids, cap, truncated
):
    text, model = _answer(tokenizer, path, new_ids, cap=cap)
    assert isinstance(text, GeneratedText)
    assert text.truncated is truncated
    assert model.caps == [cap]  # the model was asked for what the generator was built with


@pytest.mark.parametrize("path", ["scalar", "batch"])
def test_the_flag_does_not_depend_on_how_long_the_text_is(tokenizer, path):
    short, _ = _answer(tokenizer, path, [HELLO], cap=1)  # five letters, cut off
    long, _ = _answer(tokenizer, path, [HELLO, WORLD] * 20 + [EOS], cap=64)  # long, finished
    assert short == "hello" and short.truncated is True
    assert len(long) > 200 and long.truncated is False


@pytest.mark.parametrize("path", ["scalar", "batch"])
def test_the_default_budget_is_the_one_the_flag_is_measured_against(tokenizer, path):
    full, model = _answer(tokenizer, path, [HELLO] * 64)
    short, _ = _answer(tokenizer, path, [HELLO] * 63)
    ended, _ = _answer(tokenizer, path, [HELLO] * 63 + [EOS])
    assert model.caps == [64]
    assert (full.truncated, short.truncated, ended.truncated) == (True, False, False)


def test_a_cap_of_none_is_refused_and_never_treated_as_unlimited(tokenizer):
    model = ScriptedModel([HELLO])
    with pytest.raises(ValueError, match="max_new_tokens must be a positive int"):
        live_eval.make_generator(
            "unused", loaded=(model, tokenizer, "cpu"), max_new_tokens=None
        )


@pytest.mark.parametrize("path", ["scalar", "batch"])
@pytest.mark.parametrize("stop", [WORLD, [99, WORLD], (99, WORLD)], ids=["int", "list", "tuple"])
def test_a_stop_token_of_the_models_generation_config_is_a_finished_answer(
    tokenizer, path, stop
):
    # the model stops on id 4, not on the tokenizer's eos: ending on it at the cap is not a cut-off
    ended, _ = _answer(tokenizer, path, [HELLO] * 3 + [WORLD], cap=4, eos_token_id=stop)
    cut, _ = _answer(tokenizer, path, [HELLO] * 4, cap=4, eos_token_id=stop)
    assert ended.truncated is False
    assert cut.truncated is True


@pytest.mark.parametrize("path", ["scalar", "batch"])
def test_the_tokenizer_eos_still_counts_when_the_model_names_another_stop(tokenizer, path):
    text, _ = _answer(tokenizer, path, [HELLO] * 3 + [EOS], cap=4, eos_token_id=WORLD)
    assert text.truncated is False


@pytest.mark.parametrize("path", ["scalar", "batch"])
def test_a_model_with_no_generation_config_still_reports(tokenizer, path):
    generator, model = _generator(tokenizer, [HELLO, WORLD], cap=2)
    assert not hasattr(model, "generation_config")
    text = generator("hello") if path == "scalar" else generator.batch(["hello"])[0]
    assert text.truncated is True


def test_a_pad_token_that_is_not_the_eos_is_still_a_finished_row():
    tokenizer = _tokenizer(eos="<eos>", pad="<pad>")
    assert tokenizer.pad_token_id == PAD != tokenizer.eos_token_id
    # row 0 is cut off; row 1 stops after one token and is filled with the PAD id up to the width
    generator, _ = _generator(tokenizer, [HELLO, WORLD, HELLO, WORLD], [HELLO, EOS], cap=4)
    cut, finished = generator.batch(["hello", "world"])
    assert (cut.truncated, finished.truncated) == (True, False)
    # a row that ended on the stop token exactly at the cap is finished too
    ended, _ = _answer(tokenizer, "scalar", [HELLO] * 3 + [EOS], cap=4)
    assert ended.truncated is False


def test_a_batch_reports_each_row_by_its_own_tokens_across_chunks(tokenizer):
    rows = [
        [HELLO, WORLD, HELLO, WORLD],  # cut off
        [HELLO, EOS],  # stops early, filled with the pad id
        [WORLD, HELLO, WORLD, EOS],  # stop token on the last allowed position
        [EOS],
        [HELLO] * 4,  # cut off, alone in the last chunk
    ]
    generator, model = _generator(tokenizer, *rows, cap=4, batch_size=2)
    texts = generator.batch(["hello", "world", "other", "hello", "world"])
    assert [type(text) for text in texts] == [GeneratedText] * 5
    assert [text.truncated for text in texts] == [True, False, False, False, True]
    assert texts[0] == "hello world hello world" and texts[1] == "hello" and texts[3] == ""
    assert model.caps == [4, 4, 4]


def test_a_batch_where_every_row_stops_before_the_cap_reports_false_for_all(tokenizer):
    generator, _ = _generator(tokenizer, [HELLO, EOS], [HELLO, WORLD, EOS], cap=4)
    assert [text.truncated for text in generator.batch(["hello", "world"])] == [False, False]
    assert generator.batch([]) == []


def test_no_known_stop_token_means_the_cap_cannot_be_judged():
    bare = _tokenizer(eos=None, pad=None)
    assert bare.eos_token_id is None and bare.pad_token_id is None
    # nothing tells a finished answer from a cut-off one at the cap: not measured, not False
    at_cap, _ = _answer(bare, "scalar", [HELLO, WORLD], cap=2)
    assert isinstance(at_cap, GeneratedText) and at_cap.truncated is None
    # but a generation that came back short stopped on its own whatever the stop set is
    short, _ = _answer(bare, "scalar", [HELLO], cap=2)
    assert short.truncated is False


@pytest.mark.parametrize("new_ids", [5, None, object(), 3.5])
def test_token_ids_that_cannot_be_counted_are_not_measured(new_ids):
    assert live_eval._hit_new_token_cap(new_ids, 4, frozenset({EOS})) is None


def test_the_stop_set_reads_pad_tokenizer_eos_and_every_model_spelling(tokenizer):
    def stops(config_eos, pad=PAD):
        model = ScriptedModel(eos_token_id=config_eos)
        return live_eval._stop_token_ids(model, tokenizer, pad)

    assert stops(7) == frozenset({PAD, EOS, 7})
    assert stops([7, 8]) == frozenset({PAD, EOS, 7, 8})
    assert stops((7, 8)) == frozenset({PAD, EOS, 7, 8})
    assert stops(None) == frozenset({PAD, EOS})
    assert stops(True) == frozenset({PAD, EOS})  # a bool is not a token id
    assert live_eval._stop_token_ids(object(), tokenizer, None) == frozenset({EOS})


def test_the_text_is_what_the_generator_returned_before_the_flag_existed(tokenizer):
    text, _ = _answer(tokenizer, "scalar", [HELLO, WORLD, EOS], cap=8)
    assert str(text) == "hello world" and text == "hello world"


def test_existing_scorers_take_the_flagged_text_like_any_other_answer(tokenizer):
    tasks = [EvalTask("hello", "hello world"), EvalTask("world", "other")]

    def run(generator):
        result = run_eval("unused", tasks, generator)
        return [item.score for item in result.results], result.accuracy

    plain = run(lambda prompt: "hello world" if prompt == "hello" else "nope")
    generator, _model = _generator(tokenizer, [HELLO, WORLD, EOS], [HELLO, EOS], cap=8)
    assert run(generator) == plain == ([1, 0], 0.5)


def test_importing_the_module_does_not_import_torch_or_transformers():
    code = (
        "import sys; import soup_cli.utils.live_eval as module; "
        "assert module.GeneratedText and module.truncation_of; "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules, "
        "sorted(m for m in sys.modules if m in ('torch', 'transformers'))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
