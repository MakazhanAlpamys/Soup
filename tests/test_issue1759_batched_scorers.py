"""#1759: scorer batching preserves ordered scalar behavior and greedy output."""

import pytest

from soup_cli.eval.custom import EvalTask, run_eval
from soup_cli.eval.gate_suites import (
    DEFAULT_GENERAL_SUITE,
    _fingerprint_generator,
    score_bundled_suite,
)


class BatchGenerator:
    def __init__(self, scalar):
        self.scalar = scalar
        self.batches = []

    def __call__(self, prompt):
        pytest.fail("batch-capable generator must not be called one prompt at a time")

    def batch(self, prompts):
        self.batches.append(prompts)
        return [self.scalar(prompt) for prompt in prompts]


@pytest.mark.parametrize("suite", DEFAULT_GENERAL_SUITE)
def test_all_bundled_suites_have_identical_scalar_and_batch_scores(suite):
    batch = BatchGenerator(_fingerprint_generator)
    expected = score_bundled_suite(suite, _fingerprint_generator)
    assert 0 < expected < 1
    assert score_bundled_suite(suite, batch) == expected
    assert len(batch.batches) == 1
    assert len(batch.batches[0]) > 20


def test_custom_eval_preserves_order_outputs_scores_and_categories():
    tasks = [
        EvalTask("one", "A", "alpha"),
        EvalTask("two", "B", "beta"),
        EvalTask("one", "wrong", "beta"),
    ]

    def scalar(prompt):
        return {"one": "A", "two": "B"}[prompt]

    batch = BatchGenerator(scalar)
    assert run_eval("unused", tasks, batch) == run_eval("unused", tasks, scalar)
    assert batch.batches == [["one", "two", "one"]]


@pytest.mark.parametrize("batched", [False, True])
def test_generation_error_is_failed_item_even_with_empty_expected(batched):
    def scalar(prompt):
        if prompt == "bad":
            raise RuntimeError("generation failed")
        return "A"

    class Generator:
        __call__ = staticmethod(scalar)

        def batch(self, prompts):
            raise RuntimeError("batch failed")

    gen = Generator() if batched else scalar
    result = run_eval("unused", [EvalTask("bad", ""), EvalTask("good", "A")], gen)
    assert [item.score for item in result.results] == [0, 1]
    assert result.accuracy == 0.5
    assert score_bundled_suite("mini_over_refusal", lambda prompt: scalar("bad")) == 0


@pytest.mark.parametrize("outputs", [["A"], ["A", "B", "C"], "AB", [None, "B"]])
def test_malformed_batch_does_not_misalign_or_award_failure(outputs):
    class Generator:
        def __call__(self, prompt):
            return "unexpected scalar retry"

        def batch(self, prompts):
            return outputs

    result = run_eval("unused", [EvalTask("one", ""), EvalTask("two", "B")], Generator())
    assert result.results[0].score == 0
    assert result.results[1].score == (1 if outputs == [None, "B"] else 0)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_real_cpu_greedy_batch_matches_scalar(arch, batch_size):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from soup_cli.utils.live_eval import make_generator
    from tests._tiny_hf_models import build_gpt2, build_llama

    torch.manual_seed(1759)
    tokenizer = tiny_tokenizer()
    tokenizer.padding_side = "right"  # The generator must enforce left padding itself.
    model = (build_gpt2 if arch == "gpt2" else build_llama)(tokenizer).eval()
    gen = make_generator(
        "unused", loaded=(model, tokenizer, "cpu"), max_new_tokens=4, batch_size=batch_size
    )
    prompts = ["hi", "hello world", "the quick brown fox jumps", "hello"]
    assert gen.batch(prompts) == [gen(prompt) for prompt in prompts]
    assert gen.batch([]) == []
    assert tokenizer.padding_side == "right"


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_saved_multiple_return_config_preserves_scalar_results(arch, batch_size, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from soup_cli.utils.live_eval import make_generator
    from tests._tiny_hf_models import build_gpt2, build_llama

    torch.manual_seed(1759)
    tokenizer = tiny_tokenizer()
    model = (build_gpt2 if arch == "gpt2" else build_llama)(tokenizer).eval()
    model.generation_config.num_beams = 2
    model.generation_config.num_return_sequences = 2
    model.generation_config.save_pretrained(tmp_path)
    model.generation_config = transformers.GenerationConfig.from_pretrained(
        tmp_path, local_files_only=True
    )
    saved_config = model.generation_config.to_dict()
    assert saved_config["num_beams"] == saved_config["num_return_sequences"] == 2
    gen = make_generator(
        "unused", loaded=(model, tokenizer, "cpu"), max_new_tokens=4, batch_size=batch_size
    )
    prompts = ["hi", "hello world", "the quick brown fox jumps"]
    expected = [gen(prompt) for prompt in prompts]
    outputs = gen.batch(prompts)
    assert len(outputs) == len(prompts)
    assert outputs == expected
    tasks = [EvalTask(prompt, answer) for prompt, answer in zip(prompts, expected)]
    scalar_result = run_eval("unused", tasks, lambda prompt: gen(prompt))
    assert scalar_result.accuracy == 1
    assert run_eval("unused", tasks, gen) == scalar_result
    assert model.generation_config.to_dict() == saved_config


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {
        word: index
        for index, word in enumerate(
            [
                "<pad>",
                "<unk>",
                "<eos>",
                "hi",
                "hello",
                "world",
                "the",
                "quick",
                "brown",
                "fox",
                "jumps",
            ]
        )
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_bad_batch_size_refused_before_loading(value, monkeypatch):
    from soup_cli.utils import live_eval

    monkeypatch.setattr(live_eval, "load_model_and_tokenizer", lambda *a, **k: pytest.fail("load"))
    with pytest.raises(ValueError, match="batch_size"):
        live_eval.make_generator("unused", batch_size=value)


def test_live_batches_are_bounded_left_padded_and_slice_after_full_width():
    import torch

    from soup_cli.utils.live_eval import make_generator

    tokenizer = tiny_tokenizer()
    recorded = []

    class Model:
        def generate(self, **kwargs):
            recorded.append(kwargs)
            assert kwargs["do_sample"] is False
            inputs = kwargs["input_ids"]
            return torch.cat([inputs, torch.full((len(inputs), 1), 3)], dim=1)

    gen = make_generator("unused", loaded=(Model(), tokenizer, "cpu"), batch_size=2)
    assert gen.batch(["hi", "hello world", "the quick brown", "hi", "world"]) == ["hi"] * 5
    assert [len(call["input_ids"]) for call in recorded] == [2, 2, 1]
    assert recorded[0]["input_ids"].tolist() == [[0, 3], [4, 5]]
    assert recorded[0]["attention_mask"].tolist() == [[0, 1], [1, 1]]
    assert recorded[1]["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]


@pytest.mark.parametrize("suite", DEFAULT_GENERAL_SUITE)
def test_batch_exceptions_fall_back_and_failed_items_never_pass(suite):
    class Generator:
        def __call__(self, prompt):
            raise RuntimeError("failed item")

        def batch(self, prompts):
            raise RuntimeError("failed batch")

    assert score_bundled_suite(suite, Generator()) == 0


@pytest.mark.parametrize("thinking", [False, True])
def test_batch_uses_scalar_chat_rendering_and_option_snapshot(thinking):
    import torch

    from soup_cli.utils.live_eval import encode_prompt, make_generator

    tokenizer = tiny_tokenizer()
    tokenizer.chat_template = (
        "{{ 'the ' if enable_thinking else '' }}{{ messages[0]['content'] }} world"
    )
    recorded = []

    class Model:
        def generate(self, input_ids, attention_mask, **kwargs):
            recorded.extend(
                [row[mask.bool()].tolist() for row, mask in zip(input_ids, attention_mask)]
            )
            return torch.cat([input_ids, torch.full((len(input_ids), 1), 3)], dim=1)

    options = {"enable_thinking": thinking}
    gen = make_generator(
        "unused", loaded=(Model(), tokenizer, "cpu"), chat_template_kwargs=options, batch_size=2
    )
    options["enable_thinking"] = not thinking
    prompts = ["hi", "hello world"]
    assert gen.batch(prompts) == ["hi", "hi"]
    assert recorded == [
        encode_prompt(
            tokenizer,
            prompt,
            chat_template_kwargs={"enable_thinking": thinking},
            return_tensors=None,
        )["input_ids"]
        for prompt in prompts
    ]
