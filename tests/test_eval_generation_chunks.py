"""Scoring a completed chunk precedes generation of the next chunk."""

import io
import json

import pytest

from soup_cli.eval.custom import EvalTask, run_eval
from soup_cli.eval.gate_suites import _fraction_passing
from soup_cli.eval.generation import iter_outputs
from soup_cli.utils.ship_progress import group, session


def test_suite_progress_is_emitted_before_the_next_chunk_starts():
    stream = io.StringIO()
    starts = []

    class Generator:
        batch_size = 2

        def __call__(self, prompt):
            pytest.fail("successful batches must not be retried")

        def batch(self, prompts):
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
            starts.append((list(prompts), sum(event["event"] == "progress" for event in events)))
            return list(prompts)

    items = tuple({"prompt": str(index)} for index in range(5))
    with session(stream), group("suite_eval", side="tuned", suite="tiny", countable=True):
        score = _fraction_passing(items, Generator(), lambda item, text: item["prompt"] == text)

    assert score == 1.0
    assert starts == [(["0", "1"], 0), (["2", "3"], 2), (["4"], 4)]
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    progress = [event for event in events if event["event"] == "progress"]
    assert [event["done"] for event in progress] == [1, 2, 3, 4, 5]
    assert all(event["total"] == 5 and event["schema_version"] == 1 for event in progress)


@pytest.mark.parametrize("hint", [0, -1, True, 2.0, "2", None])
def test_invalid_batch_size_hint_keeps_one_batch_call(hint):
    calls = []

    class Generator:
        batch_size = hint

        def __call__(self, prompt):
            pytest.fail("successful batches must not be retried")

        def batch(self, prompts):
            calls.append(list(prompts))
            return list(prompts)

    prompts = ["a", "b", "c"]
    assert list(iter_outputs(Generator(), prompts)) == prompts
    assert calls == [prompts]


def test_failed_chunk_retries_its_items_without_repeating_successful_chunks():
    calls = []

    class Generator:
        batch_size = 2

        def __call__(self, prompt):
            calls.append(("scalar", prompt))
            if prompt == "bad":
                raise ValueError("failed item")
            return prompt

        def batch(self, prompts):
            calls.append(("batch", list(prompts)))
            if "bad" in prompts:
                raise ValueError("failed chunk")
            return list(prompts)

    assert list(iter_outputs(Generator(), ["a", "b", "bad", "d", "e"])) == [
        "a", "b", None, "d", "e",
    ]
    assert calls == [("batch", ["a", "b"]), ("batch", ["bad", "d"]),
                     ("scalar", "bad"), ("scalar", "d"), ("batch", ["e"])]


def test_bad_chunk_shape_does_not_discard_or_shift_other_chunks():
    class Generator:
        batch_size = 2

        def __call__(self, prompt):
            pytest.fail("malformed batch results cannot be aligned for scalar retry")

        def batch(self, prompts):
            return ["wrong length"] if prompts[0] == "bad" else list(prompts)

    assert list(iter_outputs(Generator(), ["a", "b", "bad", "d", "e"])) == [
        "a", "b", None, None, "e",
    ]


def test_custom_eval_keeps_order_results_and_scores_across_chunks():
    tasks = [EvalTask(str(index), str(index)) for index in range(5)]

    class Generator:
        batch_size = 2

        def __call__(self, prompt):
            return prompt

        def batch(self, prompts):
            return list(prompts)

    assert run_eval("unused", tasks, Generator()) == run_eval("unused", tasks, lambda text: text)


def test_live_generator_exposes_its_validated_batch_size():
    pytest.importorskip("torch")
    from soup_cli.utils.live_eval import make_generator

    class Tokenizer:
        pad_token_id = 0

    generator = make_generator("unused", loaded=(object(), Tokenizer(), "cpu"), batch_size=3)
    assert generator.batch_size == 3


def test_forgetting_keeps_the_same_accuracy_across_chunks():
    from soup_cli.eval.forgetting import ForgettingDetector

    class Generator:
        batch_size = 2

        def __call__(self, prompt):
            return "A"

        def batch(self, prompts):
            return ["A"] * len(prompts)

    scalar = ForgettingDetector(lambda prompt: "A").run_baseline()
    assert ForgettingDetector(Generator()).run_baseline() == scalar
