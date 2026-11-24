"""Batch-side list mutation must not change evaluation membership or retry order."""

from __future__ import annotations

import pytest

from soup_cli.eval.custom import EvalTask, run_eval
from soup_cli.eval.forgetting import ForgettingDetector
from soup_cli.eval.gate_suites import _fingerprint_generator, score_bundled_suite
from soup_cli.eval.generation import iter_outputs


class MutatingBatch:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.scalar_calls: list[str] = []
        self.batch_calls = 0

    def __call__(self, prompt: str) -> str:
        self.scalar_calls.append(prompt)
        return {"one": "A", "two": "wrong"}[prompt]

    def batch(self, prompts: list[str]) -> list[str]:
        self.batch_calls += 1
        if self.mode == "pop_short":
            prompts.pop()
            return ["A"]
        if self.mode == "short":
            return ["A"]
        if self.mode == "healthy":
            return [{"one": "A", "two": "wrong"}[prompt] for prompt in prompts]
        if self.mode == "consume":
            outputs = []
            while prompts:
                outputs.append({"one": "A", "two": "wrong"}[prompts.pop(0)])
            return outputs
        if self.mode == "clear_raise":
            prompts.clear()
        elif self.mode == "reverse_raise":
            prompts.reverse()
        else:
            raise AssertionError(self.mode)
        raise RuntimeError("retry original prompts")


def test_short_result_after_pop_fails_every_original_item() -> None:
    prompts = ["one", "two"]
    generator = MutatingBatch("pop_short")
    assert list(iter_outputs(generator, prompts)) == [None, None]
    assert prompts == ["one", "two"]
    assert generator.batch_calls == 1
    assert generator.scalar_calls == []


@pytest.mark.parametrize("mode", ["clear_raise", "reverse_raise"])
def test_batch_exception_retries_original_prompts_in_order(mode: str) -> None:
    prompts = ["one", "two"]
    generator = MutatingBatch(mode)
    assert list(iter_outputs(generator, prompts)) == ["A", "wrong"]
    assert generator.scalar_calls == ["one", "two"]
    assert prompts == ["one", "two"]


def test_custom_eval_pop_keeps_two_failed_results_and_the_denominator() -> None:
    tasks = [EvalTask("one", "A", "first"), EvalTask("two", "B", "second")]
    result = run_eval("unused", tasks, MutatingBatch("pop_short"))
    assert result.total == 2
    assert len(result.results) == len(tasks)
    assert [item.task for item in result.results] == tasks
    assert [item.score for item in result.results] == [0.0, 0.0]
    assert not any(item.matched for item in result.results)
    assert result.accuracy == 0.0
    assert result.category_scores["first"]["total"] == 1
    assert result.category_scores["second"]["total"] == 1


@pytest.mark.parametrize("mode", ["clear_raise", "reverse_raise"])
def test_custom_eval_retry_keeps_original_tasks_scores_and_order(mode: str) -> None:
    tasks = [EvalTask("one", "A"), EvalTask("two", "B")]
    generator = MutatingBatch(mode)
    result = run_eval("unused", tasks, generator)
    assert result.total == 2
    assert [item.task for item in result.results] == tasks
    assert [item.output for item in result.results] == ["A", "wrong"]
    assert [item.score for item in result.results] == [1.0, 0.0]
    assert result.accuracy == 0.5
    assert generator.scalar_calls == ["one", "two"]


def test_complete_batch_may_consume_its_private_prompt_list() -> None:
    prompts = ["one", "two"]
    generator = MutatingBatch("consume")
    assert list(iter_outputs(generator, prompts)) == ["A", "wrong"]
    assert prompts == ["one", "two"]
    assert generator.scalar_calls == []


def test_healthy_batch_and_scalar_results_remain_identical() -> None:
    tasks = [EvalTask("one", "A"), EvalTask("two", "B")]
    generator = MutatingBatch("healthy")
    batched = run_eval("unused", tasks, generator)
    scalar = run_eval("unused", tasks, MutatingBatch("healthy").__call__)
    assert batched == scalar
    assert batched.total == 2
    assert batched.accuracy == 0.5
    assert generator.batch_calls == 1
    assert generator.scalar_calls == []


def test_empty_input_calls_neither_batch_nor_scalar() -> None:
    generator = MutatingBatch("pop_short")
    assert list(iter_outputs(generator, [])) == []
    assert generator.batch_calls == 0
    assert generator.scalar_calls == []


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("failure", ["raise", "non_string", "none"])
def test_failed_item_cannot_match_an_empty_expected_answer(
    batched: bool, failure: str,
) -> None:
    class PerItemFailure:
        def __call__(self, prompt: str) -> object:
            if prompt == "two":
                return "B"
            if failure == "raise":
                raise RuntimeError("failed item")
            return None if failure == "none" else 17

        def batch(self, prompts: list[str]) -> list[object]:
            if failure == "raise":
                raise RuntimeError("retry scalar items")
            return [None if failure == "none" else 17, "B"]

    generator = PerItemFailure()
    result = run_eval(
        "unused", [EvalTask("one", ""), EvalTask("two", "B")],
        generator if batched else generator.__call__,
    )
    assert result.total == 2
    assert [item.score for item in result.results] == [0.0, 1.0]
    assert result.accuracy == 0.5


def test_unmutated_short_result_still_fails_the_batch_without_retry() -> None:
    generator = MutatingBatch("short")
    assert list(iter_outputs(generator, ["one", "two"])) == [None, None]
    assert generator.scalar_calls == []


def test_non_callable_is_still_refused_even_for_empty_input() -> None:
    with pytest.raises(TypeError, match="gen must be callable"):
        list(iter_outputs(None, []))


def test_bundled_suite_does_not_award_a_short_mutating_batch() -> None:
    class ShortBatch:
        def __call__(self, prompt: str) -> str:
            raise AssertionError("a malformed batch is not retried")

        def batch(self, prompts: list[str]) -> list[str]:
            prompts.pop()
            return [_fingerprint_generator(prompt) for prompt in prompts]

    assert score_bundled_suite("mini_over_refusal", ShortBatch()) == 0.0


def test_scalar_iteration_remains_lazy_between_outputs() -> None:
    calls: list[str] = []

    def scalar(prompt: str) -> str:
        calls.append(prompt)
        return prompt.upper()

    outputs = iter_outputs(scalar, ["one", "two"])
    assert calls == []
    assert next(outputs) == "ONE"
    assert calls == ["one"]
    assert next(outputs) == "TWO"
    assert calls == ["one", "two"]
    with pytest.raises(StopIteration):
        next(outputs)


def test_forgetting_does_not_award_a_short_mutating_batch() -> None:
    class ShortBatch:
        def __call__(self, prompt: str) -> str:
            raise AssertionError("a malformed batch is not retried")

        def batch(self, prompts: list[str]) -> list[str]:
            prompts.pop()
            return ["A"] * len(prompts)

    detector = ForgettingDetector(ShortBatch())
    assert detector.run_baseline() == 0.0


@pytest.mark.parametrize("mode", ["pop_short", "clear_raise", "reverse_raise", "consume"])
def test_mutated_middle_chunk_preserves_membership_retry_order_and_other_chunks(
    mode: str,
) -> None:
    prompts = ["one", "two", "three", "four", "five"]

    class ChunkMutator:
        batch_size = 2

        def __init__(self) -> None:
            self.batches: list[list[str]] = []
            self.scalar_calls: list[str] = []

        def __call__(self, prompt: str) -> str:
            self.scalar_calls.append(prompt)
            return prompt.upper()

        def batch(self, values: list[str]) -> list[str]:
            assert isinstance(values, list)
            self.batches.append(list(values))
            if values == ["three", "four"]:
                if mode == "pop_short":
                    values.pop()
                    return [value.upper() for value in values]
                if mode == "consume":
                    result = []
                    while values:
                        result.append(values.pop(0).upper())
                    return result
                if mode == "clear_raise":
                    values.clear()
                else:
                    values.reverse()
                raise RuntimeError("retry the original middle chunk")
            return [value.upper() for value in values]

    generator = ChunkMutator()
    tasks = [EvalTask(prompt, prompt.upper()) for prompt in prompts]
    result = run_eval("unused", tasks, generator)
    assert result.total == 5
    assert [item.task for item in result.results] == tasks
    assert generator.batches == [["one", "two"], ["three", "four"], ["five"]]
    if mode == "pop_short":
        assert result.accuracy == 0.6
        assert [item.score for item in result.results] == [1.0, 1.0, 0.0, 0.0, 1.0]
        assert generator.scalar_calls == []
    else:
        assert result.accuracy == 1.0
        assert [item.output for item in result.results] == [prompt.upper() for prompt in prompts]
        expected_retries = ["three", "four"] if mode.endswith("raise") else []
        assert generator.scalar_calls == expected_retries
