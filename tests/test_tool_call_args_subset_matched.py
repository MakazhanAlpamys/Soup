"""#2811: a ``tool_call_args_subset`` row counts as matched only at full credit.

The function name is worth 0.5 and the arguments are worth 0.5, so a ``>= 0.5``
threshold counted a call with every argument wrong, with no arguments, or to the
wrong tool as Correct. ``soup eval custom``, ``soup ship`` metric mode and the
retention check report Accuracy from ``matched``, so a model that never got one
argument right read 100%. A row is now matched only when the score is 1.0; the
partial ``score`` is unchanged, because the eval gate, ``quant_check``,
``deploy_measure`` and ``cmaes_merge`` average it.

Put ``>= 0.5`` back in ``score_task`` and the tests below fail.
"""

import json
import os

import pytest

from soup_cli.eval.custom import EvalTask, run_eval

_OMIT = object()


def call(name: str, arguments: object = _OMIT) -> str:
    function: dict = {"name": name}
    if arguments is not _OMIT:
        function["arguments"] = arguments
    return json.dumps({"function": function})


GOLD_WITH_ARGS = call("get_weather", {"city": "Paris", "unit": "celsius"})
GOLD_NO_ARGS = call("get_time", {})


def _run(gold: str, outputs: list[str]):
    tasks = [
        EvalTask(prompt=str(i), expected=gold, scoring="tool_call_args_subset")
        for i in range(len(outputs))
    ]
    replies = iter(outputs)
    return run_eval("unused", tasks, generate_fn=lambda prompt: next(replies))


@pytest.mark.parametrize(
    "output, score, matched",
    [
        (call("get_weather", {"city": "Paris", "unit": "celsius"}), 1.0, True),
        (call("get_weather", {"city": "Paris", "unit": "kelvin"}), 0.75, False),
        (call("get_weather", {"city": "Tokyo", "unit": "kelvin"}), 0.5, False),
        (call("get_weather"), 0.5, False),
        (call("delete_account", {"city": "Paris", "unit": "celsius"}), 0.5, False),
        (call("delete_account"), 0.0, False),
        # Subset matching: every expected argument is right, an extra one is allowed.
        (call("get_weather", {"city": "Paris", "unit": "celsius", "days": 3}), 1.0, True),
        (call("get_weather", "not-json"), 0.5, False),
    ],
    ids=[
        "both-args-right", "one-arg-right", "every-arg-wrong", "no-arguments",
        "wrong-tool-args-right", "wrong-tool-no-arguments", "right-tool-extra-arg",
        "right-tool-malformed-args",
    ],
)
def test_gold_with_arguments(output, score, matched):
    result = _run(GOLD_WITH_ARGS, [output]).results[0]
    assert result.score == pytest.approx(score)
    assert result.matched is matched


@pytest.mark.parametrize(
    "output, score, matched",
    [
        (call("get_time", {}), 1.0, True),
        (call("get_time"), 1.0, True),  # an omitted `arguments` keeps its #1734 treatment
        (call("delete_account", {}), 0.5, False),
        (call("get_time", {"zone": "UTC"}), 0.5, False),
        (call("get_time", "not-json"), 0.5, False),
    ],
    ids=[
        "right-tool-empty-args", "right-tool-omitted-args", "wrong-tool-no-arguments",
        "right-tool-extra-args", "right-tool-malformed-args",
    ],
)
def test_gold_without_arguments(output, score, matched):
    result = _run(GOLD_NO_ARGS, [output]).results[0]
    assert result.score == pytest.approx(score)
    assert result.matched is matched


def test_every_argument_wrong_is_zero_accuracy_with_half_mean_score():
    wrong = call("get_weather", {"city": "Tokyo", "unit": "kelvin"})
    results = _run(GOLD_WITH_ARGS, [wrong] * 4)

    assert results.correct == 0
    assert results.accuracy == 0.0
    assert sum(r.score for r in results.results) / results.total == pytest.approx(0.5)


def test_the_eval_gate_still_averages_the_partial_score(tmp_path, monkeypatch):
    from soup_cli.eval.gate import GateTask, _run_custom_task

    monkeypatch.chdir(tmp_path)
    (tmp_path / "tools.jsonl").write_text(
        "\n".join(
            json.dumps({"prompt": str(i), "expected": GOLD_WITH_ARGS,
                        "scoring": "tool_call_args_subset"})
            for i in range(4)
        ),
        encoding="utf-8",
    )
    task = GateTask(type="custom", name="tools", threshold=0.0, tasks=os.fspath("tools.jsonl"))
    wrong = call("get_weather", {"city": "Tokyo", "unit": "kelvin"})

    assert _run_custom_task(task, lambda prompt: wrong) == pytest.approx(0.5)
