"""Exact input-text overlap is a read-only gate, including short questions."""

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi


def _plain(text):
    return " ".join(strip_ansi(text).split())


def _check(train, evaluation, **kwargs):
    from soup_cli.utils.data_leakage import check_leakage

    return check_leakage(train, evaluation, **kwargs)


def _cli(tmp_path, monkeypatch, train, evaluation, *extra):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    for name, rows in (("train.jsonl", train), ("eval.jsonl", evaluation)):
        Path(name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = CliRunner().invoke(
        app, ["data", "leakage", "--train", "train.jsonl", "--eval", "eval.jsonl", *extra]
    )
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    return result


def test_short_prompt_overlap_ignores_different_answers():
    train = [{"prompt": "2+2?", "response": "four"}]
    evaluation = [{"instruction": "2+2?", "output": "4"}]
    before = copy.deepcopy((train, evaluation))
    report = _check(train, evaluation)
    assert report["overlap_eval_rows"] == 1
    assert report["overlap_percent"] == 100.0
    assert report["samples"][0]["train_row"] == 1
    assert (train, evaluation) == before


def test_duplicates_do_not_multiply_the_eval_row_denominator():
    report = _check([{"text": "q"}] * 3, [{"text": "q"}, {"text": "q"}, {"text": "other"}])
    assert report["train_rows"] == 3
    assert report["eval_rows"] == 3
    assert report["overlap_eval_rows"] == 2
    assert report["overlap_percent"] == pytest.approx(200 / 3)
    assert report["samples"][0]["train_matches"] == 3


@pytest.mark.parametrize(
    "row",
    [
        {"prompt": "q", "chosen": "answer A", "rejected": "answer B"},
        {"instruction": "q", "output": "answer"},
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "answer"}]},
        {
            "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A"}],
            "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "B"}],
        },
        {"text": "q"},
        {"content": "q"},
        {"query": "q", "answer": "a"},
    ],
)
def test_supported_prompt_formats(row):
    assert _check([row], [{"prompt": "q"}])["overlap_eval_rows"] == 1


def test_multiturn_history_is_part_of_the_prompt():
    def chat(history, response):
        return {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": history},
                {"role": "user", "content": "last"},
                {"role": "assistant", "content": response},
            ]
        }

    assert (
        _check([chat("context A", "answer")], [chat("context B", "answer")])["overlap_eval_rows"]
        == 0
    )
    assert _check([chat("context A", "one")], [chat("context A", "two")])["overlap_eval_rows"] == 1


def test_exact_matching_preserves_case_and_internal_whitespace():
    report = _check([{"prompt": " q "}], [{"prompt": "q"}, {"prompt": "Q"}, {"prompt": "q  x"}])
    assert report["overlap_eval_rows"] == 1


@pytest.mark.parametrize(
    "train,evaluation,reason",
    [
        ([], [{"text": "q"}], "train and eval must both contain records"),
        ([{"text": "q"}], [], "train and eval must both contain records"),
        ([{}], [{"text": "q"}], "train record 1: record has no usable input text"),
        ([{"prompt": ""}], [{"text": "q"}], "train record 1: record has no usable input text"),
        ([{"text": "q"}], [42], "eval record 1: record must be a JSON object"),
        (
            [{"output": "response only"}], [{"text": "q"}],
            "train record 1: record has no usable input text",
        ),
    ],
)
def test_unusable_input_is_refused(train, evaluation, reason):
    with pytest.raises(ValueError) as error:
        _check(train, evaluation)
    assert _plain(str(error.value)) == reason


@pytest.mark.parametrize("samples", [-1, 21, True, 1.5])
def test_sample_bound_is_validated(samples):
    with pytest.raises(ValueError) as error:
        _check([{"text": "q"}], [{"text": "q"}], samples=samples)
    assert _plain(str(error.value)) == "samples must be an integer in [0, 20]"


def test_examples_are_capped_and_ordered_by_eval_record():
    rows = [{"text": str(i)} for i in range(30)]
    report = _check(rows, rows, samples=2)
    assert report["overlap_eval_rows"] == 30
    assert len(report["samples"]) == 2
    assert [x["eval_row"] for x in report["samples"]] == [1, 2]
    assert not _check(rows, rows, samples=0)["samples"]


def test_overlap_cli_returns_json_and_exit_two_without_writes(tmp_path, monkeypatch):
    result = _cli(
        tmp_path,
        monkeypatch,
        [{"prompt": "q", "response": "A"}],
        [{"prompt": "q", "response": "B"}],
        "--json",
    )
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["overlap_eval_rows"] == 1


def test_clean_cli_returns_zero_and_json(tmp_path, monkeypatch):
    result = _cli(tmp_path, monkeypatch, [{"text": "train"}], [{"text": "eval"}], "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["overlap_percent"] == 0.0


def test_cli_table_includes_percentage(tmp_path, monkeypatch):
    result = _cli(tmp_path, monkeypatch, [{"text": "q"}], [{"text": "q"}, {"text": "other"}])
    assert result.exit_code == 2, result.output
    assert "50.00%" in _plain(result.output)


@pytest.mark.parametrize(
    "broken,reason",
    [
        ('{"text":', "line 1 is not valid JSON"),
        ("42\n", "line 1 must be a JSON object"),
        ("{}\n", "train record 1: record has no usable input text"),
        ("\n", "train and eval must both contain records"),
    ],
)
def test_bad_input_cli_returns_three(tmp_path, monkeypatch, broken, reason):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    Path("train.jsonl").write_text(broken)
    Path("eval.jsonl").write_text('{"text":"q"}\n')
    result = CliRunner().invoke(
        app, ["data", "leakage", "--train", "train.jsonl", "--eval", "eval.jsonl", "--json"]
    )
    assert result.exit_code == 3, result.output
    assert _plain(result.output) == f"Leakage input error: {reason}"


def test_sample_option_misuse_uses_gate_exit_three(tmp_path, monkeypatch):
    result = _cli(tmp_path, monkeypatch, [{"text": "a"}], [{"text": "b"}], "--samples", "-1")
    assert result.exit_code == 3, result.output
    assert (
        "Invalid value for '--samples': -1 is not in the range 0<=x<=20."
        in _plain(result.output)
    )


def test_cli_refuses_a_file_outside_cwd(tmp_path, monkeypatch):
    from soup_cli.cli import app

    Path(tmp_path / "outside.jsonl").write_text('{"text":"q"}\n')
    (tmp_path / "inside").mkdir()
    monkeypatch.chdir(tmp_path / "inside")
    Path("eval.jsonl").write_text('{"text":"q"}\n')
    result = CliRunner().invoke(
        app, ["data", "leakage", "--train", "../outside.jsonl", "--eval", "eval.jsonl"]
    )
    assert result.exit_code == 3
    assert _plain(result.output) == "Leakage input error: input path must stay under cwd"


def test_mismatched_preference_prompts_are_not_silently_compared():
    row = {
        "chosen": [{"role": "user", "content": "one"}],
        "rejected": [{"role": "user", "content": "two"}],
    }
    with pytest.raises(ValueError) as error:
        _check([row], [{"text": "one"}])
    assert _plain(str(error.value)) == (
        "train record 1: chosen/rejected conversations do not have the same prompt"
    )


def test_json_remains_parseable_with_long_rich_like_text(tmp_path, monkeypatch):
    text = "[red]:warning:[/red] " * 20
    result = _cli(tmp_path, monkeypatch, [{"prompt": text}], [{"prompt": text}], "--json")
    assert result.exit_code == 2
    assert json.loads(result.stdout)["samples"][0]["prompt_preview"] == text.strip()[:160]


def test_table_preserves_literal_markup_and_emoji(tmp_path, monkeypatch):
    text = "[red]:warning:[/red]"
    result = _cli(tmp_path, monkeypatch, [{"prompt": text}], [{"prompt": text}])
    assert result.exit_code == 2
    assert text in strip_ansi(result.stdout)
