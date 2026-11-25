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


@pytest.mark.parametrize("json_output", [False, True])
def test_exact_output_matches_the_pre_lexical_release_bytes(tmp_path, monkeypatch, json_output):
    from soup_cli.commands.data import console

    monkeypatch.setattr(console, "legacy_windows", False)
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("NO_COLOR", "1")
    train = [{"prompt": "same input", "response": "old"}, {"prompt": "same input"}]
    evaluation = [{"prompt": "same input", "response": "new"}, {"prompt": "other input"}]
    options = ("--json",) if json_output else ()
    result = _cli(tmp_path, monkeypatch, train, evaluation, *options)
    if json_output:
        expected = (
            '{"schema_version": 1, "matching": "exact extracted input text; outer whitespace '
            'stripped", "row_numbers": "one-based records, excluding blank lines", '
            '"train_rows": 2, "eval_rows": 2, "overlap_eval_rows": 1, "overlap_percent": 50.0, '
            '"samples": [{"train_row": 1, "eval_row": 1, "train_matches": 2, '
            '"prompt_preview": "same input"}]}\n'
        )
    else:
        expected = (
            "  Train/eval input-text overlap  \n"
            "┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓\n"
            "┃ Metric               ┃ Value  ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩\n"
            "│ Train records        │ 2      │\n"
            "│ Eval records         │ 2      │\n"
            "│ Matched eval records │ 1      │\n"
            "│ Overlap              │ 50.00% │\n"
            "└──────────────────────┴────────┘\n"
            "Train record 1 / eval record 1: same input\n"
        )
    assert result.exit_code == 2
    assert result.stdout_bytes == expected.encode("utf-8")
    assert result.stderr_bytes == b""


def test_lexical_matches_count_eval_rows_once_and_preserve_inputs(tmp_path, monkeypatch):
    pytest.importorskip("datasketch")
    base = (
        "please send a concise project status update with the latest schedule and a brief "
        "note about pending reviews"
    )
    near = base.removesuffix("reviews") + "checks"
    train = [{"prompt": text} for text in (base, base, base.upper(), "unrelated geology")]
    evaluation = [
        {"prompt": base, "response": "different answer"},
        {"prompt": near},
        {"prompt": "describe the migration route of the arctic tern"},
    ]
    exact = _cli(tmp_path, monkeypatch, train, evaluation, "--json")
    result = _cli(tmp_path, monkeypatch, train, evaluation, "--near-duplicates", "--json")
    assert exact.exit_code == result.exit_code == 2
    exact_report = json.loads(exact.stdout)
    assert exact_report["overlap_eval_rows"] == 1
    assert "method" not in exact_report["samples"][0]
    assert "score" not in exact_report["samples"][0]

    report = json.loads(result.stdout)
    assert report["overlap_eval_rows"] == 2
    assert report["overlap_percent"] == pytest.approx(200 / 3)
    exact_pair, near_pair = report["samples"]
    assert (exact_pair["eval_row"], exact_pair["method"], exact_pair["score"]) == (1, "exact", 1.0)
    assert (near_pair["eval_row"], near_pair["train_row"], near_pair["method"]) == (2, 1, "near")
    assert near_pair["train_matches"] == 3
    assert near_pair["score"] == pytest.approx((len(base.split()) - 3) / (len(base.split()) - 1))


def test_lexical_filters_candidates_by_exact_jaccard(tmp_path, monkeypatch):
    datasketch = pytest.importorskip("datasketch")
    monkeypatch.setattr(datasketch.MinHashLSH, "query", lambda *args: [1])
    result = _cli(
        tmp_path, monkeypatch, [{"text": "red green blue"}], [{"text": "one two three"}],
        "--near-duplicates", "--json",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["samples"] == []


def test_lexical_ties_use_earliest_train_row(tmp_path, monkeypatch):
    datasketch = pytest.importorskip("datasketch")
    monkeypatch.setattr(datasketch.MinHashLSH, "query", lambda *args: [2, 1])
    result = _cli(
        tmp_path, monkeypatch, [{"text": "one Two three"}, {"text": "One two three"}],
        [{"text": "ONE TWO THREE"}], "--near-duplicates", "--json",
    )
    assert result.exit_code == 2, result.output
    pair = json.loads(result.stdout)["samples"][0]
    assert (pair["train_row"], pair["train_matches"], pair["score"]) == (1, 2, 1.0)


@pytest.mark.parametrize(
    "threshold,permutations,seed",
    [("0.0001", "16", "0"), ("1", "256", "4294967295"), ("1", "16", "1")],
)
def test_lexical_option_valid_endpoints(tmp_path, monkeypatch, threshold, permutations, seed):
    pytest.importorskip("datasketch")
    result = _cli(
        tmp_path, monkeypatch, [{"text": "one two three"}], [{"text": "ONE TWO THREE"}],
        "--near-duplicates", "--threshold", threshold, "--permutations", permutations,
        "--seed", seed, "--json",
    )
    assert result.exit_code == 2, result.output
    report = json.loads(result.stdout)
    assert report["near_duplicates"] == {
        "threshold": float(threshold), "permutations": int(permutations), "seed": int(seed),
    }
    assert report["samples"][0]["score"] == 1.0


@pytest.mark.parametrize(
    "option,value,expected",
    [
        ("--threshold", "0", "greater than 0 and at most 1"),
        ("--threshold", "1.01", "greater than 0 and at most 1"),
        ("--threshold", "nan", "greater than 0 and at most 1"),
        ("--threshold", "invalid", "greater than 0 and at most 1"),
        ("--permutations", "0", "16 to 256"),
        ("--permutations", "15", "16 to 256"),
        ("--permutations", "257", "16 to 256"),
        ("--permutations", "16.5", "16 to 256"),
        ("--seed", "-1", "0 to 4294967295"),
        ("--seed", "1.5", "0 to 4294967295"),
        ("--seed", "4294967296", "0 to 4294967295"),
    ],
)
def test_invalid_lexical_options_name_the_range(tmp_path, monkeypatch, option, value, expected):
    result = _cli(
        tmp_path, monkeypatch, [{"text": "a"}], [{"text": "b"}],
        "--near-duplicates", option, value,
    )
    assert result.exit_code == 3, result.output
    assert option in _plain(result.output)
    assert expected in _plain(result.output)


def test_lexical_options_require_opt_in(tmp_path, monkeypatch):
    result = _cli(tmp_path, monkeypatch, [{"text": "a"}], [{"text": "b"}], "--seed", "1")
    assert result.exit_code == 3
    assert "require --near-duplicates" in _plain(result.output)


@pytest.mark.parametrize("option", ["--train", "--eval"])
@pytest.mark.parametrize("over_limit", [False, True])
def test_lexical_record_limit(tmp_path, monkeypatch, option, over_limit):
    from soup_cli.utils.data_leakage import NEAR_MAX_ROWS

    pytest.importorskip("datasketch")
    row = {"text": "example"}
    limited = [row] * (NEAR_MAX_ROWS + over_limit)
    train, evaluation = (limited, [row]) if option == "--train" else ([row], limited)
    result = _cli(tmp_path, monkeypatch, train, evaluation, "--near-duplicates", "--json")
    assert result.exit_code == (3 if over_limit else 2), result.output
    if over_limit:
        assert f"{option} in --near-duplicates mode: file exceeds {NEAR_MAX_ROWS} rows" in (
            _plain(result.output)
        )
        assert "smaller split or remove --near-duplicates" in _plain(result.output)


@pytest.mark.parametrize("over_limit", [False, True])
def test_lexical_byte_limit(tmp_path, monkeypatch, over_limit):
    from soup_cli.utils import data_leakage

    pytest.importorskip("datasketch")
    row = {"text": "q", "unused": ""}
    limit = len(json.dumps(row).encode("utf-8")) + 1
    monkeypatch.setattr(data_leakage, "NEAR_MAX_FILE_BYTES", limit)
    result = _cli(
        tmp_path, monkeypatch, [{"text": "q"}], [{**row, "unused": "x" * over_limit}],
        "--near-duplicates", "--json",
    )
    assert result.exit_code == (3 if over_limit else 2), result.output
    if over_limit:
        assert f"--eval in --near-duplicates mode: file exceeds {limit} bytes" in (
            _plain(result.output)
        )
        assert "smaller split or remove --near-duplicates" in _plain(result.output)


@pytest.mark.parametrize("work_budget", [0, 1])
def test_lexical_candidate_work_budget(tmp_path, monkeypatch, work_budget):
    from soup_cli.utils import data_leakage

    pytest.importorskip("datasketch")
    monkeypatch.setattr(data_leakage, "_NEAR_MAX_JACCARD_WORK", work_budget)
    result = _cli(
        tmp_path, monkeypatch, [{"text": "q"}], [{"text": "Q"}], "--near-duplicates", "--json",
    )
    assert result.exit_code == (2 if work_budget == 1 else 3), result.output
    if work_budget == 0:
        assert "shingle comparisons" in _plain(result.output)
        assert "smaller split or remove --near-duplicates" in _plain(result.output)


def test_lexical_rejects_missing_input_text(tmp_path, monkeypatch):
    pytest.importorskip("datasketch")
    result = _cli(
        tmp_path, monkeypatch, [{"text": "q"}], [{"output": "answer only"}],
        "--near-duplicates",
    )
    assert result.exit_code == 3
    assert _plain(result.output) == (
        "Leakage input error: eval record 1: record has no usable input text"
    )


def test_datasketch_required_only_in_lexical_mode(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "datasketch", None)
    train, evaluation = [{"text": "q"}], [{"text": "q"}]
    assert _cli(tmp_path, monkeypatch, train, evaluation, "--json").exit_code == 2
    result = _cli(tmp_path, monkeypatch, train, evaluation, "--near-duplicates")
    assert result.exit_code == 3
    assert 'pip install "soup-cli[data]"' in _plain(result.output)
