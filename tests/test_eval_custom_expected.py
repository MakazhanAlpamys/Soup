"""A custom eval task row must state its ``expected`` (#3240).

A row with no ``expected``, a ``null`` one, or an empty or whitespace-only one used to
load (``answer`` already refused a blank one). With no ``expected`` or an empty one,
``contains`` matched every output (``score_contains`` is True for an empty needle) and
``regex`` matched an empty pattern anywhere, so the row scored 1.0 for any answer, an
empty one included, and an eval gate built on it could never regress; ``contains``
also stripped a whitespace-only one to nothing. ``null`` loaded as the string
``"None"`` and matched any output containing that word.

The loader now refuses such a row under every ``scoring``, naming the line, for every
caller that scores ``expected``. ``soup ship --task-mode judge_score``, ``pairwise``
and ``reward`` accept a prompt-only file (``require_expected=False``): the judge modes
take a separately validated reference, and the reward function receives ``expected``
as ``answer``, ``""`` when the row has none. Values that are answers (``0``, ``false``, ``"0"``,
``4``) keep loading as ``str()`` of the value, as before.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval import custom
from soup_cli.utils.exit_codes import EXIT_USAGE_ERROR
from tests.conftest import strip_ansi

runner = CliRunner()

MISSING = "field 'expected' is missing"
NULL = "field 'expected' is null"
BLANK = "field 'expected' is empty or whitespace-only"
_ABSENT = object()

#: (id, value of ``expected`` or _ABSENT, message)
REFUSED = [
    pytest.param(_ABSENT, MISSING, id="missing"),
    pytest.param(None, NULL, id="null"),
    pytest.param("", BLANK, id="empty"),
    pytest.param("   ", BLANK, id="spaces"),
    pytest.param("\t\n", BLANK, id="tab-newline"),
]


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output."""
    return " ".join(strip_ansi(text).split())


def _row(expected: Any, scoring: str = "contains", prompt: str = "q") -> dict:
    row: dict = {"prompt": prompt, "scoring": scoring}
    if expected is not _ABSENT:
        row["expected"] = expected
    return row


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scoring", sorted(custom.VALID_SCORING))
@pytest.mark.parametrize(("expected", "message"), REFUSED)
def test_every_scoring_refuses_a_row_without_an_expected(
    tmp_path: Path, scoring: str, expected: Any, message: str,
) -> None:
    path = _write(tmp_path / "t.jsonl", [_row("4", scoring), _row(expected, scoring)])

    with pytest.raises(ValueError, match=f"^{re.escape(f'Line 2: {message}')}$"):
        custom.load_eval_tasks(path)


def test_the_line_number_counts_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(
        json.dumps(_row("4")) + "\n\n\n" + json.dumps(_row(_ABSENT)) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=f"^{re.escape(f'Line 4: {MISSING}')}$"):
        custom.load_eval_tasks(path)


@pytest.mark.parametrize(("expected", "message"), REFUSED)
def test_parse_eval_task_lines_refuses_the_same_rows(expected: Any, message: str) -> None:
    lines = [json.dumps(_row(expected))]

    with pytest.raises(ValueError, match=f"^{re.escape(f'Line 1: {message}')}$"):
        custom.parse_eval_task_lines(lines, "control.jsonl")


@pytest.mark.parametrize(
    ("value", "loaded"),
    [
        pytest.param(0, "0", id="zero"),
        pytest.param(False, "False", id="false"),
        pytest.param("0", "0", id="string-zero"),
        pytest.param(4, "4", id="int"),
        pytest.param(1.5, "1.5", id="float"),
        pytest.param([], "[]", id="empty-list"),
        pytest.param({}, "{}", id="empty-object"),
        pytest.param(" Paris ", " Paris ", id="padded-string"),
    ],
)
@pytest.mark.parametrize("scoring", ["contains", "exact", "regex"])
def test_an_expected_that_is_an_answer_keeps_loading_as_before(
    tmp_path: Path, value: Any, loaded: str, scoring: str,
) -> None:
    path = _write(tmp_path / "t.jsonl", [_row(value, scoring)])

    (task,) = custom.load_eval_tasks(path)

    assert task.expected == loaded


@pytest.mark.parametrize(
    ("expected", "loaded"),
    [
        pytest.param(_ABSENT, "", id="missing"),
        pytest.param(None, "None", id="null"),
        pytest.param("", "", id="empty"),
        pytest.param("   ", "   ", id="spaces"),
    ],
)
def test_require_expected_false_loads_the_row_as_before(
    tmp_path: Path, expected: Any, loaded: str,
) -> None:
    path = _write(tmp_path / "t.jsonl", [_row(expected)])

    (task,) = custom.load_eval_tasks(path, require_expected=False)
    (parsed,) = custom.parse_eval_task_lines(
        [json.dumps(_row(expected))], "t.jsonl", require_expected=False
    )

    assert task.expected == parsed.expected == loaded


def test_require_expected_false_still_checks_the_rest_of_the_row(tmp_path: Path) -> None:
    path = _write(tmp_path / "t.jsonl", [{"expected": "4"}])

    with pytest.raises(ValueError, match="missing required field 'prompt'"):
        custom.load_eval_tasks(path, require_expected=False)


# ---------------------------------------------------------------------------
# Callers that score ``expected``: the row is refused before anything is scored
# ---------------------------------------------------------------------------

def test_soup_eval_custom_exits_1_naming_the_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "custom-eval.db"))
    path = _write(tmp_path / "tasks.jsonl", [_row("Paris"), _row(" ")])
    (tmp_path / "model").mkdir()
    loads: list[Any] = []
    monkeypatch.setattr(custom, "_create_default_generator", lambda *a, **k: loads.append(a))

    result = runner.invoke(app, [
        "eval", "custom", "--tasks", str(path), "--model", str(tmp_path / "model"),
        "--output", str(tmp_path / "result.json"),
    ])

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"Line 2: {BLANK}" in _plain(result.output)
    assert loads == []
    assert not (tmp_path / "result.json").exists()


def test_soup_eval_quant_check_exits_with_the_usage_error(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (in_tmp / "before").mkdir()
    (in_tmp / "after").mkdir()
    _write(in_tmp / "tasks.jsonl", [_row(None)])
    loads: list[Any] = []
    monkeypatch.setattr(
        "soup_cli.eval.quant_check.make_model_generator", lambda *a, **k: loads.append(a)
    )

    result = runner.invoke(app, [
        "eval", "quant-check", "--before", "before", "--after", "after",
        "--tasks", "tasks.jsonl", "--allow-stub",
    ])

    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    assert f"Cannot load --tasks: Line 1: {NULL}" in _plain(result.output)
    assert loads == []


def test_run_quant_check_refuses_the_file(tmp_path: Path) -> None:
    from soup_cli.eval.quant_check import run_quant_check

    path = _write(tmp_path / "t.jsonl", [_row(_ABSENT)])

    with pytest.raises(ValueError, match=re.escape(f"Line 1: {MISSING}")):
        run_quant_check(before_gen=lambda p: "", after_gen=lambda p: "", tasks_file=str(path))


@pytest.mark.parametrize("scorer", [None, "contains"])
def test_the_gate_of_the_report_no_longer_passes_at_1_0(in_tmp: Path, scorer: Any) -> None:
    """The reproduction of #3240: an empty answer passed this gate at 1.0."""
    from soup_cli.eval.gate import load_suite, run_gate

    _write(in_tmp / "tasks.jsonl", [
        {"prompt": "What is the capital of France?", "expected": " ", "scoring": "contains"},
        {"prompt": "What is 6 x 7?", "expected": "", "scoring": "contains"},
    ])
    task: dict = {"type": "custom", "name": "facts", "threshold": 0.9, "tasks": "tasks.jsonl"}
    if scorer is not None:
        task["scorer"] = scorer
    (in_tmp / "gate.yaml").write_text(
        yaml.safe_dump({"suite": "t", "tasks": [task]}), encoding="utf-8"
    )

    result = run_gate(load_suite("gate.yaml"), generate_fn=lambda p: "", baseline={"facts": 1.0})

    (row,) = result.task_results
    assert result.passed is False
    assert row.score is None
    assert row.error == f"Line 1: {BLANK}"


def test_soup_eval_retention_refuses_a_control_file_and_records_nothing(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--add-task`` only hashes the file, so the refusal comes at load, before any model."""
    from soup_cli.commands import eval as eval_cmd
    from soup_cli.utils import gpu, live_eval

    monkeypatch.setattr(eval_cmd.console, "_width", 200)
    loads: list[Any] = []
    monkeypatch.setattr(live_eval, "make_generator", lambda *a, **k: loads.append(a))
    monkeypatch.setattr(live_eval, "resolve_device", lambda device=None: device or "cpu")
    monkeypatch.setattr(gpu, "bf16_fp16_flags", lambda device: (False, False))
    _write(in_tmp / "alpha.jsonl", [_row("ok"), _row("")])

    result = runner.invoke(app, [
        "eval", "retention", "--model", "base-model", "--device", "cpu", "--label", "v1",
        "--add-task", "alpha=alpha.jsonl", "--record",
    ])

    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    assert (
        f"control task 'alpha' cannot be loaded: ValueError: Line 2: {BLANK}"
        in _plain(result.output)
    )
    assert loads == []
    assert not (in_tmp / "retention.json").exists()


def test_a_retention_ledger_recorded_with_such_a_file_has_to_start_over(
    in_tmp: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The migration the fragment states: no route past the refusal keeps the old ledger.

    The first run stands for 0.78.0 (the check switched off), which recorded the file.
    """
    from soup_cli.commands import eval as eval_cmd
    from soup_cli.utils import gpu, live_eval

    monkeypatch.setattr(eval_cmd.console, "_width", 200)
    monkeypatch.setattr(live_eval, "make_generator", lambda *a, **k: (lambda prompt: "ok"))
    monkeypatch.setattr(live_eval, "resolve_device", lambda device=None: device or "cpu")
    monkeypatch.setattr(gpu, "bf16_fp16_flags", lambda device: (False, False))
    rows = [_row("ok", prompt="a"), _row("", prompt="b")]
    _write(in_tmp / "alpha.jsonl", rows)

    def retention(label: str, *args: str) -> Any:
        return runner.invoke(app, [
            "eval", "retention", "--model", "base-model", "--device", "cpu",
            "--label", label, "--record", *args,
        ])

    real = custom._require_expected
    monkeypatch.setattr(custom, "_require_expected", lambda row, line_num: None)
    assert retention("v1", "--add-task", "alpha=alpha.jsonl").exit_code == 0
    monkeypatch.setattr(custom, "_require_expected", real)
    recorded = (in_tmp / "retention.json").read_bytes()

    same = retention("v2")
    _write(in_tmp / "alpha-fixed.jsonl", [_row("ok", prompt="a"), _row("ok", prompt="b")])
    renamed = retention("v2", "--add-task", "alpha2=alpha-fixed.jsonl")
    _write(in_tmp / "alpha.jsonl", [_row("ok", prompt="a"), _row("ok", prompt="b")])
    edited = retention("v2")

    for result in (same, renamed):
        assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
        assert (
            f"control task 'alpha' cannot be loaded: ValueError: Line 2: {BLANK}"
            in _plain(result.output)
        )
    assert edited.exit_code == EXIT_USAGE_ERROR, (edited.output, repr(edited.exception))
    assert "control task 'alpha' changed" in _plain(edited.output)
    assert (in_tmp / "retention.json").read_bytes() == recorded

    fresh = retention("v1", "--ledger", "retention-new.json", "--add-task", "alpha=alpha.jsonl")
    assert fresh.exit_code == 0, (fresh.output, repr(fresh.exception))


def test_ship_metric_leg_and_its_floor_refuse_the_file(tmp_path: Path) -> None:
    from soup_cli.commands import ship

    path = str(_write(tmp_path / "t.jsonl", [_row(_ABSENT, prompt="p")]))
    generations: list[str] = []

    def gen(prompt: str) -> str:
        generations.append(prompt)
        return ""

    with pytest.raises(ValueError, match=re.escape(f"Line 1: {MISSING}")):
        ship._leg1_metric(gen, gen, "base", "tuned", path)
    with pytest.raises(ValueError, match=re.escape(f"Line 1: {MISSING}")):
        ship._build_task_floor_scorer(
            "metric", gen, base_id="base", task_eval=path, judge_model=None,
        )
    assert generations == []


def test_the_deploy_measure_scorer_refuses_the_file(tmp_path: Path) -> None:
    from soup_cli.utils.deploy_measure import _score_tasks

    path = _write(tmp_path / "t.jsonl", [_row("")])

    with pytest.raises(ValueError, match=re.escape(f"Line 1: {BLANK}")):
        _score_tasks(str(path), lambda p: "")


def test_the_cmaes_scorer_refuses_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soup_cli.utils import cmaes_merge

    monkeypatch.setattr(cmaes_merge, "_load_adapter_generator", lambda merged: (lambda p: ""))
    path = _write(tmp_path / "t.jsonl", [_row(None)])

    with pytest.raises(ValueError, match=re.escape(f"Line 1: {NULL}")):
        cmaes_merge._default_cmaes_scorer("merged", str(path))


def test_the_loop_verify_ship_check_refuses_the_file(in_tmp: Path) -> None:
    """``soup loop --verify ship`` runs ``soup ship`` in metric mode, which refuses it too."""
    from soup_cli.utils.loop_stages import check_verify_eval

    _write(in_tmp / "eval.jsonl", [_row(_ABSENT)])

    with pytest.raises(ValueError) as caught:
        check_verify_eval("ship", "eval.jsonl")

    assert "is not a soup ship task-eval JSONL" in str(caught.value)
    assert f"Line 1: {MISSING}" in str(caught.value)


# ---------------------------------------------------------------------------
# soup ship judge_score, pairwise and reward accept a prompt-only file
# ---------------------------------------------------------------------------

class _LoadedError(Exception):
    """Stops a scorer builder right after its tasks are loaded."""


@pytest.mark.parametrize("builder", ["judge_score", "pairwise", "reward"])
def test_ship_modes_that_never_score_expected_accept_a_prompt_only_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builder: str,
) -> None:
    from soup_cli.commands import ship

    real = custom.load_eval_tasks
    seen: list[tuple[dict, list]] = []

    def spy(path: Any, **kwargs: Any) -> list:
        seen.append((kwargs, real(path, **kwargs)))
        raise _LoadedError

    monkeypatch.setattr(custom, "load_eval_tasks", spy)
    path = str(_write(tmp_path / "t.jsonl", [{"prompt": "p1"}, {"prompt": "p2"}]))

    with pytest.raises(_LoadedError):
        if builder == "judge_score":
            ship._build_judge_scorer(path, "ollama://judge")
        elif builder == "pairwise":
            ship._build_pairwise_scorer(path, "ollama://judge")
        else:
            ship._build_reward_scorer(path, lambda **kwargs: 0.0)

    ((kwargs, tasks),) = seen
    assert kwargs == {"require_expected": False}
    assert [(task.prompt, task.expected) for task in tasks] == [("p1", ""), ("p2", "")]
