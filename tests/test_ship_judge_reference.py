"""soup ship --judge-reference: reference-guided LLM judging (#2520).

CPU only. No model loads and no judge is reached: ``_resolve_generators`` and
``_leg2_scores`` are replaced, and ``JudgeEvaluator._call_llm`` records every
prompt the judge would have been sent and answers from a script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, List, Optional

import pytest

from soup_cli.commands import ship as ship_cmd
from soup_cli.eval import judge as judge_mod
from soup_cli.eval import judge_templates as jt
from soup_cli.eval.judge import (
    DEFAULT_RUBRIC,
    JudgeEvaluator,
    _build_judge_prompt,
    pairwise_compare,
    pairwise_winrate,
)
from soup_cli.utils.exit_codes import EXIT_USAGE_ERROR
from soup_cli.utils.ship_verdict import (
    EVIDENCE_SCHEMA_FIELDS,
    FAILED_ADAPTER_NOOP,
    build_task_win,
    compute_benchmark_deltas,
    decide_ship,
    verdict_from_evidence,
    verdict_to_dict,
    verdict_to_evidence,
)
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

runner = CliRunner()

JUDGE_URL = "ollama://judge"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


# --------------------------------------------------------------------------
# Literal copies of the pre-#2520 prompts. Written out by hand on purpose: a
# test that rebuilt them from the templates would pass on any template change.
# --------------------------------------------------------------------------


def _legacy_judge_prompt(prompt: str, response: str, rubric: dict) -> str:
    criteria_text = "\n".join(
        f"- **{c['name']}**: {c['description']}" for c in rubric["criteria"]
    )
    scale = rubric.get("scale", {"min": 1, "max": 5})
    scale_min = scale.get("min", 1)
    scale_max = scale.get("max", 5)
    return (
        "You are an expert evaluator. Score the following response based on "
        "the criteria below.\n\n"
        f"## Criteria\n{criteria_text}\n\n"
        f"## Scale\nScore each criterion from {scale_min} to {scale_max}.\n\n"
        f"## Prompt\n{prompt}\n\n"
        f"## Response\n{response}\n\n"
        "## Instructions\n"
        "Return a JSON object with:\n"
        '- "scores": {criterion_name: score, ...}\n'
        '- "reasoning": brief explanation\n\n'
        "Return ONLY the JSON object, no other text."
    )


_LEGACY_PAIRWISE = (
    "You are comparing two AI responses to the same prompt.\n\n"
    "## Prompt\n{prompt}\n\n"
    "## Response A\n{resp_a}\n\n"
    "## Response B\n{resp_b}\n\n"
    "## Task\nWhich response is better overall (helpfulness, accuracy, "
    "safety)? Reply with a JSON object: {{\"winner\": \"A\"}} or "
    "{{\"winner\": \"B\"}}. Return ONLY the JSON object."
)

#: Changing a template's text means bumping its version and this table.
PINNED = {
    ("judge_score", "preference"): (
        "judge_score.preference", 1,
        "232cb1275b5d326913161bb0a54e7ed9c098e81876695f5f0813b7040945f25d",
    ),
    ("judge_score", "reference"): (
        "judge_score.reference", 1,
        "38c12ca5d3b7c885357e51eff40212e8959eb655707af63fb90e235301c6effd",
    ),
    ("pairwise", "preference"): (
        "pairwise.preference", 1,
        "f3ad9cb32f70326e147e0ce2c17b9bb5ee644b028befbea6075347b7bf1716a1",
    ),
    ("pairwise", "reference"): (
        "pairwise.reference", 1,
        "d0ae25323a9c94b9c51598fdfe84450a29a1fb99cb85d487fdb5f241df867192",
    ),
}

_RUBRICS = (
    DEFAULT_RUBRIC,
    {"criteria": [{"name": "x{y}", "description": "uses {braces} and {0}"}],
     "scale": {"min": 0, "max": 10}},
    {"criteria": [{"name": "ünï", "description": "コード"}]},
)
_TEXTS = ("", "plain", "has {braces} and {0} and {{double}}", "multi\nline ``` fence")


# ---------------------------------------------------------------- templates


def test_templates_are_pinned_by_id_version_and_sha256():
    assert set(jt.JUDGE_TEMPLATES) == set(PINNED)
    for key, (template_id, version, digest) in PINNED.items():
        template = jt.judge_template(*key)
        assert template.id == template_id
        assert template.version == version
        assert hashlib.sha256(template.text.encode("utf-8")).hexdigest() == digest


def test_unknown_template_is_refused():
    with pytest.raises(ValueError):
        jt.judge_template("metric", "reference")
    with pytest.raises(ValueError):
        jt.judge_template("pairwise", "other")


@pytest.mark.parametrize("rubric", _RUBRICS)
@pytest.mark.parametrize("prompt", _TEXTS)
@pytest.mark.parametrize("response", _TEXTS)
def test_judge_score_prompt_is_byte_identical_without_reference(rubric, prompt, response):
    assert _build_judge_prompt(prompt, response, rubric) == _legacy_judge_prompt(
        prompt, response, rubric
    )


@pytest.mark.parametrize("prompt", _TEXTS)
@pytest.mark.parametrize("resp", _TEXTS)
def test_pairwise_prompt_is_byte_identical_without_reference(monkeypatch, prompt, resp):
    sent: List[str] = []
    evaluator = JudgeEvaluator(provider="ollama", model="m")
    monkeypatch.setattr(evaluator, "_call_llm", lambda p: sent.append(p) or '{"winner": "A"}')
    evaluator.compare_pair(prompt, resp, resp + "!")
    assert sent == [_LEGACY_PAIRWISE.format(prompt=prompt, resp_a=resp, resp_b=resp + "!")]


def test_judge_module_holds_no_inline_prompt_text():
    source = Path(judge_mod.__file__).read_text(encoding="utf-8")
    for fragment in ("You are an expert evaluator", "You are comparing", "## Response A"):
        assert fragment not in source


def test_reference_prompt_shows_the_reference_fenced_as_data():
    reference = "the answer is {x} ```` not a fence"
    text = _build_judge_prompt("q", "r", DEFAULT_RUBRIC, reference=reference)
    fence = "`" * 5  # one longer than the longest run of backticks in the reference
    assert "## Reference answer\n" in text
    assert f"\n{fence}\n{reference}\n{fence}\n" in text
    assert text.index("## Prompt") < text.index("## Reference answer") < text.index("## Response")
    assert jt.reference_fence("no ticks") == "```"


# ------------------------------------------------------- reference propagation


class _RecordingJudge:
    """compare_pair recorder; answers come from ``verdicts`` in call order."""

    def __init__(self, verdicts: Optional[List[int]] = None):
        self.calls: List[tuple] = []
        self.verdicts = list(verdicts or [])

    def compare_pair(self, prompt, resp_a, resp_b, reference=None):
        self.calls.append((prompt, resp_a, resp_b, reference))
        return self.verdicts.pop(0) if self.verdicts else -1


class _LegacyJudge:
    """A pre-#2520 PairwiseJudge: no ``reference`` parameter at all."""

    def __init__(self):
        self.calls = 0

    def compare_pair(self, prompt, resp_a, resp_b):
        self.calls += 1
        return 1 if len(resp_b) > len(resp_a) else 0


def test_both_swap_orders_receive_the_same_reference():
    judge = _RecordingJudge([1, 0])
    assert pairwise_compare("p", "a", "bb", judge, reference="REF") == 1
    assert judge.calls == [("p", "a", "bb", "REF"), ("p", "bb", "a", "REF")]


def test_judge_that_always_answers_a_scores_a_tie():
    judge = _RecordingJudge([0, 0])
    assert pairwise_compare("p", "a", "b", judge, reference="REF") == -1
    assert pairwise_winrate([("p", "a", "b")], _RecordingJudge([0, 0]),
                            references=["REF"]) == 0.5


def test_legacy_pairwise_judge_keeps_working_without_reference():
    judge = _LegacyJudge()
    assert pairwise_winrate([("p", "s", "long"), ("q", "s", "long")], judge) == 1.0
    assert judge.calls == 4


def test_references_align_with_their_pairs():
    judge = _RecordingJudge()
    pairs = [("p0", "a0", "b0"), ("p1", "a1", "b1"), ("p2", "a2", "b2")]
    pairwise_winrate(pairs, judge, references=["r0", "r1", "r2"])
    assert [(call[0], call[3]) for call in judge.calls] == [
        ("p0", "r0"), ("p0", "r0"), ("p1", "r1"), ("p1", "r1"), ("p2", "r2"), ("p2", "r2"),
    ]


def test_misaligned_references_are_refused():
    with pytest.raises(ValueError, match="2 for 3 pairs"):
        pairwise_winrate([("p", "a", "b")] * 3, _RecordingJudge(), references=["r", "r"])
    with pytest.raises(ValueError):
        pairwise_winrate([], _RecordingJudge(), references=["r"])


def test_evaluate_batch_passes_reference_only_when_set(monkeypatch):
    seen: List[Optional[str]] = []

    def _evaluate(self, prompt, response, category="default", **kw):
        seen.append(kw.get("reference", "<absent>"))
        return judge_mod.JudgeScore(prompt=prompt, response=response, scores={},
                                    weighted_score=1.0, reasoning="", category=category)

    monkeypatch.setattr(JudgeEvaluator, "evaluate", _evaluate)
    JudgeEvaluator(provider="ollama", model="m").evaluate_batch(
        [{"prompt": "a", "response": "b"}, {"prompt": "c", "response": "d", "reference": "R"}]
    )
    assert seen == ["<absent>", "R"]


# ----------------------------------------------------------- raw-row checks


def _write_tasks(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(**fields) -> str:
    return json.dumps({"prompt": "q", "scoring": "exact", **fields})


@pytest.mark.parametrize(
    ("row", "problem"),
    (
        (json.dumps({"prompt": "q"}), "field 'expected' is missing"),
        (_row(expected=None), "field 'expected' is null"),
        (_row(expected=3), "field 'expected' must be a string, got int"),
        (_row(expected=False), "field 'expected' must be a string, got bool"),
        (_row(expected=["a"]), "field 'expected' must be a string, got list"),
        (_row(expected=""), "field 'expected' is empty or whitespace-only"),
        (_row(expected=" \t\n"), "field 'expected' is empty or whitespace-only"),
        (_row(expected="a.*", scoring="regex"), "is a pattern, not a reference answer"),
        (json.dumps(["q", "a"]), "not a JSON object"),
        ("{not json", "not valid JSON"),
    ),
)
def test_bad_expected_is_refused_with_its_physical_line(tmp_path, row, problem):
    tasks = tmp_path / "t.jsonl"
    # Blank lines are skipped by the loader, so task index 1 sits on line 4.
    _write_tasks(tasks, [_row(expected="ok"), "", "  ", row])
    with pytest.raises(ValueError) as error:
        ship_cmd._task_references(str(tasks))
    message = str(error.value)
    assert message.startswith("--judge-reference: task-eval line 4: ")
    assert problem in message


def test_references_are_returned_in_task_order(tmp_path):
    from soup_cli.eval.custom import load_eval_tasks

    tasks = tmp_path / "t.jsonl"
    _write_tasks(tasks, [_row(prompt="a", expected="A"), "", _row(prompt="b", expected="B ")])
    assert ship_cmd._task_references(str(tasks)) == ["A", "B "]
    assert [t.prompt for t in load_eval_tasks(str(tasks))] == ["a", "b"]


def test_rows_past_the_task_cap_are_refused(tmp_path, monkeypatch):
    from soup_cli.eval import custom

    monkeypatch.setattr(custom, "MAX_EVAL_TASKS", 2)
    tasks = tmp_path / "t.jsonl"
    _write_tasks(tasks, [_row(expected="A"), "", _row(expected="B"), _row(expected="C")])
    with pytest.raises(ValueError, match=r"^--judge-reference: task-eval line 4: more than 2"):
        ship_cmd._task_references(str(tasks))


def test_builder_refuses_references_that_no_longer_match_the_tasks(tmp_path):
    tasks = tmp_path / "t.jsonl"
    _write_tasks(tasks, [_row(expected="A"), _row(expected="B")])
    assert ship_cmd._aligned_references(str(tasks), 2, True) == ["A", "B"]
    assert ship_cmd._aligned_references(str(tasks), 3, False) is None
    with pytest.raises(ValueError, match="2 rows with an 'expected' for 3 loaded tasks"):
        ship_cmd._aligned_references(str(tasks), 3, True)


# --------------------------------------------------------------- live runs


class _Script:
    """Patches the live pipeline; records judge prompts and model loads."""

    def __init__(self, monkeypatch):
        self.prompts: List[str] = []
        self.loads: List[tuple] = []
        script = self

        def _resolve(base, tuned, adapter, device, quantization=None, **kw):
            script.loads.append((base, tuned, adapter))
            return (lambda p: f"base says {p}", lambda p: f"tuned says {p} at length")

        def _leg2(names, base_gen, tuned_gen, **kw):
            scores = {name: 0.5 for name in names}
            return dict(scores), dict(scores)

        def _call_llm(self_, prompt, max_tokens=1024):
            script.prompts.append(prompt)
            if "## Response A" in prompt:
                a = prompt.split("## Response A\n", 1)[1].split("\n\n", 1)[0]
                b = prompt.split("## Response B\n", 1)[1].split("\n\n", 1)[0]
                return json.dumps({"winner": "B" if len(b) > len(a) else "A"})
            score = 5 if "at length" in prompt.split("## Response\n", 1)[1] else 2
            names = [c["name"] for c in DEFAULT_RUBRIC["criteria"]]
            return json.dumps({"scores": {n: score for n in names}, "reasoning": "ok"})

        monkeypatch.setattr(ship_cmd, "_resolve_generators", _resolve)
        monkeypatch.setattr(ship_cmd, "_leg2_scores", _leg2)
        monkeypatch.setattr(JudgeEvaluator, "_call_llm", _call_llm)


def _ship(monkeypatch, tasks: List[str], *args: str, files: Callable = lambda: None):
    script = _Script(monkeypatch)
    with runner.isolated_filesystem():
        files()
        _write_tasks(Path("tasks.jsonl"), tasks)
        result = runner.invoke(
            ship_cmd.app,
            ["--base", "b", "--task-eval", "tasks.jsonl", "--general-suite", "mini_mmlu",
             "--device", "cpu", "--output", "out.json", *args],
        )
        out_path = Path("out.json")
        out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
        ev_path = Path("ev.json")
        evidence = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
    return result, script, out, evidence


_TASKS = [_row(prompt="first", expected="ALPHA"), "", _row(prompt="second", expected="BETA")]


@pytest.mark.parametrize("mode", ("judge_score", "pairwise"))
def test_reference_run_shows_each_task_its_own_reference(monkeypatch, mode):
    result, script, out, _ = _ship(
        monkeypatch, _TASKS, "--tuned", "t", "--task-mode", mode,
        "--judge-model", JUDGE_URL, "--judge-reference",
    )
    assert result.exit_code in (0, 2), (result.output, repr(result.exception))
    assert script.prompts
    for prompt in script.prompts:
        assert "## Reference answer" in prompt
        if "says first" in prompt:
            assert "ALPHA" in prompt and "BETA" not in prompt
        else:
            assert "says second" in prompt and "BETA" in prompt and "ALPHA" not in prompt
    template = jt.judge_template(mode, "reference")
    assert out["task_win"]["judge_mode"] == "reference"
    assert out["task_win"]["judge_template"] == {"id": template.id, "version": template.version}


@pytest.mark.parametrize("mode", ("judge_score", "pairwise"))
def test_default_run_sends_the_legacy_prompts(monkeypatch, mode):
    result, script, out, evidence = _ship(
        monkeypatch, _TASKS, "--tuned", "t", "--task-mode", mode,
        "--judge-model", JUDGE_URL, "--emit-evidence", "ev.json",
    )
    assert result.exit_code in (0, 2), (result.output, repr(result.exception))
    assert script.prompts
    assert all("Reference answer" not in prompt for prompt in script.prompts)
    template = jt.judge_template(mode, "preference")
    assert out["task_win"]["judge_mode"] == "preference"
    assert out["task_win"]["judge_template"] == {"id": template.id, "version": template.version}
    # Legacy evidence: no judge keys for a preference run.
    assert set(evidence["task"]) == {"mode", "base", "tuned"}


def test_metric_output_has_no_judge_fields(monkeypatch):
    result, _, out, _ = _ship(monkeypatch, _TASKS, "--tuned", "t")
    assert result.exit_code in (0, 2), (result.output, repr(result.exception))
    assert "judge_mode" not in out["task_win"]
    assert "judge_template" not in out["task_win"]


@pytest.mark.parametrize("mode", ("judge_score", "pairwise"))
def test_noise_floor_judges_with_the_same_template(monkeypatch, mode):
    result, script, out, _ = _ship(
        monkeypatch, _TASKS, "--tuned", "t", "--task-mode", mode,
        "--judge-model", JUDGE_URL, "--judge-reference", "--noise-floor", "2",
    )
    assert result.exit_code in (0, 2), (result.output, repr(result.exception))
    per_side = len(script.prompts)
    assert per_side > 0
    assert all("## Reference answer" in prompt for prompt in script.prompts)
    assert out["noise_floor"]["judge_inclusive"] is True


@pytest.mark.parametrize("mode", ("judge_score", "pairwise"))
def test_reference_evidence_round_trips(monkeypatch, mode):
    result, _, out, evidence = _ship(
        monkeypatch, _TASKS, "--tuned", "t", "--task-mode", mode,
        "--judge-model", JUDGE_URL, "--judge-reference", "--emit-evidence", "ev.json",
    )
    assert result.exit_code in (0, 2), (result.output, repr(result.exception))
    template = jt.judge_template(mode, "reference")
    assert evidence["task"]["judge_mode"] == "reference"
    assert evidence["task"]["judge_template"] == {"id": template.id, "version": template.version}
    replay = verdict_from_evidence(evidence, forgetting_threshold=out["forgetting_threshold"])
    assert replay.task_win.judge_mode == "reference"
    assert replay.task_win.judge_template == (template.id, template.version)
    assert verdict_to_evidence(replay, provenance=evidence.get("provenance")) == evidence


# ------------------------------------------------- usage errors, before loads


@pytest.mark.parametrize(
    ("tasks", "args", "needle"),
    (
        (_TASKS, ("--tuned", "t"), "only to --task-mode judge_score or pairwise, not metric"),
        (_TASKS, ("--evidence", "ev.json"), "only to a live run, not --evidence"),
        ([_row(expected="ok"), "", json.dumps({"prompt": "q"})],
         ("--tuned", "t", "--task-mode", "pairwise", "--judge-model", JUDGE_URL),
         "task-eval line 3: field 'expected' is missing"),
        ([_row(expected="a+", scoring="regex")],
         ("--tuned", "t", "--task-mode", "judge_score", "--judge-model", JUDGE_URL),
         "task-eval line 1: field 'expected' of a 'scoring: regex' row is a pattern"),
    ),
    ids=("metric", "evidence", "missing-expected", "regex"),
)
def test_invalid_use_is_a_usage_error_before_any_model_load(monkeypatch, tasks, args, needle):
    def _files():
        Path("ev.json").write_text(json.dumps({
            "task": {"mode": "pairwise", "base": 0.5, "tuned": 0.6}, "benchmarks": {},
        }), encoding="utf-8")

    result, script, out, _ = _ship(monkeypatch, tasks, *args, "--judge-reference", files=_files)
    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    assert needle in _plain(result.output)
    assert script.loads == []
    assert script.prompts == []
    assert out is None


def test_ship_refuses_metric_before_the_live_path(monkeypatch):
    """The CLI layer refuses on its own, before ``_verdict_live`` is reached."""
    reached: List[dict] = []
    monkeypatch.setattr(ship_cmd, "_verdict_live", lambda **kw: reached.append(kw))
    result, _, _, _ = _ship(monkeypatch, _TASKS, "--tuned", "t", "--judge-reference")
    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    assert reached == []


def test_verdict_live_refuses_metric_on_its_own(monkeypatch):
    """The live path refuses too: it is reachable without the CLI (tests, SDK)."""
    import typer

    script = _Script(monkeypatch)
    with runner.isolated_filesystem():
        _write_tasks(Path("tasks.jsonl"), _TASKS)
        with pytest.raises(typer.Exit) as error:
            ship_cmd._verdict_live(
                base="b", tuned="t", adapter=None, task_eval="tasks.jsonl",
                task_mode="metric", judge_model=None, general_suite="mini_mmlu",
                baseline_spec=None, device="cpu", forgetting_threshold=0.05,
                judge_reference=True,
            )
    assert error.value.exit_code == EXIT_USAGE_ERROR
    assert script.loads == []


def test_metric_from_config_is_refused_too(monkeypatch):
    def _files():
        Path("soup.yaml").write_text(
            "base: b\ntask: sft\ndata:\n  train: d.jsonl\neval:\n  ship:\n    task_mode: metric\n",
            encoding="utf-8",
        )

    result, script, _, _ = _ship(
        monkeypatch, _TASKS, "--tuned", "t", "--config", "soup.yaml", "--judge-reference",
        files=_files,
    )
    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    assert script.loads == []


def test_bad_reference_is_refused_before_the_adapter_noop_shortcut(monkeypatch):
    calls: List[str] = []
    monkeypatch.setattr(ship_cmd, "_adapter_is_noop", lambda a: calls.append(a) or True)
    result, script, _, _ = _ship(
        monkeypatch, [json.dumps({"prompt": "q"})], "--adapter", "lora",
        "--task-mode", "pairwise", "--judge-model", JUDGE_URL, "--judge-reference",
        files=lambda: Path("lora").mkdir(),
    )
    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    assert "field 'expected' is missing" in _plain(result.output)
    assert calls == []
    assert script.loads == []


def test_adapter_noop_verdict_names_the_reference_judging(monkeypatch):
    monkeypatch.setattr(ship_cmd, "_adapter_is_noop", lambda a: True)
    result, script, out, _ = _ship(
        monkeypatch, _TASKS, "--adapter", "lora", "--task-mode", "judge_score",
        "--judge-model", JUDGE_URL, "--judge-reference", files=lambda: Path("lora").mkdir(),
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert out["failed_rule"] == FAILED_ADAPTER_NOOP
    assert out["task_win"]["judge_mode"] == "reference"
    assert out["task_win"]["won"] is False
    assert script.loads == []


@pytest.mark.parametrize("mode", ("judge_score", "pairwise"))
def test_builders_recheck_the_rows(tmp_path, monkeypatch, mode):
    """A file edited after the pre-load check is refused when the scorer builds."""
    tasks = tmp_path / "t.jsonl"
    _write_tasks(tasks, [_row(expected="ok"), _row(expected="   ")])
    build = ship_cmd._build_judge_scorer if mode == "judge_score" else (
        ship_cmd._build_pairwise_scorer
    )
    with pytest.raises(ValueError, match="line 2: field 'expected' is empty"):
        build(str(tasks), JUDGE_URL, judge_reference=True)
    with pytest.raises(ValueError, match="judge_score or pairwise"):
        ship_cmd._build_task_floor_scorer(
            "metric", lambda p: "", base_id="b", task_eval=str(tasks), judge_model=None,
            judge_reference=True,
        )


# ------------------------------------------------------ verdict and evidence


def test_task_evidence_schema_names_the_judge_fields():
    assert EVIDENCE_SCHEMA_FIELDS["task"] == frozenset(
        {"mode", "base", "tuned", "judge_mode", "judge_template"}
    )


def test_build_task_win_refuses_judge_fields_on_metric():
    with pytest.raises(ValueError):
        build_task_win("metric", 0.1, 0.2, judge_mode="reference")
    with pytest.raises(ValueError):
        build_task_win("pairwise", 0.5, 0.6, judge_mode="other")
    # An explicit template skips the template lookup: the mode is still checked.
    with pytest.raises(ValueError, match="judge_mode must be one of"):
        build_task_win(
            "pairwise", 0.5, 0.6, judge_mode="other", judge_template=("pairwise.reference", 1)
        )


def test_decide_ship_keeps_the_judge_metadata():
    win = build_task_win("pairwise", 0.5, 0.75, judge_mode="reference")
    deltas = compute_benchmark_deltas({"mini_mmlu": 0.7}, {"mini_mmlu": 0.7})
    verdict = decide_ship(win, deltas)
    assert verdict.task_win.judge_mode == "reference"
    assert verdict.task_win.judge_template == ("pairwise.reference", 1)
    # A stale ``won`` is corrected without dropping the metadata.
    stale = build_task_win("judge_score", 0.5, 0.75, judge_mode="reference")
    from dataclasses import replace

    corrected = decide_ship(replace(stale, won=False), deltas)
    assert corrected.task_win.won is True
    assert corrected.task_win.judge_mode == "reference"
    assert verdict_to_dict(corrected)["task_win"]["judge_template"] == {
        "id": "judge_score.reference", "version": 1,
    }


@pytest.mark.parametrize("mode", ("judge_score", "pairwise"))
def test_legacy_evidence_reads_as_preference(mode):
    verdict = verdict_from_evidence(
        {"task": {"mode": mode, "base": 0.5, "tuned": 0.6}, "benchmarks": {}}
    )
    assert verdict.task_win.judge_mode == "preference"
    assert verdict.task_win.judge_template == (jt.judge_template(mode, "preference").id, 1)
    assert verdict_to_evidence(verdict) == {
        "task": {"mode": mode, "base": 0.5, "tuned": 0.6}, "benchmarks": {},
    }


@pytest.mark.parametrize(
    "task_extra",
    (
        {"judge_mode": "reference"},
        {"judge_template": {"id": "pairwise.reference", "version": 1}},
        {"judge_mode": "preference",
         "judge_template": {"id": "pairwise.preference", "version": 1}},
        {"judge_mode": "reference", "judge_template": {"id": "judge_score.reference",
                                                       "version": 1}},
        {"judge_mode": "reference", "judge_template": {"id": "pairwise.reference",
                                                       "version": 2}},
        {"judge_mode": "reference", "judge_template": {"id": "pairwise.reference",
                                                       "version": True}},
        {"judge_mode": "reference", "judge_template": {"id": "pairwise.reference",
                                                       "version": 1, "extra": 1}},
        {"judge_mode": "reference", "judge_template": ["pairwise.reference", 1]},
    ),
)
def test_malformed_judge_evidence_is_refused(task_extra):
    with pytest.raises((TypeError, ValueError)):
        verdict_from_evidence({
            "task": {"mode": "pairwise", "base": 0.5, "tuned": 0.6, **task_extra},
            "benchmarks": {},
        })


@pytest.mark.parametrize(
    ("task", "needle"),
    (
        # A reference template under a preference label is not reference judging.
        ({"mode": "pairwise", "judge_mode": "preference",
          "judge_template": {"id": "pairwise.reference", "version": 1}},
         "judge_mode must be 'reference'"),
        # Metric mode has no judge, whatever the fields say.
        ({"mode": "metric", "judge_mode": "reference",
          "judge_template": {"id": "pairwise.reference", "version": 1}},
         "applies only to task mode judge_score or pairwise"),
    ),
)
def test_judge_evidence_is_refused_by_its_own_check(task, needle):
    with pytest.raises(ValueError, match=needle):
        verdict_from_evidence(
            {"task": {"base": 0.5, "tuned": 0.6, **task}, "benchmarks": {}}
        )


def test_mcp_ship_evidence_reports_the_judge_fields(tmp_path, monkeypatch):
    from soup_cli.mcp_server.registry import tool_ship_evidence

    monkeypatch.chdir(tmp_path)
    Path("ev.json").write_text(json.dumps({
        "task": {"mode": "judge_score", "base": 0.4, "tuned": 0.6, "judge_mode": "reference",
                 "judge_template": {"id": "judge_score.reference", "version": 1}},
        "benchmarks": {"mini_mmlu": {"base": 0.7, "tuned": 0.7}},
    }), encoding="utf-8")
    result = tool_ship_evidence({"evidence": "ev.json"})
    assert result["task_win"]["judge_mode"] == "reference"
    assert result["task_win"]["judge_template"] == {"id": "judge_score.reference", "version": 1}
