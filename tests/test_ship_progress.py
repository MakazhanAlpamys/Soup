"""Progress describes completed scoring work and owns stderr only when opted in."""

import inspect
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.eval.custom import EvalTask, run_eval
from soup_cli.eval.gate_suites import DEFAULT_GENERAL_SUITE, score_bundled_suite
from soup_cli.utils.ship_progress import group, session
from tests.conftest import strip_ansi


def events(stream: io.StringIO) -> list[dict]:
    emitted = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert_error_contract(emitted)
    return emitted


def assert_error_contract(emitted: list[dict]) -> None:
    for event in emitted:
        if event["event"] == "error":
            assert event.get("error") == "execution_aborted"
        else:
            assert "error" not in event


def plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def capture_runner() -> CliRunner:
    # Old standalone Click defaults to mixing stderr, newer Typer always splits it.
    separate = "mix_stderr" in inspect.signature(CliRunner).parameters
    options = {"mix_stderr": False} if separate else {}
    return CliRunner(**options)


@pytest.mark.parametrize("module", ["custom", "gate_suites", "forgetting", "generation"])
def test_pure_scorer_import_does_not_load_typer(module: str) -> None:
    script = (f"import importlib, sys; importlib.import_module('soup_cli.eval.{module}'); "
              "assert 'typer' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            encoding="utf-8", timeout=60)
    assert result.returncode == 0, (result.stdout, result.stderr)


@pytest.mark.parametrize("batched", [False, True])
def test_actual_completion_including_failed_items(batched: bool) -> None:
    stream = io.StringIO()
    observed_before_generation = []

    def scalar(prompt: str) -> str:
        # Scalar progress must precede the next generation, not be replayed at end.
        observed_before_generation.append(events(stream)[-1]["done"])
        if prompt == "second":
            raise RuntimeError("secret")
        return "yes"

    class Generator:
        __call__ = staticmethod(scalar)

    generator = Generator()
    if batched:
        generator.batch = lambda prompts: ["yes", None]  # type: ignore[attr-defined]
    with session(stream), group("task_eval", side="base", countable=True):
        result = run_eval("unused", [EvalTask("first", "yes"), EvalTask("second", "")], generator)
    assert result.accuracy == 0.5
    if not batched:
        assert observed_before_generation == [0, 1]
    emitted = events(stream)
    assert [e["event"] for e in emitted] == ["start", "progress", "progress", "end"]
    assert [e["done"] for e in emitted] == [0, 1, 2, 2]
    assert emitted[-1]["total"] == 2
    assert all(e["schema_version"] == 1 and e["repeat"] is None for e in emitted)


@pytest.mark.parametrize("suite", DEFAULT_GENERAL_SUITE)
def test_every_suite_accounts_for_all_failures(suite: str) -> None:
    stream = io.StringIO()

    def fail(prompt: str) -> str:
        raise RuntimeError("secret")

    with session(stream), group("suite_eval", side="tuned", suite=suite, countable=True):
        assert score_bundled_suite(suite, fail) == 0
    emitted = events(stream)
    assert emitted[-1]["done"] == emitted[-1]["total"] > 0
    assert len([e for e in emitted if e["event"] == "progress"]) == emitted[-1]["total"]


def test_empty_and_aborted_groups_and_context_restoration() -> None:
    stream = io.StringIO()
    with session(stream):
        with group("task_eval", side="base", countable=True):
            run_eval("unused", [], lambda _: "")
        with pytest.raises(RuntimeError), group("load", side="tuned"):
            raise RuntimeError("/secret/path API_KEY prompt")
        with group("load", side="base"):
            pass
    emitted = events(stream)
    assert emitted[1]["done"] == emitted[1]["total"] == 0
    assert emitted[3]["event"] == "error"
    assert emitted[3]["done"] is emitted[3]["total"] is None
    assert "secret" not in stream.getvalue()
    assert emitted[-1]["event"] == "end"
    before = stream.getvalue()
    with group("load", side="base"):
        pass
    assert before == stream.getvalue()


def test_stream_is_ascii_escaped_for_a_non_ascii_suite_name() -> None:
    suite = "espa\u00f1ol_\u65e5\u672c\u8a9e_\U0001f600"
    stream = io.StringIO()
    with session(stream), group("suite_eval", side="tuned", suite=suite, countable=True):
        pass
    raw = stream.getvalue()
    # A consumer on a non-UTF-8 pipe must be able to read every byte as ASCII.
    assert raw.isascii(), raw
    for escaped in ("espa\\u00f1ol", "\\u65e5\\u672c\\u8a9e", "\\ud83d\\ude00"):
        assert escaped in raw, (escaped, raw)
    assert [e["suite"] for e in events(stream)] == [suite, suite]


HARNESS = '''
import os, sys, warnings
from soup_cli.commands import ship
from soup_cli.utils import live_eval
from soup_cli.eval.gate_suites import _fingerprint_generator
def make(model, **kwargs):
    warnings.warn("TOP_SECRET prompt /private/path token", UserWarning)
    os.write(2, b"NATIVE_SECRET diagnostic\\n")
    if os.environ.get("ABORT_PROGRESS") == "interrupt":
        raise KeyboardInterrupt
    if os.environ.get("ABORT_PROGRESS"):
        raise RuntimeError("ABORT_SECRET /path credentials")
    def gen(prompt):
        return "yes" if prompt.startswith("task") else _fingerprint_generator(prompt)
    return gen
live_eval.make_generator = make
import typer
typer.main.get_command(ship.app).main(args=sys.argv[1:], prog_name="ship")
'''


def invoke(
    tmp_path: Path, arguments: list[str], abort: bool | str = False,
) -> subprocess.CompletedProcess:
    (tmp_path / "tasks.jsonl").write_text(
        '{"prompt":"task one","expected":"yes","scoring":"exact"}\n', encoding="utf-8")
    env = os.environ.copy()
    env["ABORT_PROGRESS"] = ("1" if abort is True else abort) or ""
    return subprocess.run([sys.executable, "-c", HARNESS, *arguments], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=60)


def live_flags() -> list[str]:
    return ["--base", "base", "--tuned", "tuned", "--task-eval", "tasks.jsonl",
            "--general-suite", "mini_arithmetic"]


def test_subprocess_jsonl_native_warnings_repeats_and_default_parity(tmp_path: Path) -> None:
    args = live_flags() + ["--noise-floor", "2"]
    ordinary = invoke(tmp_path, args)
    result = invoke(tmp_path, args + ["--progress", "jsonl"])
    assert result.returncode == ordinary.returncode == 2, result.stderr
    assert result.stdout == ordinary.stdout
    assert "TOP_SECRET" in ordinary.stderr and "NATIVE_SECRET" in ordinary.stderr
    emitted = [json.loads(line) for line in result.stderr.splitlines()]
    assert emitted and "SECRET" not in result.stderr
    starts = [e for e in emitted if e["event"] == "start"]
    assert {(e["phase"], e["side"]) for e in starts if e["phase"] == "load"} == {
        ("load", "base"), ("load", "tuned")}
    for repeat in (1, 2):
        for phase in ("task_eval", "suite_eval"):
            identified = [e for e in emitted if e["phase"] == phase and e["repeat"] == repeat]
            assert identified[0]["done"] == 0
            assert identified[-1]["done"] == identified[-1]["total"] > 0
    assert emitted[0]["phase"] == emitted[-1]["phase"] == "verdict"
    assert emitted[-1]["event"] == "end"


@pytest.mark.parametrize("abort", [True, "interrupt"])
def test_subprocess_abort_is_terminal_content_free_jsonl(tmp_path: Path, abort: bool | str) -> None:
    result = invoke(tmp_path, live_flags() + ["--progress", "jsonl"], abort=abort)
    assert result.returncode == 1
    emitted = [json.loads(line) for line in result.stderr.splitlines()]
    assert [e["event"] for e in emitted if e["phase"] == "load"] == ["start", "error"]
    assert emitted[-1]["event"] == "error"
    assert_error_contract(emitted)
    assert "SECRET" not in result.stderr and "/path" not in result.stderr


def test_unexpected_exception_from_real_app_is_content_free_jsonl(tmp_path: Path) -> None:
    harness = '''
import sys
import typer
from soup_cli.commands import ship
from soup_cli.cli import app
def unexpected(*args, **kwargs):
    raise RuntimeError("UNEXPECTED_SECRET /private/path")
ship._validate_task_mode_flag = unexpected
typer.main.get_command(app).main(args=["ship", "--progress", "jsonl"], prog_name="soup")
'''
    result = subprocess.run([sys.executable, "-c", harness], cwd=tmp_path,
                            capture_output=True, encoding="utf-8", timeout=60)
    assert result.returncode == 1, (result.stdout, result.stderr)
    emitted = [json.loads(line) for line in result.stderr.splitlines()]
    assert [(e["phase"], e["event"]) for e in emitted] == [
        ("verdict", "start"), ("verdict", "error")]
    assert_error_contract(emitted)
    for forbidden in ("Traceback", "UNEXPECTED_SECRET", "/private/path", "RuntimeError"):
        assert forbidden not in result.stderr


def test_offline_has_only_unknown_count_verdict_events(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(json.dumps({
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.8},
        "benchmarks": {"a": {"base": 0.8, "tuned": 0.8}},
    }))
    result = invoke(tmp_path, ["--evidence", "evidence.json", "--progress", "jsonl"])
    assert result.returncode == 0, result.stdout
    emitted = [json.loads(line) for line in result.stderr.splitlines()]
    assert [e["event"] for e in emitted] == ["start", "end"]
    assert all(e["phase"] == "verdict" and e["done"] is e["total"] is None for e in emitted)


def test_external_lm_eval_unknown_totals_and_skipped_baseline(monkeypatch) -> None:
    from soup_cli.commands import eval as eval_command
    from soup_cli.commands.ship import _leg2_scores

    calls = []

    def external(model, names, *args):
        calls.append((model, names))
        return {"results": {name: {"acc,none": 0.5} for name in names}}

    monkeypatch.setattr(eval_command, "_run_lm_eval", external)
    stream = io.StringIO()
    with session(stream):
        base, tuned = _leg2_scores(
            ["external_a", "external_b", "mini_arithmetic"], lambda _: "0", lambda _: "0",
            base_id="base", tuned_id="tuned", adapter=None,
            baseline_scores={"external_a": 0.6, "mini_arithmetic": 0.5}, device="cpu")
    assert base["external_a"] == 0.6 and tuned["external_a"] == 0.5
    assert calls == [("pretrained=tuned", ["external_a", "external_b"]),
                     ("pretrained=base", ["external_b"])]
    emitted = events(stream)
    assert not any(e["suite"] == "mini_arithmetic" and e["side"] == "base" for e in emitted)
    external_events = [e for e in emitted if e["suite"].startswith("external")]
    assert [e["event"] for e in external_events] == ["start", "end", "start", "end"]
    assert all(e["done"] is e["total"] is None for e in external_events)


@pytest.mark.parametrize("mode", ["judge_score", "pairwise"])
def test_judge_completion_and_repeated_groups(mode: str, tmp_path: Path, monkeypatch) -> None:
    from soup_cli.commands.ship import _leg1_judge, _leg1_pairwise, _measure_noise_floor
    from soup_cli.eval.judge import JudgeEvaluator, JudgeScore

    path = tmp_path / "tasks.jsonl"
    path.write_text('{"prompt":"first","expected":"yes"}\n'
                    '{"prompt":"second","expected":"yes"}\n', encoding="utf-8")
    stream = io.StringIO()
    judged = []

    def evaluate(self, prompt, response, category="default"):
        # Completion is after judging, not merely after generation.
        judged.append(prompt)
        return JudgeScore(prompt, response, weighted_score=3.0)

    monkeypatch.setattr(JudgeEvaluator, "evaluate", evaluate)
    monkeypatch.setattr(JudgeEvaluator, "compare_pair", lambda *args: -1)
    with session(stream):
        _measure_noise_floor(2, ["mini_arithmetic"], lambda _: "yes", base_id="base",
                             task_mode=mode, task_eval=str(path), judge_model="ollama://judge",
                             forgetting_threshold=0.05)
        leg = _leg1_judge if mode == "judge_score" else _leg1_pairwise
        leg(lambda _: "yes", lambda _: "yes", str(path), "ollama://judge")
    emitted = events(stream)
    task_ends = [e for e in emitted if e["phase"] == "task_eval" and e["event"] == "end"]
    assert all(e["done"] == e["total"] == 2 for e in task_ends)
    assert [e["repeat"] for e in task_ends[:2]] == [1, 2]
    assert [e["side"] for e in task_ends[2:]] == (
        ["base", "tuned"] if mode == "judge_score" else ["tuned"])
    if mode == "judge_score":
        assert len(judged) == 8


def test_inprocess_runner_exit_and_stderr_restored(tmp_path: Path, monkeypatch) -> None:
    from soup_cli.commands.ship import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "evidence.json").write_text(json.dumps({
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.5},
        "benchmarks": {"a": {"base": 0.8, "tuned": 0.8}},
    }))
    result = capture_runner().invoke(app, ["--evidence", "evidence.json", "--progress", "jsonl"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stderr.splitlines()[-1])["event"] == "end"
    # Otherwise-valid arguments: without them `soup ship` exits 3 for the missing input too.
    bad = capture_runner().invoke(app, ["--evidence", "evidence.json", "--progress", "typo"])
    assert bad.exit_code == 3, (bad.output, repr(bad.exception))
    assert "--progress must be jsonl" in plain(bad.output), bad.output
    assert "schema_version" not in bad.stderr


@pytest.mark.parametrize("mode", ["judge_score", "pairwise"])
def test_unavailable_judge_aborts_without_completed_item(
    mode: str, tmp_path: Path, monkeypatch,
) -> None:
    from soup_cli.commands.ship import _leg1_judge, _leg1_pairwise
    from soup_cli.eval.judge import JudgeEvaluator, JudgeUnavailableError

    path = tmp_path / "tasks.jsonl"
    path.write_text('{"prompt":"first","expected":"yes"}\n', encoding="utf-8")

    def unavailable(*args, **kwargs):
        raise JudgeUnavailableError("secret", url="http://localhost:1234")

    monkeypatch.setattr(JudgeEvaluator, "evaluate", unavailable)
    monkeypatch.setattr(JudgeEvaluator, "compare_pair", unavailable)
    stream = io.StringIO()
    with session(stream), pytest.raises(JudgeUnavailableError):
        leg = _leg1_judge if mode == "judge_score" else _leg1_pairwise
        leg(lambda _: "yes", lambda _: "yes", str(path), "ollama://judge")
    emitted = events(stream)
    assert [e["event"] for e in emitted] == ["start", "error"]
    assert emitted[-1]["done"] == 0
    assert "secret" not in stream.getvalue()


def test_inprocess_dependency_diagnostics_stay_off_jsonl(tmp_path: Path, monkeypatch) -> None:
    from soup_cli.commands.ship import app
    from soup_cli.utils import live_eval

    monkeypatch.chdir(tmp_path)
    (tmp_path / "tasks.jsonl").write_text('{"prompt":"task","expected":"yes"}\n')

    def make(*args, **kwargs):
        sys.stderr.write("PRIVATE_DEPENDENCY_DIAGNOSTIC\n")
        return lambda prompt: "yes"

    monkeypatch.setattr(live_eval, "make_generator", make)
    result = capture_runner().invoke(app, live_flags() + ["--progress", "jsonl"])
    assert result.exit_code == 2, result.output
    assert "PRIVATE" not in result.stderr
    assert [json.loads(line) for line in result.stderr.splitlines()][-1]["event"] == "end"
