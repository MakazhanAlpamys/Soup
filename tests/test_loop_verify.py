"""The check before a loop rollout: a soup ship verdict by default, never a silent skip.

The pre-wired gate stage loaded ``--eval`` as an eval-gate suite (``suite`` +
``tasks``) while the docs passed a ``soup eval lock`` design file: the load
failed, the verdict was ``SKIPPED`` with no reason, and nothing ever rolled
out. Now ``soup loop init --verify`` picks ``ship`` (default: a ``soup ship``
verdict of the candidate against the production version, ``--eval`` = a
task-eval JSONL) or ``eval-gate``; ``--eval`` is checked at init and before a
pre-wired watch, and a ``SKIPPED`` verdict carries its reason into the
iteration record.
"""

from __future__ import annotations

import json
import os
import re

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.utils.loop_state import LoopState, read_state

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
TASK_LINE = '{"prompt": "2+2", "expected": "4"}\n'
GATE_SUITE = (
    "suite: loop\n"
    "tasks:\n"
    "  - type: custom\n"
    "    name: math\n"
    "    threshold: 0.5\n"
    "    tasks: tasks.jsonl\n"
    "    scorer: exact\n"
)


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "tasks.jsonl").write_text(TASK_LINE, encoding="utf-8")
    (tmp_path / "gate.yaml").write_text(GATE_SUITE, encoding="utf-8")
    (tmp_path / "lock.json").write_text(
        json.dumps({"dimensions": [{"name": "accuracy"}]}), encoding="utf-8"
    )
    return tmp_path


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./model")
    kwargs.setdefault("base_model", "org/base")
    kwargs.setdefault("eval_suite", "tasks.jsonl")
    kwargs.setdefault("verify", "ship")
    return LoopState(baseline="b", **kwargs)


def _invoke(*args: str):
    from soup_cli.commands.loop import app

    result = runner.invoke(app, list(args))
    return result, _clean(result.output)


class _Result:
    def __init__(self, returncode: int, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stderr = stderr


def _ship_runner(decision, *, returncode=None, stderr=b"", raw=None, calls=None):
    """A soup ship stand-in that writes ``--output`` like the real command."""

    def run(argv, **kwargs):
        if calls is not None:
            calls.append(argv)
        out = argv[argv.index("--output") + 1]
        if raw is not None:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(raw)
        elif decision is not None:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump({"decision": decision, "failed_rule": None}, fh)
        code = returncode if returncode is not None else (0 if decision == "SHIP" else 2)
        return _Result(code, stderr)

    return run


def _trained(cwd) -> dict:
    adapter = cwd / ".soup-loops" / "adapters" / "loop-train-1a2b3c4d"
    adapter.mkdir(parents=True)
    return {"skipped": False, "adapter_path": str(adapter), "run_id": "loop-train-1a2b3c4d"}


# ---------------------------------------------------------------------------
# check_verify_eval
# ---------------------------------------------------------------------------


def test_ship_accepts_a_task_eval_jsonl(cwd):
    ls.check_verify_eval("ship", "tasks.jsonl")


def test_eval_gate_accepts_a_gate_suite(cwd):
    ls.check_verify_eval("eval_gate", "gate.yaml")


@pytest.mark.parametrize("method", ["ship", "eval_gate"])
def test_a_lock_design_file_is_refused_by_what_it_is(cwd, method):
    with pytest.raises(ValueError, match="is a soup eval lock design file") as info:
        ls.check_verify_eval(method, "lock.json")
    assert "task-eval JSONL" in str(info.value) and "eval-gate suite YAML" in str(info.value)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("missing.jsonl", "is not a soup ship task-eval JSONL (FileNotFoundError"),
        ("gate.yaml", "is not a soup ship task-eval JSONL"),
    ],
)
def test_ship_refuses_what_soup_ship_cannot_read(cwd, path, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        ls.check_verify_eval("ship", path)


def test_ship_refuses_an_empty_task_file(cwd):
    (cwd / "empty.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty.jsonl"):
        ls.check_verify_eval("ship", "empty.jsonl")


def test_ship_refuses_a_task_file_outside_cwd(cwd, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "tasks.jsonl"
    outside.write_text(TASK_LINE, encoding="utf-8")
    with pytest.raises(ValueError, match="is not a soup ship task-eval JSONL"):
        ls.check_verify_eval("ship", str(outside))


def test_eval_gate_refuses_a_task_jsonl(cwd):
    with pytest.raises(ValueError, match="is not an eval-gate suite"):
        ls.check_verify_eval("eval_gate", "tasks.jsonl")


def test_unknown_method_is_refused(cwd):
    with pytest.raises(ValueError, match="--verify must be ship or eval-gate, got 'vibes'"):
        ls.check_verify_eval("vibes", "tasks.jsonl")


# ---------------------------------------------------------------------------
# verify_with_ship
# ---------------------------------------------------------------------------


def test_ship_compares_against_the_base_when_nothing_is_in_production(cwd, monkeypatch):
    calls = []
    monkeypatch.setattr(ls, "_SHIP_RUNNER", _ship_runner("SHIP", calls=calls))
    ctx = _trained(cwd)
    out = ls.verify_with_ship(_state(), ctx)
    assert out["gate_verdict"] == "OK" and out["verdict_decision"] == "SHIP"
    assert out["verdict_baseline"] == {"kind": "base", "ref": "org/base"}
    argv = calls[0]
    assert argv[argv.index("--base") + 1] == "org/base"
    assert argv[argv.index("--adapter") + 1] == ctx["adapter_path"]
    assert argv[argv.index("--task-eval") + 1] == "tasks.jsonl"
    assert "--base-adapter" not in argv
    assert os.path.isfile(out["verdict_path"])


def test_ship_compares_against_the_production_adapter(cwd, monkeypatch):
    calls = []
    monkeypatch.setattr(ls, "_SHIP_RUNNER", _ship_runner("SHIP", calls=calls))
    out = ls.verify_with_ship(_state(production_adapter="adapters/v1"), _trained(cwd))
    assert calls[0][calls[0].index("--base-adapter") + 1] == "adapters/v1"
    assert out["verdict_baseline"] == {"kind": "version", "ref": "adapters/v1"}


def test_dont_ship_blocks_the_rollout(cwd, monkeypatch):
    monkeypatch.setattr(ls, "_SHIP_RUNNER", _ship_runner("DON'T SHIP"))
    out = ls.verify_with_ship(_state(), _trained(cwd))
    assert out["gate_verdict"] == "MAJOR"
    assert out["verdict_decision"] == "DON'T SHIP"


def test_the_decision_comes_from_the_file_not_the_exit_code(cwd, monkeypatch):
    monkeypatch.setattr(ls, "_SHIP_RUNNER", _ship_runner("SHIP", returncode=2))
    assert ls.verify_with_ship(_state(), _trained(cwd))["gate_verdict"] == "OK"


def test_no_verdict_file_is_a_skip_with_the_reason(cwd, monkeypatch):
    stderr = b"Usage: soup ship\nError: No such option: --base-adapter\x1b[2J\n"
    monkeypatch.setattr(ls, "_SHIP_RUNNER", _ship_runner(None, returncode=2, stderr=stderr))
    out = ls.verify_with_ship(_state(production_adapter="adapters/v1"), _trained(cwd))
    assert out["gate_verdict"] == "SKIPPED"
    assert out["notes"].startswith("verdict_skipped: soup ship wrote no verdict (exit 2): Error:")
    assert "No such option: --base-adapter" in out["notes"]
    assert "\x1b" not in out["notes"]


@pytest.mark.parametrize("raw", ["{not json", json.dumps({"decision": "MAYBE"}), "[]"])
def test_an_unreadable_verdict_file_is_a_skip(cwd, monkeypatch, raw):
    monkeypatch.setattr(ls, "_SHIP_RUNNER", _ship_runner(None, returncode=0, raw=raw))
    out = ls.verify_with_ship(_state(), _trained(cwd))
    assert out["gate_verdict"] == "SKIPPED"
    assert "soup ship wrote no verdict (exit 0)" in out["notes"]


def test_a_runner_crash_is_a_skip_with_the_reason(cwd, monkeypatch):
    def boom(argv, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(ls, "_SHIP_RUNNER", boom)
    out = ls.verify_with_ship(_state(), _trained(cwd))
    assert out["notes"] == "verdict_skipped: soup ship could not run (OSError: disk gone)"


def test_a_missing_adapter_is_a_skip_with_the_reason(cwd, monkeypatch):
    monkeypatch.setattr(ls, "_SHIP_RUNNER", _ship_runner("SHIP"))
    out = ls.verify_with_ship(_state(), {"skipped": False, "adapter_path": "gone/dir"})
    assert out["gate_verdict"] == "SKIPPED"
    assert "the trained adapter 'gone/dir' is missing" in out["notes"]


def test_skipped_training_skips_the_verdict_without_running_ship(cwd, monkeypatch):
    calls = []
    monkeypatch.setattr(ls, "_SHIP_RUNNER", _ship_runner("SHIP", calls=calls))
    assert ls.verify_with_ship(_state(), {"skipped": True})["gate_verdict"] == "SKIPPED"
    assert calls == []


def test_an_unknown_base_is_a_skip_with_the_reason(cwd, monkeypatch):
    monkeypatch.setattr(ls, "_SHIP_RUNNER", _ship_runner("SHIP"))
    state = _state(served_model="registry://abc12", base_model=None)
    out = ls.verify_with_ship(state, _trained(cwd))
    assert out["notes"].startswith("verdict_skipped: base model unknown")


# ---------------------------------------------------------------------------
# Dispatch, the gate's reasons, and the iteration record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verify", "expected"), [("ship", "ship"), ("eval_gate", "eval_gate"), (None, "eval_gate")]
)
def test_dispatch_follows_verify_and_old_files_keep_the_gate(cwd, monkeypatch, verify, expected):
    monkeypatch.setattr(ls, "verify_with_ship", lambda s, c: {"gate_verdict": "OK", "by": "ship"})
    monkeypatch.setattr(
        ls, "gate_against_baseline", lambda s, c: {"gate_verdict": "OK", "by": "eval_gate"}
    )
    assert ls.verify_candidate(_state(verify=verify), {})["by"] == expected


def test_gate_skip_names_the_suite_problem(cwd):
    out = ls.gate_against_baseline(
        _state(verify="eval_gate", eval_suite="lock.json"), _trained(cwd)
    )
    assert out["gate_verdict"] == "SKIPPED"
    assert out["notes"].startswith("verdict_skipped: the eval-gate suite could not be loaded")


def test_iteration_record_keeps_the_skip_reason(cwd):
    from soup_cli.utils.loop_daemon import WatchConfig, run_once

    config = WatchConfig(
        max_iterations=1,
        harvest_fn=lambda s: {"pairs_harvested": 1, "pairs_path": "p.jsonl"},
        train_fn=lambda s, c: {"run_id": "r1", "skipped": False, "adapter_path": "a"},
        gate_fn=lambda s, c: ls._verdict_skipped("soup ship wrote no verdict (exit 1)"),
    )
    _, record, _ = run_once(_state(), config)
    assert record.gate_verdict == "SKIPPED"
    assert record.shipped is False
    assert "verdict_skipped: soup ship wrote no verdict (exit 1)" in record.notes


def test_prewired_config_uses_the_dispatcher():
    assert ls.build_prewired_watch_config(max_iterations=1).gate_fn is ls.verify_candidate


# ---------------------------------------------------------------------------
# soup loop init / status / watch
# ---------------------------------------------------------------------------


def test_init_defaults_to_ship_and_status_shows_it(cwd):
    result, text = _invoke(
        "init", "./model", "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired"
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_state().verify == "ship"
    assert "tasks.jsonl (verify: ship)" in text
    status, status_text = _invoke("status")
    assert re.search(r"verify\W+ship", status_text), status_text
    assert re.search(r"production\W+served model \(no adapter\)", status_text), status_text


@pytest.mark.parametrize("spelling", ["eval-gate", "eval_gate", "EVAL-GATE"])
def test_init_eval_gate_takes_a_suite(cwd, spelling):
    result, _ = _invoke(
        "init", "./model", "--eval", "gate.yaml", "--baseline", "b",
        "--pre-wired", "--verify", spelling,
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert read_state().verify == "eval_gate"


def test_init_pre_wired_refuses_the_lock_file_the_docs_used_to_pass(cwd):
    result, text = _invoke(
        "init", "./model", "--eval", "lock.json", "--baseline", "b", "--pre-wired"
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "invalid --eval / --verify:" in text
    assert "is a soup eval lock design file" in text
    assert not os.path.exists(os.path.join(".soup", "loop.yaml"))


def test_init_without_pre_wired_warns_about_a_bad_eval(cwd):
    result, text = _invoke("init", "./model", "--eval", "lock.json", "--baseline", "b")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "soup loop watch --pre-wired refuses until it is fixed" in text


def test_init_refuses_an_unknown_verify_even_without_pre_wired(cwd):
    result, text = _invoke(
        "init", "./model", "--eval", "tasks.jsonl", "--baseline", "b", "--verify", "vibes"
    )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "--verify must be ship or eval-gate" in text


def test_init_records_a_served_adapter_as_production(cwd):
    adapter = cwd / "adapters" / "v1"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base"}), encoding="utf-8"
    )
    result, _ = _invoke(
        "init", str(adapter), "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired"
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    state = read_state()
    assert state.production_adapter == str(adapter)
    assert state.base_model == "org/base"


def test_watch_pre_wired_refuses_an_old_loop_with_a_lock_file(cwd):
    os.makedirs(".soup")
    legacy = {"served_model": "./m", "eval_suite": "lock.json", "baseline": "b",
              "status": "stopped", "base_model": "org/base"}
    with open(os.path.join(".soup", "loop.yaml"), "w", encoding="utf-8") as fh:
        json.dump(legacy, fh)
    assert read_state().verify is None
    result, text = _invoke("watch", "--pre-wired", "--max-iterations", "1", "--poll-interval", "1")
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "is a soup eval lock design file" in text
    assert read_state().iteration_count == 0


def test_verify_field_is_validated():
    with pytest.raises(ValueError, match="verify must be one of"):
        _state(verify="vibes")
