"""Refused deploys back off, survive restarts and hold only after three attempts."""

import re
from datetime import datetime, timedelta, timezone

import pytest

from soup_cli.utils.loop_daemon import WatchConfig, run_once
from soup_cli.utils.loop_retry import DEPLOY_FAILURE_FINGERPRINT
from soup_cli.utils.loop_state import LoopState, read_state, write_state

CURSOR = 1_800_000_000.0
REFUSED = {"deployed": False, "failed": True, "notes": "HTTP 503: serve restarting"}
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).replace("│", " ").split())


def test_refused_deploy_waits_five_then_thirty_minutes_and_holds_after_three(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    state_path = str(tmp_path / ".soup" / "loop.yaml")
    write_state(
        LoopState(served_model="base", eval_suite="e.json", baseline="b", status="running"),
        state_path,
    )
    calls = {"cost": 0, "harvest": 0, "train": 0, "deploy": 0}

    def harvest(_state):
        calls["harvest"] += 1
        return {"pairs_harvested": 3, "traces_collected": 3, "cursor_ts": CURSOR}

    def train(_state, _harvest):
        calls["train"] += 1
        return {"run_id": "run-1", "skipped": False, "adapter_path": "out/a1"}

    def deploy(_state, _out):
        calls["deploy"] += 1
        return REFUSED

    def cost(_state):
        calls["cost"] += 1
        return 0.0

    config = WatchConfig(
        state_path=state_path,
        iteration_dir=str(tmp_path / ".soup-loops"),
        harvest_fn=harvest,
        train_fn=train,
        gate_fn=lambda *_: {"gate_verdict": "OK"},
        deploy_fn=deploy,
        cost_fn=cost,
    )
    start = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)

    state, first, _ = run_once(read_state(state_path), config, now=start)
    write_state(state, state_path)
    assert state.failed_deploy["attempts"] == 1
    assert state.pending_deploy is not None
    assert state.attempt_cursor_ts is None
    assert "2026-11-18T12:05:00+00:00" in first.notes

    # A restarted daemon can poll four times while `soup serve` is still restarting.
    # Each poll must retain the candidate and avoid repeating any stage.
    for minute in range(1, 5):
        restarted = read_state(state_path)
        state, waiting, decision = run_once(
            restarted, config, now=start + timedelta(minutes=minute)
        )
        write_state(state, state_path)
        assert not decision.proceed
        assert calls == {"cost": 1, "harvest": 1, "train": 1, "deploy": 1}
        assert state.pending_deploy == restarted.pending_deploy
        assert "deploy_backoff" in waiting.notes

    state, second, _ = run_once(state, config, now=start + timedelta(minutes=5))
    write_state(state, state_path)
    assert calls["deploy"] == 2
    assert state.failed_deploy["attempts"] == 2
    assert state.attempt_cursor_ts is None
    assert "2026-11-18T12:35:00+00:00" in second.notes

    state, _, decision = run_once(state, config, now=start + timedelta(minutes=34))
    assert not decision.proceed
    assert calls["deploy"] == 2

    state, third, _ = run_once(state, config, now=start + timedelta(minutes=35))
    assert calls == {"cost": 3, "harvest": 1, "train": 1, "deploy": 3}
    assert state.failed_deploy["attempts"] == 3 and state.failed_deploy["held"] is True
    assert state.failed_deploy["retry_after"] is None
    assert state.pending_deploy is not None
    assert state.attempt_cursor_ts == CURSOR
    assert "soup loop resume" in third.notes

    state, held, decision = run_once(state, config, now=start + timedelta(minutes=36))
    assert not decision.proceed
    assert calls["deploy"] == 3
    assert "deploy_held" in held.notes


def test_v078_state_without_failed_deploy_field_loads_with_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".soup" / "loop.yaml"
    pending = {
        "train": {"run_id": "run-legacy", "adapter_path": "out/legacy"},
        "gate": {"gate_verdict": "OK"},
        "attempts": 1,
        "cursor_ts": CURSOR,
        "iteration_id": "iter-legacy",
    }
    write_state(
        LoopState(
            served_model="base", eval_suite="e.json", baseline="b", status="running",
            pending_deploy=pending,
        ),
        str(path),
    )
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("failed_deploy")
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = read_state(str(path))
    assert loaded.failed_deploy is None
    assert loaded.pending_deploy == pending
    assert read_state(str(path)).pending_deploy == pending


def test_v078_pending_candidate_attempt_seeds_deploy_backoff(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pending = {
        "train": {"run_id": "run-legacy", "adapter_path": "out/legacy"},
        "gate": {"gate_verdict": "OK"},
        "attempts": 1,
        "cursor_ts": CURSOR,
        "iteration_id": "iter-legacy",
    }
    state = LoopState(
        served_model="base", eval_suite="e.json", baseline="b", status="running",
        pending_deploy=pending,
    )
    config = WatchConfig(
        state_path=str(tmp_path / ".soup" / "loop.yaml"),
        iteration_dir=str(tmp_path / ".soup-loops"),
        deploy_fn=lambda *_: REFUSED,
        cost_fn=lambda _state: 0.0,
    )

    retried, _, _ = run_once(
        state, config, now=datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
    )

    assert retried.failed_deploy["attempts"] == 2
    assert retried.failed_deploy["retry_after"] == "2026-11-18T12:30:00+00:00"
    assert retried.pending_deploy["attempts"] == 2


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"inputs": "not-a-fingerprint", "attempts": 1, "last_at": "t",
         "retry_after": "t", "held": False, "reason": "deploy refused"},
        {"inputs": DEPLOY_FAILURE_FINGERPRINT, "attempts": True, "last_at": "t",
         "retry_after": "t", "held": False, "reason": "deploy refused"},
        {"inputs": DEPLOY_FAILURE_FINGERPRINT, "attempts": 1, "last_at": "t",
         "retry_after": "t", "held": False, "reason": "deploy refused", "extra": 1},
    ],
)
def test_malformed_failed_deploy_state_is_refused(bad):
    with pytest.raises((ValueError, TypeError), match="failed_deploy"):
        LoopState(
            served_model="base", eval_suite="e.json", baseline="b", status="running",
            failed_deploy=bad,
        )


@pytest.mark.parametrize(
    ("held", "retry_delta", "expected"),
    [
        (False, 5, "waiting:"),
        (False, -5, "due:"),
        (True, None, "held:"),
    ],
)
def test_status_pending_deploy_reports_waiting_due_and_held(
    tmp_path, monkeypatch, held, retry_delta, expected
):
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.commands.loop import app

    now = datetime.now(timezone.utc)
    pending = {
        "train": {"run_id": "run-1", "adapter_path": "out/a1"},
        "gate": {"gate_verdict": "OK"}, "attempts": 1, "cursor_ts": CURSOR,
        "iteration_id": "iter-status",
    }
    failed = {
        "inputs": DEPLOY_FAILURE_FINGERPRINT, "attempts": 1,
        "last_at": now.isoformat(),
        "retry_after": None if retry_delta is None else
        (now + timedelta(minutes=retry_delta)).isoformat(),
        "held": held, "reason": "HTTP [bold red]503[/]",
    }
    write_state(
        LoopState(
            served_model="base", eval_suite="e.json", baseline="b", status="running",
            pending_deploy=pending, failed_deploy=failed,
        ),
        str(tmp_path / ".soup" / "loop.yaml"),
    )

    result = CliRunner().invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    plain = _plain(result.output)
    assert "pending_deploy" in plain and expected in plain
    # The refusal is untrusted terminal text; Rich markup must remain literal.
    assert "[bold red]503[/]" in plain


def test_restarted_daemon_deploys_saved_candidate_when_server_recovers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path = str(tmp_path / ".soup" / "loop.yaml")
    write_state(
        LoopState(served_model="base", eval_suite="e.json", baseline="b", status="running"),
        state_path,
    )
    calls = {"cost": 0, "harvest": 0, "train": 0, "deploy": 0}

    def deploy(_state, _out):
        calls["deploy"] += 1
        if calls["deploy"] == 1:
            return REFUSED
        return {"deployed": True, "failed": False, "notes": "HTTP 200"}

    config = WatchConfig(
        state_path=state_path,
        iteration_dir=str(tmp_path / ".soup-loops"),
        harvest_fn=lambda _state: calls.__setitem__("harvest", calls["harvest"] + 1)
        or {"pairs_harvested": 3, "traces_collected": 3, "cursor_ts": CURSOR},
        train_fn=lambda _state, _harvest: calls.__setitem__("train", calls["train"] + 1)
        or {"run_id": "run-1", "skipped": False, "adapter_path": "out/a1"},
        gate_fn=lambda *_: {"gate_verdict": "OK"},
        deploy_fn=deploy,
        cost_fn=lambda _state: calls.__setitem__("cost", calls["cost"] + 1) or 0.0,
    )
    start = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)

    state, _, _ = run_once(read_state(state_path), config, now=start)
    write_state(state, state_path)
    saved_candidate = state.pending_deploy
    assert saved_candidate is not None

    # Recreate the process state from disk on each poll: the service is unavailable
    # longer than the old three-poll retry window, then recovers at the scheduled retry.
    for minute in range(1, 5):
        restarted = read_state(state_path)
        state, _, decision = run_once(restarted, config, now=start + timedelta(minutes=minute))
        write_state(state, state_path)
        assert not decision.proceed
        assert state.pending_deploy == saved_candidate
        assert calls == {"cost": 1, "harvest": 1, "train": 1, "deploy": 1}

    state, shipped, decision = run_once(
        read_state(state_path), config, now=start + timedelta(minutes=5)
    )
    assert decision.proceed
    assert shipped.shipped is True
    assert state.pending_deploy is None
    assert state.failed_deploy is None
    assert state.attempt_cursor_ts == CURSOR
    assert calls == {"cost": 2, "harvest": 1, "train": 1, "deploy": 2}


def test_resume_clears_deploy_hold_but_keeps_candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.commands.loop import app

    path = str(tmp_path / ".soup" / "loop.yaml")
    pending = {
        "train": {"run_id": "run-1", "adapter_path": "out/a1"},
        "gate": {"gate_verdict": "OK"},
        "attempts": 3,
        "cursor_ts": CURSOR,
        "iteration_id": "iter-20261118T120000-abcdef01",
    }
    failed = {
        "inputs": DEPLOY_FAILURE_FINGERPRINT,
        "attempts": 3,
        "last_at": "2026-11-18T12:35:00+00:00",
        "retry_after": None,
        "held": True,
        "reason": "deploy refused: HTTP 503",
    }
    state = LoopState(
        served_model="base", eval_suite="e.json", baseline="b", status="running",
        pending_deploy=pending, failed_deploy=failed, attempt_cursor_ts=CURSOR,
    )
    write_state(state, path)
    result = CliRunner().invoke(app, ["resume"])

    assert result.exit_code == 0, result.output
    assert "deploy hold cleared" in _plain(result.output)
    resumed = read_state(path)
    assert resumed.failed_deploy is None
    assert resumed.pending_deploy is not None
    assert resumed.pending_deploy["attempts"] == 0

    config = WatchConfig(
        state_path=path,
        iteration_dir=str(tmp_path / ".soup-loops"),
        deploy_fn=lambda *_: REFUSED,
        cost_fn=lambda _state: 0.0,
    )
    retried, record, _ = run_once(
        resumed, config, now=datetime(2026, 11, 18, 13, 0, tzinfo=timezone.utc)
    )
    assert retried.failed_deploy["attempts"] == 1
    assert retried.pending_deploy["attempts"] == 1
    assert retried.attempt_cursor_ts == CURSOR
    assert "attempt 1 of 3" in record.notes
