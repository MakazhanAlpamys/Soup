"""Watch-daemon orchestrator for `soup loop watch` (v0.58.0).

The full production cycle is:

    traces (v0.26 from-traces) → preference pairs → DPO train
        → eval-gate (v0.26.0 Part B) → optional canary deploy
        → rollback on regression (v0.26.0 Quant-Lobotomy MAJOR)

Each stage is encapsulated as a callable so the daemon stays testable
without a GPU. A *headless* run with the default stage callbacks
exercises every state transition (state mutations, budget check,
iteration record, sticky rollback) deterministically.

The daemon is foreground by default. The CLI ``--detach`` flag launches
a subprocess via ``subprocess.Popen([sys.executable, "-m", "soup_cli.cli",
"loop", "watch", "--foreground"])`` so the operator gets a real process
id back instead of relying on shell job control.
"""

from __future__ import annotations

import logging
import math
import signal
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional

from soup_cli.utils.canary_router import (
    BucketStats,
    CanaryPolicy,
    QualityStats,
    read_bucket_stats,
    rollback,
)
from soup_cli.utils.loop_budget import (
    BudgetDecision,
    check_budget,
    reset_daily_counter_if_new_day,
    reset_monthly_spend_if_new_month,
)
from soup_cli.utils.loop_iteration import (
    IterationRecord,
    new_iteration_id,
    write_iteration,
)
from soup_cli.utils.loop_log import (
    NO_LOG,
    LoopLog,
    StageEvents,
    canary_result,
    iteration_from_record,
    progress_line,
    promotion_changes,
    resolved_line,
    start_log,
    utc_now_iso,
    version_line,
    withdrawn_line,
)
from soup_cli.utils.loop_state import (
    MAX_CURSOR_TS,
    LoopState,
    format_wait_duration,
    read_state,
    write_state,
)

_LOG = logging.getLogger(__name__)
# BucketStats.verdict's default: canary answers needed before a verdict.
CANARY_MIN_SAMPLES = 30

# A per-stage callable returns a small dict; the orchestrator merges the
# result dicts into one ``StageResult`` before recording the iteration.

HarvestFn = Callable[[LoopState], Mapping[str, object]]
TrainFn = Callable[[LoopState, Mapping[str, object]], Mapping[str, object]]
GateFn = Callable[[LoopState, Mapping[str, object]], Mapping[str, object]]
DeployFn = Callable[[LoopState, Mapping[str, object]], Mapping[str, object]]
CostFn = Callable[[LoopState], float]
PromoteFn = Callable[[LoopState], Mapping[str, object]]
# What people said about the active canary's answers; None = the quality verdict takes no part.
QualityFn = Callable[[LoopState], Optional[QualityStats]]


# ---------------------------------------------------------------------------
# Default stage callbacks — pure-Python no-ops that satisfy the contract so
# the daemon runs end-to-end on CPU without a model. Real wiring composes
# v0.26.0 trace-to-pref + v0.26.0 eval-gate + v0.30.0 multi-adapter deploy.
# ---------------------------------------------------------------------------


def default_harvest(state: LoopState) -> Mapping[str, object]:
    """Stub harvest stage — returns zero pairs.

    Real implementation wires v0.26.0 ``soup_cli.data.traces.parsers`` +
    ``pair_builder.build_pairs`` to scan trace logs. Kept as a stub here
    so headless tests don't need a live trace store.
    """
    return {"pairs_harvested": 0, "pairs_path": None}


def default_train(state: LoopState, ctx: Mapping[str, object]) -> Mapping[str, object]:
    """Stub train stage — skips the run."""
    return {"run_id": None, "skipped": True}


def default_gate(state: LoopState, ctx: Mapping[str, object]) -> Mapping[str, object]:
    """Stub gate stage — returns SKIPPED when there is nothing to evaluate."""
    if ctx.get("skipped"):
        return {"gate_verdict": "SKIPPED"}
    return {"gate_verdict": "OK"}


def default_deploy(state: LoopState, ctx: Mapping[str, object]) -> Mapping[str, object]:
    """Stub deploy stage — does not promote anything."""
    return {"deployed": False, "canary_verdict": None}


def default_promote(state: LoopState) -> Mapping[str, object]:
    """Stub promote stage — the state change alone, no server to tell."""
    return {"promoted": True}


def default_cost(state: LoopState) -> float:
    """Stub cost estimate — zero so the budget gate stays permissive."""
    return 0.0


@dataclass
class WatchConfig:
    """Daemon configuration knobs."""

    poll_interval_sec: float = 60.0
    max_iterations: Optional[int] = None  # None = unbounded (real daemon)
    state_path: Optional[str] = None
    canary_stats_path: Optional[str] = None
    iteration_dir: Optional[str] = None
    harvest_fn: HarvestFn = default_harvest
    train_fn: TrainFn = default_train
    gate_fn: GateFn = default_gate
    deploy_fn: DeployFn = default_deploy
    cost_fn: CostFn = default_cost
    promote_fn: PromoteFn = default_promote
    on_iteration: Optional[Callable[[IterationRecord], None]] = None
    # v0.71.4 #177 — pack each successful iteration as a v0.26 Soup Can +
    # append a Registry entry (default off so existing tests / stub watchers
    # have no registry side effects).
    pack_iterations: bool = False
    served_model: Optional[str] = None
    base_model: str = "unknown"
    # The feedback side of the canary verdict (loop_stages.canary_quality for the pre-wired
    # loop); None = judge by request errors alone, as the stub stages do.
    quality_fn: Optional[QualityFn] = None
    # Append the iteration, version and progress logs (docs/loop-formats.md) under
    # ``iteration_dir`` (default .soup-loops) as the loop runs. False = only the manifests.
    loop_logs: bool = True

    def __post_init__(self) -> None:
        v = self.poll_interval_sec
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("poll_interval_sec must be a finite number")
        if v < 1.0 or v > 3600.0:
            raise ValueError("poll_interval_sec must be in [1, 3600]")
        if self.max_iterations is not None:
            mi = self.max_iterations
            if isinstance(mi, bool) or not isinstance(mi, int) or mi < 0:
                raise ValueError("max_iterations must be a non-negative int or None")
        for fname in ("harvest_fn", "train_fn", "gate_fn", "deploy_fn", "cost_fn", "promote_fn"):
            if not callable(getattr(self, fname)):
                raise ValueError(f"{fname} must be callable")
        if self.on_iteration is not None and not callable(self.on_iteration):
            raise ValueError("on_iteration must be callable or None")
        if self.quality_fn is not None and not callable(self.quality_fn):
            raise ValueError("quality_fn must be callable or None")
        if not isinstance(self.pack_iterations, bool):
            raise ValueError("pack_iterations must be bool")
        if not isinstance(self.loop_logs, bool):
            raise ValueError("loop_logs must be bool")
        # Validate the #177 registry-packing identity fields like every other
        # field, so a NUL/oversize value fails at construction rather than deep
        # inside store.push (review LOW). They flow into registry_name_from /
        # store.push(base_model=...).
        if self.served_model is not None:
            sm = self.served_model
            if not isinstance(sm, str) or "\x00" in sm or len(sm) > 512:
                raise ValueError(
                    "served_model must be a NUL-free str <= 512 chars or None"
                )
        bm = self.base_model
        if not isinstance(bm, str) or not bm or "\x00" in bm or len(bm) > 512:
            raise ValueError("base_model must be a non-empty NUL-free str <= 512 chars")


def _now() -> datetime:
    """The clock ``run_once`` reads when it is not given ``now`` (tests replace this)."""
    return datetime.now(timezone.utc)


def run_once(
    state: LoopState,
    config: WatchConfig,
    *,
    now: Optional[datetime] = None,
    log: Optional[LoopLog] = None,
) -> "tuple[LoopState, IterationRecord, BudgetDecision]":
    """Execute one full iteration synchronously. Pure with respect to time.

    ``now`` fixes the clock for the day and month rollovers and for how long a canary has
    waited for ratings (tests); None reads ``_now()``, the current UTC time.

    ``log`` is where the cycle appends its lines to ``iterations.jsonl``, ``versions.jsonl``
    and ``progress.jsonl`` (``soup loop watch`` passes one); None writes nothing and leaves
    the version counter alone. A line that cannot be written is a note on the returned
    record, never an error.
    """
    if not isinstance(state, LoopState):
        raise TypeError("state must be LoopState")
    if not isinstance(config, WatchConfig):
        raise TypeError("config must be WatchConfig")
    log = NO_LOG if log is None else log
    # #2362: read the clock once, so the month and day rollovers agree even
    # when `soup loop watch` runs across midnight at the end of a month.
    if now is None:
        now = _now()
    if log.enabled:
        state = start_log(state, log, actor="loop")  # the first version, once
    # Read the month from the OLD last_run_date, before the daily reset
    # overwrites it.
    spent = reset_monthly_spend_if_new_month(
        state.spent_this_month_usd, state.last_run_date, now=now
    )
    runs_today, today = reset_daily_counter_if_new_day(
        state.runs_today, state.last_run_date, now=now
    )
    state = _state_with(
        state,
        runs_today=runs_today,
        last_run_date=today,
        spent_this_month_usd=spent,
    )
    reading = _evaluate_active_canary(state, config, now)
    canary_verdict, rollback_policy = reading.verdict, reading.rollback
    canary_samples, quality_notes = reading.samples, list(reading.notes)
    rolled_back = rollback_policy is not None
    canary_notes = []
    if rollback_policy is not None:
        canary_notes.append(
            reading.rollback_note
            or f"canary {state.canary_active} rolled back ({canary_verdict})"
        )
        canary_notes.extend(quality_notes)
        _log_withdrawn(log, state, reading, canary_notes[0])
        state = _state_with(
            state,
            canary_active=rollback_policy.canary,
            canary_traffic_pct=rollback_policy.traffic_pct,
            canary_rollout_id=None,
            canary_adapter=None,
            canary_started_ts=None,
            canary_iteration=None,
        )
    elif canary_verdict == "OK" and state.canary_active:
        canary_before = state
        log.progress(progress_line(
            _canary_iteration_id(state), "promote", "start",
            message=f"promoting {state.canary_active} to production",
        ))
        state, note = _promote(state, config)
        canary_notes.append(note)
        if reading.promote_note and state.canary_active is None:
            canary_notes.append(reading.promote_note)  # (a refused promotion keeps the canary)
        canary_notes.extend(quality_notes)
        _log_promotion(log, canary_before, state, reading, canary_notes, note)
    elif state.canary_active:
        # A canary is still collecting answers: no new iteration until it is decided.
        _log_waiting(log, state, reading)
        record = IterationRecord(
            iteration_id=new_iteration_id(),
            started_at=_utc_iso(),
            finished_at=_utc_iso(),
            pairs_harvested=0,
            run_id=None,
            gate_verdict="SKIPPED",
            canary_verdict=canary_verdict,
            shipped=False,
            rolled_back=False,
            estimated_cost_usd=0.0,
            notes=" | ".join([
                f"canary-wait: {state.canary_active} has {canary_samples} canary answers "
                f"(verdict at {CANARY_MIN_SAMPLES})",
                *quality_notes,
            ])[:4096],
            canary_quality=reading.evidence,
        )
        return state, _noted(record, log), BudgetDecision(
            proceed=False,
            reason="canary in progress",
            projected_total_usd=state.spent_this_month_usd,
            runs_today=state.runs_today,
        )
    estimated = float(config.cost_fn(state))
    decision = check_budget(
        estimated_run_usd=estimated,
        spent_so_far_usd=state.spent_this_month_usd,
        monthly_budget_usd=state.monthly_budget_usd,
        runs_today=state.runs_today,
        max_runs_per_day=state.max_runs_per_day,
    )
    iteration_id = new_iteration_id()
    started_at = _utc_iso()
    if not decision.proceed:
        record = IterationRecord(
            iteration_id=iteration_id,
            started_at=started_at,
            finished_at=_utc_iso(),
            pairs_harvested=0,
            run_id=None,
            gate_verdict="SKIPPED",
            canary_verdict=canary_verdict,
            shipped=False,
            rolled_back=rolled_back,
            estimated_cost_usd=estimated,
            notes=" | ".join([*canary_notes, f"budget-skip: {decision.reason}"])[:4096],
            canary_quality=reading.evidence,
        )
        return state, _noted(record, log), decision
    # Stage events: held back until the iteration has work to show (see StageEvents).
    events = StageEvents(log, iteration_id)
    events.emit("harvest", "start")
    harvest_out = dict(config.harvest_fn(state))
    events.emit("harvest", "end", message=_harvest_message(harvest_out))
    if harvest_out.get("pairs_path"):
        events.go_live()  # a dataset was handed to training
    events.emit("train", "start")
    train_out = dict(config.train_fn(state, harvest_out))
    trained_run = _training_ran(train_out)
    if trained_run:
        events.go_live()  # (a custom stage can train without a dataset path)
    _emit_train_end(events, train_out, trained_run)
    if trained_run:
        events.emit("verdict", "start")
    gate_out = dict(config.gate_fn(state, train_out))
    if trained_run:
        _emit_verdict_end(events, gate_out)
    deploying = trained_run and gate_out.get("gate_verdict") == "OK"
    if deploying:
        events.emit("deploy", "start")
    deploy_out = dict(config.deploy_fn(state, {**train_out, **gate_out}))
    if deploying:
        _emit_deploy_end(events, deploy_out)
    gate_verdict = str(gate_out.get("gate_verdict", "SKIPPED"))
    deploy_verdict = deploy_out.get("canary_verdict")
    if canary_verdict is None and deploy_verdict is not None:
        canary_verdict = str(deploy_verdict)
    shipped = bool(deploy_out.get("deployed", False))
    rolled_back = rolled_back or bool(deploy_out.get("rolled_back", False))
    pairs = int(harvest_out.get("pairs_harvested", 0) or 0)
    if pairs < 0:
        pairs = 0
    record = IterationRecord(
        iteration_id=iteration_id,
        started_at=started_at,
        finished_at=_utc_iso(),
        pairs_harvested=pairs,
        run_id=(str(train_out["run_id"]) if train_out.get("run_id") else None),
        gate_verdict=gate_verdict if gate_verdict in ("OK", "MAJOR", "SKIPPED") else "SKIPPED",
        canary_verdict=(
            canary_verdict
            if canary_verdict in (None, "OK", "MAJOR", "UNKNOWN")
            else None
        ),
        shipped=shipped,
        rolled_back=rolled_back,
        estimated_cost_usd=estimated,
        notes=" | ".join(
            str(n)
            for n in (
                *canary_notes,
                train_out.get("notes"),
                gate_out.get("notes"),
                deploy_out.get("notes"),
            )
            if n
        )[:4096],
        canary_quality=reading.evidence,
    )
    # The lifetime totals count what training used: an iteration that only waited for
    # more feedback (or whose training failed) would otherwise add the whole pile again.
    trained = _training_ran(train_out)
    new_state = state.bumped(
        traces_collected=int(harvest_out.get("traces_collected", 0) or 0) if trained else 0,
        pairs_distilled=pairs if trained else 0,
        runs_gated=1 if record.gate_verdict in ("OK", "MAJOR") else 0,
        adapters_shipped=1 if shipped else 0,
        iteration_count=1,
        runs_today=1,
    )
    new_state = _state_with(
        new_state,
        spent_this_month_usd=new_state.spent_this_month_usd + max(0.0, estimated),
        last_iteration_id=iteration_id,
        last_run_date=today,
    )
    new_state = _advance_cursor(new_state, harvest_out, train_out)
    if shipped and deploy_out.get("canary_active"):
        new_state = _state_with(
            new_state,
            canary_active=str(deploy_out["canary_active"]),
            canary_adapter=(
                str(deploy_out["canary_adapter"]) if deploy_out.get("canary_adapter") else None
            ),
            canary_traffic_pct=float(deploy_out.get("canary_traffic_pct") or 0.0),
            canary_rollout_id=str(deploy_out.get("canary_rollout_id") or uuid.uuid4().hex),
            canary_started_ts=_rollout_start(deploy_out),
            canary_autoroll_on_regress=True,
        )
    new_state = _log_iteration(log, events, record, state, new_state, {
        "harvest": harvest_out, "train": train_out, "gate": gate_out, "deploy": deploy_out,
    })
    return new_state, _noted(record, log), decision


# ---------------------------------------------------------------------------
# The cycle's log lines (docs/loop-formats.md). Each helper only appends: it decides
# nothing, and a line that cannot be written is a note, never an error (LoopLog).
# ---------------------------------------------------------------------------


def _noted(record: IterationRecord, log: LoopLog) -> IterationRecord:
    """``record`` with a note for each log line that could not be written this cycle."""
    failures = log.take_failures()
    if not failures:
        return record
    note = " | ".join(failures)
    notes = f"{record.notes} | {note}" if record.notes else note
    return replace(record, notes=notes[:4096])


def _canary_iteration_id(state: LoopState) -> Optional[str]:
    """The iteration the running canary belongs to, or None (a canary no iteration started)."""
    snapshot = state.canary_iteration if state.canary_active else None
    return snapshot["iteration_id"] if snapshot else None


def _canary_of(state: LoopState, reading: "_CanaryReading") -> dict:
    """The ``canary`` object of a resolved canary: what this cycle measured of it."""
    measured = reading.stable_samples is not None
    return canary_result(
        traffic_pct=state.canary_traffic_pct,
        verdict=reading.verdict,
        stable_samples=reading.stable_samples,
        canary_samples=reading.samples if measured else None,
        quality=reading.evidence,
    )


def _log_withdrawn(
    log: LoopLog, state: LoopState, reading: "_CanaryReading", reason: str
) -> None:
    """The canary was withdrawn: its iteration ends ``rolled_back``. Production is unchanged."""
    line = withdrawn_line(state, reason, canary=_canary_of(state, reading))
    if line is not None:
        log.iteration(line)
    log.progress(progress_line(
        _canary_iteration_id(state), "rollback", "end", message=reason
    ))


def _log_promotion(
    log: LoopLog,
    before: LoopState,
    after: LoopState,
    reading: "_CanaryReading",
    canary_notes: "list[str]",
    note: str,
) -> None:
    """The canary became production (or the server refused): the iteration, version, event."""
    iteration_id = _canary_iteration_id(before)
    if after.canary_active is not None:  # refused: nothing changed
        log.progress(progress_line(
            iteration_id, "promote", "error", error="promote_refused", message=note
        ))
        return
    now_iso = utc_now_iso()
    reason = " | ".join(canary_notes)[:4096]
    snapshot = before.canary_iteration
    if snapshot is not None:
        log.iteration(resolved_line(
            snapshot, outcome="promoted", reason=reason, finished_at=now_iso,
            to_version=after.production_version, canary=_canary_of(before, reading),
        ))
    log.version(version_line(
        from_version=before.production_version, to_version=after.production_version,
        cause="promote", actor="loop", iteration_id=iteration_id,
        served_as=before.canary_active, reason=reason, ts=now_iso,
    ))
    log.progress(progress_line(iteration_id, "promote", "end", message=note, ts=now_iso))


def _log_waiting(log: LoopLog, state: LoopState, reading: "_CanaryReading") -> None:
    """A canary still collecting answers: one ``canary`` progress line for this cycle.

    The numbers are the canary's rated answers out of the minimum the quality verdict needs
    when people rate them, else its answers out of ``CANARY_MIN_SAMPLES``; None when the
    statistics could not be read.
    """
    evidence = reading.evidence
    if evidence is not None:
        rated, total = evidence["rated"], evidence["min_rated"]
        done = rated["canary"]
        message = (
            f"{rated['canary']}/{total} canary and {rated['stable']}/{total} stable answers rated"
        )
    elif reading.verdict is not None:
        done, total = reading.samples, CANARY_MIN_SAMPLES
        message = f"{done}/{total} canary answers"
    else:
        done = total = None
        message = "canary statistics unavailable"
    log.progress(progress_line(
        _canary_iteration_id(state), "canary", "progress", done=done, total=total, message=message
    ))


def _harvest_message(harvest_out: Mapping[str, object]) -> str:
    pairs, new = harvest_out.get("pairs_harvested"), harvest_out.get("new_pairs")
    text = f"{pairs} rows harvested" if isinstance(pairs, int) else "harvested"
    return f"{text} ({new} new)" if isinstance(new, int) and not isinstance(new, bool) else text


def _emit_train_end(
    events: StageEvents, train_out: Mapping[str, object], trained: bool
) -> None:
    if trained:
        events.emit("train", "end", message=f"trained {train_out.get('run_id')}")
    elif train_out.get("failed"):
        events.emit("train", "error", error="train_failed", message="training failed")
    else:
        events.emit("train", "end", message="training skipped")


def _emit_verdict_end(events: StageEvents, gate_out: Mapping[str, object]) -> None:
    verdict = gate_out.get("gate_verdict")
    notes = str(gate_out.get("notes") or "")
    if verdict == "OK":
        events.emit("verdict", "end", message="SHIP")
    elif verdict == "MAJOR":
        events.emit("verdict", "end", message="DON'T SHIP")
    elif notes.startswith("verdict_skipped:"):
        events.emit("verdict", "error", error="verdict_not_reached", message=notes[:300])
    else:
        events.emit("verdict", "end", message="no verdict")


def _emit_deploy_end(events: StageEvents, deploy_out: Mapping[str, object]) -> None:
    notes = str(deploy_out.get("notes") or "")
    if deploy_out.get("deployed"):
        events.emit("deploy", "end", message=notes or "deployed")
    elif deploy_out.get("failed"):
        events.emit("deploy", "error", error="deploy_failed", message=notes or "deploy failed")
    else:
        events.emit("deploy", "end", message=notes or "not deployed")


def _log_iteration(
    log: LoopLog,
    events: StageEvents,
    record: IterationRecord,
    state: LoopState,
    new_state: LoopState,
    stages: "Mapping[str, Mapping[str, object]]",
) -> LoopState:
    """Append the iteration's line; a canary keeps its line in ``loop.yaml`` for later.

    Returns ``new_state``, with ``canary_iteration`` set when the iteration started a canary.
    """
    if not log.enabled:
        return new_state
    line = iteration_from_record(
        record, state=state, new_state=new_state, harvest=stages["harvest"],
        train=stages["train"], gate=stages["gate"], deploy=stages["deploy"],
    )
    log.iteration(line)
    if line["outcome"] != "canary":
        return new_state
    try:
        new_state = _state_with(new_state, canary_iteration=line)
    except (TypeError, ValueError):
        # (a stage reported something the snapshot cannot hold): the canary still runs,
        # but its promotion will not be able to complete this line.
        _LOG.warning("canary iteration snapshot not kept", exc_info=True)
    events.go_live()  # (a canary started: that is work, whatever the stages reported)
    events.emit(
        "canary", "start", done=0, total=CANARY_MIN_SAMPLES,
        message=f"{new_state.canary_traffic_pct:g}% of keyed traffic; verdict after "
        f"{CANARY_MIN_SAMPLES} canary answers",
    )
    return new_state


def _rollout_start(deploy_out: Mapping[str, object]) -> float:
    """When the canary the deploy stage started began: its own stamp, else now.

    A custom ``deploy_fn`` that reports none (or a value ``LoopState`` would refuse) gets
    the time of this cycle, so the quality verdict still has a start to count from.
    """
    started = deploy_out.get("canary_started_ts")
    if (
        isinstance(started, bool)
        or not isinstance(started, (int, float))
        or not (0 <= started <= MAX_CURSOR_TS)  # False for NaN too
    ):
        return datetime.now(timezone.utc).timestamp()
    return float(started)


def _training_ran(train_out: Mapping[str, object]) -> bool:
    """True when the train stage produced a run (not skipped, not failed)."""
    return bool(train_out.get("run_id")) and not train_out.get("skipped")


def _advance_cursor(
    state: LoopState, harvest_out: Mapping[str, object], train_out: Mapping[str, object]
) -> LoopState:
    """Move the data cursor to what the harvest saw, after a training run.

    Only a run counts: a skip (not enough new data, no pairs) or a failed training
    leaves it, so the same feedback is picked up again and accumulates. The verdict
    does not matter (a DON'T SHIP still consumed the data), and the cursor never
    moves back. A harvest that reports none, or a value ``LoopState`` would refuse
    (a custom ``harvest_fn``), is ignored.
    """
    cursor = harvest_out.get("cursor_ts")
    if not _training_ran(train_out):
        return state
    if isinstance(cursor, bool) or not isinstance(cursor, (int, float)):
        return state
    if not (0 <= cursor <= MAX_CURSOR_TS):  # False for NaN too
        return state
    if state.harvest_cursor_ts is not None and cursor <= state.harvest_cursor_ts:
        return state
    return _state_with(state, harvest_cursor_ts=float(cursor))


def _promote(state: LoopState, config: WatchConfig) -> "tuple[LoopState, str]":
    """Promote the active canary to production when the server agrees."""
    name = state.canary_active
    try:
        out = dict(config.promote_fn(state))
    except Exception as exc:  # noqa: BLE001 — the daemon must not crash
        out = {"promoted": False, "notes": type(exc).__name__}
    if not out.get("promoted"):
        why = out.get("notes") or "the promote stage failed"
        return state, f"canary {name} passed but was not promoted: {why}"
    promoted = _state_with(
        state,
        previous_production_name=state.production_name,
        previous_production_adapter=state.production_adapter,
        production_name=name,
        production_adapter=state.canary_adapter,
        canary_active=None,
        canary_traffic_pct=None,
        canary_rollout_id=None,
        canary_adapter=None,
        canary_started_ts=None,
        **promotion_changes(state),
    )
    note = f"canary {name} promoted to production"
    if out.get("notes"):
        note += f" ({out['notes']})"
    if state.canary_adapter is None:
        note += "; its adapter path is unknown, so the next verdict compares against the base"
    return promoted, note


@dataclass(frozen=True)
class _CanaryReading:
    """What one cycle learned about the active canary (``None``/empty: no canary)."""

    verdict: Optional[str] = None
    rollback: Optional[CanaryPolicy] = None  # set when the canary is to be withdrawn
    samples: int = 0
    # The stable bucket's answers, measured the same way; None when the statistics could not
    # be read (so the iteration line leaves the samples out instead of saying 0).
    stable_samples: Optional[int] = None
    notes: "tuple[str, ...]" = ()
    # The numbers behind the quality verdict (QualityStats.evidence), None without one.
    evidence: "Optional[dict[str, object]]" = None
    # Why a withdrawal that is not a MAJOR verdict happened (the maximum wait ran out).
    rollback_note: Optional[str] = None
    # Appended to the notes once the canary was promoted without its ratings.
    promote_note: Optional[str] = None


def _wait_is_over(state: LoopState, now: datetime) -> bool:
    """True when the rollout started at least ``canary_max_wait_sec`` before ``now``."""
    started = state.canary_started_ts
    if started is None:  # a rollout from before the field: nothing to count from
        return False
    return now.timestamp() - started >= state.canary_max_wait_sec


def _evaluate_active_canary(
    state: LoopState, config: WatchConfig, now: datetime
) -> _CanaryReading:
    """Read live bucket outcomes and feedback.

    The verdict is the request-error verdict alone, or, when ``config.quality_fn`` has
    feedback to judge by, the two combined (``combine_canary_verdicts``); the notes then
    record both. If the quality verdict would be ``UNKNOWN`` and the canary has waited
    ``state.canary_max_wait_sec`` since it started, it becomes ``INSUFFICIENT`` and
    ``state.canary_on_timeout`` decides: ``errors-only`` (the error verdict stands alone),
    ``rollback`` (withdraw it, whatever ``canary_autoroll_on_regress`` says) or ``wait``
    (no time limit: it stays ``UNKNOWN``).
    """
    active_canary = state.canary_active
    if active_canary is None:
        return _CanaryReading()
    try:
        policy = CanaryPolicy(
            stable=state.stable_name(),
            canary=active_canary,
            traffic_pct=float(state.canary_traffic_pct or 0.0),
        )
        stats = read_bucket_stats(
            stable=policy.stable,
            canary=active_canary,
            rollout_id=state.canary_rollout_id,
            path=config.canary_stats_path,
        )
        verdict = evaluate_canary_verdict(stats)
        counts = stats.snapshot()
        samples = counts["canary_ok"] + counts["canary_major"]
        stable_samples = counts["stable_ok"] + counts["stable_major"]
    except (OSError, TypeError, ValueError):
        _LOG.warning("canary stats read failed", exc_info=True)
        return _CanaryReading()
    quality = _read_quality(state, config)
    if quality is None:
        regressed = verdict == "MAJOR" and state.canary_autoroll_on_regress
        return _CanaryReading(
            verdict=verdict,
            rollback=maybe_rollback(policy, verdict) if regressed else None,
            samples=samples,
            stable_samples=stable_samples,
        )
    quality_verdict = quality.verdict(min_samples=CANARY_MIN_SAMPLES)
    timed_out = (
        quality_verdict == "UNKNOWN"
        and state.canary_on_timeout != "wait"
        and _wait_is_over(state, now)
    )
    if timed_out:
        quality_verdict = "INSUFFICIENT"
    # A verdict is made of shares; before it, show how far each side is from one.
    detail = (
        quality.summary()
        if quality_verdict in ("OK", "MAJOR")
        else quality.progress(CANARY_MIN_SAMPLES)
    )
    notes = (f"canary errors: {verdict}", f"canary quality: {detail} ({quality_verdict})")
    verdict = combine_canary_verdicts(verdict, quality_verdict)
    decided_on = "errors_and_quality"
    withdraw = None
    rollback_note = None
    promote_note = None
    if timed_out:
        waited = format_wait_duration(state.canary_max_wait_sec)
        missing = f"{quality.shortfall(CANARY_MIN_SAMPLES)} in {waited}"
        if state.canary_on_timeout == "rollback":
            # Withdrawn although nothing is known to be wrong, and whatever
            # --autoroll-on-regress says: the person asked for this on timeout.
            withdraw = rollback(policy, reason="canary quality unverified after the maximum wait")
            if verdict != "MAJOR":
                verdict = "UNKNOWN"
                rollback_note = (
                    f"canary {active_canary} rolled back: {missing}, quality unverified "
                    "(--canary-on-timeout rollback)"
                )
        else:
            decided_on = "errors_only"
            # Added to the notes only if the canary then is promoted (run_once: verdict OK).
            promote_note = f"promoted on errors only - quality unverified: {missing}"
    if withdraw is None and verdict == "MAJOR" and state.canary_autoroll_on_regress:
        withdraw = maybe_rollback(policy, verdict)
    return _CanaryReading(
        verdict=verdict,
        rollback=withdraw,
        samples=samples,
        stable_samples=stable_samples,
        notes=notes,
        evidence=quality.evidence(
            min_samples=CANARY_MIN_SAMPLES,
            verdict=quality_verdict,
            timed_out=timed_out,
            decided_on=decided_on,
        ),
        rollback_note=rollback_note,
        promote_note=promote_note,
    )


def _read_quality(state: LoopState, config: WatchConfig) -> Optional[QualityStats]:
    """What people said about the canary's answers, or ``None`` (no hook, no data, a failure).

    A failing read must not stop the loop or roll a canary back: it is logged and the
    request-error verdict stands alone for this cycle.
    """
    if config.quality_fn is None:
        return None
    try:
        quality = config.quality_fn(state)
    except (OSError, TypeError, ValueError):
        _LOG.warning("canary quality read failed", exc_info=True)
        return None
    return quality if isinstance(quality, QualityStats) else None


def combine_canary_verdicts(error_verdict: str, quality_verdict: str) -> str:
    """One canary verdict from the request-error verdict and the feedback verdict.

    Strict: ``MAJOR`` if either is, ``OK`` only when both are, ``UNKNOWN`` otherwise, so a
    canary is promoted only once its requests succeeded AND enough people rated its
    answers well enough. The exception is ``INSUFFICIENT`` (the maximum wait for ratings ran
    out): the ratings never came, so the error verdict stands alone.
    """
    if quality_verdict == "INSUFFICIENT":
        return error_verdict
    if "MAJOR" in (error_verdict, quality_verdict):
        return "MAJOR"
    if error_verdict == "OK" and quality_verdict == "OK":
        return "OK"
    return "UNKNOWN"


def watch(config: WatchConfig) -> "tuple[LoopState, int]":
    """Run the daemon loop. Returns ``(final_state, iterations_run)``.

    Stops cleanly on:
    - ``config.max_iterations`` reached (test/finite mode)
    - state file going to ``status="stopped"`` between iterations
    - SIGTERM/SIGINT (installed via ``signal.signal`` when on the main thread)
    """
    if not isinstance(config, WatchConfig):
        raise TypeError("config must be WatchConfig")
    stop = threading.Event()

    def _request_stop(signum, frame):  # pragma: no cover — signal path
        stop.set()

    try:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, _request_stop)
            signal.signal(signal.SIGINT, _request_stop)
    except (ValueError, AttributeError):
        # Non-main-thread + Windows-Python combos where signal.signal raises.
        pass

    iterations = 0
    log = LoopLog(config.iteration_dir) if config.loop_logs else None
    # v0.71.4 #177 — chain Registry entries across iterations so the loop
    # forms a real lineage DAG (parent links to the prior iteration's entry).
    prev_registry_id: Optional[str] = None
    state = read_state(config.state_path)
    # Only promote `stopped` → `running` automatically; `paused` must
    # survive a `soup loop watch` invocation so a SIGTERM + restart
    # cycle does not silently un-pause the daemon (code-review HIGH #2).
    if state.status == "stopped":
        state = state.with_status("running")
        write_state(state, config.state_path)
    try:
        while not stop.is_set():
            if config.max_iterations is not None and iterations >= config.max_iterations:
                break
            try:
                state = read_state(config.state_path)
            except (FileNotFoundError, ValueError):
                break
            if state.status == "paused":
                if stop.wait(min(config.poll_interval_sec, 60.0)):
                    break
                continue
            if state.status == "stopped":
                break
            state, record, decision = run_once(state, config, log=log)
            write_state(state, config.state_path)
            # Budget-skipped iterations DO NOT produce a manifest — the
            # cycle didn't actually run, so cluttering .soup-loops/ with
            # "I didn't run" records would surprise operators expecting
            # iteration_count to match the manifest count (code-review
            # HIGH #3 fix). The state still records the skip in notes.
            if decision.proceed:
                wrote = False
                try:
                    write_iteration(record, base_dir=config.iteration_dir)
                    wrote = True
                except (OSError, ValueError) as exc:
                    _LOG.warning("iteration write failed: %s", type(exc).__name__)
                if wrote and config.pack_iterations:
                    prev_registry_id = _pack_iteration_safely(
                        record, config, prev_registry_id
                    )
                if config.on_iteration is not None:
                    try:
                        config.on_iteration(record)
                    except Exception:  # noqa: BLE001 — daemon must not crash
                        _LOG.warning("on_iteration callback raised", exc_info=True)
            iterations += 1
            if iterations and (
                config.max_iterations is None or iterations < config.max_iterations
            ):
                if stop.wait(config.poll_interval_sec):
                    break
    finally:
        # Preserve `paused` status across daemon exit — only flip to
        # `stopped` if the daemon naturally exited (max_iterations / state
        # was running). A SIGTERM while paused must not be silently
        # promoted to stopped (code-review HIGH #2 fix).
        try:
            current = read_state(config.state_path)
            if current.status == "running":
                write_state(current.with_status("stopped"), config.state_path)
                state = current.with_status("stopped")
            else:
                state = current
        except (FileNotFoundError, ValueError):
            pass
    return state, iterations


def _pack_iteration_safely(
    record: IterationRecord,
    config: WatchConfig,
    prev_registry_id: Optional[str],
) -> Optional[str]:
    """Pack one iteration as a Soup Can; never crash the daemon.

    Returns the new Registry entry id (to chain as the next iteration's
    parent) on success, or the unchanged ``prev_registry_id`` on failure.
    On failure the iteration manifest is re-written with a ``pack-failed:``
    note appended (best-effort) so the operator can see what happened
    (#177 acceptance: swallow at WARNING + record in iteration notes).
    """
    from soup_cli.utils.loop_iteration import pack_iteration_as_can

    try:
        _, entry_id = pack_iteration_as_can(
            record.iteration_id,
            base_dir=config.iteration_dir,
            served_model=config.served_model,
            base_model=config.base_model,
            parent_registry_id=prev_registry_id,
        )
        return entry_id
    except Exception as exc:  # noqa: BLE001 — instrumentation must not crash
        _LOG.warning("iteration pack failed: %s", type(exc).__name__)
        try:
            prefix = record.notes + " | " if record.notes else ""
            note = (prefix + f"pack-failed: {type(exc).__name__}")[:4096]
            write_iteration(
                replace(record, notes=note),
                base_dir=config.iteration_dir,
            )
        except (OSError, ValueError):
            pass
        return prev_registry_id


def _state_with(state: LoopState, **kwargs: object) -> LoopState:
    """Return a copy with overrides applied (escape hatch around ``replace``).

    The dataclass already exposes ``with_status`` and ``bumped`` but the
    daemon needs to flip a handful of fields atomically per cycle (e.g.
    ``last_run_date`` + ``last_iteration_id`` together). Keeping this tiny
    helper local avoids leaking ``dataclasses.replace`` into the daemon
    surface; the import lives at module top per code-review MEDIUM #7.
    """
    return replace(state, **kwargs)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def evaluate_canary_verdict(stats: BucketStats) -> str:
    """Project a single canary verdict from accumulated bucket stats.

    Trivial wrapper around ``BucketStats.verdict`` so the daemon does
    not import the router class directly — keeps the import graph
    one-directional (utils.loop_daemon → utils.canary_router, never the
    other way).
    """
    return stats.verdict()


def maybe_rollback(
    policy: CanaryPolicy, verdict: str, *, sticky: bool = True
) -> CanaryPolicy:
    """Roll back the canary if ``verdict == "MAJOR"``.

    Non-MAJOR verdicts pass through unchanged so a flaky re-eval cannot
    flip-flop traffic. Sticky-on-rollback (the default) is documented in
    ``canary_router.rollback`` — once cleared, the operator must
    explicitly re-promote a new canary.
    """
    if not isinstance(policy, CanaryPolicy):
        raise TypeError("policy must be CanaryPolicy")
    if not isinstance(verdict, str):
        raise TypeError("verdict must be str")
    if verdict == "MAJOR":
        return rollback(policy, reason="canary regression")
    return policy
