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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional

from soup_cli.utils.canary_router import BucketStats, CanaryPolicy, rollback
from soup_cli.utils.loop_budget import (
    BudgetDecision,
    check_budget,
    reset_daily_counter_if_new_day,
)
from soup_cli.utils.loop_iteration import (
    IterationRecord,
    new_iteration_id,
    write_iteration,
)
from soup_cli.utils.loop_state import LoopState, read_state, write_state

_LOG = logging.getLogger(__name__)

# A per-stage callable returns a small dict; the orchestrator merges the
# result dicts into one ``StageResult`` before recording the iteration.

HarvestFn = Callable[[LoopState], Mapping[str, object]]
TrainFn = Callable[[LoopState, Mapping[str, object]], Mapping[str, object]]
GateFn = Callable[[LoopState, Mapping[str, object]], Mapping[str, object]]
DeployFn = Callable[[LoopState, Mapping[str, object]], Mapping[str, object]]
CostFn = Callable[[LoopState], float]


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


def default_cost(state: LoopState) -> float:
    """Stub cost estimate — zero so the budget gate stays permissive."""
    return 0.0


@dataclass
class WatchConfig:
    """Daemon configuration knobs."""

    poll_interval_sec: float = 60.0
    max_iterations: Optional[int] = None  # None = unbounded (real daemon)
    state_path: Optional[str] = None
    iteration_dir: Optional[str] = None
    harvest_fn: HarvestFn = default_harvest
    train_fn: TrainFn = default_train
    gate_fn: GateFn = default_gate
    deploy_fn: DeployFn = default_deploy
    cost_fn: CostFn = default_cost
    on_iteration: Optional[Callable[[IterationRecord], None]] = None
    # v0.71.4 #177 — pack each successful iteration as a v0.26 Soup Can +
    # append a Registry entry (default off so existing tests / stub watchers
    # have no registry side effects).
    pack_iterations: bool = False
    served_model: Optional[str] = None
    base_model: str = "unknown"

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
        for fname in ("harvest_fn", "train_fn", "gate_fn", "deploy_fn", "cost_fn"):
            if not callable(getattr(self, fname)):
                raise ValueError(f"{fname} must be callable")
        if self.on_iteration is not None and not callable(self.on_iteration):
            raise ValueError("on_iteration must be callable or None")
        if not isinstance(self.pack_iterations, bool):
            raise ValueError("pack_iterations must be bool")
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


def run_once(
    state: LoopState,
    config: WatchConfig,
) -> "tuple[LoopState, IterationRecord, BudgetDecision]":
    """Execute one full iteration synchronously. Pure with respect to time."""
    if not isinstance(state, LoopState):
        raise TypeError("state must be LoopState")
    if not isinstance(config, WatchConfig):
        raise TypeError("config must be WatchConfig")
    runs_today, today = reset_daily_counter_if_new_day(
        state.runs_today, state.last_run_date
    )
    state = _state_with(state, runs_today=runs_today, last_run_date=today)
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
            canary_verdict=None,
            shipped=False,
            rolled_back=False,
            estimated_cost_usd=estimated,
            notes=f"budget-skip: {decision.reason}",
        )
        return state, record, decision
    harvest_out = dict(config.harvest_fn(state))
    train_out = dict(config.train_fn(state, harvest_out))
    gate_out = dict(config.gate_fn(state, train_out))
    deploy_out = dict(config.deploy_fn(state, {**train_out, **gate_out}))
    gate_verdict = str(gate_out.get("gate_verdict", "SKIPPED"))
    canary_verdict = deploy_out.get("canary_verdict")
    if canary_verdict is not None:
        canary_verdict = str(canary_verdict)
    shipped = bool(deploy_out.get("deployed", False))
    rolled_back = bool(deploy_out.get("rolled_back", False))
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
        notes=str(deploy_out.get("notes", ""))[:4096],
    )
    new_state = state.bumped(
        traces_collected=int(harvest_out.get("traces_collected", 0) or 0),
        pairs_distilled=pairs,
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
    return new_state, record, decision


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
            state, record, decision = run_once(state, config)
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
