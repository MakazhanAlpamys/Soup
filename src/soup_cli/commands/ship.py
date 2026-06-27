"""soup ship — the SHIP / DON'T-SHIP verdict (v0.71.25).

Top-level CLI command (NOT a sub-group) — operators type::

    soup ship --base <m> --adapter <lora> --task-eval tasks.jsonl
    soup ship --evidence ev.json            # offline, pre-computed scores
    soup ship ... --output verdict.json

After fine-tuning, answer ONE question: did the model get better, or did I
break it? The decision fuses two legs (task win + catastrophic-forgetting
guard) into a single binary verdict — see ``utils/ship_verdict.py`` for the
moat (``decide_ship``).

Exit codes mirror ``soup diagnose`` so CI can gate on the result:
**0 = SHIP, 2 = DON'T SHIP, 1 = runtime error**.

Leg 1 (task win) modes: ``metric`` (reuses ``eval/custom.run_eval`` accuracy)
and ``judge_score`` (reuses ``eval/judge.JudgeEvaluator``). True pairwise
win-rate is reserved for a later release. Leg 2 (general suite) defaults to the
built-in mini benchmarks (``eval/forgetting``); ``--general-suite`` with
non-mini names routes through the existing lm-eval runner.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Mapping, NoReturn, Optional, Tuple

import typer
from rich.console import Console
from rich.markup import escape

from soup_cli.utils.paths import atomic_write_text, enforce_under_cwd_and_no_symlink
from soup_cli.utils.ship_verdict import (
    DECISION_SHIP,
    DEFAULT_FORGETTING_THRESHOLD,
    SUPPORTED_TASK_MODES,
    TASK_MODES,
    ShipVerdict,
    TaskWin,
    build_task_win,
    compute_benchmark_deltas,
    decide_ship,
    render_ship_panel,
    verdict_to_dict,
)

console = Console()

app = typer.Typer(no_args_is_help=False)

# 16 MiB cap on evidence JSON (mirrors `soup diagnose` — prevents a
# multi-GB / symlink-pointed file from OOMing at json.load time).
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024

# lm-eval override defaults (kept minimal; the override is for users who
# already run lm-eval — they can tune via a future flag if needed).
_LM_EVAL_BATCH_SIZE = 1

# Bounds on the leg-2 general suite (DoS / input-hygiene guards).
_MAX_SUITE_BENCHMARKS = 50
_MAX_BENCHMARK_NAME_CHARS = 256


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fail(message: str, code: int) -> NoReturn:
    """Print a friendly red error and raise ``typer.Exit(code)``."""
    console.print(f"[red]Error:[/] {escape(message)}")
    raise typer.Exit(code=code)


def _validate_threshold_flag(value: float) -> float:
    # Fast-fail a bad flag with an exit-2 USAGE error here; the engine's own
    # _validate_threshold raises ValueError (-> exit 1 runtime), which is the
    # wrong exit code for a CLI typo. Intentional, narrow duplication.
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail("--forgetting-threshold must be a number", 2)
    fvalue = float(value)
    # NaN fails both comparisons -> rejected.
    if not (0.0 <= fvalue <= 1.0):
        _fail("--forgetting-threshold must be in [0.0, 1.0]", 2)
    return fvalue


def _validate_task_mode_flag(task_mode: str) -> None:
    if task_mode not in TASK_MODES:
        _fail(
            f"--task-mode must be one of {', '.join(TASK_MODES)}; got {task_mode!r}",
            2,
        )
    if task_mode not in SUPPORTED_TASK_MODES:
        _fail(
            f"--task-mode {task_mode!r} (pairwise judge win-rate) ships in a "
            "later release; use 'metric' or 'judge_score' for now",
            2,
        )


def _validate_judge_model_url(url: str) -> None:
    """SSRF guard for --judge-model: urlparse hostname check (not startswith).

    Blocks the ``http://localhost.attacker.com`` prefix-bypass that a bare
    ``startswith("http://localhost")`` check would allow through.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme in ("ollama", "https"):
        return
    if parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1"):
        return
    _fail(
        f"--judge-model {url!r} uses a disallowed scheme/host; "
        "use ollama://, https://, or http://localhost",
        2,
    )


def _reject_lm_eval_injection(value: str, field: str) -> None:
    """Block ',' / '=' in an lm-eval model id (model_args injection guard).

    lm-eval parses ``model_args`` as comma-separated ``key=value`` pairs, so an
    adapter path like ``lora,trust_remote_code=True`` would otherwise smuggle
    extra args (e.g. remote-code execution) into the harness.
    """
    if "," in value or "=" in value:
        raise ValueError(
            f"{field} must not contain ',' or '=' "
            f"(lm-eval model_args injection guard): {value!r}"
        )


# ---------------------------------------------------------------------------
# Offline path — --evidence
# ---------------------------------------------------------------------------

def _load_evidence(path: str) -> dict:
    """Load an evidence JSON (cwd-contained, symlink-rejected, size-capped).

    Opens with ``O_NOFOLLOW`` (where available) and fstats the open fd so a
    symlink swapped in after the containment check cannot redirect the read
    (TOCTOU defence, mirrors v0.71.22 ``load_audio_mono``).
    """
    enforce_under_cwd_and_no_symlink(path, "evidence path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"evidence path unreadable: {type(exc).__name__}") from exc
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        if os.fstat(handle.fileno()).st_size > _MAX_EVIDENCE_BYTES:
            raise ValueError(f"evidence file exceeds {_MAX_EVIDENCE_BYTES} bytes")
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("evidence file must contain a JSON object")
    return payload


def _verdict_from_evidence(path: str, *, forgetting_threshold: float) -> ShipVerdict:
    """Build a verdict from pre-computed scores (no model load)."""
    try:
        payload = _load_evidence(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _fail(f"cannot read --evidence: {exc}", 1)

    task = payload.get("task")
    if not isinstance(task, dict):
        _fail("evidence.task must be an object with 'mode', 'base', 'tuned'", 1)
    mode = task.get("mode", "metric")
    if mode not in SUPPORTED_TASK_MODES:
        _fail(
            f"evidence.task.mode must be one of {', '.join(SUPPORTED_TASK_MODES)}; "
            f"got {mode!r}",
            1,
        )
    if "base" not in task or "tuned" not in task:
        _fail("evidence.task needs both 'base' and 'tuned' scores", 1)
    try:
        task_win = build_task_win(mode, task["base"], task["tuned"])
    except (TypeError, ValueError) as exc:
        _fail(f"invalid evidence.task: {exc}", 1)

    raw_benchmarks = payload.get("benchmarks", {})
    if not isinstance(raw_benchmarks, dict):
        _fail("evidence.benchmarks must be an object of {name: {base, tuned}}", 1)
    base_scores: Dict[str, object] = {}
    tuned_scores: Dict[str, object] = {}
    for name, entry in raw_benchmarks.items():
        if not isinstance(entry, dict) or "base" not in entry or "tuned" not in entry:
            _fail(f"evidence.benchmarks[{name!r}] needs 'base' and 'tuned'", 1)
        base_scores[str(name)] = entry["base"]
        tuned_scores[str(name)] = entry["tuned"]

    try:
        deltas = compute_benchmark_deltas(
            base_scores, tuned_scores, forgetting_threshold=forgetting_threshold
        )
        return decide_ship(task_win, deltas, forgetting_threshold=forgetting_threshold)
    except (TypeError, ValueError) as exc:
        _fail(f"invalid evidence.benchmarks: {exc}", 1)


# ---------------------------------------------------------------------------
# Live path — load base + tuned, evaluate both legs
# ---------------------------------------------------------------------------

def _resolve_generators(
    base: str, tuned: Optional[str], adapter: Optional[str], device: Optional[str]
) -> Tuple[Callable[[str], str], Callable[[str], str]]:
    """Build ``(base_gen, tuned_gen)`` from live_eval (greedy decode)."""
    from soup_cli.utils import live_eval

    base_gen = live_eval.make_generator(base, device=device)
    if adapter:
        tuned_gen = live_eval.make_generator(base, adapter=adapter, device=device)
    elif tuned:
        tuned_gen = live_eval.make_generator(tuned, device=device)
    else:  # pragma: no cover — _verdict_live guarantees one of tuned/adapter
        raise ValueError("need --tuned or --adapter")
    return base_gen, tuned_gen


def _leg1_metric(
    base_gen: Callable[[str], str],
    tuned_gen: Callable[[str], str],
    base_id: str,
    tuned_id: str,
    task_eval: str,
) -> TaskWin:
    from soup_cli.eval.custom import load_eval_tasks, run_eval

    tasks = load_eval_tasks(task_eval)
    if not tasks:
        raise ValueError(f"task-eval file {task_eval!r} has no tasks")
    base_acc = run_eval(base_id, tasks, generate_fn=base_gen).accuracy
    tuned_acc = run_eval(tuned_id, tasks, generate_fn=tuned_gen).accuracy
    return build_task_win("metric", base_acc, tuned_acc)


def _leg1_judge(
    base_gen: Callable[[str], str],
    tuned_gen: Callable[[str], str],
    task_eval: str,
    judge_model: str,
) -> TaskWin:
    from soup_cli.eval.custom import load_eval_tasks
    from soup_cli.eval.gate import _parse_judge_url
    from soup_cli.eval.judge import JudgeEvaluator

    tasks = load_eval_tasks(task_eval)
    if not tasks:
        raise ValueError(f"task-eval file {task_eval!r} has no tasks")
    provider, model, api_base = _parse_judge_url(judge_model)
    evaluator = JudgeEvaluator(provider=provider, model=model, api_base=api_base)
    # Normalise to [0, 1] using the judge's ACTUAL rubric scale (DEFAULT_RUBRIC
    # is 1-5, not 1-10), via min-max so the rubric floor maps to 0.0. The
    # verdict is monotonic-safe either way, but the stored/displayed numbers
    # must be honest.
    scale = evaluator.rubric.get("scale", {}) if isinstance(evaluator.rubric, dict) else {}
    if not isinstance(scale, dict):
        scale = {}

    def _num(value: object, default: float) -> float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return default

    scale_min = _num(scale.get("min", 1), 1.0)
    scale_max = _num(scale.get("max", 5), 5.0)
    span = (scale_max - scale_min) or 1.0

    def _score(gen: Callable[[str], str]) -> float:
        items = [
            {"prompt": t.prompt, "response": gen(t.prompt), "category": t.category}
            for t in tasks
        ]
        overall = float(
            getattr(evaluator.evaluate_batch(items), "overall_score", scale_min)
        )
        return max(0.0, min(1.0, (overall - scale_min) / span))

    return build_task_win("judge_score", _score(base_gen), _score(tuned_gen))


def _mini_score(gen: Callable[[str], str], benchmark: str) -> float:
    from soup_cli.eval.forgetting import ForgettingDetector

    return ForgettingDetector(generate_fn=gen, benchmark=benchmark).run_baseline()


def _extract_lm_score(bench_data: Mapping[str, object]) -> Optional[float]:
    """Pull a single accuracy metric from an lm-eval per-task result block."""
    for key in ("acc,none", "acc_norm,none", "exact_match,none", "em,none"):
        if key in bench_data:
            val = bench_data[key]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
    for key, val in bench_data.items():
        key_str = str(key)
        if "stderr" in key_str or key_str.startswith("alias"):
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return None


def _lm_eval_leg2(
    names: List[str],
    base_id: str,
    tuned_id: Optional[str],
    adapter: Optional[str],
    baseline_scores: Mapping[str, float],
    device: Optional[str],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Score non-mini benchmarks via the existing lm-eval harness runner."""
    from soup_cli.commands import eval as eval_cmd

    dev = device or "cpu"
    _reject_lm_eval_injection(base_id, "--base")
    base_arg = f"pretrained={base_id}"
    if adapter:
        _reject_lm_eval_injection(adapter, "--adapter")
        tuned_arg = f"pretrained={base_id},peft={adapter}"
    else:
        _reject_lm_eval_injection(str(tuned_id), "--tuned")
        tuned_arg = f"pretrained={tuned_id}"

    base_map: Dict[str, object] = {}
    tuned_map: Dict[str, object] = {}

    tuned_results = eval_cmd._run_lm_eval(
        tuned_arg, names, None, _LM_EVAL_BATCH_SIZE, dev
    )
    tuned_blocks = tuned_results.get("results", {})
    for name in names:
        score = _extract_lm_score(tuned_blocks.get(name, {}))
        if score is not None:
            tuned_map[name] = score

    base_to_run: List[str] = []
    for name in names:
        if name in baseline_scores:
            base_map[name] = float(baseline_scores[name])
        else:
            base_to_run.append(name)
    if base_to_run:
        base_results = eval_cmd._run_lm_eval(
            base_arg, base_to_run, None, _LM_EVAL_BATCH_SIZE, dev
        )
        base_blocks = base_results.get("results", {})
        for name in base_to_run:
            score = _extract_lm_score(base_blocks.get(name, {}))
            if score is not None:
                base_map[name] = score

    return base_map, tuned_map


def _leg2_scores(
    suite_names: List[str],
    base_gen: Callable[[str], str],
    tuned_gen: Callable[[str], str],
    *,
    base_id: str,
    tuned_id: Optional[str],
    adapter: Optional[str],
    baseline_scores: Mapping[str, float],
    device: Optional[str],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Compute leg-2 ``(base_scores, tuned_scores)`` maps over the general suite.

    Mini-benchmark names run live via ForgettingDetector; any other names route
    through the lm-eval override. ``baseline_scores`` supplies base scores
    directly (skipping the base run) for any name it covers.
    """
    from soup_cli.eval.forgetting import MINI_BENCHMARKS

    mini_names = [n for n in suite_names if n in MINI_BENCHMARKS]
    other_names = [n for n in suite_names if n not in MINI_BENCHMARKS]

    base_map: Dict[str, object] = {}
    tuned_map: Dict[str, object] = {}

    for name in mini_names:
        tuned_map[name] = _mini_score(tuned_gen, name)
        if name in baseline_scores:
            base_map[name] = float(baseline_scores[name])
        else:
            base_map[name] = _mini_score(base_gen, name)

    if other_names:
        lm_base, lm_tuned = _lm_eval_leg2(
            other_names, base_id, tuned_id, adapter, baseline_scores, device
        )
        base_map.update(lm_base)
        tuned_map.update(lm_tuned)

    # Never silently drop a requested benchmark: a name missing on either side
    # would vanish at the delta intersection and the moat would not see a
    # possible regression. Refuse loudly instead (-> exit 1).
    missing = [n for n in suite_names if n not in base_map or n not in tuned_map]
    if missing:
        raise ValueError(
            f"could not score benchmark(s) on both base and tuned: "
            f"{', '.join(sorted(missing))}"
        )

    return base_map, tuned_map


def _parse_suite(general_suite: Optional[str]) -> List[str]:
    from soup_cli.eval.forgetting import MINI_BENCHMARKS

    if not general_suite:
        return list(MINI_BENCHMARKS.keys())
    names = [chunk.strip() for chunk in general_suite.split(",")]
    return [name for name in names if name]


def _verdict_live(
    *,
    base: Optional[str],
    tuned: Optional[str],
    adapter: Optional[str],
    task_eval: Optional[str],
    task_mode: str,
    judge_model: Optional[str],
    general_suite: Optional[str],
    baseline_spec: Optional[str],
    device: Optional[str],
    forgetting_threshold: float,
) -> ShipVerdict:
    """Run a live verdict — validate flags (exit 2), then evaluate (exit 1)."""
    if not base:
        _fail("live run needs --base <model>", 2)
    if adapter and tuned:
        _fail("pass --adapter OR --tuned, not both", 2)
    if not adapter and not tuned:
        _fail("live run needs --tuned <model> or --adapter <adapter-path>", 2)
    if not task_eval:
        _fail("live run needs --task-eval <tasks.jsonl> for the leg-1 task win", 2)
    try:
        enforce_under_cwd_and_no_symlink(task_eval, "--task-eval path")
    except (ValueError, TypeError) as exc:
        _fail(str(exc), 2)

    suite_names = _parse_suite(general_suite)
    if not suite_names:
        _fail("--general-suite resolved to no benchmarks", 2)
    if len(suite_names) > _MAX_SUITE_BENCHMARKS:
        _fail(
            f"--general-suite has too many benchmarks (max {_MAX_SUITE_BENCHMARKS})",
            2,
        )
    for _name in suite_names:
        if "\x00" in _name or len(_name) > _MAX_BENCHMARK_NAME_CHARS:
            _fail(
                "--general-suite names must be null-free and "
                f"< {_MAX_BENCHMARK_NAME_CHARS} chars",
                2,
            )

    # Resolve --baseline up front so a bad spec (outside cwd / missing file /
    # unknown registry id) is a USAGE error (exit 2), not a runtime error (1).
    baseline_scores: Dict[str, float] = {}
    if baseline_spec:
        from soup_cli.eval.gate import resolve_baseline

        try:
            baseline_scores = resolve_baseline(baseline_spec)
        except (ValueError, FileNotFoundError, OSError) as exc:
            _fail(f"--baseline: {exc}", 2)

    tuned_id = tuned if tuned else base
    try:
        base_gen, tuned_gen = _resolve_generators(base, tuned, adapter, device)
        if task_mode == "judge_score":
            if not judge_model:
                _fail("--task-mode judge_score needs --judge-model <url>", 2)
            _validate_judge_model_url(judge_model)
            task_win = _leg1_judge(base_gen, tuned_gen, task_eval, judge_model)
        else:
            task_win = _leg1_metric(base_gen, tuned_gen, base, tuned_id, task_eval)
        base_scores, tuned_scores = _leg2_scores(
            suite_names,
            base_gen,
            tuned_gen,
            base_id=base,
            tuned_id=tuned_id,
            adapter=adapter,
            baseline_scores=baseline_scores,
            device=device,
        )
        deltas = compute_benchmark_deltas(
            base_scores, tuned_scores, forgetting_threshold=forgetting_threshold
        )
        return decide_ship(task_win, deltas, forgetting_threshold=forgetting_threshold)
    except typer.Exit:
        # typer.Exit subclasses RuntimeError — re-raise so in-try _fail() usage
        # errors (exit 2) keep their code instead of being re-coded as exit 1.
        raise
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
        _fail(f"live ship verdict failed: {type(exc).__name__}: {exc}", 1)


# ---------------------------------------------------------------------------
# Render + exit
# ---------------------------------------------------------------------------

def _emit_and_exit(verdict: ShipVerdict, output: Optional[str]) -> None:
    console.print(render_ship_panel(verdict))
    if output:
        try:
            atomic_write_text(
                json.dumps(verdict_to_dict(verdict), indent=2), output, field="output"
            )
            console.print(f"[green]Wrote[/] {escape(output)}")
        except (OSError, ValueError, TypeError) as exc:
            _fail(f"cannot write --output: {type(exc).__name__}: {exc}", 1)
    if verdict.decision != DECISION_SHIP:
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# CLI entrypoint (callback so `soup ship <opts>` needs no subcommand)
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def ship(
    base: Optional[str] = typer.Option(
        None, "--base", help="Base model id/path (the 'before')."
    ),
    tuned: Optional[str] = typer.Option(
        None, "--tuned", help="Tuned model id/path (a separate 'after' model)."
    ),
    adapter: Optional[str] = typer.Option(
        None, "--adapter", help="LoRA adapter path (tuned = base + this adapter)."
    ),
    task_eval: Optional[str] = typer.Option(
        None, "--task-eval", help="JSONL of leg-1 task-win eval tasks."
    ),
    task_mode: str = typer.Option(
        "metric",
        "--task-mode",
        help="Leg-1 mode: metric | judge_score (pairwise: later release).",
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="Judge model URL for --task-mode judge_score."
    ),
    general_suite: Optional[str] = typer.Option(
        None,
        "--general-suite",
        help="Comma list of leg-2 benchmarks (default: the 3 mini benchmarks; "
        "non-mini names route through lm-eval).",
    ),
    baseline: Optional[str] = typer.Option(
        None,
        "--baseline",
        help="registry://<id> or JSON file of base leg-2 scores (skips base run).",
    ),
    forgetting_threshold: float = typer.Option(
        DEFAULT_FORGETTING_THRESHOLD,
        "--forgetting-threshold",
        help="Max allowed leg-2 drop in absolute points before DON'T SHIP.",
    ),
    evidence: Optional[str] = typer.Option(
        None, "--evidence", help="JSON of pre-computed scores (offline, no model load)."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write the verdict JSON to this path."
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device for the live run (cuda / cpu)."
    ),
) -> None:
    """Decide SHIP / DON'T SHIP for a fine-tune (exit 0 = SHIP, 2 = DON'T)."""
    _validate_task_mode_flag(task_mode)
    threshold = _validate_threshold_flag(forgetting_threshold)

    if evidence:
        verdict = _verdict_from_evidence(evidence, forgetting_threshold=threshold)
    elif base or tuned or adapter or task_eval:
        verdict = _verdict_live(
            base=base,
            tuned=tuned,
            adapter=adapter,
            task_eval=task_eval,
            task_mode=task_mode,
            judge_model=judge_model,
            general_suite=general_suite,
            baseline_spec=baseline,
            device=device,
            forgetting_threshold=threshold,
        )
    else:
        _fail(
            "provide --evidence <json> for an offline verdict, or "
            "--base + (--tuned|--adapter) + --task-eval for a live run",
            2,
        )

    _emit_and_exit(verdict, output)


__all__ = ["app", "ship"]
