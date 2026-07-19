"""soup ship — the SHIP / DON'T-SHIP verdict (v0.71.25).

Top-level CLI command (NOT a sub-group) — operators type::

    soup ship --base <m> --adapter <lora> --task-eval tasks.jsonl
    soup ship --evidence ev.json            # offline, pre-computed scores
    soup ship ... --output verdict.json
    soup ship ... --config soup.yaml        # read eval.ship defaults + bind provenance
    soup ship ... --emit-evidence ev.json   # re-serialise scores as replayable input
    soup ship --evidence ev.json --push owner/repo#42   # verdict as a PR comment

After fine-tuning, answer ONE question: did the model get better, or did I
break it? The decision fuses two legs (task win + catastrophic-forgetting
guard) into a single binary verdict — see ``utils/ship_verdict.py`` for the
moat (``decide_ship``).

Exit codes so CI can gate on the result:
**0 = SHIP, 2 = DON'T SHIP, 3 = usage/validation error, 1 = runtime error**.
Usage errors moved off ``2`` in v0.71.38 — a typo'd flag was previously
indistinguishable from a caught regression (both exited ``2``); ``3`` mirrors
``soup plan`` / ``soup env check``. Offline ``--evidence`` read/parse errors
stay ``1``.

Leg 1 (task win) modes: ``metric`` (reuses ``eval/custom.run_eval`` accuracy),
``judge_score`` (reuses ``eval/judge.JudgeEvaluator``), and ``pairwise`` (true
judge win-rate, v0.71.31). Leg 2 (general suite) defaults to the bundled offline
suite (``eval/gate_suites`` — MCQ/arithmetic + tool-call/JSON/safety, scored by
the pure diagnose/custom scorers, v0.71.38); ``--general-suite`` with any
non-bundled name routes through the existing lm-eval runner.
"""

from __future__ import annotations

import json
import os
import re
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Mapping,
    NoReturn,
    Optional,
    Tuple,
)

import typer
from rich.console import Console
from rich.markup import escape

from soup_cli.utils.paths import atomic_write_text, enforce_under_cwd_and_no_symlink

if TYPE_CHECKING:  # pydantic models — import for typing only (no eager cost)
    from soup_cli.config.schema import ShipConfig, SoupConfig
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
    verdict_to_evidence,
)

console = Console()

app = typer.Typer(no_args_is_help=False)

# Exit-code taxonomy (v0.71.38): keep DON'T-SHIP distinct from a config typo.
_EXIT_RUNTIME = 1  # something went wrong actually running (IO, model load, ...)
_EXIT_DONT_SHIP = 2  # a verdict: leg 1 or leg 2 said don't ship
_EXIT_USAGE = 3  # bad flags / validation (mirrors `soup plan` / `env check`)

# 16 MiB cap on evidence JSON (mirrors `soup diagnose` — prevents a
# multi-GB / symlink-pointed file from OOMing at json.load time).
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024

# 4 MiB cap on a soup.yaml passed via --config (configs are small).
_MAX_CONFIG_BYTES = 4 * 1024 * 1024

# 8 GiB cap on the training file we fingerprint for provenance.data_sha
# (best-effort — skipped above this, never fatal).
_MAX_DATA_SHA_BYTES = 8 * 1024 * 1024 * 1024

# A canonical hex SHA-256 digest — used to sanity-check the config_sha we read
# out of an untrusted evidence file before echoing it in an error message (a
# raw value could smuggle terminal ESC bytes past rich.markup.escape).
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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
    # Fast-fail a bad flag with a usage error (exit 3) here; the engine's own
    # _validate_threshold raises ValueError (-> exit 1 runtime), which is the
    # wrong exit code for a CLI typo. Intentional, narrow duplication.
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail("--forgetting-threshold must be a number", _EXIT_USAGE)
    fvalue = float(value)
    # NaN fails both comparisons -> rejected.
    if not (0.0 <= fvalue <= 1.0):
        _fail("--forgetting-threshold must be in [0.0, 1.0]", _EXIT_USAGE)
    return fvalue


def _validate_task_mode_flag(task_mode: str) -> None:
    # All three modes (metric / judge_score / pairwise) ship as of v0.71.31, so
    # SUPPORTED_TASK_MODES == TASK_MODES and the old "pairwise reserved" gate is
    # gone (it was dead code).
    if task_mode not in TASK_MODES:
        _fail(
            f"--task-mode must be one of {', '.join(TASK_MODES)}; got {task_mode!r}",
            _EXIT_USAGE,
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
        _EXIT_USAGE,
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
# --config — read leg-1/leg-2 defaults from a committed soup.yaml (v0.71.39)
# ---------------------------------------------------------------------------

def _safe_read_text(path: str, field: str, max_bytes: int) -> str:
    """O_NOFOLLOW + fstat-capped read of a cwd-contained file.

    Shared TOCTOU-safe reader (mirrors v0.71.22 ``load_audio_mono``): opens with
    ``O_NOFOLLOW`` where available and fstats the open fd, so a symlink swapped
    in after the containment check cannot redirect the read. Raises ``ValueError``
    (incl. via ``enforce_under_cwd_and_no_symlink``) on any failure; callers map
    it to the right exit code.
    """
    enforce_under_cwd_and_no_symlink(path, field)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{field} unreadable: {type(exc).__name__}") from exc
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        if os.fstat(handle.fileno()).st_size > max_bytes:
            raise ValueError(f"{field} exceeds {max_bytes} bytes")
        return handle.read()


def _parse_ship_config(path: str) -> "Tuple[SoupConfig, Optional[ShipConfig]]":
    """Load a soup.yaml and return ``(SoupConfig, ShipConfig | None)``.

    A read / parse / validation failure is a USAGE error (exit 3), mirroring
    ``soup plan`` / ``soup env check``.
    """
    import yaml

    from soup_cli.config.loader import load_config_from_string

    try:
        text = _safe_read_text(path, "--config path", _MAX_CONFIG_BYTES)
        cfg = load_config_from_string(text)
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        _fail(f"--config: {exc}", _EXIT_USAGE)
    ship_cfg = cfg.eval.ship if cfg.eval is not None else None
    return cfg, ship_cfg


def _config_sha_of(cfg: "SoupConfig") -> str:
    """Canonical (order/whitespace-insensitive) SHA-256 of the training recipe.

    Semantic, not textual: a reformatted soup.yaml keeps the same sha but a real
    recipe change does not. Cheap — hashes only the config dict, never the data
    file (that's the ``data_sha`` in the full provenance).

    The gate's own read-time policy (``eval.ship`` — threshold / suite / judge)
    is EXCLUDED: it is applied at verdict time, not training time, so loosening
    ``forgetting_threshold`` must NOT invalidate evidence about an unchanged
    model (the staleness gate fingerprints the recipe, not the gate config).
    """
    from soup_cli.registry.hashing import hash_config

    return hash_config(cfg.model_dump(mode="json", exclude={"eval": {"ship"}}))


def _safe_hash_file(path: str, max_bytes: int) -> Optional[str]:
    """SHA-256 of a cwd-local file via an O_NOFOLLOW fd (TOCTOU + size capped).

    ``data.train`` comes from a parsed ``--config`` YAML, so it must not follow a
    symlink out of the tree or stream an unbounded file. This mirrors
    ``_safe_read_text`` but hashes bytes and is best-effort — returns ``None``
    (never raises) for an absent / oversized / unreadable / out-of-cwd path,
    because ``data_sha`` is informational provenance, not a hard requirement.
    """
    import hashlib

    try:
        enforce_under_cwd_and_no_symlink(path, "data.train")
    except (ValueError, TypeError):
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "rb") as handle:
            if os.fstat(handle.fileno()).st_size > max_bytes:
                return None
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _compute_provenance(cfg: "SoupConfig") -> Dict[str, str]:
    """Bind emitted evidence to the exact config that produced it (v0.71.39).

    ``config_sha`` (semantic recipe hash) + ``base_model`` + a best-effort
    ``data_sha`` over a cwd-local training file. Built only when we actually
    ``--emit-evidence`` (the ``data_sha`` streams the whole training file).
    """
    prov: Dict[str, str] = {"config_sha": _config_sha_of(cfg)}
    base = cfg.base
    if base:
        prov["base_model"] = base
    data = cfg.data.train
    if data:
        data_sha = _safe_hash_file(data, _MAX_DATA_SHA_BYTES)
        if data_sha is not None:
            prov["data_sha"] = data_sha
    return prov


def _check_evidence_staleness(payload: dict, expected_sha: str) -> None:
    """Refuse evidence whose ``config_sha`` != the committed config's (exit 3).

    Catches DRIFT: a PR that changed ``soup.yaml`` but forgot to recompute its
    ``ship_evidence.json`` is caught here instead of shipping a verdict about a
    *different* recipe than the one in the diff. This is staleness detection,
    NOT tamper-resistance — ``config_sha`` is an unkeyed hash, so it verifies
    "this evidence claims to describe the config at HEAD", not "these scores were
    actually produced by that config" (the ``--evidence`` trust model has always
    assumed a trusted artifact from your own pipeline; ``soup attest`` /
    ``adapters sign`` provide ed25519 signing if forgery is in scope). Pure — the
    payload is already loaded (read once per invocation).
    """
    prov = payload.get("provenance")
    got = prov.get("config_sha") if isinstance(prov, dict) else None
    # Validate the SHAPE before ever printing it: a non-hex value is both
    # malformed provenance AND a terminal-escape vector (rich.markup.escape does
    # not strip raw C0/ESC bytes). Never echo an unvalidated value.
    if not isinstance(got, str) or not _SHA256_RE.match(got):
        _fail(
            "evidence has no valid provenance.config_sha to verify against "
            "--config; re-produce it with "
            "`soup ship ... --config <cfg> --emit-evidence <ev>`",
            _EXIT_USAGE,
        )
    if got != expected_sha:
        # Both sides are now guaranteed [0-9a-f]{64}, so slicing is print-safe.
        _fail(
            "stale evidence: its config_sha does not match --config "
            f"(evidence={got[:12]}..., config={expected_sha[:12]}...). "
            "Re-run training + emit evidence against the current config.",
            _EXIT_USAGE,
        )


def _flag_is_default(ctx: typer.Context, name: str) -> bool:
    """True when ``name`` was left at its default (so --config may fill it).

    Uses Click's parameter-source tracking so an explicit CLI flag (or env var)
    always wins over the config value (CLI > config > hard default). Only a
    genuine ``DEFAULT`` source returns True; an untrackable / unknown name
    (source is None) returns False, so a future param rename cannot silently
    make a flag config-overridable.
    """
    try:
        from click.core import ParameterSource

        source = ctx.get_parameter_source(name)
    except (ImportError, AttributeError):  # pragma: no cover — defensive
        return False
    return source == ParameterSource.DEFAULT


# ---------------------------------------------------------------------------
# Offline path — --evidence
# ---------------------------------------------------------------------------

def _load_evidence(path: str) -> dict:
    """Load an evidence JSON (cwd-contained, symlink-rejected, size-capped)."""
    payload = json.loads(_safe_read_text(path, "evidence path", _MAX_EVIDENCE_BYTES))
    if not isinstance(payload, dict):
        raise ValueError("evidence file must contain a JSON object")
    return payload


def _verdict_from_evidence(payload: dict, *, forgetting_threshold: float) -> ShipVerdict:
    """Build a verdict from an already-loaded evidence payload (no model load)."""
    task = payload.get("task")
    if not isinstance(task, dict):
        _fail("evidence.task must be an object with 'mode', 'base', 'tuned'", _EXIT_RUNTIME)
    mode = task.get("mode", "metric")
    if mode not in SUPPORTED_TASK_MODES:
        _fail(
            f"evidence.task.mode must be one of {', '.join(SUPPORTED_TASK_MODES)}; "
            f"got {mode!r}",
            _EXIT_RUNTIME,
        )
    if "base" not in task or "tuned" not in task:
        _fail("evidence.task needs both 'base' and 'tuned' scores", _EXIT_RUNTIME)
    try:
        task_win = build_task_win(mode, task["base"], task["tuned"])
    except (TypeError, ValueError) as exc:
        _fail(f"invalid evidence.task: {exc}", _EXIT_RUNTIME)

    raw_benchmarks = payload.get("benchmarks", {})
    if not isinstance(raw_benchmarks, dict):
        _fail("evidence.benchmarks must be an object of {name: {base, tuned}}", _EXIT_RUNTIME)
    base_scores: Dict[str, object] = {}
    tuned_scores: Dict[str, object] = {}
    for name, entry in raw_benchmarks.items():
        if not isinstance(entry, dict) or "base" not in entry or "tuned" not in entry:
            _fail(f"evidence.benchmarks[{name!r}] needs 'base' and 'tuned'", _EXIT_RUNTIME)
        base_scores[str(name)] = entry["base"]
        tuned_scores[str(name)] = entry["tuned"]

    try:
        deltas = compute_benchmark_deltas(
            base_scores, tuned_scores, forgetting_threshold=forgetting_threshold
        )
        return decide_ship(task_win, deltas, forgetting_threshold=forgetting_threshold)
    except (TypeError, ValueError) as exc:
        _fail(f"invalid evidence.benchmarks: {exc}", _EXIT_RUNTIME)


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


def _leg1_pairwise(
    base_gen: Callable[[str], str],
    tuned_gen: Callable[[str], str],
    task_eval: str,
    judge_model: str,
) -> TaskWin:
    """Leg-1 via a true pairwise judge win-rate (#284).

    For each task prompt, generate a base and a tuned response and ask the judge
    which is better (swap-debiased). The tuned win-rate becomes leg 1, framed as
    ``TaskWin(base=0.5 coin-flip, tuned=win-rate)`` so ``won <=> win-rate > 0.5``.
    """
    from soup_cli.eval.custom import load_eval_tasks
    from soup_cli.eval.gate import _parse_judge_url
    from soup_cli.eval.judge import JudgeEvaluator, pairwise_winrate

    tasks = load_eval_tasks(task_eval)
    if not tasks:
        raise ValueError(f"task-eval file {task_eval!r} has no tasks")
    provider, model, api_base = _parse_judge_url(judge_model)
    evaluator = JudgeEvaluator(provider=provider, model=model, api_base=api_base)
    pairs = [(t.prompt, base_gen(t.prompt), tuned_gen(t.prompt)) for t in tasks]
    winrate = pairwise_winrate(pairs, evaluator)
    return build_task_win("pairwise", 0.5, winrate)


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

    Bundled suite names (v0.71.38 — the MCQ/arithmetic *and* the behavioural
    tool-call / JSON-format / safety suites) are scored offline via
    ``gate_suites.score_bundled_suite``; any other name routes through the
    lm-eval override. ``baseline_scores`` supplies base scores directly
    (skipping the base run) for any name it covers.
    """
    from soup_cli.eval.gate_suites import is_bundled_suite, score_bundled_suite

    bundled_names = [n for n in suite_names if is_bundled_suite(n)]
    other_names = [n for n in suite_names if not is_bundled_suite(n)]

    base_map: Dict[str, object] = {}
    tuned_map: Dict[str, object] = {}

    for name in bundled_names:
        tuned_map[name] = score_bundled_suite(name, tuned_gen)
        if name in baseline_scores:
            base_map[name] = float(baseline_scores[name])
        else:
            base_map[name] = score_bundled_suite(name, base_gen)

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
    from soup_cli.eval.gate_suites import DEFAULT_GENERAL_SUITE

    if not general_suite:
        return list(DEFAULT_GENERAL_SUITE)
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
        _fail("live run needs --base <model>", _EXIT_USAGE)
    if adapter and tuned:
        _fail("pass --adapter OR --tuned, not both", _EXIT_USAGE)
    if not adapter and not tuned:
        _fail("live run needs --tuned <model> or --adapter <adapter-path>", _EXIT_USAGE)
    if not task_eval:
        _fail("live run needs --task-eval <tasks.jsonl> for the leg-1 task win", _EXIT_USAGE)
    try:
        enforce_under_cwd_and_no_symlink(task_eval, "--task-eval path")
    except (ValueError, TypeError) as exc:
        _fail(str(exc), _EXIT_USAGE)

    suite_names = _parse_suite(general_suite)
    if not suite_names:
        _fail("--general-suite resolved to no benchmarks", _EXIT_USAGE)
    if len(suite_names) > _MAX_SUITE_BENCHMARKS:
        _fail(
            f"--general-suite has too many benchmarks (max {_MAX_SUITE_BENCHMARKS})",
            _EXIT_USAGE,
        )
    for _name in suite_names:
        if "\x00" in _name or len(_name) > _MAX_BENCHMARK_NAME_CHARS:
            _fail(
                "--general-suite names must be null-free and "
                f"< {_MAX_BENCHMARK_NAME_CHARS} chars",
                _EXIT_USAGE,
            )

    # Resolve --baseline up front so a bad spec (outside cwd / missing file /
    # unknown registry id) is a USAGE error (exit 2), not a runtime error (1).
    baseline_scores: Dict[str, float] = {}
    if baseline_spec:
        from soup_cli.eval.gate import resolve_baseline

        try:
            baseline_scores = resolve_baseline(baseline_spec)
        except (ValueError, FileNotFoundError, OSError) as exc:
            _fail(f"--baseline: {exc}", _EXIT_USAGE)

    tuned_id = tuned if tuned else base
    try:
        base_gen, tuned_gen = _resolve_generators(base, tuned, adapter, device)
        if task_mode == "judge_score":
            if not judge_model:
                _fail("--task-mode judge_score needs --judge-model <url>", _EXIT_USAGE)
            _validate_judge_model_url(judge_model)
            task_win = _leg1_judge(base_gen, tuned_gen, task_eval, judge_model)
        elif task_mode == "pairwise":
            if not judge_model:
                _fail("--task-mode pairwise needs --judge-model <url>", _EXIT_USAGE)
            _validate_judge_model_url(judge_model)
            task_win = _leg1_pairwise(base_gen, tuned_gen, task_eval, judge_model)
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
        # errors (exit 3) keep their code instead of being re-coded as exit 1.
        raise
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
        _fail(f"live ship verdict failed: {type(exc).__name__}: {exc}", _EXIT_RUNTIME)


# ---------------------------------------------------------------------------
# Render + exit
# ---------------------------------------------------------------------------

def _push_pr_comment(verdict: ShipVerdict, target: str) -> None:
    """Post the verdict as a GitHub PR comment — best-effort (reuses adapter_pr).

    A comment-posting failure (missing token, ``gh`` not installed, API error)
    must NOT flip the verdict's exit code: the SHIP / DON'T-SHIP decision is the
    gate's contract, and a flaky CI runner should not turn a real SHIP into a
    "runtime error". So transport failures WARN loudly and preserve the exit
    code. The target *shape* is validated up front (a typo is a usage error).
    """
    from soup_cli.utils import adapter_pr
    from soup_cli.utils.ship_verdict import render_ship_pr_markdown

    body = render_ship_pr_markdown(verdict)
    try:
        url = adapter_pr.post_pr_comment(target, body)
    except (RuntimeError, TypeError, ValueError) as exc:
        console.print(
            f"[yellow]Warning:[/] could not post the PR comment to "
            f"{escape(target)}: {escape(str(exc))}"
        )
        return
    console.print(
        f"[green]Posted PR comment to[/] {escape(target)}"
        + (f" -> {escape(url)}" if url else "")
    )


def _emit_and_exit(
    verdict: ShipVerdict,
    output: Optional[str],
    *,
    emit_evidence: Optional[str] = None,
    provenance: Optional[Mapping[str, object]] = None,
    push: Optional[str] = None,
) -> None:
    console.print(render_ship_panel(verdict))
    if output:
        try:
            atomic_write_text(
                json.dumps(verdict_to_dict(verdict), indent=2), output, field="output"
            )
            console.print(f"[green]Wrote[/] {escape(output)}")
        except (OSError, ValueError, TypeError) as exc:
            _fail(f"cannot write --output: {type(exc).__name__}: {exc}", _EXIT_RUNTIME)
    if emit_evidence:
        try:
            payload = verdict_to_evidence(verdict, provenance=provenance)
            atomic_write_text(
                json.dumps(payload, indent=2), emit_evidence, field="emit-evidence"
            )
            console.print(f"[green]Wrote evidence[/] {escape(emit_evidence)}")
        except (OSError, ValueError, TypeError) as exc:
            _fail(
                f"cannot write --emit-evidence: {type(exc).__name__}: {exc}",
                _EXIT_RUNTIME,
            )
    # Post the PR comment BEFORE the DON'T-SHIP exit so a regression is still
    # announced on the PR (the gate then blocks with exit 2).
    if push:
        _push_pr_comment(verdict, push)
    if verdict.decision != DECISION_SHIP:
        raise typer.Exit(code=_EXIT_DONT_SHIP)


# ---------------------------------------------------------------------------
# CLI entrypoint (callback so `soup ship <opts>` needs no subcommand)
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def ship(
    ctx: typer.Context,
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
        help="Leg-1 mode: metric | judge_score | pairwise (judge win-rate).",
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Judge model URL for --task-mode judge_score or pairwise.",
    ),
    general_suite: Optional[str] = typer.Option(
        None,
        "--general-suite",
        help="Comma list of leg-2 benchmarks (default: the bundled offline suite "
        "— MCQ/arithmetic + tool-call/JSON/safety; non-bundled names route "
        "through lm-eval).",
    ),
    baseline: Optional[str] = typer.Option(
        None,
        "--baseline",
        help="registry://<id> or JSON file of base leg-2 scores (skips base run). "
        "Recompute baselines captured before v0.71.38 — the leg-2 scorer changed, "
        "so an old baseline is not comparable to a freshly-scored tuned model.",
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
    emit_evidence: Optional[str] = typer.Option(
        None,
        "--emit-evidence",
        help="Write the scores in the --evidence INPUT schema so this run can be "
        "replayed offline (output-is-input). With --config it also STAMPS the "
        "config's provenance onto the emitted evidence.",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="soup.yaml whose eval.ship block supplies defaults (CLI flags win). "
        "With --evidence alone it GATES (refuses evidence whose config_sha drifted "
        "from this config); with --emit-evidence it STAMPS this config's provenance.",
    ),
    push: Optional[str] = typer.Option(
        None,
        "--push",
        help="Post the verdict as a GitHub PR comment (owner/repo#N). Auth via "
        "GITHUB_TOKEN / GH_TOKEN; needs the `gh` CLI.",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device for the live run (cuda / cpu)."
    ),
) -> None:
    """Decide SHIP / DON'T SHIP for a fine-tune (exit 0 = SHIP, 2 = DON'T)."""
    soup_config = None
    # `is not None` (not truthiness): an explicit empty --config must fail loud
    # through _parse_ship_config, not silently disable the staleness gate.
    if config is not None:
        soup_config, ship_cfg = _parse_ship_config(config)
        if ship_cfg is not None:
            if _flag_is_default(ctx, "task_eval") and ship_cfg.task_eval is not None:
                task_eval = ship_cfg.task_eval
            if _flag_is_default(ctx, "task_mode"):
                task_mode = ship_cfg.task_mode
            if _flag_is_default(ctx, "general_suite") and ship_cfg.general_suite is not None:
                general_suite = ship_cfg.general_suite
            if _flag_is_default(ctx, "judge_model") and ship_cfg.judge_model is not None:
                judge_model = ship_cfg.judge_model
            if _flag_is_default(ctx, "baseline") and ship_cfg.baseline is not None:
                baseline = ship_cfg.baseline
            if _flag_is_default(ctx, "forgetting_threshold"):
                forgetting_threshold = ship_cfg.forgetting_threshold

    _validate_task_mode_flag(task_mode)
    threshold = _validate_threshold_flag(forgetting_threshold)

    # Fail a mistyped --push target FAST (usage error) — before computing the
    # verdict — so a typo can't waste a live run; the actual POST later is
    # best-effort and never changes the verdict's exit code.
    if push is not None:
        from soup_cli.utils.adapter_pr import parse_pr_target

        try:
            parse_pr_target(push)
        except (ValueError, TypeError) as exc:
            _fail(f"--push: {exc}", _EXIT_USAGE)

    # config_sha is cheap (hashes only the config dict); the full provenance
    # (incl. data_sha over the training file) is built lazily, only when we
    # actually --emit-evidence.
    config_sha = _config_sha_of(soup_config) if soup_config is not None else None

    if evidence:
        try:
            payload = _load_evidence(evidence)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _fail(f"cannot read --evidence: {exc}", _EXIT_RUNTIME)
        # --config has two intents here:
        #   * GATE (no --emit-evidence): verify this committed evidence is bound
        #     to the committed config — refuse if config_sha drifted or is absent.
        #   * PRODUCER (--emit-evidence): STAMP the config's provenance onto these
        #     scores (raw scores from an external eval tool -> bound evidence), so
        #     the input is NOT required to already carry a matching provenance.
        if config_sha is not None and not emit_evidence:
            _check_evidence_staleness(payload, config_sha)
        verdict = _verdict_from_evidence(payload, forgetting_threshold=threshold)
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
            _EXIT_USAGE,
        )

    # Full provenance (incl. data_sha) is only needed when writing evidence.
    provenance = (
        _compute_provenance(soup_config)
        if emit_evidence and soup_config is not None
        else None
    )
    _emit_and_exit(
        verdict, output, emit_evidence=emit_evidence, push=push, provenance=provenance
    )


__all__ = ["app", "ship"]
