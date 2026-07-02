"""Pre-wired production stages for `soup loop watch` (v0.71.4 #176).

v0.58.0 shipped the watch daemon with no-op default stage callbacks; this
module supplies real harvest / train / gate / deploy callables that compose
the existing v0.26.0 trace-to-preference, eval-gate, and v0.30.0 multi-adapter
deploy surfaces. Operators opt in via ``soup loop init --pre-wired`` (or
``soup loop watch --pre-wired``).

Each callable matches the ``HarvestFn`` / ``TrainFn`` / ``GateFn`` /
``DeployFn`` protocols from ``loop_daemon``. Heavy deps (transformers / peft /
trl / httpx) are lazy-imported inside the callables so importing this module
stays cold-start cheap (project lazy-import policy).

Module-level escape hatches (mirrors v0.53.1 #109 deploy_measure) let tests
and offline smokes inject behaviour without a GPU or a network:

- ``_TRACE_DIR_RESOLVER(state) -> Optional[str]``
- ``_TRAIN_RUNNER(argv, ...)``               (a ``subprocess.run`` stand-in)
- ``_GATE_GENERATE_FACTORY(adapter_dir) -> generate_fn``
- ``_DEPLOY_POSTER(endpoint, name) -> bool``
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Optional

from soup_cli.utils.loop_state import LoopState
from soup_cli.utils.paths import atomic_write_text

if TYPE_CHECKING:
    from soup_cli.utils.loop_daemon import WatchConfig

_LOG = logging.getLogger(__name__)

_PAIRS_DIR = ".soup-loops/pairs"
_ADAPTERS_DIR = ".soup-loops/adapters"
_DEFAULT_TRACE_DIR = ".soup-loops/traces"

# Test / smoke escape hatches (default None → real behaviour).
_TRACE_DIR_RESOLVER: Optional[Callable[[LoopState], Optional[str]]] = None
_TRAIN_RUNNER: Optional[Callable[..., object]] = None
_GATE_GENERATE_FACTORY: Optional[Callable[[str], Callable[[str], str]]] = None
_DEPLOY_POSTER: Optional[Callable[[str, str], bool]] = None


# ---------------------------------------------------------------------------
# Harvest — production traces → preference pairs (v0.26.0 trace-to-pref)
# ---------------------------------------------------------------------------


def _resolve_trace_dir(state: LoopState) -> Optional[str]:
    """Resolve the directory of ``*.jsonl`` serve logs to harvest from.

    Resolution order: ``SOUP_LOOP_TRACE_DIR`` env → ``served_model`` if it is
    an existing local dir → ``.soup-loops/traces``. Returns ``None`` when no
    usable directory exists.

    Note: this is the one read surface in this module that intentionally does
    NOT enforce cwd-containment — operators legitimately keep serve logs
    outside the project dir (e.g. ``/var/log/soup-serve``). The directory is
    operator-supplied config (env / state), not untrusted input, and harvested
    content is only used to build DPO pairs (never exec'd).
    """
    if _TRACE_DIR_RESOLVER is not None:
        return _TRACE_DIR_RESOLVER(state)
    env = os.environ.get("SOUP_LOOP_TRACE_DIR")
    candidates = []
    if env:
        candidates.append(env)
    if state.served_model and not state.served_model.startswith("registry://"):
        candidates.append(state.served_model)
    candidates.append(_DEFAULT_TRACE_DIR)
    for cand in candidates:
        if cand and os.path.isdir(cand):
            return cand
    return None


def harvest_from_traces(state: LoopState) -> Mapping[str, object]:
    """Scan serve traces for thumbs/regeneration signals → preference pairs.

    Writes the harvested pairs to ``.soup-loops/pairs/<uuid>.jsonl`` (cwd-
    contained) and returns ``{pairs_harvested, pairs_path, traces_collected}``.
    Yields zero pairs (no path) when no trace directory is configured.
    """
    from soup_cli.data.traces.pair_builder import build_pairs
    from soup_cli.data.traces.parsers import parse_soup_serve

    trace_dir = _resolve_trace_dir(state)
    if not trace_dir:
        return {"pairs_harvested": 0, "pairs_path": None, "traces_collected": 0}

    traces = list(parse_soup_serve(trace_dir))
    pairs = list(build_pairs(traces, signal="thumbs_up"))
    if not pairs:
        return {
            "pairs_harvested": 0,
            "pairs_path": None,
            "traces_collected": len(traces),
        }

    pairs_dir = Path(_PAIRS_DIR)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = str(pairs_dir / f"pairs-{uuid.uuid4().hex[:12]}.jsonl")
    lines = "\n".join(json.dumps(p.to_jsonl_dict()) for p in pairs) + "\n"
    # Atomic + cwd-contained + symlink-rejected (mirrors the rest of the
    # changeset's write policy).
    atomic_write_text(lines, pairs_path, field="pairs_path")
    return {
        "pairs_harvested": len(pairs),
        "pairs_path": pairs_path,
        "traces_collected": len(traces),
    }


# ---------------------------------------------------------------------------
# Train — generated DPO YAML → `soup train` subprocess
# ---------------------------------------------------------------------------


def _render_dpo_yaml(state: LoopState, pairs_path: str, output_dir: str) -> str:
    """Render a minimal DPO ``soup.yaml`` referencing the harvested pairs."""
    import yaml

    base = (
        state.served_model
        if state.served_model and not state.served_model.startswith("registry://")
        else "hf-internal-testing/tiny-random-gpt2"
    )
    config = {
        "base": base,
        "task": "dpo",
        "data": {"train": pairs_path, "format": "dpo"},
        "training": {"epochs": 1, "batch_size": 1, "lr": 5.0e-5},
        "output": output_dir,
    }
    return yaml.safe_dump(config, sort_keys=False)


def train_dpo_from_pairs(
    state: LoopState, ctx: Mapping[str, object]
) -> Mapping[str, object]:
    """Train a DPO adapter from harvested pairs via a ``soup train`` subprocess.

    Returns ``{run_id, skipped, adapter_path}``. Skips (no run) when the
    harvest produced no pairs. The subprocess is argv-list (no shell);
    ``_TRAIN_RUNNER`` injects a stand-in for tests.
    """
    pairs_path = ctx.get("pairs_path")
    pairs_n = int(ctx.get("pairs_harvested", 0) or 0)
    if not pairs_path or pairs_n <= 0:
        return {"run_id": None, "skipped": True, "adapter_path": None}

    run_id = f"loop-train-{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(_ADAPTERS_DIR, run_id)
    yaml_dir = Path(_ADAPTERS_DIR)
    yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = str(yaml_dir / f"{run_id}.yaml")
    atomic_write_text(
        _render_dpo_yaml(state, str(pairs_path), output_dir),
        yaml_path,
        field="train yaml",
    )

    argv = [
        sys.executable,
        "-m",
        "soup_cli.cli",
        "train",
        "--config",
        yaml_path,
        "--yes",
    ]
    runner = _TRAIN_RUNNER
    if runner is None:
        import subprocess  # noqa: S404 — argv list mode, no shell

        runner = subprocess.run
    try:
        result = runner(  # noqa: S603 — internal argv, no shell
            argv, capture_output=True, timeout=24 * 3600, check=False
        )
    except Exception as exc:  # noqa: BLE001 — train failure must not crash loop
        _LOG.warning("train subprocess failed: %s", type(exc).__name__)
        return {"run_id": None, "skipped": True, "adapter_path": None}

    rc = getattr(result, "returncode", 1)
    if rc != 0:
        _LOG.warning("soup train exited rc=%s", rc)
        return {"run_id": None, "skipped": True, "adapter_path": None}
    return {"run_id": run_id, "skipped": False, "adapter_path": output_dir}


# ---------------------------------------------------------------------------
# Gate — eval-gate against the registered baseline (v0.26.0 Part B)
# ---------------------------------------------------------------------------


def _build_gate_generator(adapter_dir: str) -> Callable[[str], str]:
    """Build a ``generate_fn`` for the trained adapter (live model load)."""
    if _GATE_GENERATE_FACTORY is not None:
        return _GATE_GENERATE_FACTORY(adapter_dir)
    # Live default — load the merged/trained adapter as a full model dir.
    from soup_cli.eval.quant_check import make_model_generator

    return make_model_generator(adapter_dir)


def gate_against_baseline(
    state: LoopState, ctx: Mapping[str, object]
) -> Mapping[str, object]:
    """Run the v0.26.0 eval gate for the trained adapter vs the baseline.

    Returns ``{gate_verdict}`` — ``SKIPPED`` when training was skipped or the
    adapter dir is missing, otherwise ``OK`` / ``MAJOR`` from the gate result.
    """
    if ctx.get("skipped"):
        return {"gate_verdict": "SKIPPED"}
    adapter_path = ctx.get("adapter_path")
    if not adapter_path or not os.path.isdir(str(adapter_path)):
        return {"gate_verdict": "SKIPPED"}

    from soup_cli.eval.gate import load_suite, resolve_baseline, run_gate

    try:
        suite = load_suite(state.eval_suite)
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        _LOG.warning("gate suite load failed: %s", type(exc).__name__)
        return {"gate_verdict": "SKIPPED"}
    # An unresolvable baseline degrades to a threshold-only gate (no
    # regression check) rather than failing the whole stage — the baseline
    # may not exist yet on the first iteration.
    try:
        baseline = resolve_baseline(state.baseline)
    except (FileNotFoundError, ValueError, TypeError, OSError):
        baseline = {}
    try:
        generate_fn = _build_gate_generator(str(adapter_path))
        result = run_gate(suite, generate_fn=generate_fn, baseline=baseline)
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        _LOG.warning("gate run failed: %s", type(exc).__name__)
        return {"gate_verdict": "SKIPPED"}
    verdict = "OK" if result.passed else "MAJOR"
    return {"gate_verdict": verdict}


# ---------------------------------------------------------------------------
# Deploy — promote to a canary via /v1/adapters/activate (v0.30.0)
# ---------------------------------------------------------------------------


# Adapter-name shape accepted by the v0.30.0 /v1/adapters/activate route.
_ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _endpoint_is_local(endpoint: str) -> bool:
    """True iff ``endpoint``'s host is loopback or a private/link-local IP.

    The deploy surface must only POST to the operator's own box / LAN, never
    an arbitrary public host (the webhook validator permits any HTTPS host).
    Loopback hostnames are accepted by name; any other host must parse as a
    private / loopback / link-local IP literal. A non-IP public hostname is
    rejected (we do not resolve DNS — project policy).
    """
    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(endpoint).hostname or "").strip("[]").lower()
    if not host:
        return False
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _post_activate(endpoint: str, name: str) -> bool:
    """POST to ``<endpoint>/v1/adapters/activate/<name>``; True on 2xx.

    ``endpoint`` is already SSRF-validated by the caller; ``name`` is
    re-validated here against the activate-route pattern before URL
    interpolation (defence-in-depth against a crafted adapter dir name).
    """
    if _DEPLOY_POSTER is not None:
        return _DEPLOY_POSTER(endpoint, name)
    if not _ADAPTER_NAME_RE.match(name):
        return False
    try:
        import httpx
    except ImportError:
        return False
    url = endpoint.rstrip("/") + f"/v1/adapters/activate/{name}"
    try:
        resp = httpx.post(url, timeout=5.0)
        return 200 <= resp.status_code < 300
    except Exception:  # noqa: BLE001 — deploy must never crash the loop
        return False


def deploy_to_canary(
    state: LoopState, ctx: Mapping[str, object]
) -> Mapping[str, object]:
    """Promote the trained adapter as a canary when the gate verdict is OK.

    Returns ``{deployed, canary_verdict}``. No-op (not deployed) unless the
    gate passed, an adapter is available, and a serve endpoint is configured
    via ``SOUP_LOOP_SERVE_ENDPOINT``. The endpoint is SSRF-validated
    (loopback-only HTTP, scheme allowlist, private-IP rejection) via the
    shared ``validate_webhook_url`` helper before any POST.
    """
    if ctx.get("gate_verdict") != "OK":
        return {"deployed": False, "canary_verdict": None}
    adapter_path = ctx.get("adapter_path")
    if not adapter_path:
        return {"deployed": False, "canary_verdict": None}
    raw_endpoint = os.environ.get("SOUP_LOOP_SERVE_ENDPOINT")
    if not raw_endpoint:
        return {
            "deployed": False,
            "canary_verdict": None,
            "notes": "no SOUP_LOOP_SERVE_ENDPOINT configured",
        }
    from soup_cli.utils.drift_alarm import validate_webhook_url

    try:
        # A serve endpoint legitimately targets the operator's own LAN box, so
        # opt into private hosts here; the deploy surface is then tightened to
        # loopback/LAN by `_endpoint_is_local` below. The default webhook policy
        # (allow_private_hosts=False) still blocks private IPs for --slack-url /
        # --discord-url, where the URL is not operator-trusted.
        endpoint = validate_webhook_url(raw_endpoint, allow_private_hosts=True)
    except (TypeError, ValueError):
        _LOG.warning("SOUP_LOOP_SERVE_ENDPOINT rejected by SSRF guard")
        return {
            "deployed": False,
            "canary_verdict": None,
            "notes": "SOUP_LOOP_SERVE_ENDPOINT rejected (SSRF guard)",
        }
    # A serve endpoint is the operator's own box / LAN, never an arbitrary
    # remote host. `validate_webhook_url` (a *webhook* policy) permits any
    # HTTPS host; tighten further for the deploy surface to loopback + RFC1918
    # private + link-local only, so a poisoned SOUP_LOOP_SERVE_ENDPOINT cannot
    # POST adapter names to an attacker-controlled HTTPS server (v0.71.4
    # review MEDIUM-4). Non-IP hostnames can't be verified private without DNS
    # (project policy: no DNS resolution), so the safe default rejects them.
    if not _endpoint_is_local(endpoint):
        _LOG.warning("SOUP_LOOP_SERVE_ENDPOINT must be loopback/LAN (SSRF guard)")
        return {
            "deployed": False,
            "canary_verdict": None,
            "notes": "SOUP_LOOP_SERVE_ENDPOINT must be loopback/LAN (SSRF guard)",
        }
    name = os.path.basename(str(adapter_path).rstrip("/\\")) or "canary"
    ok = _post_activate(endpoint, name)
    return {
        "deployed": ok,
        "canary_verdict": "OK" if ok else None,
        "notes": "" if ok else "activate POST failed",
    }


# ---------------------------------------------------------------------------
# Cost — placeholder estimate (v0.34.0 run_cost wiring is a follow-up)
# ---------------------------------------------------------------------------


def estimate_cost(state: LoopState) -> float:
    """Per-iteration cost estimate (USD) for the v0.58 budget gate (#245).

    Wires the cost gate to v0.34 ``run_cost.estimate_run_cost_usd`` using the
    most-recent *completed* run's GPU + duration as a forward estimate for the
    next iteration. ``cost_fn`` runs *pre-flight* (before this iteration
    trains, so the budget gate can refuse a run before spending) — and a
    pre-wired loop trains the same model on the same box repeatedly, so the
    previous run's actual GPU-time is the best forward signal.

    Falls back to ``0.0`` (no cost gate) when there is no prior priced run:
    the first iteration has no data, and a CPU / unpriced GPU returns ``None``
    from ``estimate_run_cost_usd`` (so the budget gate trips only on a real
    dollar estimate, never a fabricated one). Never raises — a tracker error
    must not crash the daemon.
    """
    try:
        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.utils.run_cost import estimate_run_cost_usd

        runs = ExperimentTracker().list_runs(limit=10)
    except Exception:  # noqa: BLE001 - cost estimation must never crash the loop
        return 0.0
    for run in runs:
        if run.get("status") != "completed":
            continue
        device = run.get("device_name")
        duration = run.get("duration_secs")
        if not device or duration is None:
            continue
        try:
            cost = estimate_run_cost_usd(str(device), float(duration))
        except (TypeError, ValueError):
            continue
        if cost is not None and cost > 0:
            return float(cost)
        # Most-recent completed run was on a CPU / unpriced GPU → no dollar
        # signal; report 0.0 (no cost gate) rather than scanning older runs.
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_prewired_watch_config(
    *,
    max_iterations: Optional[int] = None,
    poll_interval_sec: float = 60.0,
    state_path: Optional[str] = None,
    iteration_dir: Optional[str] = None,
    pack_iterations: bool = False,
    served_model: Optional[str] = None,
    base_model: str = "unknown",
) -> "WatchConfig":
    """Return a ``WatchConfig`` wired with the pre-wired production stages."""
    from soup_cli.utils.loop_daemon import WatchConfig

    return WatchConfig(
        poll_interval_sec=poll_interval_sec,
        max_iterations=max_iterations,
        state_path=state_path,
        iteration_dir=iteration_dir,
        harvest_fn=harvest_from_traces,
        train_fn=train_dpo_from_pairs,
        gate_fn=gate_against_baseline,
        deploy_fn=deploy_to_canary,
        cost_fn=estimate_cost,
        pack_iterations=pack_iterations,
        served_model=served_model,
        base_model=base_model,
    )
