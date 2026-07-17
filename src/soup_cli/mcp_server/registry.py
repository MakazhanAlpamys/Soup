"""Pure MCP tool registry for ``soup mcp serve`` (v0.71.28).

This module has **no** dependency on the ``mcp`` SDK: it defines the tool
table (:class:`ToolSpec`), the handler functions (each a pure
``(dict) -> dict``), and the shared security guards. :mod:`soup_cli.mcp_server.server`
is the only file that imports the SDK, and it consumes this registry.

Every handler lazy-imports its light core inside the function body so that
importing this module stays cheap and torch-free.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

if TYPE_CHECKING:
    from soup_cli.config.schema import SoupConfig

# Default read cap for JSON tool arguments (mirrors ship/diagnose evidence).
_MAX_JSON_BYTES = 16 * 1024 * 1024
# Cap on `data` dataset loads — the server is long-lived, so a client must not
# be able to point `data` at an arbitrarily large file and exhaust memory
# (mirrors advise's own 1 GiB cap; security-review MEDIUM).
_MAX_DATA_BYTES = 1024 * 1024 * 1024

# C0 control bytes (keep tab / newline / CR) + DEL, stripped from every string
# in a handler result before it reaches the MCP client. ``rich.markup.escape``
# only neutralises ``[...]`` markup, not raw ESC/OSC sequences a malicious
# dataset string could smuggle into a client's terminal. Mirrors
# ``commands/data_doctor.py::_CONTROL_STRIP_TABLE``.
_CONTROL_STRIP_TABLE = {i: None for i in range(0x20) if i not in (0x09, 0x0A, 0x0D)}
_CONTROL_STRIP_TABLE[0x7F] = None


class McpToolError(Exception):
    """A tool-level failure with a pre-sanitized, path-free message.

    The MCP SDK stringifies a raised exception verbatim into an ``isError``
    result, so handlers must raise THIS (never a bare ``OSError`` whose text
    could leak a filesystem path).
    """


@dataclass(frozen=True)
class ToolSpec:
    """One entry in the MCP tool table.

    ``frozen=True`` blocks attribute rebinding; ``input_schema`` contents are
    still technically mutable, but the table is built fresh per server and
    never mutated in place.
    """

    name: str
    title: str
    description: str
    input_schema: dict
    handler: Callable[[dict], dict]
    mutating: bool = False


def _sanitize(obj: Any) -> Any:
    """Recursively strip C0/ESC/DEL bytes from every string in ``obj``.

    Leaves non-string scalars (int/float/bool/None) untouched; recurses into
    dicts and lists. Applied to every handler result as defence-in-depth.
    """
    if isinstance(obj, str):
        return obj.translate(_CONTROL_STRIP_TABLE)
    if isinstance(obj, Mapping):
        return {_sanitize(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _read_text_under_cwd(path: str, field: str, *, max_bytes: int = _MAX_JSON_BYTES) -> str:
    """Read a text file argument (cwd-contained, symlink-rejected, size-capped).

    Opens with ``O_NOFOLLOW`` (where available) and fstats the open fd so a
    symlink swapped in after the containment check cannot redirect the read
    (TOCTOU defence, mirrors ``commands/ship.py::_load_evidence``). Raises
    :class:`McpToolError` with a path-free message on any failure.
    """
    try:
        enforce_under_cwd_and_no_symlink(path, field)
    except Exception as exc:  # ValueError / OSError from the guard
        raise McpToolError(f"{field} must be a readable file under the working directory") from exc
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        handle_fd = os.open(path, flags)
    except OSError as exc:
        raise McpToolError(f"{field} is unreadable ({type(exc).__name__})") from exc
    try:
        with os.fdopen(handle_fd, "r", encoding="utf-8") as handle:
            if os.fstat(handle.fileno()).st_size > max_bytes:
                raise McpToolError(f"{field} exceeds {max_bytes} bytes")
            return handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise McpToolError(f"{field} is unreadable ({type(exc).__name__})") from exc


def _read_json_under_cwd(path: str, field: str, *, max_bytes: int = _MAX_JSON_BYTES) -> dict:
    """Read a JSON *object* argument (delegates to :func:`_read_text_under_cwd`)."""
    text = _read_text_under_cwd(path, field, max_bytes=max_bytes)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise McpToolError(f"{field} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise McpToolError(f"{field} must contain a JSON object")
    return payload


# ---------------------------------------------------------------------------
# Argument helpers — every handler validates its own args (the SDK also
# jsonschema-validates inputSchema, but handlers must not trust that alone).
# Error messages NEVER echo raw user input (avoids ANSI/path injection).
# ---------------------------------------------------------------------------


# Generous cap on free-text string args (paths, ids, goals, queries). Bounds a
# pathological input without constraining any legitimate value (security-review).
_MAX_STR_LEN = 4096


def _require_str(args: dict, key: str) -> str:
    val = args.get(key)
    if not isinstance(val, str) or not val:
        raise McpToolError(f"'{key}' must be a non-empty string")
    if len(val) > _MAX_STR_LEN:
        raise McpToolError(f"'{key}' must be at most {_MAX_STR_LEN} characters")
    return val


def _opt_str(args: dict, key: str) -> str | None:
    val = args.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise McpToolError(f"'{key}' must be a string")
    if len(val) > _MAX_STR_LEN:
        raise McpToolError(f"'{key}' must be at most {_MAX_STR_LEN} characters")
    return val


def _opt_int(args: dict, key: str, default: int, *, lo: int, hi: int) -> int:
    val = args.get(key, default)
    if isinstance(val, bool) or not isinstance(val, int):
        raise McpToolError(f"'{key}' must be an integer")
    # Reject rather than silently clamp — clamping hides the user's error and
    # bypasses the core's own bounds check (code-review MEDIUM).
    if not lo <= val <= hi:
        raise McpToolError(f"'{key}' must be between {lo} and {hi}")
    return val


def _enforce_data_path(path: str, field: str = "data") -> None:
    try:
        enforce_under_cwd_and_no_symlink(path, field)
    except Exception as exc:  # ValueError / OSError from the guard
        raise McpToolError(
            f"'{field}' must be a readable file under the working directory"
        ) from exc


# ---------------------------------------------------------------------------
# Read-only tool handlers (each a pure ``(dict) -> dict``)
# ---------------------------------------------------------------------------


def tool_advise(args: dict) -> dict:
    """`soup advise` — pre-flight PROMPT_ENG / RAG / SFT / DPO / GRPO verdict."""
    from soup_cli.utils import advise as _advise

    data = _require_str(args, "data")
    goal = _opt_str(args, "goal")
    _enforce_data_path(data)
    try:
        rows = _advise.load_advise_dataset(data)
        task = _advise.classify_task(rows, goal)
        profile = _advise.compute_dataset_profile(rows)
        verdict = _advise.build_verdict(profile, task, goal=goal)
    except (OSError, ValueError, TypeError) as exc:
        raise McpToolError(f"advise failed ({type(exc).__name__})") from exc
    return asdict(verdict)


def _load_data_rows(path: str) -> list[dict]:
    from soup_cli.data.loader import load_raw_data

    _enforce_data_path(path)
    # Best-effort size cap before load_raw_data reads the whole file into memory
    # (the path is already confirmed non-symlink + under cwd).
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise McpToolError(f"data is unreadable ({type(exc).__name__})") from exc
    if size > _MAX_DATA_BYTES:
        raise McpToolError(f"data exceeds {_MAX_DATA_BYTES} bytes")
    # load_raw_data dispatches by extension: parquet raises bare ImportError
    # without pandas, CSV raises csv.Error (NOT a ValueError subclass) on
    # malformed input. Translate every loader failure here (code-review MEDIUM).
    try:
        return load_raw_data(Path(path))
    except (OSError, ValueError, ImportError, csv.Error) as exc:
        raise McpToolError(f"cannot load data ({type(exc).__name__})") from exc


def tool_data_inspect(args: dict) -> dict:
    """`soup data inspect` — dataset stats."""
    from soup_cli.data.validator import validate_and_stats

    rows = _load_data_rows(_require_str(args, "data"))
    return validate_and_stats(rows)


def tool_data_validate(args: dict) -> dict:
    """`soup data validate` — format-compliance report."""
    from soup_cli.data.validator import validate_and_stats

    rows = _load_data_rows(_require_str(args, "data"))
    fmt = _opt_str(args, "format")
    return validate_and_stats(rows, expected_format=fmt)


def tool_data_score(args: dict) -> dict:
    """`soup data score` — PII / toxicity / language / educational scorecard."""
    from soup_cli.utils.data_score import compute_scorecard

    rows = _load_data_rows(_require_str(args, "data"))
    rep = compute_scorecard(rows)
    return {
        "total": rep.total,
        "pii_flagged": rep.pii_flagged,
        "toxic_flagged": rep.toxic_flagged,
        "decontaminated_removed": rep.decontaminated_removed,
        "languages": dict(rep.languages),
        "educational_mean": rep.educational_mean,
    }


def tool_data_doctor(args: dict) -> dict:
    """`soup data doctor` — chat-template compat report (needs the tokenizer stack)."""
    from soup_cli.data import formats as _formats
    from soup_cli.utils import data_doctor as _dd

    rows = _load_data_rows(_require_str(args, "data"))
    model = _require_str(args, "model")
    fmt = _opt_str(args, "format") or "auto"
    max_length = _opt_int(args, "max_length", 2048, lo=64, hi=1_048_576)
    sample_size = _opt_int(args, "sample_size", 200, lo=1, hi=2000)
    if fmt == "auto":
        try:
            fmt = _formats.detect_format(rows)
        except ValueError as exc:
            raise McpToolError("could not auto-detect data format; pass 'format'") from exc
    # `resolve_tokenizer` catches the transformers-missing case internally and
    # re-raises it as a ValueError, so probe the dependency directly to keep the
    # actionable "install the extra" hint (python-review HIGH).
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise McpToolError(
            "data_doctor needs the tokenizer stack: pip install \"soup-cli[train]\""
        ) from exc
    try:
        tok = _dd.resolve_tokenizer(model, trust_remote_code=False)
    except (ImportError, ValueError, TypeError, OSError) as exc:
        raise McpToolError(f"could not load tokenizer ({type(exc).__name__})") from exc
    try:
        report = _dd.run_doctor(
            rows, tok, fmt=fmt, max_length=max_length, sample_size=sample_size
        )
    except (ValueError, TypeError) as exc:
        raise McpToolError(f"data doctor failed ({type(exc).__name__})") from exc
    return report.to_dict()


def tool_recipes_search(args: dict) -> dict:
    """`soup recipes search` — compact recipe list (no yaml body)."""
    from soup_cli.recipes.catalog import RECIPES, search_recipes

    results = search_recipes(
        _opt_str(args, "query"), _opt_str(args, "task"), _opt_str(args, "size")
    )
    name_by_id = {id(meta): name for name, meta in RECIPES.items()}
    out = [
        {
            "name": name_by_id.get(id(meta), "?"),
            "model": meta.model,
            "task": meta.task,
            "size": meta.size,
            "tags": list(meta.tags),
            "description": meta.description,
        }
        for meta in results
    ]
    return {"results": out, "count": len(out)}


def tool_recipes_show(args: dict) -> dict:
    """`soup recipes show` — full recipe incl. the YAML body."""
    from soup_cli.recipes.catalog import get_recipe

    name = _require_str(args, "name")
    meta = get_recipe(name)
    if meta is None:
        raise McpToolError("unknown recipe (try recipes_search)")
    return {
        "name": name,
        "model": meta.model,
        "task": meta.task,
        "size": meta.size,
        "tags": list(meta.tags),
        "description": meta.description,
        "yaml_str": meta.yaml_str,
    }


def tool_runs_list(args: dict) -> dict:
    """`soup runs` — recent experiment runs."""
    from soup_cli.experiment.tracker import ExperimentTracker

    limit = _opt_int(args, "limit", 50, lo=1, hi=500)
    runs = ExperimentTracker().list_runs(limit=limit)
    return {"runs": runs, "count": len(runs)}


def tool_runs_show(args: dict) -> dict:
    """`soup runs show` — one run's full record."""
    from soup_cli.experiment.tracker import ExperimentTracker

    run = ExperimentTracker().get_run(_require_str(args, "run_id"))
    if run is None:
        raise McpToolError("run not found")
    return run


def tool_registry_list(args: dict) -> dict:
    """`soup registry list` — model registry entries."""
    from soup_cli.registry.store import RegistryStore

    limit = _opt_int(args, "limit", 100, lo=1, hi=500)
    with RegistryStore() as store:
        entries = store.list(
            name=_opt_str(args, "name"),
            tag=_opt_str(args, "tag"),
            base=_opt_str(args, "base"),
            task=_opt_str(args, "task"),
            limit=limit,
        )
    return {"entries": entries, "count": len(entries)}


def tool_registry_show(args: dict) -> dict:
    """`soup registry show` — one registry entry (id / prefix / name:tag / registry://)."""
    from soup_cli.registry.store import AmbiguousRefError, RegistryStore

    ref = _require_str(args, "ref")
    with RegistryStore() as store:
        try:
            entry_id = store.resolve(ref)
        except AmbiguousRefError as exc:
            raise McpToolError("ambiguous registry ref") from exc
        if entry_id is None:
            raise McpToolError("registry entry not found")
        entry = store.get(entry_id)
    if entry is None:
        raise McpToolError("registry entry not found")
    return entry


def _resolve_gpu_memory_mcp(gpu: str | None) -> float:
    """GPU memory in GB from a flag or auto-detection (non-Typer mirror of
    ``commands/profile.py::_resolve_gpu_memory``)."""
    from soup_cli.utils.profiler import GPU_MEMORY

    if gpu is not None:
        gpu_key = gpu.lower().replace(" ", "").replace("-", "")
        if gpu_key not in GPU_MEMORY:
            raise McpToolError("unknown gpu (see 'soup profile --help' for valid options)")
        return float(GPU_MEMORY[gpu_key])
    try:
        from soup_cli.utils.gpu import get_gpu_info

        info = get_gpu_info()
        mem_bytes = info.get("memory_total_bytes", 0)
        if mem_bytes > 0:
            return mem_bytes / (1024**3)
    except (ImportError, RuntimeError, OSError):
        pass
    return 24.0


def _load_config_under_cwd(config: str) -> SoupConfig:
    """Read + validate a soup.yaml via the API-safe loader.

    Uses ``load_config_from_string`` (raises ``ValueError``) NOT ``load_config``
    (which prints to stdout + ``sys.exit`` — both fatal to the MCP stdio stream).
    """
    from soup_cli.config.loader import load_config_from_string

    text = _read_text_under_cwd(config, "config")
    try:
        return load_config_from_string(text)
    except ValueError as exc:
        raise McpToolError(f"invalid config ({type(exc).__name__})") from exc


def tool_profile(args: dict) -> dict:
    """`soup profile` — memory / speed / GPU estimate from a soup.yaml (no model load)."""
    from soup_cli.utils.gpu import model_size_from_name
    from soup_cli.utils.profiler import (
        estimate_speed,
        estimate_total,
        recommend_batch_size,
        recommend_gpu,
    )

    cfg = _load_config_under_cwd(_require_str(args, "config"))
    gpu = _opt_str(args, "gpu")

    model_params_b = model_size_from_name(cfg.base)
    batch_size = cfg.training.batch_size
    batch_size = 4 if batch_size == "auto" else int(batch_size)
    gpu_memory_gb = _resolve_gpu_memory_mcp(gpu)

    result = estimate_total(
        model_name=cfg.base,
        model_params_b=model_params_b,
        quantization=cfg.training.quantization,
        lora_r=cfg.training.lora.r,
        lora_alpha=cfg.training.lora.alpha,
        batch_size=batch_size,
        seq_len=cfg.data.max_length,
        optimizer=cfg.training.optimizer,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    )
    tokens_per_sec = estimate_speed(model_params_b, cfg.training.quantization, batch_size)
    result["tokens_per_sec"] = round(tokens_per_sec, 1)
    result["samples_per_sec"] = round(tokens_per_sec / max(cfg.data.max_length, 1), 2)
    result["recommended_batch_size"] = recommend_batch_size(
        result["total_memory_gb"], gpu_memory_gb
    )
    result["compatible_gpus"] = recommend_gpu(result["total_memory_gb"])
    result["gpu_memory_gb"] = gpu_memory_gb
    return result


def tool_diagnose_evidence(args: dict) -> dict:
    """`soup diagnose --evidence` — failure-mode report card from pre-computed scores."""
    from soup_cli import __version__
    from soup_cli.utils.diagnose.report import FAILURE_MODES, FailureScore, classify_score
    from soup_cli.utils.diagnose.runner import build_report

    run_id = _require_str(args, "run_id")
    payload = _read_json_under_cwd(_require_str(args, "evidence"), "evidence")
    base = _opt_str(args, "base") or ""
    adapter = _opt_str(args, "adapter") or ""

    raw_scores = payload.get("scores", {})
    if not isinstance(raw_scores, dict):
        raise McpToolError("evidence.scores must be an object")
    scores = {}
    for mode in FAILURE_MODES:  # closed set — safe to echo in errors
        entry = raw_scores.get(mode)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise McpToolError(f"evidence.scores.{mode} must be an object")
        score = entry.get("score", 1.0)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise McpToolError(f"evidence.scores.{mode}.score must be a number")
        # classify_score rejects a score outside [0, 1] / non-finite, and
        # FailureScore.__post_init__ rejects a mismatched/unknown verdict — both
        # ValueError. Guard them into a specific McpToolError (code-review HIGH).
        try:
            verdict = entry.get("verdict") or classify_score(score)
            scores[mode] = FailureScore(
                mode=mode,
                score=float(score),
                verdict=verdict,
                evidence=str(entry.get("evidence", "supplied via evidence")),
            )
        except (ValueError, TypeError, OverflowError) as exc:
            raise McpToolError(f"evidence.scores.{mode} is invalid ({type(exc).__name__})") from exc
    try:
        report = build_report(
            run_id=run_id, base=base, adapter=adapter, scores=scores, soup_version=__version__
        )
    except (ValueError, TypeError) as exc:
        raise McpToolError(f"diagnose failed ({type(exc).__name__})") from exc
    return report.to_dict()


def tool_ship_evidence(args: dict) -> dict:
    """`soup ship --evidence` — SHIP / DON'T-SHIP verdict from pre-computed scores."""
    from soup_cli.utils.ship_verdict import (
        SUPPORTED_TASK_MODES,
        build_task_win,
        compute_benchmark_deltas,
        decide_ship,
        verdict_to_dict,
    )

    payload = _read_json_under_cwd(_require_str(args, "evidence"), "evidence")
    threshold = args.get("forgetting_threshold", 0.05)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise McpToolError("'forgetting_threshold' must be a number")
    threshold = float(threshold)
    # Inclusive [0, 1] matches ship_verdict._validate_threshold + the CLI (a
    # 0.0 zero-tolerance gate is legitimate) (code-review LOW).
    if not 0.0 <= threshold <= 1.0:
        raise McpToolError("'forgetting_threshold' must be in [0, 1]")

    task = payload.get("task")
    if not isinstance(task, dict):
        raise McpToolError("evidence.task must be an object with mode/base/tuned")
    mode = task.get("mode", "metric")
    if mode not in SUPPORTED_TASK_MODES:
        raise McpToolError("evidence.task.mode must be 'metric' or 'judge_score'")
    if "base" not in task or "tuned" not in task:
        raise McpToolError("evidence.task needs both 'base' and 'tuned'")
    try:
        task_win = build_task_win(mode, task["base"], task["tuned"])
    except (TypeError, ValueError) as exc:
        raise McpToolError(f"invalid evidence.task ({type(exc).__name__})") from exc

    raw_bench = payload.get("benchmarks", {})
    if not isinstance(raw_bench, dict):
        raise McpToolError("evidence.benchmarks must be an object of {name: {base, tuned}}")
    base_scores: dict = {}
    tuned_scores: dict = {}
    for name, entry in raw_bench.items():
        if not isinstance(entry, dict) or "base" not in entry or "tuned" not in entry:
            raise McpToolError("each evidence.benchmarks entry needs 'base' and 'tuned'")
        base_scores[str(name)] = entry["base"]
        tuned_scores[str(name)] = entry["tuned"]
    try:
        deltas = compute_benchmark_deltas(base_scores, tuned_scores, forgetting_threshold=threshold)
        verdict = decide_ship(task_win, deltas, forgetting_threshold=threshold)
    except (TypeError, ValueError) as exc:
        raise McpToolError(f"invalid evidence.benchmarks ({type(exc).__name__})") from exc
    return verdict_to_dict(verdict)


# ---------------------------------------------------------------------------
# Mutating tool handlers — PLAN-ONLY in v1: they validate + render the exact
# command that WOULD run, but never execute. Live execution is a follow-up.
# ---------------------------------------------------------------------------

_MUTATING_NOTE = (
    "plan-only: 'soup mcp serve' does not execute this. Run the command "
    "yourself to proceed."
)


def tool_train_start(args: dict) -> dict:
    """`soup train` (plan-only) — validate a soup.yaml + render the command."""
    config = _require_str(args, "config")
    cfg = _load_config_under_cwd(config)
    return {
        "config_valid": True,
        "task": cfg.task,
        "base": cfg.base,
        "would_run": f"soup train --config {shlex.quote(config)}",
        "note": _MUTATING_NOTE,
    }


def tool_export(args: dict) -> dict:
    """`soup export` (plan-only) — validate format + render the command."""
    from soup_cli.commands.export import SUPPORTED_FORMATS

    model = _require_str(args, "model")
    fmt = _require_str(args, "format")
    if fmt not in SUPPORTED_FORMATS:
        raise McpToolError("unsupported export format (see 'soup export --help')")
    output = _opt_str(args, "output")
    cmd = f"soup export --model {shlex.quote(model)} --format {fmt}"
    if output:
        cmd += f" --output {shlex.quote(output)}"
    return {"format": fmt, "would_run": cmd, "note": _MUTATING_NOTE}


# ---------------------------------------------------------------------------
# Tool table
# ---------------------------------------------------------------------------

_DATA_ARG = {
    "type": "string",
    "description": "Path to a JSONL/JSON dataset under the working directory.",
}


def _readonly_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="advise",
            title="Advise",
            description=(
                "Pre-flight recommendation (PROMPT_ENG / RAG / SFT / DPO / GRPO) "
                "for a dataset + goal."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "data": _DATA_ARG,
                    "goal": {
                        "type": "string",
                        "description": "Optional stated goal, e.g. 'improve summaries'.",
                    },
                },
                "required": ["data"],
                "additionalProperties": False,
            },
            handler=tool_advise,
        ),
        ToolSpec(
            name="data_inspect",
            title="Inspect dataset",
            description="Dataset stats: row count, columns, length distribution, duplicates.",
            input_schema={
                "type": "object",
                "properties": {"data": _DATA_ARG},
                "required": ["data"],
                "additionalProperties": False,
            },
            handler=tool_data_inspect,
        ),
        ToolSpec(
            name="data_validate",
            title="Validate dataset",
            description=(
                "Format-compliance report: issues + valid-row count "
                "(alpaca/sharegpt/chatml/dpo/...)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "data": _DATA_ARG,
                    "format": {
                        "type": "string",
                        "description": "Expected format; omit to auto-detect.",
                    },
                },
                "required": ["data"],
                "additionalProperties": False,
            },
            handler=tool_data_validate,
        ),
        ToolSpec(
            name="data_score",
            title="Score dataset",
            description="Data-quality scorecard: PII, toxicity, language mix, educational value.",
            input_schema={
                "type": "object",
                "properties": {"data": _DATA_ARG},
                "required": ["data"],
                "additionalProperties": False,
            },
            handler=tool_data_score,
        ),
        ToolSpec(
            name="data_doctor",
            title="Chat-template doctor",
            description=(
                "Chat-template compatibility report vs a tokenizer (EOS-in-labels, "
                "BOS dup, truncation risk). Needs the soup-cli[train] tokenizer stack."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "data": _DATA_ARG,
                    "model": {
                        "type": "string",
                        "description": "Tokenizer model id or local path.",
                    },
                    "format": {
                        "type": "string",
                        "description": "Data format; omit to auto-detect.",
                    },
                    "max_length": {"type": "integer", "minimum": 64, "maximum": 1048576},
                    "sample_size": {"type": "integer", "minimum": 1, "maximum": 2000},
                },
                "required": ["data", "model"],
                "additionalProperties": False,
            },
            handler=tool_data_doctor,
        ),
        ToolSpec(
            name="recipes_search",
            title="Search recipes",
            description="Search the ready-made recipe catalog by keyword / task / model size.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "task": {"type": "string"},
                    "size": {"type": "string"},
                },
                "additionalProperties": False,
            },
            handler=tool_recipes_search,
        ),
        ToolSpec(
            name="recipes_show",
            title="Show recipe",
            description="Full recipe details incl. the ready-to-use soup.yaml body.",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Recipe name."}},
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=tool_recipes_show,
        ),
        ToolSpec(
            name="runs_list",
            title="List runs",
            description="Recent experiment runs from the local tracker.",
            input_schema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 500}},
                "additionalProperties": False,
            },
            handler=tool_runs_list,
        ),
        ToolSpec(
            name="runs_show",
            title="Show run",
            description="One run's full record (config, metrics summary). Accepts an id prefix.",
            input_schema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
                "additionalProperties": False,
            },
            handler=tool_runs_show,
        ),
        ToolSpec(
            name="registry_list",
            title="List registry",
            description="Model-registry entries, filterable by name/tag/base/task.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tag": {"type": "string"},
                    "base": {"type": "string"},
                    "task": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                "additionalProperties": False,
            },
            handler=tool_registry_list,
        ),
        ToolSpec(
            name="registry_show",
            title="Show registry entry",
            description="One registry entry by id / prefix / name:tag / registry:// ref.",
            input_schema={
                "type": "object",
                "properties": {"ref": {"type": "string"}},
                "required": ["ref"],
                "additionalProperties": False,
            },
            handler=tool_registry_show,
        ),
        ToolSpec(
            name="profile",
            title="Profile training",
            description=(
                "Estimate memory / speed / GPU fit from a soup.yaml before "
                "training (no model load)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "config": {
                        "type": "string",
                        "description": "Path to a soup.yaml under cwd.",
                    },
                    "gpu": {"type": "string", "description": "Target GPU, e.g. rtx4090 / a100."},
                },
                "required": ["config"],
                "additionalProperties": False,
            },
            handler=tool_profile,
        ),
        ToolSpec(
            name="diagnose_evidence",
            title="Diagnose (evidence)",
            description=(
                "Post-training failure-mode report card from a pre-computed "
                "evidence JSON (no model load)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "evidence": {
                        "type": "string",
                        "description": "Path to a diagnose evidence JSON under cwd.",
                    },
                    "base": {"type": "string"},
                    "adapter": {"type": "string"},
                },
                "required": ["run_id", "evidence"],
                "additionalProperties": False,
            },
            handler=tool_diagnose_evidence,
        ),
        ToolSpec(
            name="ship_evidence",
            title="Ship verdict (evidence)",
            description=(
                "SHIP / DON'T-SHIP verdict from a pre-computed evidence JSON "
                "(task win AND no regression)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "string",
                        "description": "Path to a ship evidence JSON under cwd.",
                    },
                    "forgetting_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["evidence"],
                "additionalProperties": False,
            },
            handler=tool_ship_evidence,
        ),
    ]


def _refuse_mutating(name: str) -> Callable[[dict], dict]:
    """Handler used for a mutating tool when ``--allow-mutating`` is off."""

    def _handler(args: dict) -> dict:
        raise McpToolError(
            f"'{name}' can change state and is disabled; restart with "
            "'soup mcp serve --allow-mutating' to enable (still plan-only in v1)."
        )

    return _handler


def _mutating_specs(*, allow_mutating: bool) -> list[ToolSpec]:
    """The plan-only mutating tools.

    Always LISTED (so clients can discover them), but their handler refuses
    unless ``allow_mutating`` is set. Even when enabled they only render the
    command that would run — v1 never executes training or export.
    """
    entries = [
        (
            "train_start",
            "Start training (plan-only)",
            "Validate a soup.yaml and render the 'soup train' command (does not execute).",
            {
                "type": "object",
                "properties": {
                    "config": {
                        "type": "string",
                        "description": "Path to a soup.yaml under cwd.",
                    }
                },
                "required": ["config"],
                "additionalProperties": False,
            },
            tool_train_start,
        ),
        (
            "export",
            "Export model (plan-only)",
            "Validate a format and render the 'soup export' command (does not execute).",
            {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Adapter/model path or id."},
                    "format": {"type": "string", "description": "gguf / onnx / awq / gptq / ..."},
                    "output": {"type": "string", "description": "Optional output path."},
                },
                "required": ["model", "format"],
                "additionalProperties": False,
            },
            tool_export,
        ),
    ]
    return [
        ToolSpec(
            name=name,
            title=title,
            description=description,
            input_schema=schema,
            handler=real if allow_mutating else _refuse_mutating(name),
            mutating=True,
        )
        for name, title, description, schema, real in entries
    ]


def build_registry(*, allow_mutating: bool) -> list[ToolSpec]:
    """Assemble the MCP tool table.

    The read-only tools are always present and executable. The mutating tools
    are always listed but refuse unless ``allow_mutating`` is set (and are
    plan-only even then).
    """
    return _readonly_specs() + _mutating_specs(allow_mutating=allow_mutating)
