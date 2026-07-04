"""Pure MCP tool registry for ``soup mcp serve`` (v0.71.28).

This module has **no** dependency on the ``mcp`` SDK: it defines the tool
table (:class:`ToolSpec`), the handler functions (each a pure
``(dict) -> dict``), and the shared security guards. :mod:`soup_cli.mcp_server.server`
is the only file that imports the SDK, and it consumes this registry.

Every handler lazy-imports its light core inside the function body so that
importing this module stays cheap and torch-free.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

# Default read cap for JSON tool arguments (mirrors ship/diagnose evidence).
_MAX_JSON_BYTES = 16 * 1024 * 1024

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
    """One entry in the MCP tool table."""

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


def _read_json_under_cwd(path: str, field: str, *, max_bytes: int = _MAX_JSON_BYTES) -> dict:
    """Load a JSON object argument (cwd-contained, symlink-rejected, size-capped).

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
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise McpToolError(f"{field} is not valid JSON") from exc
    except OSError as exc:
        raise McpToolError(f"{field} is unreadable ({type(exc).__name__})") from exc
    if not isinstance(payload, dict):
        raise McpToolError(f"{field} must contain a JSON object")
    return payload


# ---------------------------------------------------------------------------
# Argument helpers — every handler validates its own args (the SDK also
# jsonschema-validates inputSchema, but handlers must not trust that alone).
# Error messages NEVER echo raw user input (avoids ANSI/path injection).
# ---------------------------------------------------------------------------


def _require_str(args: dict, key: str) -> str:
    val = args.get(key)
    if not isinstance(val, str) or not val:
        raise McpToolError(f"'{key}' must be a non-empty string")
    return val


def _opt_str(args: dict, key: str) -> "str | None":
    val = args.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise McpToolError(f"'{key}' must be a string")
    return val


def _opt_int(args: dict, key: str, default: int, *, lo: int, hi: int) -> int:
    val = args.get(key, default)
    if isinstance(val, bool) or not isinstance(val, int):
        raise McpToolError(f"'{key}' must be an integer")
    return max(lo, min(hi, val))


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
    import dataclasses

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
    return dataclasses.asdict(verdict)


def _load_data_rows(path: str) -> list:
    from pathlib import Path

    from soup_cli.data.loader import load_raw_data

    _enforce_data_path(path)
    try:
        return load_raw_data(Path(path))
    except (OSError, ValueError) as exc:
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
    try:
        tok = _dd.resolve_tokenizer(model, trust_remote_code=False)
    except ImportError as exc:
        raise McpToolError(
            "data_doctor needs the tokenizer stack: pip install 'soup-cli[train]'"
        ) from exc
    except (ValueError, TypeError, OSError) as exc:
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


# ---------------------------------------------------------------------------
# Tool table
# ---------------------------------------------------------------------------

_DATA_ARG = {
    "type": "string",
    "description": "Path to a JSONL/JSON dataset under the working directory.",
}


def _readonly_specs() -> "list[ToolSpec]":
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
    ]


def build_registry(*, allow_mutating: bool) -> "list[ToolSpec]":
    """Assemble the MCP tool table.

    The read-only tools are always present. ``allow_mutating`` gates the
    plan-only mutating tools (added in a later part).
    """
    specs = _readonly_specs()
    return specs
