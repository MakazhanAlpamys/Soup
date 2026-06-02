"""v0.53.6 #106 / v0.53.7 — Data Recipe DAG runner (live wave 1).

Per-node handlers for the 6 ``NODE_KINDS`` (``seed`` / ``llm_text`` /
``code`` / ``judge`` / ``validator`` / ``sampler``), topological execution
following the DAG's ``topo_order``, per-node checkpoint to
``<output_dir>/.checkpoint.json`` for resume-on-failure.

Per-node exceptions surface as ``status='failed'`` in the checkpoint and
re-raise so the caller's exit code reflects the failure. Inputs to each
node are the merged outputs of its predecessors (as listed in
``RecipeDAG.edges``).

Design notes:
- Each node's ``config`` dict is consulted for kind-specific options
  (``seed.path``, ``llm_text.prompt``, ``code.code``, ``validator.regex``
  / ``validator.schema``, ``sampler.name``). Missing keys raise
  ``ValueError`` per the loud-fail policy.
- ``llm_text`` + ``judge`` route through the v0.20.0 provider catalog via
  :func:`soup_cli.utils.data_forge.make_judge_provider_fn` (so the same
  SSRF / env-key surface applies).
- ``code`` reuses the v0.25.0 RLVR sandbox via
  :func:`soup_cli.trainer.rewards._run_code_sandbox`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat as _stat
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from soup_cli.utils.paths import is_under_cwd

if TYPE_CHECKING:  # pragma: no cover
    from soup_cli.utils.recipe_dag import RecipeDAG, RecipeNode

_LOG = logging.getLogger("soup_cli.utils.recipe_run")

# Per-node DoS caps.
_MAX_NODE_ROWS = 1_000_000
_MAX_CODE_LEN = 64 * 1024
_MAX_PROMPT_LEN = 32 * 1024


def _ensure_str(value: Any, *, name: str, max_len: int = 4096) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{name} exceeds {max_len} chars")
    return value


def _checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".checkpoint.json")


def _load_checkpoint(output_dir: str) -> Dict[str, Any]:
    path = _checkpoint_path(output_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        _LOG.debug("checkpoint load failed (%s); starting fresh", exc)
        return {}


def _save_checkpoint(output_dir: str, state: Mapping[str, Any]) -> None:
    """Atomic checkpoint write via mkstemp + os.replace.

    v0.53.7 M-E + H-C: replaces the v0.53.6 ``path + ".tmp"`` pattern which
    was both non-atomic on the temp-file creation side AND vulnerable to a
    pre-placed symlink at the predictable ``.tmp`` path redirecting the
    write to an arbitrary location. ``tempfile.mkstemp`` creates the file
    atomically with secure permissions; ``os.replace`` is the atomic
    rename.
    """
    path = _checkpoint_path(output_dir)
    parent = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp", prefix=".ckpt-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(state), f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _node_output_path(output_dir: str, node_name: str) -> str:
    """v0.53.7 M-F: per-node persisted-output sidecar (resume rehydration)."""
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", node_name)
    return os.path.join(output_dir, f".node-{safe}.jsonl")


def _save_node_output(
    output_dir: str, node_name: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Persist a node's row output for later resume rehydration."""
    parent = output_dir or "."
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp", prefix=".node-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                if isinstance(row, Mapping):
                    f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        os.replace(tmp, _node_output_path(output_dir, node_name))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_node_output(
    output_dir: str, node_name: str
) -> List[Mapping[str, Any]]:
    """Best-effort sidecar reload — empty list on missing / unparseable."""
    path = _node_output_path(output_dir, node_name)
    if not os.path.isfile(path):
        return []
    rows: List[Mapping[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, Mapping):
                    rows.append(obj)
    except OSError:
        return []
    return rows


def _redact_exc_message(exc: BaseException, limit: int = 256) -> str:
    """v0.53.7 H-D: strip path-like tokens from exception messages.

    Mirrors v0.34.0 ``crash.py`` redaction policy — file-system paths
    embedded in ``FileNotFoundError`` / ``OSError`` strings can leak the
    operator's HOME / cwd to caller-readable checkpoint state. We replace
    POSIX and Windows path runs with their basenames before persisting.
    """
    msg = f"{type(exc).__name__}: {exc}"
    # POSIX absolute paths (anything starting with ``/``).
    msg = re.sub(
        r"/[^\s:'\"]+",
        lambda m: os.path.basename(m.group(0)) or m.group(0),
        msg,
    )
    # Windows drive-letter paths (``C:\\foo\\bar`` or ``C:/foo/bar``).
    # Use a cross-platform basename: split on both ``/`` and ``\\`` so this
    # works on POSIX hosts too (``os.path.basename`` only splits on ``/``
    # on POSIX, leaving ``\\``-separated components intact).
    def _win_basename(m: "re.Match[str]") -> str:
        raw = m.group(0)
        # Split on either separator; last non-empty component is the basename.
        parts = [p for p in re.split(r"[/\\]", raw) if p]
        return parts[-1] if parts else raw

    msg = re.sub(r"[A-Za-z]:[\\/][^\s:'\"]+", _win_basename, msg)
    return msg[:limit]


# --- Node handlers ----------------------------------------------------------


def _node_seed(
    node: "RecipeNode",
    inputs: Sequence[Sequence[Mapping[str, Any]]],  # noqa: ARG001
) -> List[Mapping[str, Any]]:
    """Load a JSONL file from ``node.config.path``."""
    raw_path = node.config.get("path")
    if raw_path is None:
        raise ValueError(f"seed node {node.name!r}: missing 'path' in config")
    if not isinstance(raw_path, str):
        raise ValueError(f"seed node {node.name!r}: 'path' must be a string")
    _ensure_str(raw_path, name="seed.path")
    # v0.53.7 H-B: lstat the RAW path BEFORE realpath — realpath resolves
    # the symlink target so S_ISLNK on the result is always False
    # (matches v0.33.0 #22 TOCTOU policy).
    try:
        lst = os.lstat(raw_path)
    except OSError as exc:
        raise ValueError(
            f"seed node {node.name!r}: path not found: {raw_path!r}"
        ) from exc
    if _stat.S_ISLNK(lst.st_mode):
        raise ValueError(
            f"seed node {node.name!r}: path must not be a symlink"
        )
    real = os.path.realpath(raw_path)
    if not is_under_cwd(real):
        raise ValueError(
            f"seed node {node.name!r}: path must stay under cwd"
        )
    rows: List[Mapping[str, Any]] = []
    with open(real, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if len(rows) >= _MAX_NODE_ROWS:
                break
            try:
                obj = json.loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"seed node {node.name!r}: invalid JSON on line {line_no}"
                ) from exc
            if not isinstance(obj, Mapping):
                continue
            rows.append(obj)
    return rows


def _merge_inputs(
    inputs: Sequence[Sequence[Mapping[str, Any]]],
) -> List[Mapping[str, Any]]:
    merged: List[Mapping[str, Any]] = []
    for chunk in inputs:
        for row in chunk:
            if isinstance(row, Mapping):
                merged.append(row)
            if len(merged) >= _MAX_NODE_ROWS:
                return merged
    return merged


def _row_template(template: str, row: Mapping[str, Any]) -> str:
    """Render a simple ``{key}``-style template against ``row``.

    Missing keys fall back to an empty string (silent-degrade — matches the
    project policy on user-supplied callables). Uses ``str.format_map`` with
    a defaulting dict so partial coverage doesn't raise ``KeyError``.
    """

    class _DefaultDict(dict):
        def __missing__(self, key: str) -> str:  # type: ignore[override]
            return ""

    safe = _DefaultDict({k: str(v) for k, v in row.items()})
    try:
        return template.format_map(safe)
    except (IndexError, ValueError):
        return template


def _node_llm_text(
    node: "RecipeNode",
    inputs: Sequence[Sequence[Mapping[str, Any]]],
    *,
    judge_provider: Optional[str],
    judge_model: Optional[str],
) -> List[Mapping[str, Any]]:
    """Call an LLM provider with ``node.config.prompt.format(**row)``."""
    prompt_template = node.config.get("prompt")
    if not isinstance(prompt_template, str) or not prompt_template:
        raise ValueError(
            f"llm_text node {node.name!r}: 'prompt' must be a non-empty string"
        )
    if len(prompt_template) > _MAX_PROMPT_LEN:
        raise ValueError(
            f"llm_text node {node.name!r}: prompt exceeds {_MAX_PROMPT_LEN} chars"
        )
    if judge_provider is None:
        # Deterministic offline stub — useful for CI where no Ollama runs.
        rows = _merge_inputs(inputs)
        out: List[Mapping[str, Any]] = []
        for row in rows:
            rendered = _row_template(prompt_template, row)
            out.append({**row, node.name: f"llm_text(offline): {rendered[:80]}"})
        return out

    from soup_cli.utils.data_forge import make_judge_provider_fn

    judge_fn = make_judge_provider_fn(
        judge_provider, model=judge_model or "llama3.1"
    )
    rows = _merge_inputs(inputs)
    out_rows: List[Mapping[str, Any]] = []
    for row in rows:
        rendered = _row_template(prompt_template, row)
        try:
            reply = judge_fn(rendered)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("llm_text node %s judge raised: %s", node.name, exc)
            continue
        if not isinstance(reply, Mapping):
            continue
        text = reply.get("text") or ""
        if not isinstance(text, str):
            continue
        out_rows.append({**row, node.name: text})
    return out_rows


def _node_judge(
    node: "RecipeNode",
    inputs: Sequence[Sequence[Mapping[str, Any]]],
    *,
    judge_provider: Optional[str],
    judge_model: Optional[str],
) -> List[Mapping[str, Any]]:
    """Binary OK/REJECT classification via an LLM provider.

    Rows for which the model emits a string containing ``"OK"`` are kept
    (with ``<node.name>=True``); others are dropped.
    """
    prompt_template = node.config.get(
        "prompt",
        "Classify the following as OK or REJECT.\n\n{text}",
    )
    if not isinstance(prompt_template, str) or not prompt_template:
        raise ValueError(f"judge node {node.name!r}: 'prompt' must be a string")
    if len(prompt_template) > _MAX_PROMPT_LEN:
        raise ValueError(
            f"judge node {node.name!r}: prompt exceeds {_MAX_PROMPT_LEN} chars"
        )

    if judge_provider is None:
        rows = _merge_inputs(inputs)
        return [{**row, node.name: True} for row in rows]

    from soup_cli.utils.data_forge import make_judge_provider_fn

    judge_fn = make_judge_provider_fn(
        judge_provider, model=judge_model or "llama3.1"
    )
    rows = _merge_inputs(inputs)
    kept: List[Mapping[str, Any]] = []
    for row in rows:
        rendered = _row_template(prompt_template, row)
        try:
            reply = judge_fn(rendered)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("judge node %s raised: %s", node.name, exc)
            continue
        if not isinstance(reply, Mapping):
            continue
        text = (reply.get("text") or "").upper()
        if "OK" in text and "REJECT" not in text:
            kept.append({**row, node.name: True})
    return kept


def _node_code(
    node: "RecipeNode",
    inputs: Sequence[Sequence[Mapping[str, Any]]],
) -> List[Mapping[str, Any]]:
    """Execute ``node.config.code`` once per row via the RLVR sandbox.

    The row is injected as a JSON-encoded ``_row`` global; the code can
    ``print(json.dumps(...))`` to emit an output, which is parsed back into
    a dict + merged with the original row under ``<node.name>``.
    """
    code = node.config.get("code")
    if not isinstance(code, str) or not code:
        raise ValueError(
            f"code node {node.name!r}: 'code' must be a non-empty string"
        )
    if len(code) > _MAX_CODE_LEN:
        raise ValueError(
            f"code node {node.name!r}: code exceeds {_MAX_CODE_LEN} bytes"
        )

    from soup_cli.trainer.rewards import _run_code_sandbox

    rows = _merge_inputs(inputs)
    out_rows: List[Mapping[str, Any]] = []
    for row in rows:
        # v0.53.7 M-C: double-encode via json.dumps so a row with embedded
        # quotes / backslashes cannot break out of the Python string
        # literal that ``f"...{row_json!r}..."`` produced. ``json.dumps``
        # on a string always emits a valid Python (and JSON) string
        # literal — eliminating the repr-based injection surface.
        row_json = json.dumps(row, ensure_ascii=False)
        wrapped = (
            "import json\n"
            f"_row = json.loads({json.dumps(row_json)})\n"
            + code
        )
        try:
            stdout = _run_code_sandbox(wrapped)
        except Exception as exc:  # noqa: BLE001 — sandbox already swallows most
            _LOG.debug("code node %s sandbox raised: %s", node.name, exc)
            continue
        if stdout is None:
            continue
        try:
            parsed = json.loads(stdout)
        except (ValueError, TypeError):
            # Non-JSON output: keep as raw string under node.name.
            out_rows.append({**row, node.name: stdout})
            continue
        # v0.53.7 L-D: collapse — both branches produced the same record
        # (the isinstance check existed but did not change behaviour).
        out_rows.append({**row, node.name: parsed})
    return out_rows


def _node_validator(
    node: "RecipeNode",
    inputs: Sequence[Sequence[Mapping[str, Any]]],
) -> List[Mapping[str, Any]]:
    """Filter rows via regex match (``node.config.regex``) or JSON schema."""
    regex_src = node.config.get("regex")
    schema = node.config.get("schema")
    field = node.config.get("field", "text")

    if regex_src is None and schema is None:
        raise ValueError(
            f"validator node {node.name!r}: 'regex' or 'schema' required"
        )

    compiled = None
    if regex_src is not None:
        if not isinstance(regex_src, str):
            raise ValueError(
                f"validator node {node.name!r}: 'regex' must be a string"
            )
        if len(regex_src) > 2048:
            raise ValueError(
                f"validator node {node.name!r}: regex too int"
            )
        try:
            compiled = re.compile(regex_src)
        except re.error as exc:
            raise ValueError(
                f"validator node {node.name!r}: invalid regex: {exc}"
            ) from exc

    validator = None
    if schema is not None:
        if not isinstance(schema, Mapping):
            raise ValueError(
                f"validator node {node.name!r}: 'schema' must be a mapping"
            )
        try:
            import jsonschema  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "jsonschema is required for validator nodes with 'schema'. "
                "Run: pip install jsonschema"
            ) from exc
        try:
            validator_cls = jsonschema.validators.validator_for(dict(schema))
            validator = validator_cls(dict(schema))
        except Exception as exc:  # noqa: BLE001 — jsonschema error variety
            raise ValueError(
                f"validator node {node.name!r}: invalid JSON schema: {exc}"
            ) from exc

    rows = _merge_inputs(inputs)
    kept: List[Mapping[str, Any]] = []
    for row in rows:
        ok = True
        if compiled is not None:
            text = row.get(field, "")
            if not isinstance(text, str) or not compiled.search(text):
                ok = False
        if ok and validator is not None:
            try:
                validator.validate(dict(row))
            except Exception:  # noqa: BLE001 — schema validation error variety
                ok = False
        if ok:
            kept.append(row)
    return kept


def _node_sampler(
    node: "RecipeNode",
    inputs: Sequence[Sequence[Mapping[str, Any]]],
    *,
    output_dir: str,
) -> List[Mapping[str, Any]]:
    """Write merged inputs to ``<output_dir>/<node.name>.jsonl``.

    Atomic via tempfile + ``os.replace``. Cwd-contained by virtue of the
    pre-validated ``output_dir`` (caller checks containment before running).
    """
    rows = _merge_inputs(inputs)
    safe_name = re.sub(r"[^a-z0-9_\-]", "_", node.name.lower())
    out_path = os.path.join(output_dir, f"{safe_name}.jsonl")
    # M-E: mkstemp + os.replace for atomic write without symlink-TOCTOU
    # exposure at a predictable ``.tmp`` path.
    parent = output_dir or "."
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp", prefix=".sampler-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return rows


# --- Public runner ----------------------------------------------------------


def run_recipe(
    dag: "RecipeDAG",
    *,
    output_dir: str,
    checkpoint_dir: Optional[str] = None,
    resume: bool = False,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> Mapping[str, Any]:
    """Execute a validated :class:`RecipeDAG` end-to-end (v0.53.7 live).

    Per-node handlers dispatch on ``RecipeNode.kind`` against the closed
    ``NODE_KINDS`` allowlist. Topological order from ``dag.topo_order``.

    Args:
        dag: validated DAG (parse via ``parse_recipe`` / ``load_recipe_yaml``)
        output_dir: cwd-contained directory; sampler nodes write JSONL here
        checkpoint_dir: optional intermediate-node checkpoint dir (unused —
            checkpoint is always written under ``output_dir/.checkpoint.json``
            in v0.53.7; reserved for v0.53.8 multi-DAG runs).
        resume: when True, skip nodes whose checkpoint already reports
            ``status='completed'``.
        judge_provider: route ``llm_text`` / ``judge`` nodes through this
            v0.20.0 provider (``ollama`` / ``anthropic`` / ``vllm``). When
            ``None``, both kinds fall back to deterministic offline stubs.
        judge_model: model name forwarded to the judge provider.

    Returns:
        Mapping with ``status`` (``"completed"`` or ``"failed"``),
        ``completed_nodes`` (list of names), and ``node_row_counts``.

    Raises:
        TypeError: if ``dag`` is not a ``RecipeDAG``.
        ValueError: on bad output_dir, missing node config, unknown kind, or
            per-node validation failure.
    """
    # Late import so test code can patch the module without forcing a
    # heavy recipe_dag import at module load.
    from soup_cli.utils.recipe_dag import RecipeDAG

    if not isinstance(dag, RecipeDAG):
        raise TypeError("dag must be a RecipeDAG")
    if not isinstance(output_dir, str) or not output_dir:
        raise TypeError("output_dir must be a non-empty string")
    if checkpoint_dir is not None and not isinstance(checkpoint_dir, str):
        raise TypeError("checkpoint_dir must be a string or None")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    if judge_provider is not None and not isinstance(judge_provider, str):
        raise TypeError("judge_provider must be a string or None")
    if judge_model is not None and not isinstance(judge_model, str):
        raise TypeError("judge_model must be a string or None")

    if "\x00" in output_dir:
        raise ValueError("output_dir must not contain null bytes")
    real = os.path.realpath(output_dir)
    if not is_under_cwd(real):
        raise ValueError("output_dir must stay under cwd")
    os.makedirs(real, exist_ok=True)

    state = _load_checkpoint(real) if resume else {}
    state.setdefault("completed_nodes", [])
    state.setdefault("node_row_counts", {})
    state["status"] = "running"
    _save_checkpoint(real, state)

    node_by_name: Dict[str, "RecipeNode"] = {n.name: n for n in dag.nodes}
    predecessors: Dict[str, List[str]] = {n.name: [] for n in dag.nodes}
    for source, target in dag.edges:
        predecessors[target].append(source)

    # Persisted per-node outputs (for downstream predecessor lookup).
    outputs: Dict[str, List[Mapping[str, Any]]] = {}

    completed = set(state.get("completed_nodes", []))

    for name in dag.topo_order:
        if name in completed:
            # v0.53.7 M-F: rehydrate from the sidecar so downstream nodes
            # see the real predecessor rows on resume (the v0.53.6 stub
            # emitted ``[]``, which corrupted downstream consumers).
            outputs[name] = _load_node_output(real, name)
            continue

        node = node_by_name[name]
        pred_outputs = [outputs.get(p, []) for p in predecessors[name]]

        try:
            if node.kind == "seed":
                result = _node_seed(node, pred_outputs)
            elif node.kind == "llm_text":
                result = _node_llm_text(
                    node, pred_outputs,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                )
            elif node.kind == "code":
                result = _node_code(node, pred_outputs)
            elif node.kind == "judge":
                result = _node_judge(
                    node, pred_outputs,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                )
            elif node.kind == "validator":
                result = _node_validator(node, pred_outputs)
            elif node.kind == "sampler":
                result = _node_sampler(node, pred_outputs, output_dir=real)
            else:
                raise ValueError(
                    f"unknown node kind: {node.kind!r}"
                )  # pragma: no cover — gate by NODE_KINDS at parse
        except Exception as exc:
            state["status"] = "failed"
            state["failed_node"] = name
            # v0.53.7 H-D: redact path-like substrings from exception
            # messages so checkpoint state cannot leak absolute paths.
            state["failed_reason"] = _redact_exc_message(exc)
            _save_checkpoint(real, state)
            raise

        result_list = list(result)
        outputs[name] = result_list
        # v0.53.7 M-F: persist for resume rehydration. Best-effort — a
        # write failure does not abort the run (loud-fail at the
        # checkpoint write below is sufficient).
        try:
            _save_node_output(real, name, result_list)
        except OSError as exc:
            _LOG.debug("node output persist failed for %s: %s", name, exc)
        completed.add(name)
        state["completed_nodes"] = sorted(completed)
        state["node_row_counts"][name] = len(result_list)
        _save_checkpoint(real, state)

    state["status"] = "completed"
    _save_checkpoint(real, state)
    return {
        "status": "completed",
        "completed_nodes": tuple(state["completed_nodes"]),
        "node_row_counts": dict(state["node_row_counts"]),
        "output_dir": real,
    }


__all__ = ["run_recipe"]
