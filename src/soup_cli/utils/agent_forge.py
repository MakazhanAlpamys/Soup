"""v0.46.0 Part B — Agent Forge: spec → tool-calling SFT dataset.

Parses OpenAPI 3.x, MCP server manifests, and GraphQL introspection JSON
into a canonical ``Endpoint`` shape, then synthesises a tool-calling SFT
dataset where each row is ``{messages: [user, assistant{tool_calls}],
tool: <name>, source_endpoint: <path>}``.

The parser surface is intentionally parser-only — no network code, no
``$ref`` resolution that would let a crafted spec read arbitrary files.
``$ref`` strings are left as opaque markers and a warning is surfaced;
operators wanting full resolution should run ``openapi-spec-validator``
upstream and feed the bundled JSON in.

Live ``soup agent train`` orchestrator + ``soup agent eval`` sandbox
scoring re-use the v0.25.0 RLVR ``code_exec`` sandbox; this module ships
the parse-and-synth layer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from soup_cli.utils.paths import is_under_cwd

_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,127}$")
_MAX_ENDPOINTS = 10_000
_MAX_SPEC_BYTES = 5 * 1024 * 1024  # 5 MiB
_MAX_DESCRIPTION = 512
_MAX_ROWS_PER_ENDPOINT = 32
_ALLOWED_SPEC_KINDS = frozenset({"openapi", "mcp", "graphql"})
_HTTP_METHODS = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
)


@dataclass(frozen=True)
class Endpoint:
    """Canonical endpoint representation across all three spec kinds."""

    tool: str
    method: str
    path: str
    description: str
    parameters: Tuple[str, ...]  # parameter names only — schema details opaque
    spec_kind: str


@dataclass(frozen=True)
class SynthRow:
    """One row in the generated tool-calling SFT dataset."""

    messages: Tuple[Mapping[str, Any], ...]
    tool: str
    source_endpoint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [dict(m) for m in self.messages],
            "tool": self.tool,
            "source_endpoint": self.source_endpoint,
        }


@dataclass(frozen=True)
class SpecReport:
    """Summary of a parsed spec for the CLI."""

    spec_kind: str
    endpoint_count: int
    skipped: int
    warnings: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _validate_tool_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("tool name must be a string")
    if not _TOOL_NAME_RE.match(name):
        raise ValueError(
            "tool name must match ^[A-Za-z_][A-Za-z0-9_.-]{0,127}$"
        )
    return name


def _validate_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    canonical = method.strip().lower()
    if canonical not in _HTTP_METHODS:
        raise ValueError(f"unknown HTTP method: {method!r}")
    return canonical


def _validate_path(path: str) -> str:
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path or "\x00" in path or "\n" in path or "\r" in path:
        raise ValueError("path must be non-empty single-line NUL-free string")
    if len(path) > 1024:
        raise ValueError("path exceeds 1024 chars")
    return path


def _truncate_desc(desc: Any) -> str:
    if desc is None:
        return ""
    if not isinstance(desc, str):
        return ""
    if "\x00" in desc:
        desc = desc.replace("\x00", "")
    desc = desc.strip()
    if len(desc) > _MAX_DESCRIPTION:
        return desc[: _MAX_DESCRIPTION - 3] + "..."
    return desc


def _sanitise_tool_id(*parts: str) -> str:
    """Build a tool name from spec parts, replacing non-id chars with '_'."""
    raw = "_".join(p for p in parts if p)
    raw = re.sub(r"[^A-Za-z0-9_.\-]", "_", raw)
    if not raw or not re.match(r"^[A-Za-z_]", raw):
        raw = "tool_" + raw
    return raw[:128]


# ---------------------------------------------------------------------------
# OpenAPI parser
# ---------------------------------------------------------------------------


def parse_openapi(spec: Mapping[str, Any]) -> Tuple[List[Endpoint], List[str]]:
    """Parse an OpenAPI 3.x ``dict``. Returns (endpoints, warnings)."""
    if not isinstance(spec, dict):
        raise TypeError("openapi spec must be a dict")
    version = spec.get("openapi", "")
    warnings: List[str] = []
    if not isinstance(version, str) or not version.startswith("3."):
        warnings.append(
            f"unrecognised openapi version: {version!r}; parser is OpenAPI 3.x"
        )
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return [], ["spec has no 'paths' object"]
    endpoints: List[Endpoint] = []
    for path, ops in paths.items():
        if not isinstance(path, str) or not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if not isinstance(method, str):
                continue
            lower = method.lower()
            if lower not in _HTTP_METHODS:
                continue
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if not isinstance(op_id, str) or not op_id:
                op_id = _sanitise_tool_id(lower, path.strip("/"))
            try:
                tool = _validate_tool_name(_sanitise_tool_id(op_id))
                _validate_method(lower)
                _validate_path(path)
            except (TypeError, ValueError) as exc:
                warnings.append(f"skip {method.upper()} {path}: {exc}")
                continue
            params_raw = op.get("parameters")
            param_names: List[str] = []
            if isinstance(params_raw, list):
                for p in params_raw:
                    if not isinstance(p, dict):
                        continue
                    if "$ref" in p:
                        warnings.append("$ref left unresolved")
                        continue
                    name = p.get("name")
                    if isinstance(name, str) and name and "\x00" not in name:
                        param_names.append(name[:128])
            endpoints.append(
                Endpoint(
                    tool=tool,
                    method=lower,
                    path=path,
                    description=_truncate_desc(
                        op.get("summary") or op.get("description")
                    ),
                    parameters=tuple(param_names),
                    spec_kind="openapi",
                )
            )
            if len(endpoints) >= _MAX_ENDPOINTS:
                warnings.append(
                    f"endpoint cap {_MAX_ENDPOINTS} reached; truncating"
                )
                return endpoints, warnings
    return endpoints, warnings


# ---------------------------------------------------------------------------
# MCP manifest parser
# ---------------------------------------------------------------------------


def parse_mcp(manifest: Mapping[str, Any]) -> Tuple[List[Endpoint], List[str]]:
    """Parse an MCP server manifest's ``tools`` array."""
    if not isinstance(manifest, dict):
        raise TypeError("mcp manifest must be a dict")
    warnings: List[str] = []
    tools_raw = manifest.get("tools")
    if not isinstance(tools_raw, list):
        return [], ["mcp manifest missing 'tools' array"]
    endpoints: List[Endpoint] = []
    for entry in tools_raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            warnings.append("mcp tool missing name; skipped")
            continue
        try:
            tool = _validate_tool_name(_sanitise_tool_id(name))
        except (TypeError, ValueError) as exc:
            warnings.append(f"skip mcp tool {name!r}: {exc}")
            continue
        params: List[str] = []
        input_schema = entry.get("inputSchema")
        if isinstance(input_schema, dict):
            props = input_schema.get("properties")
            if isinstance(props, dict):
                for key in props:
                    if isinstance(key, str) and key and "\x00" not in key:
                        params.append(key[:128])
        try:
            mcp_path = _validate_path(f"mcp://{name}")
        except (TypeError, ValueError) as exc:
            warnings.append(f"skip mcp tool {name!r}: invalid path: {exc}")
            continue
        endpoints.append(
            Endpoint(
                tool=tool,
                method="invoke",
                path=mcp_path,
                description=_truncate_desc(entry.get("description")),
                parameters=tuple(params),
                spec_kind="mcp",
            )
        )
        if len(endpoints) >= _MAX_ENDPOINTS:
            warnings.append(f"endpoint cap {_MAX_ENDPOINTS} reached")
            return endpoints, warnings
    return endpoints, warnings


# ---------------------------------------------------------------------------
# GraphQL introspection parser
# ---------------------------------------------------------------------------


def parse_graphql(intro: Mapping[str, Any]) -> Tuple[List[Endpoint], List[str]]:
    """Parse a GraphQL introspection ``dict`` (``__schema`` envelope OK).

    Treats every Query / Mutation field as a tool-calling endpoint.
    """
    if not isinstance(intro, dict):
        raise TypeError("graphql introspection must be a dict")
    warnings: List[str] = []
    data = intro.get("data") if "data" in intro else intro
    if not isinstance(data, dict):
        return [], ["graphql introspection: no 'data' or schema dict"]
    schema = data.get("__schema") if "__schema" in data else data
    if not isinstance(schema, dict):
        return [], ["graphql introspection: no '__schema' field"]
    type_list = schema.get("types")
    if not isinstance(type_list, list):
        return [], ["graphql introspection: missing 'types'"]
    query_type = schema.get("queryType") or {}
    mutation_type = schema.get("mutationType") or {}
    q_name = query_type.get("name") if isinstance(query_type, dict) else None
    m_name = mutation_type.get("name") if isinstance(mutation_type, dict) else None
    endpoints: List[Endpoint] = []
    for t in type_list:
        if not isinstance(t, dict):
            continue
        type_name = t.get("name")
        if type_name not in (q_name, m_name):
            continue
        fields = t.get("fields")
        if not isinstance(fields, list):
            continue
        method = "query" if type_name == q_name else "mutation"
        for f in fields:
            if not isinstance(f, dict):
                continue
            fname = f.get("name")
            if not isinstance(fname, str) or not fname:
                continue
            try:
                tool = _validate_tool_name(_sanitise_tool_id(method, fname))
            except (TypeError, ValueError) as exc:
                warnings.append(f"skip {fname!r}: {exc}")
                continue
            args = f.get("args") or []
            arg_names: List[str] = []
            if isinstance(args, list):
                for a in args:
                    if isinstance(a, dict):
                        an = a.get("name")
                        if isinstance(an, str) and an and "\x00" not in an:
                            arg_names.append(an[:128])
            try:
                gql_path = _validate_path(f"graphql://{fname}")
            except (TypeError, ValueError) as exc:
                warnings.append(f"skip {fname!r}: invalid path: {exc}")
                continue
            endpoints.append(
                Endpoint(
                    tool=tool,
                    method=method,
                    path=gql_path,
                    description=_truncate_desc(f.get("description")),
                    parameters=tuple(arg_names),
                    spec_kind="graphql",
                )
            )
            if len(endpoints) >= _MAX_ENDPOINTS:
                warnings.append(f"endpoint cap {_MAX_ENDPOINTS} reached")
                return endpoints, warnings
    return endpoints, warnings


# ---------------------------------------------------------------------------
# Dispatcher + dataset writer
# ---------------------------------------------------------------------------


def detect_spec_kind(spec: Mapping[str, Any]) -> str:
    """Best-effort detection of spec kind from a parsed dict.

    Returns one of {"openapi", "mcp", "graphql"} or raises ``ValueError``.
    """
    if not isinstance(spec, dict):
        raise TypeError("spec must be a dict")
    if isinstance(spec.get("openapi"), str) and isinstance(spec.get("paths"), dict):
        return "openapi"
    if isinstance(spec.get("tools"), list) and not isinstance(
        spec.get("paths"), dict
    ):
        return "mcp"
    if "__schema" in spec or (
        isinstance(spec.get("data"), dict) and "__schema" in spec["data"]
    ):
        return "graphql"
    raise ValueError(
        "cannot detect spec kind — must be OpenAPI 3.x / MCP / GraphQL"
    )


def parse_spec(
    spec: Mapping[str, Any], kind: Optional[str] = None
) -> Tuple[List[Endpoint], SpecReport]:
    """Parse a spec dict with optional explicit ``kind`` override."""
    if kind is None:
        resolved = detect_spec_kind(spec)
    else:
        if not isinstance(kind, str):
            raise TypeError("kind must be a string")
        resolved = kind.strip().lower()
        if resolved not in _ALLOWED_SPEC_KINDS:
            raise ValueError(f"unknown spec kind: {kind!r}")
    if resolved == "openapi":
        endpoints, warnings = parse_openapi(spec)
    elif resolved == "mcp":
        endpoints, warnings = parse_mcp(spec)
    else:
        endpoints, warnings = parse_graphql(spec)
    # Dedup by tool name (last write wins is unsafe — first wins, preserves order).
    seen: Dict[str, Endpoint] = {}
    skipped = 0
    for ep in endpoints:
        if ep.tool in seen:
            skipped += 1
            continue
        seen[ep.tool] = ep
    report = SpecReport(
        spec_kind=resolved,
        endpoint_count=len(seen),
        skipped=skipped,
        warnings=tuple(warnings),
    )
    return list(seen.values()), report


def endpoint_to_rows(
    endpoint: Endpoint, examples_per_endpoint: int = 1
) -> List[SynthRow]:
    """Synthesise ``examples_per_endpoint`` rows for one endpoint.

    A row is one user-question / assistant-tool-call pair. We do NOT make
    network calls here; the assistant content embeds an empty arguments
    object that the trainer is meant to learn to fill from the user query.
    """
    if not isinstance(endpoint, Endpoint):
        raise TypeError("endpoint must be an Endpoint")
    if isinstance(examples_per_endpoint, bool) or not isinstance(
        examples_per_endpoint, int
    ):
        raise TypeError("examples_per_endpoint must be int (not bool)")
    if not (1 <= examples_per_endpoint <= _MAX_ROWS_PER_ENDPOINT):
        raise ValueError(
            f"examples_per_endpoint must be in [1, {_MAX_ROWS_PER_ENDPOINT}]"
        )
    desc = endpoint.description or f"Call {endpoint.tool}"
    user_templates = [
        f"Please {desc}.",
        f"How do I use {endpoint.tool}?",
        f"Run the {endpoint.tool} action with sensible defaults.",
    ]
    rows: List[SynthRow] = []
    for i in range(examples_per_endpoint):
        user_msg = user_templates[i % len(user_templates)]
        tool_args: Dict[str, str] = {p: "<value>" for p in endpoint.parameters}
        assistant_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": endpoint.tool,
                        "arguments": json.dumps(tool_args, sort_keys=True),
                    },
                }
            ],
        }
        rows.append(
            SynthRow(
                messages=(
                    {"role": "user", "content": user_msg},
                    assistant_msg,
                ),
                tool=endpoint.tool,
                source_endpoint=endpoint.path,
            )
        )
    return rows


def synthesise_dataset(
    endpoints: Sequence[Endpoint], examples_per_endpoint: int = 1
) -> List[SynthRow]:
    """Synthesise a flat list of training rows from a list of endpoints."""
    if isinstance(endpoints, (str, bytes)):
        raise TypeError("endpoints must be a sequence of Endpoint")
    out: List[SynthRow] = []
    for ep in endpoints:
        out.extend(endpoint_to_rows(ep, examples_per_endpoint))
    return out


def load_spec_file(spec_path: str) -> Mapping[str, Any]:
    """Load a YAML/JSON spec from disk with cwd containment + size cap.

    Symlinks are rejected (TOCTOU defence, mirrors v0.45.0 Part E policy).
    """
    if not isinstance(spec_path, str):
        raise TypeError("spec_path must be a string")
    if not spec_path or "\x00" in spec_path:
        raise ValueError("spec_path must be non-empty NUL-free string")
    if not is_under_cwd(spec_path):
        raise ValueError(
            f"spec_path must stay under cwd: {os.path.basename(spec_path)}"
        )
    # lstat BEFORE realpath: project-standard TOCTOU policy (v0.33.0 #22 /
    # v0.43.0 Part C / v0.44.0 Part B). The lstat must operate on the
    # original (pre-realpath) path so we see the symlink, not its target.
    import stat as _stat

    try:
        st = os.lstat(spec_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(spec_path) from exc
    if _stat.S_ISLNK(st.st_mode):
        raise ValueError(
            f"spec_path must not be a symlink: {os.path.basename(spec_path)}"
        )
    real = os.path.realpath(spec_path)
    if not os.path.isfile(real):
        raise FileNotFoundError(spec_path)
    size = os.path.getsize(real)
    if size > _MAX_SPEC_BYTES:
        raise ValueError(f"spec file exceeds {_MAX_SPEC_BYTES} bytes ({size})")
    with open(real, "r", encoding="utf-8") as fh:
        text = fh.read()
    if spec_path.lower().endswith((".yaml", ".yml")):
        import yaml

        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("spec file root must be a JSON/YAML object")
    return loaded


def write_dataset(rows: Sequence[SynthRow], output_path: str) -> str:
    """Write rows as JSONL under cwd; returns realpath written.

    Atomic via staged-tempfile + ``os.replace`` (matches v0.43.0 Part D
    ``copy_bundle_to`` policy). Validates every row BEFORE any bytes hit
    the target path — a mid-stream ``TypeError`` never leaves a partial
    file. Symlink at the target rejected via ``os.lstat`` (TOCTOU).
    """
    import stat as _stat
    import tempfile

    if not isinstance(output_path, str):
        raise TypeError("output_path must be a string")
    if not output_path or "\x00" in output_path:
        raise ValueError("output_path must be non-empty NUL-free string")
    if not is_under_cwd(output_path):
        raise ValueError(
            f"output_path must stay under cwd: {os.path.basename(output_path)}"
        )
    # Reject a pre-placed symlink at the target — defends against
    # `<output>.jsonl -> /etc/cron.d/x` overwrite.
    try:
        st = os.lstat(output_path)
        if _stat.S_ISLNK(st.st_mode):
            raise ValueError(
                f"output_path must not be a symlink: "
                f"{os.path.basename(output_path)}"
            )
    except FileNotFoundError:
        pass
    real = os.path.realpath(output_path)
    parent = os.path.dirname(real) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".agent_forge_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                if not isinstance(row, SynthRow):
                    raise TypeError("rows must all be SynthRow")
                fh.write(json.dumps(row.to_dict(), sort_keys=True))
                fh.write("\n")
        os.replace(tmp, real)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return real


__all__ = [
    "Endpoint",
    "SynthRow",
    "SpecReport",
    "parse_openapi",
    "parse_mcp",
    "parse_graphql",
    "detect_spec_kind",
    "parse_spec",
    "endpoint_to_rows",
    "synthesise_dataset",
    "load_spec_file",
    "write_dataset",
]
