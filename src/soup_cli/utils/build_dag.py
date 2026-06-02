"""v0.69.0 Part A — `soup build` (dbt-for-SFT DAG).

Parses a YAML manifest describing a DAG of dataset transforms (``ref``-connected
models with ``incremental`` / ``table`` / ``view`` materialization). Validates
topology via Kahn's algorithm and renders the dry-run plan. The live SQL/Python
runner (re-tokenize-only-diff rows materialization) is deferred to v0.69.1.

Mirrors the shape of v0.45.0 Part E `recipe_dag.py` so operators familiar with
the recipe DAG surface have one mental model. Differs in that build-DAG models
also carry a *transform* identifier and a *source* path, and the DAG edges are
derived from each model's ``refs: [<other-model>]`` field (rather than a
top-level ``edges`` list) — matching dbt's mental model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

# Closed allowlist of model kinds. Mirrors dbt's `materialized` field but
# trimmed to the three that map onto SFT-data pipelines.
_SUPPORTED_MODEL_KINDS = ("incremental", "table", "view")
SUPPORTED_MODEL_KINDS: frozenset = frozenset(_SUPPORTED_MODEL_KINDS)

# Per-build-plan caps — defence-in-depth against pathological YAML.
_MAX_MODELS = 256
_MAX_REFS_PER_MODEL = 32
_MAX_NAME_LEN = 128
_MAX_KIND_LEN = 64
_MAX_TRANSFORM_LEN = 256
_MAX_SOURCE_LEN = 4096
_MAX_FILE_BYTES = 1_048_576  # 1 MiB

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._\-]{0,127}$")


@dataclass(frozen=True)
class BuildModel:
    """One node in the build DAG.

    ``transform`` identifies the materialization function (e.g. ``identity``,
    ``filter_low_quality``, ``tokenize``) the live runner will resolve. The
    schema does not validate the function reference — the runner does.

    Field semantics:

    - ``refs=()`` + ``source`` set → *seed* model (reads from disk).
    - ``refs=(...)`` + ``source=None`` → *derived* model (consumes upstream models).
    - ``refs=()`` + ``source=None`` → degenerate; rejected by cross-validator.
    - ``refs=(...)`` + ``source`` set → ambiguous; rejected by cross-validator.

    The ``source`` path is validated for shape (non-empty, null-byte-free,
    length-capped) at schema-load. Operators wanting cwd containment on
    source paths get it via the v0.69.1 live runner — the schema permits
    relative paths so a build can be planned offline before the data lands.
    """

    name: str
    kind: str
    transform: str
    refs: Tuple[str, ...]
    source: Optional[str]
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Re-validate so callers bypassing ``parse_build_plan`` cannot smuggle
        # an inconsistent BuildModel through direct construction.
        validate_model_name(self.name)
        validate_model_kind(self.kind)
        if not isinstance(self.refs, tuple):
            raise TypeError("BuildModel.refs must be a tuple, not list/sequence")
        if isinstance(self.transform, bool) or not isinstance(self.transform, str):
            raise TypeError("BuildModel.transform must be a string")
        if "\x00" in self.transform:
            raise ValueError("BuildModel.transform must not contain null bytes")
        if len(self.transform) > _MAX_TRANSFORM_LEN:
            raise ValueError(
                f"BuildModel.transform must be <= {_MAX_TRANSFORM_LEN} chars"
            )
        if self.source is not None:
            if isinstance(self.source, bool) or not isinstance(self.source, str):
                raise TypeError("BuildModel.source must be a string or None")
            if "\x00" in self.source:
                raise ValueError("BuildModel.source must not contain null bytes")
            if len(self.source) > _MAX_SOURCE_LEN:
                raise ValueError(
                    f"BuildModel.source must be <= {_MAX_SOURCE_LEN} chars"
                )
        # Freeze config for parity with the parse path (direct construction
        # otherwise leaves a mutable dict on a frozen dataclass).
        if not isinstance(self.config, Mapping):
            raise TypeError("BuildModel.config must be a mapping")
        if not isinstance(self.config, MappingProxyType):
            object.__setattr__(
                self, "config", MappingProxyType(dict(self.config))
            )
        # Cross-validator: seed/derived shape must be unambiguous.
        if not self.refs and self.source is None:
            raise ValueError(
                f"BuildModel {self.name!r}: a model with no refs must declare a "
                "'source' (or add refs to make it derived)"
            )
        if self.refs and self.source is not None:
            raise ValueError(
                f"BuildModel {self.name!r}: a model with refs cannot also declare "
                "'source' (refs and source are mutually exclusive)"
            )


@dataclass(frozen=True)
class BuildPlan:
    """Validated build DAG with topologically sorted models."""

    models: Tuple[BuildModel, ...]
    topo_order: Tuple[str, ...]


@dataclass(frozen=True)
class IncrementalDiffReport:
    """Outcome of comparing previous + new rows for an incremental model.

    ``added``: rows present in ``new`` but not ``prev``.
    ``changed``: rows whose ``id`` exists in both but whose content hash differs.
    ``removed``: rows present in ``prev`` but absent in ``new``.
    ``unchanged``: rows with identical ``id`` and content hash in both.
    """

    added: int
    changed: int
    removed: int
    unchanged: int


@dataclass(frozen=True)
class ModelBuildResult:
    """Per-model outcome of a live build run.

    ``transform_calls`` is the number of times the transform was actually
    invoked — for ``incremental`` models this equals ``added + changed`` (the
    re-tokenize-only-diff optimisation; unchanged rows are carried over from
    the SQLite state store without re-running the transform). ``output_path``
    is the realpath of the materialised JSONL, or ``None`` for ``view`` models
    (in-memory only). ``diff`` is set for ``incremental`` models.
    """

    name: str
    kind: str
    rows_in: int
    rows_out: int
    transform_calls: int
    output_path: Optional[str]
    diff: Optional[IncrementalDiffReport]


@dataclass(frozen=True)
class BuildResult:
    """Outcome of a whole ``run_build`` invocation."""

    models: Tuple[ModelBuildResult, ...]
    output_dir: str


# -----------------------------------------------------------------------------
# Validators
# -----------------------------------------------------------------------------


def validate_model_kind(kind: object) -> str:
    """Return the canonical lower-case kind. Raise ValueError on unknown."""
    if isinstance(kind, bool) or not isinstance(kind, str):
        raise TypeError(
            f"model kind must be str, got {type(kind).__name__}"
        )
    if not kind:
        raise ValueError("model kind must be non-empty")
    if "\x00" in kind:
        raise ValueError("model kind must not contain null bytes")
    if len(kind) > _MAX_KIND_LEN:
        raise ValueError(f"model kind must be <= {_MAX_KIND_LEN} chars")
    canonical = kind.strip().lower()
    if canonical not in SUPPORTED_MODEL_KINDS:
        raise ValueError(
            f"unknown model kind: {kind!r}. supported: {sorted(SUPPORTED_MODEL_KINDS)}"
        )
    return canonical


def validate_model_name(name: object) -> str:
    """Validate the model identifier. Returns the canonical name."""
    if isinstance(name, bool) or not isinstance(name, str):
        raise TypeError(
            f"model name must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("model name must be non-empty")
    if "\x00" in name:
        raise ValueError("model name must not contain null bytes")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"model name must be <= {_MAX_NAME_LEN} chars")
    if not _NAME_RE.match(name):
        raise ValueError(
            f"model name must match {_NAME_RE.pattern}: {name!r}"
        )
    return name


def validate_build_source(source: Optional[str]) -> Optional[str]:
    """Cwd-containment + symlink-rejection boundary for a model's source path.

    This is the security boundary the v0.69.1 live ``run_build`` runner MUST
    call before opening any ``BuildModel.source`` file. ``BuildModel.__post_init__``
    only validates *shape* (non-empty / null-byte-free / length-capped) so a
    build can be *planned* offline before the data lands on disk and from any
    cwd; this helper enforces the runtime containment policy at read time.

    - ``source=None`` (derived models with no source) returns ``None``.
    - Otherwise delegates to ``utils.paths.enforce_under_cwd_and_no_symlink``
      (v0.59.0 shared TOCTOU helper — project code-review CRIT centralisation)
      and returns the validated path on success.

    Raises ``TypeError`` on non-string / non-None input and ``ValueError`` for
    empty / null-byte / outside-cwd / symlink-target paths.
    """
    if source is None:
        return None
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(source, "model source")
    return source


# -----------------------------------------------------------------------------
# Topological sort (Kahn's)
# -----------------------------------------------------------------------------


def _topological_sort(
    names: Sequence[str],
    edges: Sequence[Tuple[str, str]],
) -> List[str]:
    """Kahn's algorithm. Same shape as v0.45.0 Part E ``recipe_dag``."""
    in_degree = {name: 0 for name in names}
    successors: dict = {name: [] for name in names}
    for source, target in edges:
        in_degree[target] += 1
        successors[source].append(target)
    queue: deque = deque(
        sorted(name for name, deg in in_degree.items() if deg == 0)
    )
    order: List[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        ready: List[str] = []
        for successor in successors[current]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                ready.append(successor)
        ready.sort()
        queue.extend(ready)
    if len(order) != len(names):
        raise ValueError("build DAG contains a cycle")
    return order


# -----------------------------------------------------------------------------
# Plan parsing
# -----------------------------------------------------------------------------


def parse_build_plan(raw: Any) -> BuildPlan:
    """Validate a ``{"models": [...]}`` dict and return a sorted ``BuildPlan``.

    Each model is a dict with required ``name`` / ``kind`` / ``transform``,
    optional ``refs`` (list of upstream model names), optional ``source``
    (input file path for seed models), and optional ``config`` (free-form dict).
    """
    if not isinstance(raw, dict):
        raise TypeError("build plan must be a dict")
    raw_models = raw.get("models")
    if raw_models is None:
        raise ValueError("build plan must define a 'models' key")
    if not isinstance(raw_models, list):
        raise ValueError("build plan 'models' must be a list")
    if not raw_models:
        raise ValueError("build plan 'models' must be a non-empty list")
    if len(raw_models) > _MAX_MODELS:
        raise ValueError(
            f"build plan 'models' exceeds {_MAX_MODELS} entries"
        )

    models: List[BuildModel] = []
    seen_names: set = set()
    edges: List[Tuple[str, str]] = []
    name_set: set = set()

    # First pass: validate every model + collect names.
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise TypeError(f"build plan models[{index}] must be a dict")
        name = validate_model_name(raw_model.get("name", ""))
        if name in seen_names:
            raise ValueError(f"duplicate model name: {name!r}")
        seen_names.add(name)
        kind = validate_model_kind(raw_model.get("kind", ""))
        transform = raw_model.get("transform")
        if not isinstance(transform, str) or isinstance(transform, bool):
            raise TypeError(
                f"models[{index}].transform must be a string"
            )
        raw_refs = raw_model.get("refs", [])
        if not isinstance(raw_refs, list):
            raise TypeError(f"models[{index}].refs must be a list")
        if len(raw_refs) > _MAX_REFS_PER_MODEL:
            raise ValueError(
                f"models[{index}].refs exceeds {_MAX_REFS_PER_MODEL} entries"
            )
        refs_seen: set = set()
        normalised_refs: List[str] = []
        for ref_index, ref in enumerate(raw_refs):
            ref_name = validate_model_name(ref)
            if ref_name in refs_seen:
                raise ValueError(
                    f"duplicate ref in models[{index}].refs: {ref_name!r}"
                )
            refs_seen.add(ref_name)
            normalised_refs.append(ref_name)
        source = raw_model.get("source")
        config = raw_model.get("config", {})
        if not isinstance(config, dict):
            raise TypeError(f"models[{index}].config must be a dict")
        models.append(
            BuildModel(
                name=name,
                kind=kind,
                transform=transform,
                refs=tuple(normalised_refs),
                source=source,
                config=MappingProxyType(dict(config)),
            )
        )
        name_set.add(name)

    # Second pass: validate refs reference real models + build edges.
    for model in models:
        for ref in model.refs:
            if ref == model.name:
                raise ValueError(
                    f"self-loop edge rejected: {model.name!r}"
                )
            if ref not in name_set:
                raise ValueError(
                    f"model {model.name!r} refs missing model: {ref!r}"
                )
            # Edge direction: upstream -> downstream (ref produces model)
            edges.append((ref, model.name))

    topo = _topological_sort([m.name for m in models], edges)
    return BuildPlan(models=tuple(models), topo_order=tuple(topo))


def parse_build_yaml(text: object) -> BuildPlan:
    """Parse a YAML string into a validated ``BuildPlan``."""
    if not isinstance(text, str):
        raise TypeError("build text must be a string")
    if "\x00" in text:
        raise ValueError("build text must not contain null bytes")
    if len(text.encode("utf-8")) > _MAX_FILE_BYTES:
        raise ValueError(f"build text exceeds {_MAX_FILE_BYTES} bytes")
    import yaml  # lazy — keep CLI startup fast

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    return parse_build_plan(data)


def load_build_yaml(path: object) -> BuildPlan:
    """Load a build manifest from a path that must live under cwd.

    Delegates to ``utils.paths.enforce_under_cwd_and_no_symlink`` (v0.59.0
    shared TOCTOU helper) so this module does not reinvent the policy
    (project code-review CRIT — centralisation enforcement).
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    if isinstance(path, bool) or not isinstance(path, str):
        raise TypeError("path must be a string")
    # Shared helper validates emptiness + null-bytes + cwd + symlink upfront.
    enforce_under_cwd_and_no_symlink(path, "build path")
    if not os.path.lexists(path):
        raise FileNotFoundError(path)
    real = os.path.realpath(path)
    if not os.path.isfile(real):
        raise FileNotFoundError(real)
    if os.path.getsize(real) > _MAX_FILE_BYTES:
        raise ValueError(f"build file exceeds {_MAX_FILE_BYTES} bytes")
    with open(real, "r", encoding="utf-8") as handle:
        return parse_build_yaml(handle.read())


# -----------------------------------------------------------------------------
# Incremental diff — re-tokenize only changed rows
# -----------------------------------------------------------------------------


def compute_row_hash(row: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hex digest of a row's content.

    Keys are sorted so the hash is insertion-order independent. The hash
    EXCLUDES the ``id`` field so identity-vs-content can be reasoned about
    separately (same id + changed content = diff row).
    """
    if not isinstance(row, Mapping):
        raise TypeError(
            f"row must be a Mapping, got {type(row).__name__}"
        )
    payload = {k: row[k] for k in sorted(row.keys()) if k != "id"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def incremental_diff(
    prev: Sequence[Mapping[str, Any]],
    new: Sequence[Mapping[str, Any]],
) -> IncrementalDiffReport:
    """Compare two row sequences keyed on ``id`` and return per-bucket counts.

    Every row in both sequences MUST carry an ``id`` field — the function
    raises ``ValueError`` otherwise so silent mis-counting cannot happen.
    """
    prev_map: dict = {}
    for index, row in enumerate(prev):
        if not isinstance(row, Mapping):
            raise TypeError(f"prev[{index}] must be a Mapping")
        if "id" not in row:
            raise ValueError(
                f"prev[{index}] missing required 'id' field for incremental diff"
            )
        prev_map[row["id"]] = compute_row_hash(row)

    new_map: dict = {}
    for index, row in enumerate(new):
        if not isinstance(row, Mapping):
            raise TypeError(f"new[{index}] must be a Mapping")
        if "id" not in row:
            raise ValueError(
                f"new[{index}] missing required 'id' field for incremental diff"
            )
        new_map[row["id"]] = compute_row_hash(row)

    added = 0
    changed = 0
    unchanged = 0
    for row_id, new_hash in new_map.items():
        if row_id not in prev_map:
            added += 1
        elif prev_map[row_id] != new_hash:
            changed += 1
        else:
            unchanged += 1
    removed = sum(1 for row_id in prev_map if row_id not in new_map)
    return IncrementalDiffReport(
        added=added,
        changed=changed,
        removed=removed,
        unchanged=unchanged,
    )


# -----------------------------------------------------------------------------
# Plan rendering
# -----------------------------------------------------------------------------


def render_plan_table(plan: BuildPlan) -> str:
    """Render a plan as plain text in topological order (for dry-run output)."""
    if not isinstance(plan, BuildPlan):
        raise TypeError("plan must be a BuildPlan")
    lookup = {m.name: m for m in plan.models}
    lines = ["Build plan (topological order):"]
    for name in plan.topo_order:
        model = lookup[name]
        ref_str = ", ".join(model.refs) if model.refs else "(no refs)"
        lines.append(
            f"  {model.name} [{model.kind}] transform={model.transform} refs=[{ref_str}]"
        )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Transform registry (v0.71.6 #231) — row-wise (row, config) -> dict | None
# -----------------------------------------------------------------------------
#
# A transform receives one row plus the model's ``config`` mapping and returns
# a NEW row dict, or ``None`` to drop the row. Row-wise (not batch) so the
# incremental backend can re-run the transform on only the added/changed rows
# and carry the rest over from the SQLite state store.

TransformFn = Callable[[Mapping[str, Any], Mapping[str, Any]], Optional[Mapping[str, Any]]]

# DoS / sanity caps for the live runner.
_MAX_BUILD_ROWS = 1_000_000
_MAX_SOURCE_BYTES = 1024 * 1024 * 1024  # 1 GiB
_STATE_DB_NAME = ".soup_build_state.sqlite"


def _t_identity(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    return dict(row)


def _t_drop_empty(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> Optional[dict]:
    field = config.get("field", "text")
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return dict(row)
    return None


def _t_lowercase(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    field = config.get("field", "text")
    out = dict(row)
    value = out.get(field)
    if isinstance(value, str):
        out[field] = value.lower()
    return out


def _t_add_field(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    field = config.get("field")
    if not isinstance(field, str) or not field:
        raise ValueError("add_field transform requires config.field (non-empty string)")
    out = dict(row)
    out[field] = config.get("value")
    return out


def _t_token_count(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    field = config.get("field", "text")
    out = dict(row)
    value = out.get(field)
    out["n_tokens"] = len(str(value).split()) if value is not None else 0
    return out


# Closed built-in registry, wrapped immutable (matches project MappingProxyType
# policy for closed allowlists). Operators extend per-run via ``transforms=``.
BUILTIN_TRANSFORMS: Mapping[str, TransformFn] = MappingProxyType(
    {
        "identity": _t_identity,
        "drop_empty": _t_drop_empty,
        "lowercase": _t_lowercase,
        "add_field": _t_add_field,
        "token_count": _t_token_count,
    }
)


def resolve_transform(
    name: str,
    extra: Optional[Mapping[str, TransformFn]] = None,
) -> TransformFn:
    """Resolve a transform name to a callable.

    Per-call ``extra`` (operator-supplied for one ``run_build`` invocation)
    shadows the built-ins. No global mutable registry — that would leak state
    across runs/tests (project prefers immutable registries).
    """
    if extra and name in extra:
        fn = extra[name]
        if not callable(fn):
            raise TypeError(f"transform {name!r} must be callable")
        return fn
    if name in BUILTIN_TRANSFORMS:
        return BUILTIN_TRANSFORMS[name]
    raise ValueError(
        f"unknown transform: {name!r}. built-ins: {sorted(BUILTIN_TRANSFORMS)}; "
        "supply a custom one via transforms={name: fn}."
    )


# -----------------------------------------------------------------------------
# Live runner (v0.71.6 #231)
# -----------------------------------------------------------------------------


def _assign_ids(rows: Sequence[Mapping[str, Any]]) -> List[dict]:
    """Return dict-copies with a stable ``id`` on every row.

    Rows that already carry a non-None ``id`` keep it; id-less rows get an
    index-based surrogate so the incremental backend can diff them. Operators
    wanting stable incremental diffs across row-reorderings should provide
    their own ``id`` field (index surrogates shift on reorder — documented).
    """
    out: List[dict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a mapping")
        copy = dict(row)
        if copy.get("id") is None:
            copy["id"] = f"row-{index}"
        out.append(copy)
    return out


def _read_seed_rows(source: str) -> List[dict]:
    """Read a seed model's source JSONL under cwd (containment + symlink reject)."""
    validate_build_source(source)  # cwd-containment + symlink rejection
    if not os.path.lexists(source):
        raise FileNotFoundError(f"build source not found: {source!r}")
    real = os.path.realpath(source)
    if not os.path.isfile(real):
        raise FileNotFoundError(f"build source is not a file: {source!r}")
    if os.path.getsize(real) > _MAX_SOURCE_BYTES:
        raise ValueError(f"build source {source!r} exceeds {_MAX_SOURCE_BYTES} bytes")
    rows: List[dict] = []
    with open(real, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"build source {source!r} line {lineno}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"build source {source!r} line {lineno}: row must be a JSON object"
                )
            rows.append(obj)
            if len(rows) > _MAX_BUILD_ROWS:
                raise ValueError(
                    f"build source {source!r} exceeds {_MAX_BUILD_ROWS} rows"
                )
    return rows


def _gather_inputs(
    model: BuildModel,
    materialized: Mapping[str, List[dict]],
) -> List[dict]:
    if model.source is not None:
        return _assign_ids(_read_seed_rows(model.source))
    gathered: List[dict] = []
    for ref in model.refs:
        gathered.extend(dict(r) for r in materialized.get(ref, []))
    return gathered


def _model_fingerprint(model: BuildModel) -> str:
    """Stable hash of a model's transform + config.

    Folded into the incremental cache key so editing a model's ``config`` or
    ``transform`` (with byte-identical inputs) correctly re-runs the transform
    instead of silently carrying over stale outputs (code-review HIGH fix).
    """
    payload = json.dumps(
        {"transform": model.transform, "config": dict(model.config)},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _combine_hash(fingerprint: str, row_hash: str) -> str:
    return hashlib.sha256(f"{fingerprint}\x1f{row_hash}".encode("utf-8")).hexdigest()


def _apply_transform(
    fn: TransformFn,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    model_name: str,
) -> Optional[dict]:
    try:
        result = fn(row, config)
    except Exception as exc:  # noqa: BLE001 — transforms are operator code
        raise ValueError(
            f"transform failed on model {model_name!r} row id={row.get('id')!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if result is None:
        return None
    if not isinstance(result, Mapping):
        raise ValueError(
            f"transform on model {model_name!r} must return a dict or None, "
            f"got {type(result).__name__}"
        )
    out = dict(result)
    # Carry the row id through so derived/incremental models stay diffable
    # even when a custom transform forgets to preserve it.
    if "id" not in out and "id" in row:
        out["id"] = row["id"]
    return out


def _materialize_full(
    model: BuildModel,
    inputs: Sequence[Mapping[str, Any]],
    fn: TransformFn,
) -> Tuple[List[dict], int]:
    out_rows: List[dict] = []
    calls = 0
    for row in inputs:
        calls += 1
        result = _apply_transform(fn, row, model.config, model.name)
        if result is not None:
            out_rows.append(result)
    return out_rows, calls


def _materialize_incremental(
    conn: Any,
    model: BuildModel,
    inputs: Sequence[Mapping[str, Any]],
    fn: TransformFn,
) -> Tuple[List[dict], int, IncrementalDiffReport]:
    fingerprint = _model_fingerprint(model)
    new_hash: dict[str, str] = {}
    new_rows_by_id: dict[str, Mapping[str, Any]] = {}
    order: List[str] = []
    for row in inputs:
        rid = row.get("id")
        if rid is None:
            raise ValueError(
                f"incremental model {model.name!r} input row missing 'id'"
            )
        rid = str(rid)
        if rid in new_hash:
            raise ValueError(
                f"incremental model {model.name!r} has duplicate row id {rid!r}"
            )
        # Fold the transform+config fingerprint into the cache key so a config
        # edit (same inputs) re-runs the transform instead of carrying stale rows.
        new_hash[rid] = _combine_hash(fingerprint, compute_row_hash(row))
        new_rows_by_id[rid] = row
        order.append(rid)

    prev: dict[str, tuple] = {}
    cursor = conn.execute(
        "SELECT row_id, input_hash, output_json FROM build_rows WHERE model = ?",
        (model.name,),
    )
    for rid, input_hash, output_json in cursor.fetchall():
        prev[str(rid)] = (input_hash, output_json)

    added = changed = unchanged = 0
    calls = 0
    out_rows: List[dict] = []
    final_output: dict[str, Optional[str]] = {}  # rid -> output_json str or None
    for rid in order:
        h = new_hash[rid]
        in_prev = rid in prev
        same_hash = in_prev and prev[rid][0] == h
        reused = False
        if same_hash:
            # Carry the recorded decision over — no transform call. A
            # previously-dropped (output_json is None) unchanged row stays
            # dropped without re-running the transform.
            stored = prev[rid][1]
            if stored is None:
                final_output[rid] = None
                unchanged += 1
                continue
            try:
                res = json.loads(stored)
            except (TypeError, ValueError):
                res = None
            if isinstance(res, dict):
                final_output[rid] = stored
                out_rows.append(res)
                unchanged += 1
                reused = True
            # else: corrupt cache — fall through to re-transform (M4 fix).
        if reused:
            continue
        # added / changed / corrupt-cache → re-run the transform.
        if not in_prev:
            added += 1
        else:
            changed += 1
        calls += 1
        result = _apply_transform(fn, new_rows_by_id[rid], model.config, model.name)
        output_json = (
            json.dumps(result, ensure_ascii=False) if result is not None else None
        )
        final_output[rid] = output_json
        if result is not None:
            out_rows.append(result)

    removed = sum(1 for rid in prev if rid not in new_hash)
    # Rebuild state for this model: state == current inputs exactly.
    conn.execute("DELETE FROM build_rows WHERE model = ?", (model.name,))
    conn.executemany(
        "INSERT INTO build_rows (model, row_id, input_hash, output_json) "
        "VALUES (?, ?, ?, ?)",
        [(model.name, rid, new_hash[rid], final_output[rid]) for rid in order],
    )
    diff = IncrementalDiffReport(
        added=added, changed=changed, removed=removed, unchanged=unchanged
    )
    return out_rows, calls, diff


def _write_model_jsonl(
    output_dir: str, name: str, rows: Sequence[Mapping[str, Any]]
) -> str:
    # ``name`` is regex-validated (no path separators / ``..``) so the join
    # cannot escape ``output_dir``. Delegate the atomic write + cwd-containment
    # + symlink rejection to the v0.59.0 shared helper (single-source TOCTOU
    # defence — security/code review: no bespoke re-implementation).
    from soup_cli.utils.paths import atomic_write_bytes

    target = os.path.join(output_dir, f"{name}.jsonl")
    payload = (
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        + ("\n" if rows else "")
    ).encode("utf-8")
    return atomic_write_bytes(payload, target, prefix=".build-", field="build output")


def run_build(
    plan: BuildPlan,
    *,
    output_dir: str,
    transforms: Optional[Mapping[str, TransformFn]] = None,
    state_db: Optional[str] = None,
) -> BuildResult:
    """Execute the build DAG (v0.71.6 #231).

    Materialises each model in topological order:

    - ``table``  — apply the transform to every input row, overwrite the
      ``<output_dir>/<name>.jsonl`` artifact.
    - ``view``   — apply the transform but keep results in memory only (not
      written to disk); downstream ``refs`` still see the rows.
    - ``incremental`` — diff inputs against the SQLite state store and
      re-run the transform on only the added/changed rows (the
      re-tokenize-only-diff optimisation); unchanged rows carry over.

    Seed models read ``source`` JSONL (cwd-contained, symlink-rejected);
    derived models concatenate the outputs of their ``refs``. The SQLite
    state store lives at ``<output_dir>/.soup_build_state.sqlite`` (override
    via ``state_db``). Returns a :class:`BuildResult`.
    """
    import sqlite3
    import stat as _stat

    if not isinstance(plan, BuildPlan):
        raise TypeError("plan must be a BuildPlan")

    from soup_cli.utils.paths import is_under_cwd

    if isinstance(output_dir, bool) or not isinstance(output_dir, str) or not output_dir:
        raise TypeError("output_dir must be a non-empty string")
    if "\x00" in output_dir:
        raise ValueError("output_dir must not contain null bytes")
    if not is_under_cwd(output_dir):
        raise ValueError("output_dir must stay under cwd")
    # Symlink reject BEFORE makedirs (TOCTOU policy — lstat the raw path before
    # any filesystem-mutating call). ``makedirs(exist_ok=True)`` on a symlinked
    # dir would otherwise succeed silently (security review HIGH fix).
    if os.path.lexists(output_dir) and _stat.S_ISLNK(os.lstat(output_dir).st_mode):
        raise ValueError("output_dir must not be a symlink")
    os.makedirs(output_dir, exist_ok=True)

    lookup = {m.name: m for m in plan.models}
    # Fail fast: resolve every transform up front so an unknown transform name
    # cannot leave a partial build_out/ after earlier models materialised
    # (code review MEDIUM fix). Mirrors parse_build_plan validating the whole
    # DAG before returning.
    resolved: dict[str, TransformFn] = {
        name: resolve_transform(lookup[name].transform, transforms)
        for name in plan.topo_order
    }

    if state_db is None:
        state_db = os.path.join(output_dir, _STATE_DB_NAME)
    else:
        if not isinstance(state_db, str) or not state_db or "\x00" in state_db:
            raise ValueError("state_db must be a non-empty NUL-free string")
        if not is_under_cwd(state_db):
            raise ValueError("state_db must stay under cwd")
    if os.path.lexists(state_db) and _stat.S_ISLNK(os.lstat(state_db).st_mode):
        raise ValueError("state_db must not be a symlink")

    conn = sqlite3.connect(state_db)
    try:
        # Survive concurrent builds racing on the same state DB (matches the
        # v0.54.0 advise-history / v0.60.0 NamespacePinStore policy).
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS build_rows ("
            "model TEXT NOT NULL, row_id TEXT NOT NULL, "
            "input_hash TEXT NOT NULL, output_json TEXT, "
            "PRIMARY KEY (model, row_id))"
        )
        materialized: dict[str, List[dict]] = {}
        results: List[ModelBuildResult] = []
        for name in plan.topo_order:
            model = lookup[name]
            inputs = _gather_inputs(model, materialized)
            fn = resolved[name]
            if model.kind == "incremental":
                out_rows, calls, diff = _materialize_incremental(
                    conn, model, inputs, fn
                )
            else:
                out_rows, calls = _materialize_full(model, inputs, fn)
                diff = None
            materialized[name] = out_rows
            if model.kind == "view":
                output_path: Optional[str] = None
            else:
                output_path = _write_model_jsonl(output_dir, name, out_rows)
            results.append(
                ModelBuildResult(
                    name=name,
                    kind=model.kind,
                    rows_in=len(inputs),
                    rows_out=len(out_rows),
                    transform_calls=calls,
                    output_path=output_path,
                    diff=diff,
                )
            )
        conn.commit()
    finally:
        conn.close()

    return BuildResult(
        models=tuple(results),
        output_dir=os.path.realpath(output_dir),
    )


__all__ = [
    "BUILTIN_TRANSFORMS",
    "BuildModel",
    "BuildPlan",
    "BuildResult",
    "IncrementalDiffReport",
    "ModelBuildResult",
    "SUPPORTED_MODEL_KINDS",
    "compute_row_hash",
    "incremental_diff",
    "load_build_yaml",
    "parse_build_plan",
    "parse_build_yaml",
    "render_plan_table",
    "resolve_transform",
    "run_build",
    "validate_build_source",
    "validate_model_kind",
    "validate_model_name",
]


def _selfcheck() -> Iterable[str]:
    """Internal: list of expected public symbols."""
    return tuple(__all__)
