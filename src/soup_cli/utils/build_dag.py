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
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

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
# Live runner — deferred to v0.69.1
# -----------------------------------------------------------------------------


def run_build(plan: BuildPlan, *, output_dir: Optional[str] = None) -> None:
    """Execute the build DAG. Deferred to v0.69.1.

    Same stub-then-live cadence as v0.45.0 recipe DAG / v0.50.0 GRPO Plus /
    v0.61.0 unlearning. The CLI prints the resolved plan + a deferred-live
    advisory and exits with code 3 so CI gates can distinguish "deferred /
    not yet shipped" from "validation rejection" (which exits 2).
    """
    if not isinstance(plan, BuildPlan):
        raise TypeError("plan must be a BuildPlan")
    raise NotImplementedError(
        "soup build live runner is deferred to v0.69.1 — only --dry-run is wired today."
    )


__all__ = [
    "BuildModel",
    "BuildPlan",
    "IncrementalDiffReport",
    "SUPPORTED_MODEL_KINDS",
    "compute_row_hash",
    "incremental_diff",
    "load_build_yaml",
    "parse_build_plan",
    "parse_build_yaml",
    "render_plan_table",
    "run_build",
    "validate_build_source",
    "validate_model_kind",
    "validate_model_name",
]


def _selfcheck() -> Iterable[str]:
    """Internal: list of expected public symbols."""
    return tuple(__all__)
