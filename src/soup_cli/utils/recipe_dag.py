"""v0.45.0 Part E — Data Recipe DAG validators (schema-only).

Parses a YAML recipe describing a Seed -> LLM Text -> Code -> Judge ->
Validators -> Sampler graph and validates the topology. The live runner
(execution against a local model) is deferred to v0.45.1.
"""

from __future__ import annotations

import os
import re
import stat
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Tuple

# Closed allowlist of node kinds. Mirrors the unsloth Studio schema.
_NODE_KINDS = (
    "seed",
    "llm_text",
    "code",
    "judge",
    "validator",
    "sampler",
)
NODE_KINDS: frozenset = frozenset(_NODE_KINDS)

# Per-recipe caps — defence-in-depth against pathological YAML.
_MAX_NODES = 256
_MAX_EDGES = 1024
_MAX_NAME_LEN = 64
_MAX_FILE_BYTES = 1_048_576  # 1 MiB

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


@dataclass(frozen=True)
class RecipeNode:
    """One node in the data-recipe DAG."""

    name: str
    kind: str
    config: Mapping[str, Any]


@dataclass(frozen=True)
class RecipeDAG:
    """Validated recipe topology."""

    nodes: Tuple[RecipeNode, ...]
    edges: Tuple[Tuple[str, str], ...]
    topo_order: Tuple[str, ...]


def _check_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("node name must be a string")
    if "\x00" in name:
        raise ValueError("node name must not contain null bytes")
    if not _NAME_RE.match(name):
        raise ValueError(
            f"node name must match [a-z0-9][a-z0-9_-]{{0,{_MAX_NAME_LEN - 1}}}"
        )
    return name


def _check_kind(kind: str) -> str:
    if not isinstance(kind, str):
        raise TypeError("node kind must be a string")
    canonical = kind.strip().lower()
    if canonical not in NODE_KINDS:
        raise ValueError(
            f"unknown node kind: {kind!r}. supported: {sorted(NODE_KINDS)}"
        )
    return canonical


def _topological_sort(
    names: List[str], edges: List[Tuple[str, str]]
) -> List[str]:
    """Kahn's algorithm. Raises ``ValueError`` on cycle.

    Uses ``collections.deque`` so each pop is O(1); successors that become
    zero in-degree on the same step are appended in sorted order so the
    output is deterministic without re-sorting the queue every iteration.
    """
    in_degree: Dict[str, int] = {name: 0 for name in names}
    successors: Dict[str, List[str]] = {name: [] for name in names}
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
        raise ValueError("recipe DAG contains a cycle")
    return order


def parse_recipe(raw: Any) -> RecipeDAG:
    """Validate and topologically sort a recipe dict.

    Required shape::

        {
            "nodes": [{"name": "...", "kind": "...", "config": {...}}, ...],
            "edges": [["from_name", "to_name"], ...]
        }
    """
    if not isinstance(raw, dict):
        raise TypeError("recipe must be a dict")
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("recipe.nodes must be a non-empty list")
    if len(raw_nodes) > _MAX_NODES:
        raise ValueError(f"recipe.nodes exceeds {_MAX_NODES} entries")

    nodes: List[RecipeNode] = []
    seen_names: set[str] = set()
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            raise TypeError(f"recipe.nodes[{index}] must be a dict")
        name = _check_name(raw_node.get("name", ""))
        if name in seen_names:
            raise ValueError(f"duplicate node name: {name!r}")
        seen_names.add(name)
        kind = _check_kind(raw_node.get("kind", ""))
        config = raw_node.get("config", {})
        if not isinstance(config, dict):
            raise TypeError(f"recipe.nodes[{index}].config must be a dict")
        nodes.append(
            RecipeNode(name=name, kind=kind, config=MappingProxyType(dict(config)))
        )

    raw_edges = raw.get("edges", [])
    if not isinstance(raw_edges, list):
        raise TypeError("recipe.edges must be a list")
    if len(raw_edges) > _MAX_EDGES:
        raise ValueError(f"recipe.edges exceeds {_MAX_EDGES} entries")
    edges: List[Tuple[str, str]] = []
    edge_seen: set[Tuple[str, str]] = set()
    name_set = {node.name for node in nodes}
    for index, raw_edge in enumerate(raw_edges):
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            raise ValueError(
                f"recipe.edges[{index}] must be a 2-element [from, to] list"
            )
        source = _check_name(raw_edge[0])
        target = _check_name(raw_edge[1])
        if source == target:
            raise ValueError(f"self-loop edge rejected: {source!r}")
        if source not in name_set:
            raise ValueError(f"edge source {source!r} not in nodes")
        if target not in name_set:
            raise ValueError(f"edge target {target!r} not in nodes")
        key = (source, target)
        if key in edge_seen:
            raise ValueError(f"duplicate edge: {source!r} -> {target!r}")
        edge_seen.add(key)
        edges.append(key)

    topo = _topological_sort([n.name for n in nodes], edges)
    return RecipeDAG(
        nodes=tuple(nodes),
        edges=tuple(edges),
        topo_order=tuple(topo),
    )


def parse_recipe_yaml(text: str) -> RecipeDAG:
    """Parse a YAML string into a validated ``RecipeDAG``."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if "\x00" in text:
        raise ValueError("recipe text must not contain null bytes")
    if len(text.encode("utf-8")) > _MAX_FILE_BYTES:
        raise ValueError(f"recipe text exceeds {_MAX_FILE_BYTES} bytes")
    import yaml  # lazy import — keep CLI startup fast

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    return parse_recipe(data)


def load_recipe_yaml(path: str) -> RecipeDAG:
    """Load a recipe YAML from a path under cwd."""
    from soup_cli.utils.paths import is_under_cwd

    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if "\x00" in path:
        raise ValueError("path must not contain null bytes")
    # Symlink rejection on the *original* path (TOCTOU policy mirroring
    # v0.33.0 #22 / v0.43.0 Part C / v0.44.0 Part B). ``realpath`` below
    # would already follow symlinks, but we want to reject them as a
    # defence layer rather than silently follow.
    try:
        lstat_result = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(path) from exc
    if stat.S_ISLNK(lstat_result.st_mode):
        raise ValueError(
            f"recipe path must not be a symlink: {os.path.basename(path)}"
        )
    real = os.path.realpath(path)
    if not is_under_cwd(real):
        raise ValueError(f"recipe path must stay under cwd: {os.path.basename(real)}")
    if not os.path.isfile(real):
        raise FileNotFoundError(real)
    size = os.path.getsize(real)
    if size > _MAX_FILE_BYTES:
        raise ValueError(f"recipe file exceeds {_MAX_FILE_BYTES} bytes")
    with open(real, "r", encoding="utf-8") as handle:
        text = handle.read()
    return parse_recipe_yaml(text)


__all__ = [
    "NODE_KINDS",
    "RecipeDAG",
    "RecipeNode",
    "parse_recipe",
    "parse_recipe_yaml",
    "load_recipe_yaml",
]
