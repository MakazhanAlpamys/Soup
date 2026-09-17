"""Expanded-size limit for documents loaded with ``yaml.safe_load``.

YAML anchors and aliases let a few hundred bytes of text describe a graph in
which one list is referenced many times over. ``yaml.safe_load`` keeps those
references shared, so the loaded object is small -- but anything that walks it
as a tree (``json.dumps``, ``model_dump``, a recursive validator) visits every
path, and the number of paths grows exponentially with the nesting depth.

:func:`expanded_node_count` counts the nodes such a walk would visit without
performing it: a container's count is memoised by ``id()``, so a shared
reference is counted by multiplication rather than by re-walking it. The walk
is iterative, stops as soon as the running total passes the limit, and treats a
reference cycle (``&a [*a]``) or an excessive depth as over the limit.
"""

from __future__ import annotations

from typing import Iterator

#: Most nodes a loaded document may expand to before it is refused.
MAX_YAML_EXPANDED_NODES: int = 200_000

#: Nesting depth past which a document is treated as over the limit.
_MAX_DEPTH = 1000

_DONE = object()


def _children(node: object) -> Iterator[object]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield value
    else:
        yield from node  # type: ignore[misc]


def expanded_node_count(obj: object, *, limit: int = MAX_YAML_EXPANDED_NODES) -> int:
    """Return the number of nodes a tree walk of ``obj`` would visit.

    One per scalar, ``1 + sum(children)`` per list / tuple, and
    ``1 + sum(key + value)`` per dict. Returns ``limit + 1`` as soon as the
    count is known to exceed ``limit``, on a reference cycle, or when nesting
    is deeper than 1000 levels.
    """
    over = limit + 1
    memo: dict[int, int] = {}
    active: set[int] = set()

    def _known(node: object) -> int | None:
        if not isinstance(node, (dict, list, tuple)):
            return 1
        key = id(node)
        if key in memo:
            return memo[key]
        if key in active:
            return over
        return None

    first = _known(obj)
    if first is not None:
        return min(first, over)

    active.add(id(obj))
    stack: list[tuple[object, Iterator[object], list[int]]] = [
        (obj, _children(obj), [1]),
    ]
    while stack:
        node, children, total = stack[-1]
        child = next(children, _DONE)
        if child is _DONE:
            stack.pop()
            active.discard(id(node))
            memo[id(node)] = total[0]
            if not stack:
                return total[0]
            parent_total = stack[-1][2]
            parent_total[0] += total[0]
            if parent_total[0] > limit:
                return over
            continue
        count = _known(child)
        if count is None:
            if len(stack) >= _MAX_DEPTH:
                return over
            active.add(id(child))
            stack.append((child, _children(child), [1]))
            continue
        total[0] += count
        if total[0] > limit:
            return over
    return over  # pragma: no cover - the loop always returns


def check_yaml_expanded_size(obj: object, what: str) -> None:
    """Raise ``ValueError`` when ``obj`` expands past the node limit."""
    if expanded_node_count(obj, limit=MAX_YAML_EXPANDED_NODES) > MAX_YAML_EXPANDED_NODES:
        raise ValueError(
            f"{what} expands to more than {MAX_YAML_EXPANDED_NODES} nodes; refusing"
        )
