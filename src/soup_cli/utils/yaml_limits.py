"""Size, nesting and expansion limits for documents loaded from YAML.

Every YAML document Soup reads goes through :func:`load_yaml_limited` or
:func:`load_yaml_file_limited`. Both build the document exactly as
``yaml.safe_load`` does and raise :class:`YamlLimitError`, naming the document
and the limit, when it crosses one of three:

* **size** -- the source is at most ``max_bytes`` long, and a file is never
  read further than one byte past that;
* **nesting** -- collections nest at most :data:`MAX_YAML_NESTING` levels,
  whether written out or reached through aliases;
* **expansion** -- the loaded object has at most one node per source byte plus
  :data:`ALIAS_EXPANSION_SLACK`, and so have the pairs that merge keys (``<<``)
  copy into mappings while the document is being built.

The expansion rule exists because YAML anchors and aliases let a few hundred
bytes of text describe a graph in which one list is referenced many times over.
``yaml.safe_load`` keeps those references shared, so the loaded object is small
-- but anything that walks it as a tree (``json.dumps``, ``model_dump``, a
recursive validator) visits every path, and the number of paths grows
exponentially with the nesting depth.

:func:`expanded_node_count` counts the nodes such a walk would visit without
performing it: a container's count is memoised by ``id()``, so a shared
reference is counted by multiplication rather than by re-walking it. The walk
is iterative, stops as soon as the running total passes the limit, and treats a
reference cycle (``&a [*a]``) or an excessive depth as over the limit.

The limit is tied to the size of the source rather than fixed: a YAML document
without aliases needs at least one byte per node (``{a,b}`` is five bytes and
five nodes; nothing denser parses), so an alias-free document always passes
:func:`check_yaml_expanded_size`, whatever legitimate content it carries. Only
aliases can make a document expand to more nodes than it has bytes. They do
it in two places: in the loaded object, which is measured after loading, and
in merge keys, where PyYAML copies the pairs of every merged mapping into the
mapping that names it before any object exists. The loader charges those
copies to the same allowance as they are made and stops at the first pair
past it, so the work done during and after loading stays proportional to the
source size, which the reader already caps.
"""

from __future__ import annotations

import os
from typing import Iterator

import yaml

#: Nodes allowed beyond one per source byte, so small documents (including the
#: empty document, which loads as ``{}``) and modest alias reuse still load.
ALIAS_EXPANSION_SLACK: int = 10_000

#: Largest training config read into memory, whether it is a ``soup.yaml`` on
#: disk, a config passed as text, or the ``config.yaml`` of a can. A training
#: config is kilobytes.
MAX_CONFIG_BYTES: int = 1 * 1024 * 1024

#: Levels of nested collections (mappings and sequences) a YAML document may
#: have. PyYAML builds nested collections by recursion, so this is what keeps a
#: document's stack use fixed whatever the interpreter's recursion limit is,
#: and it leaves every later walk of the loaded object shallow. Configs and
#: manifests nest fewer than ten levels.
MAX_YAML_NESTING: int = 100

#: Nesting depth past which the node counter treats any object as over the limit.
_MAX_DEPTH = 1000

_CONTAINERS = (dict, list, tuple)

_DONE = object()


class YamlLimitError(ValueError):
    """A YAML document is over its size, nesting or expansion limit."""


def _children(node: object) -> Iterator[object]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield value
    else:
        yield from node  # type: ignore[misc]


def _measure(obj: object, *, limit: int) -> tuple[int, int]:
    """Return ``(nodes, depth)`` of ``obj`` walked as a tree.

    ``nodes`` is one per scalar, ``1 + sum(children)`` per list / tuple and
    ``1 + sum(key + value)`` per dict; ``depth`` is the number of containers on
    the longest path, so a scalar is 0 and ``[[]]`` is 2. Both are memoised by
    ``id()``: a container reached again through a shared reference contributes
    its stored count and depth without being walked again, so levels that exist
    only through shared references are still measured.

    The walk is iterative. It stops with ``nodes == limit + 1`` as soon as the
    count is known to exceed ``limit`` or on a reference cycle; ``depth`` is
    then only a lower bound.
    """
    over = limit + 1
    if not isinstance(obj, _CONTAINERS):
        return min(1, over), 0

    nodes: dict[int, int] = {}
    depths: dict[int, int] = {}
    active: set[int] = {id(obj)}
    # A container, its remaining children, and [nodes so far, deepest child].
    stack: list[tuple[object, Iterator[object], list[int]]] = [
        (obj, _children(obj), [1, 0]),
    ]
    while stack:
        node, children, state = stack[-1]
        child = next(children, _DONE)
        if child is _DONE:
            stack.pop()
            key = id(node)
            active.discard(key)
            depth = state[1] + 1
            nodes[key] = state[0]
            depths[key] = depth
            if not stack:
                return state[0], depth
            parent = stack[-1][2]
            parent[0] += state[0]
            parent[1] = max(parent[1], depth)
            if parent[0] > limit:
                return over, parent[1]
            continue
        if not isinstance(child, _CONTAINERS):
            state[0] += 1
        else:
            key = id(child)
            if key in active:
                return over, state[1]
            if key not in nodes:
                active.add(key)
                stack.append((child, _children(child), [1, 0]))
                continue
            state[0] += nodes[key]
            state[1] = max(state[1], depths[key])
        if state[0] > limit:
            return over, state[1]
    return over, 0  # pragma: no cover - the loop always returns


def expanded_node_count(obj: object, *, limit: int) -> int:
    """Return the number of nodes a tree walk of ``obj`` would visit.

    One per scalar, ``1 + sum(children)`` per list / tuple, and
    ``1 + sum(key + value)`` per dict. Returns ``limit + 1`` as soon as the
    count is known to exceed ``limit``, on a reference cycle, or when nesting
    is deeper than 1000 levels.
    """
    nodes, depth = _measure(obj, limit=limit)
    return limit + 1 if depth > _MAX_DEPTH else nodes


def check_expanded_nodes(obj: object, what: str, *, limit: int) -> None:
    """Raise ``ValueError`` when ``obj`` expands to more than ``limit`` nodes."""
    if expanded_node_count(obj, limit=limit) > limit:
        raise ValueError(f"{what} expands to more than {limit} nodes; refusing")


def _expansion_error(what: str, limit: int, source_bytes: int) -> YamlLimitError:
    return YamlLimitError(
        f"{what} expands to more than {limit} nodes from {source_bytes} "
        "bytes of YAML; refusing"
    )


def _nesting_error(what: str) -> YamlLimitError:
    return YamlLimitError(
        f"{what} nests collections more than {MAX_YAML_NESTING} levels deep; refusing"
    )


def check_yaml_expanded_size(obj: object, what: str, *, source_bytes: int) -> None:
    """Refuse ``obj`` when it nests or expands past what its YAML source allows.

    ``source_bytes`` is the length of the text ``obj`` was loaded from; the node
    allowance is that plus :data:`ALIAS_EXPANSION_SLACK` (see module docstring).
    Nesting is measured on the loaded object, so levels that exist only through
    aliases count towards :data:`MAX_YAML_NESTING` too.
    """
    limit = source_bytes + ALIAS_EXPANSION_SLACK
    nodes, depth = _measure(obj, limit=limit)
    if depth > MAX_YAML_NESTING:
        raise _nesting_error(what)
    if nodes > limit:
        raise _expansion_error(what, limit, source_bytes)


class _TooDeepError(Exception):
    """Raised while parsing when collections nest past the limit."""


class _MergeChainError(Exception):
    """Raised while building when merge keys chain past the limit."""


class _MergeBudgetError(Exception):
    """Raised while building when merge keys copy more nodes than allowed."""


_MERGE_TAG = "tag:yaml.org,2002:merge"
_VALUE_TAG = "tag:yaml.org,2002:value"


class _LimitedLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that bounds nesting and merge-key copies as it loads.

    The composer takes every event through ``get_event``, so counting
    collection starts and ends there bounds its recursion in the same pass that
    builds the document, without reading the text twice.

    ``flatten_mapping`` is the one place PyYAML copies nodes: a merge key puts
    the pairs of each mapping it names into the mapping that holds it. The
    version here builds the same pair list, charges two nodes per copied pair
    to ``node_budget`` before copying, and counts how deep merges chain.
    Sequences, sets, ``!!omap`` and ``!!pairs`` are built in one pass over
    their own node and copy nothing.
    """

    def __init__(self, stream: str | bytes, *, node_budget: int) -> None:
        super().__init__(stream)
        self._open_collections = 0
        self._node_budget = node_budget
        self._merge_depth = 0
        #: Nodes copied by merge keys so far.
        self.merged_nodes = 0

    def get_event(self) -> yaml.Event:
        event = super().get_event()
        if isinstance(event, yaml.CollectionStartEvent):
            self._open_collections += 1
            if self._open_collections > MAX_YAML_NESTING:
                raise _TooDeepError
        elif isinstance(event, yaml.CollectionEndEvent):
            self._open_collections -= 1
        return event

    def _merged_pairs(self, owner: yaml.MappingNode, source: yaml.Node) -> list:
        """The pairs ``source`` contributes to a merge, charged to the budget."""
        if not isinstance(source, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                owner.start_mark,
                f"expected a mapping for merging, but found {source.id}",
                source.start_mark,
            )
        self.flatten_mapping(source)
        self.merged_nodes += 2 * len(source.value)
        if self.merged_nodes > self._node_budget:
            raise _MergeBudgetError
        return source.value

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        """As ``SafeConstructor.flatten_mapping``, step for step, with the copies counted."""
        self._merge_depth += 1
        if self._merge_depth > MAX_YAML_NESTING:
            raise _MergeChainError
        merged: list = []
        index = 0
        while index < len(node.value):
            key_node, value_node = node.value[index]
            if key_node.tag != _MERGE_TAG:
                if key_node.tag == _VALUE_TAG:
                    key_node.tag = "tag:yaml.org,2002:str"
                index += 1
                continue
            # Taken out before the mappings it names are visited, as PyYAML does:
            # a mapping that names itself is then merged without this entry.
            del node.value[index]
            if isinstance(value_node, yaml.MappingNode):
                merged.extend(self._merged_pairs(node, value_node))
            elif isinstance(value_node, yaml.SequenceNode):
                parts = [self._merged_pairs(node, source) for source in value_node.value]
                # A later mapping in the list yields to an earlier one.
                for pairs in reversed(parts):
                    merged.extend(pairs)
            else:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "expected a mapping or list of mappings for merging, "
                    f"but found {value_node.id}",
                    value_node.start_mark,
                )
        if merged:
            node.value = merged + node.value
        self._merge_depth -= 1


class YamlScalarError(yaml.constructor.ConstructorError, ValueError):
    """A scalar that reads as a YAML type but cannot be built, such as a date
    that does not exist. Both a ``yaml.YAMLError`` (it carries the position) and
    the ``ValueError`` the underlying conversion raised."""


def _checked(name: str, build):
    def construct(self: _LimitedLoader, node: yaml.ScalarNode) -> object:
        try:
            return build(self, node)
        except ValueError as exc:
            reason = str(exc)
        except (KeyError, AttributeError, IndexError):
            # What the conversion raises for text an explicit tag puts in front
            # of it: `!!bool foo`, `!!timestamp foo`, `!!int ""`.
            reason = f"not a valid {name}"
        raise YamlScalarError(
            None, None, f"cannot build this value: {reason}", node.start_mark
        )

    return construct


# The four safe constructors that convert text with a Python call that can refuse it.
for _name in ("bool", "int", "float", "timestamp"):
    _tag = f"tag:yaml.org,2002:{_name}"
    _LimitedLoader.add_constructor(
        _tag, _checked(_name, yaml.SafeLoader.yaml_constructors[_tag])
    )


def _parse(source: str | bytes, what: str, *, source_bytes: int) -> object:
    """Build the single document in ``source`` as ``yaml.safe_load`` would."""
    limit = source_bytes + ALIAS_EXPANSION_SLACK
    # A SafeLoader subclass: the same tags and constructors as safe_load.
    loader = _LimitedLoader(source, node_budget=limit)
    try:
        return loader.get_single_data()
    except _TooDeepError:
        raise _nesting_error(what) from None
    except _MergeChainError:
        raise YamlLimitError(
            f"{what} chains merge keys more than {MAX_YAML_NESTING} levels deep; refusing"
        ) from None
    except _MergeBudgetError:
        raise _expansion_error(what, limit, source_bytes) from None
    except RecursionError:
        # Reached only when the caller's own stack is already deep.
        raise YamlLimitError(f"{what} is nested or chained too deeply to load; refusing") from None
    finally:
        loader.dispose()


def _size_error(what: str, max_bytes: int) -> YamlLimitError:
    return YamlLimitError(f"{what} is larger than {max_bytes} bytes; refusing")


def load_yaml_limited(source: str | bytes, what: str, *, max_bytes: int) -> object:
    """Load one YAML document from text, within the limits.

    Returns what ``yaml.safe_load(source)`` returns. Raises
    :class:`YamlLimitError` -- a ``ValueError`` whose message starts with
    ``what`` -- when the text is longer than ``max_bytes`` (UTF-8), nests
    collections more than :data:`MAX_YAML_NESTING` levels, or expands past the
    node allowance. A syntax error is still a ``yaml.YAMLError``,
    and a scalar that cannot be built is a :class:`YamlScalarError`.
    """
    if isinstance(source, str):
        source_bytes = len(source.encode("utf-8", "surrogatepass"))
    elif isinstance(source, bytes):
        source_bytes = len(source)
    else:
        raise TypeError(f"{what} must be str or bytes, got {type(source).__name__}")
    if source_bytes > max_bytes:
        raise _size_error(what, max_bytes)
    data = _parse(source, what, source_bytes=source_bytes)
    check_yaml_expanded_size(data, what, source_bytes=source_bytes)
    return data


def read_text_limited(
    path: str | os.PathLike[str], what: str, *, max_bytes: int
) -> str:
    """Return the text of a UTF-8 file of at most ``max_bytes`` bytes.

    Reads at most ``max_bytes + 1`` bytes and raises :class:`YamlLimitError`
    naming ``what`` when the file is larger, so the whole of a larger file is
    never held in memory.
    """
    with open(path, "rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise _size_error(what, max_bytes)
    return raw.decode("utf-8")


def load_yaml_file_limited(
    path: str | os.PathLike[str], what: str, *, max_bytes: int
) -> object:
    """Load one YAML document from a UTF-8 file, within the limits.

    The file is read with :func:`read_text_limited`, so ``max_bytes`` counts
    the bytes on disk and a larger file is refused without being held in
    memory. The result is what ``yaml.safe_load`` returns for the file's
    text: YAML treats CRLF, CR and LF line ends alike and skips a leading
    byte order mark. Raises as :func:`load_yaml_limited` does; ``OSError``
    and ``UnicodeDecodeError`` are passed on unchanged.
    """
    text = read_text_limited(path, what, max_bytes=max_bytes)
    return load_yaml_limited(text, what, max_bytes=max_bytes)
