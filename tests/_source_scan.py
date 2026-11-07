"""THE parse / walk / memo cache for guards that scan ``src/soup_cli`` or ``tests/``. One copy.

About a hundred test files walk a whole tree and ``ast.parse`` every file in it, and the slow
ones do it several times: once per test, per parametrised case, or per registry entry. Measured
on the dev box (one process, no coverage), one pass costs:

=========  =====  ======  ========  =========  =========
tree       files  list    read      parse      ast.walk
=========  =====  ======  ========  =========  =========
src        542    4-8 ms  27-47 ms  1.1-1.9 s  0.5 s
tests      876    8 ms    72 ms     1.9-2.5 s  1.8 s
=========  =====  ======  ========  =========  =========

Listing and reading are about 2% of a pass, so **they are not cached**: a guard keeps its own
``sorted(root.rglob("*.py"))`` and its own ``read_text``. What is cached is what costs:

``parse_source(text, filename="<unknown>", *, keep=False)``
    ``ast.parse(text, filename)``. By default the tree is remembered only while its text is
    one of the last two parsed, which is enough for the scanners that look at one file back
    to back to share a parse. ``keep=True`` holds it until ``release()``.
``walk_nodes(tree)``
    ``tuple(ast.walk(tree))`` without the generators: same nodes, same order. Kept for a tree
    ``parse_source`` still remembers, rebuilt each call for any other node.
``memoized_scan``
    Decorator for a pure ``scanner(text) -> findings``: one run per distinct input, and every
    caller gets its own shallow copy of the result.

**``keep=True`` has a price that outlives the module.** Private bytes of one process (Windows,
Python 3.12.10):

==========================  ======  ===================
what the process held       peak    after ``release()``
==========================  ======  ===================
nothing (parse and drop)    65 MB   23 MB
every tree of ``src``       283 MB  266 MB
every tree of ``tests``     551 MB  524 MB
==========================  ======  ===================

``release()`` frees the trees (pinned by a weakref test) but the allocator keeps the pages, so
a worker that held a whole tree once stays that much larger until it exits. The walk tuples
are not the cost (3 MB of the 283). Hence the rule: **keep a whole tree only where several
whole-tree passes share it**, and never ``tests/``. Today that is ``src`` in ``test_issue748_*``
(seven-fold faster for it), ``test_issue616_*`` and ``test_issue752_*``, whose scan held every
tree before this helper existed. A guard whose tests each make one pass gains nothing from
``keep``; prefer ``memoized_scan`` on its findings, which holds no tree at all.

**What may be relied on.** ``parse_source`` is keyed on the *text* the caller read, never on a
path or an mtime, and ``memoized_scan`` on the scanner's own arguments. A file that changes on
disk between two guards is therefore a different key and is parsed again; a seeded violation
written to ``tmp_path`` and rewritten in place is scanned for what it now contains.

**What is NOT supported.**

- A memoised scanner that reads anything but its arguments (a file, a global table): a file
  that changes on disk between two calls is then served stale until the module ends. Pass the
  text in. One scanner is knowingly keyed on a file LIST instead: ``_dead_function_inputs`` in
  ``test_issue748_*``, next to the ``_consumed_cached`` that already was. Both are right within
  one unchanged tree, which is all a test run needs; neither notices a module edited mid-run.
- Editing a tree from ``parse_source``. It is shared with every other caller, so a scanner
  that rewrites its tree (strips docstrings, sets ``node.parent``) must call ``ast.parse``
  itself and cache its *findings* with ``memoized_scan`` instead.
- Editing a tree while iterating ``walk_nodes``. ``ast.walk`` is lazy and sees such an edit;
  ``walk_nodes`` is a snapshot taken before the loop body runs.
- Results that nest mutable containers: the copy is one level deep.

**Lifetime: one test module.** ``tests/conftest.py`` calls ``release()`` when a module's tests
are done. While kept trees are alive every ``gc.collect()`` in the worker walks them (0.6 ms
-> 230 ms for ``src``). Call these from test bodies, not at import time: a tree kept during
collection is held until the first module finishes.
"""

from __future__ import annotations

import ast
import collections
import copy
import functools
import weakref
from typing import Any, Callable, TypeVar

_Scanner = TypeVar("_Scanner", bound=Callable[..., Any])

#: How many texts the default (``keep=False``) cache remembers: the file being scanned and
#: the one before it. Guards scan one file at a time.
RECENT_LIMIT = 2


class _Parsed:
    """One tree ``parse_source`` handed out, and its flattened walk once someone asked."""

    __slots__ = ("tree", "nodes")

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.nodes: tuple[ast.AST, ...] | None = None


#: source text -> its tree, for the callers that asked to keep it until ``release()``.
_KEPT: dict[str, _Parsed] = {}
#: source text -> its tree, for the last ``RECENT_LIMIT`` texts parsed without ``keep``.
_RECENT: collections.OrderedDict[str, _Parsed] = collections.OrderedDict()
#: id(tree) -> the same record, for every tree in the two maps above. The record keeps the
#: tree alive, so the id cannot be reused while the entry exists.
_BY_TREE: dict[int, _Parsed] = {}
#: Every live memoised scanner, so ``release()`` reaches them too. Weak: a scanner defined
#: inside a test goes away with that test.
_MEMOS: weakref.WeakSet[Any] = weakref.WeakSet()


def parse_source(source: str, filename: str = "<unknown>", *, keep: bool = False) -> ast.Module:
    """``ast.parse(source, filename)``, shared between callers. Do not edit the result.

    ``keep=False``: remembered while ``source`` is one of the last ``RECENT_LIMIT`` texts
    parsed. ``keep=True``: remembered until ``release()``; read the module docstring for what
    that costs before reaching for it.

    Raises exactly what ``ast.parse`` raises, every time: a failure is not cached, so its
    message names the caller's ``filename``. A tree carries no filename, which is why the
    key is the text alone.
    """
    held = _KEPT.get(source)
    if held is not None:
        return held.tree
    held = _RECENT.get(source)
    if held is None:
        held = _Parsed(ast.parse(source, filename))
        _BY_TREE[id(held.tree)] = held
    if keep:
        _RECENT.pop(source, None)
        _KEPT[source] = held
        return held.tree
    _RECENT[source] = held
    _RECENT.move_to_end(source)
    while len(_RECENT) > RECENT_LIMIT:
        _text, dropped = _RECENT.popitem(last=False)
        del _BY_TREE[id(dropped.tree)]
    return held.tree


def _flatten(root: ast.AST) -> tuple[ast.AST, ...]:
    """``ast.walk(root)`` as a tuple: breadth-first, children in field order.

    The list grows while it is iterated, which is the queue ``ast.walk`` keeps in a deque.
    One frame instead of three nested generators per node, which is where the time goes --
    more so under coverage, whose tracer is called on every generator resume.
    """
    nodes: list[ast.AST] = [root]
    for node in nodes:
        for name in node._fields:
            value = getattr(node, name, None)
            if isinstance(value, ast.AST):
                nodes.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        nodes.append(item)
    return tuple(nodes)


def walk_nodes(tree: ast.AST) -> tuple[ast.AST, ...]:
    """Every node ``ast.walk(tree)`` yields, in the same order, as a tuple."""
    held = _BY_TREE.get(id(tree))
    if held is None or held.tree is not tree:
        return _flatten(tree)
    if held.nodes is None:
        held.nodes = _flatten(tree)
    return held.nodes


def memoized_scan(scanner: _Scanner) -> _Scanner:
    """Run a pure ``scanner(*args)`` once per distinct arguments, until ``release()``.

    Each call returns ``copy.copy`` of the cached result, so a caller that sorts, extends or
    ``|=``-s what it got cannot change what the next caller gets. An exception is not cached.
    """
    cached = functools.lru_cache(maxsize=None)(scanner)
    _MEMOS.add(cached)

    @functools.wraps(scanner)
    def scan(*args: Any, **kwargs: Any) -> Any:
        return copy.copy(cached(*args, **kwargs))

    return scan  # type: ignore[return-value]


def release() -> None:
    """Drop every parsed tree, flattened walk and memoised result."""
    _KEPT.clear()
    _RECENT.clear()
    _BY_TREE.clear()
    for cached in list(_MEMOS):
        cached.cache_clear()
