"""The parse / walk / memo cache the source-scanning guards share (``tests/_source_scan.py``).

A cache underneath a guard is only acceptable if it cannot change what the guard sees. What
is pinned here:

- ``parse_source`` hands back the tree ``ast.parse`` builds, and is keyed on the TEXT, so a
  file that changed on disk can not be served stale;
- by default it remembers only the last two texts, so a guard that walks a whole tree never
  holds that tree; ``keep=True`` is the explicit way to hold one;
- ``walk_nodes`` is ``ast.walk`` node for node, in the same order;
- ``memoized_scan`` runs a scanner once per input and never lets one caller's edit reach the
  next caller;
- everything is dropped when a test module finishes (``tests/conftest.py``), and "dropped"
  means the trees are freed, not merely unreachable from the cache.
"""

from __future__ import annotations

import ast
import gc
import textwrap
import weakref
from pathlib import Path

import pytest

from tests._source_scan import (
    RECENT_LIMIT,
    memoized_scan,
    parse_source,
    release,
    walk_nodes,
)

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src" / "soup_cli"

#: One of most things the grammar has, in syntax every supported interpreter (3.10+) parses.
KITCHEN_SINK = textwrap.dedent(
    '''
    """Module docstring."""
    from __future__ import annotations

    import os.path as osp
    from typing import Any

    LIMIT: int = 3
    first, *rest = [1, 2, 3]


    def decorate(func):
        return func


    @decorate
    async def fetch(url: str, *args: Any, retries: int = LIMIT, **kwargs) -> dict:
        """Docstring."""
        global LIMIT
        async with open_session(url) as session, other() as second:
            async for chunk in session:
                yield {key: value for key, value in chunk.items() if value}
        try:
            result = await session.get(url)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"{url!r} failed: {exc:>10}") from exc
        else:
            return {"ok": result[1:-1:2], "raw": result[...]}
        finally:
            del session


    class Widget(Base, metaclass=Meta):
        size = property(lambda self: self._size if self._size else -1)

        def grow(self, by=1, /, *, cap=None):
            nonlocal_free = [item**2 for row in self.rows for item in row if item % 2]
            while (left := self.size - by) > 0 and not cap:
                by += 1
                if left in {1, 2} or left is None:
                    continue
                break
            else:
                pass
            return nonlocal_free, (yield_free for yield_free in ()), not by


    def classify(command):
        match command:
            case {"kind": "move", "to": (x, y)} if x > y:
                return x
            case [first, *others] | Point(first=first, others=others):
                return others
            case str() as text:
                return text
            case _:
                assert command is not None, "unreachable"
    '''
)


def _real_sources() -> list[Path]:
    """A fixed, sorted sample of the real tree: every 25th module, plus the largest one."""
    modules = sorted(SRC.rglob("*.py"))
    sample = modules[::25]
    sample.append(SRC / "config" / "schema.py")
    return sample


class TestParseSource:
    def test_it_is_the_tree_ast_parse_builds(self) -> None:
        texts = [KITCHEN_SINK] + [path.read_text(encoding="utf-8") for path in _real_sources()]
        assert len(texts) > 20, "the real-tree sample came back short"
        for text in texts:
            assert ast.dump(parse_source(text), include_attributes=True) == ast.dump(
                ast.parse(text), include_attributes=True
            )

    def test_one_text_is_parsed_once(self) -> None:
        equal_but_distinct = "".join(["value", " = ", "1\n"])
        assert parse_source("value = 1\n") is parse_source(equal_but_distinct)

    def test_different_text_is_a_different_tree(self) -> None:
        assert parse_source("value = 1\n") is not parse_source("value = 2\n")

    def test_a_file_rewritten_on_disk_is_never_served_stale(self, tmp_path: Path) -> None:
        """The key is the text the guard just read, not the path it read it from.

        Same path, same length, rewritten immediately: the case an mtime-or-path key
        gets wrong, and the one a guard's seeded-violation self-test produces.
        """
        module = tmp_path / "module.py"
        module.write_text("alpha = 1\n", encoding="utf-8")
        before = parse_source(module.read_text(encoding="utf-8"))
        module.write_text("omega = 1\n", encoding="utf-8")
        after = parse_source(module.read_text(encoding="utf-8"))

        assert [node.targets[0].id for node in before.body] == ["alpha"]
        assert [node.targets[0].id for node in after.body] == ["omega"]

    def test_only_the_last_texts_are_remembered_by_default(self) -> None:
        """A guard that walks a whole tree through ``parse_source`` must not end up holding it.

        Holding every tree of ``tests/`` is 551 MB of private bytes, and 524 MB of it stays
        with the worker after ``release()``.
        """
        first = parse_source("first = 1\n")
        walk_nodes(first)
        alive = weakref.ref(first)
        del first
        for index in range(RECENT_LIMIT):
            assert alive() is not None, "still one of the last texts parsed"
            parse_source(f"other_{index} = 1\n")
        gc.collect()
        assert alive() is None, "a tree that fell out of the last texts must be freed"

    def test_keep_holds_a_tree_until_release(self) -> None:
        kept = parse_source("kept = 1\n", keep=True)
        nodes = walk_nodes(kept)
        for index in range(RECENT_LIMIT + 1):
            parse_source(f"other_{index} = 1\n")
        assert parse_source("kept = 1\n") is kept
        assert parse_source("kept = 1\n", keep=True) is kept
        assert walk_nodes(kept) is nodes

    def test_keep_adopts_a_tree_already_parsed(self) -> None:
        recent = parse_source("adopted = 1\n")
        nodes = walk_nodes(recent)
        assert parse_source("adopted = 1\n", keep=True) is recent
        for index in range(RECENT_LIMIT + 1):
            parse_source(f"other_{index} = 1\n")
        assert parse_source("adopted = 1\n") is recent
        assert walk_nodes(recent) is nodes

    def test_a_syntax_error_is_raised_every_time(self) -> None:
        with pytest.raises(SyntaxError) as direct:
            ast.parse("def broken(:\n")
        for _attempt in range(2):
            with pytest.raises(SyntaxError) as cached:
                parse_source("def broken(:\n")
            assert cached.value.msg == direct.value.msg
            assert cached.value.lineno == direct.value.lineno

    def test_a_syntax_error_names_the_file_the_caller_passed(self) -> None:
        """Guards that fail on an unparseable module report its path; that must survive."""
        with pytest.raises(SyntaxError) as failure:
            parse_source("def broken(:\n", filename="src/soup_cli/broken.py")
        assert failure.value.filename == "src/soup_cli/broken.py"

        tree = parse_source("value = 1\n", filename="one.py")
        assert parse_source("value = 1\n", filename="another.py") is tree


class TestWalkNodes:
    @staticmethod
    def _same_walk(tree: ast.AST) -> None:
        ours = walk_nodes(tree)
        theirs = list(ast.walk(tree))
        assert isinstance(ours, tuple)
        assert len(ours) == len(theirs)
        assert all(mine is stdlib for mine, stdlib in zip(ours, theirs)), (
            "walk_nodes must yield ast.walk's nodes in ast.walk's order"
        )

    def test_it_is_ast_walk_on_every_kind_of_node(self) -> None:
        tree = parse_source(KITCHEN_SINK)
        kinds = {type(node).__name__ for node in ast.walk(tree)}
        assert {"AsyncFunctionDef", "Match", "Lambda", "NamedExpr", "JoinedStr"} <= kinds
        self._same_walk(tree)

    def test_it_is_ast_walk_on_the_real_tree(self) -> None:
        for path in _real_sources():
            self._same_walk(parse_source(path.read_text(encoding="utf-8")))

    def test_it_is_ast_walk_on_a_tree_it_did_not_parse(self) -> None:
        self._same_walk(ast.parse(KITCHEN_SINK))

    def test_a_subtree_is_walked_from_that_node(self) -> None:
        tree = parse_source(KITCHEN_SINK)
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef))
        self._same_walk(function)
        assert walk_nodes(function)[0] is function
        assert len(walk_nodes(function)) < len(walk_nodes(tree))

    def test_a_tree_from_parse_source_is_flattened_once(self) -> None:
        tree = parse_source(KITCHEN_SINK)
        assert walk_nodes(tree) is walk_nodes(tree)

    def test_a_tree_from_elsewhere_is_not_retained(self) -> None:
        """Only trees this module handed out are kept: nothing else has a release point."""
        foreign = ast.parse(KITCHEN_SINK)
        assert walk_nodes(foreign) is not walk_nodes(foreign)


class TestMemoizedScan:
    def test_a_scanner_runs_once_per_distinct_input(self) -> None:
        calls: list[str] = []

        @memoized_scan
        def scan(source: str) -> list[str]:
            calls.append(source)
            return source.split()

        assert scan("a b") == ["a", "b"]
        assert scan("a b") == ["a", "b"]
        assert scan("c") == ["c"]
        assert calls == ["a b", "c"]

    def test_each_caller_gets_its_own_copy(self) -> None:
        @memoized_scan
        def scan(source: str) -> list[str]:
            return source.split()

        first = scan("a b")
        first.append("smuggled")
        assert scan("a b") == ["a", "b"]

        @memoized_scan
        def names(source: str) -> set[str]:
            return set(source.split())

        mine = names("a b")
        mine |= {"smuggled"}
        assert names("a b") == {"a", "b"}

    def test_keyword_arguments_are_part_of_the_key(self) -> None:
        @memoized_scan
        def scan(source: str, upper: bool = False) -> list[str]:
            return source.upper().split() if upper else source.split()

        assert scan("a b") == ["a", "b"]
        assert scan("a b", upper=True) == ["A", "B"]

    def test_a_failure_is_not_cached(self) -> None:
        attempts: list[str] = []

        @memoized_scan
        def scan(source: str) -> list[str]:
            attempts.append(source)
            raise SyntaxError("nope")

        for _attempt in range(2):
            with pytest.raises(SyntaxError):
                scan("x")
        assert attempts == ["x", "x"]

    def test_the_scanner_keeps_its_name_and_docstring(self) -> None:
        @memoized_scan
        def find_things(source: str) -> list[str]:
            """Find things."""
            return [source]

        assert find_things.__name__ == "find_things"
        assert find_things.__doc__ == "Find things."


class TestRelease:
    def test_release_drops_trees_walks_and_memos(self) -> None:
        """ "Dropped" is checked with weak references: the trees have to die.

        Identity alone cannot see a cache that forgot the text and still holds the tree in
        its second map. That mutation (``release()`` without ``_BY_TREE.clear()``) passed
        the first version of this test, with every tree of the module still alive.
        """
        calls: list[str] = []

        @memoized_scan
        def scan(source: str) -> list[str]:
            calls.append(source)
            return [source]

        kept = parse_source("kept = 1\n", keep=True)
        recent = parse_source("recent = 1\n")
        kept_nodes, recent_nodes = walk_nodes(kept), walk_nodes(recent)
        # [1] is the Assign statement. Not [-1]: that is ast.Store(), one shared object.
        alive = [weakref.ref(node) for node in (kept, recent, kept_nodes[1], recent_nodes[1])]
        scan("x")

        release()

        assert parse_source("kept = 1\n", keep=True) is not kept
        assert parse_source("recent = 1\n") is not recent
        scan("x")
        assert calls == ["x", "x"]

        del kept, recent, kept_nodes, recent_nodes
        gc.collect()
        assert [ref() for ref in alive] == [None] * 4, (
            "release() must free the trees it held, not only forget their text"
        )

    def test_every_test_module_releases_when_it_finishes(
        self, request: pytest.FixtureRequest
    ) -> None:
        """The wiring in ``tests/conftest.py``: autouse, module-scoped, releases after the yield.

        Without it the trees a guard kept stay alive for the rest of the worker (283 MB of
        private bytes for ``src/soup_cli``), and every later ``gc.collect()`` in that worker
        walks them (0.6 ms -> 230 ms, measured on the dev box).
        """
        assert "_release_scanned_sources" in request.fixturenames, (
            "the release fixture must be autouse: this test did not ask for it"
        )

        conftest = ast.parse((TESTS / "conftest.py").read_text(encoding="utf-8"))
        fixture = next(
            node
            for node in conftest.body
            if isinstance(node, ast.FunctionDef) and node.name == "_release_scanned_sources"
        )
        decorator = ast.unparse(fixture.decorator_list[0])
        assert decorator.startswith("pytest.fixture(")
        assert 'scope="module"' in decorator.replace("'", '"')
        assert "autouse=True" in decorator

        statements = [ast.unparse(statement) for statement in fixture.body]
        assert "yield" in statements
        after_yield = statements[statements.index("yield") + 1 :]
        assert "_source_scan.release()" in after_yield, (
            "the cache must be released AFTER the module's tests ran, not before"
        )
