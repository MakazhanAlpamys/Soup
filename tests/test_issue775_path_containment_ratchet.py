"""#775 — repo-wide ratchet: containment must not ride on ``relative_to``.

``Path.resolve() + relative_to()`` is the containment idiom AGENTS.md bans:
one of the two paths can carry a Windows 8.3 short name (``C:\\Users\\RUNNER~1``)
while the other carries the long form, so ``relative_to`` raises ``ValueError``
for a path that *is* inside the base and the guard denies a legitimate write.
The migration to ``utils.paths.is_under`` (realpath + commonpath) is done; this
file is the lock that keeps it done.

**Why the previous guard was not a lock.** ``test_code_review_recurring.py``
asserted ``".relative_to(cwd)" not in src`` over **seven named files**. Two
holes, both fatal to its purpose:

1. *Scope.* A new module — or an eighth existing one — could reintroduce the
   idiom and the guard would never look at it.
2. *Spelling.* It matched one literal. ``.relative_to(base)``,
   ``.relative_to(root)``, ``.relative_to(\n    directory\n)`` and
   ``.is_relative_to(...)`` all sailed past. The variable happened to be named
   ``cwd`` at the sites that were migrated; nothing makes the next one agree.

So the scan is an AST walk over ``src/soup_cli/**/*.py`` and keys on the CALL,
not on the text around it. Every spelling of the receiver and the argument is
the same tree.

**What counts as containment** (a plain ``relative_to`` is not automatically a
bug — it is also the normal way to shorten a path for display):

* ``is_relative_to`` — always. It returns a bool and exists for no other
  purpose than answering "is this inside that".
* ``relative_to`` inside a ``try`` whose ``except`` can catch ``ValueError`` —
  that handler is the containment decision. Where the answer is used (deny,
  skip, fall back) is deliberately NOT inspected: telling "reject" from
  "cosmetic fallback" apart needs a taint analysis, and the cheap version got
  ``adapters.py:38`` wrong in both directions while being written. A benign
  site is one allowlist line with a reason instead.
  ``with contextlib.suppress(ValueError):`` is the same decision written as a
  context manager and counts identically.
* ``relative_to`` in a boolean position — ``if``/``while``/``assert`` test,
  ``not``, ``and``/``or``, a ternary condition, a comprehension filter. Nobody
  writes that for display.

Everything else — ``rel = p.relative_to(root)`` with no ``ValueError`` guard —
is left alone.

At the time of writing the scan finds **two** sites, both in
``commands/adapters.py``, both benign and both allowlisted below with their
reason. There is no cleanup left, which makes ``TestTheScannerCanActuallyFail``
load-bearing: a scanner whose only findings are allowlisted must be shown
capable of finding something, or it is indistinguishable from one that matches
nothing at all.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli"

_CONTAINMENT_METHODS = frozenset({"relative_to", "is_relative_to"})

#: Exception names whose handler catches ``ValueError``. A bare ``except:``
#: (``handler.type is None``) counts too — it catches everything.
_CATCHES_VALUE_ERROR = frozenset({"ValueError", "Exception", "BaseException"})

#: Benign call sites, each with the reason it does not decide containment.
#: Keyed by ``(path relative to src/soup_cli, line)``. Both entries are
#: verified live by ``test_the_allowlist_has_no_dead_entries`` — a moved or
#: deleted site fails there rather than rotting into a silent hole.
ALLOWLIST: dict[tuple[str, int], str] = {
    ("commands/adapters.py", 29): (
        "depth counter, not a gate. The paths come from "
        "`directory.rglob('adapter_config.json')`, so they are already under "
        "`directory` by construction; the result is only measured for its "
        "`.parts` length against max_depth, and the `except ValueError: "
        "continue` is unreachable for rglob output rather than a denial."
    ),
    ("commands/adapters.py", 98): (
        "display-only shortening. On ValueError it falls back to the full "
        "`adapter_path` and still renders the row — the value reaches "
        "`table.add_row` and nothing else, so a short-name mismatch costs a "
        "longer string in a table, never a refused path."
    ),
}


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _names_catch_value_error(nodes: list[ast.expr]) -> bool:
    """Does any exception expression in ``nodes`` catch ``ValueError``?

    Shared by ``except (A, B):`` and ``suppress(A, B)`` — one rule, two
    spellings. Tuples are recursed into: ``suppress`` forwards its arguments to
    ``issubclass``, which accepts nested tuples, so ``suppress((OSError,
    ValueError))`` is a real spelling.
    """
    for node in nodes:
        if isinstance(node, ast.Tuple):
            if _names_catch_value_error(node.elts):
                return True
            continue
        name = (
            node.id
            if isinstance(node, ast.Name)
            else node.attr
            if isinstance(node, ast.Attribute)
            else ""
        )
        if name in _CATCHES_VALUE_ERROR:
            return True
    return False


def _handler_catches_value_error(handler: ast.ExceptHandler) -> bool:
    exc = handler.type
    if exc is None:  # bare `except:`
        return True
    return _names_catch_value_error(exc.elts if isinstance(exc, ast.Tuple) else [exc])


def _suppress_names(tree: ast.AST) -> set[str]:
    """Local names bound to ``contextlib.suppress`` in this module.

    ``import contextlib as cl`` needs no entry: the attribute form below keys on
    ``.suppress`` alone.
    """
    names = {"suppress"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "contextlib":
            names.update(a.asname or a.name for a in node.names if a.name == "suppress")
    return names


def _is_suppress_call(node: ast.expr, suppress_names: set[str]) -> bool:
    func = node.func if isinstance(node, ast.Call) else None
    if isinstance(func, ast.Name):
        return func.id in suppress_names
    # ponytail: any `X.suppress(...)` counts — covers `contextlib.suppress` and
    # any module alias without following imports. Narrow to the real module if a
    # non-contextlib `suppress` attribute ever shows up.
    return isinstance(func, ast.Attribute) and func.attr == "suppress"


def _in_field(parent: ast.AST, child: ast.AST, field: str) -> bool:
    """Is ``child`` reachable from ``parent.<field>``?

    ``child`` is the node one step below ``parent`` on the walk up, so a direct
    identity check on the field (or membership in a field that is a list) says
    which branch of the parent the call came from — ``If.test`` versus
    ``If.body`` is the whole difference between a containment check and a
    display line that happens to sit in a branch.
    """
    value = getattr(parent, field, None)
    if isinstance(value, list):
        return any(item is child for item in value)
    return value is child


def _containment_reason(
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    suppress_names: set[str] = frozenset({"suppress"}),  # type: ignore[assignment]
) -> str | None:
    """Why ``call`` decides containment, or ``None`` when it does not."""
    if call.func.attr == "is_relative_to":  # type: ignore[union-attr]
        return "is_relative_to is a containment predicate"

    child: ast.AST = call
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # A `try` further out belongs to the caller, not to this call.
            return None
        if isinstance(parent, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            if _in_field(parent, child, "body") and any(
                _handler_catches_value_error(h) for h in parent.handlers
            ):
                return "relative_to guarded by an except that catches ValueError"
        if isinstance(parent, (ast.With, ast.AsyncWith)) and _in_field(
            parent, child, "body"
        ):
            # `body` only, never `items`: a `relative_to` used while building the
            # `suppress(...)` call is not guarded by it — same distinction as
            # `If.test` versus `If.body`.
            if any(
                _is_suppress_call(item.context_expr, suppress_names)
                and _names_catch_value_error(item.context_expr.args)  # type: ignore[attr-defined]
                for item in parent.items
            ):
                return "relative_to guarded by suppress() that catches ValueError"
        if isinstance(parent, (ast.If, ast.While, ast.Assert, ast.IfExp)):
            if _in_field(parent, child, "test"):
                return "relative_to used as a condition"
        if isinstance(parent, ast.BoolOp):
            return "relative_to used in a boolean expression"
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
            return "relative_to negated as a boolean"
        if isinstance(parent, ast.comprehension) and _in_field(parent, child, "ifs"):
            return "relative_to used as a comprehension filter"
        child, parent = parent, parents.get(parent)
    return None


def find_containment_relative_to(source: str) -> list[tuple[int, str, str]]:
    """Return ``(lineno, call text, reason)`` per containment-deciding call.

    Returns ``[]`` for source that cannot be parsed, so one malformed file never
    fails the whole guard — a broken file fails its own tests.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover — a broken file fails its own tests
        return []
    parents = _parent_map(tree)
    suppress_names = _suppress_names(tree)
    lines = source.splitlines()
    offenders: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _CONTAINMENT_METHODS:
            continue
        reason = _containment_reason(node, parents, suppress_names)
        if reason is None:
            continue
        # The call's own source, wrapping collapsed: `lines[lineno - 1]` is
        # `rel = (` for a formatter-split call and would report nothing useful.
        segment = ast.get_source_segment(source, node) or lines[node.lineno - 1]
        text = " ".join(segment.split())
        offenders.append((node.lineno, text[:120], reason))
    return offenders


def _scan_src() -> list[tuple[str, int, str, str]]:
    """Every containment-deciding call under ``src/soup_cli``, allowlist aside."""
    hits: list[tuple[str, int, str, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        source = path.read_text(encoding="utf-8", errors="replace")
        for lineno, text, reason in find_containment_relative_to(source):
            hits.append((rel, lineno, text, reason))
    return hits


class TestNoUnjustifiedContainmentRelativeToInSrc:
    def test_src_routes_containment_through_the_commonpath_helper(self):
        offenders = [
            f"{rel}:{lineno}: {text}  [{reason}]"
            for rel, lineno, text, reason in _scan_src()
            if (rel, lineno) not in ALLOWLIST
        ]
        assert not offenders, (
            "Containment decided by `relative_to` / `is_relative_to`. One of "
            "the two paths can carry a Windows 8.3 short name while the other "
            "carries the long form, so this denies paths that ARE inside the "
            "base (AGENTS.md: realpath + commonpath). Use "
            "`soup_cli.utils.paths.is_under(path, base)` / `is_under_cwd(path)`. "
            "If the call is genuinely not a gate, add it to ALLOWLIST in this "
            "file with the reason.\n  " + "\n  ".join(offenders)
        )

    def test_the_scan_actually_covers_src(self):
        """CONTROL for the control: a scan that stopped discovering modules
        would pass the guard above by looking at nothing."""
        scanned = list(SRC.rglob("*.py"))
        assert len(scanned) >= 400, f"only {len(scanned)} modules scanned"
        # And the walk must still reach the two known call sites.
        assert {(rel, lineno) for rel, lineno, _text, _reason in _scan_src()} == set(
            ALLOWLIST
        )

    def test_the_allowlist_has_no_dead_entries(self):
        """An allowlist entry that no longer matches a live call site is a hole:
        the code moved, the reason was never re-checked, and the next reader
        assumes it was. Fail loudly instead."""
        live = {(rel, lineno) for rel, lineno, _text, _reason in _scan_src()}
        dead = sorted(set(ALLOWLIST) - live)
        assert not dead, (
            "ALLOWLIST entries no longer point at a containment call — the code "
            "moved or was fixed. Re-verify the site and update the line number, "
            "or drop the entry: " + ", ".join(f"{rel}:{line}" for rel, line in dead)
        )

    def test_every_allowlist_entry_carries_a_reason(self):
        for key, reason in ALLOWLIST.items():
            assert len(reason.split()) >= 10, f"{key} has a placeholder reason"


class TestTheScannerCanActuallyFail:
    """The scan's only findings are allowlisted. That is meaningful only if the
    scanner is shown able to find something it has not been told about."""

    #: The exact idiom the seven migrated files used to carry.
    CWD_IDIOM = '''
def _validate(output_path):
    cwd = Path.cwd()
    try:
        output_path.resolve().relative_to(cwd)
    except ValueError:
        raise ValueError("output must stay under cwd")
    return output_path
'''

    #: Same bug, different variable name — invisible to a `".relative_to(cwd)"`
    #: text match, which is the reason this file exists.
    OTHER_NAME = '''
def _validate(target, base):
    try:
        target.resolve().relative_to(base)
    except ValueError:
        return False
    return True
'''

    #: Same bug, split across lines by the formatter — also invisible to a
    #: single-line text match.
    WRAPPED = '''
def _validate(target, root):
    try:
        relative = target.resolve().relative_to(
            root.resolve(),
        )
    except ValueError:
        return None
    return relative
'''

    def test_it_catches_the_cwd_idiom(self):
        found = find_containment_relative_to(textwrap.dedent(self.CWD_IDIOM))
        assert len(found) == 1, found
        assert "relative_to(cwd)" in found[0][1]

    def test_it_catches_a_differently_named_base(self):
        found = find_containment_relative_to(textwrap.dedent(self.OTHER_NAME))
        assert len(found) == 1, found
        assert "relative_to(base)" in found[0][1]

    def test_it_catches_a_call_split_across_lines(self):
        found = find_containment_relative_to(textwrap.dedent(self.WRAPPED))
        assert len(found) == 1, found
        # The reported text is the whole call, not `relative = (`.
        assert "relative_to( root.resolve(), )" in found[0][1]

    def test_it_catches_is_relative_to_in_a_condition(self):
        source = '''
def _validate(target, base):
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError("escape")
'''
        found = find_containment_relative_to(source)
        assert len(found) == 1, found
        assert "is_relative_to" in found[0][1]

    def test_it_catches_a_bare_is_relative_to_return(self):
        """`is_relative_to` is flagged wherever it appears — it answers exactly
        the containment question and nothing else."""
        found = find_containment_relative_to(
            "def under(p, b):\n    return p.is_relative_to(b)\n"
        )
        assert len(found) == 1, found

    def test_it_catches_a_comprehension_filter(self):
        source = '''
def _keep(paths, base):
    return [p for p in paths if p.resolve().is_relative_to(base)]
'''
        assert len(find_containment_relative_to(source)) == 1

    def test_a_bare_except_still_counts(self):
        """`except:` and `except Exception:` catch ValueError as well, so the
        containment decision is the same one — only the blast radius differs."""
        source = '''
def _validate(target, base):
    try:
        target.relative_to(base)
    except Exception:
        return False
    return True
'''
        assert len(find_containment_relative_to(source)) == 1

    # ── controls: the shapes that must NOT be flagged ──

    def test_the_migrated_helper_is_accepted(self):
        """CONTROL. The repaired form must be clean, or the guard would demand
        rewriting the fix it exists to protect."""
        source = '''
def _validate(output_path):
    from soup_cli.utils.paths import is_under_cwd

    if not is_under_cwd(output_path):
        raise ValueError("output must stay under cwd")
    return output_path
'''
        assert find_containment_relative_to(source) == []

    def test_display_only_relative_to_is_not_flagged(self):
        """CONTROL. Shortening a path for a table is the method's normal use and
        decides nothing. A guard that flagged it would be noise, and a noisy
        guard gets deleted."""
        source = '''
def _label(path, root):
    rel = path.relative_to(root)
    return str(rel)
'''
        assert find_containment_relative_to(source) == []

    def test_relative_to_inside_a_branch_body_is_not_flagged(self):
        """CONTROL. `If.body` is not `If.test`: sitting inside a branch does not
        make a call a condition."""
        source = '''
def _label(path, root, shorten):
    if shorten:
        return str(path.relative_to(root))
    return str(path)
'''
        assert find_containment_relative_to(source) == []

    def test_a_try_catching_only_os_error_is_not_flagged(self):
        """CONTROL. `except OSError` cannot catch the ValueError that
        `relative_to` raises, so that handler is not the containment decision."""
        source = '''
def _label(path, root):
    try:
        return str(path.relative_to(root))
    except OSError:
        return str(path)
'''
        assert find_containment_relative_to(source) == []

    def test_an_enclosing_function_boundary_stops_the_walk(self):
        """CONTROL. A `try` around a *definition* does not guard calls in the
        body — those run later, at the caller's mercy."""
        source = '''
try:
    def _label(path, root):
        return str(path.relative_to(root))
except ValueError:
    _label = None
'''
        assert find_containment_relative_to(source) == []

    def test_an_unrelated_method_named_similarly_is_not_flagged(self):
        source = "def f(a, b):\n    return a.relative_path(b)\n"
        assert find_containment_relative_to(source) == []

    def test_unparseable_source_does_not_explode(self):
        assert find_containment_relative_to("def broken(:\n") == []


class TestSuppressGuardedRelativeTo:
    """`with suppress(ValueError):` is `try/except ValueError:` with a shorter
    spelling — the ratchet has to see both or the idiom just moves."""

    def test_it_catches_the_issue_example(self):
        source = '''
from contextlib import suppress

def _validate(target, base):
    with suppress(ValueError):
        target.resolve().relative_to(base)
        return True
    return False
'''
        found = find_containment_relative_to(source)
        assert len(found) == 1, found
        assert "relative_to(base)" in found[0][1]
        assert found[0][0] == 6, found

    def test_it_catches_the_attribute_form(self):
        source = '''
import contextlib

def _validate(target, base):
    with contextlib.suppress(ValueError):
        target.relative_to(base)
'''
        assert len(find_containment_relative_to(source)) == 1

    def test_it_catches_a_flat_multi_argument_suppress(self):
        source = '''
from contextlib import suppress

def _validate(target, base):
    with suppress(OSError, ValueError):
        target.relative_to(base)
'''
        assert len(find_containment_relative_to(source)) == 1

    def test_it_catches_a_nested_tuple_argument(self):
        source = '''
from contextlib import suppress

def _validate(target, base):
    with suppress((OSError, ValueError)):
        target.relative_to(base)
'''
        assert len(find_containment_relative_to(source)) == 1

    def test_it_catches_an_aliased_import(self):
        source = '''
from contextlib import suppress as quiet

def _validate(target, base):
    with quiet(ValueError):
        target.relative_to(base)
'''
        assert len(find_containment_relative_to(source)) == 1

    def test_it_catches_an_aliased_module(self):
        source = '''
import contextlib as cl

def _validate(target, base):
    with cl.suppress(ValueError):
        target.relative_to(base)
'''
        assert len(find_containment_relative_to(source)) == 1

    # ── controls ──

    def test_suppressing_only_os_error_is_not_flagged(self):
        """CONTROL. OSError cannot catch the ValueError `relative_to` raises."""
        source = '''
from contextlib import suppress

def _label(path, root):
    with suppress(OSError):
        return str(path.relative_to(root))
'''
        assert find_containment_relative_to(source) == []

    def test_suppress_around_unrelated_code_is_not_flagged(self):
        source = '''
from contextlib import suppress

def _clean(path):
    with suppress(ValueError):
        path.unlink()
'''
        assert find_containment_relative_to(source) == []

    def test_relative_to_outside_the_with_body_is_not_flagged(self):
        """CONTROL. Only the `body` is guarded — not the `context_expr` that
        builds the `suppress(...)` call, and not code after the block."""
        source = '''
from contextlib import suppress

def _label(path, root, pick):
    with suppress(ValueError, pick(path.relative_to(root))):
        pass
    return str(path.relative_to(root))
'''
        assert find_containment_relative_to(source) == []

    def test_a_nested_function_boundary_stops_the_walk(self):
        """CONTROL. A `with` around a *definition* does not guard calls in the
        body — those run later, at the caller's mercy."""
        source = '''
from contextlib import suppress

with suppress(ValueError):
    def _label(path, root):
        return str(path.relative_to(root))
'''
        assert find_containment_relative_to(source) == []
