"""A typer command called as a plain function must be passed every parameter.

Typer fills a command's parameters only when *typer* invokes it. Called as an
ordinary Python function, any unfilled parameter keeps its literal default —
a `typer.models.OptionInfo`, which is **truthy**. So `if output:` fires when
the caller asked for no output, and `write_eval_json(output)` gets an
OptionInfo and raises `TypeError: expected str, bytes or os.PathLike object,
not OptionInfo`.

That is #752, and it surfaced after `soup eval auto` had already run the eval
and saved the results — a stack trace instead of a clean finish.

The class is what matters more than the two sites: the next parameter added to
a command joins the leak automatically, and the only symptom is a type error
naming an internal typer object. This walks the package and fails on any new
one. Positional arguments are matched to parameters by declaration order, so a
leak passed positionally is caught and the report names only what is unfilled.
"""
from __future__ import annotations

import ast
import pathlib
from typing import NamedTuple

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli"

_TYPER_DEFAULT_FACTORIES = {"Option", "Argument"}


class _Command(NamedTuple):
    positional: list[str]  # every parameter a positional argument can fill, in order
    typer_params: frozenset[str]  # the ones typer would have filled


class _Source(NamedTuple):
    label: str
    module: str
    tree: ast.Module


def _is_typer_default(node: ast.expr | None) -> bool:
    """True for `typer.Option(...)` / `typer.Argument(...)` and bare imports of them."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in _TYPER_DEFAULT_FACTORIES
    if isinstance(func, ast.Name):
        return func.id in _TYPER_DEFAULT_FACTORIES
    return False


def _is_command(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Decorated with something ending in `.command(...)` or `.callback(...)`."""
    for dec in node.decorator_list:
        call = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(call, ast.Attribute) and call.attr in {"command", "callback"}:
            return True
    return False


def _command(node: ast.FunctionDef | ast.AsyncFunctionDef) -> _Command:
    args = node.args
    positional = args.posonlyargs + args.args
    typer_params = {
        arg.arg
        for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults)
        if _is_typer_default(default)
    }
    typer_params |= {
        arg.arg
        for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        if _is_typer_default(default)
    }
    return _Command([arg.arg for arg in positional], frozenset(typer_params))


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_sources() -> list[_Source]:
    sources = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        sources.append(_Source(str(path.relative_to(SRC.parent.parent)), _module_name(path), tree))
    return sources


def _collect_commands(sources: list[_Source]) -> dict[tuple[str, str], _Command]:
    """Every typer command, keyed by (module, function name)."""
    commands: dict[tuple[str, str], _Command] = {}
    for source in sources:
        for node in ast.walk(source.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_command(node):
                command = _command(node)
                if command.typer_params:
                    commands[(source.module, node.name)] = command
    return commands


def _imported_commands(source: _Source, commands) -> dict[str, tuple[str, str]]:
    """Bare names in this file that resolve to a typer command elsewhere.

    Without this the walk flags every local helper that happens to share a name
    with some command in the package -- `version()`, `check()`, `convert()` --
    which is a different bug report every time somebody adds a helper.
    """
    resolved: dict[str, tuple[str, str]] = {}
    for node in ast.walk(source.tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                key = (node.module, alias.name)
                if key in commands:
                    resolved[alias.asname or alias.name] = key
    # A command defined in this very module is callable by bare name too.
    for mod, name in commands:
        if mod == source.module:
            resolved.setdefault(name, (mod, name))
    return resolved


def _unfilled(call: ast.Call, command: _Command) -> set[str] | None:
    """The typer parameters this call leaves unfilled, or None if it cannot tell.

    Positional arguments fill parameters in declaration order, so they are
    matched by index rather than counted: `widget("x")` fills only the first.
    """
    if any(kw.arg is None for kw in call.keywords):  # **kwargs
        return None
    if any(isinstance(arg, ast.Starred) for arg in call.args):  # *args
        return None
    filled = set(command.positional[: len(call.args)])
    filled |= {kw.arg for kw in call.keywords}
    return set(command.typer_params - filled)


def _leaking_calls(sources: list[_Source], commands) -> list[str]:
    findings: list[str] = []
    for source in sources:
        resolved = _imported_commands(source, commands)
        if not resolved:
            continue
        for node in ast.walk(source.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            target = resolved.get(node.func.id)
            if target is None:
                continue
            missing = _unfilled(node, commands[target])
            if missing:
                findings.append(
                    f"{source.label}:{node.lineno} "
                    f"calls {node.func.id}() without {sorted(missing)}"
                )
    return findings


_SYNTHETIC_COMMAND = """
import typer
app = typer.Typer()

@app.command()
def widget(
    name: str = typer.Argument(...),
    fmt: str = typer.Option("text"),
    *,
    verbose: bool = typer.Option(False),
):
    pass

@app.command()
def gadget(ctx: typer.Context, target: str, force: bool = typer.Option(False)):
    pass
"""


def _scan_synthetic(call: str) -> list[str]:
    sources = [
        _Source("pkg/cli.py", "pkg.cli", ast.parse(_SYNTHETIC_COMMAND)),
        _Source("pkg/use.py", "pkg.use", ast.parse(f"from pkg.cli import gadget, widget\n{call}")),
    ]
    return _leaking_calls(sources, _collect_commands(sources))


def test_the_package_declares_typer_commands_at_all():
    # If the walk silently found nothing, the real test below would pass
    # vacuously forever.
    commands = _collect_commands(_package_sources())
    assert len(commands) > 50, f"only found {len(commands)} typer commands; the walk is broken"
    key = ("soup_cli.commands.eval", "custom")
    assert key in commands
    assert {"output", "attach_to_registry"} <= commands[key].typer_params


def test_the_guard_reports_a_synthetic_leak():
    # The collector test above cannot tell a working reporter from one that
    # finds nothing, so prove the reporter can still fail.
    assert _scan_synthetic('widget("x")') == [
        "pkg/use.py:2 calls widget() without ['fmt', 'verbose']"
    ]


def test_positional_arguments_fill_parameters_by_position():
    assert _scan_synthetic('widget("x", "json", verbose=True)') == []
    assert _scan_synthetic('widget("x", fmt="json", verbose=True)') == []
    assert _scan_synthetic('widget(name="x", fmt="json", verbose=True)') == []
    assert _scan_synthetic('widget("x", "json")') == [
        "pkg/use.py:2 calls widget() without ['verbose']"
    ]


def test_leading_plain_parameters_do_not_count_as_filled_typer_ones():
    # Two positional arguments are as many as gadget's typer parameters and
    # more, but they fill ctx and target, not force.
    assert _scan_synthetic("gadget(ctx, 't')") == [
        "pkg/use.py:2 calls gadget() without ['force']"
    ]
    assert _scan_synthetic("gadget(ctx, 't', True)") == []


def test_calls_it_cannot_resolve_are_not_reported():
    assert _scan_synthetic("widget(*args)") == []
    assert _scan_synthetic("widget(**kwargs)") == []


def test_a_partial_positional_call_names_only_the_unfilled_parameters():
    """#752's call with its first two arguments passed positionally."""
    commands = _collect_commands(_package_sources())
    caller = _Source(
        "caller.py",
        "caller",
        ast.parse(
            "from soup_cli.commands.eval import custom\n"
            "custom(tasks_file, str(output_dir))\n"
        ),
    )

    assert _leaking_calls([caller], commands) == [
        "caller.py:2 calls custom() without ['attach_to_registry', 'output', 'run_id']"
    ]


def test_no_typer_command_is_called_with_parameters_left_unfilled():
    sources = _package_sources()
    findings = _leaking_calls(sources, _collect_commands(sources))
    assert not findings, (
        "A typer command is called as a plain function without every "
        "typer-defaulted parameter. The unfilled ones arrive as truthy "
        "OptionInfo objects rather than their documented defaults (#752):\n  "
        + "\n  ".join(findings)
    )
