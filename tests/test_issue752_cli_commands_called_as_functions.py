"""#752 — a typer command called as a plain Python function gets OptionInfo.

Typer fills a command's parameters only when *typer* invokes it. Called
directly from another module, any parameter the caller does not pass keeps its
literal default — a ``typer.models.OptionInfo`` object, not the declared value.

``OptionInfo`` is truthy, so ``if output:`` fires and ``output or "x"`` yields
the OptionInfo. In ``soup eval auto`` that reached ``write_eval_json`` and
raised ``TypeError: expected str, bytes or os.PathLike object, not OptionInfo``
*after* the eval had run and its results were saved — an uncaught traceback in
place of a result.

Three things are pinned here, because getting any of them wrong makes the guard
worse than nothing:

1. **The behaviour, not the call shape.** ``soup eval auto`` must complete. A
   test that only asserted "the call site lists five keywords" would pass
   against a call site that lists the wrong five.
2. **A method call is not a command call.** ``model.train()`` and
   ``tracker.list_runs()`` share their names with real typer commands. A guard
   that matched on the bare name flagged four of those, and a guard people mute
   is worse than no guard — so the false-positive shape has its own test.
3. **The guard must be able to fail.** It is asserted red against a synthetic
   leak, not merely observed green against today's tree.

No test here touches the network, a GPU, or MLX.
"""

from __future__ import annotations

import ast
import json
import pathlib
import textwrap

import pytest
import typer

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli"


# --------------------------------------------------------------------------
# the AST walker the guard is built on
# --------------------------------------------------------------------------

def _is_typer_default(node: ast.expr | None) -> bool:
    """Is this default a ``typer.Option(...)`` / ``typer.Argument(...)`` call?"""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return False
    return name in {"Option", "Argument"}


def _typer_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Parameter names of ``fn`` whose default is a typer Option/Argument."""
    args = fn.args
    params = args.posonlyargs + args.args + args.kwonlyargs
    defaults = (
        [None] * (len(args.posonlyargs) + len(args.args) - len(args.defaults))
        + list(args.defaults)
        + list(args.kw_defaults)
    )
    if len(defaults) != len(params):
        return []
    return [p.arg for p, d in zip(params, defaults) if _is_typer_default(d)]


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    """Dotted module name for a file under ``root``'s parent package."""
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def find_optioninfo_leaks(root: pathlib.Path) -> list[tuple[str, int, str, list[str]]]:
    """Bare-name calls to a typer command that leave typer params unfilled.

    The callee is resolved to a *declaration* — one in the calling module, or
    one named by a ``from X import name`` in it. Keying on the bare name alone
    merged every ``main()`` in the tree into one entry and flagged
    ``autodistill/mlx_worker.py``'s argparse ``main``, which declares no typer
    parameters at all.

    Returns ``(module, lineno, callee, leaked_params)`` tuples.
    """
    trees: dict[pathlib.Path, ast.Module] = {}
    declared: dict[str, dict[str, list[str]]] = {}

    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - the tree parses today
            continue
        trees[path] = tree
        mod = _module_name(path, root)
        declared[mod] = {
            node.name: _typer_params(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    leaks: list[tuple[str, int, str, list[str]]] = []
    for path, tree in trees.items():
        mod = _module_name(path, root)
        local = declared.get(mod, {})
        imported: dict[str, tuple[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    imported[alias.asname or alias.name] = (node.module, alias.name)

        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            # Bare-name calls only. `model.train()` is a method that happens to
            # share a name with the `train` command; flagging it is constraint 2.
            if not isinstance(call.func, ast.Name):
                continue
            name = call.func.id
            if name in local:
                params = local[name]
            elif name in imported:
                src_mod, orig = imported[name]
                params = declared.get(src_mod, {}).get(orig)
                if params is None:
                    continue
            else:
                continue
            if not params:
                continue
            # A call with *args/**kwargs cannot be judged statically.
            if call.args or any(k.arg is None for k in call.keywords):
                continue
            passed = {k.arg for k in call.keywords}
            missing = [p for p in params if p not in passed]
            if missing:
                leaks.append((path.name, call.lineno, name, missing))
    return leaks


# --------------------------------------------------------------------------
# 1. the mechanism
# --------------------------------------------------------------------------

def test_unfilled_typer_parameters_are_truthy_optioninfo():
    """The premise of the whole issue: the default is an object, and it is truthy.

    If typer ever changes this, the guard below is obsolete and should go.
    """
    import inspect

    from soup_cli.commands.eval import custom

    default = inspect.signature(custom).parameters["output"].default
    assert isinstance(default, typer.models.OptionInfo)
    assert bool(default) is True, "OptionInfo must be truthy for this bug to exist"


# --------------------------------------------------------------------------
# 2. the behaviour — soup eval auto must complete
# --------------------------------------------------------------------------

@pytest.fixture
def eval_auto_workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    """A config whose eval leg runs custom tasks and nothing else."""
    output = tmp_path / "output"
    output.mkdir()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"prompt": "2+2?", "expected": "4"}) + "\n", encoding="utf-8"
    )
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps({"instruction": "hi", "output": "hello"}) + "\n", encoding="utf-8"
    )
    config = tmp_path / "soup.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            base: some-model
            task: sft
            data:
              train: {train}
              format: alpaca
            training:
              epochs: 1
            output: {output}
            eval:
              custom_tasks: {tasks}
            """
        ),
        encoding="utf-8",
    )
    return config


def _stub_eval_internals(monkeypatch):
    """Replace only the model load and the tracker write."""
    import soup_cli.commands.eval as ce
    import soup_cli.eval.custom as ec

    monkeypatch.setattr(ec, "_create_default_generator", lambda path: (lambda p: "4"))
    monkeypatch.setattr(ce, "_save_custom_results", lambda *a, **k: None)


def test_soup_eval_auto_completes_instead_of_raising_typeerror(
    eval_auto_workspace, monkeypatch, tmp_path
):
    """The reported bug: the eval succeeds, then the command dies."""
    from soup_cli.commands.eval import auto

    _stub_eval_internals(monkeypatch)
    monkeypatch.chdir(tmp_path)

    # Every typer parameter passed explicitly — the rule this issue is about
    # applies to the test's own call too.
    auto(
        config=str(eval_auto_workspace),
        benchmarks=None,
        custom_tasks=None,
        trust_remote_code=False,
    )


def test_soup_eval_auto_writes_no_eval_json_when_none_was_requested(
    eval_auto_workspace, monkeypatch, tmp_path
):
    """The leak also made the output block run at all. It must not."""
    from soup_cli.commands.eval import auto

    _stub_eval_internals(monkeypatch)
    monkeypatch.chdir(tmp_path)

    auto(
        config=str(eval_auto_workspace),
        benchmarks=None,
        custom_tasks=None,
        trust_remote_code=False,
    )

    assert not (tmp_path / "eval_results.json").exists(), (
        "no --output was requested, so no eval JSON may be written"
    )


# --------------------------------------------------------------------------
# 3. the call sites, by what they pass rather than by their source text
# --------------------------------------------------------------------------

def _recording_custom(recorded: dict):
    def _custom(**kwargs):
        recorded.update(kwargs)
    return _custom


def test_auto_passes_every_typer_parameter_of_custom(
    eval_auto_workspace, monkeypatch, tmp_path
):
    import inspect

    import soup_cli.commands.eval as ce

    expected = set(inspect.signature(ce.custom).parameters)
    recorded: dict = {}
    monkeypatch.setattr(ce, "custom", _recording_custom(recorded))
    monkeypatch.chdir(tmp_path)

    ce.auto(
        config=str(eval_auto_workspace),
        benchmarks=None,
        custom_tasks=None,
        trust_remote_code=False,
    )

    assert set(recorded) == expected, (
        f"auto() left {sorted(expected - set(recorded))} to typer's default"
    )


def test_training_callback_passes_every_typer_parameter_of_custom(monkeypatch):
    """The second call site — mid-training auto-eval (monitoring/callback.py)."""
    import inspect

    import soup_cli.commands.eval as ce

    expected = set(inspect.signature(ce.custom).parameters)
    recorded: dict = {}
    monkeypatch.setattr(ce, "custom", _recording_custom(recorded))

    from soup_cli.monitoring.callback import SoupTrainerCallback

    callback = SoupTrainerCallback.__new__(SoupTrainerCallback)
    callback.eval_config = type(
        "EvalCfg",
        (),
        {"auto_eval": True, "benchmarks": [], "custom_tasks": "tasks.jsonl"},
    )()
    callback.output_dir = "out"
    callback.run_id = "run-1"

    callback._run_auto_eval()

    assert set(recorded) == expected, (
        f"the callback left {sorted(expected - set(recorded))} to typer's default"
    )


# --------------------------------------------------------------------------
# 4. the guard
# --------------------------------------------------------------------------

def test_no_typer_command_is_called_with_unfilled_typer_parameters():
    leaks = find_optioninfo_leaks(SRC)
    assert leaks == [], "\n".join(
        f"{m}:{ln} calls {callee}() leaving {', '.join(missing)} "
        f"bound to OptionInfo"
        for m, ln, callee, missing in leaks
    )


def test_the_guard_catches_a_synthetic_leak(tmp_path):
    """Acceptance criterion 1 — demonstrated red, not merely observed green."""
    module = tmp_path / "fake_cmd.py"
    module.write_text(
        textwrap.dedent(
            """\
            import typer

            def widget(
                name: str = typer.Option(..., "--name"),
                fmt: str = typer.Option(None, "--format"),
            ):
                return name, fmt

            def caller():
                return widget(name="x")
            """
        ),
        encoding="utf-8",
    )
    leaks = find_optioninfo_leaks(tmp_path)
    assert [(callee, missing) for _, _, callee, missing in leaks] == [("widget", ["fmt"])]


def test_the_guard_ignores_a_method_call_sharing_a_command_name(tmp_path):
    """Constraint 2 — `model.train()` is not a call to the `train` command.

    Four false positives came from matching an attribute call by its ``.attr``.
    """
    module = tmp_path / "fake_train.py"
    module.write_text(
        textwrap.dedent(
            """\
            import typer

            def train(
                config: str = typer.Option("soup.yaml", "--config"),
                gpus: int = typer.Option(1, "--gpus"),
            ):
                return config, gpus

            def caller(model, wrapper):
                model.train()
                wrapper.train(display=None)
                return train(config="a.yaml", gpus=2)
            """
        ),
        encoding="utf-8",
    )
    assert find_optioninfo_leaks(tmp_path) == []


def test_the_guard_skips_calls_it_cannot_judge(tmp_path):
    """``**kwargs`` and positional args are not statically decidable."""
    module = tmp_path / "fake_star.py"
    module.write_text(
        textwrap.dedent(
            """\
            import typer

            def widget(
                name: str = typer.Option(..., "--name"),
                fmt: str = typer.Option(None, "--format"),
            ):
                return name, fmt

            def caller(opts):
                return widget(**opts)
            """
        ),
        encoding="utf-8",
    )
    assert find_optioninfo_leaks(tmp_path) == []


def test_the_guard_ignores_a_plain_function_sharing_a_command_name(tmp_path):
    """The false positive that keying on the bare name produced.

    ``autodistill/mlx_worker.py`` defines an argparse ``main(argv=None)``. A
    name-keyed guard gave it the typer parameters of an unrelated ``main``
    command elsewhere in the tree and reported fifteen leaked options.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "cli_entry.py").write_text(
        textwrap.dedent(
            """\
            import typer

            def main(
                verbose: bool = typer.Option(False, "--verbose"),
                timeout: int = typer.Option(30, "--timeout"),
            ):
                return verbose, timeout
            """
        ),
        encoding="utf-8",
    )
    (pkg / "worker.py").write_text(
        textwrap.dedent(
            """\
            def main(argv=None):
                return 0

            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ),
        encoding="utf-8",
    )
    assert find_optioninfo_leaks(pkg) == []


def test_the_guard_follows_an_imported_command(tmp_path):
    """The callback reaches ``custom`` through ``from ... import custom``."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "commands.py").write_text(
        textwrap.dedent(
            """\
            import typer

            def widget(
                name: str = typer.Option(..., "--name"),
                fmt: str = typer.Option(None, "--format"),
            ):
                return name, fmt
            """
        ),
        encoding="utf-8",
    )
    (pkg / "caller.py").write_text(
        textwrap.dedent(
            """\
            def go():
                from pkg.commands import widget
                return widget(name="x")
            """
        ),
        encoding="utf-8",
    )
    leaks = find_optioninfo_leaks(pkg)
    assert [(callee, missing) for _, _, callee, missing in leaks] == [("widget", ["fmt"])]
