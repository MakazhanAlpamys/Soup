"""User paths and flags in ``soup data`` render as text, not Rich markup."""

import ast
import re
from pathlib import Path

import pytest
from rich.errors import MarkupError
from typer.testing import CliRunner

from soup_cli.cli import app

ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner()
DATA_PY = Path(__file__).resolve().parents[1] / "src/soup_cli/commands/data.py"

_SAFE_CALLS = frozenset({"for_terminal", "escape", "_escape", "_esc"})
_REPR_CALLS = frozenset({"for_terminal", "escape"})
_ARITH_CALLS = frozenset({"len", "min", "max", "sum", "round", "abs", "int", "float"})
_USER_NAMES = frozenset({
    "file_path", "input_file", "path", "file", "logs", "output", "out_path", "out_dir",
    "fmt", "to", "format", "final_name", "final_path", "query", "dataset_id", "exc",
    "strategy", "config_path", "output_dir", "input_path", "hf_dataset", "reason",
    "src_fmt", "signal", "signals", "embed_model", "perplexity_model", "sort", "split",
    "hub_canonical", "train_display", "target", "written", "provider", "preview",
    "train_path", "val_path", "test_path",
})
_USER_ATTRS = frozenset({
    ("file_path", "name"),
    ("input_resolved", "name"),
    ("cfg", "base"),
    ("cfg", "quality_filter"),
    ("bundle", "name"),
})


def _plain(output: str) -> str:
    return ANSI_SGR.sub("", output)


def _no_markup_error(result) -> None:
    assert not isinstance(result.exception, MarkupError)


def _safe_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _SAFE_CALLS
    )


def _repr_then_escape(node: ast.AST) -> bool:
    """``for_terminal(repr(...))`` or ``escape(repr(...))`` — quote, then escape."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _REPR_CALLS
        and len(node.args) == 1
        and not node.keywords
    ):
        return False
    arg = node.args[0]
    return (
        isinstance(arg, ast.Call)
        and isinstance(arg.func, ast.Name)
        and arg.func.id == "repr"
        and len(arg.args) == 1
    )


def _arith_operand(node: ast.AST) -> bool:
    if isinstance(node, (ast.Constant, ast.Name, ast.Attribute)):
        return True
    return _is_arith(node)


def _is_arith(node: ast.AST) -> bool:
    if isinstance(node, ast.BinOp):
        return _arith_operand(node.left) and _arith_operand(node.right)
    if isinstance(node, ast.UnaryOp):
        return _arith_operand(node.operand)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _ARITH_CALLS
    ):
        return all(_arith_operand(arg) for arg in node.args) and all(
            _arith_operand(kw.value) for kw in node.keywords
        )
    return False


def _names_inside_safe_calls(expr: ast.AST) -> bool:
    """Every ``Name`` / ``Attribute`` sits inside ``for_terminal`` / ``escape`` / aliases."""
    outside: list[ast.AST] = []

    def visit(node: ast.AST, inside: bool) -> None:
        if _safe_call(node):
            for child in ast.iter_child_nodes(node):
                if child is node.func:
                    continue
                visit(child, True)
            return
        if isinstance(node, (ast.Name, ast.Attribute)) and not inside:
            outside.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(expr, False)
    return not outside


def _is_safe_value(expr: ast.AST) -> bool:
    return _safe_call(expr) or _names_inside_safe_calls(expr) or _is_arith(expr)


def _has_user_text(expr: ast.AST) -> bool:
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in _USER_NAMES:
            return True
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (node.value.id, node.attr) in _USER_ATTRS
        ):
            return True
    return False


def markup_offenders(source: str) -> list[str]:
    """Bare user text inside a ``[...]`` f-string in ``data.py``.

    A field is safe when it is ``for_terminal`` / ``escape`` / ``_escape`` / ``_esc``,
    when every name in it sits inside such a call, or when it is only arithmetic
    (``len`` / ``min`` / ``max`` / ``sum`` / ``round`` / ``abs`` / ``int`` / ``float``).
    A ``!r`` conversion is safe only as ``for_terminal(repr(...))`` or ``escape(repr(...))``.
    """
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        constants = [
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        ]
        if not any("[" in piece for piece in constants):
            continue
        for part in node.values:
            if not isinstance(part, ast.FormattedValue):
                continue
            if part.conversion == 114 and not _repr_then_escape(part.value):
                found.append(f"line {part.lineno}: {ast.unparse(part)}")
                continue
            if _is_safe_value(part.value) or not _has_user_text(part.value):
                continue
            found.append(f"line {part.lineno}: {ast.unparse(part)}")
    return found


@pytest.mark.parametrize("name", ["nope[/]x.jsonl", "nope[red]x.jsonl"])
def test_review_missing_file_renders_markup_as_text(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["data", "review", name])
    assert result.exit_code == 1
    _no_markup_error(result)
    text = _plain(result.output)
    assert name in text
    assert "File not found" in text


@pytest.mark.parametrize("name", ["nope[/]x.jsonl", "nope[red]x.jsonl"])
def test_inspect_missing_file_renders_markup_as_text(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["data", "inspect", name])
    assert result.exit_code == 1
    _no_markup_error(result)
    text = _plain(result.output)
    # ``review`` prints the original str. ``inspect`` prints ``Path``, which
    # rewrites ``/`` to ``\`` on Windows (``nope[\]x.jsonl``).
    assert str(Path(name)) in text
    assert "File not found" in text


def test_validate_missing_markup_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = "nope[red]x.jsonl"
    result = runner.invoke(app, ["data", "validate", name])
    assert result.exit_code == 3
    _no_markup_error(result)
    assert name in _plain(result.output)


def test_validate_unknown_format_renders_markup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.jsonl").write_text('{"instruction": "a", "output": "b"}\n', encoding="utf-8")
    result = runner.invoke(app, ["data", "validate", "x.jsonl", "--format", "x[/]y"])
    assert result.exit_code == 3
    _no_markup_error(result)
    assert "x[/]y" in _plain(result.output)


def test_convert_unknown_to_renders_markup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.jsonl").write_text('{"instruction": "a", "output": "b"}\n', encoding="utf-8")
    result = runner.invoke(app, ["data", "convert", "x.jsonl", "--to", "x[/]y"])
    assert result.exit_code == 1
    _no_markup_error(result)
    assert "x[/]y" in _plain(result.output)


def test_convert_missing_markup_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = "nope[red]in.jsonl"
    result = runner.invoke(app, ["data", "convert", name, "--to", "alpaca"])
    assert result.exit_code == 1
    _no_markup_error(result)
    assert name in _plain(result.output)


@pytest.mark.parametrize("name", ["x[red]y", "x[/]y"])
def test_unregister_missing_name_renders_markup(tmp_path, monkeypatch, name):
    registry = tmp_path / "datasets.json"
    monkeypatch.setattr(
        "soup_cli.utils.registry._default_registry_path",
        lambda: registry,
    )
    result = runner.invoke(app, ["data", "unregister", name])
    assert result.exit_code == 1
    _no_markup_error(result)
    assert name in _plain(result.output)
    assert not registry.exists()


def test_unregister_strips_control_bytes(tmp_path, monkeypatch):
    registry = tmp_path / "datasets.json"
    monkeypatch.setattr(
        "soup_cli.utils.registry._default_registry_path",
        lambda: registry,
    )
    result = runner.invoke(app, ["data", "unregister", "x\x1b]0;t\x07y"])
    assert result.exit_code == 1
    _no_markup_error(result)
    text = _plain(result.output)
    assert "\x1b" not in text
    assert "\x07" not in text
    assert not registry.exists()


@pytest.mark.parametrize("name", ["nope[/]x.pdf", "nope[red]x.pdf"])
def test_ingest_missing_file_renders_markup(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["data", "ingest", name])
    assert result.exit_code == 1
    _no_markup_error(result)
    assert name in _plain(result.output)
    assert not (tmp_path / "ingested.jsonl").exists()


@pytest.mark.parametrize("logs", ["nope[/]x", "nope[red]x"])
def test_from_traces_missing_logs_renders_markup(tmp_path, monkeypatch, logs):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["data", "from-traces", "--logs", logs, "--format", "openai"],
    )
    assert result.exit_code == 1
    _no_markup_error(result)
    assert logs in _plain(result.output)


def test_from_traces_unknown_format_renders_markup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["data", "from-traces", "--logs", "nope.jsonl", "--format", "x[/]y"],
    )
    assert result.exit_code == 1
    _no_markup_error(result)
    assert "x[/]y" in _plain(result.output)


def test_data_py_markup_fields_are_wrapped():
    offenders = markup_offenders(DATA_PY.read_text(encoding="utf-8"))
    assert offenders == []


def test_markup_ratchet_flags_bare_user_text_only():
    assert markup_offenders('msg = f"[red]{file_path}[/]"')
    assert markup_offenders('msg = f"[red]{fmt!r}[/]"')
    assert markup_offenders('msg = f"[red]{for_terminal(file_path)!r}[/]"')
    assert not markup_offenders('msg = f"[red]{for_terminal(file_path)}[/]"')
    assert not markup_offenders('msg = f"[red]{for_terminal(repr(fmt))}[/]"')
    assert not markup_offenders('msg = f"[green]{len(data)} rows[/]"')
    assert not markup_offenders('msg = f"[red]{valid / total:.3f}[/]"')
    assert not markup_offenders('msg = f"[red]{\', \'.join(VALID_FORMATS)}[/]"')


def _flat(output: str) -> str:
    return " ".join(_plain(output).split())


def test_stats_table_title_renders_markup_in_file_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = "a[red]b.jsonl"
    (tmp_path / name).write_text('{"instruction": "a", "output": "b"}\n', encoding="utf-8")
    result = runner.invoke(app, ["data", "stats", name])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Extended Stats: a[red]b.jsonl" in _flat(result.output)


def test_search_table_title_renders_markup_in_query(monkeypatch):
    from types import SimpleNamespace

    hit = SimpleNamespace(id="org/ds", downloads=1, likes=1, tags=[])
    monkeypatch.setattr("soup_cli.commands.data.list_datasets", lambda **_kw: [hit])
    result = runner.invoke(app, ["data", "search", "x[red]y"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "HuggingFace Datasets: 'x[red]y'" in _flat(result.output)


@pytest.mark.parametrize(
    "argv",
    [
        ["data", "review", "x\x1b]0;t\x07y"],
        ["data", "validate", "x\x1b]0;t\x07y"],
        ["data", "ingest", "x\x1b]0;t\x07y"],
        ["data", "from-traces", "--logs", "x\x1b]0;t\x07y", "--format", "openai"],
    ],
    ids=["review", "validate", "ingest", "from-traces"],
)
def test_missing_path_strips_control_bytes(tmp_path, monkeypatch, argv):
    # ``escape`` leaves ESC / BEL in; only ``for_terminal`` strips them.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, argv)
    assert result.exit_code != 0
    text = _plain(result.output)
    assert "\x1b" not in text, (argv, repr(text))
    assert "\x07" not in text, (argv, repr(text))
