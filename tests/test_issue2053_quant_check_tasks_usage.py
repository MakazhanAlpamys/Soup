"""#2053: ``soup eval quant-check`` refuses an unparseable ``--tasks`` file with
exit 3 (usage error), reporting the loader's message *before* any model is
loaded, for every malformed-input case. On ``main`` before the fix the same
inputs escaped as ``Error: ValueError: ...`` with exit 1 after both models had
already been loaded. Related pins in the same contract:

- the loader's message goes through ``for_terminal``: file-written text is
  freed of control bytes and its Rich markup brackets render literally,
- the CLI parses the file exactly once and hands the parsed list to
  ``run_quant_check(tasks=...)``,
- a ``--tasks`` path that exists but cannot be read (directory,
  PermissionError, decoding failure) is caught by the same ``except``
  (``FileNotFoundError`` is a subclass of ``OSError``),
- ``run_quant_check(tasks=[])`` and a call with neither task source raise
  ``ValueError`` instead of reporting before == after == 0.0 / verdict OK.
"""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.custom import MAX_EVAL_TASKS

from .conftest import strip_ansi

runner = CliRunner()


def _clean(output: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output."""
    return " ".join(strip_ansi(output).split())


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    """cwd = tmp_path with two empty dirs standing in for the models."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "before").mkdir()
    (tmp_path / "after").mkdir()


@pytest.fixture()
def forbid_models(monkeypatch):
    """Fail loudly if make_model_generator runs: --tasks must be refused first."""
    calls = []

    def _boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("model load reached; --tasks should have been refused first")

    monkeypatch.setattr("soup_cli.eval.quant_check.make_model_generator", _boom)
    return calls


def _invoke(tasks_name: str, *extra_args: str) -> object:
    """Invoke ``eval quant-check`` with two stub-model dirs and --allow-stub."""
    return runner.invoke(app, [
        "eval", "quant-check",
        "--before", "before",
        "--after", "after",
        "--tasks", tasks_name,
        "--allow-stub",
        *extra_args,
    ])


BAD_FILE_CASES = [
    pytest.param(
        lambda td: (td / "tasks.txt", b"hello\n"),
        "Expected .jsonl file, got: .txt",
        id="wrong-extension",
    ),
    pytest.param(
        lambda td: (td / "bad.jsonl", b"not json\n"),
        "Invalid JSON on line 1",
        id="invalid-json",
    ),
    pytest.param(
        lambda td: (td / "bad.jsonl", json.dumps({"expected": "4"}).encode() + b"\n"),
        "missing required field 'prompt'",
        id="missing-prompt",
    ),
    pytest.param(
        lambda td: (td / "empty.jsonl", b""),
        "No eval tasks found in",
        id="empty-jsonl",
    ),
    pytest.param(
        lambda td: (td / "toomany.jsonl", b"\n".join(
            json.dumps({"prompt": f"p{i}", "expected": "a"}).encode()
            for i in range(MAX_EVAL_TASKS + 1)
        ) + b"\n"),
        f"exceeds maximum of {MAX_EVAL_TASKS} tasks",
        id="over-max-eval-tasks",
    ),
    pytest.param(
        lambda td: (td / "bom.jsonl", b"\xef\xbb\xbf" + json.dumps(
            {"prompt": "p", "expected": "a"},
        ).encode() + b"\n"),
        "Invalid JSON on line 1",
        id="utf8-bom",
    ),
    pytest.param(
        lambda td: (td / "utf16.jsonl", "hi\n".encode("utf-16")),
        "codec can't decode",
        id="utf16-file",
    ),
    pytest.param(
        lambda td: (td / "nonobject.jsonl", b"[1]\n"),
        "expected JSON object, got list",
        id="non-object-row",
    ),
    pytest.param(
        lambda td: (td / "badscoring.jsonl", json.dumps({
            "prompt": "p", "expected": "a", "scoring": "bogus",
        }).encode() + b"\n"),
        "invalid scoring 'bogus'",
        id="invalid-scoring",
    ),
    pytest.param(
        lambda td: (td / "blank.jsonl", b"   \n\n"),
        "No eval tasks found in",
        id="blank-lines-only",
    ),
]


@pytest.mark.parametrize(("make_file", "message"), BAD_FILE_CASES)
def test_unparseable_tasks_is_a_usage_error_before_any_model_load(
    cli_env, forbid_models, tmp_path, make_file, message,
):
    """Every malformed --tasks input exits 3 with the loader's message."""
    path, payload = make_file(tmp_path)
    path.write_bytes(payload)

    result = _invoke(path.name)

    assert result.exit_code == 3, (_clean(result.output), repr(result.exception))
    assert message in _clean(result.output)
    assert forbid_models == []  # refused before a single model was touched


def test_missing_tasks_file_still_exits_3(cli_env, tmp_path):
    """The up-front existence check keeps its message and exit code."""
    result = _invoke("missing.jsonl")
    assert result.exit_code == 3, (_clean(result.output), repr(result.exception))
    assert "--tasks not found: missing.jsonl" in _clean(result.output)


def test_tasks_outside_cwd_is_refused_before_the_file_is_read(
    cli_env, forbid_models, tmp_path_factory,
):
    """The containment refusal wins over the loader: an outside-cwd --tasks
    path is named and refused with exit 3 and its contents are never parsed,
    whatever a row's scoring says ('SECRET' must not surface anywhere)."""
    outside = tmp_path_factory.mktemp("outside") / "refusal.jsonl"
    outside.write_text(
        json.dumps({"prompt": "p", "expected": "a", "scoring": "SECRET"}),
        encoding="utf-8",
    )

    result = _invoke(str(outside))

    assert result.exit_code == 3, (_clean(result.output), repr(result.exception))
    assert "is outside cwd - refusing" in _clean(result.output)
    assert "SECRET" not in _clean(result.output)
    assert forbid_models == []


def test_loader_message_is_terminal_safe(cli_env, forbid_models, tmp_path, monkeypatch):
    """File-written hostile text never reaches the terminal raw.

    ``for_terminal`` must strip the OSC escape and BEL from the loader's
    message while leaving the markup brackets as literal text: plain
    ``str()`` would keep the control bytes, ``rich.markup.escape`` would
    keep both.
    """
    buf = io.StringIO()
    monkeypatch.setattr(
        "soup_cli.commands.eval.console",
        Console(file=buf, color_system=None, width=400),
    )
    hostile = "[bold red]x" + chr(27) + "]0;t" + chr(7)
    (tmp_path / "tasks.jsonl").write_text(
        json.dumps({"prompt": "p", "scoring": hostile}) + "\n", encoding="utf-8",
    )

    result = _invoke("tasks.jsonl")

    out = buf.getvalue()
    assert result.exit_code == 3, (out, repr(result.exception))
    assert chr(27) not in out and chr(7) not in out
    assert "[bold red]x" in out
    assert forbid_models == []


def test_loader_message_path_markup_is_shown_literally(
    cli_env, forbid_models, tmp_path, monkeypatch,
):
    """A markup-named --tasks path is printed as text, not rendered or escaped."""
    buf = io.StringIO()
    monkeypatch.setattr(
        "soup_cli.commands.eval.console",
        Console(file=buf, color_system=None, width=400),
    )
    (tmp_path / "a[red]b.jsonl").write_bytes(b"")

    result = _invoke("a[red]b.jsonl")

    assert result.exit_code == 3, (buf.getvalue(), repr(result.exception))
    assert "No eval tasks found in a[red]b.jsonl" in buf.getvalue()
    assert forbid_models == []


def test_loader_file_not_found_is_a_usage_error(cli_env, forbid_models, tmp_path, monkeypatch):
    """FileNotFoundError from the loader (file deleted between the checks)
    lands in the same usage-error handler (#2053's FileNotFoundError edge)."""
    import soup_cli.eval.custom as custom

    def gone(path):
        raise FileNotFoundError(f"Eval tasks file not found: {path}")

    monkeypatch.setattr(custom, "load_eval_tasks", gone)
    (tmp_path / "tasks.jsonl").write_text("{}\n", encoding="utf-8")

    result = _invoke("tasks.jsonl")

    assert result.exit_code == 3, (_clean(result.output), repr(result.exception))
    assert "Eval tasks file not found: tasks.jsonl" in _clean(result.output)
    assert forbid_models == []


def test_unopenable_tasks_path_is_a_usage_error(cli_env, forbid_models, tmp_path):
    """A --tasks path that exists but cannot be opened: directory, unreadable
    file, or decoding failure — all OSError children on either platform —
    exit 3 with the loader's message, never escape as exit 1."""
    (tmp_path / "d.jsonl").mkdir()

    result = _invoke("d.jsonl")

    assert result.exit_code == 3, (_clean(result.output), repr(result.exception))
    assert "Cannot load --tasks:" in _clean(result.output)
    assert "d.jsonl" in _clean(result.output)
    assert forbid_models == []


def test_tasks_file_is_parsed_exactly_once(cli_env, tmp_path, monkeypatch):
    """The CLI pre-parses --tasks and hands the list down: the loader runs once."""
    import soup_cli.eval.custom as custom

    real, calls = custom.load_eval_tasks, []

    def counting(path):
        calls.append(str(path))
        return real(path)

    monkeypatch.setattr(custom, "load_eval_tasks", counting)
    monkeypatch.setattr(
        "soup_cli.eval.quant_check.make_model_generator",
        lambda path, **kwargs: (lambda prompt: "4"),
    )
    (tmp_path / "tasks.jsonl").write_text(
        json.dumps({"prompt": "2+2=", "expected": "4"}) + "\n",
        encoding="utf-8",
    )

    result = _invoke("tasks.jsonl", "--format", "json")

    assert result.exit_code == 0, (_clean(result.output), repr(result.exception))
    assert calls == ["tasks.jsonl"]


def test_valid_tasks_reach_the_scorer(cli_env, tmp_path, monkeypatch):
    """A valid file runs end to end on a fake generator: the parsed tasks
    score 1.0 on both sides and --allow-stub never engages (no stub key)."""
    monkeypatch.setattr(
        "soup_cli.eval.quant_check.make_model_generator",
        lambda path, **kwargs: (lambda prompt: "4"),
    )
    (tmp_path / "tasks.jsonl").write_text(
        json.dumps({"prompt": "2+2=", "expected": "4"}) + "\n",
        encoding="utf-8",
    )

    result = _invoke("tasks.jsonl", "--format", "json")

    assert result.exit_code == 0, (_clean(result.output), repr(result.exception))
    payload = json.loads(_clean(result.output))
    assert payload["rows"][0]["before"] == 1.0
    assert payload["rows"][0]["after"] == 1.0
    assert "stub" not in payload


def test_run_quant_check_uses_preparsed_tasks_without_rereading(tmp_path, monkeypatch):
    """run_quant_check accepts the pre-parsed list and never re-opens the file."""
    from soup_cli.eval.custom import load_eval_tasks
    from soup_cli.eval.quant_check import run_quant_check

    monkeypatch.chdir(tmp_path)
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        '{"prompt": "2+2=", "expected": "4", "scoring": "exact"}\n',
        encoding="utf-8",
    )
    tasks = load_eval_tasks(str(tasks_file))
    tasks_file.unlink()  # gone — scoring must not depend on re-reading it

    def gen(prompt):
        return "4"

    result = run_quant_check(before_gen=gen, after_gen=gen, tasks=tasks)
    row = result.rows[0]
    assert row.before == 1.0 and row.after == 1.0 and row.verdict == "OK"


def test_run_quant_check_keeps_tasks_file_working(tmp_path, monkeypatch):
    """The documented tasks_file path still loads and scores as before."""
    from soup_cli.eval.quant_check import run_quant_check

    monkeypatch.chdir(tmp_path)
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        '{"prompt": "3+3=", "expected": "6", "scoring": "exact"}\n',
        encoding="utf-8",
    )

    def gen(prompt):
        return "6"

    result = run_quant_check(
        before_gen=gen, after_gen=gen, tasks_file=str(tasks_file),
    )
    row = result.rows[0]
    assert row.before == 1.0 and row.after == 1.0 and row.verdict == "OK"


def test_run_quant_check_prefers_preparsed_tasks_over_tasks_file(tmp_path):
    """When both sources are given, the pre-parsed list wins: tasks_file is
    never touched (the docstring's precedence contract)."""
    from soup_cli.eval.custom import load_eval_tasks
    from soup_cli.eval.quant_check import run_quant_check

    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        '{"prompt": "2+2=", "expected": "4", "scoring": "exact"}\n',
        encoding="utf-8",
    )
    tasks = load_eval_tasks(str(tasks_file))

    def gen(prompt):
        return "4"

    row = run_quant_check(
        before_gen=gen, after_gen=gen,
        tasks=tasks, tasks_file="missing.jsonl",  # would raise if it were read
    ).rows[0]
    assert row.before == 1.0 and row.after == 1.0


def test_run_quant_check_rejects_empty_task_list():
    """tasks=[] must not report before == after == 0.0 with verdict OK: a
    pass that scored nothing used to be impossible (the loader refuses an
    empty file), so the new parameter needs the guard explicitly."""
    from soup_cli.eval.quant_check import run_quant_check

    def gen(prompt):
        return "4"

    with pytest.raises(ValueError, match="No eval tasks"):
        run_quant_check(before_gen=gen, after_gen=gen, tasks=[])


def test_run_quant_check_requires_a_task_source():
    """Neither tasks nor tasks_file is a caller bug: a loud ValueError beats
    'Expected .jsonl file, got: '."""
    from soup_cli.eval.quant_check import run_quant_check

    def gen(prompt):
        return "4"

    with pytest.raises(ValueError, match="run_quant_check needs tasks or tasks_file"):
        run_quant_check(before_gen=gen, after_gen=gen)
