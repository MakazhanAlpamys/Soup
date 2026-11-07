"""The edit CLI keeps request, result and diff text literal."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import edit
from soup_cli.registry import attach
from soup_cli.utils import edit_diff, edit_governor, knowledge_edit
from tests.conftest import make_cli_runner, strip_ansi

RAW = r"text:warning:[red]\[end"
STORED = RAW + "\x9b\u202e"


@pytest.fixture(autouse=True)
def cwd_and_console(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        edit,
        "console",
        Console(
            width=500,
            height=100,
            force_terminal=True,
            no_color=False,
            color_system="standard",
            legacy_windows=False,
        ),
    )


def run(*args):
    return make_cli_runner().invoke(app, ["edit", *args])


def edit_set(*args, method="rome", governor=False):
    governor_option = ["--no-governor"] if not governor else []
    return run(
        "set",
        "--base",
        "base-" + STORED,
        "--method",
        method,
        "--layer",
        "6",
        "--subject",
        "subject-" + STORED,
        "--target",
        "target-" + STORED,
        *governor_option,
        *args,
    )


def panel_rows(output):
    return [
        line.removeprefix("│ ").removesuffix("│").rstrip()
        for line in strip_ansi(output).splitlines()
        if line.startswith("│")
    ]


def test_plan_preserves_values_labels_and_blank_row():
    result = edit_set("--plan-only")
    assert result.exit_code == 0, (result.output, result.exception)
    rows = panel_rows(result.output)
    assert rows[:6] == [
        "Method: rome",
        f"Base:   base-{RAW}",
        "Layer:  6",
        f"Subject: subject-{RAW}",
        f"Target:  target-{RAW}",
        "",
    ]
    assert rows[6].startswith("Rank-One Model Editing")
    for label in ("Method:", "Base:", "Layer:", "Subject:", "Target:"):
        assert f"\x1b[1m{label}\x1b[0m" in result.output
    assert "\x1b[2mRank-One Model Editing" in result.output


@pytest.mark.parametrize(
    "stage", ["plan", "corpus", "governor", "apply", "attach", "diff", "write"]
)
def test_edit_failures_preserve_literal_details(monkeypatch, stage):
    def fail(*args, **kwargs):
        if stage == "governor":
            raise edit_governor.GovernedEditError(STORED)
        raise ValueError(STORED)

    if stage == "attach":
        directory = Path("edited")
        directory.mkdir()
        (directory / "model.safetensors").touch()
        monkeypatch.setattr(
            knowledge_edit,
            "apply_edit",
            lambda *a, **k: knowledge_edit.EditResult(
                "rome", 6, 1.0, (6,), str(directory), 0.25, 0.75, False
            ),
        )
    module, function, label = {
        "plan": (knowledge_edit, "build_edit_plan", "Invalid edit request:"),
        "corpus": (edit, "_load_cov_corpus", "Cannot load --cov-corpus:"),
        "governor": (knowledge_edit, "apply_edit", "Edit refused by governor:"),
        "apply": (knowledge_edit, "apply_edit", "Edit failed:"),
        "attach": (attach, "attach_artifact", "Error:"),
        "diff": (edit_diff, "build_diff_report", "Cannot build diff:"),
        "write": (edit_diff, "write_diff_report", "Cannot write diff:"),
    }[stage]
    monkeypatch.setattr(module, function, fail)
    if stage in {"diff", "write"}:
        result = run("diff", "before", "after", "-o", "out.json")
    else:
        extra = {
            "corpus": ["--cov-corpus", "corpus.txt"],
            "attach": ["-o", "edited", "--registry-id", "entry"],
        }.get(stage, [])
        result = edit_set(*extra)

    assert result.exit_code == (1 if stage == "attach" else 2), (result.output, result.exception)
    detail = "could not attach to registry: " if stage == "attach" else ""
    assert f"\x1b[31m{label}\x1b[0m {detail}{RAW}" in result.output
    assert not Path("out.json").exists()


def test_governor_failure_warns_and_keeps_edit_going(monkeypatch):
    def fail(path):
        raise OSError(STORED)

    def apply(plan, **kwargs):
        assert kwargs["governor"] is None
        return knowledge_edit.EditResult("rome", 6, 1.0, (6,), None, 0.25, 0.75, False)

    monkeypatch.setattr(edit_governor, "EditGovernorStore", fail)
    monkeypatch.setattr(knowledge_edit, "apply_edit", apply)
    result = edit_set(governor=True)
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"\x1b[2mGovernor disabled ({RAW}).\x1b[0m" in result.output


def test_edit_result_displays_stored_governor_verdict_literally(monkeypatch):
    governor = edit_governor.EditGovernor("base", edit_count=3, last_verdict=STORED)
    store = SimpleNamespace(close=lambda: None)
    saved = []
    monkeypatch.setattr(edit_governor, "EditGovernorStore", lambda path: store)
    monkeypatch.setattr(edit_governor, "load_governor", lambda store, base: governor)
    monkeypatch.setattr(
        edit_governor, "save_governor", lambda store, governor: saved.append(governor)
    )
    monkeypatch.setattr(
        knowledge_edit,
        "apply_edit",
        lambda *a, **k: knowledge_edit.EditResult("rome", 6, 1.0, (6,), None, 0.25, 0.75, False),
    )
    result = edit_set(governor=True)
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"\x1b[1mGovernor:\x1b[0m count=3 verdict={RAW} " in result.output
    assert "\x9b" not in result.output and "\u202e" not in result.output
    assert saved == [governor]
    assert governor.last_verdict == STORED


def test_saved_result_and_attach_use_original_values(monkeypatch):
    path = Path("edited-[weights]\x9b\u202e")
    path.mkdir()
    weight = path / "model.safetensors"
    weight.touch()
    called = []

    def apply(plan, *, output_dir, **kwargs):
        assert (plan.base, plan.subject, plan.target) == (
            "base-" + STORED,
            "subject-" + STORED,
            "target-" + STORED,
        )
        assert output_dir == str(path)
        return knowledge_edit.EditResult("rome", 6, 1.0, (6,), output_dir, 0.25, 0.75, False)

    def record(entry_id, *, path, kind):
        called.append((entry_id, path, kind))

    monkeypatch.setattr(knowledge_edit, "apply_edit", apply)
    monkeypatch.setattr(attach, "attach_artifact", record)
    result = edit_set("-o", str(path), "--registry-id", STORED)
    assert result.exit_code == 0, (result.output, result.exception)
    shown = str(path).replace("\x9b\u202e", "")
    assert panel_rows(result.output)[-6:] == [
        "Method: rome",
        "Layers edited: [6]",
        "Norm delta: 1.0000",
        "Target prob: 0.2500 -> 0.7500",
        f"Saved: {shown}",
        "Governor disabled.",
    ]
    assert "\x1b[1mSaved:\x1b[0m " in result.output
    assert "\x1b[2mGovernor disabled.\x1b[0m" in result.output
    assert (
        f"\x1b[32mAttached\x1b[0m edited_model to registry entry \x1b[1m{RAW}\x1b[0m "
        f"\x1b[2m({str(weight).replace(chr(155) + chr(8238), '')})\x1b[0m"
    ) in result.output
    assert called == [(STORED, str(weight), "edited_model")]


def test_diff_table_and_json_keep_probes_and_labels(monkeypatch):
    report = edit_diff.DiffReport(
        "before-" + STORED,
        "after-" + STORED,
        (edit_diff.FactChange("prompt-" + STORED, "old-" + STORED, "new-" + STORED, True),),
        1,
        "0",
    )
    monkeypatch.setattr(edit_diff, "build_diff_report", lambda **kwargs: report)
    spelling = "diff-[draft].json" if sys.platform == "win32" else "diff-:warning:.json"
    result = run("diff", "before", "after", "-o", spelling)
    assert result.exit_code == 0, (result.output, result.exception)
    plain = " ".join(strip_ansi(result.output).split())
    assert f"Edit diff — before-{RAW} -> after-{RAW}" in plain
    for label in ("prompt", "old", "new"):
        assert f"{label}-{RAW}" in plain
    assert "\x1b[32myes\x1b[0m" in result.output
    assert f"Wrote diff -> {spelling}" in plain
    assert json.loads(Path(spelling).read_text(encoding="utf-8")) == report.to_dict()
