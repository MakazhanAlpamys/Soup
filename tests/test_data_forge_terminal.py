"""Forge diagnostics and receipts show source text while storing the original values."""

import json
import os
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()
VALUES = ["plain", "x[red]n", "[/]n", "n:warning:", "x\\[n", "n\\\\", "x\x9b\u202en"]


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.commands.data_forge.console", Console(width=500, height=100))
    Path("docs").mkdir()
    Path("docs/input.txt").write_text("A useful source document.", encoding="utf-8")


def run(*extra):
    return runner.invoke(app, ["data", "forge", "--docs", "docs", "--target-rows", "1", *extra])


def plain(result):
    return " ".join(strip_ansi(result.output).split())


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("field", ["teacher", "output", "provenance"])
def test_receipt_names_real_outputs_and_original_teacher(value, field):
    if sys.platform == "win32" and field != "teacher" and ":" in value:
        pytest.skip("colon is not valid in a Windows filename")
    typed = value if field == "teacher" else f"out-{value}.jsonl"
    if field != "teacher":
        Path(typed).parent.mkdir(parents=True, exist_ok=True)

    result = run(f"--{field}", typed)

    assert result.exit_code == 0, result.output
    shown = typed.replace("\x9b\u202e", "")
    teacher = shown if field == "teacher" else "local-judge"
    dataset = os.path.realpath(typed if field == "output" else "forge_dataset.jsonl")
    dataset = dataset.replace("\x9b\u202e", "")
    provenance = os.path.realpath(typed if field == "provenance" else "forge_provenance.json")
    provenance = provenance.replace("\x9b\u202e", "")
    text = plain(result)
    assert re.search(
        r"Task: sft (?:\W )+Docs scanned: 1 (?:\W )+Rows kept: 1 / target 1 "
        r"(?:\W )+Teacher: " + re.escape(teacher) + r" (?:\W )+Threshold: 0\.00 "
        r"(?:\W )+Dataset: "
        + re.escape(dataset)
        + r" (?:\W )+Provenance: "
        + re.escape(provenance),
        text,
    )
    dataset_path = typed if field == "output" else "forge_dataset.jsonl"
    manifest_path = typed if field == "provenance" else "forge_provenance.json"
    assert len(Path(dataset_path).read_text(encoding="utf-8").splitlines()) == 1
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if field == "teacher":
        assert manifest["records"][0]["judge_id"] == value
    assert Path("docs/input.txt").read_text(encoding="utf-8") == "A useful source document."


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize(
    "stage,label,code,extra",
    [
        ("discover_documents", "Plan failed:", 1, []),
        ("validate_hub_name", "", 2, []),
        ("prefetch_import", "", 1, ["--hub", "modelscope", "--teacher", "owner/model"]),
        (
            "prefetch_model_from_hub",
            "Teacher pre-fetch failed:",
            1,
            ["--hub", "modelscope", "--teacher", "owner/model"],
        ),
        (
            "make_judge_provider_fn",
            "Failed to build judge backend:",
            1,
            ["--judge-provider", "ollama"],
        ),
        ("build_forge_plan", "Plan failed:", 1, []),
        ("write_forge_dataset", "Write failed:", 1, []),
        ("write_provenance", "Write failed:", 1, []),
    ],
)
def test_failure_sites_are_literal(monkeypatch, value, stage, label, code, extra):
    import_failure = stage == "prefetch_import"

    def fail(*args, **kwargs):
        if import_failure:
            raise ImportError(value)
        raise ValueError(value)

    if stage == "prefetch_import":
        stage = "prefetch_model_from_hub"
    module = "hubs" if stage in {"validate_hub_name", "prefetch_model_from_hub"} else "data_forge"
    monkeypatch.setattr(f"soup_cli.utils.{module}.{stage}", fail)

    result = run(*extra)

    assert result.exit_code == code
    message = f"{label} {value}".strip().replace("\x9b\u202e", "")
    assert message in plain(result)
    assert not Path("forge_provenance.json").exists()
    assert Path("docs/input.txt").read_text(encoding="utf-8") == "A useful source document."


@pytest.mark.parametrize("value", VALUES)
def test_unknown_provider_is_named_literally(value):
    result = run("--judge-provider", value)
    assert result.exit_code == 2
    assert plain(result) == (
        f"Unknown --judge-provider: {value.replace(chr(155) + chr(8238), '')}. "
        "Choose from ['anthropic', 'ollama', 'vllm']."
    )


@pytest.mark.parametrize("value", VALUES)
def test_non_repo_teacher_warning_is_literal(value):
    if "/" in value:
        pytest.skip("this spelling is routed to hub prefetch")
    result = run("--hub", "modelscope", "--teacher", value)
    assert result.exit_code == 0, result.output
    assert (
        f"--hub modelscope ignored: --teacher {value!r} is not a repo id (owner/name), "
        "so there is nothing to pre-fetch."
    ) in plain(result)


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("partial", [False, True])
def test_judge_failures_and_live_model_are_literal(monkeypatch, value, partial):
    calls = []
    Path("docs/second.txt").write_text("A second source document.", encoding="utf-8")

    def judge(prompt):
        calls.append(prompt)
        if partial and len(calls) == 2:
            return {"text": "An answer independent of the source."}
        raise ValueError(value)

    def provider(name, **kwargs):
        assert name == "ollama" and kwargs["model"] == value and kwargs["raise_on_error"]
        return judge

    monkeypatch.setattr("soup_cli.utils.data_forge.make_judge_provider_fn", provider)
    result = run("--target-rows", "2", "--judge-provider", "ollama", "--judge-model", value)

    assert result.exit_code == (0 if partial else 1), result.output
    text = plain(result)
    assert value.replace("\x9b\u202e", "") in text
    assert ("Warning:" if partial else "No usable rows produced:") in text
    assert len(calls) == 2
    if partial:
        assert (
            f"Live judge backend: ollama (model: {value.replace(chr(155) + chr(8238), '')})."
            in text
        )
        assert "Judge calls: 1 of 2 failed" in text
    else:
        assert not Path("forge_dataset.jsonl").exists()


def test_panel_and_warning_styles(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.data_forge.console",
        Console(
            width=500, height=100, force_terminal=True, no_color=False, color_system="standard"
        ),
    )
    result = run("--hub", "modelscope", "--teacher", "label")
    assert result.exit_code == 0, result.output
    assert "\x1b[33m--hub modelscope ignored:\x1b[0m" in result.output
    for value in (
        "sft",
        " 1",
        "1",
        "label",
        "0.00",
        os.path.realpath("forge_dataset.jsonl"),
        os.path.realpath("forge_provenance.json"),
    ):
        assert f"\x1b[1m{value}\x1b[0m" in result.output
