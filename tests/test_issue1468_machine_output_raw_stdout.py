"""Regression tests for issue #1468.

Four commands print machine-readable output (JSON / JSONL / CSV) that used to go
through Rich's ``console.print``. When stdout is not a terminal Rich folds lines
at the console width (80 columns) and parses ``[...]`` as console markup, so a
piped record longer than the line was split across lines and any ``[...]`` token
was stripped from the data — or crashed the command with ``MarkupError``.

The fix routes those four sites through ``typer.echo`` (plain stdout), matching
the convention the rest of the CLI already uses for machine-readable output
(``adapters.py``, ``recipes.py``, ``data_clean.py``).

Each test pins the affected command's module ``console`` to width 80 so the
pre-fix fold is deterministic; the fix bypasses that console entirely, so the
output stays verbatim regardless of width, and a bracketed token no longer
crashes the command.
"""

from __future__ import annotations

import csv
import io
import json

from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()

# A token that Rich would treat as a (broken) closing markup tag, and enough
# padding to push every record well past 80 columns.
_MARKUP = "[/x]"
_PAD = "x" * 80


def _narrow(monkeypatch, module_path: str) -> None:
    """Force the command module's Console to 80 cols (piped-terminal width)."""
    monkeypatch.setattr(module_path, Console(width=80))


def test_audit_log_tail_json_is_raw_jsonl(monkeypatch, tmp_path):
    _narrow(monkeypatch, "soup_cli.commands.audit_log.console")
    record = {
        "timestamp": "2026-09-16T10:20:30.123456+00:00",
        "command": "train",
        "args": ["--config", f"runs/exp{_MARKUP}/soup.yaml", "--note", _PAD],
        "exit_code": 0,
        "operator_id": "ci",
        "host_id": "build-host-01",
    }
    log = tmp_path / "audit.jsonl"
    log.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["audit-log", "tail", "--json", "--path", str(log)])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    # One JSONL record on exactly one line, round-tripping to the original.
    assert len(lines) == 1, lines
    assert json.loads(lines[0]) == record
    assert _MARKUP in lines[0]


def test_bom_emit_stdout_is_valid_json_with_markup_preserved(monkeypatch):
    _narrow(monkeypatch, "soup_cli.commands.bom.console")
    name = f"adapter{_MARKUP}-{_PAD}"
    result = runner.invoke(
        app,
        [
            "bom",
            "emit",
            "--name",
            name,
            "--base-model",
            f"org/really-long-base-model-name-{_PAD}",
            "--base-sha",
            "a" * 64,
            "--config-sha",
            "c" * 64,
            "--format",
            "spdx",
        ],
    )

    assert result.exit_code == 0, result.output
    # Not folded / not markup-mangled: still a single parseable JSON document
    # that carries the bracketed name verbatim.
    doc = json.loads(result.output)
    assert isinstance(doc, dict)
    assert name in result.output


def test_eval_leaderboard_json_and_csv_are_raw(monkeypatch):
    _narrow(monkeypatch, "soup_cli.commands.eval.console")
    from soup_cli.eval.leaderboard import Leaderboard, LeaderboardEntry

    model = f"org/really-long-model-path-{_MARKUP}-{_PAD}"

    def _fake_build(_tracker):
        lb = Leaderboard(
            entries=[
                LeaderboardEntry(model_path=model, benchmark="mmlu", score=0.8),
            ]
        )
        lb.compute()
        return lb

    monkeypatch.setattr("soup_cli.eval.leaderboard.build_leaderboard_from_tracker", _fake_build)

    json_res = runner.invoke(app, ["eval", "leaderboard", "--format", "json"])
    assert json_res.exit_code == 0, json_res.output
    data = json.loads(json_res.output)
    assert data[0]["model"] == model

    csv_res = runner.invoke(app, ["eval", "leaderboard", "--format", "csv"])
    assert csv_res.exit_code == 0, csv_res.output
    rows = list(csv.reader(io.StringIO(csv_res.output)))
    body = [r for r in rows if r and r[0] == model]
    # The long model path stays in one field on one row (not folded into
    # several short rows), header + this row share the same column count.
    assert len(body) == 1, rows
    assert len(body[0]) == len(rows[0])


def test_profile_json_keeps_long_model_name(monkeypatch, tmp_path):
    _narrow(monkeypatch, "soup_cli.commands.profile.console")
    base = f"org/really-long-7b-model-name-{_MARKUP}-{_PAD}"
    cfg = tmp_path / "soup.yaml"
    cfg.write_text(
        f"base: {base}\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 512\n"
        "training:\n"
        "  batch_size: 4\n"
        "  quantization: none\n"
        "output: ./output\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["profile", "--json", "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["model_name"] == base
