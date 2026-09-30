"""Regression tests for issue #1480.

Two more commands print machine-readable JSON through Rich's ``console.print``,
the defect #1468 described and #1470 fixed for ``soup audit-log tail --json``,
``soup bom emit``, ``soup eval leaderboard --format json|csv`` and
``soup profile --json``:

- ``soup version --json --full`` emits one line that grows with every installed
  library, so once it is longer than the console width (80 columns when piped)
  Rich folds it across lines and the output is no longer one JSON document.
- ``soup cost --json`` passes ``json.dumps(results, indent=2)`` through
  ``console.print``, so any string value containing square brackets is read as
  Rich markup and the token is silently dropped from the data.

The fix routes both sites through ``typer.echo`` (plain stdout), matching the
convention the rest of the CLI already uses for machine-readable output. Each
test pins the command's ``console`` to width 80 so the pre-fix fold is
deterministic, and passes a bracketed token so the markup stripping is
observable even at a wide console.
"""

from __future__ import annotations

import json

from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()

# A token Rich treats as a (broken) closing markup tag, and enough padding to
# push the record well past 80 columns.
_MARKUP = "[/x]"
_PAD = "x" * 80


def _narrow(monkeypatch, module_path: str) -> None:
    """Force the command module's Console to 80 cols (piped-terminal width)."""
    monkeypatch.setattr(module_path, Console(width=80))


def test_version_json_full_is_one_line_with_markup_preserved(monkeypatch):
    """``soup version --json --full`` is a single JSON document, not folded."""
    _narrow(monkeypatch, "soup_cli.cli.console")
    # `--full` only lists libraries that are importable, so the record is short
    # in a bare environment. Pin the version to a long value to make the record
    # exceed the console width, exactly as an installed training stack does.
    monkeypatch.setattr("soup_cli.cli.__version__", f"0.75.0+local-{_MARKUP}-{_PAD}")

    result = runner.invoke(app, ["version", "--json", "--full"])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    # One JSON document on exactly one line, carrying the token verbatim.
    assert len(lines) == 1, lines
    assert _MARKUP in lines[0]
    assert json.loads(lines[0])["version"].endswith(f"-{_PAD}")


def test_version_json_without_full_is_single_line(monkeypatch):
    """The shorter record is raw stdout too, so it never depends on width."""
    _narrow(monkeypatch, "soup_cli.cli.console")

    result = runner.invoke(app, ["version", "--json"])

    assert result.exit_code == 0, result.output
    assert len([ln for ln in result.output.splitlines() if ln.strip()]) == 1
    json.loads(result.output)


def _cost_config(tmp_path):
    cfg = tmp_path / "soup.yaml"
    cfg.write_text(
        "base: org/really-long-7b-model-name\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 512\n"
        "training:\n"
        "  batch_size: 4\n"
        "  quantization: none\n"
        "  epochs: 1\n"
        "output: ./output\n",
        encoding="utf-8",
    )
    return cfg


def test_cost_json_keeps_bracketed_values(monkeypatch, tmp_path):
    """A bracketed string in the cost rows survives as data, not markup."""
    _narrow(monkeypatch, "soup_cli.commands.cost.console")
    import soup_cli.commands.cost as cost_mod

    monkeypatch.setattr(
        cost_mod,
        "GPU_PRICING",
        [
            {
                "provider": f"RunPod{_MARKUP}",
                "gpu": f"RTX 4090 {_PAD}",
                "cost_per_hr": 0.35,
                "speed_mult": 0.7,
            }
        ],
    )

    result = runner.invoke(app, ["cost", "--json", "--config", str(_cost_config(tmp_path))])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list) and data, result.output
    assert data[0]["provider"] == f"RunPod{_MARKUP}"
    assert data[0]["gpu"] == f"RTX 4090 {_PAD}"
