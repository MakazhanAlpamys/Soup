"""Regression tests for issue #1480.

Two more commands printed machine-readable JSON through Rich's ``console.print``,
the defect #1468 described and #1470 fixed for ``audit-log tail --json``,
``bom emit``, ``eval leaderboard --format json|csv`` and ``profile --json``:

- ``soup version --json --full`` — the record is one line and grows with every
  installed library. Rich folds it at the console width (80 columns when stdout
  is not a terminal), so the output stops being one JSON document.
- ``soup cost --json`` — Rich parses ``[...]`` in the string as console markup,
  so a value such as ``[v2]`` is dropped and the command still exits 0, or a
  value such as ``[/x]`` kills the command with ``MarkupError``. Both shapes
  are pinned below, because the silent drop is the one a test can miss.

Both now go through ``typer.echo`` like the other machine-readable paths, so the
bytes are written verbatim at any width.
"""

from __future__ import annotations

import builtins
import json
import types

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()

# `[/x]` is a closing tag Rich cannot match, and enough padding pushes the record
# well past 80 columns.
_MARKUP = "[/x]"
# `[v2]` is an opening tag Rich accepts, so it is dropped instead of raised:
# the command exits 0 with the value quietly gone.
_SILENT_MARKUP = "[v2]"
_PAD = "x" * 120

# The libraries `soup version --json --full` probes, with a version long enough
# that the assembled document is several times the console width.
_FAKE_LIBS = {
    "torch": "2.9.0+cu128." + _PAD,
    "transformers": "5.1.0." + _PAD,
    "peft": "0.19.0." + _PAD,
    "trl": "0.25.0." + _PAD,
    "datasets": "4.2.0." + _PAD,
    "accelerate": "1.12.0." + _PAD,
    "fastapi": "0.121.0." + _PAD,
    "vllm": "0.13.0." + _PAD,
    "datasketch": "1.8.0." + _PAD,
    "lm_eval": "0.4.12." + _PAD,
    "deepspeed": "0.18.0." + _PAD,
    "wandb": "0.24.0." + _PAD,
}


def _narrow(monkeypatch, module_path: str) -> None:
    """Force the command module's Console to 80 cols (piped-terminal width)."""
    monkeypatch.setattr(module_path, Console(width=80))


def _fake_import(real_import):
    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in _FAKE_LIBS and not fromlist:
            mod = types.ModuleType(name)
            mod.__version__ = _FAKE_LIBS[name]
            return mod
        return real_import(name, globals, locals, fromlist, level)

    return _import


def test_version_json_full_is_one_line_at_any_width(monkeypatch):
    """`soup version --json --full` must emit one JSON document, not folded text."""
    _narrow(monkeypatch, "soup_cli.cli.console")
    monkeypatch.setattr(builtins, "__import__", _fake_import(builtins.__import__))

    result = runner.invoke(app, ["version", "--json", "--full"])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 1, lines
    info = json.loads(lines[0])
    assert info["torch"] == _FAKE_LIBS["torch"]
    assert info["wandb"] == _FAKE_LIBS["wandb"]


@pytest.mark.parametrize("markup", [_MARKUP, _SILENT_MARKUP])
def test_cost_json_keeps_bracketed_value(monkeypatch, tmp_path, markup):
    """`soup cost --json` must not read `[...]` in a value as Rich markup.

    `[/x]` makes Rich raise `MarkupError`; `[v2]` makes it drop the tag and
    carry on, so the run looks clean and the value is missing from the record.
    """
    _narrow(monkeypatch, "soup_cli.commands.cost.console")
    gpu = f"H100{markup}"
    monkeypatch.setattr(
        "soup_cli.commands.cost.GPU_PRICING",
        [{"provider": "Test", "gpu": gpu, "cost_per_hr": 5.92, "speed_mult": 2.5}],
    )

    cfg = tmp_path / "soup.yaml"
    cfg.write_text(
        "base: meta-llama/Llama-3.2-1B\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 512\n"
        "training:\n"
        "  batch_size: 4\n"
        "  quantization: none\n"
        "output: ./output\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["cost", "--json", "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    results = json.loads(result.output)
    assert results[0]["gpu"] == gpu


def test_eval_against_json_only_keeps_bracketed_run_id(monkeypatch, tmp_path):
    """`soup eval against --json-only` must emit raw JSON too (third site).

    The command closes over a ``console`` passed to ``register``, so there is no
    module attribute to pin to width 80; the runner is already a non-terminal.
    """
    class _FakeTracker:
        def get_metric_series(self, run_id, source_metric):
            return [0.50, 0.50, 0.50, 0.50]

    monkeypatch.setattr("soup_cli.experiment.tracker.ExperimentTracker", _FakeTracker)

    baseline = f"run-baseline{_MARKUP}"
    result = runner.invoke(
        app,
        [
            "eval",
            "against",
            baseline,
            "--candidate",
            "run-candidate",
            "--json-only",
            "--n-samples",
            "100",
        ],
    )

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 1, lines
    verdict = json.loads(lines[0])
    assert verdict["baseline_run_id"] == baseline
    assert verdict["regressed"] is False
