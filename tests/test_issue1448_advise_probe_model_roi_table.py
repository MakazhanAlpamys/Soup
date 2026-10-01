"""#1448 — `soup advise run --probe-model X` runs the live probe but hid its ROI table.

`--probe-model` implies `--probe` (``run_probe = probe or probe_model is not None``),
and the probe is run and paid for under that derived flag, but the table was gated
on the raw ``probe`` option. The table must follow the same flag as the probe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.commands.advise as advise
from soup_cli.cli import app

ROWS = [
    {"instruction": f"Summarise item {i}", "output": f"Item {i} short."}
    for i in range(12)
]


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # keep advise_history / last-verdict out of the repo
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {"baselines": 0, "lora": 0, "proximity": 0}

    def baselines(rows, *, model=None, device=None):
        calls["baselines"] += 1
        return {"zero_shot": 0.1, "few_shot": 0.2, "rag": 0.3}

    def lora(rows, *, model=None, device=None):
        calls["lora"] += 1
        return 0.4, 12.0

    def proximity(rows, *, model=None, device=None):
        calls["proximity"] += 1
        return 0.5

    monkeypatch.setattr(advise, "synth_probe_baselines", baselines)
    monkeypatch.setattr(advise, "synth_probe_lora_delta", lora)
    monkeypatch.setattr(advise, "measure_base_model_proximity", proximity)
    return calls


def test_probe_model_alone_prints_the_roi_table(dataset: Path, fake_probe: dict) -> None:
    result = CliRunner().invoke(
        app, ["advise", "run", str(dataset), "--probe-model", "org/tiny"]
    )
    assert result.exit_code == 0, result.output
    # The probe ran (and on a real model would have been paid for)...
    assert fake_probe == {"baselines": 1, "lora": 1, "proximity": 1}
    # ...so its result must be shown.
    assert "ROI deltas" in result.output
    assert "prompt_eng" in result.output and "sft ETA" in result.output


def test_probe_model_with_explicit_probe_still_prints(dataset: Path, fake_probe: dict) -> None:
    result = CliRunner().invoke(
        app, ["advise", "run", str(dataset), "--probe-model", "org/tiny", "--probe"]
    )
    assert result.exit_code == 0, result.output
    assert "ROI deltas" in result.output


def test_no_probe_prints_no_table(dataset: Path, fake_probe: dict) -> None:
    result = CliRunner().invoke(app, ["advise", "run", str(dataset)])
    assert result.exit_code == 0, result.output
    assert fake_probe == {"baselines": 0, "lora": 0, "proximity": 0}
    assert "ROI deltas" not in result.output
