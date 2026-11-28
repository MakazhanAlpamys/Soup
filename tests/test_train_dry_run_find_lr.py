"""#3270 - `soup train --dry-run --find-lr` refuses before the live LR sweep.

`--dry-run` promises to validate the config and data without training, but the
`--find-lr` fast path ran ahead of the dry-run branch, so the combination started the
live sweep and wrote the LR report. It now exits 2 before any model work, naming both
flags and which one to drop. Each flag alone works as before.
"""

from __future__ import annotations

import json
import os

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi

runner = CliRunner()

LRS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
LOSSES = [3.0, 2.0, 1.5, 1.8, 4.0]
REFUSAL = (
    "--dry-run cannot be combined with --find-lr: --find-lr runs a live learning-rate "
    "sweep. Drop --find-lr to validate the config and data, or drop --dry-run to run "
    "the sweep."
)


@pytest.fixture
def train_env(tmp_path, monkeypatch):
    """A local plaintext config, with the live sweep replaced by a recording stub."""
    from soup_cli.commands import train as train_command

    monkeypatch.chdir(tmp_path)
    with open("train.jsonl", "w", encoding="utf-8") as handle:
        for index in range(8):
            handle.write(json.dumps({"text": f"row {index}"}) + "\n")
    with open("soup.yaml", "w", encoding="utf-8") as handle:
        handle.write(
            "base: fake-org/fake-model\ntask: sft\n"
            "data:\n  train: ./train.jsonl\n  format: plaintext\n  max_length: 64\n"
            "output: ./output\n"
        )
    sweeps = []

    def fake_sweep(*args, **kwargs):
        sweeps.append((args, kwargs))
        return list(LRS), list(LOSSES)

    monkeypatch.setattr(train_command, "_run_live_lr_sweep", fake_sweep)

    def run(*args):
        from soup_cli.cli import app

        result = runner.invoke(app, ["train", "--config", "soup.yaml", *args])
        return result, " ".join(strip_ansi(result.output).split())

    return run, sweeps


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(("--find-lr", "--dry-run"), id="find-lr-first"),
        pytest.param(("--dry-run", "--find-lr"), id="dry-run-first"),
        pytest.param(
            ("--dry-run", "--find-lr", "--find-lr-output", "sweep/report.json"),
            id="custom-output",
        ),
    ],
)
def test_dry_run_with_find_lr_is_refused_before_the_sweep(train_env, flags):
    run, sweeps = train_env

    result, text = run(*flags)

    assert result.exit_code == 2, text
    assert REFUSAL in text, text
    assert sweeps == []
    assert not os.path.exists("lr_finder.json")
    assert not os.path.exists("sweep")
    assert not os.path.exists("output")
    assert "Dry run" not in text, text


def test_find_lr_alone_still_runs_the_sweep_and_writes_the_report(train_env):
    run, sweeps = train_env

    result, text = run("--find-lr")

    assert result.exit_code == 0, text
    assert len(sweeps) == 1
    assert "LR finder report written to: lr_finder.json" in text, text
    with open("lr_finder.json", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["lrs"] == LRS
    assert report["losses"] == LOSSES
    assert "cannot be combined" not in text, text


def test_dry_run_alone_still_validates_without_the_sweep(train_env):
    run, sweeps = train_env

    result, text = run("--dry-run")

    assert result.exit_code == 0, text
    assert sweeps == []
    assert "Config valid. Ready to train!" in text, text
    assert not os.path.exists("lr_finder.json")
    assert "cannot be combined" not in text, text
