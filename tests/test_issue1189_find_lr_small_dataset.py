"""#1189 — `soup train --find-lr` on a dataset smaller than `--find-lr-steps`.

The live sweep takes one row per step. It used to stop when the rows ran out (or when
the loss went non-finite) while the report was written against the full schedule, so
the length mismatch surfaced as "Invalid --find-lr-output". Now the sweep is sized to
the rows over the same LR range, refused below the 4-pair minimum before the model
loads, and trimmed to the steps that ran when the loss diverges.
"""

from __future__ import annotations

import json
import math

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi

pytest.importorskip("torch")

START, END = 1e-7, 1e-1
# Descends, then climbs: a curve find_optimal_lr can read.
CURVE = [3.0, 2.6, 2.2, 1.9, 1.7, 1.8, 2.4, 3.5, 5.0, 8.0]


@pytest.fixture
def find_lr(tmp_path, monkeypatch):
    """Run `soup train --find-lr` on `rows` rows with a fake tokenizer and model.

    `nan_at` makes the model's loss non-finite from that step on. Returns the
    normalised output, the exit code, the report (or None) and the model-load count.
    """
    import torch
    import transformers

    monkeypatch.chdir(tmp_path)
    loads = []

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

        def __call__(self, text, **kwargs):
            ids = torch.ones((1, 4), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def run(rows: int, steps: int, nan_at: "int | None" = None, output="lr_finder.json"):
        class ScriptedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.step = 0

            def forward(self, **batch):
                value = CURVE[self.step % len(CURVE)]
                if nan_at is not None and self.step >= nan_at:
                    value = math.nan
                self.step += 1
                return {"loss": self.weight.sum() * 0 + value}

        def load_model(*args, **kwargs):
            loads.append(args)
            return ScriptedModel()

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
        )
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)

        with open("train.jsonl", "w", encoding="utf-8") as handle:
            for index in range(rows):
                handle.write(json.dumps({"text": f"row {index}"}) + "\n")
        with open("soup.yaml", "w", encoding="utf-8") as handle:
            handle.write(
                "base: fake-org/fake-model\ntask: sft\n"
                "data:\n  train: ./train.jsonl\n  format: plaintext\n  max_length: 64\n"
                "output: ./output\n"
            )

        from soup_cli.cli import app

        result = CliRunner().invoke(app, [
            "train", "--config", "soup.yaml", "--find-lr",
            "--find-lr-start", str(START), "--find-lr-end", str(END),
            "--find-lr-steps", str(steps), "--find-lr-output", output,
        ])
        text = " ".join(strip_ansi(result.output).split())
        report = None
        if output == "lr_finder.json" and (tmp_path / output).exists():
            report = json.loads((tmp_path / output).read_text(encoding="utf-8"))
        return text, result.exit_code, report, len(loads)

    return run


def test_small_dataset_sweeps_its_rows_over_the_full_range(find_lr):
    text, code, report, _ = find_lr(rows=4, steps=5)
    assert code == 0, text
    assert len(report["lrs"]) == len(report["losses"]) == 4
    # Rebuilt over the whole range, not the first 4 entries of the 5-step schedule.
    assert report["lrs"][0] == pytest.approx(START, rel=1e-6)
    assert report["lrs"][-1] == pytest.approx(END, rel=1e-6)
    assert "4 rows" in text and "--find-lr-steps 5" in text, text
    assert "--find-lr-output" not in text


def test_enough_rows_keeps_the_requested_steps(find_lr):
    # Control: the note and the rebuild only happen when rows run short.
    text, code, report, _ = find_lr(rows=6, steps=5)
    assert code == 0, text
    assert len(report["lrs"]) == len(report["losses"]) == 5
    assert "fewer than --find-lr-steps" not in text


@pytest.mark.parametrize("rows", [0, 1, 3])
def test_fewer_than_four_rows_is_refused_before_the_model_loads(find_lr, rows):
    text, code, report, loads = find_lr(rows=rows, steps=5)
    assert code == 1, text
    count = f"{rows} row" + ("" if rows == 1 else "s")
    assert f"has {count}," in text and "--find-lr-steps 5" in text, text
    assert loads == 0
    # Not swallowed by the synthetic-curve fallback.
    assert report is None
    assert "synthetic" not in text


def test_too_few_steps_is_rejected_before_the_model_loads(find_lr):
    text, code, report, loads = find_lr(rows=10, steps=3)
    assert code == 1, text
    assert "num_steps must be in [4," in text, text
    assert loads == 0
    assert report is None
    assert "--find-lr-output" not in text


def test_divergence_writes_a_report_for_the_steps_that_ran(find_lr):
    text, code, report, _ = find_lr(rows=8, steps=8, nan_at=6)
    assert code == 0, text
    assert len(report["lrs"]) == len(report["losses"]) == 6
    # Names the LR of the step that went non-finite, not the last one that ran.
    assert "non-finite at lr 0.0139;" in text, text


def test_early_divergence_is_refused_with_its_own_message(find_lr):
    text, code, report, _ = find_lr(rows=8, steps=8, nan_at=2)
    assert code == 1, text
    assert "non-finite at lr 5.18e-06 after 2 steps" in text, text
    assert report is None
    assert "--find-lr-output" not in text
    assert "synthetic" not in text


def test_a_bad_output_path_still_names_find_lr_output(find_lr):
    # Control: the label stays for the one error it describes.
    text, code, _, _ = find_lr(rows=6, steps=5, output="../outside.json")
    assert code == 1, text
    assert "Invalid --find-lr-output" in text, text


def test_a_non_finite_first_step_points_away_from_the_lr_range(find_lr):
    # No update has run, so lowering the LR range cannot help.
    text, code, report, _ = find_lr(rows=8, steps=8, nan_at=0)
    assert code == 1, text
    assert "non-finite on the first step (lr 1e-07), before any update" in text, text
    assert "lower --find-lr-start" not in text
    assert report is None
