"""#1203 — `soup train --find-lr` invented a curve when the sweep could not run.

The fallback answered any failure — a base model that does not exist, a config
that will not parse — with a curve shaped by the LR schedule alone, then wrote a
report with a confident `recommended_lr` and exited 0. Two different nonexistent
model ids therefore reported the identical number. Both now refuse with exit 1
and write no report, while a sweep that can run still writes its report.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi

pytest.importorskip("torch")

START, END = 1e-7, 1e-1
# The words transformers uses for an id that resolves to nothing (#1189's repro).
MISSING = (
    "soup-audit-1189/this-model-does-not-exist is not a local folder and is not "
    "a valid model identifier listed on 'https://huggingface.co/models'"
)
# Descends, then climbs: a curve find_optimal_lr can read.
CURVE = [3.0, 2.6, 2.2, 1.9, 1.7, 1.8, 2.4, 3.5, 5.0, 8.0]


@pytest.fixture
def find_lr(tmp_path, monkeypatch):
    """Run `soup train --find-lr` on an 8-row dataset.

    `model_error` makes the model load raise that error, `scripted` swaps in a
    model that walks `CURVE`, and `broken_yaml` writes an unparseable config.
    Returns the normalised output, the exit code and the report (or None).
    """
    monkeypatch.chdir(tmp_path)

    def run(model_error=None, scripted=False, broken_yaml=False, output="lr_finder.json"):
        import torch
        import transformers

        class FakeTokenizer:
            pad_token = None
            eos_token = "</s>"

            def __call__(self, text, **kwargs):
                ids = torch.ones((1, 4), dtype=torch.long)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        class ScriptedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.step = 0

            def forward(self, **batch):
                value = CURVE[self.step % len(CURVE)]
                self.step += 1
                return {"loss": self.weight.sum() * 0 + value}

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
        )
        if model_error is not None:
            def unreachable(*args, **kwargs):
                raise model_error

            monkeypatch.setattr(
                transformers.AutoModelForCausalLM, "from_pretrained", unreachable
            )
        elif scripted:
            monkeypatch.setattr(
                transformers.AutoModelForCausalLM,
                "from_pretrained",
                lambda *a, **k: ScriptedModel(),
            )

        with open("train.jsonl", "w", encoding="utf-8") as handle:
            for index in range(8):
                handle.write(json.dumps({"text": f"row {index}"}) + "\n")
        with open("soup.yaml", "w", encoding="utf-8") as handle:
            handle.write(
                "base: fake-org/fake-model\ndata: [unclosed\n"
                if broken_yaml
                else f"base: {MISSING.split(' is not ')[0]}\ntask: sft\n"
                "data:\n  train: ./train.jsonl\n  format: plaintext\n  max_length: 64\n"
                "output: ./output\n"
            )

        from soup_cli.cli import app

        result = CliRunner().invoke(app, [
            "train", "--config", "soup.yaml", "--find-lr",
            "--find-lr-start", str(START), "--find-lr-end", str(END),
            "--find-lr-steps", "8", "--find-lr-output", output,
        ])
        text = " ".join(strip_ansi(result.output).split())
        path = tmp_path / output
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return text, result.exit_code, report

    return run


def test_unreachable_base_refuses_and_writes_no_report(find_lr):
    text, code, report = find_lr(model_error=OSError(MISSING))
    assert code == 1, text
    assert report is None, text
    assert "live sweep unavailable" in text, text
    assert "not a valid model identifier" in text, text
    assert "LR finder report written" not in text


def test_broken_yaml_refuses_with_the_loader_message(find_lr):
    text, code, report = find_lr(broken_yaml=True)
    assert code == 1, text
    assert report is None, text
    assert "config load failed" in text, text
    # The loader's own words, not just our prefix: Rich used to read
    # "[unclosed ... ']" in the message as a markup tag and drop it (#1203).
    assert "while parsing a flow sequence" in text, text
    assert "data: [unclosed" in text, text
    assert "LR finder report written" not in text


def test_two_unreachable_bases_both_refuse_instead_of_sharing_a_curve(find_lr):
    # The reported symptom: two different missing ids returned one identical
    # fabricated curve and the identical recommended_lr.
    first = find_lr(model_error=OSError(MISSING))
    second = find_lr(model_error=OSError("another-org/also-missing-model is not a local folder"))
    assert first[1] == second[1] == 1, (first[0], second[0])
    assert first[2] is None and second[2] is None, (first[0], second[0])


def test_a_sweep_that_can_run_still_writes_its_report(find_lr):
    # Control: the refusal must not swallow a sweep that actually ran.
    text, code, report = find_lr(scripted=True)
    assert code == 0, text
    assert report is not None, text
    assert len(report["lrs"]) == len(report["losses"]) == 8, text
    assert report["recommended_lr"] > 0
