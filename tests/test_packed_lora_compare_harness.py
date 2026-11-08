"""benchmarks/harness/packed_lora_compare.py without a GPU: the recipe it writes, the summary it
reads back from the meter's files, and that it declines to run where there is no CUDA."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "packed_lora_compare.py"


def _load():
    spec = importlib.util.spec_from_file_location("packed_lora_compare", _HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load()


def _recipe(packed=None, batch=8, accum=1):
    return harness.RECIPE.format(
        model="Qwen/Qwen2.5-1.5B-Instruct", data="rows.jsonl", max_length=512, batch=batch,
        accum=accum, seed=3407, out="/tmp/run",
        packed=f"  packed_lora: {packed}\n" if packed else "")


class TestTheRecipe:
    def test_plain_soup_loads(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_recipe(batch=2, accum=4))
        assert cfg.training.packed_lora is None
        assert (cfg.training.batch_size, cfg.training.gradient_accumulation_steps) == (2, 4)
        assert cfg.training.gradient_checkpointing is True and cfg.training.seed == 3407

    def test_the_packed_recipe_loads_with_the_block(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_recipe("{enabled: true, gpu_budget_mib: 3000}"))
        assert cfg.training.packed_lora.enabled is True
        assert cfg.training.packed_lora.gpu_budget_mib == 3000
        assert cfg.training.lora.dropout == 0 and cfg.training.quantization == "4bit"

    def test_the_two_arms_differ_only_by_the_block_and_the_batch_split(self):
        plain = _recipe(batch=2, accum=4).splitlines()
        packed = _recipe("{enabled: true}", batch=8, accum=1).splitlines()
        different = [line for line in packed if line not in plain]
        assert sorted(line.strip() for line in different) == [
            "batch_size: 8", "gradient_accumulation_steps: 1", "packed_lora: {enabled: true}"]


class TestTheSummary:
    def _files(self, tmp_path, steps=8, per_step=100, seconds=0.5):
        with open(tmp_path / "steps.jsonl", "w", encoding="utf-8") as steps_file, \
                open(tmp_path / "log.jsonl", "w", encoding="utf-8") as log_file:
            for step in range(1, steps + 1):
                steps_file.write(json.dumps({
                    "step": step, "seconds": seconds * (2 if step == 1 else 1),
                    "alloc_peak_mib": 1000.0 + step, "reserved_peak_mib": 1500.0}) + "\n")
                log_file.write(json.dumps({
                    "step": step, "loss": 2.0 - step / 100, "grad_norm": 1.0,
                    "num_tokens": step * per_step}) + "\n")

    def test_the_rate_counts_only_the_steps_after_the_warm_up(self, tmp_path):
        self._files(tmp_path)
        summary = harness.summarise(str(tmp_path), warmup=3)
        assert summary["measured_steps"] == 5
        assert summary["tokens_per_s"] == pytest.approx(500 / 2.5)  # steps 4-8: 500 tokens, 2.5 s
        assert summary["step_s_median"] == 0.5 and summary["step1_s"] == 1.0
        assert summary["alloc_peak_mib"] == 1008.0 and summary["reserved_peak_mib"] == 1500.0
        assert summary["loss"][0] == 1.99 and len(summary["loss"]) == 8

    def test_a_run_that_wrote_nothing_has_an_empty_summary(self, tmp_path):
        assert harness.summarise(str(tmp_path), warmup=3) == {
            "steps_done": 0, "loss": [], "grad_norm": []}


class TestWithoutCuda:
    def test_it_declines_and_exits_zero(self, tmp_path, monkeypatch, capsys):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        code = harness.main(["--out", str(tmp_path), "--model", "m", "--data", "d.jsonl"])
        assert code == 0
        assert "no CUDA device" in capsys.readouterr().err

    @pytest.mark.parametrize("flag,expected", [([], False), (["--allow-oom-attempt"], True)])
    def test_allow_oom_attempt_reaches_soup_train_only_when_asked(
            self, tmp_path, monkeypatch, capsys, flag, expected):
        import torch

        import soup_cli.cli as cli

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(harness, "install_meter", lambda *args: None)
        calls = []
        monkeypatch.setattr(cli, "app", lambda: calls.append(list(sys.argv)))
        monkeypatch.setattr(sys, "argv", ["harness"])
        rows = tmp_path / "rows.jsonl"
        rows.write_text('{"messages": []}\n', encoding="utf-8")
        code = harness.main(["--out", str(tmp_path / "out"), "--model", "m", "--data", str(rows),
                             "--batch", "1", "--accum", "1", "--steps", "1", *flag])
        assert code == 0 and len(calls) == 1
        assert calls[0][:2] == ["soup", "train"] and "--yes" in calls[0]
        assert ("--allow-oom-attempt" in calls[0]) is expected
        assert '"batch": 1' in capsys.readouterr().out

    @pytest.mark.parametrize("flag,expected", [([], False),
                                               (["--packed", "{enabled: true}"], True)])
    def test_the_recipe_it_writes_has_the_block_only_when_asked(
            self, tmp_path, monkeypatch, capsys, flag, expected):
        import torch

        import soup_cli.cli as cli

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(harness, "install_meter", lambda *args: None)
        monkeypatch.setattr(cli, "app", lambda: None)
        monkeypatch.setattr(sys, "argv", ["harness"])
        rows = tmp_path / "rows.jsonl"
        rows.write_text('{"messages": []}\n', encoding="utf-8")
        out = tmp_path / "out"
        harness.main(["--out", str(out), "--model", "m", "--data", str(rows), "--batch", "1",
                      "--accum", "1", "--steps", "1", *flag])
        text = (out / "soup.yaml").read_text(encoding="utf-8")
        assert ("  packed_lora: {enabled: true}" in text) is expected

    def test_too_few_rows_are_refused_with_the_count(self, tmp_path, monkeypatch, capsys):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        rows = tmp_path / "rows.jsonl"
        rows.write_text('{"messages": []}\n' * 5, encoding="utf-8")
        code = harness.main(["--out", str(tmp_path / "out"), "--model", "m", "--data", str(rows),
                             "--batch", "2", "--accum", "1", "--steps", "10"])
        assert code == 2
        assert "has 5 rows, 20 needed" in capsys.readouterr().err
