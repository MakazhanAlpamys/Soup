"""Tests for autopilot gradient_checkpointing decision and monotonicity across VRAM budgets.

Fixes #2865.
"""

from __future__ import annotations

import json
import pytest

from soup_cli.autopilot.decisions import decide_performance_flags
from soup_cli.autopilot.generate_config import build_soup_config


class TestAutopilotGradientCheckpointing:
    def _write_data(self, tmp_path):
        rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(100)]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return path

    @pytest.mark.parametrize("budget", [10.0, 12.0, 16.0, 18.0])
    def test_autopilot_writes_gradient_checkpointing_true_for_tight_budgets(self, tmp_path, budget):
        """For an 8B model (16GB fp16 weights), budgets <= 16GB have deficits or zero headroom,
        and 18GB has 2GB headroom (< 4GB threshold). All must write gradient_checkpointing: true.
        """
        data_file = self._write_data(tmp_path)
        cfg = build_soup_config(
            model="meta-llama/Llama-3.1-8B-Instruct",
            data_path=str(data_file),
            goal="chat",
            vram_gb=budget,
        )
        assert cfg.training.quantization == "4bit"
        assert cfg.training.gradient_checkpointing is True

    @pytest.mark.parametrize("budget", [24.0, 48.0])
    def test_autopilot_writes_gradient_checkpointing_false_for_spacious_budgets(self, tmp_path, budget):
        """For an 8B model, budgets >= 24GB have >= 8GB headroom (> 4GB threshold),
        so gradient_checkpointing must be false for normal sequence lengths.
        """
        data_file = self._write_data(tmp_path)
        cfg = build_soup_config(
            model="meta-llama/Llama-3.1-8B-Instruct",
            data_path=str(data_file),
            goal="chat",
            vram_gb=budget,
        )
        assert cfg.training.gradient_checkpointing is False

    def test_monotonic_gradient_checkpointing_across_budgets(self, tmp_path):
        """Across budgets from 10GB to 48GB, no budget gets checkpointing off while a smaller budget gets it on."""
        data_file = self._write_data(tmp_path)
        budgets = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 32.0, 48.0]
        results = []
        for b in budgets:
            cfg = build_soup_config(
                model="meta-llama/Llama-3.1-8B-Instruct",
                data_path=str(data_file),
                goal="chat",
                vram_gb=b,
            )
            results.append((b, cfg.training.gradient_checkpointing))

        for i in range(len(results) - 1):
            b1, gc1 = results[i]
            b2, gc2 = results[i + 1]
            assert not (not gc1 and gc2), (
                f"Non-monotonic gradient checkpointing: budget {b1} has gc={gc1} but budget {b2} has gc={gc2}"
            )

    def test_decide_performance_flags_headroom_pinning(self):
        """Pin decide_performance_flags behavior for None, 0.0, negative, tight, and spacious headroom."""
        kwargs = dict(gpu_name="rtx4090", compute_capability=8.9, max_length=2048)

        # Unknown headroom (None or omitted): left off for normal sequence lengths
        assert decide_performance_flags(**kwargs, vram_headroom_gb=None)["gradient_checkpointing"] is False
        assert decide_performance_flags(**kwargs)["gradient_checkpointing"] is False

        # Zero headroom (0.0): tight (< 4.0), enabled
        assert decide_performance_flags(**kwargs, vram_headroom_gb=0.0)["gradient_checkpointing"] is True

        # Deficits / negative headroom (< 0.0): tight, enabled
        assert decide_performance_flags(**kwargs, vram_headroom_gb=-6.0)["gradient_checkpointing"] is True
        assert decide_performance_flags(**kwargs, vram_headroom_gb=-4.0)["gradient_checkpointing"] is True
        assert decide_performance_flags(**kwargs, vram_headroom_gb=-0.1)["gradient_checkpointing"] is True

        # Tight positive headroom (< 4.0): enabled
        assert decide_performance_flags(**kwargs, vram_headroom_gb=1.5)["gradient_checkpointing"] is True
        assert decide_performance_flags(**kwargs, vram_headroom_gb=3.9)["gradient_checkpointing"] is True

        # Boundary and spacious headroom (>= 4.0): left off
        assert decide_performance_flags(**kwargs, vram_headroom_gb=4.0)["gradient_checkpointing"] is False
        assert decide_performance_flags(**kwargs, vram_headroom_gb=8.0)["gradient_checkpointing"] is False
        assert decide_performance_flags(**kwargs, vram_headroom_gb=18.0)["gradient_checkpointing"] is False

    def test_decide_performance_flags_long_sequence_always_enabled(self):
        """Long sequence (>8192) enables checkpointing even with spacious or unknown headroom."""
        kwargs = dict(gpu_name="rtx4090", compute_capability=8.9, max_length=16384)
        assert decide_performance_flags(**kwargs, vram_headroom_gb=None)["gradient_checkpointing"] is True
        assert decide_performance_flags(**kwargs, vram_headroom_gb=24.0)["gradient_checkpointing"] is True
