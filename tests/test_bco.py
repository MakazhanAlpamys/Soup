"""Tests for BCO (Binary Classifier Optimization) — v0.40.0 Part A.

Mirrors ORPO/SimPO/IPO test layout: schema, data format gate, template,
trainer wrapper init + routing (train + sweep), edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Schema Tests ───────────────────────────────────────────────────────────


class TestBCOConfig:
    """Test BCO task config validation."""

    def test_bco_task_accepted(self):
        cfg = SoupConfig(
            base="some-model",
            task="bco",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert cfg.task == "bco"

    def test_bco_beta_default(self):
        cfg = SoupConfig(
            base="some-model",
            task="bco",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert cfg.training.bco_beta == 0.1

    def test_bco_beta_custom(self):
        cfg = SoupConfig(
            base="some-model",
            task="bco",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"bco_beta": 0.05},
        )
        assert cfg.training.bco_beta == pytest.approx(0.05)

    def test_bco_beta_must_be_positive(self):
        with pytest.raises(ValidationError, match="bco_beta"):
            SoupConfig(
                base="some-model",
                task="bco",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"bco_beta": 0},
            )

    def test_bco_beta_negative_rejected(self):
        with pytest.raises(ValidationError, match="bco_beta"):
            SoupConfig(
                base="some-model",
                task="bco",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"bco_beta": -0.1},
            )

    def test_bco_full_config(self):
        cfg = SoupConfig(
            base="meta-llama/Llama-3.1-8B-Instruct",
            task="bco",
            data={"train": "./data.jsonl", "format": "dpo", "max_length": 2048},
            training={
                "epochs": 3,
                "lr": 1e-5,
                "bco_beta": 0.2,
                "lora": {"r": 64, "alpha": 16},
                "quantization": "4bit",
            },
        )
        assert cfg.task == "bco"
        assert cfg.training.bco_beta == pytest.approx(0.2)


# ─── Data Format Tests ──────────────────────────────────────────────────────


class TestBCODataFormat:
    """Test that BCO uses the DPO data format (prompt+chosen+rejected)."""

    def test_dpo_format_works_for_bco(self):
        from soup_cli.data.formats import detect_format

        data = [{"prompt": "Q", "chosen": "A", "rejected": "B"}]
        assert detect_format(data) == "dpo"


# ─── Split helper Tests ─────────────────────────────────────────────────────


class TestSplitDpoRowsToBco:
    def test_two_rows_become_four_with_correct_labels(self):
        from soup_cli.trainer.bco import _split_dpo_rows_to_bco

        rows = [
            {"prompt": "p1", "chosen": "c1", "rejected": "r1"},
            {"prompt": "p2", "chosen": "c2", "rejected": "r2"},
        ]
        out = _split_dpo_rows_to_bco(rows)
        assert len(out) == 4
        assert out[0] == {"prompt": "p1", "completion": "c1", "label": True}
        assert out[1] == {"prompt": "p1", "completion": "r1", "label": False}
        assert out[2] == {"prompt": "p2", "completion": "c2", "label": True}
        assert out[3] == {"prompt": "p2", "completion": "r2", "label": False}

    def test_empty_input_returns_empty_list(self):
        from soup_cli.trainer.bco import _split_dpo_rows_to_bco

        assert _split_dpo_rows_to_bco([]) == []

    @pytest.mark.parametrize(
        "row",
        [
            {"chosen": "c", "rejected": "r"},
            {"prompt": "p", "rejected": "r"},
            {"prompt": "p", "chosen": "c"},
            {},
            {"unrelated": "field"},
        ],
    )
    def test_missing_required_field_skipped(self, row):
        from soup_cli.trainer.bco import _split_dpo_rows_to_bco

        assert _split_dpo_rows_to_bco([row]) == []

    def test_extra_keys_ignored(self):
        from soup_cli.trainer.bco import _split_dpo_rows_to_bco

        out = _split_dpo_rows_to_bco(
            [{"prompt": "p", "chosen": "c", "rejected": "r", "extra": "x"}]
        )
        assert len(out) == 2
        assert all("extra" not in row for row in out)

    def test_skipped_rows_logged_at_debug(self, caplog):
        import logging

        from soup_cli.trainer.bco import _split_dpo_rows_to_bco

        caplog.set_level(logging.DEBUG, logger="soup_cli.trainer.bco")
        _split_dpo_rows_to_bco([{"prompt": "p", "chosen": "c"}])
        assert any("skipped" in r.message.lower() for r in caplog.records)


# ─── Template Tests ─────────────────────────────────────────────────────────


class TestBCOTemplate:
    def test_bco_template_exists(self):
        assert "bco" in TEMPLATES

    def test_bco_template_valid_yaml(self):
        import yaml

        config = yaml.safe_load(TEMPLATES["bco"])
        assert config["task"] == "bco"
        assert config["training"]["bco_beta"] == 0.1
        assert config["data"]["format"] == "dpo"

    def test_bco_template_valid_config(self):
        import yaml

        raw = yaml.safe_load(TEMPLATES["bco"])
        cfg = SoupConfig(**raw)
        assert cfg.task == "bco"
        assert cfg.training.bco_beta == 0.1


# ─── Trainer Wrapper Tests ──────────────────────────────────────────────────


class TestBCOTrainerWrapper:
    def test_bco_import_exists(self):
        from soup_cli.trainer.bco import BCOTrainerWrapper

        assert BCOTrainerWrapper is not None

    def test_bco_wrapper_init(self):
        from soup_cli.trainer.bco import BCOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="bco",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        wrapper = BCOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.task == "bco"
        assert wrapper.device == "cpu"
        assert wrapper.model is None
        assert wrapper.trainer is None

    def test_bco_wrapper_init_with_options(self):
        from soup_cli.trainer.bco import BCOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="bco",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        wrapper = BCOTrainerWrapper(
            cfg, device="cuda", report_to="wandb", deepspeed_config="ds.json",
        )
        assert wrapper.report_to == "wandb"
        assert wrapper.deepspeed_config == "ds.json"

    def test_bco_train_before_setup_raises(self):
        from soup_cli.trainer.bco import BCOTrainerWrapper

        cfg = SoupConfig(
            base="some-model",
            task="bco",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        wrapper = BCOTrainerWrapper(cfg, device="cpu")
        with pytest.raises(RuntimeError, match="setup"):
            wrapper.train()


# ─── Routing Tests ──────────────────────────────────────────────────────────


class TestBCOTrainRouting:
    """Test that train + sweep route to BCO trainer."""

    def test_sweep_routes_to_bco_trainer(self):
        from soup_cli.commands.sweep import _run_single

        cfg = SoupConfig(
            base="some-model",
            task="bco",
            data={"train": "./data.jsonl", "format": "dpo"},
        )

        fake_dataset = {
            "train": [{"prompt": "Q?", "chosen": "A", "rejected": "B"}],
        }
        fake_result = {
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "total_steps": 10,
            "duration_secs": 60.0,
            "output_dir": "./output",
            "duration": "1m",
        }
        fake_gpu_info = {"memory_total": "0 MB", "memory_total_bytes": 0}

        with mock_patch(
            "soup_cli.data.loader.load_dataset", return_value=fake_dataset,
        ), mock_patch(
            "soup_cli.utils.gpu.detect_device", return_value=("cpu", "CPU"),
        ), mock_patch(
            "soup_cli.utils.gpu.get_gpu_info", return_value=fake_gpu_info,
        ), mock_patch(
            "soup_cli.experiment.tracker.ExperimentTracker",
        ) as mock_tracker_cls, mock_patch(
            "soup_cli.monitoring.display.TrainingDisplay",
        ), mock_patch(
            "soup_cli.trainer.bco.BCOTrainerWrapper.setup",
        ), mock_patch(
            "soup_cli.trainer.bco.BCOTrainerWrapper.train",
            return_value=fake_result,
        ) as mock_train:
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value = "run-bco-1"
            mock_tracker_cls.return_value = mock_tracker
            result = _run_single(cfg, {}, "bco_run_1", None)

        mock_train.assert_called_once()
        assert result["run_id"] == "run-bco-1"


# ─── Sweep Shortcut Tests ───────────────────────────────────────────────────


class TestBCOSweepParams:
    def test_bco_beta_shortcut(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {"training": {"bco_beta": 0.1}}
        _set_nested_param(config, "bco_beta", 0.05)
        assert config["training"]["bco_beta"] == 0.05

    def test_bco_beta_shortcut_creates_nested_key(self):
        from soup_cli.commands.sweep import _set_nested_param

        config = {}
        _set_nested_param(config, "bco_beta", 0.2)
        assert config["training"]["bco_beta"] == pytest.approx(0.2)
