"""Tests for HF_DEACTIVATE_ASYNC_LOAD default on Apple Silicon MPS (#2076).

Prevents Transformers threaded weight-loading SIGABRT crash in _materialize_copy
while preserving explicit user overrides and keeping non-MPS targets untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.gpu import configure_mps_async_load
from tests.conftest import strip_ansi


@pytest.fixture(autouse=True)
def clean_hf_async_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure HF_DEACTIVATE_ASYNC_LOAD is restored cleanly after every test.

    monkeypatch.delenv(..., raising=False) on an absent key does not register
    an undo action in pytest. Setting a temporary sentinel first guarantees
    teardown restores the environment to clean unset state (#2040 review).
    """
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "sentinel_for_teardown")
    monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD")


class TestMPSAsyncLoadEnvironment:
    def test_train_mps_device_sets_async_load_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "soup.yaml"
        config_path.write_text(
            "base: owner/model\n"
            "data:\n"
            "  train: dummy.jsonl\n"
            "training:\n"
            "  quantization: none\n",
            encoding="utf-8",
        )
        (tmp_path / "dummy.jsonl").write_text('{"text": "hi"}\n', encoding="utf-8")

        monkeypatch.setattr(
            "soup_cli.commands.train.detect_device",
            lambda **kwargs: ("mps", "Apple Silicon (MPS)"),
        )
        monkeypatch.setattr(
            "soup_cli.utils.gpu.detect_device",
            lambda **kwargs: ("mps", "Apple Silicon (MPS)"),
        )
        monkeypatch.setattr(
            "soup_cli.commands.train._hardware_fit_preflight",
            lambda *args, **kwargs: None,
        )

        env_at_build: list[str | None] = []

        with patch("soup_cli.trainer.dispatch.build_trainer") as mock_build, \
             patch("soup_cli.commands.train.load_dataset") as mock_load:
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {
                "output_dir": str(tmp_path),
                "initial_loss": 1.0,
                "final_loss": 0.5,
                "total_steps": 10,
                "duration_secs": 1.0,
                "epochs": 1,
                "training_loss": 0.5,
            }
            mock_build.side_effect = lambda *a, **kw: (
                env_at_build.append(os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")) or mock_trainer
            )
            mock_load.return_value = {"train": [{"text": "hi"}]}

            runner = CliRunner()
            result = runner.invoke(
                app,
                ["train", "--config", str(config_path), "--yes"],
            )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert env_at_build == ["1"], "HF_DEACTIVATE_ASYNC_LOAD must be set before model loading"
        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1"

    def test_train_mps_preserves_user_specified_async_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "0")

        config_path = tmp_path / "soup.yaml"
        config_path.write_text(
            "base: owner/model\n"
            "data:\n"
            "  train: dummy.jsonl\n"
            "training:\n"
            "  quantization: none\n",
            encoding="utf-8",
        )
        (tmp_path / "dummy.jsonl").write_text('{"text": "hi"}\n', encoding="utf-8")

        monkeypatch.setattr(
            "soup_cli.commands.train.detect_device",
            lambda **kwargs: ("mps", "Apple Silicon (MPS)"),
        )
        monkeypatch.setattr(
            "soup_cli.utils.gpu.detect_device",
            lambda **kwargs: ("mps", "Apple Silicon (MPS)"),
        )
        monkeypatch.setattr(
            "soup_cli.commands.train._hardware_fit_preflight",
            lambda *args, **kwargs: None,
        )

        env_at_build: list[str | None] = []

        with patch("soup_cli.trainer.dispatch.build_trainer") as mock_build, \
             patch("soup_cli.commands.train.load_dataset") as mock_load:
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {
                "output_dir": str(tmp_path),
                "initial_loss": 1.0,
                "final_loss": 0.5,
                "total_steps": 10,
                "duration_secs": 1.0,
                "epochs": 1,
                "training_loss": 0.5,
            }
            mock_build.side_effect = lambda *a, **kw: (
                env_at_build.append(os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")) or mock_trainer
            )
            mock_load.return_value = {"train": [{"text": "hi"}]}

            runner = CliRunner()
            result = runner.invoke(
                app,
                ["train", "--config", str(config_path), "--yes"],
            )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert env_at_build == ["0"], "User-set async load must be preserved before model loading"
        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "0"

    @pytest.mark.parametrize("target_device", ["cuda", "cpu", "mlx"])
    def test_train_non_mps_device_keeps_async_load_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target_device: str
    ) -> None:
        config_path = tmp_path / "soup.yaml"
        config_path.write_text(
            "base: owner/model\n"
            "data:\n"
            "  train: dummy.jsonl\n"
            "training:\n"
            "  quantization: none\n",
            encoding="utf-8",
        )
        (tmp_path / "dummy.jsonl").write_text('{"text": "hi"}\n', encoding="utf-8")

        monkeypatch.setattr(
            "soup_cli.commands.train.detect_device",
            lambda **kwargs: (target_device, f"Accelerator ({target_device})"),
        )
        monkeypatch.setattr(
            "soup_cli.utils.gpu.detect_device",
            lambda **kwargs: (target_device, f"Accelerator ({target_device})"),
        )
        monkeypatch.setattr(
            "soup_cli.commands.train._hardware_fit_preflight",
            lambda *args, **kwargs: None,
        )

        env_at_build: list[str | None] = []

        with patch("soup_cli.trainer.dispatch.build_trainer") as mock_build, \
             patch("soup_cli.commands.train.load_dataset") as mock_load:
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {
                "output_dir": str(tmp_path),
                "initial_loss": 1.0,
                "final_loss": 0.5,
                "total_steps": 10,
                "duration_secs": 1.0,
                "epochs": 1,
                "training_loss": 0.5,
            }
            mock_build.side_effect = lambda *a, **kw: (
                env_at_build.append(os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")) or mock_trainer
            )
            mock_load.return_value = {"train": [{"text": "hi"}]}

            runner = CliRunner()
            result = runner.invoke(
                app,
                ["train", "--config", str(config_path), "--yes"],
            )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert env_at_build == [None], (
            f"HF_DEACTIVATE_ASYNC_LOAD must remain unset on non-MPS device {target_device}"
        )
        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") is None

    def test_configure_mps_async_load_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 1. Explicit mps device sets default to 1
        configure_mps_async_load("mps")
        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1"

        # 2. Reset and test mps sub-device (e.g. mps:0)
        monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
        configure_mps_async_load("mps:0")
        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1"

        # 3. Explicit non-mps devices leave variable unset
        for non_mps in ["cuda", "cpu", "mlx"]:
            monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
            configure_mps_async_load(non_mps)
            assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") is None

        # 4. Device is None, MPS is available -> sets 1
        monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
            configure_mps_async_load(None)
            assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1"

        # 5. Device is None, MPS is unavailable -> leaves unset
        monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
            configure_mps_async_load(None)
            assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") is None

        # 6. User override preserved
        monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "custom_val")
        configure_mps_async_load("mps")
        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "custom_val"

    def test_sweep_mps_device_sets_async_load_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from soup_cli.commands.sweep import _run_single
        from soup_cli.config.schema import SoupConfig

        cfg = SoupConfig(
            base="owner/model",
            data={"train": "dummy.jsonl"},
            training={"quantization": "none"},
        )
        (tmp_path / "dummy.jsonl").write_text('{"text": "hi"}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        tracker = MagicMock()
        tracker.start_run.return_value = "run-1"
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "output_dir": str(tmp_path),
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "total_steps": 10,
            "duration_secs": 1.0,
        }

        env_at_build = []

        def capture_build_trainer(*args, **kwargs):
            env_at_build.append(os.environ.get("HF_DEACTIVATE_ASYNC_LOAD"))
            return mock_trainer

        dev_mock = ("mps", "Apple Silicon (MPS)")
        with patch("soup_cli.utils.gpu.detect_device", return_value=dev_mock), \
             patch("soup_cli.utils.gpu.get_gpu_info", return_value={}), \
             patch("soup_cli.experiment.tracker.ExperimentTracker", return_value=tracker), \
             patch("soup_cli.monitoring.display.TrainingDisplay"), \
             patch("soup_cli.trainer.dispatch.build_trainer", side_effect=capture_build_trainer), \
             patch("soup_cli.data.loader.load_dataset", return_value={"train": [{"text": "x"}]}):
            _run_single(cfg, {}, "sweep_1", None)

        assert env_at_build == ["1"], "HF_DEACTIVATE_ASYNC_LOAD must be set before trainer setup"
        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1"

    def test_bench_train_mps_device_sets_async_load_default(self) -> None:
        from soup_cli.bench.train_run import run_bench_train
        from soup_cli.config.schema import SoupConfig

        cfg = SoupConfig(
            base="owner/model",
            data={"train": "dummy.jsonl"},
            training={"quantization": "none"},
        )

        class StopAfterConfigureError(Exception):
            pass

        with patch("soup_cli.trainer.sft.SFTTrainerWrapper", side_effect=StopAfterConfigureError):
            with pytest.raises(StopAfterConfigureError):
                run_bench_train(
                    cfg,
                    steps=1,
                    warmup=0,
                    device="mps",
                    load_dataset=lambda c: {"train": [{"text": "hi"}]},
                )

        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1"

    def test_doctor_reports_mps_async_load_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        runner = CliRunner()
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain = strip_ansi(result.output)
        assert "MPS (Apple Silicon)" in plain
        assert "HF_DEACTIVATE_ASYNC_LOAD:" in plain
        assert "not set (soup train sets it to 1 on MPS)" in plain

    def test_doctor_reports_mps_async_load_user_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "custom_override")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        runner = CliRunner()
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain = strip_ansi(result.output)
        assert "MPS (Apple Silicon)" in plain
        assert "HF_DEACTIVATE_ASYNC_LOAD:" in plain
        assert "custom_override" in plain

    def test_doctor_reports_mps_async_load_escapes_markup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "[/oops]")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        runner = CliRunner()
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        plain = strip_ansi(result.output)
        assert "[/oops]" in plain

    def test_transformers_thread_pool_contract_unset_vs_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify transformers reads HF_DEACTIVATE_ASYNC_LOAD dynamically on CPU (#2076)."""
        import transformers.core_model_loading as cml
        from transformers import AutoModelForCausalLM, LlamaConfig

        model_dir = tmp_path / "tiny_llama"
        cfg = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        tiny_model = AutoModelForCausalLM.from_config(cfg)
        tiny_model.save_pretrained(str(model_dir))

        # 1. Unset -> ThreadPoolExecutor is used
        monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
        with patch.object(cml, "ThreadPoolExecutor", wraps=cml.ThreadPoolExecutor) as spy:
            AutoModelForCausalLM.from_pretrained(str(model_dir))
            assert spy.call_count >= 1, "Unset variable must instantiate ThreadPoolExecutor"

        # 2. Set to '1' -> ThreadPoolExecutor is deactivated
        monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
        with patch.object(cml, "ThreadPoolExecutor", wraps=cml.ThreadPoolExecutor) as spy:
            AutoModelForCausalLM.from_pretrained(str(model_dir))
            assert (
                spy.call_count == 0
            ), "HF_DEACTIVATE_ASYNC_LOAD=1 must deactivate ThreadPoolExecutor"

        # 3. Set to '0' -> ThreadPoolExecutor is used (working opt-out)
        monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "0")
        with patch.object(cml, "ThreadPoolExecutor", wraps=cml.ThreadPoolExecutor) as spy:
            AutoModelForCausalLM.from_pretrained(str(model_dir))
            assert (
                spy.call_count >= 1
            ), "HF_DEACTIVATE_ASYNC_LOAD=0 must re-enable ThreadPoolExecutor"
