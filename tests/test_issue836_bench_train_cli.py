"""``soup bench train`` through a real trainer on a real tiny model (#836).

The pure builder and the collector have their own unit tests. These drive the
whole path -- config, dataset, SFTTrainerWrapper, transformers' Trainer -- and
break the run on purpose, so each check is shown to fire on a real training
loop, and a control shows the same config passes when nothing is broken.
"""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")
    from tests.test_issue341_seed_and_fullft import _ROWS, _tiny_llama_dir

    base = _tiny_llama_dir(tmp_path)
    rows = [
        {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
        for u, a in _ROWS
    ]
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (tmp_path / "soup.yaml").write_text(
        yaml.safe_dump({
            "base": base,
            "task": "sft",
            "backend": "transformers",
            "modality": "text",
            "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
            "training": {
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "quantization": "none",
                "lr": 1e-2,
                "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj"]},
            },
            "output": str(tmp_path / "out"),
        }),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda backend=None: ("cpu", "CPU"))
    return tmp_path


def _run(steps=5, warmup=1):
    from soup_cli.bench.train_run import run_bench_train
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset

    def go(before_train=None):
        return run_bench_train(
            load_config("soup.yaml"), steps=steps, warmup=warmup, device="cpu",
            load_dataset=lambda c: load_dataset(c.data), before_train=before_train,
        )

    return go


def _failed(report):
    return {f["check"] for f in report["failures"]}


class TestTheCommand:
    def test_the_control_trains_and_writes_a_valid_report(self, workdir, monkeypatch):
        import subprocess

        spawned = []
        real_popen = subprocess.Popen.__init__

        def watching(self, args, *a, **k):
            listed = isinstance(args, (list, tuple))
            spawned.append(" ".join(map(str, args)) if listed else str(args))
            return real_popen(self, args, *a, **k)

        # Memory is torch's allocator counters, never nvidia-smi: watched on
        # the real run rather than grepped out of the source.
        monkeypatch.setattr(subprocess.Popen, "__init__", watching)
        result = runner.invoke(
            app, ["bench", "train", "--config", "soup.yaml", "--steps", "5",
                  "--warmup", "1", "-o", "r.json"],
        )
        assert result.exit_code == 0, result.output
        report = json.loads((workdir / "r.json").read_text(encoding="utf-8"))
        assert report["valid"] is True, report["failures"]
        assert report["steps_measured"] == 5
        assert report["timing"]["counted_steps"] == 4
        assert report["timing"]["warmup_steps_discarded"] == 1
        assert report["checks"]["grad_norm"] == "all reported steps finite and non-zero"
        assert report["checks"]["parameters_changed"] is True
        assert 0 < report["tokens"]["useful"] < report["tokens"]["total"]
        assert set(report["memory"]) == {
            "max_memory_allocated_bytes", "max_memory_reserved_bytes"
        }
        assert len(report["config_hash"]) == 64
        assert report["resolved_config"]["training"]["logging_steps"] == 1
        assert not [cmd for cmd in spawned if "nvidia-smi" in cmd], spawned
        assert "output" not in report["resolved_config"]
        # A benchmark never writes into the config's own output directory.
        assert not (workdir / "out").exists()

    def test_a_run_that_does_not_train_exits_non_zero_and_still_writes(
        self, workdir, monkeypatch
    ):
        """A zero learning rate: the backend reports real norms and nothing
        moves, so only the fingerprint can see it -- and the report is kept as
        evidence. The schema refuses ``lr: 0``, so it is set on the real
        trainer after setup, before the optimizer is built."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        setup = SFTTrainerWrapper.setup

        def setup_then_stall(self, dataset):
            setup(self, dataset)
            self.trainer.args.learning_rate = 0.0

        monkeypatch.setattr(SFTTrainerWrapper, "setup", setup_then_stall)
        result = runner.invoke(
            app, ["bench", "train", "--config", "soup.yaml", "--steps", "3",
                  "--warmup", "1", "-o", "r.json"],
        )
        assert result.exit_code == 1, result.output
        report = json.loads((workdir / "r.json").read_text(encoding="utf-8"))
        assert report["valid"] is False
        assert _failed(report) == {"parameters_changed"}
        assert "NOT valid" in result.output

    def test_a_task_it_does_not_measure_is_refused_by_name(self, workdir):
        cfg = yaml.safe_load((workdir / "soup.yaml").read_text(encoding="utf-8"))
        cfg["task"] = "dpo"
        cfg["data"]["format"] = "dpo"
        (workdir / "soup.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        result = runner.invoke(app, ["bench", "train", "--config", "soup.yaml"])
        assert result.exit_code == 1
        assert "task: dpo" in result.output

    def test_warmup_that_eats_every_step_is_refused(self, workdir):
        result = runner.invoke(
            app, ["bench", "train", "--config", "soup.yaml", "--steps", "2", "--warmup", "2"],
        )
        assert result.exit_code == 1
        assert "--warmup" in result.output


class TestTheConfigHash:
    def test_two_runs_of_one_config_hash_the_same(self, workdir):
        """The hash is there to show a config change moved a run, so the
        per-run scratch directory must not be part of it."""
        go = _run(steps=2, warmup=1)
        assert go()["config_hash"] == go()["config_hash"]


class TestTheChecksFireOnARealTrainer:
    def test_zero_trainable_parameters_fails_naming_the_cause(self, workdir):
        def freeze(wrapper):
            for param in wrapper.trainer.model.parameters():
                param.requires_grad_(False)

        with pytest.raises(ValueError, match="0 trainable parameter tensors"):
            _run()(before_train=freeze)

    def test_gradients_forced_to_zero_fail_on_grad_norm(self, workdir):
        import torch

        def zero_grads(wrapper):
            for param in wrapper.trainer.model.parameters():
                if param.requires_grad:
                    param.register_hook(torch.zeros_like)

        report = _run()(before_train=zero_grads)
        assert report["valid"] is False
        assert "grad_norm" in _failed(report)
        assert "grad_norm == 0.0" in next(
            f["message"] for f in report["failures"] if f["check"] == "grad_norm"
        )

    def test_without_a_norm_the_fingerprint_still_fails_a_run_that_did_not_train(
        self, workdir
    ):
        """The MLX/DeepSpeed shape: no grad_norm at all, nothing moved."""

        def no_norm_no_update(wrapper):
            trainer = wrapper.trainer
            trainer._get_grad_norm = lambda model, grad_norm=None: None
            trainer.args.learning_rate = 0.0

        report = _run()(before_train=no_norm_no_update)
        assert report["checks"]["grad_norm"] == "not reported by this backend"
        assert _failed(report) == {"parameters_changed"}

    def test_without_a_norm_a_run_that_trains_passes(self, workdir):
        """The control for the one above: an absent norm alone is not a failure."""

        def no_norm(wrapper):
            wrapper.trainer._get_grad_norm = lambda model, grad_norm=None: None

        report = _run()(before_train=no_norm)
        assert report["checks"]["grad_norm"] == "not reported by this backend"
        assert report["valid"] is True, report["failures"]


class TestInferStillAnswersToTheOldForm:
    """``soup bench <model>`` is in the README and in shell history. A group
    that must keep accepting a bare argument is the shape that regresses when a
    third subcommand lands, so each form is pinned to reach ``infer``."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["bench", "nonexistent_model_path"],
            ["bench", "infer", "nonexistent_model_path"],
            ["bench", "--max-tokens", "8", "nonexistent_model_path"],
        ],
    )
    def test_it_reaches_infer(self, argv, monkeypatch):
        seen = {}

        def fake_resolve(model):
            seen["model"] = model
            raise FileNotFoundError("stop here")

        monkeypatch.setattr(
            "soup_cli.commands.infer._resolve_model_source", fake_resolve
        )
        result = runner.invoke(app, argv)
        assert seen == {"model": "nonexistent_model_path"}, result.output
        assert result.exit_code == 1

    def test_the_group_help_lists_both(self):
        result = runner.invoke(app, ["bench", "--help"])
        assert result.exit_code == 0
        assert "infer" in result.output and "train" in result.output
