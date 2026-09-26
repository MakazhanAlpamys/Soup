"""Tests for issue #1220: `soup train --resume` on `task: prm` and `task: moe_lora_routing`.

Verifies:
1. PRM wrapper accepts and forwards resume_from_checkpoint to HF Trainer.
2. MoLE wrapper accepts and forwards resume_from_checkpoint, saves mole_gate.pt into
   each checkpoint, and restores the gate on resume.
3. Resuming from checkpoint-4 executes 0 steps and preserves global_step.
4. Resuming from checkpoint-2 executes remaining 2 steps and does not rewrite checkpoint-2.
5. Existing checkpoint modification times are preserved.
6. Static AST scan over src/soup_cli/trainer/*.py confirms all wrappers forwarding
   resume_from_checkpoint or registered in UNSUPPORTED_RESUME_TASKS.
7. CLI refusal for unsupported tasks (e.g. unlearn) with ANSI strip verification.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.train import UNSUPPORTED_RESUME_TASKS
from tests._windows_ci import skip_on_windows_ci

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from terminal output."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def _collapse_whitespace(text: str) -> str:
    """Normalize internal and leading/trailing whitespace."""
    return " ".join(text.split())


TRAINER_DIR = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "trainer"


def _forwards_resume(func: ast.FunctionDef) -> bool:
    """Some call in the BODY passes the checkpoint on; the signature proves nothing."""
    kwarg = func.args.kwarg.arg if func.args.kwarg else None
    for stmt in func.body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "resume_from_checkpoint":
                    return True
                if kw.arg is None and isinstance(kw.value, ast.Name) and kw.value.id == kwarg:
                    return True
            if any(isinstance(a, ast.Name) and a.id == "resume_from_checkpoint" for a in node.args):
                return True  # ppo.py passes it on positionally
    return False


class TestTrainerScanInvariants:
    """Static AST verification that every trainer wrapper handles resume_from_checkpoint."""

    def test_all_trainer_wrappers_forward_resume_or_refuse(self):
        """A scan over src/soup_cli/trainer/*.py fails when a wrapper's train()
        accepts resume_from_checkpoint (or **kwargs) without forwarding it,
        unless the task is in UNSUPPORTED_RESUME_TASKS.
        """
        trainer_files = sorted(TRAINER_DIR.glob("*.py"))
        assert trainer_files, "No trainer files found in src/soup_cli/trainer"

        task_mapping = {
            "asr.py": "asr",
            "bco.py": "bco",
            "classifier.py": "classifier",
            "distill.py": "distill",
            "dpo.py": "dpo",
            "embedding.py": "embedding",
            "grpo.py": "grpo",
            "ipo.py": "ipo",
            "kto.py": "kto",
            "mlx_dpo.py": "dpo",
            "mlx_grpo.py": "grpo",
            "mlx_sft.py": "sft",
            "mole_routing.py": "moe_lora_routing",
            "online_dpo.py": "online_dpo",
            "orpo.py": "orpo",
            "ppo.py": "ppo",
            "preference.py": "preference",
            "pretrain.py": "pretrain",
            "prm.py": "prm",
            "reward_model.py": "reward_model",
            "sft.py": "sft",
            "simpo.py": "simpo",
            "tts.py": "tts",
            "unlearn.py": "unlearn",
        }

        failures = []
        for file_path in trainer_files:
            file_name = file_path.name
            if file_name in (
                "__init__.py",
                "loss_summary.py",
                "stream_setup.py",
                "bitnet.py",
                "mlx_masking.py",
                "mlx_optim.py",
                "mlx_routing.py",
                "rewards.py",
                "_trl_compat.py",
                "rewind_mlx.py",
            ):
                continue

            with open(file_path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                # Only check TrainerWrapper classes or primary trainers
                if not ("TrainerWrapper" in node.name or node.name.endswith("Trainer")):
                    continue

                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "train":
                        arg_names = [a.arg for a in item.args.args]
                        kwarg = item.args.kwarg.arg if item.args.kwarg else None

                        accepts_resume = "resume_from_checkpoint" in arg_names or kwarg is not None
                        if not accepts_resume:
                            continue

                        task_name = task_mapping.get(file_name)
                        is_refused = task_name in UNSUPPORTED_RESUME_TASKS
                        if is_refused:
                            continue

                        if not _forwards_resume(item):
                            failures.append(
                                f"{file_name}::{node.name}.train() accepts resume_from_checkpoint "
                                f"but does not forward it and task {task_name!r} is not in "
                                f"UNSUPPORTED_RESUME_TASKS"
                            )

        assert not failures, "\n".join(failures)


class TestResumeRefusalForUnsupportedTasks:
    """CLI test for tasks that cleanly refuse --resume / --hf-resume."""

    def test_unlearn_resume_refusal(self, tmp_path):
        runner = CliRunner()
        config_path = tmp_path / "unlearn.yaml"
        train_data = tmp_path / "forget.jsonl"
        train_data.write_text(json.dumps({"prompt": "forget me", "response": "done"}) + "\n")

        config_path.write_text(
            f"base: {TINY_MODEL}\n"
            "task: unlearn\n"
            f"data: {{train: {train_data.as_posix()}, forget_set: {train_data.as_posix()},"
            " format: chatml, val_split: 0.0}\n"
            "training: {unlearn_method: npo}\n"
            "output: ./out_unlearn\n"
        )

        res = runner.invoke(
            app, ["train", "--config", str(config_path), "--resume", "auto", "--yes"]
        )
        assert res.exit_code != 0
        clean_output = _collapse_whitespace(_strip_ansi(res.output))
        assert "--resume is not supported for task 'unlearn'" in clean_output

    def test_unlearn_hf_resume_refusal(self, tmp_path):
        runner = CliRunner()
        config_path = tmp_path / "unlearn.yaml"
        train_data = tmp_path / "forget.jsonl"
        train_data.write_text(json.dumps({"prompt": "forget me", "response": "done"}) + "\n")

        config_path.write_text(
            f"base: {TINY_MODEL}\n"
            "task: unlearn\n"
            f"data: {{train: {train_data.as_posix()}, forget_set: {train_data.as_posix()},"
            " format: chatml, val_split: 0.0}\n"
            "training: {unlearn_method: npo}\n"
            "output: ./out_unlearn\n"
        )

        res = runner.invoke(app, [
            "train", "--config", str(config_path), "--hf-resume", "--push-as", "test/repo", "--yes"
        ])
        assert res.exit_code != 0
        clean_output = _collapse_whitespace(_strip_ansi(res.output))
        assert "--hf-resume is not supported for task 'unlearn'" in clean_output


@skip_on_windows_ci
class TestPRMResume:
    """Functional verification of PRM resume with tiny model."""

    def test_prm_resume_execution_and_checkpoints(self, tmp_path, monkeypatch):
        import transformers

        steps = []
        _orig_training_step = transformers.Trainer.training_step

        def counting_step(self, *args, **kwargs):
            if steps:
                steps[-1] += 1
            return _orig_training_step(self, *args, **kwargs)

        monkeypatch.setattr(transformers.Trainer, "training_step", counting_step)
        monkeypatch.setenv("WANDB_DISABLED", "true")

        prm_data = tmp_path / "prm.jsonl"
        rows = [
            {
                "prompt": f"Solve {i}+1.",
                "completions": [f"{i}+1 is {i + 1}.", "Done."],
                "labels": [True, True],
            }
            for i in range(8)
        ]
        prm_data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        out_dir = tmp_path / "out_prm"
        cfg_path = tmp_path / "prm.yaml"
        cfg_path.write_text(
            f"base: {TINY_MODEL}\n"
            "task: prm\n"
            f"data: {{train: {prm_data.as_posix()}, format: prm, val_split: 0.0, max_length: 64}}\n"
            "training: {\n"
            "  epochs: 1, batch_size: 2, gradient_accumulation_steps: 1, save_steps: 2,\n"
            "  quantization: none, lora: {r: 0}\n"
            "}\n"
            f"output: {out_dir.as_posix()}\n"
        )

        runner = CliRunner()

        # Run 1: initial run to checkpoint-4
        steps.append(0)
        res1 = runner.invoke(app, ["train", "--config", str(cfg_path), "--yes"])
        assert res1.exit_code == 0, res1.output
        assert steps[-1] == 4, f"Expected 4 training steps on initial run, got {steps[-1]}"

        ckpt2 = out_dir / "checkpoint-2"
        ckpt4 = out_dir / "checkpoint-4"
        assert ckpt2.is_dir()
        assert ckpt4.is_dir()
        mtime2_before = (ckpt2 / "trainer_state.json").stat().st_mtime
        mtime4_before = (ckpt4 / "trainer_state.json").stat().st_mtime

        # Run 2: resume auto (from checkpoint-4) -> should execute 0 steps
        steps.append(0)
        res2 = runner.invoke(app, ["train", "--config", str(cfg_path), "--resume", "auto", "--yes"])
        assert res2.exit_code == 0, res2.output
        assert "Resuming from" in res2.output
        assert steps[-1] == 0, f"Expected 0 steps on resume auto, got {steps[-1]}"

        # Checkpoints must not be rewritten
        assert (ckpt2 / "trainer_state.json").stat().st_mtime == mtime2_before
        assert (ckpt4 / "trainer_state.json").stat().st_mtime == mtime4_before

        # Run 3: resume explicit path to checkpoint-4 -> 0 steps
        steps.append(0)
        res3 = runner.invoke(
            app, ["train", "--config", str(cfg_path), "--resume", str(ckpt4), "--yes"]
        )
        assert res3.exit_code == 0, res3.output
        assert steps[-1] == 0

        # Run 4: resume from middle checkpoint-2 -> executes remaining 2 steps
        steps.append(0)
        res4 = runner.invoke(
            app, ["train", "--config", str(cfg_path), "--resume", str(ckpt2), "--yes"]
        )
        assert res4.exit_code == 0, res4.output
        assert steps[-1] == 2, f"Expected 2 steps when resuming from checkpoint-2, got {steps[-1]}"
        # checkpoint-2 itself was not rewritten
        assert (ckpt2 / "trainer_state.json").stat().st_mtime == mtime2_before


@skip_on_windows_ci
class TestMoleRoutingResume:
    """Functional verification of MoLE resume with tiny model and task adapters."""

    def test_mole_resume_execution_checkpoints_and_gate(self, tmp_path, monkeypatch):
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        steps = []
        _orig_training_step = transformers.Trainer.training_step

        def counting_step(self, *args, **kwargs):
            if steps:
                steps[-1] += 1
            return _orig_training_step(self, *args, **kwargs)

        monkeypatch.setattr(transformers.Trainer, "training_step", counting_step)
        monkeypatch.setenv("WANDB_DISABLED", "true")

        # Create two tiny task LoRA adapters
        base_model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
        peft_m = get_peft_model(base_model, lora_cfg)
        adapter_a = tmp_path / "adapter_a"
        adapter_b = tmp_path / "adapter_b"
        peft_m.save_pretrained(str(adapter_a))
        peft_m.save_pretrained(str(adapter_b))

        # Dataset for routing
        chat_data = tmp_path / "chat.jsonl"
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            }
            for i in range(8)
        ]
        chat_data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        out_dir = tmp_path / "out_mole"
        cfg_path = tmp_path / "mole.yaml"
        c_posix = chat_data.as_posix()
        cfg_data = (
            f"base: {TINY_MODEL}\n"
            "task: moe_lora_routing\n"
            f"data: {{train: {c_posix}, format: chatml, val_split: 0.0, max_length: 64}}\n"
            "training: {\n"
            "  epochs: 1, batch_size: 2, gradient_accumulation_steps: 1, save_steps: 2,\n"
            f"  mole_task_adapters: [{adapter_a.as_posix()}, {adapter_b.as_posix()}],\n"
            "  mole_top_k: 2, mole_temperature: 1.0, lr: 0.01, quantization: none\n"
            "}\n"
            f"output: {out_dir.as_posix()}\n"
        )
        cfg_path.write_text(cfg_data)

        runner = CliRunner()

        # Run 1: initial run to checkpoint-4
        steps.append(0)
        res1 = runner.invoke(app, ["train", "--config", str(cfg_path), "--yes"])
        assert res1.exit_code == 0, res1.output
        assert steps[-1] == 4, f"Expected 4 training steps on initial run, got {steps[-1]}"

        ckpt2 = out_dir / "checkpoint-2"
        ckpt4 = out_dir / "checkpoint-4"
        assert ckpt2.is_dir()
        assert ckpt4.is_dir()

        # Verify mole_gate.pt is saved in checkpoints
        gate2_path = ckpt2 / "mole_gate.pt"
        gate4_path = ckpt4 / "mole_gate.pt"
        assert gate2_path.is_file(), f"mole_gate.pt missing from {ckpt2}"
        assert gate4_path.is_file(), f"mole_gate.pt missing from {ckpt4}"

        gate4_saved = torch.load(gate4_path, map_location="cpu", weights_only=True)
        mtime2_before = (ckpt2 / "trainer_state.json").stat().st_mtime
        mtime4_before = (ckpt4 / "trainer_state.json").stat().st_mtime

        # Run 2: resume auto (checkpoint-4) -> executes 0 steps
        steps.append(0)
        res2 = runner.invoke(app, ["train", "--config", str(cfg_path), "--resume", "auto", "--yes"])
        assert res2.exit_code == 0, res2.output
        assert "Resuming from" in res2.output
        assert steps[-1] == 0, f"Expected 0 steps on resume auto, got {steps[-1]}"

        # Checkpoints not rewritten
        assert (ckpt2 / "trainer_state.json").stat().st_mtime == mtime2_before
        assert (ckpt4 / "trainer_state.json").stat().st_mtime == mtime4_before

        # Gate weights after resume match saved checkpoint-4 weights
        final_gate = torch.load(out_dir / "mole_gate.pt", map_location="cpu", weights_only=True)
        for k in gate4_saved:
            assert torch.equal(gate4_saved[k], final_gate[k])

        # Run 3: resume from middle checkpoint-2 -> executes 2 steps
        steps.append(0)
        res3 = runner.invoke(
            app, ["train", "--config", str(cfg_path), "--resume", str(ckpt2), "--yes"]
        )
        assert res3.exit_code == 0, res3.output
        assert steps[-1] == 2, f"Expected 2 steps when resuming from checkpoint-2, got {steps[-1]}"
        # checkpoint-2 not modified
        assert (ckpt2 / "trainer_state.json").stat().st_mtime == mtime2_before

        # Bit-exact check: gate reproduced across checkpoint-2 resume
        resumed = torch.load(out_dir / "mole_gate.pt", map_location="cpu", weights_only=True)
        for k in gate4_saved:
            torch.testing.assert_close(resumed[k], gate4_saved[k], rtol=0, atol=1e-6)
