"""``soup train --resume`` / ``--hf-resume`` on ``task: ppo`` is refused, not restarted.

trl's experimental ``PPOTrainer.train()`` takes no checkpoint, so a resume request used
to print one warning and train again from step 0, overwriting the progress already in the
output directory. The refusal now sits in ``commands/train.py:_resume_refusal`` -- the one
place both flags go through, before the dataset is read, a model is loaded or the output
directory is touched. The wrapper raises the same text if a caller reaches it directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.train import _resolve_resume_or_exit, _resume_refusal
from soup_cli.config.schema import SoupConfig
from tests.conftest import strip_ansi


def _plain(text: str) -> str:
    """CLI output without SGR escapes, whitespace collapsed (Rich colours and wraps it
    on Linux/macOS CI, under FORCE_COLOR=1 or a narrow COLUMNS)."""
    return " ".join(strip_ansi(text).split())


def _expected(flag: str) -> str:
    return (
        f"soup train {flag} is not supported for task: ppo (resume is not implemented for "
        "PPO). trl's PPOTrainer.train() takes no checkpoint, so the run would start over "
        "from step 0 and overwrite the progress already in the output directory. "
        f"Remove {flag} to start a new run, or point output at a new directory to keep "
        "the old one."
    )


def _config(task: str = "ppo", **overrides) -> SoupConfig:
    return SoupConfig(
        base="test-model",
        task=task,
        data={"train": "./data.jsonl"},
        **overrides,
    )


def _tree(root) -> dict[str, tuple[bytes | None, int]]:
    """Every path under ``root`` with its bytes (None for a directory) and mtime."""
    return {
        str(path.relative_to(root)): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
    }


@pytest.fixture
def ppo_run(tmp_path, monkeypatch):
    """A PPO config on disk, an output directory holding checkpoint-2 and a
    final adapter, and every loader the train command could reach replaced by a recorder."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "prompts.jsonl"
    data.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    out = tmp_path / "out"
    checkpoint = out / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step": 2}', encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter-bytes-of-step-2")
    (out / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    config = tmp_path / "soup.yaml"
    config.write_text(
        "base: test-model\n"
        "task: ppo\n"
        f"data: {{train: {data.as_posix()}, format: chatml, val_split: 0.0}}\n"
        "training: {epochs: 1, batch_size: 1, reward_model: test-reward-model}\n"
        f"output: {out.as_posix()}\n",
        encoding="utf-8",
    )

    reached: list[str] = []

    def _recorder(name):
        def _record(*_args, **_kwargs):
            reached.append(name)
            raise RuntimeError(f"{name} must not be reached before the resume refusal")

        return _record

    import transformers

    import soup_cli.commands.train as train_module
    import soup_cli.trainer.dispatch as dispatch_module
    import soup_cli.trainer.ppo as ppo_module

    monkeypatch.setattr(train_module, "load_dataset", _recorder("load_dataset"))
    monkeypatch.setattr(dispatch_module, "build_trainer", _recorder("build_trainer"))
    monkeypatch.setattr(ppo_module.PPOTrainerWrapper, "setup", _recorder("PPOTrainerWrapper.setup"))
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", _recorder("from_pretrained"),
    )
    return config, out, reached


class TestResumeRefusalFunction:
    @pytest.mark.parametrize("flag", ["--resume", "--hf-resume"])
    def test_ppo_refuses_each_flag_with_the_exact_message(self, flag):
        assert _resume_refusal(_config(), flag) == _expected(flag)

    def test_message_names_the_flag_the_task_and_the_way_out(self):
        message = _resume_refusal(_config(), "--resume")
        assert "--resume" in message
        assert "task: ppo" in message
        assert "Remove --resume to start a new run" in message

    @pytest.mark.parametrize("task", ["sft", "dpo", "kto", "orpo", "simpo", "ipo", "online_dpo"])
    def test_other_tasks_are_not_refused(self, task):
        kwargs = {}
        if task == "online_dpo":
            kwargs["training"] = {"online_dpo_judge": "pairwise"}
        assert _resume_refusal(_config(task, **kwargs), "--resume") is None

    def test_grpo_on_transformers_is_not_refused(self):
        assert _resume_refusal(_config("grpo"), "--resume") is None

    def test_unlearn_keeps_its_own_message(self):
        cfg = SoupConfig(
            base="test-model",
            task="unlearn",
            data={"train": "./d.jsonl", "forget_set": "./d.jsonl"},
            training={"unlearn_method": "npo"},
        )
        assert _resume_refusal(cfg, "--resume") == "--resume is not supported for task 'unlearn'"

    def test_no_resume_request_resolves_to_none_for_ppo(self):
        assert _resolve_resume_or_exit("", _config()) is None
        assert _resolve_resume_or_exit(None, _config()) is None


class TestTrainCommandRefusesBeforeAnythingLoads:
    @pytest.mark.parametrize("flag_args, flag", [
        (["--resume", "auto"], "--resume"),
        (["--resume", "{checkpoint}"], "--resume"),
        (["--hf-resume", "--push-as", "org/repo"], "--hf-resume"),
    ])
    def test_refuses_and_leaves_the_output_directory_untouched(
        self, ppo_run, flag_args, flag,
    ):
        config, out, reached = ppo_run
        args = [a.replace("{checkpoint}", str(out / "checkpoint-2")) for a in flag_args]
        before = _tree(out)

        result = CliRunner().invoke(app, ["train", "--config", str(config), *args, "--yes"])

        output = _plain(result.output)
        assert result.exit_code == 1, (output, repr(result.exception))
        assert _plain(_expected(flag)) in output
        assert "Resuming from" not in output
        assert reached == [], f"loaders reached before the refusal: {reached}"
        assert _tree(out) == before

    def test_auto_with_no_checkpoint_gets_the_ppo_refusal_not_a_missing_checkpoint(
        self, ppo_run,
    ):
        config, out, reached = ppo_run
        import shutil

        shutil.rmtree(out)

        result = CliRunner().invoke(
            app, ["train", "--config", str(config), "--resume", "auto", "--yes"],
        )

        output = _plain(result.output)
        assert result.exit_code == 1, (output, repr(result.exception))
        assert _plain(_expected("--resume")) in output
        assert "No checkpoint found" not in output
        assert reached == []
        assert not out.exists()


class TestWrapperBackstop:
    """A caller that reaches ``PPOTrainerWrapper.train`` without the CLI is refused too."""

    def _wrapper(self):
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        wrapper = PPOTrainerWrapper(_config(), device="cpu")
        wrapper._output_dir = "/tmp/test"
        wrapper._dataset_in_constructor = True
        wrapper._train_ds = MagicMock()
        wrapper.tokenizer = MagicMock()
        return wrapper

    def test_builtin_trainer_without_resume_support_raises_instead_of_restarting(self):
        from soup_cli.trainer.ppo import RESUME_REFUSAL

        wrapper = self._wrapper()
        calls = []
        mock_trainer = MagicMock()
        mock_trainer.train = lambda: calls.append("train")
        wrapper.trainer = mock_trainer

        with pytest.raises(ValueError) as excinfo:
            wrapper._train_builtin(
                display=None, tracker=None, run_id="", resume_from_checkpoint="/tmp/ckpt",
            )

        assert str(excinfo.value) == RESUME_REFUSAL
        assert calls == [], "train() must not run after the refusal"

    def test_manual_loop_raises_instead_of_dropping_the_checkpoint(self):
        from soup_cli.trainer.ppo import RESUME_REFUSAL

        wrapper = self._wrapper()
        wrapper.trainer = MagicMock(spec=["step"])  # trl < 0.28: step(), no train()
        wrapper._train_manual = MagicMock()

        with pytest.raises(ValueError) as excinfo:
            wrapper.train(resume_from_checkpoint="/tmp/ckpt")

        assert str(excinfo.value) == RESUME_REFUSAL
        wrapper._train_manual.assert_not_called()

    def test_the_wrapper_text_is_the_cli_text(self):
        from soup_cli.trainer.ppo import RESUME_REFUSAL

        assert _resume_refusal(_config(), "--resume") == RESUME_REFUSAL
        assert _plain(RESUME_REFUSAL) == _expected("--resume")
