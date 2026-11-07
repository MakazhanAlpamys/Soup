"""#2585 -- layer streaming under a distributed launch (WORLD_SIZE > 1) is a decision.

`refuse_if_data_parallel` returns early under torchrun / accelerate / deepspeed,
because `nn.DataParallel` is never applied there. That left a distributed launch
with no check at all, although the decoder's base weights stay on `meta`, which DDP
refuses, and where the launcher does not create a process group each process would
train its own unsynchronised copy. `refuse_if_distributed_launch` closes that gap, at
two points. It first refused every WORLD_SIZE > 1; since the adapter-gradient sync
(`utils/adapter_grad_sync.py`) exists it lets ONE slice through as BETA (task=sft, one
rank per host, nothing that shards, resumes, pushes or gates) and refuses the rest by
name. This file keeps the refusals and the single-process controls from #2586; the
whole matrix of refused items is in `test_stream_distributed_trainer.py`.

What changed with that decision: a WORLD_SIZE > 1 launch of an SFT config with
`LOCAL_WORLD_SIZE=1` used to be refused and now streams, so every case that stood for
"a distributed launch" now carries a blocker (`LOCAL_WORLD_SIZE` above one, or a
preference task) to stay a refusal, and the guard takes the config. The message text
changed (it names the blocking items); its phrases, the issue number and the exit
behaviour did not.

* `soup train`, right after the config is loaded -- before the dataset, the
  Hub resume pull, the run record or any model work;
* `StreamingSetupMixin._setup_streaming_transformers`, first statement -- for the
  callers that never go through the command (`soup sweep`, the web UI, the API).

It is deliberately NOT a config-load validator: the same config is read by
`soup validate` and `soup export` in shells where a launcher variable may be left
over, and those must not be refused.

CLI output is compared through an ANSI-strip + whitespace-collapse helper: Rich
puts colour escapes inside words on Linux/macOS CI (FORCE_COLOR=1 reproduces it).
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.trainer.stream_setup import (
    DISTRIBUTED_STREAM_ISSUE,
    _distributed_launch,
    refuse_if_data_parallel,
    refuse_if_distributed_launch,
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

#: The sentence a refused user reads. Pinned whole: it names the setting, the
#: launch, the blocking items, the tracking issue, what is allowed and the ways out.
_MESSAGE = (
    "training.stream_layers=true under a distributed launch (WORLD_SIZE={size}): "
    "distributed layer streaming is not supported yet with {items}, see #2585. "
    "It is BETA for task='sft' on one rank per host, without --deepspeed, --fsdp, --resume, "
    "--hf-resume, --push-as, an eval gate (--gate / training.eval_gate), training.multipack, "
    "training.moe_lora or data.format='raft'. "
    "Run on one process (no torchrun/accelerate launch), set stream_layers: false, "
    "or remove the blocking setting."
)

#: The blocking item of a launch whose ranks share a host (the whole world on one host).
_SHARED_HOST = "LOCAL_WORLD_SIZE={size} (more than one rank per host)"

#: Every spelling of "not a distributed launch" that must keep today's behaviour.
_NOT_DISTRIBUTED = ["1", "0", "-1", "-8", "", " ", "abc", "2.5", "1e3", "two", "0x2"]


def _plain(text: str | None) -> str:
    """Rich colours per character and wraps, so compare stripped + collapsed."""
    return " ".join(_ANSI.sub("", text or "").replace("│", " ").split())


class _PastTheGuardError(Exception):
    """Raised by a sentinel placed just AFTER the guard: reaching it proves no refusal."""


class _ReachedModelWorkError(AssertionError):
    """Raised by a sentinel on model download / sharding: reaching it means the guard was late."""


# ==========================================================================
# the guard itself (pure -- no torch, no GPU)
# ==========================================================================
class TestRefuseIfDistributedLaunch:
    """Every case that used to stand for "any distributed launch" now sets
    `LOCAL_WORLD_SIZE` to the whole world (all ranks on one host), the blocker that keeps
    it a refusal; the guard takes the config since it now decides on it."""

    @pytest.mark.parametrize("size", [2, 3, 8, 64])
    def test_refuses_when_world_size_is_above_one(self, tmp_path, monkeypatch, size):
        monkeypatch.setenv("WORLD_SIZE", str(size))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(size))
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_config(tmp_path, "sft"))
        assert _plain(str(excinfo.value)) == _MESSAGE.format(
            size=size, items=_SHARED_HOST.format(size=size)
        )

    def test_message_carries_the_phrase_the_tests_and_users_grep_for(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_config(tmp_path, "sft"))
        message = _plain(str(excinfo.value))
        assert "distributed layer streaming is not supported yet" in message
        assert f"#{DISTRIBUTED_STREAM_ISSUE}" in message
        assert DISTRIBUTED_STREAM_ISSUE == 2585
        assert "WORLD_SIZE=2" in message
        assert "training.stream_layers=true" in message
        assert "stream_layers: false" in message
        assert "no torchrun/accelerate launch" in message
        assert "BETA" in message, "a refused user is told what IS allowed"

    def test_whitespace_around_the_number_is_parsed_like_the_dataparallel_guard(
        self, tmp_path, monkeypatch
    ):
        """`int(" 4 ")` is 4, so `_distributed_launch()` says distributed; the new guard
        must agree with it rather than carry a second parser."""
        monkeypatch.setenv("WORLD_SIZE", " 4 ")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_config(tmp_path, "sft"))
        assert "WORLD_SIZE=4" in str(excinfo.value)

    @pytest.mark.parametrize("value", _NOT_DISTRIBUTED)
    def test_not_distributed_values_are_not_refused(self, tmp_path, monkeypatch, value):
        """Control. Without it a guard that always raises would pass the test above and
        break every single-process run. A malformed value keeps today's behaviour."""
        monkeypatch.setenv("WORLD_SIZE", value)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        assert refuse_if_distributed_launch(_config(tmp_path, "dpo")) is False

    def test_unset_world_size_is_not_refused(self, tmp_path, monkeypatch):
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        assert refuse_if_distributed_launch(_config(tmp_path, "dpo")) is False

    @pytest.mark.parametrize("value", [None, *_NOT_DISTRIBUTED, "2", " 4 ", "16"])
    def test_refuses_exactly_when_the_dataparallel_guard_calls_it_distributed(
        self, tmp_path, monkeypatch, value
    ):
        """One definition of "distributed launch" for both guards (the config here always
        carries a blocker, so a distributed launch is always refused)."""
        if value is None:
            monkeypatch.delenv("WORLD_SIZE", raising=False)
        else:
            monkeypatch.setenv("WORLD_SIZE", value)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
        refused = False
        try:
            refuse_if_distributed_launch(_config(tmp_path, "sft"))
        except ValueError:
            refused = True
        assert refused is _distributed_launch()

    def test_one_rank_per_host_is_the_slice_that_is_no_longer_refused(self, tmp_path, monkeypatch):
        """CHANGED from #2586, which refused every WORLD_SIZE > 1: an SFT config launched with
        one rank per host now streams (BETA), and the guard says so by returning True."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
        assert refuse_if_distributed_launch(_config(tmp_path, "sft")) is True


class TestDataParallelGuardIsUnchanged:
    """`refuse_if_data_parallel` keeps its own contract (see test_stream_multi_gpu_guard);
    the new guard is separate and must not leak into it."""

    @staticmethod
    def _fake_cuda(monkeypatch, count):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)

    def test_still_refuses_several_visible_gpus_without_a_launcher(self, monkeypatch):
        self._fake_cuda(monkeypatch, 4)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        with pytest.raises(ValueError) as excinfo:
            refuse_if_data_parallel("cuda")
        message = str(excinfo.value)
        assert "4 CUDA devices are visible" in message
        assert "DataParallel" in message
        assert "CUDA_VISIBLE_DEVICES=0" in message
        assert "distributed layer streaming" not in message

    def test_still_lets_a_distributed_launch_through_on_its_own(self, monkeypatch):
        """The DataParallel concern does not apply under a launcher; refusing the
        distributed case is the job of `refuse_if_distributed_launch`, not of this one."""
        self._fake_cuda(monkeypatch, 8)
        monkeypatch.setenv("WORLD_SIZE", "8")
        refuse_if_data_parallel("cuda")

    def test_still_ignores_cpu_and_single_gpu(self, monkeypatch):
        self._fake_cuda(monkeypatch, 8)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        refuse_if_data_parallel("cpu")
        self._fake_cuda(monkeypatch, 1)
        refuse_if_data_parallel("cuda")


# ==========================================================================
# setup: every streaming wrapper refuses before any model work
# ==========================================================================
_WRAPPERS = {
    "sft": ("soup_cli.trainer.sft", "SFTTrainerWrapper"),
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper"),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper"),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper"),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper"),
}


def _config(tmp_path, task: str, *, stream_layers: bool = True):
    from soup_cli.config.loader import load_config_from_string

    training = {
        # KTO is refused below batch 2 (its KL term is degenerate at 1).
        "batch_size": 2 if task == "kto" else 1,
        "gradient_accumulation_steps": 1,
        "quantization": "none",
        "epochs": 1,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    if stream_layers:
        training["stream_layers"] = True
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": str(tmp_path / "never-downloaded-base"),
                "task": task,
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": training,
                "output": str(tmp_path / "out"),
            }
        )
    )


def _wrapper(tmp_path, task: str):
    import importlib

    module_name, class_name = _WRAPPERS[task]
    wrapper_cls = getattr(importlib.import_module(module_name), class_name)
    return wrapper_cls(_config(tmp_path, task), device="cpu")


def _trap_model_work(monkeypatch):
    """Everything `_setup_streaming_transformers` does before the model exists
    (config fetch, tokenizer, weight resolve, shard write) now raises: a guard
    that fires late is caught by `_ReachedModelWorkError` instead of by a download."""
    import transformers

    def _boom(*_args, **_kwargs):
        raise _ReachedModelWorkError("the setup reached model work")

    monkeypatch.setattr("soup_cli.utils.kimi_k2.load_stream_model_config", _boom)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _boom)
    monkeypatch.setattr("soup_cli.utils.spectrum_scan.resolve_model_weights", _boom)
    monkeypatch.setattr("soup_cli.utils.layer_shard.shard_checkpoint", _boom)


_DATASET = {"train": [{"messages": [{"role": "user", "content": "hi"}]}]}


class TestSetupRefusesBeforeAnyModelWork:
    @pytest.mark.parametrize("task", sorted(_WRAPPERS))
    def test_refuses_under_a_distributed_launch(self, tmp_path, monkeypatch, task):
        """Both blockers at once for a preference task (the task, and ranks sharing a host);
        for sft only the shared host, since sft with one rank per host now streams."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
        _trap_model_work(monkeypatch)
        wrapper = _wrapper(tmp_path, task)
        with pytest.raises(ValueError) as excinfo:
            wrapper.setup(_DATASET)
        items = _SHARED_HOST.format(size=2)
        if task != "sft":
            items = f"task='{task}', {items}"
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items=items)

    @pytest.mark.parametrize("task", ["dpo", "kto", "orpo", "simpo"])
    def test_a_preference_task_is_refused_even_with_one_rank_per_host(
        self, tmp_path, monkeypatch, task
    ):
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
        _trap_model_work(monkeypatch)
        wrapper = _wrapper(tmp_path, task)
        with pytest.raises(ValueError) as excinfo:
            wrapper.setup(_DATASET)
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items=f"task='{task}'")

    def test_sft_with_one_rank_per_host_reaches_the_model(self, tmp_path, monkeypatch):
        """CHANGED from #2586, which refused this: the sentinel on model work is now reached,
        i.e. the decision let the setup go on."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
        _trap_model_work(monkeypatch)
        wrapper = _wrapper(tmp_path, "sft")
        with pytest.raises(_ReachedModelWorkError):
            wrapper.setup(_DATASET)

    @pytest.mark.parametrize("task", sorted(_WRAPPERS))
    def test_nothing_is_written_before_the_refusal(self, tmp_path, monkeypatch, task):
        """No shard directory, no cache root, no output directory."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
        cache = tmp_path / "cache"
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(cache))
        _trap_model_work(monkeypatch)
        wrapper = _wrapper(tmp_path, task)
        with pytest.raises(ValueError, match="distributed layer streaming is not supported"):
            wrapper.setup(_DATASET)
        assert not cache.exists()
        assert not (tmp_path / "out").exists()
        assert not (tmp_path / "never-downloaded-base").exists()

    @pytest.mark.parametrize("value", [None, "1", "", "abc", "-3"])
    @pytest.mark.parametrize("task", ["sft", "dpo"])
    def test_a_single_process_is_not_refused(self, tmp_path, monkeypatch, task, value):
        """Control: the sentinel on model work is reached, i.e. the guard let the setup go on."""
        if value is None:
            monkeypatch.delenv("WORLD_SIZE", raising=False)
        else:
            monkeypatch.setenv("WORLD_SIZE", value)
        _trap_model_work(monkeypatch)
        wrapper = _wrapper(tmp_path, task)
        with pytest.raises(_ReachedModelWorkError):
            wrapper.setup(_DATASET)


# ==========================================================================
# soup train: the earliest point that knows WORLD_SIZE
# ==========================================================================
def _write_config(tmp_path, *, stream_layers: bool) -> str:
    cfg = tmp_path / "soup.yaml"
    streaming = "  stream_layers: true\n" if stream_layers else ""
    cfg.write_text(
        "base: smollm2-135m\n"
        "task: sft\n"
        "backend: transformers\n"
        "data:\n"
        "  train: /dev/null\n"
        "training:\n"
        f"{streaming}"
        "  batch_size: 1\n"
    )
    return str(cfg)


def _sentinel_after_the_guard(monkeypatch):
    """`_resolve_resume_or_exit` is the first step after the guard's position; a run that
    gets there was not refused."""
    from soup_cli.commands import train as train_command

    def _past(*_args, **_kwargs):
        raise _PastTheGuardError

    monkeypatch.setattr(train_command, "_resolve_resume_or_exit", _past)


def _no_run_record(monkeypatch):
    from soup_cli.experiment.tracker import ExperimentTracker

    start_run = MagicMock()
    monkeypatch.setattr(ExperimentTracker, "start_run", start_run)
    return start_run


class TestTrainCommandRefusesEarly:
    @pytest.mark.parametrize("dry_run", [False, True], ids=["train", "dry-run"])
    def test_refused_right_after_the_config_is_loaded(self, tmp_path, monkeypatch, dry_run):
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
        monkeypatch.setenv("FORCE_COLOR", "1")
        start_run = _no_run_record(monkeypatch)
        _sentinel_after_the_guard(monkeypatch)
        args = ["train", "--config", _write_config(tmp_path, stream_layers=True)]
        args.append("--dry-run" if dry_run else "--yes")

        result = CliRunner().invoke(app, args)
        output = _plain(result.output)

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert not isinstance(result.exception, _PastTheGuardError), "the guard let the run through"
        assert _MESSAGE.format(size=2, items=_SHARED_HOST.format(size=2)) in output, output
        assert "Loaded:" not in output, output
        assert "Run ID" not in output, output
        assert "Config valid" not in output, output
        assert not start_run.called

    def test_the_refusal_is_a_clean_exit_not_a_traceback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        _sentinel_after_the_guard(monkeypatch)
        result = CliRunner().invoke(
            app, ["train", "--config", _write_config(tmp_path, stream_layers=True), "--yes"]
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        assert "Traceback" not in result.output
        assert "WORLD_SIZE=4" in _plain(result.output)

    def test_one_rank_per_host_is_no_longer_refused(self, tmp_path, monkeypatch):
        """CHANGED from #2586, which refused every WORLD_SIZE > 1: the sentinel placed just
        after the guard is reached, and no refusal is printed."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
        _sentinel_after_the_guard(monkeypatch)
        result = CliRunner().invoke(
            app, ["train", "--config", _write_config(tmp_path, stream_layers=True), "--yes"]
        )
        assert isinstance(result.exception, _PastTheGuardError), (
            result.output, repr(result.exception)
        )
        assert "distributed layer streaming" not in _plain(result.output)

    def test_stream_layers_off_is_not_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "2")
        _sentinel_after_the_guard(monkeypatch)
        result = CliRunner().invoke(
            app, ["train", "--config", _write_config(tmp_path, stream_layers=False), "--yes"]
        )
        assert isinstance(result.exception, _PastTheGuardError), (
            result.output, repr(result.exception)
        )
        assert "distributed layer streaming" not in _plain(result.output)

    @pytest.mark.parametrize("value", [None, "1", "", "abc"])
    def test_a_single_process_is_not_refused(self, tmp_path, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("WORLD_SIZE", raising=False)
        else:
            monkeypatch.setenv("WORLD_SIZE", value)
        _sentinel_after_the_guard(monkeypatch)
        result = CliRunner().invoke(
            app, ["train", "--config", _write_config(tmp_path, stream_layers=True), "--yes"]
        )
        assert isinstance(result.exception, _PastTheGuardError), (
            result.output, repr(result.exception)
        )
        assert "distributed layer streaming" not in _plain(result.output)

    def test_config_load_is_not_environment_dependent(
        self, tmp_path, monkeypatch
    ):
        """Why this is not a config-load validator: the same config must still load and
        validate in a shell that has WORLD_SIZE left over."""
        from soup_cli.config.loader import load_config

        monkeypatch.setenv("WORLD_SIZE", "8")
        cfg = load_config(_write_config(tmp_path, stream_layers=True))
        assert cfg.training.stream_layers is True
