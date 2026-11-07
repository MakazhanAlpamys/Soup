"""Data-parallel layer streaming: the trainer hook, the DDP bypass, the decision (C1, C3, C4).

A streamed model keeps its base on ``meta``, so accelerate cannot wrap it in DDP and Soup
averages the LoRA gradients itself (``utils/adapter_grad_sync.py``). The pieces under test
here connect that sync to a real run:

* ``TestAdapterSyncTrainerClass`` (C3): the ``training_step`` override runs the sync after
  the backward of the last micro-batch of a window, once per optimizer step, and before
  HF clips (a capability test pins the order in transformers' loop).
* ``TestAccelerateBypass`` (C4): the production helper sets ``_is_accelerate_prepared``,
  and a model carrying it comes back from ``prepare_model`` as the same object.
* ``TestDecision`` / ``TestSetupDecision`` / ``TestTrainCommandDecision`` (C1, T4): what a
  distributed launch may stream (task=sft, one rank per host, ...) and each refusal, with
  the message a user would grep, in process (``WORLD_SIZE``/``LOCAL_WORLD_SIZE`` set by
  monkeypatch, no process group), at the function, at the SFT setup (before any model
  work) and at ``soup train``.
* ``TestEquivalence`` and friends (T1, T1a, T1b, T1c): two real processes, gloo over
  127.0.0.1, accelerate in its distributed state (``MULTI_CPU``, group created by
  accelerate from ``MASTER_ADDR``/``MASTER_PORT``), against ONE process with twice the
  per-device batch on the union of the rows. Each process runs the production setup
  (``SFTTrainerWrapper.setup`` -> ``_setup_streaming_transformers`` ->
  ``build_streamed_model`` -> the SFT trainer class, on a tiny Llama) and then
  transformers' own ``Trainer.train()``. Five SGD steps, fp32, ``lora_dropout=0``, a
  sequential sampler (rows in a fixed order); the adapters are compared after every step.

What equality means, per tolerance: the two ranks' adapters are BITWISE equal (they apply
one reduced gradient with a deterministic optimizer); they equal the one-rank adapters
within ``atol=1e-6, rtol=1e-5`` (the summation order differs). SGD and not Adam, because
Adam divides by its own running gradient scale: a gradient that is 2x too large (SUM
instead of mean) would yield nearly the same update and the test could not see it.

How HF scales the loss in transformers 5.17 (``Trainer._get_num_items_in_batch`` and
``compute_loss``): with ``average_tokens_across_devices`` (default True) and a model whose
forward takes loss kwargs, ``num_items_in_batch`` is the label-token count of the whole
accumulation window GATHERED OVER ALL RANKS, each micro-batch loss is the token SUM over
that global count, and it is multiplied by the number of processes. The mean over ranks of
the per-rank gradients is therefore the gradient of the global per-token mean, which is
what a one-rank run on the union batch computes, however the tokens are spread. That is
why a plain mean is the right sync and why T1b (ranks holding different token counts)
fails on a SUM, and on any averaging of per-rank means.

The two-process tests need torch, transformers, peft, trl, datasets, accelerate,
safetensors and gloo and skip without them (and on macOS, where two-process rendezvous can
time out on loaded runners). Each process has its own shard cache and
``CUDA_VISIBLE_DEVICES=-1``; the worker is ``_stream_distributed_trainer_worker.py``.
CLI output is compared through an ANSI-strip + whitespace-collapse helper.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.trainer.stream_setup import (
    DISTRIBUTED_STREAM_ISSUE,
    distributed_stream_blockers,
    refuse_if_distributed_launch,
)
from soup_cli.utils.adapter_sync_trainer import (
    ACCELERATE_PREPARED_FLAG,
    assert_adapter_sync_installed,
    has_adapter_sync,
    is_accelerate_prepared,
    make_adapter_sync_trainer_class,
    mark_accelerate_prepared,
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

_WORKER = Path(__file__).resolve().with_name("_stream_distributed_trainer_worker.py")
_TIMEOUT_SECONDS = 420
_STEPS = 5
_ROWS = 20
_ATOL = 1e-6
_RTOL = 1e-5

#: The refusal a user reads. Pinned whole, spelled out here and not imported, so a change
#: of the wording in src/ fails by name.
_MESSAGE = (
    "training.stream_layers=true under a distributed launch (WORLD_SIZE={size}): "
    "distributed layer streaming is not supported yet with {items}, see #2585. "
    "It is BETA for task='sft' on one rank per host, without --deepspeed, --fsdp, --resume, "
    "--hf-resume, --push-as, an eval gate (--gate / training.eval_gate), training.multipack, "
    "training.moe_lora or data.format='raft'. "
    "Run on one process (no torchrun/accelerate launch), set stream_layers: false, "
    "or remove the blocking setting."
)

#: Every spelling of "not a distributed launch" that must keep today's behaviour.
_NOT_DISTRIBUTED = ["1", "0", "-1", "", " ", "abc", "2.5", "two"]

_LOCAL_SIZE_INVALID = "LOCAL_WORLD_SIZE unset or invalid (one rank per host cannot be confirmed)"


def _plain(text: str | None) -> str:
    """Rich colours per character and wraps, so compare stripped + collapsed."""
    return " ".join(_ANSI.sub("", text or "").replace("│", " ").split())


@pytest.fixture
def one_rank_per_host(monkeypatch):
    """Rank 0 of a two-host launch: the one distributed shape that may stream."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    for name in ("ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_FSDP"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------
# C3: the trainer class
# --------------------------------------------------------------------------


class _FakeTrainer:
    """Stands in for HF's ``Trainer``: one backward per ``training_step``, and HF's own
    "last micro-batch of the window" flag on ``accelerator``."""

    def __init__(self):
        self.accelerator = SimpleNamespace(sync_gradients=False)
        self.events = []
        self.seen = []

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.events.append("backward")
        self.seen.append((model, inputs, num_items_in_batch))
        return "the-loss"


@pytest.fixture
def sync_events(monkeypatch):
    """Make ``AdapterGradSync.sync`` record itself into the trainer's event list."""
    pytest.importorskip("torch")
    from soup_cli.utils.adapter_grad_sync import AdapterGradSync, SyncStats

    state = {"trainer": None}

    def sync(self, group=None):
        state["trainer"].events.append("sync")
        return SyncStats(
            bytes=4 * self.n_elements, seconds=0.25, n_params=self.n_params, n_chunks=1
        )

    monkeypatch.setattr(AdapterGradSync, "sync", sync)
    return state


def _trainer_and_model(sync_events):
    import torch

    trainer = make_adapter_sync_trainer_class(_FakeTrainer)()
    sync_events["trainer"] = trainer
    return trainer, torch.nn.Linear(2, 2)


class TestAdapterSyncTrainerClass:
    def test_the_last_micro_batch_syncs_after_its_backward_and_returns_the_loss(self, sync_events):
        trainer, model = _trainer_and_model(sync_events)
        trainer.accelerator.sync_gradients = True
        assert trainer.training_step(model, {"x": 1}) == "the-loss"
        assert trainer.events == ["backward", "sync"], "the sync must follow the backward"

    def test_a_micro_batch_inside_the_window_does_not_sync(self, sync_events):
        trainer, model = _trainer_and_model(sync_events)
        trainer.accelerator.sync_gradients = False
        trainer.training_step(model, {"x": 1})
        assert trainer.events == ["backward"]
        assert trainer.sync_calls == 0
        assert trainer.last_sync_stats is None

    def test_one_sync_per_accumulation_window(self, sync_events):
        """Window of three micro-batches, two windows: two syncs, each after a window's
        last backward and never between two backwards of one window."""
        trainer, model = _trainer_and_model(sync_events)
        for last in (False, False, True, False, False, True):
            trainer.accelerator.sync_gradients = last
            trainer.training_step(model, {})
        assert trainer.events == (["backward"] * 3 + ["sync"]) * 2
        assert trainer.events.count("sync") == trainer.sync_calls == 2

    def test_the_sync_is_built_once_and_lazily_from_the_model(self, sync_events):
        from soup_cli.utils.adapter_grad_sync import AdapterGradSync

        trainer, model = _trainer_and_model(sync_events)
        assert trainer._adapter_grad_sync is None, "built before any step"
        trainer.accelerator.sync_gradients = False
        trainer.training_step(model, {})
        assert trainer._adapter_grad_sync is None, "built on a step that does not sync"
        trainer.accelerator.sync_gradients = True
        trainer.training_step(model, {})
        first = trainer._adapter_grad_sync
        assert isinstance(first, AdapterGradSync)
        assert first.n_elements == 6 and first.n_params == 2
        trainer.training_step(model, {})
        assert trainer._adapter_grad_sync is first

    def test_the_last_stats_stay_on_the_trainer(self, sync_events):
        trainer, model = _trainer_and_model(sync_events)
        trainer.accelerator.sync_gradients = True
        trainer.training_step(model, {})
        stats = trainer.last_sync_stats
        assert (stats.bytes, stats.seconds, stats.n_params, stats.n_chunks) == (24, 0.25, 2, 1)
        assert trainer.sync_calls == 1

    def test_arguments_reach_the_base_step_and_the_model_may_be_a_keyword(self, sync_events):
        trainer, model = _trainer_and_model(sync_events)
        trainer.accelerator.sync_gradients = True
        trainer.training_step(model=model, inputs={"a": 1}, num_items_in_batch=7)
        assert trainer.seen == [(model, {"a": 1}, 7)]
        assert trainer.events == ["backward", "sync"]

    def test_a_model_wrapped_in_ddp_is_refused_by_name(self, sync_events):
        import torch

        trainer, _model = _trainer_and_model(sync_events)
        wrapped = MagicMock(spec=torch.nn.parallel.DistributedDataParallel)
        trainer.accelerator.sync_gradients = True
        with pytest.raises(RuntimeError, match=r"wrapped the streamed model in Distributed"):
            trainer.training_step(wrapped, {})
        assert trainer.events == ["backward"], "synced a model that DDP also averages"

    def test_the_class_is_named_after_its_base_and_marked(self):
        cls = make_adapter_sync_trainer_class(_FakeTrainer)
        assert cls.__name__ == "AdapterSync_FakeTrainer"
        assert issubclass(cls, _FakeTrainer)
        assert has_adapter_sync(cls()) and not has_adapter_sync(_FakeTrainer())

    def test_a_trainer_without_the_sync_is_refused_before_it_trains(self):
        assert_adapter_sync_installed(make_adapter_sync_trainer_class(_FakeTrainer)())
        with pytest.raises(RuntimeError, match=r"needs the adapter gradient sync.*_FakeTrainer"):
            assert_adapter_sync_installed(_FakeTrainer())

    def test_hf_runs_training_step_before_clipping_and_the_optimizer(self):
        """The override point precedes ``_clip_grad_norm``: a callback does not (it fires
        after clipping), which is why the sync is not one. Pins transformers' loop order."""
        transformers = pytest.importorskip("transformers")
        source = "".join(
            inspect.getsource(getattr(transformers.Trainer, name))
            for name in ("_inner_training_loop", "_run_epoch")
            if hasattr(transformers.Trainer, name)
        )
        order = [
            "self.training_step(",
            "self._clip_grad_norm(",
            "on_pre_optimizer_step(",
            "self.optimizer.step()",
        ]
        assert all(source.count(token) == 1 for token in order), "the loop changed shape"
        positions = [source.index(token) for token in order]
        assert positions == sorted(positions), dict(zip(order, positions))
        assert "sync_gradients" in source, "HF no longer sets the flag the override reads"


# --------------------------------------------------------------------------
# C4: the accelerate bypass
# --------------------------------------------------------------------------


class TestAccelerateBypass:
    def test_the_helper_sets_the_flag_accelerate_reads(self):
        pytest.importorskip("torch")
        import torch

        model = torch.nn.Linear(2, 2)
        assert not is_accelerate_prepared(model)
        mark_accelerate_prepared(model)
        assert getattr(model, ACCELERATE_PREPARED_FLAG) is True
        assert ACCELERATE_PREPARED_FLAG == "_is_accelerate_prepared"
        assert is_accelerate_prepared(model)

    def test_the_setup_helper_flags_the_streamed_model_and_says_so_once(self, capsys):
        """C4 as the SFT setup calls it right after ``build_streamed_model``: the flag, and
        one console line. (The call itself is checked on the real setup by the two-process
        tests below, through the facts the worker reads off the trained model.)"""
        from soup_cli.trainer.stream_setup import StreamingSetupMixin

        model = SimpleNamespace()
        assert not is_accelerate_prepared(model)
        StreamingSetupMixin._bypass_accelerate_wrap(model)
        assert is_accelerate_prepared(model)
        output = _plain(capsys.readouterr().out)
        assert output.count("DistributedDataParallel wrap is bypassed") == 1, output
        assert "averaged across ranks by Soup" in output

    def test_a_marked_model_comes_back_from_prepare_model_unwrapped(self, monkeypatch):
        """On a GPU launch ``multi_device`` is true and an unmarked model is DDP-wrapped
        (which torch refuses for a streamed model); the marked one is returned as is. A
        CPU run cannot show it (``MULTI_CPU`` is never wrapped), so the capability is
        checked here on the production helper, with ``multi_device`` forced true."""
        pytest.importorskip("torch")
        pytest.importorskip("accelerate")
        import torch
        from accelerate import Accelerator
        from accelerate.state import AcceleratorState, GradientState

        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR"):
            monkeypatch.delenv(name, raising=False)
        AcceleratorState._reset_state(True)
        GradientState._reset_state()
        try:
            accelerator = Accelerator(cpu=True)
            monkeypatch.setattr(Accelerator, "multi_device", property(lambda self: True))
            assert accelerator.multi_device is True
            model = torch.nn.Linear(2, 2)
            mark_accelerate_prepared(model)
            returned = accelerator.prepare_model(model)
            assert returned is model
            assert not isinstance(returned, torch.nn.parallel.DistributedDataParallel)
            with pytest.raises(ValueError, match=r"Default process group has not been init"):
                accelerator.prepare_model(torch.nn.Linear(2, 2))
        finally:
            AcceleratorState._reset_state(True)
            GradientState._reset_state()


# --------------------------------------------------------------------------
# C1 / T4: the decision, in process
# --------------------------------------------------------------------------


def _sft_cfg(tmp_path, task: str = "sft", *, training=None, data=None):
    """A valid streaming config; ``training`` / ``data`` are applied WITHOUT validation, to
    reach combinations the schema refuses (multipack, ...) that the decision still names."""
    from soup_cli.config.loader import load_config_from_string

    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": str(tmp_path / "never-downloaded-base"),
                "task": task,
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": {
                    "batch_size": 2 if task == "kto" else 1,
                    "gradient_accumulation_steps": 1,
                    "quantization": "none",
                    "epochs": 1,
                    "stream_layers": True,
                    "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    updates = {}
    if training:
        updates["training"] = cfg.training.model_copy(update=training)
    if data:
        updates["data"] = cfg.data.model_copy(update=data)
    return cfg.model_copy(update=updates) if updates else cfg


def _eval_gate():
    from soup_cli.config.schema import EvalGateConfig

    return EvalGateConfig(enabled=True, suite="evals/gate.yaml")


class TestDecision:
    def test_the_allowed_slice_streams_and_says_so(self, tmp_path, one_rank_per_host):
        cfg = _sft_cfg(tmp_path)
        assert distributed_stream_blockers(cfg) == []
        assert refuse_if_distributed_launch(cfg) is True

    def test_the_rewind_recorder_is_not_a_blocker(self, tmp_path, one_rank_per_host):
        """``rewind_log`` defaults to on, and the SFT setup already switches it off under a
        distributed launch with one console line; refusing it would refuse every config."""
        cfg = _sft_cfg(tmp_path)
        assert cfg.training.rewind_log is True
        assert refuse_if_distributed_launch(cfg) is True

    @pytest.mark.parametrize("value", [None, *_NOT_DISTRIBUTED])
    def test_a_process_that_is_not_distributed_is_never_refused(self, tmp_path, monkeypatch, value):
        """Today's behaviour, even for a config that a distributed launch would refuse."""
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
        if value is None:
            monkeypatch.delenv("WORLD_SIZE", raising=False)
        else:
            monkeypatch.setenv("WORLD_SIZE", value)
        cfg = _sft_cfg(tmp_path, "dpo")
        assert refuse_if_distributed_launch(cfg, resume="auto", push_as="u/r") is False

    @pytest.mark.parametrize("task", ["dpo", "kto", "orpo", "simpo"])
    def test_every_other_streaming_task_is_refused(self, tmp_path, one_rank_per_host, task):
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path, task))
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items=f"task='{task}'")

    @pytest.mark.parametrize("task", ["pretrain", "grpo", "ppo"])
    def test_a_task_outside_streaming_is_named_too(self, tmp_path, one_rank_per_host, task):
        cfg = _sft_cfg(tmp_path).model_copy(update={"task": task})
        assert distributed_stream_blockers(cfg) == [f"task='{task}'"]

    @pytest.mark.parametrize("size", [2, 3, 8])
    def test_more_than_one_rank_per_host_is_refused(self, tmp_path, monkeypatch, size):
        monkeypatch.setenv("WORLD_SIZE", str(size))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(size))
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path))
        assert _plain(str(excinfo.value)) == _MESSAGE.format(
            size=size, items=f"LOCAL_WORLD_SIZE={size} (more than one rank per host)"
        )

    @pytest.mark.parametrize("value", [None, "", " ", "abc", "0", "-1"])
    def test_an_unconfirmed_host_layout_is_refused(self, tmp_path, monkeypatch, value):
        """LOCAL_WORLD_SIZE is what torchrun and accelerate launch always set; without a
        1 there is no proof that the ranks sit on different hosts."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        if value is None:
            monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
        else:
            monkeypatch.setenv("LOCAL_WORLD_SIZE", value)
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path))
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items=_LOCAL_SIZE_INVALID)

    @pytest.mark.parametrize(
        ("kwargs", "item"),
        [
            ({"deepspeed": "zero2"}, "--deepspeed"),
            ({"deepspeed": "ds_config.json"}, "--deepspeed"),
            ({"fsdp": "full_shard"}, "--fsdp"),
            ({"resume": "auto"}, "--resume"),
            ({"resume": "output/checkpoint-10"}, "--resume"),
            ({"hf_resume": True}, "--hf-resume"),
            ({"push_as": "user/model"}, "--push-as"),
            ({"gate": "evals/gate.yaml"}, "--gate"),
        ],
        ids=[
            "deepspeed-preset",
            "deepspeed-file",
            "fsdp",
            "resume-auto",
            "resume-path",
            "hf-resume",
            "push-as",
            "gate",
        ],
    )
    def test_each_command_line_option_that_is_not_supported_is_refused_by_name(
        self, tmp_path, one_rank_per_host, kwargs, item
    ):
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path), **kwargs)
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items=item)

    @pytest.mark.parametrize(
        ("training", "item"),
        [
            ({"multipack": True}, "training.multipack"),
            ({"moe_lora": True}, "training.moe_lora"),
            ({"eval_gate": _eval_gate()}, "training.eval_gate"),
        ],
        ids=["multipack", "moe_lora", "eval_gate"],
    )
    def test_each_config_setting_that_is_not_supported_is_refused_by_name(
        self, tmp_path, one_rank_per_host, training, item
    ):
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path, training=training))
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items=item)

    def test_raft_is_refused_because_its_trainer_has_no_sync(self, tmp_path, one_rank_per_host):
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path, data={"format": "raft"}))
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items="data.format='raft'")

    @pytest.mark.parametrize("name", ["ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_FSDP"])
    def test_a_launcher_that_enables_deepspeed_or_fsdp_is_refused_without_the_flag(
        self, tmp_path, one_rank_per_host, monkeypatch, name
    ):
        """A caller that never passes ``--deepspeed`` (sweep, the API) is still caught by
        what ``accelerate launch --use_deepspeed/--use_fsdp`` put in the environment."""
        monkeypatch.setenv(name, "true")
        item = "--deepspeed" if name.endswith("DEEPSPEED") else "--fsdp"
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path))
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items=item)
        monkeypatch.setenv(name, "false")
        assert refuse_if_distributed_launch(_sft_cfg(tmp_path)) is True

    def test_everything_that_blocks_is_named_in_one_message(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
        cfg = _sft_cfg(tmp_path, "dpo", training={"multipack": True, "moe_lora": True})
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(
                cfg, resume="auto", push_as="u/r", deepspeed="zero2", fsdp="full_shard"
            )
        items = (
            "task='dpo', LOCAL_WORLD_SIZE=2 (more than one rank per host), --deepspeed, "
            "--fsdp, --resume, --push-as, training.multipack, training.moe_lora"
        )
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=4, items=items)

    def test_the_message_keeps_the_phrases_and_the_tracking_issue(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path), resume="auto")
        message = _plain(str(excinfo.value))
        assert DISTRIBUTED_STREAM_ISSUE == 2585
        for phrase in (
            "distributed layer streaming is not supported yet",
            f"#{DISTRIBUTED_STREAM_ISSUE}",
            "WORLD_SIZE=2",
            "training.stream_layers=true",
            "stream_layers: false",
            "no torchrun/accelerate launch",
            "BETA",
        ):
            assert phrase in message, phrase


# --------------------------------------------------------------------------
# the SFT setup decides before any model work
# --------------------------------------------------------------------------

_WRAPPERS = {
    "sft": ("soup_cli.trainer.sft", "SFTTrainerWrapper"),
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper"),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper"),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper"),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper"),
}
_DATASET = {"train": [{"messages": [{"role": "user", "content": "hi"}]}]}


class _ReachedModelWorkError(AssertionError):
    """A sentinel on model download / sharding: reaching it means the guard let the run go on."""


def _wrapper(tmp_path, task, **kwargs):
    import importlib

    module_name, class_name = _WRAPPERS[task]
    wrapper_cls = getattr(importlib.import_module(module_name), class_name)
    return wrapper_cls(_sft_cfg(tmp_path, task), device="cpu", **kwargs)


def _trap_model_work(monkeypatch):
    pytest.importorskip("transformers")
    import transformers

    def _boom(*_args, **_kwargs):
        raise _ReachedModelWorkError("the setup reached model work")

    monkeypatch.setattr("soup_cli.utils.kimi_k2.load_stream_model_config", _boom)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _boom)
    monkeypatch.setattr("soup_cli.utils.spectrum_scan.resolve_model_weights", _boom)
    monkeypatch.setattr("soup_cli.utils.layer_shard.shard_checkpoint", _boom)


class TestSetupDecision:
    @pytest.mark.parametrize("task", ["dpo", "kto", "orpo", "simpo"])
    def test_a_preference_task_is_refused_before_any_model_work(
        self, tmp_path, monkeypatch, one_rank_per_host, task
    ):
        cache = tmp_path / "cache"
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(cache))
        _trap_model_work(monkeypatch)
        with pytest.raises(ValueError) as excinfo:
            _wrapper(tmp_path, task).setup(_DATASET)
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items=f"task='{task}'")
        assert not cache.exists()
        assert not (tmp_path / "out").exists()

    @pytest.mark.parametrize(
        ("kwargs", "item"),
        [
            ({"deepspeed_config": "ds_config.json"}, "--deepspeed"),
            ({"fsdp_config": {"fsdp": "full_shard"}}, "--fsdp"),
        ],
        ids=["deepspeed", "fsdp"],
    )
    def test_a_sharding_launch_is_refused_before_any_model_work(
        self, tmp_path, monkeypatch, one_rank_per_host, kwargs, item
    ):
        _trap_model_work(monkeypatch)
        with pytest.raises(ValueError) as excinfo:
            _wrapper(tmp_path, "sft", **kwargs).setup(_DATASET)
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items=item)

    def test_the_allowed_slice_goes_on_to_the_model_and_says_it_is_beta_once(
        self, tmp_path, monkeypatch, one_rank_per_host, capsys
    ):
        _trap_model_work(monkeypatch)
        wrapper = _wrapper(tmp_path, "sft")
        with pytest.raises(_ReachedModelWorkError):
            wrapper.setup(_DATASET)
        assert wrapper._distributed_stream is True
        output = _plain(capsys.readouterr().out)
        assert output.count("Distributed layer streaming is BETA") == 1, output
        assert "data-parallel over 2 ranks" in output
        assert f"#{DISTRIBUTED_STREAM_ISSUE}" in output

    def test_a_single_process_says_nothing_about_beta(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        _trap_model_work(monkeypatch)
        wrapper = _wrapper(tmp_path, "sft")
        with pytest.raises(_ReachedModelWorkError):
            wrapper.setup(_DATASET)
        assert wrapper._distributed_stream is False
        assert "Distributed layer streaming" not in capsys.readouterr().out

    def test_a_resume_is_refused_at_train_for_callers_that_skip_the_command(
        self, tmp_path, one_rank_per_host
    ):
        """Sweep / web UI / API reach a resume only through ``train(resume_from_checkpoint=)``;
        refused before a callback is added or a step is taken."""
        wrapper = _wrapper(tmp_path, "sft")
        wrapper._distributed_stream = True
        wrapper.trainer = MagicMock()
        with pytest.raises(ValueError) as excinfo:
            wrapper.train(resume_from_checkpoint="out/checkpoint-5")
        assert _plain(str(excinfo.value)) == _MESSAGE.format(size=2, items="--resume")
        wrapper.trainer.train.assert_not_called()


# --------------------------------------------------------------------------
# soup train: the earliest point that knows the options
# --------------------------------------------------------------------------


class _PastTheGuardError(Exception):
    """Raised by a sentinel placed just AFTER the guard: reaching it proves no refusal."""


def _write_config(tmp_path) -> str:
    cfg = tmp_path / "soup.yaml"
    cfg.write_text(
        "base: smollm2-135m\n"
        "task: sft\n"
        "backend: transformers\n"
        "data:\n"
        "  train: /dev/null\n"
        "training:\n"
        "  stream_layers: true\n"
        "  batch_size: 1\n"
    )
    return str(cfg)


def _invoke_train(tmp_path, monkeypatch, *extra):
    from soup_cli.commands import train as train_command
    from soup_cli.experiment.tracker import ExperimentTracker

    def _past(*_args, **_kwargs):
        raise _PastTheGuardError

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr(train_command, "_resolve_resume_or_exit", _past)
    start_run = MagicMock()
    monkeypatch.setattr(ExperimentTracker, "start_run", start_run)
    result = CliRunner().invoke(
        app, ["train", "--config", _write_config(tmp_path), "--yes", *extra]
    )
    return result, start_run


class TestTrainCommandDecision:
    def test_the_allowed_slice_is_not_refused(self, tmp_path, monkeypatch, one_rank_per_host):
        result, _start = _invoke_train(tmp_path, monkeypatch)
        assert isinstance(result.exception, _PastTheGuardError), (
            result.output,
            repr(result.exception),
        )
        assert "distributed layer streaming" not in _plain(result.output)

    @pytest.mark.parametrize(
        ("args", "item"),
        [
            (["--resume", "auto"], "--resume"),
            (["--resume", "output/checkpoint-10"], "--resume"),
            (["--push-as", "user/model"], "--push-as"),
            (["--push-as", "user/model", "--hf-resume"], "--hf-resume, --push-as"),
            (["--deepspeed", "zero2"], "--deepspeed"),
            (["--fsdp", "full_shard"], "--fsdp"),
            (["--gate", "evals/gate.yaml"], "--gate"),
        ],
        ids=["resume-auto", "resume-path", "push-as", "hf-resume", "deepspeed", "fsdp", "gate"],
    )
    def test_each_option_is_refused_by_name_before_anything_else(
        self, tmp_path, monkeypatch, one_rank_per_host, args, item
    ):
        result, start_run = _invoke_train(tmp_path, monkeypatch, *args)
        output = _plain(result.output)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert not isinstance(result.exception, _PastTheGuardError), "the guard let it through"
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        assert _MESSAGE.format(size=2, items=item) in output, output
        assert "Traceback" not in result.output
        assert "Loaded:" not in output and "Run ID" not in output
        assert not start_run.called

    def test_a_single_process_may_use_every_one_of_them(self, tmp_path, monkeypatch):
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        result, _start = _invoke_train(tmp_path, monkeypatch, "--resume", "auto")
        assert isinstance(result.exception, _PastTheGuardError), (
            result.output,
            repr(result.exception),
        )
        assert "distributed layer streaming" not in _plain(result.output)


# --------------------------------------------------------------------------
# two real processes against one: T1, T1a, T1b, T1c
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return "<no output>"
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _letters(seed: int, count: int) -> str:
    return "".join(chr(97 + (seed * 7 + index * 5) % 26) for index in range(count))


def _row(index: int, answer_length: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"item {index:02d} " + _letters(index, 6)},
            {"role": "assistant", "content": _letters(index + 3, answer_length)},
        ]
    }


def _equal_rows() -> list:
    return [_row(index, 24) for index in range(_ROWS)]


def _ragged_rows(per_device_batch: int) -> list:
    """Rows of one rank are long, of the other short: the global batch ``j`` (``b`` rows) goes
    to rank ``j % 2``, so rank 0 holds the long rows and rank 1 the short ones."""
    return [
        _row(index, 40 if (index // per_device_batch) % 2 == 0 else 2) for index in range(_ROWS)
    ]


def _worker_env() -> dict:
    import soup_cli

    source_root = str(Path(soup_cli.__file__).resolve().parent.parent)
    env = os.environ.copy()
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "ACCELERATE_USE_CPU",
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_USE_FSDP",
        "FORCE_COLOR",
        "NO_COLOR",
    ):
        env.pop(name, None)  # a launcher variable left in the shell must not reach a rank
    env.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONPATH=os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH", "")])),
        PYTHONIOENCODING="utf-8",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TQDM_DISABLE="1",
    )
    return env


def _read_run(out_dir: Path, tag: str):
    from safetensors.torch import load_file

    return SimpleNamespace(
        tag=tag,
        facts=json.loads((out_dir / f"{tag}.json").read_text("utf-8")),
        tensors=load_file(str(out_dir / f"{tag}.safetensors")),
        log=_plain((out_dir / f"{tag}.log").read_text("utf-8", errors="replace")),
    )


def _run_group(base_dir, out_dir: Path, *, rows, per_device_batch, grad_accum, max_grad_norm):
    """Two ranks (batch ``b``) and one reference process (batch ``2b``), all at once."""
    distributed = pytest.importorskip("torch.distributed")
    if not distributed.is_gloo_available():
        pytest.skip("installed torch has no CPU Gloo backend")
    if sys.platform == "darwin":
        # Same reason as tests/test_stream_adapter_sync.py: two-process rendezvous can time
        # out on loaded macOS runners.
        pytest.skip("two-process Gloo runs are skipped on macOS (runner timeouts)")
    for module in ("transformers", "peft", "trl", "datasets", "accelerate", "safetensors"):
        pytest.importorskip(module)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_file = out_dir / "rows.json"
    rows_file.write_text(json.dumps(rows), encoding="utf-8")
    port = _free_port()
    plans = [
        ("dp0", 0, 2, per_device_batch),
        ("dp1", 1, 2, per_device_batch),
        ("ref", 0, 1, 2 * per_device_batch),
    ]
    handles, processes, logs = [], [], {}
    failure = None
    try:
        for tag, rank, world, batch in plans:
            env = _worker_env()
            if world > 1:
                env.update(
                    RANK=str(rank),
                    WORLD_SIZE=str(world),
                    LOCAL_RANK="0",
                    LOCAL_WORLD_SIZE="1",  # one rank per host: what the decision requires
                    MASTER_ADDR="127.0.0.1",
                    MASTER_PORT=str(port),
                    ACCELERATE_USE_CPU="true",  # accelerate: MULTI_CPU, gloo, env://
                )
            logs[tag] = out_dir / f"{tag}.log"
            handle = logs[tag].open("w", encoding="utf-8")
            handles.append(handle)
            command = [sys.executable, str(_WORKER)]
            for flag, value in (
                ("--rank", rank),
                ("--world-size", world),
                ("--tag", tag),
                ("--out-dir", out_dir),
                ("--base-dir", base_dir),
                ("--rows-file", rows_file),
                ("--batch-size", batch),
                ("--grad-accum", grad_accum),
                ("--max-grad-norm", max_grad_norm),
            ):
                command += [flag, str(value)]
            if world == 1:
                command.append("--no-rewind-log")
            processes.append(
                subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            )
        deadline = time.monotonic() + _TIMEOUT_SECONDS
        while failure is None and any(process.poll() is None for process in processes):
            for (tag, *_rest), process in zip(plans, processes):
                if process.poll() not in (None, 0):
                    failure = f"{tag} exited with {process.poll()}"
            if failure is None and time.monotonic() > deadline:
                failure = f"still running after {_TIMEOUT_SECONDS} s (a hang)"
            time.sleep(0.25)
        for (tag, *_rest), process in zip(plans, processes):
            if failure is None and process.returncode != 0:
                failure = f"{tag} exited with {process.returncode}"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
        for handle in handles:
            handle.close()
    if failure is not None:
        report = "\n".join(
            f"--- {tag} (last 40 lines) ---\n{_tail(logs[tag])}" for tag, *_rest in plans
        )
        pytest.fail(f"two-rank run failed: {failure}\n{report}", pytrace=False)
    dp0, dp1, ref = (_read_run(out_dir, tag) for tag, *_rest in plans)
    return SimpleNamespace(dp=(dp0, dp1), ref=ref, rows=rows)


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    """A tiny Llama with its tokenizer, built ONCE with a fixed seed and shared read-only
    (each process shards it into its own cache)."""
    for module in ("torch", "transformers", "tokenizers"):
        pytest.importorskip(module)
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from tests._tiny_hf_models import build_tokenizer

    target = tmp_path_factory.mktemp("base") / "model"
    torch.manual_seed(7)
    tokenizer = build_tokenizer()
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tokenizer),
        max_position_embeddings=256,
    )
    LlamaForCausalLM(config).to(torch.float32).save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return str(target)


@pytest.fixture(scope="module")
def group_plain(base_dir, tmp_path_factory):
    """T1: equal-length rows, per-device batch 2 (reference 4), no accumulation, no clipping."""
    return _run_group(
        base_dir,
        tmp_path_factory.mktemp("plain"),
        rows=_equal_rows(),
        per_device_batch=2,
        grad_accum=1,
        max_grad_norm=1000.0,
    )


@pytest.fixture(scope="module")
def group_clip(base_dir, group_plain, tmp_path_factory):
    """T1a: the same run, with ``max_grad_norm`` below every step's gradient norm."""
    norms = group_plain.ref.facts["grad_norms"]
    assert len(norms) == _STEPS and min(norms) > 0, norms
    return _run_group(
        base_dir,
        tmp_path_factory.mktemp("clip"),
        rows=_equal_rows(),
        per_device_batch=2,
        grad_accum=1,
        max_grad_norm=0.3 * min(norms),
    )


@pytest.fixture(scope="module")
def group_ragged(base_dir, tmp_path_factory):
    """T1b: rank 0 holds long rows, rank 1 short ones (unequal token counts per step)."""
    return _run_group(
        base_dir,
        tmp_path_factory.mktemp("ragged"),
        rows=_ragged_rows(2),
        per_device_batch=2,
        grad_accum=1,
        max_grad_norm=1000.0,
    )


@pytest.fixture(scope="module")
def group_accum(base_dir, tmp_path_factory):
    """T1c: accumulation 2 with ragged rows, per-device batch 1 (reference batch 2)."""
    return _run_group(
        base_dir,
        tmp_path_factory.mktemp("accum"),
        rows=_ragged_rows(1),
        per_device_batch=1,
        grad_accum=2,
        max_grad_norm=1000.0,
    )


def _lora_keys(run):
    return sorted(key for key in run.tensors if key != "per_step")


def _worst(left, right) -> float:
    return float((left.double() - right.double()).abs().max())


def _assert_took_the_distributed_path(group):
    """The facts that make the comparison mean what it says."""
    for run in group.dp:
        facts = run.facts
        assert facts["world_size"] == 2, facts
        assert facts["distributed_type"].endswith("MULTI_CPU"), facts
        assert facts["distributed_stream"] is True, facts
        assert facts["trainer_class"].startswith("AdapterSync"), facts
        assert facts["has_adapter_sync"] is True
        # read after setup, before train(): accelerate flags every model it prepares
        assert facts["accelerate_prepared"] is True, "C4: the streamed model was not flagged"
        assert facts["ddp_wrapped"] is False
        assert facts["average_tokens_across_devices"] is True
        assert facts["accepts_loss_kwargs"] is True, "HF no longer scales the loss globally"
        assert facts["global_step"] == _STEPS, facts
    assert [run.facts["process_index"] for run in group.dp] == [0, 1]
    ref = group.ref.facts
    assert ref["world_size"] == 1 and ref["distributed_stream"] is False, ref
    assert not ref["trainer_class"].startswith("AdapterSync"), ref
    assert ref["accelerate_prepared"] is False and ref["sync_calls"] is None, ref
    assert ref["global_step"] == _STEPS, ref


def _assert_not_vacuous(group):
    ref = group.ref
    assert ref.facts["initial_sha"] == group.dp[0].facts["initial_sha"], "unequal starts"
    assert group.dp[0].facts["initial_sha"] == group.dp[1].facts["initial_sha"]
    assert ref.facts["final_sha"] != ref.facts["initial_sha"], "the adapters never moved"
    per_step = ref.tensors["per_step"]
    assert per_step.shape[0] == _STEPS
    for step in range(1, _STEPS):
        assert not (per_step[step] == per_step[step - 1]).all(), f"step {step} changed nothing"
    assert min(ref.facts["grad_norms"]) > 0


def _assert_equivalent(group):
    """Both ranks equal each other bit for bit, and the one-rank run within tolerance."""
    import torch

    zero, one = group.dp
    ref = group.ref
    assert _lora_keys(zero) == _lora_keys(one) == _lora_keys(ref)
    assert len(_lora_keys(ref)) == 8  # 2 layers x (q_proj, v_proj) x (lora_A, lora_B)
    assert zero.facts["final_sha"] == one.facts["final_sha"], "the ranks ended different"
    for key in (*_lora_keys(ref), "per_step"):
        assert torch.equal(zero.tensors[key], one.tensors[key]), f"ranks differ at {key}"
        assert torch.allclose(zero.tensors[key], ref.tensors[key], atol=_ATOL, rtol=_RTOL), (
            f"{key}: 2 ranks vs 1 rank differ by {_worst(zero.tensors[key], ref.tensors[key])}"
        )
    assert zero.facts["losses"] == one.facts["losses"], "the logged loss is a cross-rank mean"
    losses = torch.tensor(zero.facts["losses"], dtype=torch.float64)
    reference = torch.tensor(ref.facts["losses"], dtype=torch.float64)
    assert losses.numel() == reference.numel() == _STEPS
    assert torch.allclose(losses, reference, atol=_ATOL, rtol=_RTOL), (losses, reference)
    assert min(zero.facts["grad_norms"]) > 0


def _assert_one_sync_per_optimizer_step(group):
    for run in group.dp:
        facts = run.facts
        assert facts["sync_calls"] == _STEPS, facts
        assert facts["sync_bytes"] == 4 * facts["n_trainable"] > 0, facts
        assert facts["sync_chunks"] == 1 and facts["sync_seconds"] > 0, facts


class TestEquivalence:
    """T1 (design note, section 6): 2 ranks with batch ``b`` == 1 rank with batch ``2b``."""

    def test_the_run_took_the_production_distributed_path(self, group_plain):
        _assert_took_the_distributed_path(group_plain)

    def test_the_comparison_is_not_vacuous(self, group_plain):
        _assert_not_vacuous(group_plain)

    def test_two_ranks_equal_one_rank_on_the_union_batch(self, group_plain):
        """Mutation target (T3a): without the sync the ranks drift apart and away from the
        one-rank run. Also T3b: a SUM without the divide doubles every update."""
        _assert_equivalent(group_plain)

    def test_one_sync_per_optimizer_step_of_exactly_the_adapter_bytes(self, group_plain):
        _assert_one_sync_per_optimizer_step(group_plain)

    def test_c4_and_beta_are_each_said_once_per_rank(self, group_plain):
        for run in group_plain.dp:
            assert run.log.count("Distributed layer streaming is BETA") == 1, run.log[-2000:]
            assert run.log.count("DistributedDataParallel wrap is bypassed") == 1, run.log[-2000:]
            assert "averaged across ranks by Soup" in run.log
            assert "Rewind log off: the recorder is single-process" in run.log
        assert "Distributed layer streaming" not in group_plain.ref.log
        assert "wrap is bypassed" not in group_plain.ref.log


class TestClipping:
    """T1a: clipping sees the AVERAGED gradient, because the sync precedes it."""

    def test_the_clipping_really_fires_on_every_step(self, group_plain, group_clip):
        limit = 0.3 * min(group_plain.ref.facts["grad_norms"])
        for run in (*group_clip.dp, group_clip.ref):
            norms = run.facts["grad_norms"]  # HF logs the norm BEFORE clipping
            assert len(norms) == _STEPS and all(norm > limit for norm in norms), (norms, limit)

    def test_two_ranks_equal_one_rank_when_every_step_is_clipped(self, group_clip):
        """Mutation target (T3c): sync after clipping gives mean(clip(g_rank)), not
        clip(mean(g_rank)); the clipped update is then too small, rank-dependent weighted."""
        _assert_took_the_distributed_path(group_clip)
        _assert_not_vacuous(group_clip)
        _assert_equivalent(group_clip)
        _assert_one_sync_per_optimizer_step(group_clip)

    def test_the_clipped_run_is_not_the_unclipped_one(self, group_plain, group_clip):
        import torch

        assert not torch.allclose(
            group_clip.ref.tensors["per_step"],
            group_plain.ref.tensors["per_step"],
            atol=_ATOL,
            rtol=_RTOL,
        ), "clipping changed nothing: the threshold is not exercising anything"


class TestRaggedRanks:
    """T1b: the ranks hold different token counts in the same step."""

    def test_the_ranks_really_hold_different_amounts_of_text(self, group_ragged):
        lengths = {0: [], 1: []}
        for index, row in enumerate(group_ragged.rows):
            lengths[(index // 2) % 2].append(len(row["messages"][1]["content"]))
        assert set(lengths[0]) == {40} and set(lengths[1]) == {2}

    def test_two_ranks_equal_one_rank_however_the_tokens_are_spread(self, group_ragged):
        """Mutation target (T3b): SUM without the divide, or a mean of per-rank means, fails."""
        _assert_took_the_distributed_path(group_ragged)
        _assert_not_vacuous(group_ragged)
        _assert_equivalent(group_ragged)
        _assert_one_sync_per_optimizer_step(group_ragged)


class TestAccumulation:
    """T1c: the sync fires once per OPTIMIZER step, on the last micro-batch of a window."""

    @pytest.mark.parametrize("window", [1, 2])
    def test_equal_to_the_one_rank_run_and_one_sync_per_step(self, request, window):
        group = request.getfixturevalue("group_plain" if window == 1 else "group_accum")
        _assert_took_the_distributed_path(group)
        _assert_not_vacuous(group)
        _assert_equivalent(group)
        _assert_one_sync_per_optimizer_step(group)

    def test_a_window_of_two_means_half_as_many_syncs_as_micro_batches(self, group_accum):
        """Mutation evidence for 'sync on every micro-batch': 10 micro-batches per rank, 5
        optimizer steps, so 5 syncs; a sync per micro-batch would count 10."""
        for run in group_accum.dp:
            assert run.facts["sync_calls"] == _STEPS == _ROWS // 2 // 2
