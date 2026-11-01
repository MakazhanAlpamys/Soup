"""Manual checkpoint controls drive real local HF/TRL/PEFT training."""

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest


def _train_case(tmp_path, monkeypatch, mode):
    from transformers import TrainerCallback

    from tests.test_issue691_sft_packing import _sft_wrapper

    wrapper, _ = _sft_wrapper(tmp_path, monkeypatch, n_rows=4,
                              gradient_accumulation_steps=1,
                              save_steps=1 if mode == "scheduled" else 1000)
    output = Path(wrapper._output_dir)
    output.mkdir(exist_ok=True)
    trigger = output / ".checkpoint_now"
    if mode in ("trigger", "save-error"):
        trigger.write_text("")
    saved = []

    class Observe(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if mode in ("first-signal", "second-signal", "signal-during-save") \
                    and state.global_step == 1:
                signal.raise_signal(signal.SIGINT)
            if mode == "second-signal" and state.global_step == 2:
                signal.raise_signal(signal.SIGINT)
            if mode == "step-error" and state.global_step == 1:
                raise RuntimeError("deliberate step failure")

        def on_save(self, args, state, control, **kwargs):
            checkpoint = output / f"checkpoint-{state.global_step}"
            assert checkpoint.is_dir()
            assert (checkpoint / "trainer_state.json").is_file()
            assert (checkpoint / "adapter_model.safetensors").is_file()
            saved.append(state.global_step)

    observer = Observe()
    wrapper.trainer.add_callback(observer)
    # Deliver SIGINT before DefaultFlowCallback clears TrainerControl at step
    # begin: the request must survive until the next completed optimizer step.
    wrapper.trainer.callback_handler.callbacks.remove(observer)
    wrapper.trainer.callback_handler.callbacks.insert(0, observer)
    if mode == "save-error":
        def fail_save(*args, **kwargs):
            raise RuntimeError("deliberate checkpoint failure")
        monkeypatch.setattr(wrapper.trainer, "_save_checkpoint", fail_save)
    if mode == "signal-during-save":
        original_save = wrapper.trainer._save_checkpoint

        def signal_while_saving(*args, **kwargs):
            signal.raise_signal(signal.SIGINT)
            return original_save(*args, **kwargs)

        monkeypatch.setattr(wrapper.trainer, "_save_checkpoint", signal_while_saving)
    previous = signal.getsignal(signal.SIGINT)
    error = None
    try:
        wrapper.train()
    except (KeyboardInterrupt, RuntimeError) as exception:
        error = type(exception).__name__ + ": " + str(exception)
    restored = signal.getsignal(signal.SIGINT) == previous
    if not restored:
        signal.signal(signal.SIGINT, previous)
    return {"saved": saved, "step": wrapper.trainer.state.global_step,
            "trigger_exists": trigger.exists(), "restored": restored, "error": error}


@pytest.mark.parametrize("mode,expected", [("none", [4]), ("scheduled", [1, 2, 3, 4]),
                                          ("trigger", [1, 4])])
def test_trigger_adds_a_real_checkpoint_without_changing_scheduled_saves(
    tmp_path, monkeypatch, mode, expected,
):
    result = _train_case(tmp_path, monkeypatch, mode)
    assert result == {"saved": expected, "step": 4, "trigger_exists": False,
                      "restored": True, "error": None}


@pytest.mark.parametrize("mode,expected,step", [("first-signal", [2, 4], 4),
                                                ("second-signal", [2, 3], 3),
                                                ("signal-during-save", [2], 2)])
def test_signals_save_continue_then_save_and_stop_in_a_fresh_process(
    tmp_path, mode, expected, step,
):
    script = (
        "import json,sys; from pathlib import Path; import pytest; "
        "from tests.test_manual_training_saves import _train_case; "
        "patch=pytest.MonkeyPatch(); "
        "result=_train_case(Path(sys.argv[1]),patch,sys.argv[2]); patch.undo(); "
        "print('RESULT='+json.dumps(result))"
    )
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path), mode],
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert process.returncode == 0, (process.stdout[-2000:], process.stderr[-2000:])
    result = json.loads(next(line[7:] for line in process.stdout.splitlines()
                             if line.startswith("RESULT=")))
    assert result == {"saved": expected, "step": step, "trigger_exists": False,
                      "restored": True, "error": None}


@pytest.mark.parametrize("mode,message,step", [("step-error", "deliberate step failure", 1),
                                              ("save-error", "deliberate checkpoint failure", 1)])
def test_handler_is_restored_and_unsaved_trigger_is_kept_on_failure(
    tmp_path, monkeypatch, mode, message, step,
):
    result = _train_case(tmp_path, monkeypatch, mode)
    assert result["error"] == "RuntimeError: " + message
    assert result["step"] == step
    assert result["restored"] is True
    assert result["saved"] == []
    assert result["trigger_exists"] is (mode == "save-error")


@pytest.mark.requires_symlink
def test_trigger_symlink_is_neither_followed_nor_consumed(tmp_path, monkeypatch):
    from soup_cli.utils.checkpoint_trigger import consume_trigger, should_save_now

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    original = tmp_path / "keep.txt"
    original.write_bytes(b"original data")
    trigger = output / ".checkpoint_now"
    trigger.symlink_to(original)
    assert should_save_now(str(output)) is False
    assert consume_trigger(str(output)) is False
    assert trigger.is_symlink()
    assert original.read_bytes() == b"original data"


def test_handler_is_restored_if_startup_fails_after_install(tmp_path, monkeypatch):
    from soup_cli.utils.graceful_save import GracefulSaveHandler

    original = GracefulSaveHandler.install

    def fail_after_install(self):
        original(self)
        raise RuntimeError("startup after signal install")

    monkeypatch.setattr(GracefulSaveHandler, "install", fail_after_install)
    result = _train_case(tmp_path, monkeypatch, "none")
    assert result["error"] == "RuntimeError: startup after signal install"
    assert result["restored"] is True
    assert result["step"] == 0 and result["saved"] == []


def _gloo_worker(rank, rendezvous, directory):
    from datetime import timedelta

    import torch.distributed as distributed

    from soup_cli.monitoring.manual_save import _shared_requests

    distributed.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                                   rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        result = _shared_requests(rank == 0, rank == 1)
        (Path(directory) / f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        distributed.destroy_process_group()


def test_save_and_stop_requests_are_shared_by_two_cpu_ranks(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU distributed test needs Gloo")
    torch.multiprocessing.spawn(_gloo_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    assert [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)] == [
        [True, True], [True, True],
    ]


def test_custom_trainer_without_hf_callbacks_keeps_its_behavior():
    from soup_cli.monitoring.manual_save import manual_save_context

    calls = []

    class CustomTrainer:
        def train(self):
            calls.append("trained")
            return "result"

    trainer = CustomTrainer()
    with manual_save_context(trainer):
        assert trainer.train() == "result"
    assert calls == ["trained"]


def test_distributed_step_combines_save_and_stop_requests(monkeypatch):
    import torch
    import torch.distributed as distributed

    from soup_cli.monitoring.manual_save import _shared_requests

    monkeypatch.setattr(distributed, "is_available", lambda: True)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_backend", lambda: "gloo")

    def reduce(flags, op):
        assert flags.device.type == "cpu" and flags.dtype == torch.int32
        assert flags.tolist() == [1, 0]
        assert op == distributed.ReduceOp.MAX
        flags[1] = 1  # Another rank requested stop.

    monkeypatch.setattr(distributed, "all_reduce", reduce)
    assert _shared_requests(True, False) == (True, True)
