"""Manual checkpoints scoped to a HuggingFace Trainer training call."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

from soup_cli.utils.checkpoint_trigger import consume_trigger, should_save_now
from soup_cli.utils.graceful_save import GracefulSaveHandler


class _StopAfterCheckpoint(BaseException):
    """Leave a custom HF-backed loop only after its checkpoint succeeded."""


def _shared_requests(save: bool, stop: bool) -> tuple[bool, bool]:
    """All initialized distributed ranks must take the same safe-step decision."""
    import torch
    import torch.distributed as distributed

    if not distributed.is_available() or not distributed.is_initialized():
        return save, stop
    device = (torch.device("cuda", torch.cuda.current_device())
              if distributed.get_backend() == "nccl" else torch.device("cpu"))
    flags = torch.tensor([save, stop], dtype=torch.int32, device=device)
    distributed.all_reduce(flags, op=distributed.ReduceOp.MAX)
    return bool(flags[0].item()), bool(flags[1].item())


@contextmanager
def manual_save_context(trainer: Any, *, stop_after_save: bool = False) -> Iterator[None]:
    """Run a HF-backed trainer with checkpoint triggers and scoped SIGINT handling.

    A separate callback keeps these controls available without a live display.
    The finally block also covers startup, training and checkpoint exceptions:
    HF does not emit on_train_end when its loop raises.
    ``stop_after_save`` also leaves custom TRL loops that emit callbacks but
    ignore TrainerControl.should_training_stop (experimental PPO).
    """
    # Custom loops and lightweight trainer implementations without the HF
    # callback protocol keep their existing behavior and need no HF import.
    if not all(callable(getattr(trainer, name, None))
               for name in ("add_callback", "remove_callback")):
        yield
        return
    from transformers import TrainerCallback

    requests = SimpleNamespace(should_save=False, should_training_stop=False)
    handler = GracefulSaveHandler()
    handler.attach_state(requests)

    class ManualSaveCallback(TrainerCallback):
        trigger_pending = False

        def on_train_begin(self, args, state, control, **event_kwargs):
            handler.install()

        def on_step_end(self, args, state, control, **event_kwargs):
            # Signal requests live outside TrainerControl, whose flags the
            # default flow resets at step begin. Thus a signal during loading
            # or gradient accumulation remains latched until a safe step.
            self.trigger_pending = self.trigger_pending or should_save_now(args.output_dir)
            save, stop = _shared_requests(
                requests.should_save or self.trigger_pending,
                requests.should_training_stop,
            )
            if save or stop:
                control.should_save = True
            if stop:
                control.should_training_stop = True

        def on_save(self, args, state, control, **event_kwargs):
            if self.trigger_pending and state.is_world_process_zero:
                consume_trigger(args.output_dir)
            self.trigger_pending = False
            requests.should_save = False
            # A second signal can arrive while serializing this checkpoint.
            # It is already safe to stop after this successful save.
            _save, stop = _shared_requests(False, requests.should_training_stop)
            if stop:
                control.should_training_stop = True
                if stop_after_save:
                    raise _StopAfterCheckpoint

        def on_train_end(self, args, state, control, **event_kwargs):
            handler.restore()

    callback = ManualSaveCallback()
    trainer.add_callback(callback)
    try:
        yield
    except _StopAfterCheckpoint:
        trainer.control = trainer.callback_handler.on_train_end(
            trainer.args, trainer.state, trainer.control
        )
    finally:
        handler.restore()
        trainer.remove_callback(callback)
