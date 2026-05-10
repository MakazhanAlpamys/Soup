"""v0.44.0 Part B — Ctrl+C graceful-save SIGINT handler.

First SIGINT writes a checkpoint by setting `should_save=True` on the HF
Trainer state; second SIGINT exits via `should_training_stop=True` (or raises
KeyboardInterrupt if no trainer state is wired).
"""

from __future__ import annotations

import signal
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional


@dataclass
class GracefulSaveHandler:
    """SIGINT handler that survives a first Ctrl+C by requesting a save."""

    state: Optional[Any] = None  # HF TrainerState (duck-typed)
    sigint_count: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)
    _previous_handler: Any = field(default=None, repr=False, compare=False)
    _installed: bool = field(default=False, repr=False, compare=False)

    def attach_state(self, state: Any) -> None:
        """Plug an HF TrainerState in. Must be called before SIGINT fires."""
        with self._lock:
            self.state = state

    def install(self) -> None:
        """Install ourselves as SIGINT handler. Idempotent."""
        with self._lock:
            if self._installed:
                return
            try:
                self._previous_handler = signal.signal(
                    signal.SIGINT, self._handle_sigint
                )
                self._installed = True
            except (ValueError, OSError):
                # signal() raises ValueError when called from a non-main thread,
                # OSError on platforms where SIGINT is unavailable. Both are
                # acceptable degradations — the trainer just behaves like before.
                self._installed = False

    def restore(self) -> None:
        """Restore the prior SIGINT handler. Idempotent."""
        with self._lock:
            if not self._installed:
                return
            try:
                signal.signal(signal.SIGINT, self._previous_handler or signal.SIG_DFL)
            except (ValueError, OSError):
                pass
            self._installed = False

    def _handle_sigint(self, signum, frame) -> None:  # noqa: ARG002
        with self._lock:
            self.sigint_count += 1
            count = self.sigint_count
            state = self.state
        if count == 1 and state is not None:
            # First Ctrl+C: ask the trainer to save & continue.
            try:
                state.should_save = True
            except AttributeError:
                pass
            return
        # Second Ctrl+C (or first when no state attached): stop training.
        if state is not None:
            try:
                state.should_training_stop = True
                state.should_save = True
                return
            except AttributeError:
                pass
        # Last resort: behave like the default SIGINT.
        raise KeyboardInterrupt
