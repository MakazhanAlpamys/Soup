"""Opt-in, context-local progress for ship. No model or task content is serialized."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager, redirect_stderr
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterable, Iterator, TextIO, TypeVar, get_type_hints

_writer: ContextVar[TextIO | None] = ContextVar("ship_progress_writer", default=None)
_current: ContextVar[_Group | None] = ContextVar("ship_progress_group", default=None)
T = TypeVar("T")


@dataclass
class _Group:
    phase: str
    side: str | None
    suite: str | None
    repeat: int | None
    done: int | None
    total: int | None = None

    def emit(self, event: str) -> None:
        writer = _writer.get()
        if writer is None:
            return
        payload = {"schema_version": 1, "event": event, "phase": self.phase,
                   "side": self.side, "suite": self.suite, "repeat": self.repeat,
                   "done": self.done, "total": self.total}
        if event == "error":
            payload["error"] = "execution_aborted"
        writer.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
        writer.flush()


@contextmanager
def session(stream: TextIO) -> Iterator[None]:
    """Bind one execution's writer, including nested SDK/test use."""
    token = _writer.set(stream)
    try:
        yield
    finally:
        _writer.reset(token)


@contextmanager
def group(
    phase: str, *, side: str | None = None, suite: str | None = None,
    repeat: int | None = None, countable: bool = False,
) -> Iterator[None]:
    """Bracket actual work; a skipped call creates no group."""
    if _writer.get() is None:
        yield
        return
    state = _Group(phase, side, suite, repeat, 0 if countable else None)
    token = _current.set(state)
    state.emit("start")
    try:
        yield
    except BaseException:
        state.emit("error")
        raise
    else:
        state.emit("end")
    finally:
        _current.reset(token)


def track(items: Iterable[T], total: int) -> Iterator[T]:
    """Report only after the consumer scored an item, including failed items.

    An exception that escapes the consumer aborts its group; it is not reported
    as completed. In particular unavailable judges retain their runtime error.
    """
    state = _current.get() if _writer.get() is not None else None
    if state is not None and state.done is not None:
        state.total = total
    else:
        state = None
    for item in items:
        yield item
        if state is not None:
            state.done += 1
            state.emit("progress")


@contextmanager
def _exclusive_stderr() -> Iterator[TextIO]:
    """Keep native fd-2 writes, subprocesses, Python warnings and logging off JSONL.

    Duplicate the original channel before redirecting fd 2; event writes use
    the duplicate. For in-process runners with a StringIO stderr, retain that
    stream while suppressing the process's actual fd 2 as well.
    """
    original = sys.stderr
    saved_fd = os.dup(2)
    event_stream = None
    try:
        try:
            native = original.fileno() == 2
        except (AttributeError, OSError, ValueError):
            native = False
        if native:
            event_stream = os.fdopen(os.dup(saved_fd), "w", encoding="utf-8", buffering=1)
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), 2)
            with redirect_stderr(sink):
                yield event_stream if event_stream is not None else original
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        if event_stream is not None:
            event_stream.close()


def jsonl_command(function: Callable[..., Any]) -> Callable[..., Any]:
    """Enter before command validation; preserve Typer's public signature."""
    import typer

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("progress") != "jsonl":
            return function(*args, **kwargs)
        with _exclusive_stderr() as stream, session(stream):
            # Verdict encompasses validation, offline evidence and live groups.
            # Exit 2 is a completed DON'T SHIP verdict, not a runtime failure.
            try:
                completed_exit = None
                with group("verdict"):
                    try:
                        result = function(*args, **kwargs)
                    except typer.Exit as exc:
                        if exc.exit_code in (0, 2):
                            completed_exit = exc.exit_code
                        else:
                            raise
            except typer.Exit:
                raise
            except (Exception, KeyboardInterrupt):
                raise typer.Exit(code=1) from None
        if completed_exit is not None:
            raise typer.Exit(code=completed_exit)
        return result

    # Typer resolves hints using callback.__globals__, which belongs here.
    wrapped.__annotations__ = get_type_hints(function)
    return wrapped
