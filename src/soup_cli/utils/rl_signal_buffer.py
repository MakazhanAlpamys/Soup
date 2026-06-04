"""Shared RL signal buffer — v0.71.11 (#235 / #240).

Reward-fn capture mechanism that lets the live reward-hacking detector
(#235) and echo-trap detector (#240) callbacks observe the real
per-completion rewards + the generated completions of a GRPO step
without monkey-patching TRL internals.

How it works: when ``reward_hack_detector`` or ``echo_trap_enabled`` is
set, the GRPO wrapper wraps every reward function with a small capturing
shim (:func:`wrap_reward_funcs`) that records the completions + returned
rewards into a single shared :class:`RLSignalBuffer` before forwarding
the verbatim result. The callbacks then read a snapshot in
``on_step_end``.

Design notes:
- The reward-function signature ``reward_func(prompts, completions,
  **kwargs) -> list[float]`` is stable across TRL versions, so wrapping
  it is far more robust than hooking ``_generate_and_score_completions``.
- The capture shim is exception-safe — a buffer error MUST NEVER break
  the reward computation (training would crash).
- ``__name__`` is preserved so TRL's per-function logging keys stay
  correct (TRL logs ``rewards/<func_name>``).
- No torch import at module top (pure Python; the buffer stores plain
  floats + strings).

Security:
- Bounded buffers (``_MAX_COMPLETIONS`` / ``_MAX_COMPLETION_CHARS``) so a
  pathological run can't blow up RAM via the captured completions.
- Reward values coerced to floats; non-finite / non-numeric dropped.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Optional, Sequence

_MAX_COMPLETIONS = 1024
_MAX_COMPLETION_CHARS = 100_000


def _completion_to_text(completion: Any) -> str:
    """Best-effort extraction of the assistant text from a completion.

    TRL completions come in two shapes:
    - plain string (non-conversational reward), OR
    - list of message dicts ``[{"role": "assistant", "content": "..."}]``
      (conversational). We concatenate the ``content`` of every dict.
    """
    if isinstance(completion, str):
        text = completion
    elif isinstance(completion, dict):
        text = str(completion.get("content", ""))
    elif isinstance(completion, (list, tuple)):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        text = "".join(parts)
    else:
        text = str(completion)
    if len(text) > _MAX_COMPLETION_CHARS:
        text = text[:_MAX_COMPLETION_CHARS]
    return text


def _coerce_rewards(rewards: Any) -> list[Optional[float]]:
    """Coerce a reward list to floats; non-numeric / non-finite → None."""
    out: list[Optional[float]] = []
    try:
        iterator = list(rewards)
    except TypeError:
        return out
    for r in iterator:
        if isinstance(r, bool) or not isinstance(r, (int, float)):
            out.append(None)
            continue
        fv = float(r)
        out.append(fv if math.isfinite(fv) else None)
    return out


class RLSignalBuffer:
    """Thread-safe ring of the most-recently observed GRPO step signal.

    Stores the latest completions (as strings) and the per-reward-function
    reward lists. ``snapshot`` returns a consistent copy under the lock
    plus an element-wise aggregate reward (sum across functions).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._completions: list[str] = []
        self._per_func: dict[str, list[Optional[float]]] = {}

    def record(
        self,
        *,
        func_name: str,
        completions: Any,
        rewards: Any,
    ) -> None:
        """Record one reward-function call. Exception-safe by contract.

        Completions are normalised to strings and capped at
        ``_MAX_COMPLETIONS``. Within a single GRPO step every reward
        function sees the *same* completions, so overwriting on each call
        is correct.
        """
        texts: list[str] = []
        if completions is not None:
            try:
                seq = list(completions)
            except TypeError:
                seq = []
            for c in seq[:_MAX_COMPLETIONS]:
                texts.append(_completion_to_text(c))
        coerced = _coerce_rewards(rewards)
        name = func_name if isinstance(func_name, str) and func_name else "reward"
        with self._lock:
            if texts:
                self._completions = texts
            self._per_func[name] = coerced

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent copy of the latest step signal.

        Keys:
        - ``completions``: list[str] (latest observed).
        - ``per_func``: dict name -> list[float] (None-filtered per entry).
        - ``rewards``: element-wise sum across functions (aggregate reward
          GRPO uses for advantages). None values are treated as 0 in the
          sum but tracked so a fully-None column drops out.
        """
        with self._lock:
            completions = list(self._completions)
            per_func = {k: list(v) for k, v in self._per_func.items()}
        aggregate = _aggregate_rewards(per_func.values())
        return {
            "completions": completions,
            "per_func": per_func,
            "rewards": aggregate,
        }

    def clear(self) -> None:
        """Reset the buffer (used by tests / between runs)."""
        with self._lock:
            self._completions = []
            self._per_func = {}


def _aggregate_rewards(
    columns: "Sequence[list[Optional[float]]] | Any",
) -> list[float]:
    """Element-wise sum across per-function reward lists.

    Aligns on the shortest list length; ``None`` entries contribute 0.
    A position that is ``None`` in every function is dropped (returns the
    truncated prefix up to the first all-None column-position only when no
    finite value appears — simpler: we keep finite floats, treating None
    as 0).
    """
    col_lists = [c for c in columns if c]
    if not col_lists:
        return []
    n = min(len(c) for c in col_lists)
    out: list[float] = []
    for i in range(n):
        total = 0.0
        seen = False
        for c in col_lists:
            v = c[i]
            if v is not None:
                total += v
                seen = True
        if seen:
            out.append(total)
        else:
            out.append(0.0)
    return out


def _make_capturing(fn: Any, buffer: RLSignalBuffer) -> Any:
    """Wrap a single reward function with a capturing shim.

    Forwards ``*args, **kwargs`` verbatim, records into ``buffer``, and
    returns the inner result unchanged. The capture is wrapped in a broad
    ``except`` so a buffer bug can never break the reward computation.
    """
    name = getattr(fn, "__name__", "reward")

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        try:
            completions = kwargs.get("completions")
            if completions is None:
                # Positional: TRL calls reward_func(prompts, completions, ...)
                if len(args) >= 2:
                    completions = args[1]
                elif len(args) == 1:
                    completions = args[0]
            buffer.record(func_name=name, completions=completions, rewards=result)
        except Exception:  # noqa: BLE001 — capture MUST NOT break the reward
            pass
        return result

    _wrapped.__name__ = name
    return _wrapped


def wrap_reward_funcs(reward_funcs: Any, buffer: RLSignalBuffer) -> Any:
    """Wrap one reward function (or a list) for capture into ``buffer``.

    Preserves the single-callable-vs-list shape so the caller can hand the
    result straight back to ``GRPOTrainer(reward_funcs=...)``.
    """
    if isinstance(reward_funcs, (list, tuple)):
        return [_make_capturing(fn, buffer) for fn in reward_funcs]
    return _make_capturing(reward_funcs, buffer)
