"""Ordered optional batch generation shared by the pure evaluation scorers."""

from typing import Callable, Iterator, Optional


def iter_outputs(gen: Callable[[str], str], prompts: list[str]) -> Iterator[Optional[str]]:
    """Return one output per prompt; ``None`` marks a failed generation.

    A callable ``batch(prompts) -> list[str]`` is optional. If a batch raises,
    scalar retries isolate failures to their items. A malformed result cannot
    be aligned safely and fails the batch; non-string items fail individually.
    Empty strings remain valid outputs, distinct from generation failures.
    """
    if not callable(gen):
        raise TypeError("gen must be callable")
    if not prompts:
        return
    batch = getattr(gen, "batch", None)
    if callable(batch):
        try:
            outputs = batch(prompts)
        except Exception:  # noqa: BLE001 — retry scalars to isolate failed items
            pass
        else:
            if not isinstance(outputs, list) or len(outputs) != len(prompts):
                yield from [None] * len(prompts)
                return
            yield from (output if isinstance(output, str) else None for output in outputs)
            return
    for prompt in prompts:
        try:
            output = gen(prompt)
        except Exception:  # noqa: BLE001 — a generation error is a failed item
            output = None
        yield output if isinstance(output, str) else None
