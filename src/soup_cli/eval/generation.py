"""Ordered optional batch generation shared by the pure evaluation scorers."""

from typing import Callable, Iterator, Optional


def iter_outputs(gen: Callable[[str], str], prompts: list[str]) -> Iterator[Optional[str]]:
    """Return one output per prompt; ``None`` marks a failed generation.

    A callable ``batch(prompts) -> list[str]`` is optional. A positive integer
    ``batch_size`` bounds each call, so consumers score its outputs before the
    next chunk starts. Without that hint the generator owns the batch boundary.
    If a chunk raises, scalar retries isolate failures to its items. A malformed
    result fails that chunk; non-string items fail individually.
    Empty strings remain valid outputs, distinct from generation failures.
    """
    if not callable(gen):
        raise TypeError("gen must be callable")
    if not prompts:
        return
    batch = getattr(gen, "batch", None)
    if callable(batch):
        size = getattr(gen, "batch_size", len(prompts))
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            size = len(prompts)
        for start in range(0, len(prompts), size):
            chunk = prompts[start:start + size]
            try:
                outputs = batch(list(chunk))
            except Exception:  # noqa: BLE001 — retry this chunk's items only
                yield from _scalar_outputs(gen, chunk)
            else:
                if not isinstance(outputs, list) or len(outputs) != len(chunk):
                    yield from [None] * len(chunk)
                else:
                    yield from (output if isinstance(output, str) else None for output in outputs)
        return
    yield from _scalar_outputs(gen, prompts)


def _scalar_outputs(gen: Callable[[str], str], prompts: list[str]) -> Iterator[Optional[str]]:
    for prompt in prompts:
        try:
            output = gen(prompt)
        except Exception:  # noqa: BLE001 — a generation error is a failed item
            output = None
        yield output if isinstance(output, str) else None
