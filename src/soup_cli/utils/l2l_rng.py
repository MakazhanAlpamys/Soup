"""Replay of the random stream gradient accumulation draws, for the L2L step.

``lora.dropout`` > 0 draws a mask from the global generator on decoder and head calls.
Plain accumulation runs each micro-batch through every decoder layer and then its head,
so each call starts where the preceding one left the generator. The L2L step instead
visits decoder calls layer by layer and recomputes them in backward. One shared stream
cannot give both orders the same masks, and saving states in layer-major order cannot
recover the accumulation schedule.

So the step learns the state of every pair before it runs anything:

1. ``record_draws`` notes, per decoder layer and head, which dropouts a call draws (shape,
   strides, dtype, device, probability) while the capture forward runs.
2. ``replay`` starts from the generator state at the step and runs those dropouts on
   zeros of the same metadata in accumulation order. A dropout consumes the generator as a
   function of that metadata only, so the state before each pair equals the one
   accumulation reaches. The state after the last pair is the one accumulation leaves.
3. ``state_at`` runs a layer call at the state of its pair, in the forward and in the
   recompute, and puts the surrounding stream back afterwards. ``leave_stream_at`` sets the outer
   stream to the replay's end once the step is done.

Result: the masks, and the stream the next optimizer step starts from, equal those of an
unmodified accumulation (with gradient checkpointing, which restores the state of each
layer for its recompute).

Not replayable, and refused: attention dropout inside ``scaled_dot_product_attention``
(its draws are inside a fused kernel), and any dropout that sits between the layers, such
as NEFTune noise, which accumulation interleaves with the layers.

torch is imported inside functions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Draw:
    """One dropout a layer call draws: what fixes how much of the stream it consumes."""

    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: Any
    device: str
    p: float


@dataclass(frozen=True)
class RngState:
    """The CPU generator state and the state of every initialised CUDA card."""

    cpu: Any
    cuda: Tuple[Tuple[int, Any], ...] = ()


def _cards() -> List[int]:
    import torch

    return list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []


def snapshot() -> RngState:
    import torch

    return RngState(
        cpu=torch.get_rng_state(),
        cuda=tuple((card, torch.cuda.get_rng_state(card)) for card in _cards()),
    )


def restore(state: RngState) -> None:
    """Set the global generators to ``state``."""
    import torch

    torch.set_rng_state(state.cpu)
    for card, cuda_state in state.cuda:
        torch.cuda.set_rng_state(cuda_state, card)


def leave_stream_at(state: RngState) -> None:
    """Leave the outer generators where accumulation would leave them after the step."""
    restore(state)


@contextmanager
def _forked() -> Iterator[None]:
    import warnings

    import torch

    with warnings.catch_warnings():
        # fork_rng warns when it forks more than one card; that is the intent here
        warnings.simplefilter("ignore")
        with torch.random.fork_rng(devices=_cards()):
            yield


@contextmanager
def state_at(state: RngState) -> Iterator[None]:
    """Run a layer call at ``state``, then put the surrounding stream back."""
    with _forked():
        restore(state)
        yield


@contextmanager
def record_draws(layers: Sequence[Any]) -> Iterator[List[List[Draw]]]:
    """Yield, per layer, the dropouts drawn while the block runs (filled as it runs).

    Patches ``torch.nn.functional.dropout`` (what ``nn.Dropout``, PEFT's LoRA dropout and
    eager attention call) for the block. Raises ``ValueError`` on a dropout inside
    ``scaled_dot_product_attention``, which cannot be replayed.
    """
    import torch

    functional = torch.nn.functional
    real_dropout = functional.dropout
    real_sdpa = functional.scaled_dot_product_attention
    plan: List[List[Draw]] = [[] for _ in layers]
    current: Dict[str, Optional[int]] = {"layer": None}

    def dropout(
        input: Any, p: float = 0.5, training: bool = True, inplace: bool = False,
    ) -> Any:  # noqa: A002
        if training and p > 0 and current["layer"] is not None:
            plan[current["layer"]].append(
                Draw(
                    shape=tuple(input.shape),
                    stride=tuple(input.stride()),
                    dtype=input.dtype,
                    device=str(input.device),
                    p=float(p),
                )
            )
        return real_dropout(input, p, training, inplace)

    def sdpa(*args: Any, **kwargs: Any) -> Any:
        dropout_p = kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0)
        if dropout_p and current["layer"] is not None:
            raise ValueError(
                "training.l2l cannot replay attention dropout that runs inside "
                "scaled_dot_product_attention (attention_dropout above 0 with the sdpa "
                "attention implementation). Set attention_dropout to 0 or load the model "
                "with attn_implementation='eager'."
            )
        return real_sdpa(*args, **kwargs)

    handles = []
    for idx, layer in enumerate(layers):
        handles.append(
            layer.register_forward_pre_hook(lambda _m, _a, idx=idx: current.update(layer=idx))
        )
        handles.append(layer.register_forward_hook(lambda _m, _a, _o: current.update(layer=None)))
    functional.dropout = dropout
    functional.scaled_dot_product_attention = sdpa
    try:
        yield plan
    finally:
        functional.dropout = real_dropout
        functional.scaled_dot_product_attention = real_sdpa
        for handle in handles:
            handle.remove()


def has_draws(plan: Sequence[Sequence[Draw]]) -> bool:
    return any(layer for layer in plan)


def replay(
    plan: Sequence[Sequence[Draw]], k: int
) -> Tuple[List[List[RngState]], RngState]:
    """The state before each (micro-batch, layer) pair and the state after the last one.

    Starts from the generators as they are now and leaves them as they were.
    """
    import torch

    states: List[List[RngState]] = []
    with _forked():
        for _mb in range(k):
            row: List[RngState] = []
            for layer_draws in plan:
                row.append(snapshot())
                for draw in layer_draws:
                    zeros = torch.empty_strided(
                        draw.shape, draw.stride, dtype=draw.dtype, device=draw.device
                    ).zero_()
                    torch.nn.functional.dropout(zeros, draw.p, True)
            states.append(row)
        end = snapshot()
    return states, end
