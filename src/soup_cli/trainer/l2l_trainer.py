"""SFT trainer mode for L2L (``training.l2l``).

HF's loop prefetches ``gradient_accumulation_steps`` batches per optimizer step
(``Trainer.get_batch_samples``) and then calls ``training_step`` once per batch. The
subclass keeps that loop and everything it owns (``num_items_in_batch``, lr schedule,
logging, checkpoints, resume) and changes two calls:

* ``get_batch_samples`` remembers the window of k batches HF just built;
* ``training_step`` runs one layer-major step over the whole window on the window's
  first call and returns the summed loss. The other k-1 calls of the window return a
  zero and compute nothing, so HF's accumulation of the returned losses and its one
  optimizer step per window stay as they are.

Each micro-batch runs at its own length, as in gradient accumulation: nothing is padded.
The layer arguments (position ids, mask) are captured once per distinct shape and each
micro-batch gets the call of its shape. The activation store sizes every micro-batch's
chunks to that micro-batch, so a short row does not pay for the longest row of its window.

FP8 is an opt-in boundary codec: capacities retain the logical hidden width for
per-token scales, and the host lease includes codec scratch, not device workspace.
It is lossy and does not relax the routing guard. An already-resolved memory plan
can bound each window; the trainer never selects k or changes HF's schedule.

Each micro-batch position keeps its largest capacity seen so far. A larger window grows
the store after releasing its old allocations. When the store does not fit the host-RAM
budget, the lowest layers spill to one file (``soup_cli.utils.l2l_spill``); ``train`` removes
it on every exit it sees and sweeps files of dead runs at start.

torch is imported inside functions.
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from soup_cli.utils.expert_stream import EXPERT_LAYOUT_PER_EXPERT
from soup_cli.utils.host_memory_budget import HostMemoryBudget, MemoryLease
from soup_cli.utils.l2l_step import (
    LoadCounters,
    LossHead,
    ReferencePass,
    as_passes,
    refuse_moe_l2l,
    sft_head,
)

if TYPE_CHECKING:
    from soup_cli.utils.moe_l2l_budget import MoEL2LMemoryPlan

#: Architectures ``training.l2l`` admits at launch with any weight layout.
PUBLIC_ARCHS = ("llama", "mistral")

#: Architectures ``training.l2l`` admits only with ``training.stream_expert_layout``
#: set to ``EXPERT_LAYOUT_PER_EXPERT``: the DeepSeek-V3 step reads each routed expert from its
#: own cache entry, which a fused layer cache does not have.
PER_EXPERT_ARCHS = ("deepseek_v3",)

logger = logging.getLogger(__name__)

#: Tasks whose L2L result is verified to match gradient accumulation. Any other task the
#: caller admits runs with a warning.
L2L_READY_TASKS = ("sft", "dpo")


def unready_task_warning(task: Optional[str]) -> Optional[str]:
    """The warning for a task that ``training.l2l`` accepts but that is not ready; else None."""
    if task is None or task in L2L_READY_TASKS:
        return None
    return (
        f"training.l2l with task '{task}' is not ready: its result is not verified to "
        "match gradient accumulation."
    )


_ABSENT = object()


def refuse_unsupported_trainer(
    trainer: Any, expected: Sequence[Tuple[str, Any, str]], *, reproduces: str
) -> None:
    """Refuse a built TRL trainer whose loss is not the one an L2L window loss reproduces.

    ``expected`` lists ``(attribute, wanted value, name shown)``. An attribute the trainer
    does not have is refused too: a TRL that renames it would otherwise pass unchecked.
    """
    for attribute, wanted, name in expected:
        value = getattr(trainer, attribute, _ABSENT)
        if value is _ABSENT:
            raise ValueError(
                f"{reproduces}; this trainer has no attribute '{attribute}' ({name}), so "
                "the check cannot tell whether its loss is still the one reproduced here "
                "(TRL may have renamed or removed it)."
            )
        if value != wanted:
            raise ValueError(
                f"{reproduces}; this trainer has {name}={value!r} (wanted {wanted!r})."
            )

#: Captured layer arguments kept per shape; the cache restarts when a run exceeds it. A new
#: shape costs one forward to capture them.
MAX_CACHED_SHAPES = 64


def refuse_unsupported_l2l(
    model_config: Any, *, n_gpu: int, fp16: bool = False, expert_layout: str = "fused"
) -> None:
    """Refuse at launch what the schema cannot see: the model, the GPU count, fp16.

    ``expert_layout`` is ``training.stream_expert_layout``; DeepSeek-V3 needs ``per_expert``.
    """
    arch = str(getattr(model_config, "model_type", ""))
    refuse_moe_l2l(arch)
    if arch in PER_EXPERT_ARCHS:
        if expert_layout != EXPERT_LAYOUT_PER_EXPERT:
            raise ValueError(
                f"training.l2l with model_type {arch!r} requires "
                f"training.stream_expert_layout={EXPERT_LAYOUT_PER_EXPERT!r}; "
                f"got {expert_layout!r}."
            )
    elif arch not in PUBLIC_ARCHS:
        raise ValueError(
            f"training.l2l does not support model_type {arch!r}: it is enabled for "
            f"{' and '.join(PUBLIC_ARCHS)}, and for {' and '.join(PER_EXPERT_ARCHS)} "
            f"with training.stream_expert_layout={EXPERT_LAYOUT_PER_EXPERT!r}."
        )
    if n_gpu > 1:
        raise ValueError(
            f"training.l2l is single-GPU, but {n_gpu} GPUs are visible. Re-run with "
            f"one card visible, e.g. CUDA_VISIBLE_DEVICES=0."
        )
    if fp16:
        raise ValueError(
            "training.l2l does not support fp16: the L2L step runs backward itself, "
            "so the fp16 grad scaler would never scale the loss. Use bf16 or fp32."
        )


@dataclass
class L2LState:
    """What the trainer subclass needs besides the HF trainer."""

    runtime: Any
    dtype: Any
    window: Optional[List[Any]] = None
    position: int = 0
    steps: int = 0
    has_dropout: Optional[bool] = None
    totals: LoadCounters = field(default_factory=LoadCounters)
    store: Any = None
    stepper: Any = None
    calls_by_shape: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    #: How a window is read, which head it runs and what its loss is divided by.
    #: ``None`` means SFT (:data:`SFT_WINDOW_LOSS`).
    loss: Optional["WindowLoss"] = None
    #: Capture forwards run, the layer loads they cost and their wall time (running totals).
    captures: int = 0
    capture_loads: int = 0
    capture_seconds: float = 0.0
    #: ``training.l2l_spill_dir``; None means ``<output_dir>/_l2l_spill``.
    spill_dir: Optional[str] = None
    #: Directories the weights stream from (layer cache and stripe folders).
    weight_dirs: Tuple[str, ...] = ()
    output_dir: Optional[str] = None
    #: The plan of the current store (kept after it is closed); None before the first step.
    spill_plan: Optional[Dict[str, Any]] = None
    #: Files of dead runs the last ``train`` removed from the spill directory.
    stale_removed: List[str] = field(default_factory=list)
    c3_plan: Any = None
    #: Boundary storage only; compute and routing checks retain their existing behavior.
    activation_codec: Literal["none", "fp8_e4m3"] = "none"
    #: An already-resolved MoEL2LMemoryPlan. The trainer never selects accumulation k.
    memory_plan: Optional[MoEL2LMemoryPlan] = None
    host_budget: Optional[HostMemoryBudget] = None
    _routing_lease: Optional[MemoryLease] = field(default=None, init=False, repr=False)
    #: The recipe's effective k (``soup_cli.utils.l2l_recipe``); None when the caller gave none.
    effective_k: Optional[int] = None
    two_read_experts: bool = False
    grouped_expert_routes: bool = False
    route_preserving_mbr: bool = False
    mbr_group_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.loss is None:
            self.loss = SFT_WINDOW_LOSS
        if self.activation_codec not in ("none", "fp8_e4m3"):
            raise ValueError(f"unsupported L2L activation codec {self.activation_codec!r}")
        if self.activation_codec == "fp8_e4m3":
            if self.loss is not SFT_WINDOW_LOSS:
                raise ValueError("FP8 activation storage supports the SFT window loss only")
            if str(self.dtype).removeprefix("torch.") not in ("float32", "bfloat16"):
                raise ValueError("FP8 activation compute dtype must be float32 or bfloat16")
        if (
            getattr(self.runtime, "expert_pool", None) is not None
            and self.memory_plan is None
            and (
                self.activation_codec != "none"
                or self.host_budget is not None or self.c3_plan is not None
            )
        ):
            raise ValueError("budgeted MoE routing requires a resolved memory_plan")
        if self.memory_plan is not None:
            if self.host_budget is None:
                raise ValueError("an L2L memory_plan requires an explicit host_budget")
            if self.memory_plan.codec != self.activation_codec:
                raise ValueError("L2L activation codec differs from the resolved memory plan")
        if (
            self.c3_plan is not None and self.host_budget is not None
            and self.host_budget is not self.c3_plan.activations
        ):
            raise ValueError(
                "L2L host_budget must be the reserved activation partition of a shared-budget store"
            )
        if self.c3_plan is not None and self.memory_plan is not None:
            routing_bytes = (
                self.memory_plan.routing_record_bytes + self.memory_plan.routing_comparison_bytes
            )
            if self.c3_plan.routing_bytes != routing_bytes:
                raise ValueError(
                    "combined shared-budget and L2L memory plans require matching routing_bytes "
                    "reserved "
                    "in the shared activation partition before weight allocation"
                )

    def log_fields(self) -> Dict[str, Any]:
        fields = {
            "l2l_steps": self.steps,
            "l2l_captures": self.captures,
            "l2l_capture_layer_loads": self.capture_loads,
            "l2l_capture_seconds": self.capture_seconds,
            "l2l_layer_loads": self.totals.layer_loads,
            "l2l_large_loads": self.totals.large_loads,
            "l2l_spill_layers": int(self.spill_plan["spill_layers"]) if self.spill_plan else 0,
        }
        if self.totals.restores is not None:
            fields["l2l_weight_restores"] = self.totals.restores
        if self.effective_k is not None:
            fields["l2l_effective_k"] = self.effective_k
        # Only a runtime that streams routed experts has them to count; a dense run's log
        # keeps the fields it always had.
        if getattr(self.runtime, "expert_pool", None) is not None:
            fields["l2l_expert_fetches"] = self.totals.expert_fetches
            fields["l2l_expert_bytes_read"] = self.totals.expert_bytes_read
            fields["l2l_two_read_experts"] = self.two_read_experts
            fields["l2l_grouped_expert_routes"] = self.grouped_expert_routes
        if self.route_preserving_mbr:
            fields["l2l_route_preserving_mbr"] = True
            fields["l2l_mbr_group_size"] = self.mbr_group_size or self.effective_k
        return fields

    def spill_directory(self) -> str:
        """The configured spill directory, or the default one under the output directory."""
        if self.spill_dir is not None:
            return self.spill_dir
        from soup_cli.utils.l2l_spill import SPILL_DIR_NAME

        return os.path.join(self.output_dir or ".", SPILL_DIR_NAME)

    def close_store(self) -> None:
        """Drain the current store before growth without releasing routing headroom."""
        if self.store is not None:
            self.store.close()
            self.store = None

    def close(self) -> None:
        """Release the routing reservation only after activation users have drained."""
        self.close_store()
        if self._routing_lease is not None:
            self._routing_lease.close()
            self._routing_lease = None


def require_right_padded(mask: Any) -> None:
    """Refuse a window row whose attention mask has a padding token before a real token."""
    if mask is not None and bool((mask[:, 1:] > mask[:, :-1]).any()):
        raise ValueError(
            "training.l2l needs right-padded rows: a padding token precedes "
            "a real token in the attention mask."
        )


def refuse_mixed_shapes(ids_list: List[Any], *, neftune: bool, has_dropout: bool) -> None:
    """Refuse a window whose micro-batches differ in shape with NEFTune or dropout on.

    Dropout: ``l2l_rng.replay`` builds the generator states from one draw plan shared by
    all micro-batches, and rows of different lengths draw different masks. NEFTune: the
    noise scales with ``1 / sqrt(seq * hidden)`` and the combination has not been checked
    against gradient accumulation on rows of different lengths.
    """
    shapes = {tuple(ids.shape) for ids in ids_list}
    mixed = len(shapes) > 1
    if mixed and neftune:
        raise ValueError(
            "training.l2l with training.neftune_alpha needs every micro-batch of a window "
            f"to have one shape, but this window has {sorted(shapes)}: the L2L step has "
            "not been checked against gradient accumulation for NEFTune noise on rows of "
            "different lengths. Use rows of one length, or turn neftune_alpha off."
        )
    if mixed and has_dropout:
        raise ValueError(
            "training.l2l with dropout (training.lora.dropout or attention_dropout above 0) "
            f"needs every micro-batch of a window to have one shape, but this window has "
            f"{sorted(shapes)}: rows of different lengths draw different dropout masks, and "
            "the replay of l2l_rng uses one draw plan for all micro-batches. Use rows of "
            "one length, or set the dropout to 0."
        )


def stack_pair_window(
    window: List[Any],
    device: Any,
    *,
    trainer_cls: Any,
    padding_value: int,
    neftune: bool,
    has_dropout: bool,
) -> Tuple[List[Any], List[Dict[str, Any]], List[int]]:
    """Read a window of ``chosen_*`` / ``rejected_*`` micro-batches as TRL stacks them.

    TRL's ``ORPOTrainer`` and ``CPOTrainer`` share one ``concatenated_inputs``: chosen rows
    over rejected rows, both padded to the longer width, ``input_ids`` with ``padding_value``.
    Returns, per micro-batch, the stacked ids, TRL's ``concatenated_*`` dict and the number of
    chosen rows. Refuses left padding and the mixed-shape cases like the SFT reader.
    """
    import torch

    ids_list: List[Any] = []
    stacked_list: List[Dict[str, Any]] = []
    chosen_rows: List[int] = []
    for inputs in window:
        moved = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        stacked = trainer_cls.concatenated_inputs(
            moved, is_encoder_decoder=False, padding_value=padding_value, device=device
        )
        require_right_padded(stacked["concatenated_attention_mask"])
        ids_list.append(stacked["concatenated_input_ids"])
        stacked_list.append(stacked)
        chosen_rows.append(int(moved["chosen_labels"].shape[0]))
    refuse_mixed_shapes(ids_list, neftune=neftune, has_dropout=has_dropout)
    return ids_list, stacked_list, chosen_rows


def _window_tensors(
    window: List[Any], device: Any, *, neftune: bool = False, has_dropout: bool = False
) -> Tuple[List[Any], List[Any]]:
    """Input ids and labels of each micro-batch, each at its own shape."""
    ids_list, label_list = [], []
    for inputs in window:
        ids = inputs["input_ids"].to(device)
        labels = inputs.get("labels")
        labels = ids if labels is None else labels.to(device)
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
            require_right_padded(mask)
        ids_list.append(ids)
        label_list.append(labels)
    refuse_mixed_shapes(ids_list, neftune=neftune, has_dropout=has_dropout)
    return ids_list, label_list


@dataclass(frozen=True)
class WindowLoss:
    """What a task contributes to the L2L trainer: reading, head, and the window's scale.

    ``divide_by="items"``: the head divides by HF's ``num_items_in_batch`` (SFT sums token
    losses over the window). ``divide_by="micro_batches"``: the head divides by
    ``k = len(window)``, as HF's ``training_step`` does when ``num_items_in_batch`` is
    ``None`` (every preference trainer averages its rows and ignores the item count). The
    last window of an epoch holds fewer than ``gradient_accumulation_steps`` micro-batches,
    and ``k`` is that smaller count.
    """

    divide_by: Literal["items", "micro_batches"]
    #: ``(window, device, *, neftune, has_dropout) -> (ids_list, extras)``. ``ids_list`` is
    #: what the layers walk; ``extras`` is handed to ``make_head``.
    read_window: Callable[..., Tuple[List[Any], Any]]
    #: ``(extras, divisor) -> LossHead``; the head returns the loss already scaled.
    make_head: Callable[[Any, int], LossHead]
    #: ``extras -> ReferencePass`` or a sequence of them, for a loss that needs gradient-free
    #: passes: a reference model (DPO, KTO) or an estimate from the policy itself (KTO's KL);
    #: ``None`` for a loss that does not. The policy head reads the passes' ``outputs`` after
    #: the step filled them, so both come from the same ``extras``.
    make_reference: Optional[
        Callable[[Any], Union[ReferencePass, Sequence[ReferencePass]]]
    ] = None

    def divisor(self, k: int, num_items: Optional[int]) -> int:
        """The number the window's loss is divided by. The one place the scale rule lives."""
        if self.divide_by == "micro_batches":
            return int(k)
        if num_items is None:
            raise RuntimeError(
                "training.l2l needs num_items_in_batch from the HF loop to scale "
                "the loss over the window; this trainer did not receive one."
            )
        return int(num_items)


SFT_WINDOW_LOSS = WindowLoss(
    divide_by="items", read_window=_window_tensors, make_head=sft_head
)


def _has_dropout(model: Any) -> bool:
    """True when a forward in training draws dropout masks: any ``nn.Dropout`` with p > 0
    (LoRA dropout) or a non-zero ``attention_dropout`` on the config or a module."""
    import torch

    config = getattr(model, "config", None)
    if float(getattr(config, "attention_dropout", 0.0) or 0.0) != 0:
        return True
    return any(
        (isinstance(module, torch.nn.Dropout) and module.p > 0)
        or float(getattr(module, "attention_dropout", 0.0) or 0.0) != 0
        for module in model.modules()
    )


@contextmanager
def _timed_capture(state: L2LState) -> Iterator[None]:
    """Add one capture, its layer loads and its wall time to the state's totals."""
    import time

    import torch

    from soup_cli.utils.l2l_step import count_loads

    def sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    sync()
    start = time.perf_counter()
    with count_loads(state.runtime) as counters:
        try:
            yield
        finally:
            sync()
            state.captures += 1
            state.capture_seconds += time.perf_counter() - start
    state.capture_loads += counters.layer_loads


def _calls_for(state: L2LState, model: Any, ids_list: Sequence[Any]) -> List[Any]:
    """One captured ``LayerCall`` per micro-batch, from the shapes the run has seen."""
    from soup_cli.utils.l2l_step import capture_layer_call

    if state.has_dropout is None:
        state.has_dropout = _has_dropout(model)
    calls = []
    for ids in ids_list:
        shape = (int(ids.shape[0]), int(ids.shape[1]))
        call = state.calls_by_shape.get(shape)
        if call is None:
            if len(state.calls_by_shape) >= MAX_CACHED_SHAPES:
                state.calls_by_shape.clear()
            # Only a model with dropout needs the full forward, to record the draws.
            with _timed_capture(state):
                call = capture_layer_call(model, ids, draws=state.has_dropout)
            state.calls_by_shape[shape] = call
        calls.append(call)
    return calls


def _l2l_step_for(state: L2LState, model: Any, ids_list: List[Any]) -> Tuple[Any, List[Any]]:
    """The window's L2L step and one captured ``LayerCall`` per micro-batch."""
    from soup_cli.utils.l2l_step import L2LStep

    calls = _calls_for(state, model, ids_list)
    if state.stepper is None:
        state.stepper = L2LStep(model, state.runtime, calls[0])
    return state.stepper, calls


def _spill_line(plan: Dict[str, Any], budget: Any) -> str:
    from soup_cli.utils.terminal import for_terminal

    return (
        f"[yellow]L2L:[/] activations do not fit host RAM; spilling "
        f"{plan['spill_layers']} of {plan['n_layers']} layers "
        f"({plan['spill_file_bytes'] / 1e9:.2f} GB) to {for_terminal(plan['spill_path'])}; "
        f"{for_terminal(budget.describe())}"
    )


def _store_for(state: L2LState, stepper: Any, ids_list: List[Any], device: Any) -> Any:
    """The activation store for the window, grown only when a micro-batch does not fit.

    Micro-batch ``i`` of every window uses chunk ``i``, so the store keeps the largest
    size each position has needed. The plan sums those allocations for resident layers;
    spill slots and staging buffers hold the largest chunk. Replanning and directory
    validation happen after the old store is released and before the new allocation.
    """
    import math

    import torch

    from soup_cli.utils import l2l_spill
    from soup_cli.utils.l2l_activations import ActivationStore

    hidden = int(stepper.causal.config.hidden_size)
    store = state.store
    n_layers = len(stepper.layers)
    pin = str(device).startswith("cuda")
    dtype = str(state.dtype).removeprefix("torch.")
    if not ids_list or any(ids.ndim != 2 or min(ids.shape) < 1 for ids in ids_list):
        raise ValueError("L2L activation windows need nonempty two-dimensional micro-batches")
    if n_layers < 1 or hidden < 1:
        raise ValueError("L2L activation windows need positive layer and hidden dimensions")
    shapes = [(int(ids.shape[0]), int(ids.shape[1]), hidden) for ids in ids_list]
    if state.c3_plan is not None:
        # Before any release: an out-of-plan window retains the current store and lease.
        state.c3_plan.validate_microbatches(
            n_layers=n_layers, shapes=shapes, dtype=dtype, pin=pin,
            codec=state.activation_codec,
            device_type=torch.device(device).type,
        )
    if state.memory_plan is not None:
        state.memory_plan.validate_microbatches(
            n_layers=n_layers, shapes=shapes, dtype=dtype, pin=pin,
            codec=state.activation_codec,
        )
        if torch.device(device).type != state.memory_plan.device_type:
            raise ValueError("L2L device differs from the resolved memory plan")
        if getattr(state.runtime, "expert_pool", None) is not None:
            moe_layers = len(state.runtime.expert_readers)
            top_k = getattr(stepper.causal.config, "num_experts_per_tok", None)
            if (
                not moe_layers or moe_layers > state.memory_plan.n_moe_layers
                or type(top_k) is not int or top_k < 1
                or top_k > state.memory_plan.experts_per_token
            ):
                raise ValueError("MoE routing dimensions exceed the resolved memory plan")
    if store is not None and (state.activation_codec != "none" or state.memory_plan is not None):
        expected_device = torch.device(device)
        actual_device = torch.device(store.device)
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())
        if actual_device.type == "cuda" and actual_device.index is None:
            actual_device = torch.device("cuda", torch.cuda.current_device())
        if (
            store.n_layers != n_layers or store.dtype != state.dtype
            or actual_device != expected_device
        ):
            raise ValueError("L2L activation store layers, dtype or device changed between windows")
        if state.activation_codec == "fp8_e4m3" and any(
            len(shape) < 2 or shape[-1] != hidden for shape in store.mb_shapes
        ):
            raise ValueError("FP8 activation store hidden width changed between windows")
    need = [math.prod(shape) for shape in shapes]
    have = [math.prod(shape) for shape in store.mb_shapes] if store is not None else []
    caps = [max(n, h) for n, h in zip(need, have)] + need[len(have):] + have[len(need):]
    fp8 = state.activation_codec == "fp8_e4m3"
    mb_shapes = [(cap // hidden, hidden) for cap in caps] if fp8 else [(cap,) for cap in caps]
    if fp8:
        from soup_cli.utils.fp8_activations import fp8_activation_allocation

        allocation = fp8_activation_allocation(
            n_layers=n_layers, mb_shapes=mb_shapes, pin=pin,
            compute_dtype=dtype, device_type=torch.device(device).type,
        ).host_bytes
    elif state.memory_plan is not None:
        from soup_cli.utils.budgeted_activations import activation_allocation_bytes

        allocation = activation_allocation_bytes(
            n_layers=n_layers, k=len(caps),
            chunk_bytes=max(caps) * torch.empty((), dtype=state.dtype).element_size(),
            spill_layers=0, ring=1, pin=pin,
        )
    if state.memory_plan is not None:
        if allocation > state.memory_plan.activation_bytes:
            raise ValueError("L2L activation allocation exceeds the resolved memory plan")
        routing = (
            state.memory_plan.routing_record_bytes + state.memory_plan.routing_comparison_bytes
        )
        if state._routing_lease is None:
            state._routing_lease = state.host_budget.reserve("l2l.routing", routing)
        elif state._routing_lease.nbytes != routing:
            raise ValueError("L2L routing reservation differs from the resolved memory plan")
    if store is not None and len(need) <= len(have) and all(
        n <= h for n, h in zip(need, have)
    ):
        return store
    rebuilding_cuda = store is not None and pin
    state.close_store()
    # Drop old storage before replanning. Routing headroom survives routine store growth.
    del store
    if rebuilding_cuda:
        empty_host_cache = getattr(getattr(torch, "accelerator", None), "empty_host_cache", None)
        if empty_host_cache is not None:
            empty_host_cache()
        else:
            # Older torch has no supported cache-release API. Do not invent reusable-byte
            # credit: a larger window can need different pinned allocation size buckets.
            logger.warning(
                "training.l2l is growing its CUDA activation store, but this torch has no "
                "torch.accelerator.empty_host_cache(); idle pinned blocks may remain charged "
                "to host RAM, so the new plan may need extra free RAM."
            )
    host_budget = state.host_budget
    if state.c3_plan is not None:
        host_budget = state.c3_plan.activations
    if fp8 or host_budget is not None:
        if host_budget is None:
            safe_budget = l2l_spill.ram_budget().budget_bytes
            if safe_budget is None or safe_budget <= 0:
                raise ValueError("FP8 activation storage requires a known positive host RAM budget")
            host_budget = HostMemoryBudget(safe_budget)
        if fp8:
            from soup_cli.utils.fp8_activations import FP8ActivationStore

            factory = FP8ActivationStore
            chunk_shape = (max(caps) // hidden, hidden)
        else:
            from soup_cli.utils.budgeted_activations import BudgetedActivationStore

            factory = BudgetedActivationStore
            chunk_shape = (max(caps),)
        try:
            state.store = factory(
                host_budget=host_budget, n_layers=n_layers, k=len(caps),
                chunk_shape=chunk_shape, mb_shapes=mb_shapes,
                dtype=state.dtype, device=device, pin=pin,
            )
        except BaseException:
            # A failed store constructor owns its own cleanup; no routing guard ran yet.
            state.close()
            raise
        state.spill_plan = {
            "n_layers": n_layers, "spill_layers": 0, "codec": state.activation_codec,
        }
        return state.store
    itemsize = torch.empty((), dtype=state.dtype).element_size()
    budget = l2l_spill.ram_budget()
    logger.info(
        "training.l2l activation budget: %s [limiting level %s: %s]",
        budget.describe(), budget.limiting_level, budget.limiting_numbers,
    )
    plan = l2l_spill.plan_spill(
        budget.budget_bytes,
        n_layers=n_layers,
        k=len(caps),
        chunk_bytes=max(caps) * itemsize,
        mb_bytes=[cap * itemsize for cap in caps],
        ring=l2l_spill.TRAINER_RING,
        pin=pin,
        budget_source=budget.describe(),
    )
    record = {
        **plan.as_dict(),
        "available_bytes": budget.available_bytes,
        "limiting_level": budget.limiting_level,
        "limiting_numbers": dict(budget.limiting_numbers),
        "spill_dir": state.spill_directory(),
        "spill_path": None,
        "explicit_dir": state.spill_dir is not None,
    }
    factory = None
    if plan.spill_layers:
        directory = l2l_spill.prepare_spill_dir(
            record["spill_dir"],
            explicit=record["explicit_dir"],
            spill_layers=plan.spill_layers,
            file_bytes=plan.spill_file_bytes,
            tier=str(getattr(state.runtime, "tier", "ram")),
            weight_dirs=state.weight_dirs,
        )
        record["spill_dir"] = directory
        record["spill_path"] = os.path.join(directory, l2l_spill.spill_file_name())
        factory = functools.partial(l2l_spill.open_spill_file, record["spill_path"])
    state.store = ActivationStore(
        n_layers=n_layers,
        k=len(caps),
        chunk_shape=(max(caps),),
        mb_shapes=[(cap,) for cap in caps],
        dtype=state.dtype,
        device=device,
        pin=pin,
        spill_layers=plan.spill_layers,
        spill_factory=factory,
        ring=l2l_spill.TRAINER_RING,
    )
    state.spill_plan = record
    if plan.spill_layers:
        l2l_spill.say(_spill_line(record, budget))
    return state.store


def run_window(
    state: L2LState,
    model: Any,
    ids_list: List[Any],
    label_list: Any,
    n_items: Optional[int],
    device: Any,
) -> Tuple[List[Any], Any]:
    """One L2L step over the window; returns the losses and the step's load counters.

    ``label_list`` is what the state's strategy read from the window besides the ids;
    ``n_items`` is HF's ``num_items_in_batch``, ``None`` for the preference trainers.
    """
    strategy = state.loss
    divisor = strategy.divisor(len(ids_list), n_items)
    stepper, calls = _l2l_step_for(state, model, ids_list)
    store = _store_for(state, stepper, ids_list, device)
    head = strategy.make_head(label_list, divisor)
    reference = None if strategy.make_reference is None else strategy.make_reference(label_list)
    for extra in as_passes(reference):
        # A pass over ids of its own (KTO's KL completions) takes calls of its own shapes.
        if extra.micro_batches is not None and extra.calls is None:
            extra.calls = _calls_for(state, model, extra.micro_batches)
    routing = {}
    if state.memory_plan is not None:
        routing["routing_budget"] = (
            state.memory_plan.routing_record_bytes, state.memory_plan.routing_comparison_bytes
        )
    losses = stepper.run(
        ids_list,
        store,
        calls=calls,
        head=head,
        reference=reference,
        **routing,
        route_preserving_mbr=state.route_preserving_mbr,
        mbr_group_size=state.mbr_group_size,
    )
    return losses, stepper.counters


@functools.lru_cache(maxsize=None)
def make_l2l_trainer_class(base_cls: type) -> type:
    """Return a subclass of ``base_cls`` that runs each accumulation window as one L2L step.

    Cached, so two calls with the same ``base_cls`` return the same subclass. Without
    :func:`attach_l2l_state` the subclass behaves as ``base_cls``.
    """

    class L2LTrainer(base_cls):  # type: ignore[misc, valid-type]
        soup_l2l: bool = True
        _soup_l2l: Optional[L2LState] = None

        def train(self, *args, **kwargs):  # type: ignore[override]
            """HF's ``train`` with the spill directory swept before it and the activation
            store (and its spill file) closed after it, on a normal end, an exception or
            Ctrl-C alike."""
            state = self._soup_l2l
            if state is None:
                return super().train(*args, **kwargs)
            from soup_cli.utils.l2l_spill import sweep_stale

            state.stale_removed = sweep_stale(state.spill_directory())
            try:
                return super().train(*args, **kwargs)
            finally:
                state.close()

        def get_batch_samples(self, epoch_iterator, num_batches, device):  # type: ignore[override]
            samples, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
            if self._soup_l2l is not None:
                self._soup_l2l.window = list(samples)
                self._soup_l2l.position = 0
            return samples, num_items

        def training_step(  # type: ignore[override]
            self, model, inputs, num_items_in_batch=None
        ):
            import torch

            state = self._soup_l2l
            if state is None or state.window is None:
                return super().training_step(model, inputs, num_items_in_batch)
            first = state.position == 0
            state.position += 1
            window = state.window
            if state.position >= len(window):
                state.window = None
            if not first:
                return torch.zeros((), device=self.args.device)
            # Refuse a missing item count (SFT) before anything runs.
            state.loss.divisor(len(window), num_items_in_batch)
            model.train()
            if state.has_dropout is None:
                state.has_dropout = _has_dropout(model)
            if self.args.neftune_noise_alpha and state.has_dropout:
                raise ValueError(
                    "training.l2l cannot combine neftune_alpha with dropout "
                    "(training.lora.dropout or attention_dropout above 0): accumulation "
                    "draws the NEFTune noise from the same stream as the dropout masks, "
                    "between the layers, and a layer-major pass cannot replay that. Turn "
                    "neftune_alpha off or set the dropout to 0."
                )
            device = state.runtime.device
            ids_list, label_list = state.loss.read_window(
                window,
                device,
                neftune=bool(self.args.neftune_noise_alpha),
                has_dropout=state.has_dropout,
            )
            # The step bypasses the wrapped forward, where the Trainer applies its mixed
            # precision; forward, head and the recompute in backward all run inside it.
            with self.accelerator.autocast():
                losses, counters = run_window(
                    state,
                    model,
                    ids_list,
                    label_list,
                    None if num_items_in_batch is None else int(num_items_in_batch),
                    device,
                )
            state.steps += 1
            state.totals.add(counters)
            return torch.stack(losses).sum().to(self.args.device)

        def log(self, logs, *args, **kwargs):  # type: ignore[override]
            state = self._soup_l2l
            if state is not None and state.steps and "loss" in logs:
                logs = {**logs, **state.log_fields()}
            return super().log(logs, *args, **kwargs)

    return L2LTrainer


def attach_l2l_state(
    trainer: Any,
    runtime: Any,
    *,
    dtype: Any,
    loss: Optional[WindowLoss] = None,
    c3_plan: Any = None,
    activation_codec: Literal["none", "fp8_e4m3"] = "none",
    memory_plan: Optional[MoEL2LMemoryPlan] = None,
    host_budget: Optional[HostMemoryBudget] = None,
    spill_dir: Optional[str] = None,
    weight_dirs: Sequence[str] = (),
    effective_k: Optional[int] = None,
    two_read_experts: bool = False,
    grouped_expert_routes: bool = False,
    route_preserving_mbr: bool = False,
    mbr_group_size: Optional[int] = None,
) -> L2LState:
    """Give a trainer from :func:`make_l2l_trainer_class` its streaming runtime.

    ``loss`` is the task's :class:`WindowLoss`; the default is SFT. ``spill_dir`` is
    ``training.l2l_spill_dir`` (None: ``<output_dir>/_l2l_spill``), and ``weight_dirs`` lists
    the directories the weights stream from on the disk tier.
    ``memory_plan`` is already resolved and must match HF's accumulation count. Its
    routing allocation and activation storage use the supplied ``host_budget``; a shared-budget
    store instead uses its reserved activation partition. Without either budget, FP8
    reads the safe RAM budget and refuses an unknown limit rather than spilling.
    """
    args = getattr(trainer, "args", None)
    if memory_plan is not None and (
        args is None or args.gradient_accumulation_steps != memory_plan.effective_k
    ):
        raise ValueError(
            "L2L gradient_accumulation_steps must equal the resolved memory_plan.effective_k; "
            "the trainer does not change the scheduler or trim windows"
        )
    state = L2LState(
        runtime=runtime,
        dtype=dtype,
        loss=loss,
        c3_plan=c3_plan,
        activation_codec=activation_codec,
        memory_plan=memory_plan,
        host_budget=host_budget,
        effective_k=effective_k,
        spill_dir=spill_dir,
        weight_dirs=tuple(str(path) for path in weight_dirs),
        two_read_experts=two_read_experts,
        grouped_expert_routes=grouped_expert_routes,
        route_preserving_mbr=route_preserving_mbr,
        mbr_group_size=mbr_group_size,
    )
    # The trainer's run directory is the fallback location for activation spill.
    state.output_dir = getattr(args, "output_dir", None)
    trainer._soup_l2l = state
    return state
