"""Select trainable decoder LoRA adapters while retaining every attached weight."""

from __future__ import annotations

import json
import math
import random
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
RECORD_NAME = "lora_layers.json"
# The save guard counts these parameters even while the policy has frozen them.
FROZEN_BY_SELECTION = "_soup_frozen_by_layer_select"
AdapterLayers = dict[int, list[tuple[str, Any]]]


@dataclass(frozen=True)
class LayerSelectPolicy:
    method: str
    k: int | None = None
    layers: tuple[int, ...] = ()
    probe_rows: int = 64
    interval_steps: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method == "fixed":
            if not self.layers:
                raise ValueError("training.lora_layers.method='fixed' requires nonnegative layers")
        elif self.method not in ("random", "grad", "angular", "lisa"):
            raise ValueError(f"training.lora_layers: unknown method {self.method!r}")
        elif self.k is None or self.k < 1:
            raise ValueError(f"training.lora_layers.method={self.method!r} requires k")


def adapter_params_by_layer(model: Any) -> AdapterLayers:
    """Every LoRA parameter of a decoder layer, keyed by the layer's index.

    Only parameters that are trainable when this is called count: the adapters as peft
    created them. Raises when the model has no per-layer adapter.
    """
    by_layer: AdapterLayers = {}
    for name, param in model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        match = _LAYER_RE.search(name)
        if match is None:
            continue
        by_layer.setdefault(int(match.group(1)), []).append((name, param))
    if not by_layer:
        raise ValueError(
            "training.lora_layers: the model has no trainable LoRA parameter inside a "
            "numbered decoder layer ('layers.N.'), so there is no layer to select"
        )
    return by_layer


def apply_layer_mask(by_layer: AdapterLayers, active: Iterable[int]) -> int:
    """Make the adapters of ``active`` layers trainable and freeze the others.

    Returns the number of trainable adapter parameters (elements).
    """
    keep = set(active)
    unknown = keep - set(by_layer)
    if unknown:
        raise ValueError(
            f"training.lora_layers: layer(s) {sorted(unknown)} have no adapter; the model's "
            f"adapted layers are {min(by_layer)}..{max(by_layer)}"
        )
    count = 0
    for idx, params in by_layer.items():
        on = idx in keep
        for _name, param in params:
            param.requires_grad_(on)
            setattr(param, FROZEN_BY_SELECTION, not on)
            if not on:
                param.grad = None
            else:
                count += param.numel()
    return count


def top_k(scores: dict[int, float], k: int) -> list[int]:
    """The ``k`` highest-scoring layers, ties broken by the lower index; sorted."""
    ranked = sorted(scores, key=lambda idx: (-scores[idx], idx))
    return sorted(ranked[: min(k, len(ranked))])


def draw(layers: Sequence[int], k: int, rng: random.Random) -> list[int]:
    return sorted(rng.sample(sorted(layers), min(k, len(layers))))


def _model_inputs(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    keep = ("input_ids", "attention_mask", "labels", "position_ids")
    return {
        key: value.to(device) for key, value in batch.items() if key in keep and value is not None
    }


def grad_scores(
    model: Any, by_layer: AdapterLayers, batches: Iterable[dict[str, Any]], device: Any
) -> dict[int, float]:
    """Mean squared gradient per adapter parameter, averaged over probe batches."""
    import torch

    sums = {idx: 0.0 for idx in by_layer}
    sizes = {idx: sum(param.numel() for _, param in params) for idx, params in by_layer.items()}
    count = 0
    try:
        with torch.enable_grad():
            for batch in batches:
                model.zero_grad(set_to_none=True)
                model(**_model_inputs(batch, device)).loss.backward()
                for idx, params in by_layer.items():
                    for _, param in params:
                        if param.grad is not None:
                            sums[idx] += float(param.grad.detach().float().square().sum())
                count += 1
    finally:
        model.zero_grad(set_to_none=True)
    if not count:
        raise ValueError("training.lora_layers: the probe saw no batch")
    return {idx: sums[idx] / sizes[idx] / count for idx in by_layer}


def angular_scores(
    model: Any, by_layer: AdapterLayers, batches: Iterable[dict[str, Any]], device: Any
) -> dict[int, float]:
    """Mean angular distance between each adapted block's input and output residual."""
    import torch

    sums = dict.fromkeys(by_layer, 0.0)
    counts = dict.fromkeys(by_layer, 0)
    mask = None

    def record(idx):
        def hook(_module, args, kwargs, output):
            before = args[0] if args else kwargs["hidden_states"]
            after = output[0] if isinstance(output, tuple) else output
            cosine = torch.nn.functional.cosine_similarity(before.float(), after.float(), dim=-1)
            distance = cosine.clamp(-1, 1).acos() / torch.pi
            if mask is not None:
                distance = distance[mask.to(device=distance.device, dtype=torch.bool)]
            sums[idx] += float(distance.sum())
            counts[idx] += distance.numel()

        return hook

    handles = []
    try:
        for idx, params in by_layer.items():
            name = params[0][0]
            module = model.get_submodule(name[: _LAYER_RE.search(name).end() - 1])
            handles.append(module.register_forward_hook(record(idx), with_kwargs=True))
        with torch.no_grad():
            for batch in batches:
                inputs = _model_inputs(batch, device)
                inputs.pop("labels", None)
                mask = inputs.get("attention_mask")
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    if not all(counts.values()):
        raise ValueError(
            "training.lora_layers: the angular probe saw no tokens for an adapted layer"
        )
    return {idx: sums[idx] / counts[idx] for idx in by_layer}


def probe_batches(dataset: Any, collator: Any, rows: int, batch_size: int) -> Iterator[Any]:
    """Collate a sequential prefix of the trainer's prepared dataset."""
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise ValueError(
            "training.lora_layers grad and angular require an indexable training dataset"
        )
    n = min(rows, len(dataset))
    for start in range(0, n, batch_size):
        yield collator([dataset[i] for i in range(start, min(n, start + batch_size))])


@dataclass
class Selection:
    method: str
    layers: list[int]
    scores: dict[int, float] = field(default_factory=dict)
    history: list[tuple[int, list[int]]] = field(default_factory=list)
    backward_cut: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "layers": self.layers,
            "backward_cut": self.backward_cut,
            "scores": {str(idx): score for idx, score in sorted(self.scores.items())},
            "history": [{"step": step, "layers": layers} for step, layers in self.history],
        }


def choose(
    policy: LayerSelectPolicy,
    model: Any,
    by_layer: AdapterLayers,
    batches: Iterable[Any] | None,
    device: Any,
    rng: random.Random,
) -> Selection:
    """Choose the initial mask and isolate scoring from the training RNG and model mode."""
    if policy.method == "fixed":
        return Selection(policy.method, sorted(set(policy.layers)))
    if policy.method in ("random", "lisa"):
        return Selection(policy.method, draw(sorted(by_layer), policy.k, rng))
    if batches is None:
        raise ValueError(f"training.lora_layers.method={policy.method!r} requires probe batches")

    import torch

    device = torch.device(device)
    cuda = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if (device.type == "cuda")
        else []
    )
    was_training = model.training
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=cuda):
            if policy.method == "grad":
                model.train()
                scores = grad_scores(model, by_layer, batches, device)
            else:
                model.eval()
                scores = angular_scores(model, by_layer, batches, device)
    finally:
        model.train(was_training)
        random.setstate(python_state)
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("training.lora_layers: the probe produced non-finite scores")
    return Selection(policy.method, top_k(scores, policy.k), scores)


_INPUT_GRAD_HOOK_NAMES = {"make_inputs_require_grads", "make_inputs_require_grad"}


def cut_input_grads(model: Any) -> str:
    """Remove known input-grad hooks when the frozen prefix needs no autograd graph."""
    for module in model.modules():
        checkpoint = getattr(module, "_gradient_checkpointing_func", None)
        if checkpoint is None or not getattr(module, "gradient_checkpointing", False):
            continue
        if (getattr(checkpoint, "keywords", None) or {}).get("use_reentrant", True) is not False:
            return "kept: reentrant gradient checkpointing needs the input grads"

    getter = getattr(model, "get_input_embeddings", None)
    embedding = getter() if callable(getter) else None
    if embedding is None:
        return "kept: no input embedding"
    if any(param.requires_grad for param in embedding.parameters()):
        return "kept: the input embedding trains"

    for key, hook in list(embedding._forward_hooks.items()):
        if getattr(hook, "__name__", "") in _INPUT_GRAD_HOOK_NAMES:
            del embedding._forward_hooks[key]
    return "cut: the backward stops at the lowest trainable layer"


class _LayerSelectCallback:
    """Choose after the optimizer has collected every adapter parameter."""

    def __init__(
        self,
        policy: LayerSelectPolicy,
        *,
        dataset: Any,
        collator: Any,
        batch_size: int,
        output_dir: str,
        console: Any = None,
        autocast: Any = nullcontext,
    ) -> None:
        self.policy = policy
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.console = console
        self.autocast = autocast
        self.rng = random.Random(policy.seed)
        self.resume_checkpoint: str | None = None
        self.by_layer: AdapterLayers = {}
        self.selection: Selection | None = None

    def _policy_record(self) -> dict[str, Any]:
        return {**asdict(self.policy), "layers": list(self.policy.layers)}

    def _write_record(self, directory: str | Path, step: int) -> None:
        path = Path(directory) / RECORD_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.selection.to_json(), "policy": self._policy_record(), "step": step}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _read_checkpoint(self, step: int) -> Selection:
        path = Path(self.resume_checkpoint) / RECORD_NAME
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["policy"] != self._policy_record() or record["step"] != step:
                raise ValueError("policy or optimizer step differs")
            selection = Selection(
                self.policy.method,
                record["layers"],
                {int(idx): score for idx, score in record["scores"].items()},
                [(draw["step"], draw["layers"]) for draw in record["history"]],
            )
            if self.policy.method == "lisa":
                # Replaying the recorded draws advances the private RNG to the same state.
                for _, layers in selection.history:
                    if draw(sorted(self.by_layer), self.policy.k, self.rng) != layers:
                        raise ValueError("the saved LISA draws differ from this model or policy")
                if not selection.history or selection.history[-1][1] != selection.layers:
                    raise ValueError("the saved LISA mask differs from the last draw")
            return selection
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"training.lora_layers cannot resume from {path}: {exc}") from exc

    def on_train_begin(self, args: Any, state: Any, control: Any = None, **kwargs: Any) -> Any:
        model = kwargs["model"]
        step = state.global_step
        # A repeated train() may reuse the model. Keep the original parameter set for
        # scoring and resampling, including adapters that the last mask froze.
        if not self.by_layer:
            self.by_layer = adapter_params_by_layer(model)
        self.rng.seed(self.policy.seed)
        if self.resume_checkpoint is not None:
            selection = self._read_checkpoint(step)
        else:
            apply_layer_mask(self.by_layer, self.by_layer)
            batches = None
            if self.policy.method in ("grad", "angular"):
                batches = probe_batches(
                    self.dataset, self.collator, self.policy.probe_rows, self.batch_size
                )
            with self.autocast():
                selection = choose(
                    self.policy, model, self.by_layer, batches, args.device, self.rng
                )
            selection.history.append((step, selection.layers))
        trainable = apply_layer_mask(self.by_layer, selection.layers)
        selection.backward_cut = cut_input_grads(model)
        self.selection = selection
        self._write_record(self.output_dir, step)
        if self.console is not None:
            self.console.print(
                f"[green]LoRA layers ({self.policy.method}):[/] training "
                f"{len(selection.layers)} of {len(self.by_layer)} decoder layers "
                f"{selection.layers} ({trainable:,} adapter parameters)"
            )
        return control

    def on_step_end(self, args: Any, state: Any, control: Any = None, **kwargs: Any) -> Any:
        step = state.global_step
        if (
            self.selection is None
            or self.policy.method != "lisa"
            or step <= 0
            or step % self.policy.interval_steps
        ):
            return control
        active = draw(sorted(self.by_layer), self.policy.k, self.rng)
        stale = set(self.selection.layers) - set(active)
        apply_layer_mask(self.by_layer, active)
        optimizer_state = getattr(kwargs.get("optimizer"), "state", None)
        if optimizer_state is not None:
            for idx in stale:
                for _, param in self.by_layer[idx]:
                    optimizer_state.pop(param, None)
        self.selection.layers = active
        self.selection.history.append((step, active))
        self._write_record(self.output_dir, step)
        return control

    def on_save(self, args: Any, state: Any, control: Any = None, **kwargs: Any) -> Any:
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        if self.selection is not None:
            directory = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            self._write_record(directory, state.global_step)
        return control


def __getattr__(name: str) -> Any:
    if name == "LayerSelectCallback":
        from transformers import TrainerCallback

        cls = type(name, (_LayerSelectCallback, TrainerCallback), {"__module__": __name__})
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def attach_layer_select_callback(
    trainer: Any, tcfg: Any, output_dir: str, console: Any = None
) -> _LayerSelectCallback | None:
    cfg = tcfg.lora_layers
    if cfg is None:
        return None

    from soup_cli.utils.layer_select import LayerSelectCallback

    defaults = LayerSelectPolicy(cfg.method, k=cfg.k, layers=tuple(cfg.layers or ()))
    callback = LayerSelectCallback(
        LayerSelectPolicy(
            defaults.method,
            k=defaults.k,
            layers=defaults.layers,
            probe_rows=cfg.probe_rows or defaults.probe_rows,
            interval_steps=cfg.interval_steps or defaults.interval_steps,
            seed=cfg.seed if cfg.seed is not None else defaults.seed,
        ),
        dataset=trainer.train_dataset,
        collator=trainer.data_collator,
        batch_size=trainer.args.per_device_train_batch_size,
        output_dir=output_dir,
        console=console,
        autocast=getattr(getattr(trainer, "accelerator", None), "autocast", nullcontext),
    )
    trainer.add_callback(callback)
    return callback
