"""Read-only diagnostics for a progressive LoRA run."""

from __future__ import annotations

import math
import random
import sys
from contextlib import contextmanager
from statistics import mean, pstdev
from typing import Any, Iterable, Iterator

LayerFeatures = dict[int, dict[str, float]]


def adapter_update_features(layers: dict[int, Any]) -> LayerFeatures:
    """Sum scaled update norms and average the effective ranks of the linear adapters."""
    import torch

    features = {}
    for idx, layer in layers.items():
        updates, ranks = [], []
        for module in layer.modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            for name in module.active_adapters:
                if name not in module.lora_A or name not in module.lora_B:
                    continue
                a = module.lora_A[name].weight.detach().float()
                b = module.lora_B[name].weight.detach().float()
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                    continue
                # This rank-by-rank product has the squared singular values of B @ A.
                gram = (b.T @ b).cpu().double() @ (a @ a.T).cpu().double()
                if not torch.isfinite(gram).all():
                    continue
                eig = torch.linalg.eigvals(gram).real.clamp_min(0)
                total = float(eig.sum())
                squared = float(eig.square().sum())
                scale = abs(float(module.scaling[name]))
                updates.append(scale * math.sqrt(total))
                if squared and scale:
                    ranks.append(min(a.shape[0], total * total / squared))
        if updates:
            features[idx] = {"update": sum(updates), "update_rank": mean(ranks) if ranks else 0.0}
    return features


def grad_features(norms: Iterable[tuple[int, dict[int, float]]]) -> LayerFeatures:
    """Summarize completed windows, including finite-window checks for older records."""
    by_layer: dict[int, list[float]] = {}
    for _, window in norms:
        if all(math.isfinite(norm) for norm in window.values()):
            for idx, norm in window.items():
                by_layer.setdefault(idx, []).append(norm)
    features = {}
    for idx, values in by_layer.items():
        average = mean(values)
        features[idx] = {
            "grad_mean": average,
            "grad_cv": pstdev(values) / average if average else 0.0,
        }
    return features


@contextmanager
def _probe_context(model: Any, device: Any) -> Iterator[None]:
    import torch

    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    modes = [(module, module.training) for module in model.modules()]
    cuda = []
    if device.type == "cuda":
        cuda = [device.index if device.index is not None else torch.cuda.current_device()]
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        with torch.random.fork_rng(devices=cuda), torch.no_grad():
            model.eval()
            yield
    finally:
        for module, was_training in modes:
            module.training = was_training
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


class LayerProbe:
    """Keep fixed inputs and up to 32 sampled output tokens per row and layer."""

    def __init__(self, layers: dict[int, Any]) -> None:
        self.layers = layers
        self.batches: list[dict[str, Any]] = []
        self.before: dict[int, Any] = {}

    def start(
        self,
        model: Any,
        batches: Iterable[dict[str, Any]],
        device: Any,
        *,
        capture_baseline: bool = True,
    ) -> None:
        import torch

        device = torch.device(device)
        with _probe_context(model, device):
            self.batches = [
                {key: batch[key] for key in ("input_ids", "attention_mask", "position_ids")
                 if batch.get(key) is not None}
                for batch in batches
            ]
            if capture_baseline:
                self.before, _ = self._capture(model, device, measure_turn=False)

    def finish(self, model: Any, device: Any) -> LayerFeatures:
        import torch

        device = torch.device(device)
        with _probe_context(model, device):
            after, turns = self._capture(model, device, measure_turn=True)
        features = {}
        previous = 0.0
        for idx, output in sorted(after.items()):
            entry = {}
            if self.before:
                change = 1.0 - linear_cka(self.before[idx], output)
                if math.isfinite(change):
                    entry.update(cka_change=change, cka_own=max(0.0, change - previous))
                    previous = change
            if turns[idx]:
                entry["turn"] = float(torch.cat(turns[idx]).mean())
            if entry:
                features[idx] = entry
        return features

    def _capture(
        self, model: Any, device: Any, *, measure_turn: bool
    ) -> tuple[dict[int, Any], dict[int, list[Any]]]:
        import torch

        outputs: dict[int, list[Any]] = {idx: [] for idx in self.layers}
        turns: dict[int, list[Any]] = {idx: [] for idx in self.layers}
        indices = None

        def record(idx: int) -> Any:
            def hook(_module: Any, args: tuple, kwargs: dict, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                sample = hidden[0, indices.to(hidden.device)].detach().float()
                outputs[idx].append(sample.cpu())
                before = args[0] if args else kwargs.get("hidden_states")
                if measure_turn and torch.is_tensor(before):
                    incoming = before[0, indices.to(before.device)].detach().float()
                    cosine = torch.nn.functional.cosine_similarity(incoming, sample, dim=-1)
                    turns[idx].append((cosine.clamp(-1.0, 1.0).acos() / math.pi).cpu())

            return hook

        handles = []
        rows = 0
        try:
            for idx, layer in self.layers.items():
                handles.append(layer.register_forward_hook(record(idx), with_kwargs=True))
            for batch in self.batches:
                ids = batch["input_ids"]
                mask = batch.get("attention_mask")
                valid = torch.arange(ids.shape[-1]) if mask is None else mask[0].nonzero().flatten()
                count = valid.numel()
                if not count:
                    continue
                offsets = torch.linspace(0, count - 1, min(count, 32)).round().long()
                indices = valid[offsets.to(valid.device)].to(device)
                model(**{key: value.to(device) for key, value in batch.items()}, use_cache=False)
                rows += 1
        finally:
            for handle in handles:
                handle.remove()
        if not rows or any(len(samples) != rows for samples in outputs.values()):
            raise ValueError("the activation probe did not see every decoder layer on every row")
        return {idx: torch.cat(samples) for idx, samples in outputs.items()}, turns


def linear_cka(before: Any, after: Any) -> float:
    """Centered linear CKA using the smaller of token and hidden dimensions."""
    x = before - before.mean(0)
    y = after - after.mean(0)
    if x.shape[0] <= x.shape[1]:
        xx, yy = x @ x.T, y @ y.T
        numerator = (xx * yy).sum()
    else:
        xx, yy = x.T @ x, y.T @ y
        numerator = (x.T @ y).norm().square()
    denominator = float(xx.norm() * yy.norm())
    return float((numerator / denominator).clamp(0, 1)) if denominator else 1.0


def merge_features(*parts: LayerFeatures) -> dict[str, dict[str, float]]:
    layers: LayerFeatures = {}
    for part in parts:
        for idx, values in part.items():
            layers.setdefault(idx, {}).update(
                (key, round(value, 6)) for key, value in values.items() if math.isfinite(value)
            )
    return {str(idx): dict(sorted(values.items())) for idx, values in sorted(layers.items())}


def loss_summary(logs: Iterable[dict[str, Any]]) -> dict[str, float | int | None] | None:
    losses = []
    for entry in logs:
        if "loss" not in entry:
            continue
        try:
            loss = float(entry["loss"])
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(loss):
            losses.append(loss)
    if len(losses) < 4:
        return None
    quarter = len(losses) // 4
    first, last = mean(losses[:quarter]), mean(losses[-quarter:])
    drop = (first - last) / first if first else None
    return {
        "first_quarter": first,
        "last_quarter": last,
        "drop": drop if drop is not None and math.isfinite(drop) else None,
        "steps": len(losses),
    }
