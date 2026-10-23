"""Native A/B AdamW groups with a stable state tree for compiled Soup SFT."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any


def check_loraplus_output(output: Any, metadata: dict) -> None:
    """Preserve existing weights when their reconstruction recipe differs."""
    import json

    if not any(output.glob("*adapters.safetensors")):
        return
    try:
        previous = json.loads((output / "adapter_config.json").read_text())
        fields = ("fine_tune_type", "lora_parameters", "model", "num_layers", "soup_lora")
        if isinstance(previous, dict) and all(
            previous.get(key) == metadata.get(key) for key in fields
        ):
            return
    except (OSError, ValueError):
        pass
    raise ValueError("MLX LoRA+ requires a fresh output directory for a different adapter recipe")


def build_loraplus_optimizer(plan: Any, model: Any, ratio: float) -> Any:
    """Route only complete linear A/B pairs; clip the whole tree at the caller."""
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    from soup_cli.trainer.mlx_optim import build_optimizer

    if (type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0 or
            not math.isfinite(plan.peak_lr * ratio) or plan.optimizer_name != "AdamW"):
        raise ValueError("MLX loraplus_lr_ratio requires a finite positive ratio and AdamW")
    parameters = dict(tree_flatten(model.trainable_parameters()))
    groups = {suffix: {name[:-len(suffix)] for name in parameters if name.endswith(suffix)}
              for suffix in (".lora_a", ".lora_b")}
    if (not groups[".lora_a"] or groups[".lora_a"] != groups[".lora_b"] or
            len(parameters) != 2 * len(groups[".lora_a"]) or
            any(value.ndim != 2 for value in parameters.values())):
        raise ValueError("MLX LoRA+ requires only complete trainable linear LoRA A/B pairs")
    expected = set(parameters)

    class StableMultiOptimizer(optim.MultiOptimizer):
        # MultiOptimizer.state normally returns a NEW root on every access.
        # Soup Rewind adds recorder arrays before mx.compile captures this root;
        # retain its identity and extras while native optimizers own the moments.
        def __init__(self) -> None:
            super().__init__(
                [build_optimizer(plan),
                 build_optimizer(replace(plan, peak_lr=plan.peak_lr * ratio))],
                [lambda name, _: name.endswith(".lora_a")],
            )
            self._state = {"states": [opt.state for opt in self.optimizers]}

        @property
        def state(self) -> dict:
            return self._state

        @state.setter
        def state(self, value: dict) -> None:
            optim.MultiOptimizer.state.fset(self, value)
            self._state = value
            self._state["states"] = [opt.state for opt in self.optimizers]

        @property
        def step(self) -> Any:
            return self.optimizers[0].step

        @property
        def learning_rate(self) -> Any:
            return self.optimizers[0].learning_rate

        @learning_rate.setter
        def learning_rate(self, value: Any) -> None:
            self.optimizers[0].learning_rate = value
            self.optimizers[1].learning_rate = value * ratio

        def apply_gradients(self, gradients: dict, parameters: dict) -> dict:
            if {name for name, _ in tree_flatten(gradients)} != expected:
                raise ValueError("MLX LoRA+ gradients must contain exactly the trained A/B pairs")
            # MLX 0.31 tree_merge rejects two empty placeholders at the
            # same list index (sparse adapted layers, e.g. hybrid Nemotron).
            # Groups have disjoint validated leaves; join those leaves once.
            updates = []
            for optimizer, group in zip(self.optimizers, self._split_dictionary(gradients)):
                updates.extend(tree_flatten(optimizer.apply_gradients(group, parameters)))
            return tree_unflatten(updates)

    return StableMultiOptimizer()
