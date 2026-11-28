"""Capture one pretrained streaming arm for a separate, unchanged-tolerance comparison.

Local snapshot and finite dataset only; never provisions hardware. All trainable
adapters are explicitly initialized, including nonzero B, before either arm trains.
The comparison consumes saved tensors on CPU and checks every element.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def configure_determinism() -> None:
    """Seed alone does not make CUDA SDPA backward repeatable on A40."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def initialize_adapters(model: object) -> dict:
    """Identical, nonzero trainable state in every correctness and timing arm."""
    import torch

    adapters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert adapters and all("lora_" in name for name in adapters)
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for name, value in sorted(adapters.items()):
            scale = 0.003 if "lora_B" in name else 0.02
            value.copy_(torch.randn(value.shape, generator=generator) * scale)
    return adapters


def capture(config_path: str, arm: str, output: Path) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import TrainerCallback, set_seed

    from benchmarks.harness.liger_streaming_bench import build_config
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.seeding import resolve_training_seed

    configure_determinism()
    output.mkdir(parents=True, exist_ok=False)
    config = build_config(load_config(config_path), arm=arm, output=str(output / "adapter"))
    set_seed(resolve_training_seed(config.training))
    wrapper = SFTTrainerWrapper(config, device="cuda", report_to="none")
    tensors, batches, nodes, losses = {}, [], set(), []
    report = {"arm": arm, "status": "STARTED", "rtol": 1e-2, "atol": 1e-5}
    try:
        wrapper.setup(load_dataset(config.data))
        trainer = wrapper.trainer
        trainer.args.max_steps = 2
        trainer.args.disable_tqdm = True
        trainer.args.logging_steps = 1
        trainer.args.logging_nan_inf_filter = False
        adapters = initialize_adapters(wrapper.model)
        assert (
            len(adapters) == wrapper.model.get_base_model().config.num_hidden_layers * 14
            and all("lora_" in name for name in adapters)
        )
        with torch.no_grad():
            for name, value in sorted(adapters.items()):
                tensors["initial/" + name] = value.detach().cpu().clone()

        def observe(_module, _args, kwargs, result):
            index = len(batches)
            batch = {}
            for name, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    saved = value.detach().cpu().contiguous()
                    tensors[f"batch/{index}/{name}"] = saved.clone()
                    batch[name] = {"shape": list(saved.shape), "dtype": str(saved.dtype),
                                   "sha256": hashlib.sha256(saved.numpy().tobytes()).hexdigest()}
            batches.append(batch)
            tensors[f"logits/{index}"] = result.logits.detach().cpu().clone()
            losses.append(float(result.loss.detach()))
            pending, visited = [result.logits.grad_fn], set()
            while pending:
                node = pending.pop()
                if node is None or node in visited:
                    continue
                visited.add(node)
                name = node.name()
                if "Liger" in name:
                    node.register_hook(lambda *_args, name=name: nodes.add(name))
                pending.extend(child for child, _index in node.next_functions)

        handle = wrapper.model.register_forward_hook(observe, with_kwargs=True)

        class Capture(TrainerCallback):
            step = 0

            def on_pre_optimizer_step(self, _args, _state, _control, **_kwargs):
                for name, value in adapters.items():
                    assert value.grad is not None
                    tensors[f"gradient/{self.step}/{name}"] = value.grad.detach().cpu().clone()
                self.step += 1

        callback = Capture()
        trainer.add_callback(callback)
        wrapper.train()
        handle.remove()
        assert callback.step == 2 and len(batches) == 4
        for name, value in adapters.items():
            tensors["final/" + name] = value.detach().cpu().clone()
        tensors["loss"] = torch.tensor(losses, dtype=torch.float64)
        expected_nodes = {"rms_norm": "RMS", "swiglu": "SiLUMul"}
        for component, marker in expected_nodes.items():
            assert any(marker in name for name in nodes) == (component in arm)
        assert all(torch.isfinite(value).all() for value in tensors.values())
        report.update(status="CAPTURED", patch_counts=wrapper._streaming_liger_patches,
                      batches=batches, backward_nodes=sorted(nodes), losses=losses,
                      runtime=wrapper._stream_runtime.stats(), adapter_tensors=len(adapters),
                      amp={"bf16": trainer.args.bf16, "fp16": trainer.args.fp16},
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      log_history=trainer.state.log_history)
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_file({name: value.contiguous() for name, value in tensors.items()},
                  str(output / "trace.safetensors"))
        (output / "capture.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        wrapper._close_stream_runtime()


def compare(reference: Path, candidate: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    first = load_file(str(reference / "trace.safetensors"))
    second = load_file(str(candidate / "trace.safetensors"))
    assert set(first) == set(second), "trace keys differ"
    groups = {}
    for name, value in first.items():
        other = second[name]
        group = name.split("/")[0]
        stats = groups.setdefault(group, {"tensors": 0, "failed_tensors": 0,
                                         "max_abs": 0.0, "failures": []})
        stats["tensors"] += 1
        exact = group in {"batch", "initial"}
        finite = bool(torch.isfinite(value).all() and torch.isfinite(other).all())
        close = torch.equal(value, other) if exact else torch.allclose(
            value, other, rtol=1e-2, atol=1e-5)
        if not (finite and close):
            stats["failed_tensors"] += 1
            stats["failures"].append(name)
        difference = float((value.double() - other.double()).abs().max())
        stats["max_abs"] = max(stats["max_abs"], difference)
    loss_match = all(f"{a:.3f}" == f"{b:.3f}" for a, b in
                     zip(first["loss"].tolist(), second["loss"].tolist(), strict=True))
    return {"passed": loss_match and all(g["failed_tensors"] == 0 for g in groups.values()),
            "loss_three_decimals": loss_match, "groups": groups, "rtol": 1e-2, "atol": 1e-5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.config, args.arm, args.output)


if __name__ == "__main__":
    main()
