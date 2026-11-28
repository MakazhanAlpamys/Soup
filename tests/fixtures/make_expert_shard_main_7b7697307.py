"""Generator of the ``expert_shard_main_7b7697307`` fixture.

The fixture is a pair of tiny per_expert shard caches written by Soup's UNMODIFIED sharder
at commit 7b7697307 (the last commit before ``utils/expert_shard.py`` existed), so it proves
that a cache written before the storage layer was split out still loads through today's code.

Regenerate (only ever against that commit, never against a newer tree)::

    git worktree add --detach <scratch> 7b7697307
    PYTHONPATH=<scratch>/src python tests/fixtures/make_expert_shard_main_7b7697307.py <out>

``<out>/fp32`` is a float32 cache; ``<out>/nf4`` is a CPU NF4 cache with the default double
quantisation. ``source_tensors()`` is the one definition of the weights, imported by the
test; the values are integer-derived so no platform regenerates different bytes (a
torch.randn oracle did).
"""

from __future__ import annotations

import os
import sys

N_LAYERS = 2
N_EXPERTS = 3
HIDDEN = 8
INTER = 8


def _values(count, salt):
    """``count`` exactly-representable float32 values in [-2, 2), identical on every platform.

    Built from integer arithmetic only (no RNG, no libm): ``torch.randn`` is vectorised per
    CPU ISA, so the same seed gave different bytes on Windows, Linux and macOS.
    """
    import torch

    steps = torch.arange(count, dtype=torch.int64)
    return (((steps * 37 + salt * 101) % 257) - 128).to(torch.float32) / 64.0


def source_tensors():
    """The checkpoint every fixture cache was sharded from (deterministic, portable)."""
    import torch

    tensors = {"model.norm.weight": torch.ones(HIDDEN)}
    salt = 0

    def make(*shape):
        nonlocal salt
        salt += 1
        count = 1
        for dim in shape:
            count *= dim
        return _values(count, salt).reshape(*shape)

    for layer in range(N_LAYERS):
        prefix = f"model.layers.{layer}."
        tensors[prefix + "self_attn.q_proj.weight"] = make(HIDDEN, HIDDEN)
        tensors[prefix + "mlp.gate.e_score_correction_bias"] = make(N_EXPERTS)
        tensors[prefix + "mlp.shared_experts.down_proj.weight"] = make(HIDDEN, INTER)
        for expert in range(N_EXPERTS):
            for projection, shape in (
                ("gate_proj", (INTER, HIDDEN)),
                ("up_proj", (INTER, HIDDEN)),
                ("down_proj", (HIDDEN, INTER)),
            ):
                tensors[prefix + f"mlp.experts.{expert}.{projection}.weight"] = make(*shape)
    return tensors


def main(out_dir: str) -> None:
    import tempfile

    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import shard_checkpoint

    with tempfile.TemporaryDirectory() as source:
        save_file(source_tensors(), os.path.join(source, "model.safetensors"))
        shard_checkpoint(
            source,
            os.path.join(out_dir, "fp32"),
            dtype="float32",
            arch="deepseek_v3",
            num_hidden_layers=N_LAYERS,
            expert_layout="per_expert",
        )
        shard_checkpoint(
            source,
            os.path.join(out_dir, "nf4"),
            dtype="float32",
            arch="deepseek_v3",
            num_hidden_layers=N_LAYERS,
            expert_layout="per_expert",
            quant="nf4",
            quant_suffixes=("self_attn.q_proj.weight",),
            quant_device="cpu",
        )


if __name__ == "__main__":
    main(sys.argv[1])
