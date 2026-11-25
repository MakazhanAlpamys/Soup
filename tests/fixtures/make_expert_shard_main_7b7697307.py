"""Generator of the ``expert_shard_main_7b7697307`` fixture.

The fixture is a pair of tiny per_expert shard caches written by Soup's UNMODIFIED sharder
at commit 7b7697307 (the last commit before ``utils/expert_shard.py`` existed), so it proves
that a cache written before the storage layer was split out still loads through today's code.

Regenerate (only ever against that commit, never against a newer tree)::

    git worktree add --detach <scratch> 7b7697307
    PYTHONPATH=<scratch>/src python tests/fixtures/make_expert_shard_main_7b7697307.py <out>

``<out>/fp32`` is a float32 cache; ``<out>/nf4`` is a CPU NF4 cache with the default double
quantisation. ``source_tensors()`` is the one definition of the weights, imported by the test.
"""

from __future__ import annotations

import os
import sys

N_LAYERS = 2
N_EXPERTS = 3
HIDDEN = 8
INTER = 8
SEED = 20260725


def source_tensors():
    """The checkpoint every fixture cache was sharded from (deterministic)."""
    import torch

    generator = torch.Generator().manual_seed(SEED)
    tensors = {"model.norm.weight": torch.ones(HIDDEN)}
    for layer in range(N_LAYERS):
        prefix = f"model.layers.{layer}."
        tensors[prefix + "self_attn.q_proj.weight"] = torch.randn(
            HIDDEN, HIDDEN, generator=generator
        )
        tensors[prefix + "mlp.gate.e_score_correction_bias"] = torch.randn(
            N_EXPERTS, generator=generator
        )
        tensors[prefix + "mlp.shared_experts.down_proj.weight"] = torch.randn(
            HIDDEN, INTER, generator=generator
        )
        for expert in range(N_EXPERTS):
            for projection, shape in (
                ("gate_proj", (INTER, HIDDEN)),
                ("up_proj", (INTER, HIDDEN)),
                ("down_proj", (HIDDEN, INTER)),
            ):
                tensors[prefix + f"mlp.experts.{expert}.{projection}.weight"] = torch.randn(
                    *shape, generator=generator
                )
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
