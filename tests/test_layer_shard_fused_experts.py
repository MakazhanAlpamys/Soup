"""Round-trip sharding for fused routed-expert checkpoint tensors."""

from pathlib import Path


def test_fused_checkpoint_shards_each_expert(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.expert_shard import ExpertShardReader, aligned_staging
    from soup_cli.utils.layer_shard import expert_paths, shard_checkpoint

    weights = tmp_path / "weights"
    weights.mkdir()
    gate_up = torch.arange(72, dtype=torch.float32).reshape(3, 6, 4)
    down = torch.arange(36, dtype=torch.float32).reshape(3, 4, 3) + 100
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.eye(4),
            "model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.layers.0.mlp.experts.down_proj": down,
            "model.norm.weight": torch.ones(4),
        },
        str(weights / "model.safetensors"),
    )

    out = str(tmp_path / "shards")
    index = shard_checkpoint(
        str(weights), out, arch="olmoe", dtype="float32", expert_layout="per_expert"
    )
    assert index.expert_layers == (0,)
    assert index.expert_count == gate_up.shape[0]
    reader = ExpertShardReader(
        expert_paths(out, index)[0],
        count=index.expert_count,
        shapes=index.expert_shapes,
        quant_specs=index.expert_quant_specs,
        dtype=index.dtype,
        direct=False,
    )
    try:
        _owner, staging = aligned_staging(reader.max_span)
        for expert in range(index.expert_count):
            stored = reader.read_host(expert, staging)
            torch.testing.assert_close(stored[f"{expert}.gate_up"], gate_up[expert], rtol=0, atol=0)
            torch.testing.assert_close(stored[f"{expert}.down"], down[expert], rtol=0, atol=0)
    finally:
        reader.close()
