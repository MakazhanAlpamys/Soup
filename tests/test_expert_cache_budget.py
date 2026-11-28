"""The conservative header-only budget for a new per-expert NF4 cache."""

from __future__ import annotations

from pathlib import Path

import pytest


def _checkpoint(root: Path, count: int = 4):
    import torch
    from safetensors.torch import save_file

    root.mkdir()
    tensors = {"model.norm.weight": torch.ones(4)}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        tensors[prefix + "self_attn.q_proj.weight"] = torch.eye(4)
        tensors[prefix + "mlp.gate.e_score_correction_bias"] = torch.arange(count).float()
        tensors[prefix + "mlp.shared_experts.down_proj.weight"] = torch.ones(4, 3)
        for expert in range(count):
            for projection, shape in (
                ("gate_proj", (3, 4)), ("up_proj", (3, 4)), ("down_proj", (4, 3)),
            ):
                tensors[prefix + f"mlp.experts.{expert}.{projection}.weight"] = (
                    torch.arange(12).reshape(shape).float() + 100 * expert + 1000 * layer
                )
    save_file(tensors, str(root / "model.safetensors"))
    return tensors


def _cache_size(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def test_header_budget_covers_serialized_expert_cache(tmp_path):
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget
    from soup_cli.utils.layer_shard import shard_checkpoint

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, count=4)
    paths = [str(source / "model.safetensors")]
    budget = new_expert_cache_budget(paths)
    selected = new_expert_cache_budget(
        paths, quant_suffixes=("self_attn.q_proj.weight",), n_layers=1,
    )
    shard_checkpoint(
        str(source), str(output), dtype="float32", arch="deepseek_v3", num_hidden_layers=2,
        expert_layout="per_expert", quant="nf4", quant_suffixes=("self_attn.q_proj.weight",),
        double_quant=False,
    )
    assert _cache_size(output) < budget
    assert _cache_size(output) < selected
    assert selected < budget


def test_budget_refuses_truncated_source_before_using_shapes(tmp_path):
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget

    path = tmp_path / "broken.safetensors"
    path.write_bytes((100).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError, match="truncated safetensors header"):
        new_expert_cache_budget([str(path)])


def test_budget_needs_source_files():
    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget

    with pytest.raises(ValueError, match="requires complete source files"):
        new_expert_cache_budget([])


@pytest.mark.parametrize("n_layers", [0, -1, True, 1.5])
def test_budget_rejects_invalid_explicit_layer_count(n_layers):
    from soup_cli.utils.expert_cache_budget import expert_cache_budget_from_headers

    with pytest.raises(ValueError, match="budget layer count must be a positive integer"):
        expert_cache_budget_from_headers([], n_layers=n_layers)


def test_dropped_source_layer_still_requires_valid_offsets():
    from soup_cli.utils.expert_cache_budget import expert_cache_budget_from_headers

    header = {"model.layers.2.mlp.experts.0.gate_proj.weight":
              {"shape": [64, 64], "dtype": "F32", "data_offsets": [0, 16384]}}
    with pytest.raises(ValueError, match="source tensor offsets do not fit the file"):
        expert_cache_budget_from_headers([(1024, 128, header)], n_layers=1)


def test_quantization_suffixes_must_name_weight_tensors():
    from soup_cli.utils.expert_cache_budget import expert_cache_budget_from_headers

    with pytest.raises(ValueError, match="must name weight tensors"):
        expert_cache_budget_from_headers([], quant_suffixes=("self_attn.q_proj.bias",))


def test_a_source_without_routed_experts_is_refused():
    from soup_cli.utils.expert_cache_budget import expert_cache_budget_from_headers

    header = {"model.norm.weight": {"shape": [4], "dtype": "F32", "data_offsets": [0, 16]}}
    with pytest.raises(ValueError, match="source contains no supported routed experts"):
        expert_cache_budget_from_headers([(64, 40, header)])


def test_a_duplicate_tensor_key_across_files_is_refused():
    from soup_cli.utils.expert_cache_budget import expert_cache_budget_from_headers

    header = {"model.layers.0.mlp.experts.0.gate_proj.weight":
              {"shape": [2, 2], "dtype": "F32", "data_offsets": [0, 16]}}
    with pytest.raises(ValueError, match="duplicate or oversized source tensor key"):
        expert_cache_budget_from_headers([(256, 128, header), (256, 128, header)])


def test_an_unsupported_dtype_is_refused():
    from soup_cli.utils.expert_cache_budget import expert_cache_budget_from_headers

    header = {"model.layers.0.mlp.experts.0.gate_proj.weight":
              {"shape": [2, 2], "dtype": "I64", "data_offsets": [0, 32]}}
    with pytest.raises(ValueError, match="unsupported tensor shape or dtype"):
        expert_cache_budget_from_headers([(256, 128, header)])
