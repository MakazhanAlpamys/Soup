"""#1241 — `soup shrink` must keep per-layer config lists in step with the layers.

`prune_model_layers` drops a block of decoder layers and patches
`config.num_hidden_layers`. Qwen2, Qwen3, Qwen2-MoE and SmolLM3 configs also
carry per-layer lists (`layer_types`, and SmolLM3's `no_rope_layers`). Left at
the old length, transformers 5 refuses to save the pruned model, so the shrink
run dies after scoring with nothing written.

The lists must be sliced with the same kept indices as the layers, not cut at
the tail: a mixed full/sliding Qwen2 list and SmolLM3's `no_rope_layers` both
have a different entry in the dropped block than at the end, so a tail cut
fails here.
"""
import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_COMMON = dict(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=6,
    num_attention_heads=2,
    num_key_value_heads=2,
)

_LINEAR_ATTN = dict(
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
)

_CASES = [
    ("LlamaConfig", {}),
    ("Qwen2Config", {}),
    (
        "Qwen2Config",
        {"use_sliding_window": True, "max_window_layers": 3, "sliding_window": 8},
    ),
    ("Qwen3Config", {"head_dim": 8}),
    (
        "Qwen2MoeConfig",
        {
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
        },
    ),
    (
        "Qwen3MoeConfig",
        {"head_dim": 8, "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 16},
    ),
    ("SmolLM3Config", {"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2}),
    # Hybrid: `layer_types` decides whether each layer is built with linear or
    # full attention, so a wrong slice changes the architecture, not a flag.
    ("Qwen3_5TextConfig", {"head_dim": 8, **_LINEAR_ATTN}),
]

# (start, block_size) on 6 layers: first allowed position, middle, ending at
# n - 1, and the largest block the first/last-layer guard allows.
_DROPS = [(1, 1), (2, 2), (4, 1), (1, 4)]


def _per_layer_lists(config, n_layers: int) -> dict:
    return {
        key: list(value)
        for key, value in vars(config).items()
        if isinstance(value, (list, tuple)) and len(value) == n_layers
    }


@pytest.mark.parametrize("start,block_size", _DROPS)
@pytest.mark.parametrize(
    "config_name,extra", _CASES, ids=[f"{name}-{i}" for i, (name, _) in enumerate(_CASES)]
)
def test_prune_slices_per_layer_lists_and_round_trips(
    tmp_path, config_name, extra, start, block_size
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from soup_cli.utils.shrink import prune_model_layers

    config = getattr(transformers, config_name)(**_COMMON, **extra)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    before = _per_layer_lists(model.config, 6)
    kept = [i for i in range(6) if not (start <= i < start + block_size)]

    prune_model_layers(model, start, block_size)

    for key, old in before.items():
        assert list(getattr(model.config, key)) == [old[i] for i in kept], key

    model.save_pretrained(str(tmp_path))
    reloaded = AutoConfig.from_pretrained(str(tmp_path))
    assert reloaded.num_hidden_layers == len(kept)
    for key, old in before.items():
        assert list(getattr(reloaded, key)) == [old[i] for i in kept], key
    AutoModelForCausalLM.from_pretrained(str(tmp_path))


@pytest.mark.parametrize(
    "config_name,extra", _CASES, ids=[f"{name}-{i}" for i, (name, _) in enumerate(_CASES)]
)
def test_every_per_layer_list_is_in_the_declared_table(config_name, extra):
    """A per-layer list a supported config carries, missing from the table,
    would be left at the old length by the prune — the next #1241."""
    from soup_cli.utils.shrink import _PER_LAYER_CONFIG_ATTRS

    config = getattr(transformers, config_name)(**_COMMON, **extra)
    assert set(_per_layer_lists(config, 6)) <= set(_PER_LAYER_CONFIG_ATTRS)


_QWEN2_MOE = (
    "Qwen2MoeConfig",
    {
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
    },
)
_QWEN3_MOE = (
    "Qwen3MoeConfig",
    {"head_dim": 8, "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 16},
)
_LLAMA4_TEXT = (
    "Llama4TextConfig",
    {"head_dim": 8, "num_local_experts": 2, "num_experts_per_tok": 1, "intermediate_size_mlp": 32},
)

# Index-valued MoE layout attributes: a non-default value names layers by
# number, which a slice cannot remap, so the prune must refuse it.
_LAYER_INDEX_CASES = [
    (*_QWEN2_MOE, "mlp_only_layers", [3]),
    (*_QWEN3_MOE, "decoder_sparse_step", 2),
    (*_LLAMA4_TEXT, "moe_layers", [1, 3]),
    (*_LLAMA4_TEXT, "interleave_moe_layer_step", 2),
]
_LAYER_INDEX_IDS = [attr for _, _, attr, _ in _LAYER_INDEX_CASES]


@pytest.mark.parametrize(
    "config_name,extra,attr,value", _LAYER_INDEX_CASES, ids=_LAYER_INDEX_IDS
)
def test_prune_refuses_a_non_default_layer_index_attr(config_name, extra, attr, value):
    import torch
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.shrink import prune_model_layers

    config = getattr(transformers, config_name)(**_COMMON, **extra, **{attr: value})
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)

    with pytest.raises(ValueError, match=attr):
        prune_model_layers(model, 1, 1)
    assert len(model.model.layers) == 6
    assert model.config.num_hidden_layers == 6


@pytest.mark.parametrize(
    "config_name,extra,attr,value", _LAYER_INDEX_CASES, ids=_LAYER_INDEX_IDS
)
def test_prune_accepts_the_default_layer_index_attr(tmp_path, config_name, extra, attr, value):
    import torch
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.shrink import prune_model_layers

    config = getattr(transformers, config_name)(**_COMMON, **extra)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)

    prune_model_layers(model, 1, 1)

    assert len(model.model.layers) == 5
    model.save_pretrained(str(tmp_path))
    _, info = AutoModelForCausalLM.from_pretrained(str(tmp_path), output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"], info


def _write_tiny_qwen2(dir_path, lines):
    """Tiny random-init Qwen2 + a byte-level BPE tokenizer trained on ``lines``
    (no download)."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(lines, vocab_size=300, min_frequency=1, special_tokens=["<eos>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<eos>", pad_token="<eos>")
    config = transformers.Qwen2Config(
        **{**_COMMON, "vocab_size": len(tok)},
        use_sliding_window=True,
        max_window_layers=3,
        sliding_window=8,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(str(dir_path))
    tok.save_pretrained(str(dir_path))
    return str(dir_path)


def test_shrink_cli_saves_a_pruned_qwen2(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from soup_cli.cli import app
    from tests.conftest import strip_ansi

    monkeypatch.chdir(tmp_path)
    lines = [
        "the quick brown fox jumps over the lazy dog",
        "a small model keeps the first and last layer",
        "pruning drops a contiguous block of layers",
        "the calibration set scores each block",
    ]
    model_dir = _write_tiny_qwen2(tmp_path / "tiny-qwen2", lines)
    (tmp_path / "calib.jsonl").write_text(
        "\n".join(json.dumps({"text": line}) for line in lines), encoding="utf-8"
    )
    out_dir = tmp_path / "shrunk"
    r = CliRunner().invoke(
        app,
        ["shrink", "--model", model_dir, "--drop-layers", "2", "--calib", "calib.jsonl",
         "--device", "cpu", "--output-dir", str(out_dir), "--tolerance", "5.0"],
    )
    plain = " ".join(strip_ansi(r.output).split())
    assert r.exit_code == 0, (plain, repr(r.exception))
    cfg = json.loads((out_dir / "model" / "config.json").read_text(encoding="utf-8"))
    assert cfg["num_hidden_layers"] == 4
    assert len(cfg["layer_types"]) == 4
    report = json.loads((out_dir / "shrink_report.json").read_text(encoding="utf-8"))
    assert report["layers_before"] == 6 and report["layers_after"] == 4
    assert "Layers: 6 -> 4" in plain, plain
