"""`soup merge --low-memory` must produce exactly what the regular merge produces."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _tiny_base(directory: Path, shard_size: str = "40KB") -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).to(torch.float32).save_pretrained(
        directory, max_shard_size=shard_size
    )
    return directory


def _tiny_adapter(base: Path, directory: Path, **lora_kwargs) -> Path:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    options = dict(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"])
    options.update(lora_kwargs)
    peft_model = get_peft_model(model, LoraConfig(**options))
    torch.manual_seed(1)
    for name, parameter in peft_model.named_parameters():
        if "lora_" in name:
            parameter.data = torch.randn_like(parameter) * 0.05
    peft_model.save_pretrained(directory)
    return directory


def _regular_merge(base: Path, adapter: Path, dtype: torch.dtype) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, device_map="cpu")
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return {key: value.detach().clone() for key, value in merged.state_dict().items()}


def _read_all(directory: Path) -> dict:
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        tensors.update(safetensors_torch.load_file(str(path)))
    return tensors


@pytest.fixture()
def checkpoints(tmp_path):
    base = _tiny_base(tmp_path / "base")
    adapter = _tiny_adapter(base, tmp_path / "adapter")
    return base, adapter, tmp_path / "merged"


@pytest.mark.parametrize(
    "name, dtype",
    [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)],
)
def test_matches_regular_merge_bit_for_bit(checkpoints, name, dtype):
    from soup_cli.utils.lowmem_merge import merge_low_memory

    base, adapter, output = checkpoints
    expected = _regular_merge(base, adapter, dtype)

    merge_low_memory(base, adapter, output, name)

    actual = _read_all(output)
    assert actual.keys() == expected.keys()
    for key, tensor in expected.items():
        assert actual[key].dtype == tensor.dtype, key
        assert torch.equal(actual[key], tensor), key


def test_checkpoint_is_sharded_and_loadable(checkpoints):
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.lowmem_merge import merge_low_memory

    base, adapter, output = checkpoints
    assert (base / "model.safetensors.index.json").is_file()

    total = merge_low_memory(base, adapter, output, "float16")

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == total
    assert sorted(set(index["weight_map"].values())) == sorted(
        path.name for path in output.glob("*.safetensors")
    )
    model = AutoModelForCausalLM.from_pretrained(output)
    assert model.config.torch_dtype == torch.float16
    assert next(model.parameters()).dtype == torch.float16


def test_single_file_base(tmp_path):
    from soup_cli.utils.lowmem_merge import merge_low_memory

    base = _tiny_base(tmp_path / "base", shard_size="10MB")
    assert (base / "model.safetensors").is_file()
    adapter = _tiny_adapter(base, tmp_path / "adapter")

    merge_low_memory(base, adapter, tmp_path / "merged", "float32")

    expected = _regular_merge(base, adapter, torch.float32)
    actual = _read_all(tmp_path / "merged")
    assert all(torch.equal(actual[key], value) for key, value in expected.items())


def test_rslora_scaling_matches(tmp_path):
    from soup_cli.utils.lowmem_merge import merge_low_memory

    base = _tiny_base(tmp_path / "base")
    adapter = _tiny_adapter(base, tmp_path / "adapter", use_rslora=True)

    merge_low_memory(base, adapter, tmp_path / "merged", "float32")

    expected = _regular_merge(base, adapter, torch.float32)
    actual = _read_all(tmp_path / "merged")
    assert all(torch.equal(actual[key], value) for key, value in expected.items())


def test_dora_adapter_is_refused(tmp_path):
    from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

    base = _tiny_base(tmp_path / "base")
    adapter = _tiny_adapter(base, tmp_path / "adapter", use_dora=True)

    with pytest.raises(LowMemoryMergeError, match="use_dora"):
        merge_low_memory(base, adapter, tmp_path / "merged", "float16")


def test_adapter_weights_matching_nothing_are_refused(checkpoints):
    from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

    base, adapter, output = checkpoints
    weights = safetensors_torch.load_file(str(adapter / "adapter_model.safetensors"))
    renamed = {
        key.replace("layers.0.", "layers.9."): value for key, value in weights.items()
    }
    safetensors_torch.save_file(renamed, str(adapter / "adapter_model.safetensors"))

    with pytest.raises(LowMemoryMergeError, match="match no base tensor"):
        merge_low_memory(base, adapter, output, "float16")


def test_output_must_differ_from_base(checkpoints):
    from soup_cli.utils.lowmem_merge import merge_low_memory

    base, adapter, _ = checkpoints
    with pytest.raises(ValueError, match="differ"):
        merge_low_memory(base, adapter, base, "float16")


def test_integer_tensors_keep_their_dtype(tmp_path):
    from safetensors import safe_open

    from soup_cli.utils.lowmem_merge import merge_low_memory

    base = _tiny_base(tmp_path / "base", shard_size="10MB")
    weights = safetensors_torch.load_file(str(base / "model.safetensors"))
    weights["extra.positions"] = torch.arange(6, dtype=torch.int64)
    safetensors_torch.save_file(
        weights, str(base / "model.safetensors"), metadata={"format": "pt"}
    )
    adapter = _tiny_adapter(_tiny_base(tmp_path / "plain"), tmp_path / "adapter")

    merge_low_memory(base, adapter, tmp_path / "merged", "float16")

    with safe_open(str(tmp_path / "merged" / "model.safetensors"), framework="pt") as merged:
        assert merged.get_tensor("extra.positions").dtype == torch.int64
        assert torch.equal(merged.get_tensor("extra.positions"), torch.arange(6))


class _StubTokenizer:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return SimpleNamespace(save_pretrained=lambda path: Path(path, "tokenizer.json").touch())


class TestCli:
    @pytest.fixture()
    def run(self, checkpoints, monkeypatch):
        import transformers
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.setattr(transformers, "AutoTokenizer", _StubTokenizer)
        base, adapter, _ = checkpoints
        monkeypatch.chdir(base.parent)

        def invoke(*extra):
            return CliRunner().invoke(
                app,
                ["merge", "--adapter", str(adapter), "--base", str(base), *extra],
            )

        return invoke

    def test_low_memory_merge_writes_checkpoint(self, run, checkpoints):
        base, adapter, output = checkpoints
        result = run("--output", "merged_out", "--low-memory")
        assert result.exit_code == 0, result.output
        merged = base.parent / "merged_out"
        expected = _regular_merge(base, adapter, torch.float16)
        actual = _read_all(merged)
        assert all(torch.equal(actual[key], value) for key, value in expected.items())
        assert (merged / "tokenizer.json").is_file()

    def test_rejects_non_fp16_save_format(self, run):
        result = run("--output", "merged_out", "--low-memory", "--save-format", "4bit")
        assert result.exit_code == 2
        assert "--low-memory only supports" in result.output

    def test_unsupported_adapter_points_back_to_regular_merge(self, run, checkpoints):
        _, adapter, _ = checkpoints
        config_path = adapter / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["use_dora"] = True
        config_path.write_text(json.dumps(config))
        result = run("--output", "merged_out", "--low-memory")
        assert result.exit_code == 1
        assert "without" in result.output and "--low-memory" in result.output
