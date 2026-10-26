"""`soup merge --low-memory` must produce exactly what the regular merge produces."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import strip_ansi

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _tiny_base(directory: Path, shard_size: str = "40KB", tie: bool = False) -> Path:
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
        tie_word_embeddings=tie,
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
        assert result.exit_code == 0, (result.output, repr(result.exception))
        merged = base.parent / "merged_out"
        expected = _regular_merge(base, adapter, torch.float16)
        actual = _read_all(merged)
        assert all(torch.equal(actual[key], value) for key, value in expected.items())
        assert (merged / "tokenizer.json").is_file()

    def test_rejects_non_fp16_save_format(self, run):
        result = run("--output", "merged_out", "--low-memory", "--save-format", "4bit")
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--low-memory only supports" in strip_ansi(result.output)

    def test_unsupported_adapter_points_back_to_regular_merge(self, run, checkpoints):
        _, adapter, _ = checkpoints
        config_path = adapter / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["use_dora"] = True
        config_path.write_text(json.dumps(config))
        result = run("--output", "merged_out", "--low-memory")
        assert result.exit_code == 1, (result.output, repr(result.exception))
        output = strip_ansi(result.output)
        assert "Low-memory merge not possible" in output
        assert "without" in output and "--low-memory" in output

    def test_hub_base_is_fetched_by_snapshot_download(self, checkpoints, monkeypatch):
        import transformers
        from typer.testing import CliRunner

        from soup_cli.cli import app

        base, adapter, _ = checkpoints
        monkeypatch.setattr(transformers, "AutoTokenizer", _StubTokenizer)
        monkeypatch.chdir(base.parent)
        fetched = {}

        def fake_download(repo_id, **kwargs):
            fetched["repo_id"] = repo_id
            fetched.update(kwargs)
            return str(base)

        monkeypatch.setattr("soup_cli.utils.hubs.snapshot_download", fake_download)

        result = CliRunner().invoke(
            app,
            [
                "merge", "--adapter", str(adapter), "--base", "org/tiny-llama",
                "--output", "merged_out", "--low-memory",
            ],
        )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert fetched["repo_id"] == "org/tiny-llama"
        assert fetched["allow_patterns"] == ["*.safetensors", "*.json"]
        expected = _regular_merge(base, adapter, torch.float16)
        actual = _read_all(base.parent / "merged_out")
        assert all(torch.equal(actual[key], value) for key, value in expected.items())

    def test_tokenizer_failure_leaves_no_output(self, run, checkpoints, monkeypatch):
        import transformers

        class _Broken:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise RuntimeError("tokenizer exploded")

        monkeypatch.setattr(transformers, "AutoTokenizer", _Broken)
        base, _, _ = checkpoints
        result = run("--output", "merged_out", "--low-memory")
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert sorted(path.name for path in base.parent.iterdir()) == ["adapter", "base"]

    def test_progress_text_is_neutralised_and_trust_flag_is_forwarded(
        self, checkpoints, monkeypatch
    ):
        import transformers
        from typer.testing import CliRunner

        from soup_cli.cli import app

        base, adapter, _ = checkpoints
        monkeypatch.setattr(transformers, "AutoTokenizer", _StubTokenizer)
        monkeypatch.chdir(base.parent)
        seen = {}

        def fake_merge(base_dir, adapter_dir, output_dir, dtype, **kwargs):
            seen.update(kwargs)
            kwargs["progress"]("Merging [red]EVIL[/red]\x1b[31m shard")
            output_dir.mkdir()
            return 0

        monkeypatch.setattr("soup_cli.utils.lowmem_merge.merge_low_memory", fake_merge)
        result = CliRunner().invoke(
            app,
            [
                "merge", "--adapter", str(adapter), "--base", str(base),
                "--output", "merged_out", "--low-memory", "--trust-remote-code",
            ],
        )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert seen["trust_remote_code"] is True
        output = strip_ansi(result.output)
        assert "[red]EVIL[/red]" in output
        assert "\x1b" not in output


def _rewrite_adapter_weights(adapter: Path, edit) -> None:
    path = adapter / "adapter_model.safetensors"
    weights = safetensors_torch.load_file(str(path))
    edit(weights)
    safetensors_torch.save_file(weights, str(path), metadata={"format": "pt"})


def _rewrite_adapter_config(adapter: Path, **changes) -> None:
    path = adapter / "adapter_config.json"
    config = json.loads(path.read_text())
    config.update(changes)
    path.write_text(json.dumps(config))


def _first_adapter_target(base: Path, adapter: Path) -> str:
    weights = safetensors_torch.load_file(str(adapter / "adapter_model.safetensors"))
    key = sorted(weights)[0]
    return key.split(".lora_")[0].replace("base_model.model.", "") + ".weight"


class TestAdapterCoverage:
    def test_fan_in_fan_out_matches_regular_merge(self, tmp_path):
        from transformers import GPT2Config, GPT2LMHeadModel

        from soup_cli.utils.lowmem_merge import merge_low_memory

        torch.manual_seed(0)
        config = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=4)
        base = tmp_path / "base"
        GPT2LMHeadModel(config).save_pretrained(base)
        adapter = _tiny_adapter(
            base, tmp_path / "adapter",
            target_modules=["c_attn", "c_proj"], fan_in_fan_out=True,
        )

        merge_low_memory(base, adapter, tmp_path / "merged", "float32")

        expected = _regular_merge(base, adapter, torch.float32)
        actual = _read_all(tmp_path / "merged")
        original = safetensors_torch.load_file(str(base / "model.safetensors"))
        assert all(torch.equal(value, expected[key]) for key, value in actual.items())
        assert any(not torch.equal(value, original[key]) for key, value in actual.items())

    def test_tied_embeddings_match_regular_merge(self, tmp_path):
        from soup_cli.utils.lowmem_merge import merge_low_memory

        base = _tiny_base(tmp_path / "base", tie=True)
        adapter = _tiny_adapter(base, tmp_path / "adapter")

        merge_low_memory(base, adapter, tmp_path / "merged", "float32")

        expected = _regular_merge(base, adapter, torch.float32)
        actual = _read_all(tmp_path / "merged")
        assert set(actual) <= set(expected)
        assert all(torch.equal(value, expected[key]) for key, value in actual.items())

    @pytest.mark.parametrize(
        "field, value",
        [
            ("modules_to_save", ["lm_head"]),
            ("rank_pattern", {"q_proj": 2}),
            ("alpha_pattern", {"q_proj": 2}),
            ("trainable_token_indices", {"embed_tokens": [1]}),
            ("loftq_config", {"loftq_bits": 4}),
            ("lora_bias", True),
            ("layer_replication", [[0, 1]]),
            ("target_parameters", ["mlp.experts.gate_up_proj"]),
            ("alora_invocation_tokens", [1, 2]),
            ("use_qalora", True),
        ],
    )
    def test_unsupported_config_fields_are_refused(self, checkpoints, field, value):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints
        _rewrite_adapter_config(adapter, **{field: value})
        with pytest.raises(LowMemoryMergeError, match=field):
            merge_low_memory(base, adapter, output, "float16")
        assert not output.exists()

    @pytest.mark.parametrize("changes", [{"peft_type": "IA3"}, {"bias": "all"}])
    def test_other_adapter_types_are_refused(self, checkpoints, changes):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints
        _rewrite_adapter_config(adapter, **changes)
        with pytest.raises(LowMemoryMergeError, match=next(iter(changes))):
            merge_low_memory(base, adapter, output, "float16")

    def test_unpaired_lora_halves_are_refused(self, checkpoints):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints

        def drop_one_b(weights):
            del weights[next(key for key in weights if key.endswith(".lora_B.weight"))]

        _rewrite_adapter_weights(adapter, drop_one_b)
        with pytest.raises(LowMemoryMergeError, match="unpaired"):
            merge_low_memory(base, adapter, output, "float16")

    def test_shape_mismatch_is_refused_before_writing(self, checkpoints):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints

        def shrink_one_a(weights):
            key = next(key for key in weights if key.endswith(".lora_A.weight"))
            weights[key] = weights[key][:, :16].contiguous()

        _rewrite_adapter_weights(adapter, shrink_one_a)
        with pytest.raises(LowMemoryMergeError, match="does not match"):
            merge_low_memory(base, adapter, output, "float16")
        assert not output.exists()

    @pytest.mark.parametrize("key", ["lm_head.base_layer.weight", "embed_tokens.weight"])
    def test_saved_embedding_or_head_weights_get_a_clear_message(self, checkpoints, key):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints
        _rewrite_adapter_weights(
            adapter,
            lambda weights: weights.update(
                {f"base_model.model.{key}": torch.zeros(4, 4)}
            ),
        )
        with pytest.raises(LowMemoryMergeError, match="embeddings or lm_head"):
            merge_low_memory(base, adapter, output, "float16")

    def test_quantized_base_weight_under_an_adapter_is_refused(self, tmp_path):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base = _tiny_base(tmp_path / "base", shard_size="10MB")
        adapter = _tiny_adapter(base, tmp_path / "adapter")
        target = _first_adapter_target(base, adapter)
        weights = safetensors_torch.load_file(str(base / "model.safetensors"))
        weights[target] = torch.zeros(weights[target].numel(), dtype=torch.uint8)
        safetensors_torch.save_file(
            weights, str(base / "model.safetensors"), metadata={"format": "pt"}
        )

        with pytest.raises(LowMemoryMergeError, match="not a float type"):
            merge_low_memory(base, adapter, tmp_path / "merged", "float16")
        assert not (tmp_path / "merged").exists()


class TestHostileAdapterValues:
    def test_rank_and_shape_are_compared_before_any_matmul(self, monkeypatch):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merged_tensor

        def boom(*_args, **_kwargs):
            raise AssertionError("matmul ran before the shape check")

        monkeypatch.setattr(torch.Tensor, "__matmul__", boom)
        with pytest.raises(LowMemoryMergeError, match="does not match"):
            merged_tensor(
                torch.zeros(4, 4), torch.zeros(1, 20000), torch.zeros(20000, 1),
                scaling=1.0, fan_in_fan_out=False, dtype=torch.float32,
            )
        with pytest.raises(LowMemoryMergeError, match="rank mismatch"):
            merged_tensor(
                torch.zeros(4, 4), torch.zeros(2, 4), torch.zeros(4, 3),
                scaling=1.0, fan_in_fan_out=False, dtype=torch.float32,
            )

    def test_huge_delta_is_refused_before_writing(self, checkpoints):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints

        def make_huge(weights):
            key_a = next(key for key in weights if key.endswith(".lora_A.weight"))
            key_b = key_a.replace("lora_A", "lora_B")
            weights[key_a] = torch.zeros(1, 20000)
            weights[key_b] = torch.zeros(20000, 1)

        _rewrite_adapter_weights(adapter, make_huge)
        with pytest.raises(LowMemoryMergeError, match="does not match"):
            merge_low_memory(base, adapter, output, "float16")
        assert not output.exists()

    def test_rank_mismatch_between_a_and_b_is_refused(self, checkpoints):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints

        def drop_a_row(weights):
            key = next(key for key in weights if key.endswith(".lora_A.weight"))
            weights[key] = weights[key][:3].contiguous()

        _rewrite_adapter_weights(adapter, drop_a_row)
        with pytest.raises(LowMemoryMergeError, match="rank mismatch"):
            merge_low_memory(base, adapter, output, "float16")
        assert not output.exists()

    @pytest.mark.parametrize(
        "field, value",
        [
            ("r", 0),
            ("r", -4),
            ("r", "4"),
            ("r", 4.0),
            ("r", True),
            ("r", None),
            ("lora_alpha", float("nan")),
            ("lora_alpha", float("inf")),
            ("lora_alpha", "8"),
            ("lora_alpha", True),
            ("lora_alpha", None),
            ("use_rslora", "false"),
            ("use_rslora", 1),
            ("fan_in_fan_out", "false"),
            ("fan_in_fan_out", 0),
        ],
    )
    def test_adapter_config_values_are_validated(self, checkpoints, field, value):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints
        path = adapter / "adapter_config.json"
        config = json.loads(path.read_text())
        if value is None:
            config.pop(field)
        else:
            config[field] = value
        path.write_text(json.dumps(config))
        with pytest.raises(LowMemoryMergeError, match=field):
            merge_low_memory(base, adapter, output, "float16")
        assert not output.exists()


def _set_weight_map_name(base: Path, name: str) -> None:
    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    first = next(iter(index["weight_map"]))
    index["weight_map"][first] = name
    index_path.write_text(json.dumps(index))


class TestContainment:
    @pytest.mark.parametrize(
        "name",
        [
            "../escaped.safetensors",
            "..\\escaped.safetensors",
            "sub/escaped.safetensors",
            "/tmp/escaped.safetensors",
            "C:\\escaped.safetensors",
            "C:escaped.safetensors",
            "CON.safetensors",
            "escaped.bin",
            ".hidden.safetensors",
            "",
            7,
        ],
    )
    def test_index_shard_names_must_be_plain_file_names(self, checkpoints, name):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints
        _set_weight_map_name(base, name)
        with pytest.raises(LowMemoryMergeError, match="plain .safetensors file name"):
            merge_low_memory(base, adapter, output, "float16")
        assert not output.exists()
        assert not (base.parent / "escaped.safetensors").exists()
        assert not any(base.parent.glob("*escaped*"))

    def test_case_colliding_shard_names_are_refused(self, checkpoints):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints
        index_path = base / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        shard = next(iter(index["weight_map"].values()))
        index["weight_map"]["extra.weight"] = shard.upper().replace(".SAFETENSORS", ".safetensors")
        index_path.write_text(json.dumps(index))
        with pytest.raises(LowMemoryMergeError, match="differ only by case"):
            merge_low_memory(base, adapter, output, "float16")

    @pytest.mark.parametrize("where", ["adapter_inside_output", "output_inside_base"])
    def test_overlapping_directories_are_refused(self, tmp_path, where):
        from soup_cli.utils.lowmem_merge import merge_low_memory

        base = _tiny_base(tmp_path / "base")
        if where == "adapter_inside_output":
            output = tmp_path / "out"
            adapter = _tiny_adapter(base, output / "adapter")
        else:
            output = base / "merged"
            adapter = _tiny_adapter(base, tmp_path / "adapter")
        with pytest.raises(ValueError, match="overlap"):
            merge_low_memory(base, adapter, output, "float16")

    @pytest.mark.requires_symlink
    def test_symlinked_output_is_refused(self, checkpoints, tmp_path):
        from soup_cli.utils.lowmem_merge import merge_low_memory

        base, adapter, output = checkpoints
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        output.symlink_to(elsewhere, target_is_directory=True)
        with pytest.raises(ValueError, match="symbolic link"):
            merge_low_memory(base, adapter, output, "float16")
        assert list(elsewhere.iterdir()) == []

    def test_existing_output_is_not_overwritten(self, checkpoints):
        from soup_cli.utils.lowmem_merge import merge_low_memory

        base, adapter, output = checkpoints
        output.mkdir()
        (output / "keep.txt").write_text("mine")
        with pytest.raises(FileExistsError):
            merge_low_memory(base, adapter, output, "float16")
        assert (output / "keep.txt").read_text() == "mine"
        assert [path.name for path in output.iterdir()] == ["keep.txt"]

    def test_empty_existing_output_directory_is_used(self, checkpoints):
        from soup_cli.utils.lowmem_merge import merge_low_memory

        base, adapter, output = checkpoints
        output.mkdir()
        merge_low_memory(base, adapter, output, "float16")
        assert (output / "model.safetensors.index.json").is_file()

    @pytest.mark.requires_symlink
    def test_shard_symlinked_outside_the_model_is_refused(self, checkpoints, tmp_path):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints
        shard = next(base.glob("model-*.safetensors"))
        outside = tmp_path / "outside.safetensors"
        shard.rename(outside)
        shard.symlink_to(outside)
        with pytest.raises(LowMemoryMergeError, match="outside the base model directory"):
            merge_low_memory(base, adapter, output, "float16")
        assert not output.exists()

    @pytest.mark.requires_symlink
    def test_support_file_symlinked_outside_the_model_is_not_copied(self, checkpoints, tmp_path):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints
        secret = tmp_path / "secret.json"
        secret.write_text('{"token": "hunter2"}')
        (base / "tokenizer_config.json").symlink_to(secret)
        with pytest.raises(LowMemoryMergeError, match="outside the base model directory"):
            merge_low_memory(base, adapter, output, "float16")
        assert not output.exists()

    def test_python_files_are_copied_only_when_remote_code_is_trusted(self, tmp_path):
        from soup_cli.utils.lowmem_merge import merge_low_memory

        base = _tiny_base(tmp_path / "base")
        adapter = _tiny_adapter(base, tmp_path / "adapter")
        (base / "modeling_custom.py").write_text("print('hi')")

        merge_low_memory(base, adapter, tmp_path / "plain", "float16")
        merge_low_memory(base, adapter, tmp_path / "trusted", "float16", trust_remote_code=True)

        assert not (tmp_path / "plain" / "modeling_custom.py").exists()
        assert (tmp_path / "plain" / "config.json").is_file()
        assert (tmp_path / "trusted" / "modeling_custom.py").read_text() == "print('hi')"

    @pytest.mark.requires_symlink
    def test_trusted_python_symlink_outside_the_model_is_still_refused(self, checkpoints, tmp_path):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints
        outside = tmp_path / "outside.py"
        outside.write_text("print('outside')")
        (base / "modeling_custom.py").symlink_to(outside)
        with pytest.raises(LowMemoryMergeError, match="outside the base model directory"):
            merge_low_memory(base, adapter, output, "float16", trust_remote_code=True)
        assert not output.exists()

    @pytest.mark.requires_symlink
    def test_huggingface_cache_snapshot_links_are_followed(self, tmp_path):
        from soup_cli.utils.lowmem_merge import merge_low_memory

        original = _tiny_base(tmp_path / "original")
        adapter = _tiny_adapter(original, tmp_path / "adapter")
        repo = tmp_path / "models--org--tiny"
        blobs = repo / "blobs"
        snapshot = repo / "snapshots" / "abc123"
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        for position, source in enumerate(sorted(original.iterdir())):
            blob = blobs / f"{position:040x}"
            shutil.copyfile(source, blob)
            (snapshot / source.name).symlink_to(Path("..", "..", "blobs", blob.name))

        merge_low_memory(snapshot, adapter, tmp_path / "merged", "float32")

        expected = _regular_merge(original, adapter, torch.float32)
        actual = _read_all(tmp_path / "merged")
        assert all(torch.equal(actual[key], value) for key, value in expected.items())
        assert (tmp_path / "merged" / "config.json").is_file()


class TestAtomicOutput:
    def test_late_failure_leaves_no_output_and_no_staging(self, checkpoints, tmp_path):
        from soup_cli.utils.lowmem_merge import merge_low_memory

        base, adapter, output = checkpoints
        before = sorted(path.name for path in tmp_path.iterdir())

        def fail(staging):
            assert any(staging.glob("*.safetensors"))
            raise RuntimeError("late failure")

        with pytest.raises(RuntimeError, match="late failure"):
            merge_low_memory(base, adapter, output, "float16", finalize=fail)

        assert not output.exists()
        assert sorted(path.name for path in tmp_path.iterdir()) == before

    def test_failure_while_writing_a_shard_leaves_no_output(
        self, checkpoints, tmp_path, monkeypatch
    ):
        from soup_cli.utils import lowmem_merge

        base, adapter, output = checkpoints
        before = sorted(path.name for path in tmp_path.iterdir())
        real_write = lowmem_merge._write_shard
        calls = []

        def second_shard_fails(*args, **kwargs):
            calls.append(args[0].name)
            if len(calls) == 2:
                raise OSError("disk full")
            return real_write(*args, **kwargs)

        monkeypatch.setattr(lowmem_merge, "_write_shard", second_shard_fails)
        with pytest.raises(OSError, match="disk full"):
            lowmem_merge.merge_low_memory(base, adapter, output, "float16")

        assert len(calls) == 2
        assert not output.exists()
        assert sorted(path.name for path in tmp_path.iterdir()) == before

    def test_finalize_output_is_part_of_the_published_directory(self, checkpoints):
        from soup_cli.utils.lowmem_merge import merge_low_memory

        base, adapter, output = checkpoints
        merge_low_memory(
            base, adapter, output, "float16",
            finalize=lambda staging: (staging / "tokenizer.json").write_text("{}"),
        )
        assert (output / "tokenizer.json").is_file()
        assert not any(path.name.startswith(".merged.") for path in output.parent.iterdir())


class TestOrderAndDefenceInDepth:
    def test_non_2d_lora_tensors_are_refused(self, checkpoints):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints

        def flatten_one_a(weights):
            key = next(key for key in weights if key.endswith(".lora_A.weight"))
            weights[key] = weights[key].reshape(-1)

        _rewrite_adapter_weights(adapter, flatten_one_a)
        with pytest.raises(LowMemoryMergeError, match="2-D"):
            merge_low_memory(base, adapter, output, "float16")
        assert not output.exists()

    def test_a_bad_shape_is_found_before_any_shard_is_written(self, checkpoints, monkeypatch):
        from soup_cli.utils import lowmem_merge

        base, adapter, output = checkpoints

        def shrink_one_a(weights):
            key = next(key for key in weights if key.endswith(".lora_A.weight"))
            weights[key] = weights[key][:, :16].contiguous()

        _rewrite_adapter_weights(adapter, shrink_one_a)

        def no_write(*args, **kwargs):
            raise AssertionError("a shard was written before the shapes were checked")

        monkeypatch.setattr(lowmem_merge, "_write_shard", no_write)
        with pytest.raises(lowmem_merge.LowMemoryMergeError, match="does not match"):
            lowmem_merge.merge_low_memory(base, adapter, output, "float16")

    def test_load_adapter_itself_refuses_bad_config_values(self, checkpoints):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, load_adapter

        _, adapter, _ = checkpoints
        _rewrite_adapter_config(adapter, r=0)
        with pytest.raises(LowMemoryMergeError, match="'r'"):
            load_adapter(adapter)

    def test_write_shard_alone_refuses_a_quantized_tensor_under_an_adapter(self, tmp_path):
        from soup_cli.utils import lowmem_merge

        base = _tiny_base(tmp_path / "base", shard_size="10MB")
        adapter = _tiny_adapter(base, tmp_path / "adapter")
        target = _first_adapter_target(base, adapter)
        weights = safetensors_torch.load_file(str(base / "model.safetensors"))
        weights[target] = torch.zeros(weights[target].numel(), dtype=torch.uint8)
        safetensors_torch.save_file(
            weights, str(base / "model.safetensors"), metadata={"format": "pt"}
        )
        config, paired = lowmem_merge.load_adapter(adapter)
        with pytest.raises(lowmem_merge.LowMemoryMergeError, match="not a float type"):
            lowmem_merge._write_shard(
                base / "model.safetensors", tmp_path / "out.safetensors", "float16", paired, config
            )

    def test_write_shard_checks_the_bytes_it_writes(self, checkpoints, tmp_path, monkeypatch):
        from soup_cli.utils import lowmem_merge

        base, adapter, _ = checkpoints
        config, paired = lowmem_merge.load_adapter(adapter)
        monkeypatch.setattr(lowmem_merge, "merged_tensor", lambda *args, **kwargs: torch.zeros(3))
        weight_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
        shard = base / weight_map[_first_adapter_target(base, adapter)]
        with pytest.raises(lowmem_merge.LowMemoryMergeError, match="header declares"):
            lowmem_merge._write_shard(
                shard, tmp_path / "out.safetensors", "float16", paired, config
            )


class TestIndexContainment:
    @pytest.mark.requires_symlink
    def test_index_symlinked_outside_the_model_is_refused(self, checkpoints, tmp_path):
        from soup_cli.utils.lowmem_merge import LowMemoryMergeError, merge_low_memory

        base, adapter, output = checkpoints
        index = base / "model.safetensors.index.json"
        outside = tmp_path / "outside.index.json"
        index.rename(outside)
        index.symlink_to(outside)
        with pytest.raises(LowMemoryMergeError, match="outside the base model directory"):
            merge_low_memory(base, adapter, output, "float16")
        assert not output.exists()
