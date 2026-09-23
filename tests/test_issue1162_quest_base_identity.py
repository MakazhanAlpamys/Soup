"""Issue #1162: local QuEST bases use portable, content-bound identities."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.utils.quest import (
    EXPECTED_MODULES,
    build_metadata,
    load_metadata,
    resolve_base_model_identity,
    validate_resume_metadata,
    write_metadata,
)


def _base(directory: Path, *, sharded: bool = False) -> Path:
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    if sharded:
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 2},
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors",
                    },
                }
            ),
            encoding="utf-8",
        )
        (directory / "model-00001-of-00002.safetensors").write_bytes(b"first")
        (directory / "model-00002-of-00002.safetensors").write_bytes(b"second")
    else:
        (directory / "model.safetensors").write_bytes(b"safe-weights")
    return directory


def _metadata(base_model: str, *, format_version: int = 2) -> dict:
    return build_metadata(
        activation_scales={name: 3.0 for name in EXPECTED_MODULES},
        base_model=base_model,
        calibration_sha256="ab" * 32,
        format_version=format_version,
    )


def test_hub_id_stays_byte_identical():
    assert resolve_base_model_identity("ahxt/LiteLlama-460M-1T") == "ahxt/LiteLlama-460M-1T"


def test_selected_safetensors_matches_real_transformers_loader(tmp_path):
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    base = tmp_path / "base"
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=256,
        )
    )
    model.save_pretrained(base, safe_serialization=True)
    selected = resolve_base_model_identity(str(base))
    (base / "pytorch_model.bin").write_bytes(b"invalid and deliberately unused")
    assert isinstance(AutoModelForCausalLM.from_pretrained(base), LlamaForCausalLM)
    assert resolve_base_model_identity(str(base)) == selected


def test_sharded_safetensors_matches_real_transformers_loader(tmp_path):
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    base = tmp_path / "sharded"
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=256,
        )
    )
    model.save_pretrained(base, safe_serialization=True, max_shard_size="250KB")
    assert (base / "model.safetensors.index.json").is_file()
    first = resolve_base_model_identity(str(base))
    assert isinstance(AutoModelForCausalLM.from_pretrained(base), LlamaForCausalLM)
    (base / "unselected.safetensors").write_bytes(b"not in the index")
    assert resolve_base_model_identity(str(base)) == first


def test_local_identity_survives_relocation_and_ignores_unread_files(tmp_path):
    original = _base(tmp_path / "original")
    first = resolve_base_model_identity(str(original))
    assert first.startswith("local-sha256:")
    relocated = tmp_path / "elsewhere" / "model"
    relocated.parent.mkdir()
    shutil.copytree(original, relocated)
    (relocated / "README.md").write_text("not loaded", encoding="utf-8")
    (relocated / "pytorch_model.bin").write_bytes(b"not selected when safetensors exists")
    assert resolve_base_model_identity(str(relocated)) == first

    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, _metadata(first))
    validate_resume_metadata(checkpoint, _metadata(resolve_base_model_identity(str(relocated))))
    assert str(original) not in (checkpoint / "quest_mixed_precision.json").read_text()


def test_setup_writes_only_the_local_digest_into_both_artifact_declarations(tmp_path, monkeypatch):
    from transformers import LlamaConfig

    import soup_cli.utils.quest as quest
    from soup_cli.trainer import stream_setup
    from soup_cli.trainer.sft import SFTTrainerWrapper

    # Keep the fixture's source outside the artifact directory, as in a real run.
    source = tmp_path / "private"
    source.mkdir()
    base = _base(source / "base")
    wrapper = object.__new__(SFTTrainerWrapper)
    wrapper.config = SimpleNamespace(base=str(base))
    wrapper.model = SimpleNamespace(config=LlamaConfig())
    wrapper.deepspeed_config = None
    wrapper.fsdp_config = None
    wrapper._quest_base_identity_before = resolve_base_model_identity(str(base))
    monkeypatch.setattr(stream_setup, "_distributed_launch", lambda: False)
    monkeypatch.setattr(quest, "validate_cuda_hardware", lambda: ("RTX 3090", (8, 6)))
    monkeypatch.setattr(quest, "calibration_rows_sha256", lambda rows: "ab" * 32)
    monkeypatch.setattr(
        quest,
        "calibrate_activation_scales",
        lambda model, rows: {name: 3.0 for name in EXPECTED_MODULES},
    )
    monkeypatch.setattr(
        quest,
        "install_mixed_quest",
        lambda model, *, activation_scales, base_model, calibration_sha256: build_metadata(
            activation_scales=activation_scales,
            base_model=base_model,
            calibration_sha256=calibration_sha256,
        ),
    )

    wrapper._setup_quest([{"input_ids": [1]}] * 32)
    assert wrapper._quest_metadata["base_model"].startswith("local-sha256:")
    assert wrapper.model.config.soup_quest == wrapper._quest_metadata
    assert str(base) not in json.dumps(wrapper.model.config.soup_quest)
    artifact = tmp_path / "artifact"
    wrapper.model.config.save_pretrained(artifact)
    write_metadata(artifact, wrapper._quest_metadata)
    assert str(base) not in (artifact / "config.json").read_text(encoding="utf-8")
    assert str(base) not in (artifact / "quest_mixed_precision.json").read_text(encoding="utf-8")

    (base / "model.safetensors").write_bytes(b"changed-after-model-load")
    with pytest.raises(ValueError, match="changed while the model was loading"):
        wrapper._setup_quest([{"input_ids": [1]}] * 32)


def test_content_change_at_the_same_path_is_refused(tmp_path):
    base = _base(tmp_path / "base")
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, _metadata(resolve_base_model_identity(str(base))))
    (base / "model.safetensors").write_bytes(b"changed-weights")
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_metadata(checkpoint, _metadata(resolve_base_model_identity(str(base))))


def test_config_change_at_the_same_path_is_refused(tmp_path):
    base = _base(tmp_path / "base")
    first = resolve_base_model_identity(str(base))
    (base / "config.json").write_text('{"model_type":"llama","hidden_size":256}')
    assert resolve_base_model_identity(str(base)) != first


def test_generation_config_change_at_the_same_path_is_refused(tmp_path):
    base = _base(tmp_path / "base")
    (base / "generation_config.json").write_text('{"max_length":32}', encoding="utf-8")
    first = resolve_base_model_identity(str(base))
    (base / "generation_config.json").write_text('{"max_length":64}', encoding="utf-8")
    assert resolve_base_model_identity(str(base)) != first


def test_tokenizer_change_at_the_same_path_is_refused(tmp_path):
    base = _base(tmp_path / "base")
    (base / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    first = resolve_base_model_identity(str(base))
    (base / "tokenizer.json").write_text('{"version":"1.1"}', encoding="utf-8")
    assert resolve_base_model_identity(str(base)) != first


def test_custom_tokenizer_with_untracked_code_is_refused(tmp_path):
    base = _base(tmp_path / "base")
    (base / "tokenizer_config.json").write_text(
        '{"auto_map":{"AutoTokenizer":["custom.Tokenizer",null]}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="standard tokenizer"):
        resolve_base_model_identity(str(base))


def test_sharded_identity_uses_index_and_only_referenced_shards(tmp_path):
    base = _base(tmp_path / "base", sharded=True)
    first = resolve_base_model_identity(str(base))
    (base / "unused.safetensors").write_bytes(b"ignored")
    assert resolve_base_model_identity(str(base)) == first
    (base / "model-00002-of-00002.safetensors").write_bytes(b"changed")
    assert resolve_base_model_identity(str(base)) != first
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"a": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shard"):
        resolve_base_model_identity(str(base))


@pytest.mark.parametrize(
    "shard_name",
    ["../outside.safetensors", "/outside.safetensors", "C:/outside.safetensors", "nested\\file"],
)
def test_shard_index_rejects_paths_outside_the_selected_base(tmp_path, shard_name):
    base = _base(tmp_path / "base", sharded=True)
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shard_name}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="shard name"):
        resolve_base_model_identity(str(base))


def test_bin_fallback_when_no_safetensors_exist(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (base / "pytorch_model.bin").write_bytes(b"torch-weights")
    first = resolve_base_model_identity(str(base))
    (base / "pytorch_model.bin").write_bytes(b"different")
    assert resolve_base_model_identity(str(base)) != first


def test_v1_sidecar_remains_readable_with_its_original_path_contract(tmp_path):
    original = _base(tmp_path / "original")
    historical = _metadata(str(original), format_version=1)
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, historical)
    assert load_metadata(checkpoint) == historical

    current = _metadata(resolve_base_model_identity(str(original)))
    validate_resume_metadata(checkpoint, current, legacy_base_model=str(original))
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_metadata(checkpoint, current, legacy_base_model=str(tmp_path / "moved"))
    with pytest.raises(ValueError, match="original base model reference"):
        validate_resume_metadata(checkpoint, current)


def test_v1_and_v2_do_not_silently_compare_as_the_same_format(tmp_path):
    base = _base(tmp_path / "base")
    historical = _metadata(str(base), format_version=1)
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, historical)
    current = _metadata(resolve_base_model_identity(str(base)))
    current["activation_scales"][EXPECTED_MODULES[0]] = 4.0
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_metadata(checkpoint, current, legacy_base_model=str(base))


def test_v1_reference_cannot_override_a_different_current_base(tmp_path):
    original = _base(tmp_path / "original")
    changed = _base(tmp_path / "changed")
    (changed / "model.safetensors").write_bytes(b"different-weights")
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, _metadata(str(original), format_version=1))
    current = _metadata(resolve_base_model_identity(str(changed)))
    with pytest.raises(ValueError, match="does not match current identity"):
        validate_resume_metadata(checkpoint, current, legacy_base_model=str(original))


@pytest.mark.parametrize("absolute", ["/home/alice/models/base", "C:\\Users\\alice\\models\\base"])
def test_v2_rejects_raw_absolute_paths(absolute):
    with pytest.raises(ValueError, match="base_model"):
        _metadata(absolute)
