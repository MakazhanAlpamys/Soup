"""Keys, fingerprints and on-disk storage of the frozen-prefix activation cache."""

import os

import pytest
import torch
from transformers import MistralConfig, MistralModel

from soup_cli.utils.frozen_prefix_cache import (
    CACHE_DIRNAME,
    FrozenPrefixCache,
    build_cache_metadata,
    cache_key,
    fingerprint_frozen_prefix,
    fingerprint_prefix_configuration,
    resolve_cache_directory,
)


def tiny_config(**overrides):
    values = dict(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    values.update(overrides)
    return MistralConfig(**values)


COMMON = {
    "model_revision": "tiny-mistral-v1",
    "frozen_prefix_fingerprint": "weights-v1",
    "config_fingerprint": "config-v1",
    "cutoff": 2,
}


def test_cache_save_load(tmp_path):
    cache = FrozenPrefixCache(tmp_path)
    activation = torch.randn(1, 4, 64)
    metadata = {"model": "tiny-mistral", "revision": "test-v1", "cutoff": 2}

    cache.save(activation, metadata)
    loaded = cache.load(metadata)

    assert loaded is not None
    assert torch.equal(activation, loaded)


def test_cache_persists_across_instances(tmp_path):
    metadata = build_cache_metadata(
        **COMMON,
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        position_ids=torch.tensor([[0, 1, 2]]),
    )
    activation = torch.randn(1, 3, 8)

    FrozenPrefixCache(tmp_path).save(activation, metadata)
    restored = FrozenPrefixCache(tmp_path).load(metadata)

    assert restored is not None
    assert torch.equal(restored, activation)


def test_cache_key_is_deterministic():
    assert cache_key({"model": "mistral", "cutoff": 2}) == cache_key(
        {"cutoff": 2, "model": "mistral"}
    )


def test_cache_miss_on_changed_metadata(tmp_path):
    cache = FrozenPrefixCache(tmp_path)
    original = {"model": "tiny-mistral", "revision": "test-v1", "cutoff": 2}
    cache.save(torch.randn(1, 4, 64), original)

    assert cache.load({**original, "revision": "test-v2"}) is None


def test_every_input_that_changes_the_activation_changes_the_key(tmp_path):
    cache = FrozenPrefixCache(tmp_path)
    inputs = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[1, 1, 1, 1]])
    positions = torch.tensor([[0, 1, 2, 3]])
    original = build_cache_metadata(
        **COMMON, input_ids=inputs, attention_mask=mask, position_ids=positions
    )
    cache.save(torch.randn(1, 4, 8), original)
    assert cache.load(original) is not None

    variants = {
        "input ids": build_cache_metadata(
            **COMMON,
            input_ids=torch.tensor([[1, 2, 3, 5]]),
            attention_mask=mask,
            position_ids=positions,
        ),
        "attention mask": build_cache_metadata(
            **COMMON,
            input_ids=inputs,
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
            position_ids=positions,
        ),
        "position ids": build_cache_metadata(
            **COMMON,
            input_ids=inputs,
            attention_mask=mask,
            position_ids=torch.tensor([[4, 5, 6, 7]]),
        ),
        "frozen weights": build_cache_metadata(
            **{**COMMON, "frozen_prefix_fingerprint": "weights-v2"},
            input_ids=inputs,
            attention_mask=mask,
            position_ids=positions,
        ),
        "configuration": build_cache_metadata(
            **{**COMMON, "config_fingerprint": "config-v2"},
            input_ids=inputs,
            attention_mask=mask,
            position_ids=positions,
        ),
        "cutoff": build_cache_metadata(
            **{**COMMON, "cutoff": 3},
            input_ids=inputs,
            attention_mask=mask,
            position_ids=positions,
        ),
    }
    for name, metadata in variants.items():
        assert cache_key(metadata) != cache_key(original), name
        assert cache.load(metadata) is None, name


def test_metadata_rejects_a_non_tensor_input():
    with pytest.raises(TypeError, match="Expected a torch.Tensor"):
        build_cache_metadata(**COMMON, input_ids=[1, 2, 3])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"model_revision": ""}, "Model identity and fingerprint are required"),
        ({"frozen_prefix_fingerprint": ""}, "Model identity and fingerprint are required"),
        ({"config_fingerprint": ""}, "Configuration fingerprint is required"),
        ({"cutoff": 0}, "cutoff must be positive"),
    ],
)
def test_metadata_refuses_missing_identity(overrides, message):
    with pytest.raises(ValueError, match=message):
        build_cache_metadata(**{**COMMON, **overrides}, input_ids=torch.tensor([[1, 2]]))


def test_atomic_save_leaves_no_temporary_file(tmp_path):
    cache = FrozenPrefixCache(tmp_path)
    metadata = {"model": "tiny-mistral", "revision": "v1", "cutoff": 2}
    activation = torch.randn(1, 4, 64)

    path = cache.save(activation, metadata)

    assert path.exists()
    assert torch.equal(cache.load(metadata), activation)
    assert list(tmp_path.glob(".frozen-prefix-*.tmp")) == []


def test_corrupted_entry_is_refused_not_read_as_a_hit(tmp_path):
    cache = FrozenPrefixCache(tmp_path)
    metadata = {"model": "tiny-mistral", "revision": "v1", "cutoff": 2}
    path = cache.save(torch.randn(1, 4, 64), metadata)

    path.write_bytes(b"corrupted-cache")

    with pytest.raises(Exception):  # noqa: B017 - safetensors names its own error type
        cache.load(metadata)


def test_prefix_fingerprint_tracks_frozen_weights_only():
    model = MistralModel(tiny_config())
    original = fingerprint_frozen_prefix(model, cutoff=2)

    assert fingerprint_frozen_prefix(model, cutoff=2) == original

    with torch.no_grad():
        next(model.layers[3].parameters()).add_(0.01)
    assert fingerprint_frozen_prefix(model, cutoff=2) == original, "an upper layer is not prefix"

    with torch.no_grad():
        next(model.layers[0].parameters()).add_(0.01)
    assert fingerprint_frozen_prefix(model, cutoff=2) != original


def test_prefix_fingerprint_tracks_embeddings():
    model = MistralModel(tiny_config())
    original = fingerprint_frozen_prefix(model, cutoff=2)

    with torch.no_grad():
        model.embed_tokens.weight[0, 0].add_(0.01)

    assert fingerprint_frozen_prefix(model, cutoff=2) != original


def test_prefix_fingerprint_tracks_the_cutoff():
    model = MistralModel(tiny_config())

    assert fingerprint_frozen_prefix(model, cutoff=1) != fingerprint_frozen_prefix(
        model, cutoff=2
    )


@pytest.mark.parametrize("cutoff", [0, 5, -1])
def test_fingerprints_refuse_an_out_of_range_cutoff(cutoff):
    model = MistralModel(tiny_config())

    with pytest.raises(ValueError, match="Invalid frozen-prefix cutoff"):
        fingerprint_frozen_prefix(model, cutoff)
    with pytest.raises(ValueError, match="Invalid frozen-prefix cutoff"):
        fingerprint_prefix_configuration(model, cutoff)


@pytest.mark.parametrize("cutoff", [True, 2.0, "2"])
def test_fingerprints_refuse_a_non_integer_cutoff(cutoff):
    model = MistralModel(tiny_config())

    with pytest.raises(ValueError, match="cutoff must be an integer"):
        fingerprint_frozen_prefix(model, cutoff)


def test_rope_scaling_configuration_changes_the_fingerprint():
    model = MistralModel(tiny_config()).eval().requires_grad_(False)
    old = fingerprint_prefix_configuration(model, 2)

    model.config.rope_scaling = {"rope_type": "linear", "factor": 2.0}

    assert fingerprint_prefix_configuration(model, 2) != old


def test_resolve_cache_directory_lives_inside_the_output_directory(tmp_path):
    directory = resolve_cache_directory(tmp_path)

    assert directory.name == CACHE_DIRNAME
    assert os.path.commonpath([os.path.realpath(tmp_path), str(directory)]) == (
        os.path.realpath(tmp_path)
    )


def test_resolve_cache_directory_refuses_a_link_that_leaves_the_output_directory(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    link = output / CACHE_DIRNAME
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")

    with pytest.raises(ValueError, match="must stay inside the training output directory"):
        resolve_cache_directory(output)
