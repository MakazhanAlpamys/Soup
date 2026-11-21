"""Configuration regressions for streaming and sharding options that did nothing."""

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string


def _load_data(train, **settings):
    return load_config_from_string(
        yaml.safe_dump({"base": "org/model", "data": {"train": train, **settings}})
    ).data


@pytest.mark.parametrize(
    ("train", "settings"),
    [
        ("./train.jsonl", {}),
        ("org/dataset", {"streaming": True}),
        ("s3://bucket/train.jsonl", {"streaming": True}),
        (["a.jsonl", "b.jsonl"], {"interleave": "concat", "streaming": True}),
    ],
)
@pytest.mark.parametrize("shards", [4, 0, "all"])
def test_shard_count_is_refused_at_config_load(train, settings, shards):
    with pytest.raises(ValueError) as error:
        _load_data(train, shards=shards, **settings)
    assert (
        "data.shards is refused as of v0.78.0. A shard count alone does not select "
        "which shard to keep. Remove data.shards."
    ) in str(error.value)


@pytest.mark.parametrize("train", ["./train.jsonl", "data/training.PARQUET"])
@pytest.mark.parametrize(
    ("settings", "keys"),
    [
        ({"streaming": True}, "data.streaming"),
        ({"buffer_size": 16}, "data.buffer_size"),
        ({"streaming": True, "buffer_size": 16}, "data.streaming and data.buffer_size"),
    ],
)
def test_single_local_file_refuses_ignored_streaming_options(train, settings, keys):
    with pytest.raises(ValueError) as error:
        _load_data(train, **settings)
    assert (
        f"{keys} cannot be used with a single local file in data.train. "
        "Streaming supports a Hugging Face Hub dataset name, a remote URI, or a "
        "list of local files or remote URIs with data.interleave."
    ) in str(error.value)


@pytest.mark.parametrize(
    ("train", "settings"),
    [
        ("org/dataset.1", {}),
        ("s3://bucket/train.jsonl", {}),
        (["a.jsonl", "b.jsonl"], {"interleave": "concat"}),
        (["s3://bucket/a.jsonl", "s3://bucket/b.jsonl"], {"interleave": "concat"}),
    ],
)
def test_supported_sources_still_accept_streaming_with_a_buffer(train, settings):
    data = _load_data(train, streaming=True, buffer_size=16, **settings)
    assert data.streaming is True
    assert data.buffer_size == 16


def test_single_local_file_can_still_be_read_eagerly():
    data = _load_data("./train.jsonl", streaming=False, buffer_size=None, shards=None)
    assert data.train == "./train.jsonl"
    assert data.streaming is False
    assert data.buffer_size is None
    assert data.shards is None
