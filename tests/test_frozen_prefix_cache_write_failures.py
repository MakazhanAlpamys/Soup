"""CPU fault-injection tests for the frozen-prefix disk cache.

Only pytest-owned temporary directories are modified. These tests simulate
recoverable exceptions; they do not fill a disk, kill a process, or establish
power-loss durability or general concurrent-writer correctness.
"""
from __future__ import annotations

import errno
import hashlib
from pathlib import Path

import pytest
import torch
from safetensors import torch as st

from soup_cli.utils import frozen_prefix_cache as implementation


def make_case(tmp_path: Path, entry_exists: bool):
    directory = tmp_path / "isolated-frozen-prefix-cache"
    cache = implementation.FrozenPrefixCache(directory)
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.int64, device="cpu")
    common = {
        "model_revision": "write-failure-fixture-v1",
        "frozen_prefix_fingerprint": "fixture-weights-v1",
        "config_fingerprint": "fixture-config-v1",
        "cutoff": 2,
        "attention_mask": torch.ones_like(inputs),
        "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cpu"),
    }
    metadata = implementation.build_cache_metadata(input_ids=inputs, **common)
    other_metadata = implementation.build_cache_metadata(input_ids=inputs + 10, **common)
    old = torch.arange(12, dtype=torch.float32, device="cpu").reshape(1, 3, 4)
    new = old + 100
    unrelated = old + 200
    cache.save(unrelated, other_metadata)
    if entry_exists:
        cache.save(old, metadata)
    target = directory / f"{implementation.cache_key(metadata)}.safetensors"
    return cache, metadata, other_metadata, old, new, unrelated, target


def disk_state(directory: Path) -> dict[str, str]:
    """Fingerprint every entry, including hidden .tmp files."""
    state = {}
    for path in directory.iterdir():
        assert path.is_file() and not path.is_symlink()
        state[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return state


def assert_value(cache, metadata, expected) -> None:
    loaded = cache.load(metadata)
    if expected is None:
        assert loaded is None
    else:
        assert loaded is not None
        assert loaded.device.type == "cpu"
        assert loaded.dtype == expected.dtype
        assert loaded.shape == expected.shape
        assert torch.equal(loaded, expected)


@pytest.mark.parametrize("entry_exists", [False, True], ids=["new-key", "existing-key"])
@pytest.mark.parametrize(
    "failure_stage",
    ["create-temp", "serialize-before-write", "serialize-partial", "publish"],
)
def test_save_failure_preserves_disk_state_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_exists: bool,
    failure_stage: str,
) -> None:
    cache, metadata, other, old, new, unrelated, target = make_case(tmp_path, entry_exists)
    before = disk_state(cache.directory)
    expected_before = old if entry_exists else None
    original_new = new.clone()
    reached = []
    error_number = errno.ENOSPC if failure_stage == "create-temp" else errno.EIO
    injected = OSError(error_number, f"simulated-{failure_stage}")

    def failed_create(*args, **kwargs):
        assert Path(kwargs["dir"]) == cache.directory
        reached.append("create-temp")
        raise injected  # Simulated ENOSPC only; no real disk is filled.

    def failed_serialize(tensors, filename, *args, **kwargs):
        temporary = Path(filename)
        assert temporary.parent == cache.directory
        assert temporary != target and temporary.is_file()
        assert set(tensors) == {"hidden_states"}
        assert torch.equal(tensors["hidden_states"], new)
        reached.append(failure_stage)
        if failure_stage == "serialize-partial":
            temporary.write_bytes(b"deliberately incomplete test serialization")
        raise injected

    def failed_publish(src, dst):
        temporary = Path(src)
        assert Path(dst) == target
        assert temporary.parent == cache.directory
        assert temporary != target
        # A complete candidate has been serialized, but is not published yet.
        serialized = st.load_file(str(temporary), device="cpu")
        assert torch.equal(serialized["hidden_states"], new)
        assert_value(cache, metadata, expected_before)
        reached.append("publish")
        raise injected

    # Scope replacement to the one attempted save. Real operations are restored
    # before the retry, and no production file or function definition is edited.
    with monkeypatch.context() as patch:
        if failure_stage == "create-temp":
            patch.setattr(implementation.tempfile, "mkstemp", failed_create)
        elif failure_stage.startswith("serialize-"):
            patch.setattr(st, "save_file", failed_serialize)
        else:
            patch.setattr(implementation.os, "replace", failed_publish)
        with pytest.raises(OSError) as caught:
            cache.save(new, metadata)

    assert caught.value is injected  # Neither swallowed nor masked by cleanup.
    assert reached == [failure_stage]
    assert torch.equal(new, original_new)
    assert disk_state(cache.directory) == before
    assert list(cache.directory.glob(".frozen-prefix-*.tmp")) == []

    # Check persistence through a separate reader, not only the original object.
    reader = implementation.FrozenPrefixCache(cache.directory)
    assert_value(reader, metadata, expected_before)
    assert_value(reader, other, unrelated)
    if not entry_exists:
        assert not target.exists()

    # Retry the same key using the real serializer and the real os.replace.
    assert reader.save(new, metadata) == target
    fresh_reader = implementation.FrozenPrefixCache(cache.directory)
    assert_value(fresh_reader, metadata, new)
    assert_value(fresh_reader, other, unrelated)
    assert list(cache.directory.glob(".frozen-prefix-*.tmp")) == []
    after = disk_state(cache.directory)
    assert set(after) == set(before) | {target.name}
    assert {k: v for k, v in after.items() if k != target.name} == {
        k: v for k, v in before.items() if k != target.name
    }
    if entry_exists:
        assert after[target.name] != before[target.name]


@pytest.mark.parametrize("entry_exists", [False, True], ids=["new-key", "existing-key"])
def test_reader_sees_old_or_absent_entry_until_real_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_exists: bool,
) -> None:
    cache, metadata, other, old, new, unrelated, target = make_case(tmp_path, entry_exists)
    before = disk_state(cache.directory)
    real_replace = implementation.os.replace
    calls = []

    def observe_publication(src, dst):
        temporary = Path(src)
        assert Path(dst) == target
        assert temporary.parent == cache.directory and temporary != target
        assert temporary.name.startswith(".frozen-prefix-")
        assert temporary.suffix == ".tmp"
        tensors = st.load_file(str(temporary), device="cpu")
        assert torch.equal(tensors["hidden_states"], new)
        reader = implementation.FrozenPrefixCache(cache.directory)
        assert_value(reader, metadata, old if entry_exists else None)
        assert_value(reader, other, unrelated)
        current = disk_state(cache.directory)
        assert {k: v for k, v in current.items() if k != temporary.name} == before
        calls.append("before")
        real_replace(src, dst)  # Actual replacement, not a simulated success.
        calls.append("after")
        assert not temporary.exists()
        assert_value(reader, metadata, new)

    with monkeypatch.context() as patch:
        patch.setattr(implementation.os, "replace", observe_publication)
        assert cache.save(new, metadata) == target

    assert calls == ["before", "after"]
    assert_value(implementation.FrozenPrefixCache(cache.directory), metadata, new)
    assert_value(cache, other, unrelated)
    assert list(cache.directory.glob(".frozen-prefix-*.tmp")) == []
