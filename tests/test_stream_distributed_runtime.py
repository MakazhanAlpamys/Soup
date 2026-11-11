"""Data-parallel layer streaming, the rest of the BETA slice (#2585): C5, C6, C7, C8.

``test_stream_distributed_trainer.py`` shows that two ranks equal one rank on the union
batch; this file covers what has to be true around that, for a real launch:

* **C5, init consistency** (``utils/distributed_init_check.py``, T2): every rank fingerprints
  its trainable layout and frozen base, the ranks all-gather it and refuse a difference BY
  NAME (which item, which rank); then the adapters are broadcast from rank 0, so ranks
  launched with different ``seed`` values end bitwise equal. In process: the fingerprint, the
  comparison and the control flow of the collective on a scripted fake group. With two real
  gloo ranks: different seeds (proceeds, ends equal), a different LoRA ``r``, different
  target modules, and a different base on one rank (all three refuse at startup).
* **C6, rank gating** (T6): the Soup callback draws the live panel, the watchdog banner and
  the tracker rows on rank 0 only (``TrainerState.is_world_process_zero``) and still stops
  every rank together; ``soup train`` gives every other rank a tracker that writes nothing;
  the tokenizer is saved where the model is. With two real ranks and one output directory
  each: which files each rank wrote.
* **C7, per-step metrics** (``utils/stream_dp_runtime.py``, T8): one JSON line per optimizer
  step per rank, with the sync bytes of that step.
* **C8, group timeout** (T9): shortened once the setup is done through
  ``torch.distributed.set_timeout`` (torch 2.14), skipped cleanly where torch has no API.
* **T5, a dead peer**: a rank that exits (and one that goes silent) ends the other rank's run
  with a non-zero exit inside the timeout plus a slack, with the failure in its output.
* **T7, sharding**: which rows each rank sees under the production sampler and accelerate's
  batch sharding, including the rows that are repeated to complete the last step.

The two-process tests run ``_stream_distributed_runtime_worker.py`` (the production setup and
``SFTTrainerWrapper.train()``) on CPU, gloo over 127.0.0.1, one output directory, shard cache
and tracker database per rank, and skip without torch, transformers, peft, trl, datasets,
accelerate or gloo (and on macOS, where two-process rendezvous can time out on loaded
runners). Text from the processes is compared through an ANSI-strip + whitespace-collapse
helper.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.adapter_grad_sync import SyncStats, collect_trainable
from soup_cli.utils.distributed_init_check import (
    BASE_ITEMS,
    ITEM_FIRST_SHARD,
    ITEM_LAST_SHARD,
    ITEM_LAYOUT,
    ITEM_QUANT,
    ITEM_REVISION,
    ITEM_SHARD_INDEX,
    DistributedInitMismatchError,
    InitReport,
    describe_local,
    find_mismatches,
    first_layout_difference,
    format_mismatches,
    frozen_base_items,
    hash_file,
    layout_digest,
    quantisation_label,
    shard_index_digest,
    trainable_layout,
    verify_and_sync_init,
)
from soup_cli.utils.stream_dp_runtime import (
    LEGACY_TIMEOUT_API,
    SET_TIMEOUT_API,
    GroupTimeout,
    apply_group_timeout,
    build_stream_dp_metrics_callback,
    stream_dp_metrics_path,
)
from tests.test_stream_distributed_trainer import (
    _ROWS,
    _STEPS,
    _free_port,
    _plain,
    _row,
    _tail,
    _worker_env,
)

_WORKER = Path(__file__).resolve().with_name("_stream_distributed_runtime_worker.py")
_RUN_DEADLINE = 420
_GROUP_TIMEOUT = 30
_SILENT_TIMEOUT = 15
_SLACK = 90
_TAGS = ("dp0", "dp1")
_HEX = "0123456789abcdef"

#: What a survivor's output says when its peer is gone: a closed or reset socket, or the
#: group timeout. Torch words it differently per platform and backend.
_PEER_GONE = re.compile(
    r"connection (closed|reset)|forcibly closed|timed out|timeout|distbackenderror|gloo",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# C5: the fingerprint, in process
# --------------------------------------------------------------------------


def _index(**overrides):
    from soup_cli.utils.layer_shard import ShardIndex

    fields = {
        "n_layers": 2,
        "layer_keys": ("self_attn.q_proj.weight",),
        "extra_keys": ("model.embed_tokens.weight",),
        "dtype": "float32",
        "total_params": 1000,
        "arch": "llama",
        "soup_version": "0.77.2",
        "source_files": (("model.safetensors", 4096, 111),),
        "source_fingerprint": "abc",
    }
    fields.update(overrides)
    return ShardIndex(**fields)


def _shards(tmp_path, *, first=b"first-layer-bytes", last=b"last-layer-bytes"):
    root = tmp_path / "shards"
    root.mkdir(parents=True, exist_ok=True)
    (root / "layer_000.safetensors").write_bytes(first)
    (root / "layer_001.safetensors").write_bytes(last)
    return str(root)


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


class TestShardIndexDigest:
    def test_it_ignores_what_differs_between_two_hosts_that_hold_the_same_base(self):
        baseline = shard_index_digest(_index())
        elsewhere = _index(
            source_files=(("model.safetensors", 4096, 999999),),  # another mtime
            source_fingerprint="other",
            soup_version="0.99.0",
            stripe_roots=("/mnt/other-disk",),
            layer_roots=(0, 1),
        )
        assert shard_index_digest(elsewhere) == baseline

    @pytest.mark.parametrize(
        "override",
        [
            {"dtype": "bfloat16"},
            {"arch": "qwen2"},
            {"n_layers": 3},
            {"layer_keys": ("self_attn.k_proj.weight",)},
            {"extra_keys": ()},
            {"large_keys": ("lm_head.weight",)},
            {"total_params": 1001},
            {"source_files": (("model.safetensors", 8192, 111),)},
            {"source_files": (("other.safetensors", 4096, 111),)},
            {"expert_layout": "per_expert"},
            {"expert_count": 8},
            {"external_mode": "none"},
            {"format_version": 4},
        ],
        ids=lambda override: next(iter(override)),
    )
    def test_it_changes_with_what_the_frozen_base_is(self, override):
        assert shard_index_digest(_index(**override)) != shard_index_digest(_index())


class TestQuantisationLabel:
    def test_an_unquantised_base_is_none(self):
        assert quantisation_label(_index()) == "none"

    def test_nf4_names_the_double_quant_flag_and_the_device(self):
        label = quantisation_label(_index(quant="nf4", double_quant=True, quant_device="cuda"))
        assert label.startswith("nf4 (double_quant=True, quantised on cuda, specs ")

    def test_each_part_of_it_changes_the_label(self):
        base = {"quant": "nf4", "double_quant": True, "quant_device": "cuda"}
        labels = {
            quantisation_label(_index(**base)),
            quantisation_label(_index(**{**base, "double_quant": False})),
            quantisation_label(_index(**{**base, "quant_device": "cpu"})),
            quantisation_label(_index(**base, quant_specs={"q": "spec-a"})),
            quantisation_label(_index(**base, quant_specs={"q": "spec-b"})),
        }
        assert len(labels) == 5


class TestFrozenBaseItems:
    def test_the_items_are_the_documented_five(self, tmp_path):
        items = frozen_base_items(_shards(tmp_path), _index())
        assert tuple(items) == BASE_ITEMS
        assert BASE_ITEMS == (
            ITEM_SHARD_INDEX,
            ITEM_QUANT,
            ITEM_FIRST_SHARD,
            ITEM_LAST_SHARD,
            ITEM_REVISION,
        )

    def test_the_two_shard_digests_are_the_sha256_of_the_first_and_last_layer_file(self, tmp_path):
        items = frozen_base_items(_shards(tmp_path), _index())
        assert items[ITEM_FIRST_SHARD] == _sha(b"first-layer-bytes")
        assert items[ITEM_LAST_SHARD] == _sha(b"last-layer-bytes")

    def test_a_changed_last_layer_changes_only_the_last_shard_item(self, tmp_path):
        left = frozen_base_items(_shards(tmp_path / "a"), _index())
        right = frozen_base_items(_shards(tmp_path / "b", last=b"last-layer-byteZ"), _index())
        differing = [item for item in BASE_ITEMS if left[item] != right[item]]
        assert differing == [ITEM_LAST_SHARD]

    def test_a_changed_first_layer_changes_only_the_first_shard_item(self, tmp_path):
        left = frozen_base_items(_shards(tmp_path / "a"), _index())
        right = frozen_base_items(_shards(tmp_path / "b", first=b"first-layer-byteZ"), _index())
        differing = [item for item in BASE_ITEMS if left[item] != right[item]]
        assert differing == [ITEM_FIRST_SHARD]

    def test_the_revision_is_carried_and_an_unbound_base_is_unknown(self, tmp_path):
        shards = _shards(tmp_path)
        assert frozen_base_items(shards, _index())[ITEM_REVISION] == "unknown"
        assert frozen_base_items(shards, _index(), revision="a" * 40)[ITEM_REVISION] == "a" * 40

    def test_a_one_layer_base_has_the_same_first_and_last_digest(self, tmp_path):
        items = frozen_base_items(_shards(tmp_path), _index(n_layers=1))
        assert items[ITEM_FIRST_SHARD] == items[ITEM_LAST_SHARD] == _sha(b"first-layer-bytes")

    def test_an_index_without_layers_is_refused_by_name(self, tmp_path):
        with pytest.raises(ValueError, match=r"lists no decoder layer"):
            frozen_base_items(_shards(tmp_path), _index(n_layers=0))

    def test_a_missing_layer_file_raises_instead_of_hashing_nothing(self, tmp_path):
        shards = _shards(tmp_path)
        (Path(shards) / "layer_001.safetensors").unlink()
        with pytest.raises(OSError):
            frozen_base_items(shards, _index())

    def test_a_file_larger_than_the_read_chunk_hashes_the_same(self, tmp_path):
        target = tmp_path / "blob"
        target.write_bytes(b"0123456789" * 10)
        assert (
            hash_file(str(target), chunk_bytes=3)
            == hash_file(str(target))
            == _sha(b"0123456789" * 10)
        )


# --------------------------------------------------------------------------
# C5: the comparison, in process
# --------------------------------------------------------------------------


def _adapters(rank: int = 4, seed: int = 0, *, extra: bool = False):
    """A module with two trainable tensors (and one frozen, which the check must ignore)."""
    torch = pytest.importorskip("torch")

    class _Adapters(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(seed)
            self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)
            self.lora_A = torch.nn.Parameter(torch.randn(rank, 6, generator=generator))
            self.lora_B = torch.nn.Parameter(torch.randn(6, rank, generator=generator))
            if extra:
                self.lora_C = torch.nn.Parameter(torch.randn(2, generator=generator))

    return _Adapters()


_BASE = {
    ITEM_SHARD_INDEX: "i" * 64,
    ITEM_QUANT: "none",
    ITEM_FIRST_SHARD: "a" * 64,
    ITEM_LAST_SHARD: "b" * 64,
    ITEM_REVISION: "unknown",
}


def _local(model, base=None):
    return describe_local(collect_trainable(model), _BASE if base is None else base)


class TestFindMismatches:
    def test_equal_ranks_have_no_mismatch(self):
        gathered = [_local(_adapters(seed=0)), _local(_adapters(seed=1))]
        assert find_mismatches(gathered) == []  # values differ, which is not a mismatch

    def test_a_different_layout_is_named_with_the_counts(self):
        gathered = [_local(_adapters(rank=4)), _local(_adapters(rank=8))]
        (found,) = find_mismatches(gathered)
        assert (found.item, found.rank) == (ITEM_LAYOUT, 1)
        assert "rank 0 has 2 tensors / 48 elements, rank 1 has 2 tensors / 96 elements" in (
            found.detail
        )

    def test_the_full_layouts_name_the_first_differing_tensor(self):
        left, right = _adapters(rank=4), _adapters(rank=8)
        layouts = [
            trainable_layout(collect_trainable(left)),
            trainable_layout(collect_trainable(right)),
        ]
        (found,) = find_mismatches([_local(left), _local(right)], layouts)
        assert "first difference: tensor 0 'lora_A' is (4, 6) torch.float32 on rank 0" in (
            found.detail
        )
        assert "(8, 6) torch.float32 on rank 1" in found.detail

    @pytest.mark.parametrize("item", BASE_ITEMS)
    def test_each_base_item_is_named_on_its_own(self, item):
        other = dict(_BASE, **{item: "c" * 64 if item != ITEM_QUANT else "nf4 (x)"})
        gathered = [_local(_adapters()), _local(_adapters(), other)]
        (found,) = find_mismatches(gathered)
        assert (found.item, found.rank) == (item, 1)
        assert "on rank 1" in found.detail and "on rank 0" in found.detail

    def test_digests_are_shortened_and_other_values_are_not(self):
        other = dict(_BASE, **{ITEM_FIRST_SHARD: "c" * 64, ITEM_REVISION: "a" * 40})
        found = find_mismatches([_local(_adapters()), _local(_adapters(), other)])
        by_item = {entry.item: entry.detail for entry in found}
        assert by_item[ITEM_FIRST_SHARD] == "cccccccccccc... on rank 1, aaaaaaaaaaaa... on rank 0"
        assert by_item[ITEM_REVISION] == f"{'a' * 40} on rank 1, unknown on rank 0"

    def test_only_the_rank_that_differs_is_named(self):
        gathered = [_local(_adapters()), _local(_adapters()), _local(_adapters(rank=2))]
        (found,) = find_mismatches(gathered)
        assert found.rank == 2

    def test_a_refusal_lists_every_item_in_the_documented_order(self):
        other = dict(_BASE, **{ITEM_LAST_SHARD: "c" * 64, ITEM_FIRST_SHARD: "d" * 64})
        gathered = [_local(_adapters(rank=4)), _local(_adapters(rank=8), other)]
        message = format_mismatches(find_mismatches(gathered))
        positions = [
            message.index(f"- {item}: rank 1 differs from rank 0")
            for item in (ITEM_LAYOUT, ITEM_FIRST_SHARD, ITEM_LAST_SHARD)
        ]
        assert positions == sorted(positions)
        assert "Refusing to train" in message
        assert "same GPU model" in message


class TestFirstLayoutDifference:
    def test_a_shape_difference(self):
        left = [["a", [4, 6], "torch.float32"]]
        right = [["a", [8, 6], "torch.float32"]]
        assert first_layout_difference(left, right, 1) == (
            "tensor 0 'a' is (4, 6) torch.float32 on rank 0 and (8, 6) torch.float32 on rank 1"
        )

    def test_a_name_difference(self):
        left = [["a", [1], "torch.float32"], ["v_proj", [1], "torch.float32"]]
        right = [["a", [1], "torch.float32"], ["k_proj", [1], "torch.float32"]]
        assert first_layout_difference(left, right, 2) == (
            "tensor 1 is 'v_proj' on rank 0 and 'k_proj' on rank 2"
        )

    def test_a_dtype_difference(self):
        left = [["a", [1], "torch.float32"]]
        right = [["a", [1], "torch.bfloat16"]]
        assert "torch.float32 on rank 0 and (1,) torch.bfloat16 on rank 1" in (
            first_layout_difference(left, right, 1)
        )

    def test_a_tensor_that_only_one_rank_has(self):
        left = [["a", [1], "torch.float32"]]
        right = [["a", [1], "torch.float32"], ["extra", [2], "torch.float32"]]
        assert first_layout_difference(left, right, 1) == "tensor 1 'extra' only exists on rank 1"
        assert first_layout_difference(right, left, 1) == "tensor 1 'extra' only exists on rank 0"

    def test_braces_in_a_name_survive(self):
        left = [["a{0}", [1], "torch.float32"]]
        right = [["b{1}", [1], "torch.float32"]]
        assert "'a{0}' on rank 0 and 'b{1}' on rank 1" in first_layout_difference(left, right, 1)

    def test_equal_layouts(self):
        layout = [["a", [1], "torch.float32"]]
        assert first_layout_difference(layout, layout, 1) == "the layouts are equal"

    def test_the_digest_follows_the_layout(self):
        layout = [["a", [1], "torch.float32"]]
        assert layout_digest(layout) == layout_digest([["a", [1], "torch.float32"]])
        assert layout_digest(layout) != layout_digest([["a", [2], "torch.float32"]])
        assert len(layout_digest(layout)) == 64 and set(layout_digest(layout)) <= set(_HEX)


# --------------------------------------------------------------------------
# C5: the collective, on a scripted fake group
# --------------------------------------------------------------------------


@pytest.fixture
def fake_world(monkeypatch):
    """A scripted process group: this process is ``rank`` of ``size``; what the peers send
    to each ``all_gather_object`` is queued in ``peer_queue``; ``broadcast`` copies rank 0's
    flat values into the tensor it is handed (when this process is not rank 0)."""
    torch = pytest.importorskip("torch")
    import torch.distributed as dist

    world = SimpleNamespace(
        size=2,
        rank=1,
        peer_queue=[],
        gather_calls=[],
        broadcasts=[],
        source_flat=None,
        cursor=0,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: world.size)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: world.rank)

    def all_gather_object(output, obj, group=None):
        world.gather_calls.append(obj)
        peers = world.peer_queue.pop(0)
        for index in range(world.size):
            output[index] = obj if index == world.rank else peers[index]

    def broadcast(tensor, src, group=None):
        world.broadcasts.append((tensor.numel(), src))
        if world.source_flat is not None and world.rank != src:
            tensor.copy_(world.source_flat[world.cursor : world.cursor + tensor.numel()])
        world.cursor += tensor.numel()

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(dist, "broadcast", broadcast)
    world.torch = torch
    return world


def _flat(model):
    import torch

    return torch.cat([param.detach().reshape(-1) for _name, param in collect_trainable(model)])


class TestVerifyAndSyncInit:
    def test_values_that_differ_are_overwritten_from_rank_0_in_chunks(self, fake_world):
        rank0, mine = _adapters(seed=0), _adapters(seed=1)
        fake_world.peer_queue = [{0: _local(rank0)}]
        fake_world.source_flat = _flat(rank0)
        assert not fake_world.torch.equal(_flat(mine), _flat(rank0))

        report = verify_and_sync_init(mine, _BASE, chunk_bytes=16)

        assert fake_world.torch.equal(_flat(mine), _flat(rank0)), "rank 0's values not taken"
        assert fake_world.broadcasts == [(4, 0)] * 12  # 48 elements, 4 per 16-byte chunk
        assert report.differing_ranks == (1,)
        assert (report.world_size, report.rank) == (2, 1)
        assert (report.n_tensors, report.n_elements) == (2, 48)
        assert report.broadcast_bytes == 4 * 48
        assert len(fake_world.gather_calls) == 1, "equal layouts need no second gather"

    def test_equal_values_are_reported_as_such_and_left_alone(self, fake_world):
        rank0, mine = _adapters(seed=3), _adapters(seed=3)
        fake_world.peer_queue = [{0: _local(rank0)}]
        fake_world.source_flat = _flat(rank0)
        before = _flat(mine).clone()
        report = verify_and_sync_init(mine, _BASE)
        assert report.differing_ranks == ()
        assert fake_world.torch.equal(_flat(mine), before)
        assert fake_world.broadcasts == [(48, 0)]

    def test_the_frozen_parameter_is_neither_fingerprinted_nor_broadcast(self, fake_world):
        mine = _adapters()
        fake_world.peer_queue = [{0: _local(mine)}]
        fake_world.source_flat = _flat(mine)
        verify_and_sync_init(mine, _BASE)
        assert sum(count for count, _src in fake_world.broadcasts) == 48
        assert fake_world.gather_calls[0]["layout"]["n_tensors"] == 2

    def test_a_different_layout_is_refused_by_name_before_any_broadcast(self, fake_world):
        rank0, mine = _adapters(rank=4), _adapters(rank=8)
        fake_world.peer_queue = [
            {0: _local(rank0)},
            {0: trainable_layout(collect_trainable(rank0))},  # the second, full gather
        ]
        with pytest.raises(DistributedInitMismatchError) as excinfo:
            verify_and_sync_init(mine, _BASE)
        message = str(excinfo.value)
        assert f"- {ITEM_LAYOUT}: rank 1 differs from rank 0" in message
        assert "rank 0 has 2 tensors / 48 elements, rank 1 has 2 tensors / 96 elements" in message
        assert "tensor 0 'lora_A' is (4, 6) torch.float32 on rank 0" in message
        assert "layer shard" not in message, "the base was equal and must not be accused"
        assert len(fake_world.gather_calls) == 2
        assert fake_world.broadcasts == [], "a buffer of another length would hang the group"

    def test_a_different_base_is_refused_by_name_before_any_broadcast(self, fake_world):
        rank0, mine = _adapters(), _adapters()
        theirs = dict(_BASE, **{ITEM_FIRST_SHARD: "9" * 64})
        fake_world.peer_queue = [{0: _local(rank0, theirs)}]
        with pytest.raises(DistributedInitMismatchError) as excinfo:
            verify_and_sync_init(mine, _BASE)
        message = str(excinfo.value)
        assert f"- {ITEM_FIRST_SHARD}: rank 1 differs from rank 0" in message
        assert f"- {ITEM_LAST_SHARD}" not in message
        assert f"- {ITEM_LAYOUT}" not in message
        assert len(fake_world.gather_calls) == 1
        assert fake_world.broadcasts == []

    def test_a_peer_that_could_not_fingerprint_is_named(self, fake_world):
        fake_world.peer_queue = [{0: {"error": "OSError: shard gone"}}]
        with pytest.raises(DistributedInitMismatchError) as excinfo:
            verify_and_sync_init(_adapters(), _BASE)
        assert "rank 0: OSError: shard gone" in str(excinfo.value)
        assert fake_world.broadcasts == []

    def test_a_rank_that_cannot_fingerprint_still_reaches_the_gather(self, fake_world):
        """Else the others would wait for it until the group timeout."""
        fake_world.peer_queue = [{0: _local(_adapters())}]
        with pytest.raises(DistributedInitMismatchError) as excinfo:
            verify_and_sync_init(_adapters(), {"error": "OSError: shard unreadable"})
        assert len(fake_world.gather_calls) == 1
        sent = fake_world.gather_calls[0]
        assert "shard unreadable" in sent["error"] and "frozen-base fingerprint" in sent["error"]
        assert "rank 1:" in str(excinfo.value)

    def test_a_model_without_trainable_parameters_is_sent_as_an_error(self, fake_world):
        torch = fake_world.torch
        frozen = torch.nn.Linear(2, 2).requires_grad_(False)
        fake_world.peer_queue = [{0: _local(_adapters())}]
        with pytest.raises(DistributedInitMismatchError, match=r"no trainable parameters"):
            verify_and_sync_init(frozen, _BASE)
        assert len(fake_world.gather_calls) == 1

    def test_the_third_rank_is_the_one_named(self, fake_world):
        fake_world.size, fake_world.rank = 3, 0
        mine = _adapters()
        theirs = dict(_BASE, **{ITEM_REVISION: "f" * 40})
        fake_world.peer_queue = [{1: _local(mine), 2: _local(mine, theirs)}]
        with pytest.raises(DistributedInitMismatchError) as excinfo:
            verify_and_sync_init(mine, _BASE)
        message = str(excinfo.value)
        assert f"- {ITEM_REVISION}: rank 2 differs from rank 0" in message
        assert "rank 1 differs" not in message

    def test_a_launch_without_a_process_group_is_refused(self, fake_world, monkeypatch):
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        with pytest.raises(RuntimeError, match=r"has no process group.*train its own copy"):
            verify_and_sync_init(_adapters(), _BASE)
        assert fake_world.gather_calls == []


# --------------------------------------------------------------------------
# C8: the group timeout, in process
# --------------------------------------------------------------------------


@pytest.fixture
def live_group(monkeypatch):
    pytest.importorskip("torch")
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    return dist


class TestApplyGroupTimeout:
    def test_it_calls_torch_distributed_set_timeout(self, live_group, monkeypatch):
        calls = []
        monkeypatch.setattr(
            live_group,
            "set_timeout",
            lambda timeout, group=None: calls.append((timeout, group)),
            raising=False,
        )
        result = apply_group_timeout(20)
        assert result == GroupTimeout(SET_TIMEOUT_API, 20.0, "")
        assert SET_TIMEOUT_API == "torch.distributed.set_timeout"
        assert calls == [(timedelta(seconds=20), None)]

    def test_a_float_number_of_seconds_is_kept(self, live_group, monkeypatch):
        calls = []
        monkeypatch.setattr(
            live_group,
            "set_timeout",
            lambda timeout, group=None: calls.append(timeout),
            raising=False,
        )
        apply_group_timeout(7.5)
        assert calls == [timedelta(seconds=7.5)]

    def test_older_torch_falls_back_to_the_private_function(self, live_group, monkeypatch):
        calls = []
        monkeypatch.delattr(live_group, "set_timeout", raising=False)
        monkeypatch.setattr(
            live_group.distributed_c10d,
            "_set_pg_timeout",
            lambda timeout, group=None: calls.append(timeout),
            raising=False,
        )
        result = apply_group_timeout(20)
        assert result.api == LEGACY_TIMEOUT_API
        assert calls == [timedelta(seconds=20)]

    def test_where_torch_has_neither_the_timeout_is_skipped_cleanly(self, live_group, monkeypatch):
        import torch

        monkeypatch.delattr(live_group, "set_timeout", raising=False)
        monkeypatch.delattr(live_group.distributed_c10d, "_set_pg_timeout", raising=False)
        result = apply_group_timeout(20)
        assert result.api is None
        assert result.reason == (
            f"torch {torch.__version__} has no API to change a running group's timeout"
        )

    def test_a_refusing_api_is_reported_not_raised(self, live_group, monkeypatch):
        def refuse(timeout, group=None):
            raise RuntimeError("the backend cannot change its timeout")

        monkeypatch.setattr(live_group, "set_timeout", refuse, raising=False)
        result = apply_group_timeout(20)
        assert result.api is None
        assert result.reason == (
            f"{SET_TIMEOUT_API} failed: RuntimeError: the backend cannot change its timeout"
        )

    def test_without_a_process_group_nothing_is_called(self, monkeypatch):
        pytest.importorskip("torch")
        import torch.distributed as dist

        calls = []
        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        monkeypatch.setattr(dist, "set_timeout", lambda *a: calls.append(a), raising=False)
        result = apply_group_timeout(20)
        assert result.api is None and result.reason == "there is no process group"
        assert calls == []

    @pytest.mark.parametrize("seconds", [True, 0, -1, float("nan"), float("inf"), "30", None])
    def test_a_bad_number_of_seconds_is_a_bug_of_the_caller(self, live_group, seconds):
        with pytest.raises(ValueError, match=r"group timeout must be"):
            apply_group_timeout(seconds)


class TestFinishDistributedSetup:
    """``StreamingSetupMixin._finish_distributed_stream_setup``: C5, then C8, then C7."""

    @pytest.fixture
    def host(self, tmp_path, monkeypatch):
        steps = []
        report = InitReport(
            world_size=2,
            rank=0,
            n_tensors=8,
            n_elements=1024,
            differing_ranks=(),
            broadcast_bytes=4096,
            seconds=0.5,
        )

        def verify(model, base_items):
            steps.append(("init", model, base_items))
            return report

        def attach(trainer, output_dir):
            steps.append(("metrics", trainer, output_dir))

        monkeypatch.setattr("soup_cli.utils.distributed_init_check.verify_and_sync_init", verify)
        monkeypatch.setattr("soup_cli.utils.stream_dp_runtime.attach_stream_dp_metrics", attach)
        host = SimpleNamespace(
            model="the-model",
            trainer="the-trainer",
            _output_dir=str(tmp_path / "out"),
            _stream_base_items={"item": "value"},
        )
        return SimpleNamespace(self=host, steps=steps, report=report)

    def _run(self, host):
        from soup_cli.trainer.stream_setup import StreamingSetupMixin

        StreamingSetupMixin._finish_distributed_stream_setup(host.self)

    def test_the_timeout_is_applied_after_the_init_check_and_before_the_metrics(
        self, host, monkeypatch, capsys
    ):
        applied = []

        def apply(seconds):
            host.steps.append(("timeout", seconds))
            applied.append(seconds)
            return GroupTimeout(SET_TIMEOUT_API, float(seconds))

        monkeypatch.setattr("soup_cli.utils.stream_dp_runtime.apply_group_timeout", apply)
        self._run(host)
        assert [step[0] for step in host.steps] == ["init", "timeout", "metrics"]
        assert host.steps[0][1:] == ("the-model", {"item": "value"})
        assert host.steps[2][1:] == ("the-trainer", host.self._output_dir)
        assert applied == [300], "the shipped value is 300 s"
        output = _plain(capsys.readouterr().out)
        assert "Group timeout set to 300 s (torch.distributed.set_timeout)" in output
        assert "a peer that stops answering ends this run within that time" in output

    def test_the_value_comes_from_the_module_constant(self, host, monkeypatch):
        import soup_cli.trainer.stream_setup as stream_setup

        applied = []
        monkeypatch.setattr(stream_setup, "DISTRIBUTED_STREAM_TIMEOUT_SECONDS", 7)
        monkeypatch.setattr(
            "soup_cli.utils.stream_dp_runtime.apply_group_timeout",
            lambda seconds: applied.append(seconds) or GroupTimeout(SET_TIMEOUT_API, seconds),
        )
        self._run(host)
        assert applied == [7]

    def test_a_torch_without_the_api_is_said_once_and_does_not_stop_the_run(
        self, host, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            "soup_cli.utils.stream_dp_runtime.apply_group_timeout",
            lambda seconds: GroupTimeout(None, float(seconds), "torch 2.0 has no API"),
        )
        self._run(host)
        output = _plain(capsys.readouterr().out)
        assert output.count("Group timeout left at its launch value") == 1, output
        assert "torch 2.0 has no API" in output
        assert "Group timeout set to" not in output
        assert [step[0] for step in host.steps] == ["init", "metrics"], "the metrics must follow"

    def test_the_init_line_says_what_was_checked_and_broadcast(self, host, monkeypatch, capsys):
        monkeypatch.setattr(
            "soup_cli.utils.stream_dp_runtime.apply_group_timeout",
            lambda seconds: GroupTimeout(SET_TIMEOUT_API, float(seconds)),
        )
        self._run(host)
        output = _plain(capsys.readouterr().out)
        assert output.count("Distributed init:") == 1, output
        assert (
            "2 ranks agree on the trainable layout (8 tensors, 1,024 elements) and the "
            "frozen base; adapters broadcast from rank 0 (0.0 MB in 0.5 s)." in output
        )
        assert "started from different adapter values" not in output

    def test_the_init_line_names_the_ranks_that_were_overwritten(self, host, monkeypatch, capsys):
        host.report = InitReport(2, 0, 8, 1024, (1,), 4096, 0.5)
        monkeypatch.setattr(
            "soup_cli.utils.distributed_init_check.verify_and_sync_init",
            lambda model, items: host.report,
        )
        monkeypatch.setattr(
            "soup_cli.utils.stream_dp_runtime.apply_group_timeout",
            lambda seconds: GroupTimeout(SET_TIMEOUT_API, float(seconds)),
        )
        self._run(host)
        assert (
            "rank(s) 1 started from different adapter values (a different seed?) and took "
            "rank 0's" in _plain(capsys.readouterr().out)
        )

    def test_a_refusal_of_the_init_check_stops_before_the_timeout_and_the_metrics(
        self, host, monkeypatch
    ):
        def refuse(model, base_items):
            raise DistributedInitMismatchError("the ranks differ")

        applied = []
        monkeypatch.setattr("soup_cli.utils.distributed_init_check.verify_and_sync_init", refuse)
        monkeypatch.setattr(
            "soup_cli.utils.stream_dp_runtime.apply_group_timeout",
            lambda seconds: applied.append(seconds),
        )
        with pytest.raises(DistributedInitMismatchError, match=r"the ranks differ"):
            self._run(host)
        assert applied == [] and host.steps == []


class TestDistributedBaseItems:
    def test_a_failure_is_returned_for_the_gather_not_raised(self, tmp_path):
        from soup_cli.trainer.stream_setup import StreamingSetupMixin

        items = StreamingSetupMixin._distributed_base_items(str(tmp_path / "nowhere"), _index())
        assert set(items) == {"error"}
        assert items["error"].startswith(("FileNotFoundError:", "OSError:", "PermissionError:"))

    def test_the_active_revision_is_part_of_the_items(self, tmp_path):
        from soup_cli.trainer.stream_setup import StreamingSetupMixin
        from soup_cli.utils.run_provenance import BaseRevision, base_revision_scope

        identity = BaseRevision(str(tmp_path), binding_supported=False)
        identity.revision, identity.source = "c" * 40, "hub_commit"
        with base_revision_scope(identity):
            items = StreamingSetupMixin._distributed_base_items(_shards(tmp_path), _index())
        assert items[ITEM_REVISION] == "c" * 40
        outside = StreamingSetupMixin._distributed_base_items(_shards(tmp_path), _index())
        assert outside[ITEM_REVISION] == "unknown"


# --------------------------------------------------------------------------
# C7: the per-step metrics callback, in process
# --------------------------------------------------------------------------


class _Trainer:
    """The two attributes the adapter-sync trainer keeps for the metrics callback."""

    def __init__(self):
        self.sync_calls = 0
        self.last_sync_stats = None

    def sync(self, stats):
        self.sync_calls += 1
        self.last_sync_stats = stats


_STATS = SyncStats(bytes=192, seconds=0.5, n_params=2, n_chunks=3)


def _drive(callback, trainer, steps, *, rank=1, world=2, log_every=1, sync=True, logs=None):
    """The events HF fires: begin, then per step begin / (sync) / end / log, then end."""
    args = SimpleNamespace(process_index=rank, world_size=world)
    state = SimpleNamespace(global_step=0)
    control = SimpleNamespace(should_log=False)
    callback.on_train_begin(args, state, control)
    for step in range(1, steps + 1):
        callback.on_step_begin(args, state, control)
        if sync:
            trainer.sync(_STATS)
        state.global_step = step
        control.should_log = step % log_every == 0
        callback.on_step_end(args, state, control)
        if control.should_log:
            callback.on_log(
                args, state, control, logs=logs or {"loss": 1.0 / step, "grad_norm": 0.5 * step}
            )
    callback.on_log(args, state, control, logs={"train_runtime": 3.0})  # HF's final summary
    callback.on_train_end(args, state, control)
    return args


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text("utf-8").splitlines()]


class TestMetricsCallback:
    def test_one_line_per_optimizer_step_with_the_sync_of_that_step(self, tmp_path):
        pytest.importorskip("transformers")
        trainer = _Trainer()
        callback = build_stream_dp_metrics_callback(trainer, str(tmp_path))
        _drive(callback, trainer, 3, rank=1)
        lines = _lines(tmp_path / "stream_dp_metrics.rank1.jsonl")
        assert [line["step"] for line in lines] == [1, 2, 3]
        for line in lines:
            assert set(line) == {
                "step",
                "loss",
                "grad_norm",
                "t_step",
                "t_sync",
                "sync_bytes",
                "sync_chunks",
                "world_size",
                "rank",
            }
            assert line["sync_bytes"] == 192 and line["sync_chunks"] == 3
            assert line["t_sync"] == 0.5
            assert (line["world_size"], line["rank"]) == (2, 1)
            assert line["t_step"] is not None and line["t_step"] >= 0
        assert [line["loss"] for line in lines] == [1.0, 0.5, 1.0 / 3]
        assert [line["grad_norm"] for line in lines] == [0.5, 1.0, 1.5]

    def test_a_step_that_is_not_logged_has_no_loss_and_is_written_at_once(self, tmp_path):
        pytest.importorskip("transformers")
        trainer = _Trainer()
        callback = build_stream_dp_metrics_callback(trainer, str(tmp_path))
        args = SimpleNamespace(process_index=0, world_size=2)
        state, control = SimpleNamespace(global_step=0), SimpleNamespace(should_log=False)
        callback.on_train_begin(args, state, control)
        callback.on_step_begin(args, state, control)
        trainer.sync(_STATS)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        # the heartbeat: on disk before any log event
        assert _lines(tmp_path / "stream_dp_metrics.rank0.jsonl")[0]["loss"] is None

    def test_every_other_step_logged_gives_every_step_a_line(self, tmp_path):
        pytest.importorskip("transformers")
        trainer = _Trainer()
        callback = build_stream_dp_metrics_callback(trainer, str(tmp_path))
        _drive(callback, trainer, 4, rank=0, log_every=2)
        lines = _lines(tmp_path / "stream_dp_metrics.rank0.jsonl")
        assert [line["step"] for line in lines] == [1, 2, 3, 4]
        assert [line["loss"] is None for line in lines] == [True, False, True, False]

    def test_a_step_without_a_sync_reports_zeros_not_the_previous_steps_numbers(self, tmp_path):
        pytest.importorskip("transformers")
        trainer = _Trainer()
        callback = build_stream_dp_metrics_callback(trainer, str(tmp_path))
        trainer.sync(_STATS)  # an earlier sync, before training begins
        _drive(callback, trainer, 2, sync=False)
        for line in _lines(tmp_path / "stream_dp_metrics.rank1.jsonl"):
            assert (line["t_sync"], line["sync_bytes"], line["sync_chunks"]) == (0.0, 0, 0)

    def test_a_loss_that_is_not_finite_is_null_and_the_file_stays_valid_json(self, tmp_path):
        pytest.importorskip("transformers")
        trainer = _Trainer()
        callback = build_stream_dp_metrics_callback(trainer, str(tmp_path))
        _drive(callback, trainer, 1, logs={"loss": float("nan"), "grad_norm": float("inf")})
        text = (tmp_path / "stream_dp_metrics.rank1.jsonl").read_text("utf-8")
        assert "NaN" not in text and "Infinity" not in text
        (line,) = _lines(tmp_path / "stream_dp_metrics.rank1.jsonl")
        assert line["loss"] is None and line["grad_norm"] is None

    def test_a_record_that_no_log_event_claimed_is_written_when_training_ends(self, tmp_path):
        pytest.importorskip("transformers")
        trainer = _Trainer()
        callback = build_stream_dp_metrics_callback(trainer, str(tmp_path))
        args = SimpleNamespace(process_index=0, world_size=2)
        state, control = SimpleNamespace(global_step=0), SimpleNamespace(should_log=True)
        callback.on_train_begin(args, state, control)
        callback.on_step_begin(args, state, control)
        trainer.sync(_STATS)
        state.global_step = 1
        callback.on_step_end(args, state, control)
        callback.on_train_end(args, state, control)
        assert [line["step"] for line in _lines(tmp_path / "stream_dp_metrics.rank0.jsonl")] == [1]

    def test_the_file_is_truncated_when_training_begins(self, tmp_path):
        pytest.importorskip("transformers")
        path = tmp_path / "stream_dp_metrics.rank1.jsonl"
        path.write_text('{"step": 99}\n', encoding="utf-8")
        trainer = _Trainer()
        _drive(build_stream_dp_metrics_callback(trainer, str(tmp_path)), trainer, 2)
        assert [line["step"] for line in _lines(path)] == [1, 2]

    def test_ranks_that_share_a_directory_write_different_files(self, tmp_path):
        pytest.importorskip("transformers")
        for rank in (0, 1):
            trainer = _Trainer()
            _drive(build_stream_dp_metrics_callback(trainer, str(tmp_path)), trainer, 2, rank=rank)
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "stream_dp_metrics.rank0.jsonl",
            "stream_dp_metrics.rank1.jsonl",
        ]
        for rank in (0, 1):
            lines = _lines(tmp_path / f"stream_dp_metrics.rank{rank}.jsonl")
            assert {line["rank"] for line in lines} == {rank}

    def test_an_unwritable_file_is_one_warning_and_training_goes_on(self, tmp_path, capsys):
        pytest.importorskip("transformers")
        blocker = tmp_path / "a-file-not-a-directory"
        blocker.write_text("x", encoding="utf-8")
        trainer = _Trainer()
        _drive(build_stream_dp_metrics_callback(trainer, str(blocker)), trainer, 3)
        output = _plain(capsys.readouterr().out)
        assert output.count("Per-step metrics file could not be written") == 1, output
        assert "training goes on without it" in output

    def test_the_path_names_the_rank(self, tmp_path):
        assert stream_dp_metrics_path(str(tmp_path), 3) == str(
            tmp_path / "stream_dp_metrics.rank3.jsonl"
        )
        for bad in (-1, True, "1", 1.0, None):
            with pytest.raises(ValueError, match=r"rank must be a non-negative int"):
                stream_dp_metrics_path(str(tmp_path), bad)


# --------------------------------------------------------------------------
# C6: rank gating, in process
# --------------------------------------------------------------------------


def _state(*, chief=True, step=3):
    return SimpleNamespace(global_step=step, max_steps=10, epoch=1.0, is_world_process_zero=chief)


def _soup_callback(*, rank_gated=True, **kwargs):
    from soup_cli.monitoring.callback import SoupTrainerCallback

    callback = SoupTrainerCallback(**kwargs)
    callback.rank_gated = rank_gated
    return callback


_LOGS = {"loss": 1.0, "learning_rate": 1e-4, "grad_norm": 0.3}


class TestCallbackRankGate:
    def test_without_the_streaming_gate_every_rank_keeps_its_outputs(self):
        # A resident distributed run (no stream_layers) keeps one tracker row per rank,
        # so rank 1 must keep writing its metrics rows and drawing its panel, as before C6.
        display, tracker = MagicMock(), MagicMock()
        callback = _soup_callback(
            rank_gated=False, display=display, tracker=tracker, run_id="run-1"
        )
        state, control = _state(chief=False), MagicMock()
        callback.on_train_begin(MagicMock(), state, control)
        callback.on_log(MagicMock(), state, control, logs=dict(_LOGS))
        callback.on_train_end(MagicMock(), state, control)
        display.start.assert_called_once()
        display.update.assert_called()
        display.stop.assert_called_once()
        tracker.log_metrics.assert_called()

    def test_the_builder_gates_only_when_asked(self):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.monitoring.callback import build_soup_trainer_callback

        cfg = SoupConfig(base="m", data={"train": "d.jsonl"})
        plain = build_soup_trainer_callback(MagicMock(), config=cfg)
        gated = build_soup_trainer_callback(MagicMock(), config=cfg, rank_gated=True)
        assert plain.rank_gated is False
        assert gated.rank_gated is True

    def test_a_rank_that_is_not_the_chief_draws_nothing_and_records_nothing(self):
        display, tracker = MagicMock(), MagicMock()
        callback = _soup_callback(display=display, tracker=tracker, run_id="run-1")
        state, control = _state(chief=False), MagicMock()
        callback.on_train_begin(MagicMock(), state, control)
        callback.on_log(MagicMock(), state, control, logs=dict(_LOGS))
        callback.on_train_end(MagicMock(), state, control)
        display.start.assert_not_called()
        display.update.assert_not_called()
        display.stop.assert_not_called()
        tracker.log_metrics.assert_not_called()

    def test_the_chief_keeps_every_output(self):
        display, tracker = MagicMock(), MagicMock()
        callback = _soup_callback(display=display, tracker=tracker, run_id="run-1")
        state, control = _state(chief=True), MagicMock()
        callback.on_train_begin(MagicMock(), state, control)
        callback.on_log(MagicMock(), state, control, logs=dict(_LOGS))
        callback.on_train_end(MagicMock(), state, control)
        display.start.assert_called_once_with(total_steps=10)
        display.update.assert_called_once()
        display.stop.assert_called_once()
        tracker.log_metrics.assert_called_once()
        assert tracker.log_metrics.call_args.kwargs["run_id"] == "run-1"
        assert tracker.log_metrics.call_args.kwargs["step"] == 3

    def test_a_state_without_the_field_is_the_chief(self):
        """A single process (and every existing caller with a bare state) behaves as before."""
        display, tracker = MagicMock(), MagicMock()
        callback = _soup_callback(display=display, tracker=tracker, run_id="run-1")
        state = SimpleNamespace(global_step=1, max_steps=2, epoch=1.0)
        callback.on_train_begin(MagicMock(), state, MagicMock())
        callback.on_log(MagicMock(), state, MagicMock(), logs=dict(_LOGS))
        display.start.assert_called_once()
        tracker.log_metrics.assert_called_once()

    def test_the_watchdog_stops_every_rank_but_only_the_chief_speaks(self, capsys):
        outcomes = {}
        for chief in (True, False):
            display = MagicMock()
            callback = _soup_callback(
                display=display,
                loss_watchdog=True,
                loss_watchdog_threshold=1.0,
                loss_watchdog_patience=1,
            )
            control = SimpleNamespace(should_training_stop=False)
            callback.on_log(
                MagicMock(), _state(chief=chief), control, logs={"loss": 5.0, "grad_norm": 1.0}
            )
            outcomes[chief] = (control.should_training_stop, display.stop.called)
            outcomes[(chief, "out")] = _plain(capsys.readouterr().out)
        assert outcomes[True] == (True, True)
        assert outcomes[False] == (True, False), "a rank that goes on would hang its peers"
        assert "Loss watchdog triggered" in outcomes[(True, "out")]
        assert "Loss watchdog triggered" not in outcomes[(False, "out")]

    def test_the_spike_hint_is_written_by_the_chief_alone(self):
        calls = {}
        for chief in (True, False):
            callback = _soup_callback(
                display=MagicMock(),
                loss_watchdog=True,
                loss_watchdog_threshold=1.0,
                loss_watchdog_patience=1,
                spike_recovery=True,
            )
            callback._write_spike_recovery_hint = MagicMock()
            callback.on_log(
                MagicMock(),
                _state(chief=chief),
                SimpleNamespace(should_training_stop=False),
                logs={"loss": 5.0},
            )
            calls[chief] = callback._write_spike_recovery_hint.call_count
        assert calls == {True: 1, False: 0}


class TestRunTracker:
    @pytest.fixture
    def database(self, tmp_path, monkeypatch):
        path = tmp_path / "runs.db"
        monkeypatch.setenv("SOUP_DB_PATH", str(path))
        for name in ("RANK", "LOCAL_RANK"):
            monkeypatch.delenv(name, raising=False)
        return path

    def test_the_chief_gets_the_sqlite_tracker(self, database, monkeypatch):
        from soup_cli.commands.train import _open_run_tracker
        from soup_cli.experiment.tracker import ExperimentTracker

        monkeypatch.setenv("RANK", "0")
        tracker = _open_run_tracker(distributed_stream=True)
        assert type(tracker) is ExperimentTracker and tracker.records_runs is True
        assert database.exists()

    def test_every_other_rank_gets_a_tracker_that_writes_nothing(self, database, monkeypatch):
        from soup_cli.commands.train import _open_run_tracker
        from soup_cli.experiment.tracker import NullTracker

        monkeypatch.setenv("RANK", "1")
        tracker = _open_run_tracker(distributed_stream=True)
        assert type(tracker) is NullTracker and tracker.records_runs is False
        assert not database.exists(), "a database was opened for a rank that records nothing"

    def test_a_launch_that_is_not_streaming_distributed_records_on_every_rank(
        self, database, monkeypatch
    ):
        from soup_cli.commands.train import _open_run_tracker
        from soup_cli.experiment.tracker import ExperimentTracker

        monkeypatch.setenv("RANK", "1")
        assert type(_open_run_tracker(distributed_stream=False)) is ExperimentTracker

    def test_without_rank_the_local_rank_decides(self, database, monkeypatch):
        from soup_cli.commands.train import _open_run_tracker
        from soup_cli.experiment.tracker import NullTracker

        monkeypatch.setenv("LOCAL_RANK", "1")
        assert type(_open_run_tracker(distributed_stream=True)) is NullTracker


class TestNullTracker:
    _CALLED_BY_TRAIN = (
        "start_run",
        "mark_running",
        "record_training_data_hash",
        "record_base_revision",
        "finish_run",
        "fail_run",
        "get_run",
        "get_metrics",
    )

    def test_it_has_every_method_soup_train_calls_on_a_tracker(self):
        import soup_cli.commands.train as train_command
        from soup_cli.experiment.tracker import NullTracker

        called = set(re.findall(r"\btracker\.(\w+)\(", inspect.getsource(train_command)))
        assert set(self._CALLED_BY_TRAIN) <= called, called
        assert not [name for name in called if not hasattr(NullTracker, name)], called

    def test_its_signatures_are_the_sqlite_trackers(self):
        from soup_cli.experiment.tracker import ExperimentTracker, NullTracker

        public = [
            name
            for name, member in vars(NullTracker).items()
            if callable(member) and not name.startswith("_")
        ]
        assert set(public) >= set(self._CALLED_BY_TRAIN) | {"log_metrics", "close"}
        for name in public:
            assert str(inspect.signature(getattr(NullTracker, name))) == str(
                inspect.signature(getattr(ExperimentTracker, name))
            ), name

    def test_it_records_nothing_and_answers_like_an_empty_database(self, tmp_path, monkeypatch):
        from soup_cli.experiment.tracker import NullTracker

        monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
        tracker = NullTracker()
        generated = tracker.start_run({}, "cpu", "cpu", {})
        # --profile refuses an empty id (utils/profiling.py resolve_trace_path), on this
        # rank alone: the id must look like a real one
        assert re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{8}", generated), generated
        from soup_cli.utils.profiling import resolve_trace_path

        resolve_trace_path(Path("."), generated)
        assert tracker.start_run({}, "cpu", "cpu", {}, run_id="given") == "given"
        tracker.mark_running("", pid=1)
        tracker.log_metrics("", 1, loss=1.0)
        tracker.finish_run("", 1.0, 0.5, 5, 1.0, "out")
        tracker.fail_run("", error="boom")
        assert tracker.get_run("") is None and tracker.get_metrics("") == []
        assert not (tmp_path / "runs.db").exists()


class TestWritesArtifacts:
    @pytest.mark.parametrize(
        ("distributed", "should_save", "expected"),
        [
            (True, True, True),
            (True, False, False),
            (False, True, True),
            (False, False, True),  # a run that is not distributed saves as it always did
        ],
    )
    def test_only_a_rank_that_does_not_save_the_model_skips_the_tokenizer(
        self, distributed, should_save, expected
    ):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper._distributed_stream = distributed
        wrapper.trainer = SimpleNamespace(args=SimpleNamespace(should_save=should_save))
        assert wrapper._writes_artifacts() is expected


class _StopAtTrainerError(Exception):
    """Raised by the stub trainer: the command got as far as building one."""


def _invoke_train_to_the_trainer(tmp_path, monkeypatch, *, rank, world="2"):
    """``soup train`` as one rank of a streaming launch (``world`` ranks), up to the trainer."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(_row(index, 8)) for index in range(3)) + "\n", encoding="utf-8"
    )
    (tmp_path / "soup.yaml").write_text(
        "base: smollm2-135m\n"
        "task: sft\n"
        "backend: transformers\n"
        "data:\n"
        "  train: train.jsonl\n"
        "training:\n"
        "  stream_layers: true\n"
        "  batch_size: 1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    for name, value in (
        ("WORLD_SIZE", world),
        ("LOCAL_WORLD_SIZE", "1"),
        ("RANK", str(rank)),
        ("LOCAL_RANK", "0"),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    for name in ("ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_FSDP"):
        monkeypatch.delenv(name, raising=False)
    # No process group is created here: the base-revision exchange would wait for a peer.
    monkeypatch.setattr(
        "soup_cli.utils.run_provenance.synchronize_base_revision", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "soup_cli.utils.run_provenance.confirm_distributed_base_revision", lambda *a, **k: None
    )

    class _Wrapper:
        def setup(self, dataset):
            raise _StopAtTrainerError

    monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", lambda cfg, **kw: _Wrapper())
    return CliRunner().invoke(app, ["train", "--config", str(tmp_path / "soup.yaml"), "--yes"])


class TestTrainCommandTracker:
    def test_rank_0_records_the_run(self, tmp_path, monkeypatch):
        from soup_cli.experiment.tracker import ExperimentTracker

        result = _invoke_train_to_the_trainer(tmp_path, monkeypatch, rank=0)
        assert isinstance(result.exception, _StopAtTrainerError), (
            result.output,
            repr(result.exception),
        )
        assert "Run ID:" in _plain(result.output)
        runs = ExperimentTracker(tmp_path / "runs.db").list_runs()
        assert len(runs) == 1 and runs[0]["status"] == "failed"

    def test_every_other_rank_records_nothing_and_says_so(self, tmp_path, monkeypatch):
        result = _invoke_train_to_the_trainer(tmp_path, monkeypatch, rank=1)
        assert isinstance(result.exception, _StopAtTrainerError), (
            result.output,
            repr(result.exception),
        )
        output = _plain(result.output)
        assert "Run record: rank 0 keeps it; this rank records nothing." in output
        assert "Run ID:" not in output
        assert not (tmp_path / "runs.db").exists()

    def test_a_run_that_is_not_distributed_records_on_whatever_rank(self, tmp_path, monkeypatch):
        from soup_cli.experiment.tracker import ExperimentTracker

        # the launch variables of rank 1, minus the world: one process, not a distributed run
        result = _invoke_train_to_the_trainer(tmp_path, monkeypatch, rank=1, world=None)
        assert isinstance(result.exception, _StopAtTrainerError), (
            result.output,
            repr(result.exception),
        )
        assert len(ExperimentTracker(tmp_path / "runs.db").list_runs()) == 1


# --------------------------------------------------------------------------
# two real ranks
# --------------------------------------------------------------------------


def _need_two_rank_runs():
    distributed = pytest.importorskip("torch.distributed")
    if not distributed.is_gloo_available():
        pytest.skip("installed torch has no CPU Gloo backend")
    if sys.platform == "darwin":
        pytest.skip("two-process Gloo runs are skipped on macOS (runner timeouts)")
    for module in ("transformers", "peft", "trl", "datasets", "accelerate", "safetensors"):
        pytest.importorskip(module)


def _build_base(target: Path, seed: int) -> str:
    """A tiny Llama with its tokenizer, from a fixed seed (two seeds: two different bases)."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from tests._tiny_hf_models import build_tokenizer

    torch.manual_seed(seed)
    tokenizer = build_tokenizer()
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tokenizer),
        max_position_embeddings=256,
    )
    LlamaForCausalLM(config).to(torch.float32).save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return str(target)


@pytest.fixture(scope="module")
def bases(tmp_path_factory):
    for module in ("torch", "transformers", "tokenizers"):
        pytest.importorskip(module)
    root = tmp_path_factory.mktemp("bases")
    return SimpleNamespace(same=_build_base(root / "a", 7), other=_build_base(root / "b", 8))


def _launch(
    out_dir: Path,
    *,
    base_dirs,
    rows,
    extra=None,
    batch=2,
    wait_for=_TAGS,
    deadline=_RUN_DEADLINE,
    marker=None,
):
    """Both ranks at once; wait until the ``wait_for`` ranks have exited (or the deadline).

    ``extra[tag]`` are further worker arguments. ``marker`` is a line a rank prints at the
    moment of a fault, whose time of appearance is recorded. Whatever is still running
    after the wait is killed. Nothing here fails on a non-zero exit: the callers decide.
    """
    _need_two_rank_runs()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_file = out_dir / "rows.json"
    rows_file.write_text(json.dumps(rows), encoding="utf-8")
    port = _free_port()
    extra = extra or {}
    processes, handles, logs = {}, [], {}
    exited_at, marker_at, codes = {}, {}, {}
    hung = False
    start = time.monotonic()
    try:
        for rank, tag in enumerate(_TAGS):
            env = _worker_env()
            env.update(
                RANK=str(rank),
                WORLD_SIZE="2",
                LOCAL_RANK="0",
                LOCAL_WORLD_SIZE="1",  # one rank per host: what the decision requires
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(port),
                ACCELERATE_USE_CPU="true",  # accelerate: MULTI_CPU, gloo, env://
            )
            logs[tag] = out_dir / f"{tag}.log"
            handle = logs[tag].open("w", encoding="utf-8")
            handles.append(handle)
            command = [sys.executable, str(_WORKER)]
            for flag, value in (
                ("--rank", rank),
                ("--world-size", 2),
                ("--tag", tag),
                ("--out-dir", out_dir),
                ("--base-dir", base_dirs[tag]),
                ("--rows-file", rows_file),
                ("--batch-size", batch),
            ):
                command += [flag, str(value)]
            command += [str(part) for part in extra.get(tag, [])]
            processes[tag] = subprocess.Popen(
                command, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
        while True:
            now = time.monotonic() - start
            for tag, process in processes.items():
                if tag not in exited_at and process.poll() is not None:
                    exited_at[tag] = now
                    codes[tag] = process.returncode
                if marker and tag not in marker_at:
                    try:
                        text = logs[tag].read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        text = ""
                    if marker in text:
                        marker_at[tag] = now
            if all(tag in exited_at for tag in wait_for):
                break
            if now > deadline:
                hung = True
                break
            time.sleep(0.25)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
        for handle in handles:
            handle.close()
    runs = {
        tag: SimpleNamespace(
            tag=tag,
            code=codes.get(tag),
            exited_at=exited_at.get(tag),
            marker_at=marker_at.get(tag),
            log=_plain(logs[tag].read_text(encoding="utf-8", errors="replace")),
            raw=logs[tag],
        )
        for tag in _TAGS
    }
    return SimpleNamespace(out_dir=out_dir, runs=runs, hung=hung)


def _report(outcome) -> str:
    return "\n".join(
        f"--- {tag} (exit {run.code}, last 40 lines) ---\n{_tail(run.raw)}"
        for tag, run in outcome.runs.items()
    )


def _facts(outcome, tag: str) -> dict:
    return json.loads((outcome.out_dir / f"{tag}.json").read_text("utf-8"))


def _require_clean(outcome) -> None:
    bad = [tag for tag, run in outcome.runs.items() if run.code != 0]
    if bad or outcome.hung:
        pytest.fail(
            f"two-rank run failed: {'hung' if outcome.hung else 'exit codes ' + str(bad)}\n"
            + _report(outcome),
            pytrace=False,
        )


def _rows(count: int = _ROWS) -> list:
    return [_row(index, 24) for index in range(count)]


def _same_base(bases):
    return {tag: bases.same for tag in _TAGS}


def _args_for(**per_tag) -> dict:
    return {tag: [str(part) for part in parts] for tag, parts in per_tag.items()}


@pytest.fixture(scope="module")
def run_seeds(bases, tmp_path_factory):
    """T2 (seed), T6, T8, T9: different seeds per rank, checkpoints every 2 steps."""
    common = ["--save-steps", 2, "--sequential-sampler", "--group-timeout", _GROUP_TIMEOUT]
    outcome = _launch(
        tmp_path_factory.mktemp("seeds"),
        base_dirs=_same_base(bases),
        rows=_rows(),
        extra=_args_for(dp0=[*common, "--seed", 42], dp1=[*common, "--seed", 43]),
    )
    _require_clean(outcome)
    return outcome


@pytest.fixture(scope="module")
def run_sampling(bases, tmp_path_factory):
    """T7: the production sampler (no override), one seed, 21 rows that do not divide."""
    common = ["--group-timeout", _GROUP_TIMEOUT]
    outcome = _launch(
        tmp_path_factory.mktemp("sampling"),
        base_dirs=_same_base(bases),
        rows=_rows(21),
        extra=_args_for(dp0=common, dp1=common),
    )
    _require_clean(outcome)
    return outcome


class TestSeedsEndEqual:
    """T2, the value half: a seed that differs between ranks is fixed by the broadcast."""

    def test_the_ranks_start_and_end_bitwise_equal(self, run_seeds):
        zero, one = _facts(run_seeds, "dp0"), _facts(run_seeds, "dp1")
        assert zero["initial_sha"] == one["initial_sha"], "the broadcast did not equalise them"
        assert zero["final_sha"] == one["final_sha"]
        assert zero["final_sha"] != zero["initial_sha"], "the adapters never moved"
        assert zero["global_step"] == one["global_step"] == _STEPS
        assert zero["losses"] == one["losses"]

    def test_each_rank_says_once_that_rank_1_took_rank_0s_values(self, run_seeds):
        for tag, run in run_seeds.runs.items():
            assert run.log.count("Distributed init:") == 1, (tag, run.log[-1500:])
            assert "2 ranks agree on the trainable layout (8 tensors" in run.log
            assert "adapters broadcast from rank 0" in run.log
            assert (
                "rank(s) 1 started from different adapter values (a different seed?) "
                "and took rank 0's" in run.log
            ), (tag, run.log[-1500:])


class TestRankGatedArtifacts:
    """T6: which rank wrote what. Each rank has its own output directory."""

    @staticmethod
    def _files(run, tag):
        return _facts(run, tag)["files"]

    def test_only_rank_0_wrote_the_adapter_and_its_checkpoints(self, run_seeds):
        zero, one = self._files(run_seeds, "dp0"), self._files(run_seeds, "dp1")
        assert "adapter_model.safetensors" in zero and "adapter_config.json" in zero
        for step in (2, 4):
            assert f"checkpoint-{step}/adapter_model.safetensors" in zero, zero
        assert not [name for name in one if name.endswith("adapter_model.safetensors")], one
        assert not [name for name in one if name.endswith("adapter_config.json")], one

    def test_a_checkpoint_directory_on_the_other_rank_holds_only_its_rng_state(self, run_seeds):
        """HF's ``_save_checkpoint`` creates ``checkpoint-N`` on every rank for the per-rank
        ``rng_state_<rank>.pth``; the design note's 'only rank 0 has checkpoint-*' is not
        literally true. Nothing that makes a checkpoint loadable is written there."""
        one = self._files(run_seeds, "dp1")
        inside = [name for name in one if name.startswith("checkpoint-")]
        assert all(name.rsplit("/", 1)[-1] == "rng_state_1.pth" for name in inside), inside

    def test_the_tokenizer_is_saved_where_the_model_is_and_nowhere_else(self, run_seeds):
        zero, one = self._files(run_seeds, "dp0"), self._files(run_seeds, "dp1")
        assert any(name.startswith("tokenizer") and "/" not in name for name in zero), zero
        assert not [name for name in one if "tokenizer" in name], one

    def test_each_rank_wrote_its_own_metrics_file_and_only_that(self, run_seeds):
        zero, one = self._files(run_seeds, "dp0"), self._files(run_seeds, "dp1")
        assert "stream_dp_metrics.rank0.jsonl" in zero
        assert "stream_dp_metrics.rank1.jsonl" not in zero
        assert "stream_dp_metrics.rank1.jsonl" in one
        assert "stream_dp_metrics.rank0.jsonl" not in one

    def test_the_live_panel_was_drawn_on_rank_0_alone(self, run_seeds):
        zero, one = _facts(run_seeds, "dp0"), _facts(run_seeds, "dp1")
        assert zero["display_calls"]["start"] == 1 and zero["display_calls"]["stop"] == 1
        assert zero["display_calls"]["update"] >= _STEPS
        assert one["display_calls"] == {"start": 0, "update": 0, "stop": 0}

    def test_tracker_rows_were_written_for_rank_0_alone(self, run_seeds):
        """Both ranks hand the callback a REAL tracker (each its own database), so only the
        callback's rank gate keeps rank 1 from writing."""
        zero, one = _facts(run_seeds, "dp0"), _facts(run_seeds, "dp1")
        assert zero["metric_rows"] >= _STEPS, zero
        assert one["metric_rows"] == 0, one


class TestStepMetricsFile:
    """T8: one line per optimizer step per rank, with exactly the adapter bytes."""

    @pytest.mark.parametrize("tag", _TAGS)
    def test_one_line_per_step_with_the_exact_wire_bytes(self, run_seeds, tag):
        facts = _facts(run_seeds, tag)
        rank = facts["process_index"]
        lines = _lines(Path(facts["output_dir"]) / f"stream_dp_metrics.rank{rank}.jsonl")
        assert [line["step"] for line in lines] == list(range(1, _STEPS + 1)), lines
        for line in lines:
            assert line["sync_bytes"] == 4 * facts["n_trainable"] > 0, line
            assert line["t_sync"] > 0, line
            assert line["sync_chunks"] == 1, line
            assert (line["world_size"], line["rank"]) == (2, rank), line
            assert line["t_step"] >= line["t_sync"], line
            assert line["grad_norm"] is not None and line["grad_norm"] > 0, line
        assert [line["loss"] for line in lines] == facts["losses"]


class TestGroupTimeoutInARun:
    """T9, on real ranks: the setup asks torch to shorten the group timeout."""

    @pytest.mark.parametrize("tag", _TAGS)
    def test_it_was_applied_once_with_the_configured_seconds(self, run_seeds, tag):
        assert _facts(run_seeds, tag)["timeout_calls"] == [float(_GROUP_TIMEOUT)]
        log = run_seeds.runs[tag].log
        assert (
            log.count(f"Group timeout set to {_GROUP_TIMEOUT} s (torch.distributed.set_timeout)")
            == 1
        )


class TestSharding:
    """T7: what each rank sees under the production sampler and accelerate's sharding."""

    @staticmethod
    def _seen(run):
        seen = [_facts(run, tag)["rows_seen"] for tag in _TAGS]
        assert len(seen[0]) == len(seen[1]), "the ranks took a different number of steps"
        return seen

    def test_each_step_gives_the_two_ranks_different_rows(self, run_sampling):
        zero, one = self._seen(run_sampling)
        for step, (left, right) in enumerate(zip(zero, one)):
            assert not set(left) & set(right), f"step {step}: {left} and {right} overlap"
            assert len(set(left)) == len(left) and len(set(right)) == len(right)

    def test_no_row_is_missing_from_the_epoch(self, run_sampling):
        zero, one = self._seen(run_sampling)
        covered = {row for batch in (*zero, *one) for row in batch}
        assert covered == set(range(21)), sorted(set(range(21)) - covered)

    def test_the_rows_repeated_to_complete_the_last_step_are_counted_and_all_in_the_tail(
        self, run_sampling
    ):
        """21 rows, 2 per rank per step: 6 steps of 4 slots is 24 slots. Accelerate's
        ``even_batches`` completes the short last batch (and the batch the other rank lacks)
        with rows from the START of the epoch, so 3 rows are seen twice, none dropped."""
        zero, one = self._seen(run_sampling)
        steps = len(zero)
        assert steps == 6, steps
        slots = [row for step in range(steps) for row in (*zero[step], *one[step])]
        assert len(slots) == 24
        repeated = len(slots) - len(set(slots))
        assert repeated == 3, f"{repeated} repeated slots"
        before_the_tail = [row for step in range(steps - 1) for row in (*zero[step], *one[step])]
        assert len(before_the_tail) == len(set(before_the_tail)) == 20, "a row repeated early"
        tail = [*zero[-1], *one[-1]]
        assert len([row for row in tail if row not in before_the_tail]) == 1
        assert len([row for row in tail if row in before_the_tail]) == 3


# --------------------------------------------------------------------------
# T2, the structural half: refusals at startup
# --------------------------------------------------------------------------


def _mismatch_run(tmp_path_factory, name, base_dirs, extra):
    outcome = _launch(
        tmp_path_factory.mktemp(name),
        base_dirs=base_dirs,
        rows=_rows(),
        extra=extra,
        deadline=240,
    )
    assert not outcome.hung, "a refusal must not hang the ranks\n" + _report(outcome)
    return outcome


@pytest.fixture(scope="module")
def run_other_r(bases, tmp_path_factory):
    return _mismatch_run(
        tmp_path_factory,
        "other_r",
        _same_base(bases),
        _args_for(dp0=["--lora-r", 4], dp1=["--lora-r", 8]),
    )


@pytest.fixture(scope="module")
def run_other_targets(bases, tmp_path_factory):
    return _mismatch_run(
        tmp_path_factory,
        "other_targets",
        _same_base(bases),
        _args_for(
            dp0=["--target-modules", "q_proj,v_proj"], dp1=["--target-modules", "q_proj,k_proj"]
        ),
    )


@pytest.fixture(scope="module")
def run_other_base(bases, tmp_path_factory):
    return _mismatch_run(
        tmp_path_factory, "other_base", {"dp0": bases.same, "dp1": bases.other}, {}
    )


class TestRefusedAtStartup:
    @staticmethod
    def _both_refuse(outcome):
        for tag, run in outcome.runs.items():
            assert run.code == 1, (tag, run.code, _report(outcome))
            assert "DistributedInitMismatchError" in run.log, (tag, run.log[-1500:])
            assert "Refusing to train" in run.log
            assert not (outcome.out_dir / f"{tag}.json").exists(), "training went on"

    def test_a_different_lora_r_names_the_layout_and_the_first_tensor(self, run_other_r):
        self._both_refuse(run_other_r)
        for run in run_other_r.runs.values():
            assert f"- {ITEM_LAYOUT}: rank 1 differs from rank 0" in run.log, run.log[-1500:]
            assert "rank 0 has 8 tensors" in run.log and "rank 1 has 8 tensors" in run.log
            assert re.search(
                r"tensor 0 '[^']*layers\.0\.self_attn\.q_proj\.lora_A[^']*' is \(4, 32\) "
                r"torch\.float32 on rank 0 and \(8, 32\) torch\.float32 on rank 1",
                run.log,
            ), run.log[-1500:]
            # the base was equal: no mismatch line accuses a shard (the setup log itself
            # says "Preparing layer shards", so match the mismatch line format)
            for item in (ITEM_FIRST_SHARD, ITEM_LAST_SHARD, ITEM_SHARD_INDEX):
                assert f"- {item}: rank" not in run.log, "the base was equal"

    def test_different_target_modules_name_the_first_tensor_that_differs(self, run_other_targets):
        self._both_refuse(run_other_targets)
        for run in run_other_targets.runs.values():
            assert f"- {ITEM_LAYOUT}: rank 1 differs from rank 0" in run.log, run.log[-1500:]
            assert "tensor 2 is" in run.log
            assert "v_proj" in run.log and "k_proj" in run.log
            # the base was equal: no mismatch line accuses a shard (the setup log itself
            # says "Preparing layer shards", so match the mismatch line format)
            for item in (ITEM_FIRST_SHARD, ITEM_LAST_SHARD, ITEM_SHARD_INDEX):
                assert f"- {item}: rank" not in run.log, "the base was equal"

    def test_a_different_base_on_one_rank_names_the_shards_that_differ(self, run_other_base):
        self._both_refuse(run_other_base)
        for run in run_other_base.runs.values():
            assert f"- {ITEM_FIRST_SHARD}: rank 1 differs from rank 0" in run.log, run.log[-1500:]
            assert f"- {ITEM_LAST_SHARD}: rank 1 differs from rank 0" in run.log
            assert f"- {ITEM_LAYOUT}" not in run.log, "the trainable layout was equal"
            assert f"- {ITEM_SHARD_INDEX}" not in run.log, "same shapes, same index"


# --------------------------------------------------------------------------
# T5: a peer that dies or goes silent
# --------------------------------------------------------------------------


def _fault_run(bases, tmp_path_factory, name, *, fault_rank, fault, timeout):
    common = [
        "--group-timeout",
        timeout,
        "--fault",
        fault,
        "--fault-step",
        2,
        "--fault-rank",
        fault_rank,
    ]
    alive = _TAGS[1 - fault_rank]
    return _launch(
        tmp_path_factory.mktemp(name),
        base_dirs=_same_base(bases),
        rows=_rows(),
        extra=_args_for(dp0=common, dp1=common),
        wait_for=_TAGS if fault == "exit" else (alive,),
        deadline=timeout + _SLACK + 240,
        marker="fault -- ",
    )


@pytest.fixture(scope="module")
def rank_1_exits(bases, tmp_path_factory):
    return _fault_run(
        bases, tmp_path_factory, "rank1_exits", fault_rank=1, fault="exit", timeout=_GROUP_TIMEOUT
    )


@pytest.fixture(scope="module")
def rank_0_exits(bases, tmp_path_factory):
    return _fault_run(
        bases, tmp_path_factory, "rank0_exits", fault_rank=0, fault="exit", timeout=_GROUP_TIMEOUT
    )


@pytest.fixture(scope="module")
def rank_1_goes_silent(bases, tmp_path_factory):
    return _fault_run(
        bases,
        tmp_path_factory,
        "rank1_silent",
        fault_rank=1,
        fault="hang",
        timeout=_SILENT_TIMEOUT,
    )


class TestPeerDeath:
    @staticmethod
    def _survivor_ends_the_run(outcome, dead_tag, timeout):
        alive_tag = _TAGS[1 - _TAGS.index(dead_tag)]
        dead, alive = outcome.runs[dead_tag], outcome.runs[alive_tag]
        assert not outcome.hung, f"the survivor hung\n{_report(outcome)}"
        assert dead.marker_at is not None, f"the fault never fired\n{_report(outcome)}"
        assert alive.code not in (None, 0), f"the survivor did not fail\n{_report(outcome)}"
        waited = alive.exited_at - dead.marker_at
        assert waited >= -1, f"the survivor ended before the fault ({waited:.1f} s)"
        assert waited <= timeout + _SLACK, f"{waited:.1f} s after the fault\n{_report(outcome)}"
        assert "Traceback" in alive.log and _PEER_GONE.search(alive.log), alive.log[-2000:]
        return alive, waited

    def test_rank_0_ends_the_run_when_rank_1_exits(self, rank_1_exits):
        self._survivor_ends_the_run(rank_1_exits, "dp1", _GROUP_TIMEOUT)
        assert rank_1_exits.runs["dp1"].code == 1

    def test_rank_1_ends_the_run_when_rank_0_exits(self, rank_0_exits):
        self._survivor_ends_the_run(rank_0_exits, "dp0", _GROUP_TIMEOUT)
        assert rank_0_exits.runs["dp0"].code == 1

    def test_a_peer_that_goes_silent_is_cut_off_at_the_shortened_group_timeout(
        self, rank_1_goes_silent
    ):
        """The case the timeout exists for. Without C8 the group keeps the trainer's 1800 s
        and this run hits the deadline instead."""
        alive, _waited = self._survivor_ends_the_run(rank_1_goes_silent, "dp1", _SILENT_TIMEOUT)
        assert re.search(r"(?i)timed out|timeout", alive.log), alive.log[-2000:]
