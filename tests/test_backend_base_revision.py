"""Streamed base pins and caches, with unchanged Unsloth mapping as a control."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soup_cli.utils import run_provenance as provenance

BASE = "owner/base"
SHA = "a" * 40


@pytest.fixture
def moving_hub(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("UNSLOTH_USE_MODELSCOPE", raising=False)
    probe = Mock(side_effect=[SimpleNamespace(sha=SHA), SimpleNamespace(sha="b" * 40)])
    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(model_info=probe))
    return probe


def test_stream_setup_pins_tokenizer_and_architecture_config(monkeypatch, moving_hub):
    from transformers import AutoConfig, AutoTokenizer

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper

    calls = []

    def tokenizer(model_id, *, revision=None, **kwargs):
        assert model_id == BASE and revision == SHA
        calls.append("tokenizer")
        return SimpleNamespace(pad_token=None, eos_token="eos")

    def config(model_id, *, revision=None, **kwargs):
        assert model_id == BASE and revision == SHA
        calls.append("config")
        return object()

    class AfterConfigError(Exception):
        pass

    def stop_before_allocation(config):
        raise AfterConfigError

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(AutoConfig, "from_pretrained", config)
    monkeypatch.setattr("soup_cli.utils.layer_stream.stream_arch_of", stop_before_allocation)
    cfg = SoupConfig(base=BASE, data={"train": "unused.jsonl"},
                     training={"quantization": "none", "stream_layers": True, "batch_size": 1})
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    identity = provenance.BaseRevision(BASE)
    with provenance.base_revision_scope(identity):
        with pytest.raises(AfterConfigError):
            wrapper._setup_streaming_transformers(cfg, cfg.training)
    assert identity.revision == SHA
    # The model config is read first (#2107: a refused Kimi K2 config is reported before
    # the tokenizer load), and both reads are pinned to the one bound commit.
    assert calls == ["config", "tokenizer"]
    moving_hub.assert_called_once()


@pytest.mark.parametrize("second_call", [False, True])
def test_stream_skeleton_and_weight_downloads_share_one_commit(
    tmp_path, monkeypatch, moving_hub, second_call,
):
    from transformers import AutoConfig

    from soup_cli.utils import hubs, spectrum_scan
    from soup_cli.utils.layer_shard import resolve_shard_dir, shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton
    from tests._tiny_hf_models import tiny_model_dir

    local = tiny_model_dir("llama")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    # An index with a nonstandard weight spelling exercises the second fetch too.
    if second_call:
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "part.safetensors"}}), encoding="utf-8",
        )
    calls = []

    def download(model_id, *, revision=None, **kwargs):
        assert model_id == BASE
        assert revision == SHA, "a download must not follow the moving default branch"
        calls.append(kwargs)
        if not second_call or len(calls) == 2:
            import shutil

            shutil.copyfile(Path(local) / "model.safetensors",
                            snapshot / ("part.safetensors" if second_call else "model.safetensors"))
        return str(snapshot)

    original_config = AutoConfig.from_pretrained
    configs = []

    def config(model_id, **kwargs):
        assert model_id == BASE
        assert kwargs.pop("revision", None) == SHA
        configs.append(model_id)
        return original_config(local, **kwargs)

    monkeypatch.setattr(hubs, "snapshot_download", download)
    monkeypatch.setattr(AutoConfig, "from_pretrained", config)
    identity = provenance.BaseRevision(BASE)
    with provenance.base_revision_scope(identity):
        # Probe and final skeleton both read the same source config.
        for _ in range(2):
            skeleton = build_meta_skeleton(BASE, dtype="float32")
            assert skeleton.name_or_path == BASE
            assert all(parameter.is_meta for parameter in skeleton.parameters())
        weights = spectrum_scan.resolve_model_weights(BASE)
        shards = resolve_shard_dir(BASE, str(tmp_path / "shards"))
        index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    assert index is not None and os.path.isdir(shards)
    assert configs == [BASE, BASE]
    assert len(calls) == (2 if second_call else 1)
    assert moving_hub.call_count == 1
    assert identity.metadata("run")["base_revision"] == SHA


def test_shard_slots_separate_commits_and_leave_other_models_unchanged(tmp_path, moving_hub):
    from soup_cli.utils.layer_shard import resolve_shard_dir

    cache = str(tmp_path / "cache")
    unscoped = resolve_shard_dir(BASE, cache)
    other = resolve_shard_dir("owner/teacher", cache)
    slots = []
    for sha in (SHA, "b" * 40):
        with provenance.base_revision_scope(provenance.BaseRevision(BASE)):
            slot = resolve_shard_dir(BASE, cache)
            assert Path(slot).name.endswith("@" + sha)
            assert resolve_shard_dir(BASE, cache) == slot
            assert resolve_shard_dir("owner/teacher", cache) == other
            assert os.path.commonpath([os.path.realpath(slot), os.path.realpath(cache)]) == cache
            slots.append(slot)
    assert len(set([unscoped, *slots])) == 3
    assert moving_hub.call_count == 2


def test_materialized_weight_slots_separate_pinned_commits(tmp_path, monkeypatch, moving_hub):
    from soup_cli.utils import spectrum_scan

    monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "snapshot"
    source.mkdir()
    slots = []
    # Use the planner directly: no Windows symlink privilege is required.
    for sha in (SHA, "b" * 40):
        with provenance.base_revision_scope(provenance.BaseRevision(BASE)):
            plan = spectrum_scan._hf_snapshot_plan(BASE, str(source), (), sha)
            assert Path(plan.weights_dir).name.endswith("@" + sha)
            slots.append(plan.weights_dir)
    assert slots[0] != slots[1]
    assert moving_hub.call_count == 2


@pytest.mark.parametrize("quantization", ["none", "4bit"])
def test_unsloth_keeps_native_mapping_without_a_revision_override(
    monkeypatch, moving_hub, quantization,
):
    from soup_cli.utils.unsloth import load_model_and_tokenizer

    calls = []
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path=BASE, _commit_hash=SHA))
    tokenizer = object()

    def load(*, model_name, revision=None, use_exact_model_name=False, **kwargs):
        # This receiver is an API-contract fixture, not an installed Unsloth run.
        # Native quantized mapping must remain enabled for official model IDs.
        assert model_name == BASE and use_exact_model_name is False
        assert revision is None
        assert kwargs["load_in_4bit"] is (quantization == "4bit")
        calls.append(model_name + "-bnb-4bit" if kwargs["load_in_4bit"] else model_name)
        return model, tokenizer

    peft = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=SimpleNamespace(
        from_pretrained=load, get_peft_model=peft,
    )))
    identity = provenance.BaseRevision(BASE)
    with provenance.base_revision_scope(identity):
        assert load_model_and_tokenizer(BASE, 64, quantization=quantization) == (model, tokenizer)
    assert calls == [BASE + "-bnb-4bit" if quantization == "4bit" else BASE]
    assert peft.call_args.kwargs["use_gradient_checkpointing"] == "unsloth"
    moving_hub.assert_not_called()
    assert identity.revision is None


def test_local_stream_cache_keeps_unverified_path_hint(tmp_path, monkeypatch, moving_hub):
    from soup_cli.utils.layer_shard import resolve_shard_dir

    local = tmp_path / "snapshots" / SHA
    local.mkdir(parents=True)
    base = str(local)
    cache = str(tmp_path / "cache")
    expected = resolve_shard_dir(base, cache)
    identity = provenance.BaseRevision(base)
    with provenance.base_revision_scope(identity):
        assert provenance.base_revision_kwargs(base) == {}
        assert resolve_shard_dir(base, cache) == expected
    assert identity.revision == SHA and identity.source == "cache_path"
    moving_hub.assert_not_called()


@pytest.mark.parametrize("mode", ["offline", "failed"])
def test_unknown_stream_weights_keep_unpinned_download_and_cache(
    tmp_path, monkeypatch, moving_hub, mode,
):
    from soup_cli.utils import hubs, spectrum_scan
    from soup_cli.utils.layer_shard import resolve_shard_dir

    if mode == "offline":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:
        moving_hub.side_effect = OSError("metadata unavailable")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"existing local weight fixture")
    calls = []

    def download(model_id, **kwargs):
        assert model_id == BASE
        assert "revision" not in kwargs
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(hubs, "snapshot_download", download)
    cache = str(tmp_path / "cache")
    original_slot = resolve_shard_dir(BASE, cache)
    identity = provenance.BaseRevision(BASE)
    with provenance.base_revision_scope(identity):
        assert spectrum_scan.resolve_model_weights(BASE) == str(snapshot)
        assert resolve_shard_dir(BASE, cache) == original_slot
    assert len(calls) == 1
    assert identity.revision is None
    assert identity.source == ("offline" if mode == "offline" else "resolution_failed")
    assert moving_hub.call_count == (0 if mode == "offline" else 1)
