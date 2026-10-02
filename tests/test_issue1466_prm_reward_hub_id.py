"""#1466, the ``prm_reward`` item: a Hub id is resolved before training, not refused after it.

``training.prm_reward`` is documented as "Path (or HF id)". A Hub id parsed and built,
but the PRM's reward head was read only from a local directory, so the GRPO run died at
the first reward call, after both ``from_pretrained`` calls, with "path is not a
directory". The id is now downloaded to a local snapshot when the reward is built, the
head is read from it (and refused there, before training, when the repo is not a
Soup-trained PRM), and a value that is neither an existing directory nor a repo id is
refused by name.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import prm_reward

_HUB_ID = "my-org/soup-prm-135m"


def _tcfg(value: str):
    cfg = load_config_from_string(
        "base: HuggingFaceTB/SmolLM2-135M-Instruct\ntask: grpo\n"
        "data:\n  train: p.jsonl\n  format: chatml\n"
        f"training:\n  prm_reward: {value}\n  num_generations: 2\n"
        "output: out\n"
    )
    return cfg.training


def _prm_dir(path: Path, *, with_head: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    tensors = {"model.embed_tokens.weight": torch.zeros(4, 8)}
    if with_head:
        tensors["reward_head.weight"] = torch.ones(1, 8)
        tensors["reward_head.bias"] = torch.zeros(1)
    save_file(tensors, str(path / "model.safetensors"))
    return path


@pytest.fixture
def no_hub_probe(monkeypatch: pytest.MonkeyPatch):
    # The trust-remote-code probe would ask the Hub; the issue's reproduction stubs it too.
    monkeypatch.setattr(prm_reward, "_resolve_trust", lambda base, requested, console: False)


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Stub ``huggingface_hub.snapshot_download``: record the call, hand back a prepared dir."""
    import huggingface_hub

    calls: list[dict] = []
    target = _prm_dir(tmp_path / "snapshot")

    def fake(repo_id: str, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        return str(target)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return calls, target


class TestHubIdResolution:
    def test_a_hub_id_is_downloaded_and_the_head_read_from_the_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_hub_probe, snapshot
    ) -> None:
        calls, target = snapshot
        monkeypatch.chdir(tmp_path)
        scorer = prm_reward.build_prm_reward_fn(_tcfg(_HUB_ID), "cpu", False)
        assert Path(scorer.prm_path) == target
        assert calls[0]["repo_id"] == _HUB_ID and calls[0].get("revision") is None
        head = prm_reward.load_reward_head_weights(scorer.prm_path)
        assert set(head) == {"weight", "bias"}

    def test_an_at_revision_spelling_becomes_the_revision_argument(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_hub_probe, snapshot
    ) -> None:
        calls, _ = snapshot
        monkeypatch.chdir(tmp_path)
        prm_reward.build_prm_reward_fn(_tcfg(f"{_HUB_ID}@main"), "cpu", False)
        assert calls[0]["repo_id"] == _HUB_ID and calls[0]["revision"] == "main"

    def test_only_weights_and_tokenizer_files_are_requested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_hub_probe, snapshot
    ) -> None:
        calls, _ = snapshot
        monkeypatch.chdir(tmp_path)
        prm_reward.build_prm_reward_fn(_tcfg(_HUB_ID), "cpu", False)
        patterns = calls[0]["allow_patterns"]
        assert "*.safetensors" in patterns and "*.json" in patterns
        assert "*.py" in patterns  # a trust_remote_code base ships its modules as .py
        assert not any(p.endswith(".bin") or p.endswith(".pt") for p in patterns)

    def test_a_hub_repo_without_a_reward_head_is_refused_at_build_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_hub_probe
    ) -> None:
        import huggingface_hub

        target = _prm_dir(tmp_path / "plain-base", with_head=False)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **kw: str(target))
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="not a Soup-trained PRM"):
            prm_reward.build_prm_reward_fn(_tcfg(_HUB_ID), "cpu", False)

    def test_a_download_failure_is_a_refusal_naming_the_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_hub_probe
    ) -> None:
        import huggingface_hub

        def offline(repo_id: str, **kw):
            raise OSError("offline")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="my-org/soup-prm-135m") as info:
            prm_reward.build_prm_reward_fn(_tcfg(_HUB_ID), "cpu", False)
        assert "offline" in str(info.value)

    def test_neither_a_directory_nor_a_repo_id_is_refused_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_hub_probe
    ) -> None:
        import huggingface_hub

        def never(repo_id: str, **kw):
            raise AssertionError("the Hub was queried for a path-looking value")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", never)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="missing-prm") as info:
            prm_reward.build_prm_reward_fn(_tcfg("./missing-prm"), "cpu", False)
        # the by-name refusal, not a failed download attempt dressed as one
        assert "neither an existing directory nor a Hub repo id" in str(info.value)


class TestLocalPathsUnchanged:
    def test_an_existing_local_prm_is_used_in_place_and_the_hub_is_not_asked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_hub_probe
    ) -> None:
        import huggingface_hub

        def never(repo_id: str, **kw):
            raise AssertionError("snapshot_download called for a local directory")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", never)
        monkeypatch.chdir(tmp_path)
        local = _prm_dir(tmp_path / "my-prm")
        scorer = prm_reward.build_prm_reward_fn(_tcfg("./my-prm"), "cpu", False)
        assert os.path.realpath(scorer.prm_path) == os.path.realpath(local)

    def test_a_local_directory_without_a_head_is_refused_before_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_hub_probe
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _prm_dir(tmp_path / "base-only", with_head=False)
        with pytest.raises(ValueError, match="not a Soup-trained PRM"):
            prm_reward.build_prm_reward_fn(_tcfg("./base-only"), "cpu", False)
