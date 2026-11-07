"""The RL checkpoint manifest reports what a save wrote (#808, item 2).

`training.rl_checkpoint_include_ref_model` and
`training.rl_checkpoint_include_rollout_buffer` are accepted, but no save
writes reference-model or rollout-buffer state. The manifest used to copy both
flags from the config, so a checkpoint holding only the adapter and the
optimizer claimed `has_ref_model: true, has_rollout_buffer: true`.
"""

from __future__ import annotations

import json
import os

import pytest


def _optimizer():
    import torch

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    return opt


def _save(tcfg, optimizer):
    """Build the callback the trainers build from a `training:` block, and save once."""
    from soup_cli.utils.peft_wiring import _build_rl_checkpoint_cb

    cb = _build_rl_checkpoint_cb(tcfg, output_dir="run", task="grpo")
    ckpt_dir = cb.save_checkpoint(step=1, model=None, optimizer=optimizer)
    with open(os.path.join(ckpt_dir, "manifest.json"), encoding="utf-8") as fh:
        return sorted(os.listdir(ckpt_dir)), json.load(fh)


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the checkpoint root must stay under cwd
    return tmp_path


def test_both_flags_true_save_neither_state_and_the_manifest_says_so(cwd):
    from soup_cli.config.schema import TrainingConfig

    tcfg = TrainingConfig(
        rl_checkpoint_save_every_steps=1,
        rl_checkpoint_include_ref_model=True,
        rl_checkpoint_include_rollout_buffer=True,
    )

    files, manifest = _save(tcfg, _optimizer())

    assert files == ["manifest.json", "optimizer.pt"], files
    assert manifest["has_ref_model"] is False, manifest
    assert manifest["has_rollout_buffer"] is False, manifest


@pytest.mark.parametrize("ref, buf", [(True, False), (False, True)])
def test_each_flag_alone_is_reported_false_too(cwd, ref, buf):
    from soup_cli.config.schema import TrainingConfig

    tcfg = TrainingConfig(
        rl_checkpoint_save_every_steps=1,
        rl_checkpoint_include_ref_model=ref,
        rl_checkpoint_include_rollout_buffer=buf,
    )

    _files, manifest = _save(tcfg, _optimizer())

    assert (manifest["has_ref_model"], manifest["has_rollout_buffer"]) == (False, False)


def test_has_optimizer_still_follows_the_optimizer_save(cwd):
    """Control: the one flag that was already honest is unchanged."""
    from soup_cli.config.schema import TrainingConfig

    saved_files, saved = _save(
        TrainingConfig(rl_checkpoint_save_every_steps=1), _optimizer()
    )
    assert "optimizer.pt" in saved_files
    assert saved["has_optimizer"] is True

    from soup_cli.utils.peft_wiring import _build_rl_checkpoint_cb

    cb = _build_rl_checkpoint_cb(
        TrainingConfig(
            rl_checkpoint_save_every_steps=1, rl_checkpoint_include_optimizer=False
        ),
        output_dir="run-no-opt",
        task="grpo",
    )
    ckpt_dir = cb.save_checkpoint(step=1, model=None, optimizer=_optimizer())
    with open(os.path.join(ckpt_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert "optimizer.pt" not in os.listdir(ckpt_dir)
    assert manifest["has_optimizer"] is False



@pytest.mark.parametrize(("ref", "buf"), [(True, True), (True, False), (False, True)])
def test_the_manifest_ignores_include_flags_set_on_the_callback_config(cwd, ref, buf):
    """Set directly on RLCheckpointConfig, the flags still save nothing,
    so the manifest says false."""
    from soup_cli.utils.rl_checkpoint import RLCheckpointConfig, build_rl_checkpoint_callback

    cfg = RLCheckpointConfig(
        save_every_steps=1,
        include_optimizer_state=True,
        include_ref_model=ref,
        include_rollout_buffer=buf,
    )
    cb = build_rl_checkpoint_callback(cfg, output_dir="run", task="grpo")
    ckpt_dir = cb.save_checkpoint(step=1, model=None, optimizer=_optimizer())
    with open(os.path.join(ckpt_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert sorted(os.listdir(ckpt_dir)) == ["manifest.json", "optimizer.pt"]
    assert manifest["has_ref_model"] is False
    assert manifest["has_rollout_buffer"] is False
    assert manifest["has_optimizer"] is True
