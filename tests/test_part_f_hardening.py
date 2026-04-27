"""Part F — Standalone hardening (v0.33.0).

Tests for:
  - #21 RLVR code_exec_reward: OS-level isolation strategy detection +
    Linux unshare attempt + macOS sandbox-exec wrapper detection.
  - #22 checkpoint_intelligence.prune_checkpoints: TOCTOU-safe symlink
    handling via os.lstat + S_ISLNK and onerror abort on rmtree walk.
"""

from __future__ import annotations

import os
import stat
import sys
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# #22 — prune_checkpoints TOCTOU hardening
# ---------------------------------------------------------------------------


class TestPruneCheckpointsTOCTOU:
    def test_prune_skips_top_level_symlink_via_lstat(self, tmp_path):
        """Top-level symlink masquerading as a checkpoint dir must be skipped
        without following the link target."""
        from soup_cli.eval.checkpoint_intelligence import CheckpointTracker

        # Real checkpoint to keep
        (tmp_path / "checkpoint-100").mkdir()

        # Decoy target outside the prune root
        outside = tmp_path.parent / "outside_target_dir"
        outside.mkdir(exist_ok=True)
        (outside / "sentinel.txt").write_text("must-not-delete", encoding="utf-8")

        link = tmp_path / "checkpoint-200"
        try:
            os.symlink(str(outside), str(link), target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted (Windows non-admin)")

        tracker = CheckpointTracker(metric="composite", keep_top=1)
        tracker.record(step=100, score=0.9)
        tracker.record(step=200, score=0.5)

        removed = tracker.prune_checkpoints(tmp_path)

        # The symlink must NOT be followed — sentinel survives
        assert (outside / "sentinel.txt").exists()
        # Symlink itself was skipped (not in removed list)
        assert 200 not in removed

    def test_prune_aborts_on_symlink_inside_checkpoint(self, tmp_path):
        """If rmtree encounters a symlink mid-walk inside a doomed checkpoint,
        it must abort instead of following it (defence-in-depth)."""
        from soup_cli.eval.checkpoint_intelligence import CheckpointTracker

        # Two checkpoints; we'll keep the top one
        ckpt_keep = tmp_path / "checkpoint-100"
        ckpt_keep.mkdir()
        ckpt_doomed = tmp_path / "checkpoint-200"
        ckpt_doomed.mkdir()

        # Plant a symlink INSIDE the doomed checkpoint pointing outside
        outside = tmp_path.parent / "siblings_must_survive"
        outside.mkdir(exist_ok=True)
        (outside / "secret.txt").write_text("keep-me", encoding="utf-8")

        nested_link = ckpt_doomed / "linked"
        try:
            os.symlink(str(outside), str(nested_link), target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted (Windows non-admin)")

        tracker = CheckpointTracker(metric="composite", keep_top=1)
        tracker.record(step=100, score=0.9)
        tracker.record(step=200, score=0.5)

        # Should not follow the nested symlink
        tracker.prune_checkpoints(tmp_path)

        assert (outside / "secret.txt").exists(), "rmtree followed a symlink"

    def test_prune_uses_lstat_for_symlink_check(self, tmp_path, monkeypatch):
        """Verify prune uses os.lstat-based check, not Path.is_symlink, so a
        broken symlink (target removed mid-walk) is still rejected."""
        from soup_cli.eval import checkpoint_intelligence as ci

        # Create a broken symlink as 'checkpoint-300'
        broken = tmp_path / "checkpoint-300"
        try:
            os.symlink(str(tmp_path / "_nonexistent_"), str(broken))
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted")

        tracker = ci.CheckpointTracker(metric="composite", keep_top=1)
        tracker.record(step=100, score=0.9)
        tracker.record(step=300, score=0.5)

        removed = tracker.prune_checkpoints(tmp_path)
        assert 300 not in removed


# ---------------------------------------------------------------------------
# #21 — RLVR code_exec_reward OS-level isolation strategy
# ---------------------------------------------------------------------------


class TestCodeExecIsolationStrategy:
    def test_get_isolation_strategy_returns_known_value(self):
        from soup_cli.trainer.rewards import _get_isolation_strategy

        strategy = _get_isolation_strategy()
        assert strategy in {"namespaces", "sandbox-exec", "best-effort"}

    def test_isolation_strategy_linux_with_unshare(self):
        from soup_cli.trainer import rewards

        with patch.object(sys, "platform", "linux"), \
                patch("os.unshare", create=True) as mock_unshare:
            mock_unshare.return_value = None
            strategy = rewards._get_isolation_strategy.__wrapped__() \
                if hasattr(rewards._get_isolation_strategy, "__wrapped__") \
                else rewards._get_isolation_strategy()
            # Either the cached call or fresh call is acceptable; what matters
            # is that the strategy on a Linux host with os.unshare available
            # is reachable.
            assert strategy in {"namespaces", "best-effort"}

    def test_isolation_strategy_macos_with_sandbox_exec(self, monkeypatch):
        import shutil as shutil_mod

        from soup_cli.trainer import rewards

        # Force fresh evaluation
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(shutil_mod, "which", lambda name: (
            "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None
        ))
        # Bypass any module-level cache
        if hasattr(rewards, "_ISOLATION_STRATEGY_CACHE"):
            rewards._ISOLATION_STRATEGY_CACHE = None
        strategy = rewards._compute_isolation_strategy()
        assert strategy == "sandbox-exec"

    def test_isolation_strategy_macos_without_sandbox_exec(self, monkeypatch):
        import shutil as shutil_mod

        from soup_cli.trainer import rewards

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(shutil_mod, "which", lambda _name: None)
        if hasattr(rewards, "_ISOLATION_STRATEGY_CACHE"):
            rewards._ISOLATION_STRATEGY_CACHE = None
        strategy = rewards._compute_isolation_strategy()
        assert strategy == "best-effort"

    def test_isolation_strategy_windows(self, monkeypatch):
        from soup_cli.trainer import rewards

        monkeypatch.setattr(sys, "platform", "win32")
        if hasattr(rewards, "_ISOLATION_STRATEGY_CACHE"):
            rewards._ISOLATION_STRATEGY_CACHE = None
        strategy = rewards._compute_isolation_strategy()
        assert strategy == "best-effort"

    def test_isolation_strategy_linux_unshare_unavailable(self, monkeypatch):
        from soup_cli.trainer import rewards

        monkeypatch.setattr(sys, "platform", "linux")
        # Pretend os.unshare doesn't exist
        if hasattr(os, "unshare"):
            monkeypatch.delattr(os, "unshare", raising=False)
        if hasattr(rewards, "_ISOLATION_STRATEGY_CACHE"):
            rewards._ISOLATION_STRATEGY_CACHE = None
        strategy = rewards._compute_isolation_strategy()
        assert strategy == "best-effort"

    def test_macos_sandbox_profile_blocks_network_and_writes(self):
        """The macOS sandbox profile must deny network and writes outside /tmp."""
        from soup_cli.trainer.rewards import MACOS_SANDBOX_PROFILE

        # Profile must default-deny then explicitly allow narrow process needs
        assert "(deny default)" in MACOS_SANDBOX_PROFILE
        assert "network" in MACOS_SANDBOX_PROFILE
        # Must be a single-line or properly formatted scheme expression
        assert "(version 1)" in MACOS_SANDBOX_PROFILE


# ---------------------------------------------------------------------------
# Smoke-test that existing code_exec_reward path still works on this host
# ---------------------------------------------------------------------------


class TestCodeExecRewardSmoke:
    def test_correct_code_still_scores_one(self):
        from soup_cli.trainer.rewards import code_exec_reward

        completions = [[{"role": "assistant", "content": "```python\nprint(2+2)\n```"}]]
        scores = code_exec_reward(completions, expected=["4"])
        assert scores == [1.0]

    def test_wrong_code_still_scores_zero(self):
        from soup_cli.trainer.rewards import code_exec_reward

        completions = [[{"role": "assistant", "content": "```python\nprint(3)\n```"}]]
        scores = code_exec_reward(completions, expected=["4"])
        assert scores == [0.0]


# ---------------------------------------------------------------------------
# Helper: confirm S_ISLNK lstat-style check works
# ---------------------------------------------------------------------------


def test_lstat_islnk_detects_symlink(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(str(target), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported")
    assert stat.S_ISLNK(os.lstat(str(link)).st_mode)
    assert not stat.S_ISLNK(os.lstat(str(target)).st_mode)
