"""#1450 — a failed replace must never delete the last good copy.

Two call sites used to remove the old artifact before the new one was known
to be valid and in place:

* ``merge_adapter_to_dense`` (shared by ``soup shrink`` and ``soup draft
  distill``) used to ``shutil.rmtree(out_dir)`` then ``os.replace(staging,
  out_dir)``; its ``except BaseException`` handler deleted ``staging`` too, so
  a failure between those two calls left neither copy. It now renames
  ``out_dir`` aside instead of deleting it, restores that backup if the swap
  fails, and — if the restore itself also fails — keeps both the backup and
  the staged model on disk rather than guessing which one to delete.
* ``soup reward synth -o reward.py --force`` overwrote the existing verifier
  with the candidate BEFORE calibrating it, then ``_restore_or_cleanup``
  deleted the file on refusal or error — destroying the user's previous,
  working verifier. It now moves the previous verifier to a ``.tmp`` sibling
  first and restores it on any failure, including a failed candidate write.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# reward.py synth — --force must not destroy the previous verifier
# ---------------------------------------------------------------------------
class TestRewardSynthForceKeepsPreviousOnFailure:
    def _runner(self):
        from typer.testing import CliRunner
        return CliRunner()

    def test_refusal_leaves_previous_verifier_byte_identical(self, tmp_path, monkeypatch):
        from soup_cli.commands.reward import app

        monkeypatch.chdir(tmp_path)
        previous = (
            "# my working verifier\n"
            "def reward_fn(completions, **kw):\n"
            "    return [1.0 for _ in completions]\n"
        )
        Path("reward.py").write_text(previous, encoding="utf-8")

        # A huge tolerance makes every reference AND every perturbed negative
        # (gold + 9999) accept -> discrimination below the default floor ->
        # REFUSED (exit 2). This must not touch the existing --force target.
        _write_jsonl(Path("refs.jsonl"), [{"answer": str(i)} for i in range(4)])
        res = self._runner().invoke(app, [
            "synth", "refs.jsonl", "-o", "reward.py",
            "--force", "--tolerance", "10000",
        ])

        assert res.exit_code == 2, (res.output, repr(res.exception))
        assert Path("reward.py").exists(), "the previous verifier was deleted on refusal"
        assert Path("reward.py").read_text(encoding="utf-8") == previous, (
            "the previous verifier's bytes changed on a refused --force replace"
        )

    def test_calibration_error_leaves_previous_verifier_byte_identical(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.commands.reward import app

        monkeypatch.chdir(tmp_path)
        previous = "# my working verifier\ndef reward_fn(completions, **kw):\n    return [1.0]\n"
        Path("reward.py").write_text(previous, encoding="utf-8")

        # Make calibrate() raise so the command takes the "calibration failed"
        # (exit 1) branch instead of the "refused" (exit 2) branch — both
        # branches call _restore_or_cleanup(output, backup_output).
        from soup_cli.utils import reward_synth as rs

        def _boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(rs, "calibrate", _boom)

        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])
        res = self._runner().invoke(app, [
            "synth", "refs.jsonl", "-o", "reward.py", "--force",
        ])

        assert res.exit_code == 1, (res.output, repr(res.exception))
        assert Path("reward.py").exists(), (
            "the previous verifier was deleted on a calibration error"
        )
        assert Path("reward.py").read_text(encoding="utf-8") == previous

    def test_candidate_write_failure_leaves_previous_verifier_and_no_orphan_backup(
        self, tmp_path, monkeypatch
    ):
        """``atomic_write_text`` runs after the previous verifier is already
        moved aside to its ``.soup.reward-prev.*.tmp`` backup. If the write
        itself fails (disk full, a permission error), that call used to sit
        outside any guard: ``output`` stayed missing and the backup was
        never restored, so a write failure destroyed the previous verifier
        AND orphaned the backup file."""
        from soup_cli.commands.reward import app

        monkeypatch.chdir(tmp_path)
        previous = "# my working verifier\ndef reward_fn(completions, **kw):\n    return [1.0]\n"
        Path("reward.py").write_text(previous, encoding="utf-8")

        import soup_cli.commands.reward as reward_cmd

        def _boom(*a, **kw):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(reward_cmd, "atomic_write_text", _boom)

        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])
        res = self._runner().invoke(app, [
            "synth", "refs.jsonl", "-o", "reward.py", "--force",
        ])

        assert res.exit_code == 1, (res.output, repr(res.exception))
        assert Path("reward.py").exists(), (
            "the previous verifier was deleted on a failed candidate write"
        )
        assert Path("reward.py").read_text(encoding="utf-8") == previous
        leftover = list(tmp_path.glob(".soup.reward-prev.*.tmp"))
        assert not leftover, f"orphaned backup file(s) left behind: {leftover}"

    def test_accepted_replace_still_writes_the_new_verifier(self, tmp_path, monkeypatch):
        """The fix must not break the ordinary accepted path."""
        from soup_cli.commands.reward import app

        monkeypatch.chdir(tmp_path)
        Path("reward.py").write_text("# stale\n", encoding="utf-8")
        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])

        res = self._runner().invoke(app, [
            "synth", "refs.jsonl", "-o", "reward.py", "--force",
        ])

        assert res.exit_code == 0, (res.output, repr(res.exception))
        new_source = Path("reward.py").read_text(encoding="utf-8")
        assert "stale" not in new_source
        assert "reward_fn" in new_source


# ---------------------------------------------------------------------------
# adapter_fuse.merge_adapter_to_dense — a failed swap must not delete out_dir
# ---------------------------------------------------------------------------
class _Merged:
    def save_pretrained(self, path):
        Path(path).joinpath("model.safetensors").write_bytes(b"NEW")


class _Base:
    @staticmethod
    def from_pretrained(*a, **kw):
        return _Base()


class _Peft:
    @staticmethod
    def from_pretrained(*a, **kw):
        obj = _Peft()
        obj.merge_and_unload = lambda: _Merged()  # noqa: E731
        return obj


class _Tok:
    @staticmethod
    def from_pretrained(*a, **kw):
        return _Tok()

    def save_pretrained(self, path):
        Path(path).joinpath("tokenizer.json").write_bytes(b"NEW")


def _fake_train_stack(monkeypatch):
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModelForCausalLM = _Base
    fake_tf.AutoTokenizer = _Tok
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = _Peft
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)


def _fresh_adapter_and_out(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (out / name).write_bytes(b"OLD")
    return adapter, out


class TestMergeAdapterToDenseSurvivesAFailedSwap:
    def test_out_dir_survives_when_os_replace_raises(self, tmp_path, monkeypatch):
        """A real-world failure: an antivirus scanner or search indexer holds
        the staging dir just long enough that the final os.replace raises.
        Before the fix, the destination was already rmtree'd by then, and the
        except-handler deleted the staging copy too — both copies gone."""
        from soup_cli.utils import adapter_fuse

        monkeypatch.chdir(tmp_path)
        _fake_train_stack(monkeypatch)
        adapter, out = _fresh_adapter_and_out(tmp_path)

        real_replace = adapter_fuse.os.replace

        def _replace_fails(src, dst, *a, **kw):
            if os.path.basename(src).startswith(".fuse_"):
                raise PermissionError(5, "Access is denied")
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(adapter_fuse.os, "replace", _replace_fails)

        with pytest.raises(PermissionError):
            adapter_fuse.merge_adapter_to_dense(
                base_model="base", adapter_dir=str(adapter), out_dir="out"
            )

        # The swap failed but the rename-back succeeded (only os.replace of a
        # ``.fuse_*`` source was poisoned above): out/ must hold the ORIGINAL
        # bytes, restored, with nothing left behind — no leftover staging
        # dir and no leftover ``.old-`` backup. A test that merely asked for
        # "old survives OR a copy landed somewhere" would still pass even if
        # the restore-on-failure block were deleted outright (the untouched
        # staging dir alone would satisfy it), so both conditions are
        # asserted together.
        assert (out / "config.json").read_bytes() == b"OLD"
        assert (out / "model.safetensors").read_bytes() == b"OLD"
        assert (out / "tokenizer.json").read_bytes() == b"OLD"
        leftover = list(tmp_path.glob(".fuse_*")) + list(tmp_path.glob(".out.old-*"))
        assert not leftover, f"orphaned staging/backup dir(s) left behind: {leftover}"

    def test_out_dir_survives_when_rmtree_is_reintroduced_and_fails_partway(
        self, tmp_path, monkeypatch
    ):
        """Regression guard for the original bug shape: the pre-fix code
        called ``shutil.rmtree(out_dir)`` directly, and a version of that
        call which fails after deleting only SOME files left ``out_dir``
        half-destroyed with no backup to restore. The fixed implementation
        must never call ``shutil.rmtree`` on ``out_dir`` at all — it renames
        it aside instead — so poisoning rmtree for that exact path must have
        no effect on an otherwise-successful merge."""
        from soup_cli.utils import adapter_fuse

        monkeypatch.chdir(tmp_path)
        _fake_train_stack(monkeypatch)
        adapter, out = _fresh_adapter_and_out(tmp_path)
        out_real = str(out.resolve())

        real_rmtree = shutil.rmtree
        poisoned_calls = []

        def _rmtree_partial_then_raise(path, *a, **kw):
            if os.path.realpath(path) == out_real:
                poisoned_calls.append(path)
                # Simulate the old failure mode: delete one file, then blow up
                # before the rest (and before any replacement is in place).
                victim = os.path.join(path, "config.json")
                if os.path.exists(victim):
                    os.remove(victim)
                raise OSError(13, "Permission denied (partial rmtree)")
            return real_rmtree(path, *a, **kw)

        # adapter_fuse does ``import shutil`` locally inside the function body
        # (heavy imports stay function-local there); that binds the same
        # cached module object, so patching the shutil module directly here
        # reaches it.
        monkeypatch.setattr(shutil, "rmtree", _rmtree_partial_then_raise)

        adapter_fuse.merge_adapter_to_dense(
            base_model="base", adapter_dir=str(adapter), out_dir="out"
        )

        assert not poisoned_calls, (
            "merge_adapter_to_dense called shutil.rmtree(out_dir) directly — "
            "the destination must be renamed aside, never rmtree'd"
        )
        # The fake save_pretrained()s only write model.safetensors /
        # tokenizer.json (no config.json) — a successful merge replaces
        # out_dir with exactly the staged content, so config.json from the
        # old model is gone and must not be asserted here.
        assert (out / "model.safetensors").read_bytes() == b"NEW"
        assert (out / "tokenizer.json").read_bytes() == b"NEW"
        leftover = list(tmp_path.glob(".fuse_*")) + list(tmp_path.glob(".out.old-*"))
        assert not leftover, f"orphaned staging/backup dir(s) left behind: {leftover}"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only file-lock semantics")
    def test_out_dir_survives_an_open_handle_on_model_safetensors(self, tmp_path, monkeypatch):
        """Windows-only: another process (a viewer, a server, an antivirus
        scan) holds ``out_dir/model.safetensors`` open. The rename-aside or
        the final swap can then raise ``PermissionError`` — either way, the
        guarantee is the same as every other failure mode here: out_dir ends
        up with a complete model (old or new), never neither, and nothing is
        orphaned on disk."""
        from soup_cli.utils import adapter_fuse

        monkeypatch.chdir(tmp_path)
        _fake_train_stack(monkeypatch)
        adapter, out = _fresh_adapter_and_out(tmp_path)

        handle = open(out / "model.safetensors", "rb")
        try:
            try:
                adapter_fuse.merge_adapter_to_dense(
                    base_model="base", adapter_dir=str(adapter), out_dir="out"
                )
            except OSError:
                pass
        finally:
            handle.close()

        old_intact = (
            (out / "config.json").exists()
            and (out / "model.safetensors").exists()
            and (out / "tokenizer.json").exists()
            and (out / "model.safetensors").read_bytes() == b"OLD"
        )
        new_intact = (out / "model.safetensors").exists() and (
            (out / "model.safetensors").read_bytes() == b"NEW"
        )
        assert old_intact or new_intact, (
            "an open handle on model.safetensors left out_dir with neither a "
            "complete old model nor a complete new one"
        )

