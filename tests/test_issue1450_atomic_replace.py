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
  with the candidate BEFORE calibrating it, then deleted the file on refusal
  or error — destroying the user's previous, working verifier. It now writes
  the candidate to a ``.soup.reward-candidate.*.py`` temp sibling, loads and
  calibrates THAT file, and ``os.replace``s it onto ``-o`` only once
  calibration accepts it. ``-o`` is never moved, overwritten or otherwise
  touched before that final replace, so a refusal, a calibration error, or a
  crash at any point before it leaves the previous verifier exactly as it
  was — there is nothing to restore.
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
        leftover = list(tmp_path.glob(".soup.reward-candidate.*.py"))
        assert not leftover, f"orphaned candidate file(s) left behind: {leftover}"

    def test_calibration_error_leaves_previous_verifier_byte_identical(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.commands.reward import app

        monkeypatch.chdir(tmp_path)
        previous = "# my working verifier\ndef reward_fn(completions, **kw):\n    return [1.0]\n"
        Path("reward.py").write_text(previous, encoding="utf-8")

        # Make calibrate() raise so the command takes the "calibration failed"
        # (exit 1) branch instead of the "refused" (exit 2) branch — both
        # branches call _cleanup(candidate_path), never touching ``output``.
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
        leftover = list(tmp_path.glob(".soup.reward-candidate.*.py"))
        assert not leftover, f"orphaned candidate file(s) left behind: {leftover}"

    def test_crash_mid_calibration_leaves_previous_verifier_and_no_orphan_candidate(
        self, tmp_path, monkeypatch
    ):
        """Simulates a crash partway through the load-back + calibrate step
        (an abrupt worker failure, not a clean "calibration refused" result):
        under the old "move old file aside, write candidate in place" design
        this left the candidate sitting at ``reward.py`` with the real
        previous verifier hidden under its backup name until a human
        recovered it by hand. Under the temp-sibling design ``output`` is
        never moved or written to before the final ``os.replace``, so it
        survives untouched, and the single ``except Exception`` around the
        whole load+calibrate block still reaches ``_cleanup`` to remove the
        orphaned candidate temp file."""
        from soup_cli.commands.reward import app

        monkeypatch.chdir(tmp_path)
        previous = "# my working verifier\ndef reward_fn(completions, **kw):\n    return [1.0]\n"
        Path("reward.py").write_text(previous, encoding="utf-8")

        from soup_cli.utils import reward_synth as rs

        def _crash(*a, **kw):
            raise MemoryError("worker crashed mid-calibration")

        monkeypatch.setattr(rs, "calibrate", _crash)

        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])
        res = self._runner().invoke(app, [
            "synth", "refs.jsonl", "-o", "reward.py", "--force",
        ])

        assert res.exit_code == 1, (res.output, repr(res.exception))
        assert Path("reward.py").exists(), (
            "the previous verifier was deleted by a crash mid-calibration"
        )
        assert Path("reward.py").read_text(encoding="utf-8") == previous
        leftover = list(tmp_path.glob(".soup.reward-candidate.*.py"))
        assert not leftover, f"orphaned candidate file(s) left behind: {leftover}"

    def test_candidate_write_failure_leaves_previous_verifier_untouched(
        self, tmp_path, monkeypatch
    ):
        """The candidate is written to its own temp sibling, never to
        ``output``. If that write fails (disk full, a permission error),
        ``output`` was never touched in the first place — nothing to
        restore, and no backup file to orphan."""
        from soup_cli.commands.reward import app

        monkeypatch.chdir(tmp_path)
        previous = "# my working verifier\ndef reward_fn(completions, **kw):\n    return [1.0]\n"
        Path("reward.py").write_text(previous, encoding="utf-8")

        import os as os_module

        real_fdopen = os_module.fdopen

        def _boom(fd, mode="r", *a, **kw):
            # Only the candidate write ("w") is poisoned — _read_jsonl's own
            # os.fdopen(fd, "r", ...) of refs.jsonl must keep working.
            if mode == "w":
                os_module.close(fd)
                raise OSError(28, "No space left on device")
            return real_fdopen(fd, mode, *a, **kw)

        monkeypatch.setattr(os_module, "fdopen", _boom)

        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])
        res = self._runner().invoke(app, [
            "synth", "refs.jsonl", "-o", "reward.py", "--force",
        ])

        assert res.exit_code == 1, (res.output, repr(res.exception))
        assert Path("reward.py").exists(), (
            "the previous verifier was deleted on a failed candidate write"
        )
        assert Path("reward.py").read_text(encoding="utf-8") == previous
        leftover = list(tmp_path.glob(".soup.reward-candidate.*.py"))
        assert not leftover, f"orphaned candidate file(s) left behind: {leftover}"

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
        leftover = list(tmp_path.glob(".soup.reward-candidate.*.py"))
        assert not leftover, f"orphaned candidate file(s) left behind: {leftover}"

    def test_final_replace_failure_leaves_previous_verifier_and_no_orphan(
        self, tmp_path, monkeypatch
    ):
        """The very last step — ``os.replace(candidate_path, output)`` on an
        ACCEPTED candidate — must also be guarded. A lock held on ``output``
        or a disk-full error at that exact point used to propagate as a raw
        ``OSError`` traceback; it must instead exit cleanly (1), leave
        ``output`` untouched, and clean up the now-useless candidate."""
        from soup_cli.commands.reward import app

        monkeypatch.chdir(tmp_path)
        previous = "# my working verifier\ndef reward_fn(completions, **kw):\n    return [1.0]\n"
        Path("reward.py").write_text(previous, encoding="utf-8")

        import os as os_module

        real_replace = os_module.replace

        def _replace_fails(src, dst, *a, **kw):
            if os.path.basename(src).startswith(".soup.reward-candidate."):
                raise OSError(28, "No space left on device")
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(os_module, "replace", _replace_fails)

        _write_jsonl(Path("refs.jsonl"), [{"answer": "4"}, {"answer": "6"}])
        res = self._runner().invoke(app, [
            "synth", "refs.jsonl", "-o", "reward.py", "--force",
        ])

        assert res.exit_code == 1, (res.output, repr(res.exception))
        assert res.exception is None or isinstance(res.exception, SystemExit), (
            "a failed final replace must exit cleanly, not raise an uncaught OSError"
        )
        assert Path("reward.py").exists(), (
            "the previous verifier was deleted by a failed final replace"
        )
        assert Path("reward.py").read_text(encoding="utf-8") == previous
        leftover = list(tmp_path.glob(".soup.reward-candidate.*.py"))
        assert not leftover, f"orphaned candidate file(s) left behind: {leftover}"


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

    def test_both_swap_and_restore_fail_keeps_both_copies_and_names_both_paths(
        self, tmp_path, monkeypatch
    ):
        """The double-failure path the issue calls out explicitly: the swap
        AND the rename-back both fail. Neither ``backup`` nor ``staging`` may
        be deleted then — the code must not guess which one holds the good
        copy — and the raised error must name both paths so a human can
        recover by hand.

        Mutation check: this is what catches ``if not keep_staging:`` being
        replaced by ``if True:`` at the bottom of the function — that
        mutation makes the ``except BaseException`` handler always
        ``rmtree(staging)``, which would delete the *only* surviving copy in
        this exact scenario. Without this test, that mutation still leaves
        the other tests in this file green.
        """
        from soup_cli.utils import adapter_fuse

        monkeypatch.chdir(tmp_path)
        _fake_train_stack(monkeypatch)
        adapter, out = _fresh_adapter_and_out(tmp_path)

        real_replace = adapter_fuse.os.replace

        def _replace_fails_both(src, dst, *a, **kw):
            name = os.path.basename(src)
            if name.startswith(".fuse_") or ".old-" in name:
                raise PermissionError(5, "Access is denied")
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(adapter_fuse.os, "replace", _replace_fails_both)

        with pytest.raises(RuntimeError) as excinfo:
            adapter_fuse.merge_adapter_to_dense(
                base_model="base", adapter_dir=str(adapter), out_dir="out"
            )

        message = str(excinfo.value)
        staging_dirs = list(tmp_path.glob(".fuse_*"))
        backup_dirs = list(tmp_path.glob(".out.old-*"))
        assert len(staging_dirs) == 1, f"staging dir missing or duplicated: {staging_dirs}"
        assert len(backup_dirs) == 1, f"backup dir missing or duplicated: {backup_dirs}"
        assert str(backup_dirs[0]) in message, "error does not name the backup path"
        assert str(staging_dirs[0]) in message, "error does not name the staging path"

        # Both copies must be complete, not partially written or removed.
        assert (backup_dirs[0] / "config.json").read_bytes() == b"OLD"
        assert (backup_dirs[0] / "model.safetensors").read_bytes() == b"OLD"
        assert (staging_dirs[0] / "model.safetensors").read_bytes() == b"NEW"
        assert not out.exists(), "out_dir should be absent — neither replace landed there"

    def test_backup_rename_failure_keeps_old_model_and_new_staging(
        self, tmp_path, monkeypatch
    ):
        """The rename that moves the previous model aside can hit the same
        lock as the swap. out_dir is then untouched, so the new merge in
        staging is the only copy of it and must not be deleted."""
        from soup_cli.utils import adapter_fuse

        monkeypatch.chdir(tmp_path)
        _fake_train_stack(monkeypatch)
        adapter, out = _fresh_adapter_and_out(tmp_path)

        real_replace = adapter_fuse.os.replace

        def _backup_fails(src, dst, *a, **kw):
            if ".old-" in os.path.basename(dst):
                raise PermissionError(5, "Access is denied")
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(adapter_fuse.os, "replace", _backup_fails)

        with pytest.raises(RuntimeError) as excinfo:
            adapter_fuse.merge_adapter_to_dense(
                base_model="base", adapter_dir=str(adapter), out_dir="out"
            )

        assert (out / "model.safetensors").read_bytes() == b"OLD"
        staging_dirs = list(tmp_path.glob(".fuse_*"))
        assert len(staging_dirs) == 1, f"staging dir missing: {staging_dirs}"
        assert (staging_dirs[0] / "model.safetensors").read_bytes() == b"NEW"
        assert str(staging_dirs[0]) in str(excinfo.value)
        assert not list(tmp_path.glob(".out.old-*"))

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only file-lock semantics")
    def test_out_dir_survives_an_open_handle_on_model_safetensors(self, tmp_path, monkeypatch):
        """Windows-only: another process (a viewer, a server, an antivirus
        scan) holds ``out_dir/model.safetensors`` open, so the rename-aside
        fails. The old model must be untouched and the merged model kept at
        the staging path the error names."""
        from soup_cli.utils import adapter_fuse

        monkeypatch.chdir(tmp_path)
        _fake_train_stack(monkeypatch)
        adapter, out = _fresh_adapter_and_out(tmp_path)

        handle = open(out / "model.safetensors", "rb")
        try:
            # Windows refuses to rename a directory that holds an open file, so
            # the rename-aside fails: a RuntimeError that names the staged model.
            with pytest.raises(RuntimeError, match="move the previous model") as excinfo:
                adapter_fuse.merge_adapter_to_dense(
                    base_model="base", adapter_dir=str(adapter), out_dir="out"
                )
        finally:
            handle.close()

        # The old model is complete and untouched ...
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            assert (out / name).read_bytes() == b"OLD"
        # ... and the merged model survives at the path the error names.
        staging_dirs = list(tmp_path.glob(".fuse_*"))
        assert len(staging_dirs) == 1, staging_dirs
        assert (staging_dirs[0] / "model.safetensors").read_bytes() == b"NEW"
        assert staging_dirs[0].name in str(excinfo.value)
        assert not list(tmp_path.glob(".out.old-*"))

