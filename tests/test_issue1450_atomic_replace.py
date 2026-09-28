"""#1450 — a failed replace must never delete the last good copy.

Two call sites remove the old artifact before the new one is known to be
valid and in place:

* ``merge_adapter_to_dense`` (shared by ``soup shrink`` and ``soup draft
  distill``) does ``shutil.rmtree(out_dir)`` then ``os.replace(staging,
  out_dir)``; its ``except BaseException`` handler deletes ``staging`` too, so
  a failure between those two calls leaves neither copy.
* ``soup reward synth -o reward.py --force`` overwrites the existing verifier
  with the candidate BEFORE calibrating it, then ``_cleanup`` deletes the file
  on refusal or error — destroying the user's previous, working verifier.
"""

from __future__ import annotations

import json
import os
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
        # (exit 1) branch instead of the "refused" (exit 2) branch — both call
        # _cleanup(output) today.
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


class TestMergeAdapterToDenseSurvivesAFailedSwap:
    def test_out_dir_survives_when_os_replace_raises(self, tmp_path, monkeypatch):
        """cloudflare-style failure: antivirus/indexer/AV holds the staging
        dir just long enough that the final os.replace raises. Today the
        destination is already rmtree'd by then, and the except-handler
        deletes the staging copy too -> both copies gone."""
        from soup_cli.utils import adapter_fuse

        monkeypatch.chdir(tmp_path)
        _fake_train_stack(monkeypatch)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            (out / name).write_bytes(b"OLD")

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

        # The last good copy must survive SOMEWHERE: either the original
        # out_dir was never removed, or the new model landed at a path named
        # in the error. Today it is neither.
        old_survives = (
            (out / "config.json").exists()
            and (out / "config.json").read_bytes() == b"OLD"
        )
        staging_left = list(tmp_path.glob(".fuse_*")) or list(tmp_path.glob(".out.old-*"))
        assert old_survives or staging_left, (
            "a failed os.replace destroyed BOTH the previous out_dir and the "
            "staged replacement — no good copy survives"
        )

