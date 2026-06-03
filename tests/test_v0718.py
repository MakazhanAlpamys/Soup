"""v0.71.8 — "Probes & SAE" (tiny GPU). Closes #215, #216, #217, #218, #219.

- #215 Real probe weights — operator-supplied (`load_probe_weights`) +
  contrast-pair computation (`compute_contrast_probe`); synthetic seed retained
  as the offline fallback.
- #216 Live HF Hub auto-download for SAE weights (`download_sae` +
  `hubs.snapshot_download`); `soup probe sae-diff --auto-download`.
- #217 Live truth + harm probe utilities (`apply_truth_probe`/`apply_harm_probe`
  + bundled packs + `soup probe truth/harm`).
- #218 Auto-measure `soup probe interference --measure <eval_suite>`.
- #219 `soup train --capture-activations <layer> --capture-prompts <jsonl>`.

Heavy model loads are mocked at the `live_eval` boundary; the real loads are
covered by the release-step-6 smoke on SmolLM2-135M.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types

import numpy as np
import pytest
from typer.testing import CliRunner

runner = CliRunner()


def _clean_help(text: str) -> str:
    """Strip ANSI + remove all whitespace so flag substrings survive CI color.

    Under CI (``FORCE_COLOR``) Rich/Typer renders a long flag like
    ``--auto-download`` with ANSI escapes between styled segments (the internal
    hyphen is colorized separately) AND may line-wrap at the hyphen. Stripping
    the escapes and removing every whitespace char concatenates the literal
    flag back together (matches the v0.71.1 / test_v0717 pattern).
    """
    return re.sub(r"\s+", "", re.sub(r"\x1b\[[0-9;]*m", "", text))


# ===========================================================================
# Tiny real-torch harness for activation capture (#215 / #217 / #219).
# ===========================================================================


def _tiny_model_and_tok(d: int = 8, n_layers: int = 3, vocab: int = 64):
    import torch
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.lin = nn.Linear(dim, dim)

        def forward(self, x):  # noqa: D401 — decoder-layer-shaped tuple output
            return (self.lin(x),)

    class _Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, d)
            self.layers = nn.ModuleList([_Block(d) for _ in range(n_layers)])

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _Inner()

        def forward(self, input_ids, **_kw):
            x = self.model.embed_tokens(input_ids)
            for layer in self.model.layers:
                x = layer(x)[0]
            return types.SimpleNamespace(logits=x)

    class _DictBatch(dict):
        def to(self, _device):
            return self

    class _Tok:
        chat_template = None

        def __call__(self, text, return_tensors=None, truncation=None,
                     max_length=None, **_kw):
            ids = [(len(w) % (vocab - 1)) + 1 for w in text.split()] or [1]
            return _DictBatch(input_ids=torch.tensor([ids], dtype=torch.long))

    return _Model(), _Tok()


# ===========================================================================
# Shared probe kernel (#215 / #217)
# ===========================================================================


class TestProbeKernel:
    def test_module_imports(self) -> None:
        from soup_cli.utils import probe_kernel

        for name in (
            "compute_contrast_probe",
            "apply_linear_probe",
            "flagged_rate",
            "classify_probe_rate",
            "PROBE_VERDICTS",
        ):
            assert hasattr(probe_kernel, name), name

    def test_contrast_probe_separates_classes(self) -> None:
        from soup_cli.utils.probe_kernel import (
            apply_linear_probe,
            compute_contrast_probe,
        )

        rng = np.random.default_rng(0)
        d = 16
        pos = rng.standard_normal((20, d)) + np.array([3.0] + [0.0] * (d - 1))
        neg = rng.standard_normal((20, d)) + np.array([-3.0] + [0.0] * (d - 1))
        w, threshold = compute_contrast_probe(pos, neg)
        assert w.shape == (d,)
        assert np.isfinite(threshold)
        # A fresh positive example should score above threshold, negative below.
        new_pos = np.array([[5.0] + [0.0] * (d - 1)])
        new_neg = np.array([[-5.0] + [0.0] * (d - 1)])
        assert apply_linear_probe(new_pos, w, threshold)[0] > threshold
        assert apply_linear_probe(new_neg, w, threshold)[0] < threshold

    def test_contrast_probe_unit_norm(self) -> None:
        from soup_cli.utils.probe_kernel import compute_contrast_probe

        w, _ = compute_contrast_probe(
            np.ones((4, 8)) * 2.0, np.ones((4, 8)) * -1.0
        )
        assert np.linalg.norm(w) == pytest.approx(1.0, abs=1e-5)

    def test_contrast_probe_degenerate_rejected(self) -> None:
        from soup_cli.utils.probe_kernel import compute_contrast_probe

        same = np.ones((4, 8))
        with pytest.raises(ValueError, match="degenerate"):
            compute_contrast_probe(same, same)

    def test_contrast_probe_dim_mismatch(self) -> None:
        from soup_cli.utils.probe_kernel import compute_contrast_probe

        with pytest.raises(ValueError, match="hidden-dim mismatch"):
            compute_contrast_probe(np.ones((4, 8)), np.ones((4, 6)))

    def test_contrast_probe_non_2d_rejected(self) -> None:
        from soup_cli.utils.probe_kernel import compute_contrast_probe

        with pytest.raises(ValueError, match="2D"):
            compute_contrast_probe(np.ones(8), np.ones((4, 8)))

    def test_contrast_probe_empty_rejected(self) -> None:
        from soup_cli.utils.probe_kernel import compute_contrast_probe

        with pytest.raises(ValueError, match="non-empty"):
            compute_contrast_probe(np.empty((0, 8)), np.ones((4, 8)))

    def test_contrast_probe_non_finite_rejected(self) -> None:
        from soup_cli.utils.probe_kernel import compute_contrast_probe

        bad = np.ones((4, 8))
        bad[0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            compute_contrast_probe(bad, np.ones((4, 8)))

    def test_apply_linear_probe_shape_mismatch(self) -> None:
        from soup_cli.utils.probe_kernel import apply_linear_probe

        with pytest.raises(ValueError, match="shape mismatch"):
            apply_linear_probe(np.ones((3, 8)), np.ones(6), 0.0)

    def test_apply_linear_probe_bool_threshold(self) -> None:
        from soup_cli.utils.probe_kernel import apply_linear_probe

        with pytest.raises(TypeError, match="threshold"):
            apply_linear_probe(np.ones((3, 8)), np.ones(8), True)

    def test_apply_linear_probe_nan_threshold(self) -> None:
        from soup_cli.utils.probe_kernel import apply_linear_probe

        with pytest.raises(ValueError, match="finite"):
            apply_linear_probe(np.ones((3, 8)), np.ones(8), float("nan"))

    def test_flagged_rate(self) -> None:
        from soup_cli.utils.probe_kernel import flagged_rate

        assert flagged_rate(np.array([1.0, 2.0, 3.0, 4.0]), 2.5) == 0.5
        assert flagged_rate(np.array([]), 0.0) == 0.0
        with pytest.raises(ValueError, match="1D"):
            flagged_rate(np.ones((2, 2)), 0.0)

    def test_classify_probe_rate_bands(self) -> None:
        from soup_cli.utils.probe_kernel import classify_probe_rate

        assert classify_probe_rate(0.0, minor=0.05, major=0.20) == "OK"
        # boundary lands in the MORE SEVERE bucket
        assert classify_probe_rate(0.05, minor=0.05, major=0.20) == "MINOR"
        assert classify_probe_rate(0.20, minor=0.05, major=0.20) == "MAJOR"
        assert classify_probe_rate(0.10, minor=0.05, major=0.20) == "MINOR"

    def test_classify_probe_rate_invalid(self) -> None:
        from soup_cli.utils.probe_kernel import classify_probe_rate

        with pytest.raises(TypeError):
            classify_probe_rate(True, minor=0.05, major=0.20)
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            classify_probe_rate(1.5, minor=0.05, major=0.20)
        with pytest.raises(ValueError, match="≤ major"):
            classify_probe_rate(0.1, minor=0.3, major=0.2)


# ===========================================================================
# Shared activation capture (#215 / #217 / #219)
# ===========================================================================


class TestExtractLayerActivations:
    def test_resolve_layer_module(self) -> None:
        from soup_cli.utils.live_eval import resolve_layer_module

        model, _ = _tiny_model_and_tok()
        mod = resolve_layer_module(model, "model.layers.1")
        assert mod is model.model.layers[1]

    def test_resolve_layer_module_bad_path(self) -> None:
        from soup_cli.utils.live_eval import resolve_layer_module

        model, _ = _tiny_model_and_tok()
        with pytest.raises(ValueError, match="could not resolve"):
            resolve_layer_module(model, "model.nope.5")

    def test_resolve_layer_module_rejects_dunder(self) -> None:
        from soup_cli.utils.live_eval import resolve_layer_module

        model, _ = _tiny_model_and_tok()
        with pytest.raises(ValueError, match="invalid layer path"):
            resolve_layer_module(model, "model.__class__")

    def test_resolve_layer_module_empty(self) -> None:
        from soup_cli.utils.live_eval import resolve_layer_module

        model, _ = _tiny_model_and_tok()
        with pytest.raises(ValueError, match="non-empty"):
            resolve_layer_module(model, "  ")

    def test_extract_mean_pooled(self) -> None:
        from soup_cli.utils.live_eval import extract_layer_activations

        model, tok = _tiny_model_and_tok(d=8)
        acts = extract_layer_activations(
            model, tok, ["hello world", "foo bar baz", "one"],
            layer="model.layers.1", device="cpu", pool="mean",
        )
        assert acts.shape == (3, 8)
        assert acts.dtype == np.float32

    def test_extract_per_token(self) -> None:
        from soup_cli.utils.live_eval import extract_layer_activations

        model, tok = _tiny_model_and_tok(d=8)
        acts = extract_layer_activations(
            model, tok, ["hello world", "foo bar baz"],
            layer="model.layers.0", device="cpu", pool="none",
        )
        # 2 + 3 tokens = 5 rows
        assert acts.shape == (5, 8)

    def test_extract_per_token_token_cap(self) -> None:
        from soup_cli.utils.live_eval import extract_layer_activations

        model, tok = _tiny_model_and_tok(d=8)
        acts = extract_layer_activations(
            model, tok, ["a b c d e", "f g h i j"],
            layer="model.layers.0", device="cpu", pool="none", max_tokens=4,
        )
        assert acts.shape[0] == 4

    def test_extract_bad_pool(self) -> None:
        from soup_cli.utils.live_eval import extract_layer_activations

        model, tok = _tiny_model_and_tok()
        with pytest.raises(ValueError, match="pool"):
            extract_layer_activations(
                model, tok, ["x"], layer="model.layers.0", device="cpu",
                pool="bogus",
            )

    def test_extract_empty_prompts(self) -> None:
        from soup_cli.utils.live_eval import extract_layer_activations

        model, tok = _tiny_model_and_tok()
        with pytest.raises(ValueError, match="non-empty"):
            extract_layer_activations(
                model, tok, ["", "  "], layer="model.layers.0", device="cpu",
            )

    def test_hook_removed_after_extract(self) -> None:
        from soup_cli.utils.live_eval import extract_layer_activations

        model, tok = _tiny_model_and_tok()
        target = model.model.layers[1]
        extract_layer_activations(
            model, tok, ["a b"], layer="model.layers.1", device="cpu",
        )
        # No lingering forward hooks.
        assert len(target._forward_hooks) == 0


# ===========================================================================
# #215 — real sleeper-probe weights (operator-supplied + contrast-pair)
# ===========================================================================


class TestSleeperRealWeights:
    def test_reexports(self) -> None:
        from soup_cli.utils import sleeper_probe

        assert hasattr(sleeper_probe, "compute_contrast_probe")
        assert hasattr(sleeper_probe, "load_probe_weights")

    def test_load_npz(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.sleeper_probe import load_probe_weights

        p = tmp_path / "probe.npz"
        np.savez(p, w=np.ones(8, dtype=np.float32), threshold=np.array(2.5))
        w, thr = load_probe_weights("probe.npz")
        assert w.shape == (8,)
        assert thr == pytest.approx(2.5)

    def test_load_npy_threshold_default(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.sleeper_probe import load_probe_weights

        np.save(tmp_path / "probe.npy", np.ones(8, dtype=np.float32))
        w, thr = load_probe_weights("probe.npy")
        assert w.shape == (8,)
        assert thr == 0.0

    def test_load_safetensors(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from safetensors.numpy import save_file

        from soup_cli.utils.sleeper_probe import load_probe_weights

        save_file(
            {
                "W_probe": np.ones(8, dtype=np.float32),
                "threshold": np.array([1.5], dtype=np.float32),
            },
            str(tmp_path / "probe.safetensors"),
        )
        w, thr = load_probe_weights("probe.safetensors")
        assert w.shape == (8,)
        assert thr == pytest.approx(1.5)

    def test_load_bad_extension(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.sleeper_probe import load_probe_weights

        (tmp_path / "probe.bin").write_bytes(b"x")
        with pytest.raises(ValueError, match="npz / .npy / .safetensors"):
            load_probe_weights("probe.bin")

    def test_load_outside_cwd(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.sleeper_probe import load_probe_weights

        with pytest.raises(ValueError):
            load_probe_weights(str(tmp_path.parent / "x.npz"))

    def test_load_npz_missing_w(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.sleeper_probe import load_probe_weights

        np.savez(tmp_path / "probe.npz", notw=np.ones(8))
        with pytest.raises(KeyError, match="'w'"):
            load_probe_weights("probe.npz")

    def test_load_non_1d(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.sleeper_probe import load_probe_weights

        np.save(tmp_path / "probe.npy", np.ones((2, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="1D"):
            load_probe_weights("probe.npy")

    def test_load_non_finite(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.sleeper_probe import load_probe_weights

        bad = np.ones(8, dtype=np.float32)
        bad[0] = np.inf
        np.save(tmp_path / "probe.npy", bad)
        with pytest.raises(ValueError, match="finite"):
            load_probe_weights("probe.npy")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_load_symlink_rejected(self, tmp_path, monkeypatch) -> None:
        import os

        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.sleeper_probe import load_probe_weights

        real = tmp_path / "real.npz"
        np.savez(real, w=np.ones(8))
        link = tmp_path / "link.npz"
        os.symlink(real, link)
        with pytest.raises(ValueError):
            load_probe_weights("link.npz")

    def test_run_with_weights_arbitrary_base(self) -> None:
        from soup_cli.utils.sleeper_probe import run_sleeper_probe

        acts = np.zeros((10, 8), dtype=np.float32)
        acts[:2] = 10.0  # 2 of 10 tokens score high → 20% → MAJOR (1%/5% bands)
        w = np.array([1.0] + [0.0] * 7, dtype=np.float32)
        result = run_sleeper_probe(acts, "my-org/custom-model", weights=(w, 1.0))
        assert result.base == "my-org/custom-model"
        assert result.num_tokens == 10
        assert result.verdict == "MAJOR"

    def test_run_with_weights_dim_mismatch(self) -> None:
        from soup_cli.utils.sleeper_probe import run_sleeper_probe

        with pytest.raises(ValueError, match="hidden_dim mismatch"):
            run_sleeper_probe(
                np.ones((3, 8), dtype=np.float32), "x", weights=(np.ones(6), 0.0)
            )

    def test_run_with_bad_weights_tuple(self) -> None:
        from soup_cli.utils.sleeper_probe import run_sleeper_probe

        with pytest.raises(TypeError, match="W, threshold"):
            run_sleeper_probe(np.ones((3, 8), dtype=np.float32), "x", weights=[1, 2, 3])

    def test_run_with_bad_threshold(self) -> None:
        from soup_cli.utils.sleeper_probe import run_sleeper_probe

        with pytest.raises(TypeError, match="threshold"):
            run_sleeper_probe(
                np.ones((3, 8), dtype=np.float32), "x", weights=(np.ones(8), True)
            )

    def test_synthetic_fallback_still_works(self) -> None:
        # The SHA-256 fallback path must remain (back-compat, #215 criterion).
        from soup_cli.utils.sleeper_probe import run_sleeper_probe

        acts = np.zeros((50, 4096), dtype=np.float32)
        result = run_sleeper_probe(acts, "meta-llama/Llama-3-8B")
        assert result.base == "meta-llama/Llama-3-8B"
        assert result.verdict in {"OK", "MINOR", "MAJOR"}

    def test_contrast_to_run_end_to_end(self) -> None:
        from soup_cli.utils.sleeper_probe import (
            compute_contrast_probe,
            run_sleeper_probe,
        )

        rng = np.random.default_rng(1)
        d = 8
        pos = rng.standard_normal((20, d)) + np.array([4.0] + [0.0] * (d - 1))
        neg = rng.standard_normal((20, d)) + np.array([-4.0] + [0.0] * (d - 1))
        w, thr = compute_contrast_probe(pos, neg)
        # Score a held-out positive-heavy activation set → high defection rate.
        test_acts = rng.standard_normal((10, d)) + np.array([4.0] + [0.0] * (d - 1))
        result = run_sleeper_probe(test_acts, "base", weights=(w, thr))
        assert result.defection_rate > 0.5


class TestSleeperCli215:
    def _write_acts(self, tmp_path, n: int, d: int, high: int) -> str:
        acts = np.zeros((n, d), dtype=np.float32)
        acts[:high, 0] = 10.0
        p = tmp_path / "acts.json"
        p.write_text(json.dumps({"activations": acts.tolist()}), encoding="utf-8")
        return "acts.json"

    def test_sleeper_with_weights(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        acts = self._write_acts(tmp_path, 10, 8, 1)
        np.savez(tmp_path / "w.npz", w=np.array([1.0] + [0.0] * 7), threshold=np.array(1.0))
        res = runner.invoke(
            app, ["sleeper", "custom/model", "--evidence", acts, "--weights", "w.npz"]
        )
        assert res.exit_code in (0, 2), res.output
        assert "Sleeper probe" in res.output or "Defection" in res.output

    def test_sleeper_weights_requires_evidence(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        np.savez(tmp_path / "w.npz", w=np.ones(8))
        res = runner.invoke(app, ["sleeper", "custom/model", "--weights", "w.npz"])
        assert res.exit_code == 2
        assert "requires --evidence" in res.output

    def test_sleeper_bad_weights_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        acts = self._write_acts(tmp_path, 4, 8, 0)
        res = runner.invoke(
            app, ["sleeper", "custom/model", "--evidence", acts, "--weights", "missing.npz"]
        )
        assert res.exit_code == 2

    def test_sleeper_help_lists_weights(self) -> None:
        from soup_cli.commands.probe import app

        res = runner.invoke(app, ["sleeper", "--help"])
        assert res.exit_code == 0
        clean = _clean_help(res.output)
        assert "weights" in clean


# ===========================================================================
# #216 — live HF Hub auto-download for SAE weights
# ===========================================================================


def _fake_sae_dir(tmp_path, d_model: int = 4, n_feats: int = 8) -> str:
    from safetensors.numpy import save_file

    d = tmp_path / "snapshot"
    d.mkdir()
    save_file(
        {
            "W_enc": np.ones((d_model, n_feats), dtype=np.float32),
            "b_enc": np.zeros(n_feats, dtype=np.float32),
        },
        str(d / "sae.safetensors"),
    )
    return str(d)


class TestHubsSnapshotDownload:
    def test_validate_cache_dir_under_tmp(self, tmp_path) -> None:
        from soup_cli.utils.hubs import _validate_cache_dir

        sub = tmp_path / "cache"
        assert _validate_cache_dir(str(sub)).endswith("cache")

    def test_validate_cache_dir_outside(self) -> None:
        from soup_cli.utils.hubs import _validate_cache_dir

        with pytest.raises(ValueError, match="HOME"):
            _validate_cache_dir("/etc/passwd-dir")

    def test_snapshot_download_bad_repo(self, tmp_path) -> None:
        from soup_cli.utils.hubs import snapshot_download

        with pytest.raises(ValueError, match="null bytes"):
            snapshot_download("a\x00b", cache_dir=str(tmp_path / "c"))

    def test_snapshot_download_happy(self, tmp_path, monkeypatch) -> None:
        import huggingface_hub

        from soup_cli.utils import hubs

        captured = {}

        def _fake(repo_id, local_dir, revision, allow_patterns):
            captured["repo_id"] = repo_id
            captured["local_dir"] = local_dir
            captured["allow_patterns"] = allow_patterns
            return local_dir

        monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake)
        out = hubs.snapshot_download(
            "google/gemma-scope-2b-pt-res",
            cache_dir=str(tmp_path / "cache"),
            allow_patterns=["*.safetensors"],
            namespace_check=False,
        )
        assert captured["repo_id"] == "google/gemma-scope-2b-pt-res"
        assert captured["allow_patterns"] == ["*.safetensors"]
        assert out.endswith("cache")

    def test_snapshot_download_bad_allow_patterns(self, tmp_path) -> None:
        from soup_cli.utils.hubs import snapshot_download

        with pytest.raises(TypeError, match="allow_patterns"):
            snapshot_download(
                "google/gemma-scope-2b-pt-res",
                cache_dir=str(tmp_path / "c"),
                allow_patterns="*.safetensors",  # not a list
                namespace_check=False,
            )


class TestDownloadSae:
    def test_rejects_non_allowlisted(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils import hubs
        from soup_cli.utils.sae_diff import download_sae

        # Must reject BEFORE any network call.
        called = {"n": 0}
        monkeypatch.setattr(
            hubs, "snapshot_download",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1),
        )
        with pytest.raises(ValueError, match="HF_HUB_ALLOWLIST"):
            download_sae("evil/not-a-real-sae")
        assert called["n"] == 0

    def test_happy_path(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils import hubs
        from soup_cli.utils.sae_diff import download_sae

        snap = _fake_sae_dir(tmp_path)
        captured = {}

        def _fake(repo_id, *, cache_dir, revision=None, allow_patterns=None):
            captured["repo_id"] = repo_id
            captured["cache_dir"] = cache_dir
            return snap

        monkeypatch.setattr(hubs, "snapshot_download", _fake)
        weights = download_sae("EleutherAI/sae-pythia-70m-deduped")
        assert "W_enc" in weights
        # Operator's exact case preserved for the actual download.
        assert captured["repo_id"] == "EleutherAI/sae-pythia-70m-deduped"
        # Cache lands under ~/.soup/sae-cache/
        import os

        assert os.path.join(".soup", "sae-cache") in captured["cache_dir"]

    def test_custom_cache_dir(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils import hubs
        from soup_cli.utils.sae_diff import download_sae

        snap = _fake_sae_dir(tmp_path)
        captured = {}

        def _fake(repo_id, *, cache_dir, revision=None, allow_patterns=None):
            captured["cache_dir"] = cache_dir
            return snap

        monkeypatch.setattr(hubs, "snapshot_download", _fake)
        download_sae("openai/sae-gpt2-small", cache_dir=str(tmp_path / "mycache"))
        assert captured["cache_dir"] == str(tmp_path / "mycache")

    def test_no_safetensors(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils import hubs
        from soup_cli.utils.sae_diff import download_sae

        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(
            hubs, "snapshot_download",
            lambda repo_id, **k: str(empty),
        )
        with pytest.raises(FileNotFoundError, match="no .safetensors"):
            download_sae("openai/sae-gpt2-small")

    def test_tofu_rejection_propagates(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils import hubs
        from soup_cli.utils.sae_diff import download_sae

        def _boom(repo_id, **k):
            raise ValueError("namespace-pin refused: author changed")

        monkeypatch.setattr(hubs, "snapshot_download", _boom)
        with pytest.raises(ValueError, match="namespace-pin"):
            download_sae("openai/sae-gpt2-small")

    def test_default_cache_dir_under_home(self) -> None:
        import os

        from soup_cli.utils.sae_diff import default_sae_cache_dir

        d = default_sae_cache_dir()
        assert d.startswith(os.path.expanduser("~"))
        assert d.endswith(os.path.join(".soup", "sae-cache"))


class TestSaeDiffCli216:
    def test_auto_download_in_help(self) -> None:
        from soup_cli.commands.probe import app

        res = runner.invoke(app, ["sae-diff", "--help"])
        assert res.exit_code == 0
        assert "auto-download" in _clean_help(res.output)

    def test_auto_download_unknown_repo(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        (tmp_path / "pre.json").write_text(
            json.dumps({"activations": [[1.0, 2.0, 3.0, 4.0]]}), encoding="utf-8"
        )
        (tmp_path / "post.json").write_text(
            json.dumps({"activations": [[1.0, 2.0, 3.0, 4.0]]}), encoding="utf-8"
        )
        res = runner.invoke(
            app,
            ["sae-diff", "evil/nope", "pre.json", "post.json", "--auto-download"],
        )
        assert res.exit_code == 2
        assert "HF_HUB_ALLOWLIST" in res.output

    def test_auto_download_happy(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands import probe as probe_cmd
        from soup_cli.utils import sae_diff

        # Mock download_sae to return a small in-memory SAE.
        fake_sae = {
            "W_enc": np.ones((4, 6), dtype=np.float32),
            "b_enc": np.zeros(6, dtype=np.float32),
        }
        monkeypatch.setattr(sae_diff, "download_sae", lambda *a, **k: fake_sae)
        (tmp_path / "pre.json").write_text(
            json.dumps({"activations": [[1.0, 2.0, 3.0, 4.0]] * 3}),
            encoding="utf-8",
        )
        (tmp_path / "post.json").write_text(
            json.dumps({"activations": [[2.0, 1.0, 4.0, 3.0]] * 3}),
            encoding="utf-8",
        )
        res = runner.invoke(
            probe_cmd.app,
            ["sae-diff", "openai/sae-gpt2-small", "pre.json", "post.json",
             "--auto-download"],
        )
        assert res.exit_code == 0, res.output
        assert "SAE feature diff" in res.output


# ===========================================================================
# #217 — truth + harm probes
# ===========================================================================


@pytest.mark.parametrize("mod_name", ["truth_probe", "harm_probe"])
class TestTruthHarmProbes:
    def _mod(self, mod_name):
        import importlib

        return importlib.import_module(f"soup_cli.utils.{mod_name}")

    def test_module_imports(self, mod_name) -> None:
        mod = self._mod(mod_name)
        kind = mod_name.split("_")[0]
        for name in (
            f"BUNDLED_{kind.upper()}_PROBES",
            f"apply_{kind}_probe",
            f"classify_{kind}_score",
            f"run_{kind}_probe",
            f"validate_base_for_{kind}",
            f"render_{kind}_json",
            f"render_{kind}_markdown",
            f"{kind.upper()}_CONTRAST_PROMPTS",
            "compute_contrast_probe",
            "load_probe_weights",
        ):
            assert hasattr(mod, name), name

    def test_bundled_immutable_6_bases(self, mod_name) -> None:
        from types import MappingProxyType

        mod = self._mod(mod_name)
        kind = mod_name.split("_")[0]
        bundled = getattr(mod, f"BUNDLED_{kind.upper()}_PROBES")
        assert isinstance(bundled, MappingProxyType)
        assert len(bundled) == 6
        with pytest.raises(TypeError):
            bundled["x"] = "y"  # type: ignore[index]

    def test_apply_shape_mismatch(self, mod_name) -> None:
        mod = self._mod(mod_name)
        kind = mod_name.split("_")[0]
        apply = getattr(mod, f"apply_{kind}_probe")
        with pytest.raises(ValueError, match="shape mismatch"):
            apply(np.ones((3, 8)), np.ones(6), threshold=0.0)

    def test_classify_bands(self, mod_name) -> None:
        mod = self._mod(mod_name)
        kind = mod_name.split("_")[0]
        classify = getattr(mod, f"classify_{kind}_score")
        assert classify(0.0) == "OK"
        assert classify(0.04) == "OK"
        assert classify(0.05) == "MINOR"  # 5% boundary -> MINOR
        assert classify(0.19) == "MINOR"
        assert classify(0.20) == "MAJOR"  # 20% boundary -> MAJOR

    def test_run_synthetic_fallback(self, mod_name) -> None:
        mod = self._mod(mod_name)
        kind = mod_name.split("_")[0]
        run = getattr(mod, f"run_{kind}_probe")
        result = run(np.zeros((20, 4096), dtype=np.float32), "meta-llama/Llama-3-8B")
        assert result.kind == kind
        assert result.base == "meta-llama/Llama-3-8B"
        assert result.verdict in {"OK", "MINOR", "MAJOR"}

    def test_run_unknown_base(self, mod_name) -> None:
        mod = self._mod(mod_name)
        kind = mod_name.split("_")[0]
        run = getattr(mod, f"run_{kind}_probe")
        with pytest.raises(ValueError, match="no bundled probe"):
            run(np.zeros((4, 4096), dtype=np.float32), "evil/unknown")

    def test_run_dim_mismatch(self, mod_name) -> None:
        mod = self._mod(mod_name)
        kind = mod_name.split("_")[0]
        run = getattr(mod, f"run_{kind}_probe")
        with pytest.raises(ValueError, match="hidden_dim mismatch"):
            run(np.zeros((4, 100), dtype=np.float32), "meta-llama/Llama-3-8B")

    def test_run_with_weights_major(self, mod_name) -> None:
        mod = self._mod(mod_name)
        kind = mod_name.split("_")[0]
        run = getattr(mod, f"run_{kind}_probe")
        acts = np.zeros((10, 8), dtype=np.float32)
        acts[:3, 0] = 5.0  # 30% of tokens > threshold 0.5 -> MAJOR (>=20%)
        w = np.array([1.0] + [0.0] * 7, dtype=np.float32)
        result = run(acts, "any/base", weights=(w, 0.5))
        assert result.base == "any/base"
        assert result.verdict == "MAJOR"

    def test_render_json_roundtrip(self, mod_name) -> None:
        mod = self._mod(mod_name)
        kind = mod_name.split("_")[0]
        run = getattr(mod, f"run_{kind}_probe")
        render_json = getattr(mod, f"render_{kind}_json")
        result = run(np.zeros((5, 2304), dtype=np.float32), "google/gemma-2-2b")
        payload = json.loads(render_json(result))
        assert payload["kind"] == kind
        assert "verdict" in payload

    def test_contrast_prompts_shape(self, mod_name) -> None:
        mod = self._mod(mod_name)
        kind = mod_name.split("_")[0]
        pos, neg = getattr(mod, f"{kind.upper()}_CONTRAST_PROMPTS")
        assert len(pos) == len(neg) >= 3
        assert all(isinstance(p, str) and p for p in pos + neg)


class TestProbePackTruthHarm:
    def test_packs_contain_three_kinds(self) -> None:
        from soup_cli.utils.probe_pack import BUNDLED_PACKS

        for base, pack in BUNDLED_PACKS.items():
            kinds = {p.kind for p in pack.probes}
            assert kinds == {"sleeper", "truth", "harm"}, (base, kinds)
            assert len(pack.probes) == 3

    def test_pack_json_lists_truth_harm(self) -> None:
        from soup_cli.utils.probe_pack import get_probe_pack, render_pack_json

        pack = get_probe_pack("meta-llama/Llama-3-8B")
        payload = json.loads(render_pack_json(pack))
        names = " ".join(p["name"] for p in payload["probes"])
        assert "truth:" in names
        assert "harm:" in names


@pytest.mark.parametrize("kind", ["truth", "harm"])
class TestTruthHarmCli:
    def _acts(self, tmp_path, n: int, d: int, high: int, val: float = 5.0) -> str:
        acts = np.zeros((n, d), dtype=np.float32)
        acts[:high, 0] = val
        p = tmp_path / "acts.json"
        p.write_text(json.dumps({"activations": acts.tolist()}), encoding="utf-8")
        return "acts.json"

    def test_help(self, kind) -> None:
        from soup_cli.commands.probe import app

        res = runner.invoke(app, [kind, "--help"])
        assert res.exit_code == 0

    def test_no_evidence_metadata_panel(self, kind, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        res = runner.invoke(app, [kind, "meta-llama/Llama-3-8B"])
        assert res.exit_code == 0, res.output
        assert "no evidence" in res.output.lower()

    def test_unknown_base(self, kind, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        res = runner.invoke(app, [kind, "evil/nope"])
        assert res.exit_code == 2
        assert "no bundled probe" in res.output

    def test_with_evidence_runs(self, kind, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        acts = self._acts(tmp_path, 10, 4096, 0)
        res = runner.invoke(app, [kind, "meta-llama/Llama-3-8B", "--evidence", acts])
        assert res.exit_code in (0, 2), res.output
        assert "probe" in res.output.lower()

    def test_with_weights_major_exit2(self, kind, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        acts = self._acts(tmp_path, 10, 8, 5)
        np.savez(tmp_path / "w.npz", w=np.array([1.0] + [0.0] * 7), threshold=np.array(0.5))
        res = runner.invoke(
            app, [kind, "custom/model", "--evidence", acts, "--weights", "w.npz"]
        )
        assert res.exit_code == 2  # 50% flagged -> MAJOR
        assert "MAJOR" in res.output


# ===========================================================================
# #218 — auto-measure interference
# ===========================================================================


class TestMeasureInterferenceValidation:
    def test_bad_base(self) -> None:
        from soup_cli.utils.interference_live import measure_interference_losses

        with pytest.raises(ValueError, match="base"):
            measure_interference_losses("", {"a": "p", "b": "q"}, [])

    def test_too_few_adapters(self) -> None:
        from soup_cli.utils.interference_live import measure_interference_losses

        with pytest.raises(ValueError, match="at least 2"):
            measure_interference_losses("base", {"a": "p"}, [])

    def test_too_many_adapters(self) -> None:
        from soup_cli.utils.interference_live import measure_interference_losses

        adapters = {f"a{i}": f"p{i}" for i in range(17)}
        with pytest.raises(ValueError, match="too many"):
            measure_interference_losses("base", adapters, [])

    def test_bad_adapter_path_type(self) -> None:
        from soup_cli.utils.interference_live import measure_interference_losses

        with pytest.raises(ValueError, match="path"):
            measure_interference_losses("base", {"a": "p", "b": 123}, [])

    def test_adapters_not_mapping(self) -> None:
        from soup_cli.utils.interference_live import measure_interference_losses

        with pytest.raises(TypeError, match="mapping"):
            measure_interference_losses("base", ["a", "b"], [])


class TestParseAdapterSpecs:
    def test_happy(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        from soup_cli.commands.probe import _parse_adapter_specs

        out = _parse_adapter_specs(["a=a", "b=b"])
        assert out == {"a": "a", "b": "b"}

    def test_no_equals(self) -> None:
        import typer

        from soup_cli.commands.probe import _parse_adapter_specs

        with pytest.raises(typer.BadParameter, match="name=path"):
            _parse_adapter_specs(["aaa"])

    def test_duplicate(self, tmp_path, monkeypatch) -> None:
        import typer

        monkeypatch.chdir(tmp_path)
        (tmp_path / "x").mkdir()
        from soup_cli.commands.probe import _parse_adapter_specs

        with pytest.raises(typer.BadParameter, match="duplicate"):
            _parse_adapter_specs(["a=x", "a=x"])

    def test_outside_cwd(self, tmp_path, monkeypatch) -> None:
        import typer

        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import _parse_adapter_specs

        with pytest.raises(typer.BadParameter, match="under the cwd"):
            _parse_adapter_specs([f"a={tmp_path.parent}"])

    def test_empty(self) -> None:
        import typer

        from soup_cli.commands.probe import _parse_adapter_specs

        with pytest.raises(typer.BadParameter, match="2 --adapter"):
            _parse_adapter_specs(None)


class TestInterferenceMeasureCli:
    def _eval_suite(self, tmp_path) -> str:
        p = tmp_path / "eval.jsonl"
        p.write_text(
            "\n".join(
                json.dumps({"prompt": f"q{i}", "response": f"a{i}"})
                for i in range(5)
            ),
            encoding="utf-8",
        )
        return "eval.jsonl"

    def test_measure_in_help(self) -> None:
        from soup_cli.commands.probe import app

        res = runner.invoke(app, ["interference", "--help"])
        assert res.exit_code == 0
        clean = _clean_help(res.output)
        assert "measure" in clean

    def test_measure_requires_base_model(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        suite = self._eval_suite(tmp_path)
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        res = runner.invoke(
            app,
            ["interference", "--measure", suite, "--adapter", "a=a", "--adapter", "b=b"],
        )
        assert res.exit_code == 2
        assert "base-model" in res.output

    def test_measure_bad_adapter_spec(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        suite = self._eval_suite(tmp_path)
        res = runner.invoke(
            app,
            ["interference", "--measure", suite, "--base-model", "tiny",
             "--adapter", "nope"],
        )
        assert res.exit_code == 2
        assert "name=path" in _clean_help(res.output)

    def test_measure_missing_suite(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        res = runner.invoke(
            app,
            ["interference", "--measure", "missing.jsonl", "--base-model", "tiny",
             "--adapter", "a=a", "--adapter", "b=b"],
        )
        assert res.exit_code == 2

    def test_measure_happy(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app
        from soup_cli.utils import interference_live

        suite = self._eval_suite(tmp_path)
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        # Mock the live measurement at the boundary.
        monkeypatch.setattr(
            interference_live, "measure_interference_losses",
            lambda base, adapters, rows, **k: {
                ("a", "a"): 1.0, ("b", "b"): 1.0,
                ("a", "b"): 1.02, ("b", "a"): 1.03,
            },
        )
        res = runner.invoke(
            app,
            ["interference", "--measure", suite, "--base-model", "tiny",
             "--adapter", "a=a", "--adapter", "b=b"],
        )
        assert res.exit_code == 0, res.output
        assert "interference matrix" in res.output.lower()

    def test_measure_major_exit2(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app
        from soup_cli.utils import interference_live

        suite = self._eval_suite(tmp_path)
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        monkeypatch.setattr(
            interference_live, "measure_interference_losses",
            lambda base, adapters, rows, **k: {
                ("a", "a"): 1.0, ("b", "b"): 1.0,
                ("a", "b"): 1.5, ("b", "a"): 1.5,  # 50% interference -> MAJOR
            },
        )
        res = runner.invoke(
            app,
            ["interference", "--measure", suite, "--base-model", "tiny",
             "--adapter", "a=a", "--adapter", "b=b"],
        )
        assert res.exit_code == 2

    def test_legacy_losses_path_still_works(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        (tmp_path / "losses.json").write_text(
            json.dumps({
                "adapters": ["a", "b"],
                "losses": {"a|a": 1.0, "b|b": 1.0, "a|b": 1.01, "b|a": 1.01},
            }),
            encoding="utf-8",
        )
        res = runner.invoke(app, ["interference", "losses.json"])
        assert res.exit_code == 0, res.output

    def test_no_args_errors(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.probe import app

        res = runner.invoke(app, ["interference"])
        assert res.exit_code == 2
        assert "--measure" in res.output


# ===========================================================================
# #219 — soup train --capture-activations
# ===========================================================================


class TestLoadCapturePrompts:
    def test_jsonl_objects(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.train import _load_capture_prompts

        p = tmp_path / "p.jsonl"
        p.write_text(
            json.dumps({"prompt": "hello"}) + "\n" + json.dumps({"text": "world"}) + "\n",
            encoding="utf-8",
        )
        assert _load_capture_prompts(str(p)) == ["hello", "world"]

    def test_json_strings(self, tmp_path) -> None:
        from soup_cli.commands.train import _load_capture_prompts

        p = tmp_path / "p.jsonl"
        p.write_text('"one"\n"two"\n', encoding="utf-8")
        assert _load_capture_prompts(str(p)) == ["one", "two"]

    def test_raw_text_lines(self, tmp_path) -> None:
        from soup_cli.commands.train import _load_capture_prompts

        p = tmp_path / "p.txt"
        p.write_text("plain one\nplain two\n\n", encoding="utf-8")
        assert _load_capture_prompts(str(p)) == ["plain one", "plain two"]

    def test_cap(self, tmp_path) -> None:
        from soup_cli.commands.train import _load_capture_prompts

        p = tmp_path / "p.jsonl"
        p.write_text("\n".join(f'"line{i}"' for i in range(400)), encoding="utf-8")
        assert len(_load_capture_prompts(str(p))) == 256


class TestCaptureActivations:
    def _fake_live(self, monkeypatch, capture_call=None):
        from soup_cli.utils import live_eval

        def _fake_load(model_id, *, adapter=None, device=None, trust_remote_code=False):
            if capture_call is not None:
                capture_call["model_id"] = model_id
                capture_call["adapter"] = adapter
            return (object(), object(), "cpu")

        def _fake_extract(model, tok, prompts, *, layer, device, pool="mean", **k):
            if capture_call is not None:
                capture_call["layer"] = layer
                capture_call["pool"] = pool
                capture_call["n_prompts"] = len(prompts)
            return np.ones((len(prompts) * 2, 6), dtype=np.float32)

        monkeypatch.setattr(live_eval, "load_model_and_tokenizer", _fake_load)
        monkeypatch.setattr(live_eval, "extract_layer_activations", _fake_extract)

    def test_writes_activations_json(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.train import _capture_activations

        self._fake_live(monkeypatch)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "probes.jsonl").write_text(
            json.dumps({"prompt": "a"}) + "\n" + json.dumps({"prompt": "b"}),
            encoding="utf-8",
        )
        written = _capture_activations(
            "model.layers.3", "probes.jsonl", "base", str(out_dir)
        )
        assert os.path.exists(written)
        payload = json.loads((out_dir / "activations" / "activations.json").read_text())
        assert payload["layer"] == "model.layers.3"
        assert payload["hidden_dim"] == 6
        assert payload["num_tokens"] == 4  # 2 prompts * 2 tokens each
        assert len(payload["activations"]) == 4
        # The captured snapshot is directly consumable by sae-diff (per-token 2D).
        arr = np.asarray(payload["activations"])
        assert arr.shape == (4, 6)

    def test_requires_prompts(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.train import _capture_activations

        (tmp_path / "out").mkdir()
        with pytest.raises(ValueError, match="requires --capture-prompts"):
            _capture_activations("model.layers.0", "", "base", str(tmp_path / "out"))

    def test_empty_prompts(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.train import _capture_activations

        (tmp_path / "out").mkdir()
        (tmp_path / "empty.jsonl").write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no usable prompts"):
            _capture_activations(
                "model.layers.0", "empty.jsonl", "base", str(tmp_path / "out")
            )

    def test_bad_layer(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.train import _capture_activations

        (tmp_path / "out").mkdir()
        (tmp_path / "p.jsonl").write_text('"x"', encoding="utf-8")
        with pytest.raises(ValueError, match="layer must be"):
            _capture_activations("  ", "p.jsonl", "base", str(tmp_path / "out"))

    def test_prompts_outside_cwd(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.train import _capture_activations

        (tmp_path / "out").mkdir()
        with pytest.raises(ValueError):
            _capture_activations(
                "model.layers.0", str(tmp_path.parent / "p.jsonl"), "base",
                str(tmp_path / "out"),
            )

    def test_detects_lora_adapter(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.train import _capture_activations

        call: dict = {}
        self._fake_live(monkeypatch, capture_call=call)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "p.jsonl").write_text('"x"', encoding="utf-8")
        _capture_activations("model.layers.0", "p.jsonl", "base/model", str(out_dir))
        # LoRA adapter → load base + adapter=output_dir.
        assert call["model_id"] == "base/model"
        assert call["adapter"] == str(out_dir)
        assert call["pool"] == "none"

    def test_full_model_no_adapter(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.train import _capture_activations

        call: dict = {}
        self._fake_live(monkeypatch, capture_call=call)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # config.json but no adapter_config.json → full fine-tune checkpoint.
        (out_dir / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "p.jsonl").write_text('"x"', encoding="utf-8")
        _capture_activations("model.layers.0", "p.jsonl", "base/model", str(out_dir))
        # Full fine-tune → load output_dir directly.
        assert call["model_id"] == str(out_dir)
        assert call["adapter"] is None


class TestTrainCaptureCli:
    def test_flags_in_help(self) -> None:
        from soup_cli.cli import app

        res = runner.invoke(app, ["train", "--help"])
        assert res.exit_code == 0
        clean = _clean_help(res.output)
        assert "capture-activations" in clean
        assert "capture-prompts" in clean


# ===========================================================================
# Review-agent follow-ups — direct kernel unit tests + new-guard coverage.
# ===========================================================================


class TestReviewFollowups:
    # --- validate_bundled_base (only indirectly tested before) ---
    def test_validate_bundled_base_canonicalises(self) -> None:
        from soup_cli.utils import probe_kernel as pk

        spec = pk.ProbeSpec(base="Meta/Foo", hidden_dim=8, threshold=0.0, description="")
        bundled = {"Meta/Foo": spec}
        assert pk.validate_bundled_base("meta/foo", bundled) == "Meta/Foo"

    def test_validate_bundled_base_rejects_bool(self) -> None:
        from soup_cli.utils import probe_kernel as pk

        with pytest.raises(TypeError, match="bool"):
            pk.validate_bundled_base(True, {})

    def test_validate_bundled_base_rejects_int(self) -> None:
        from soup_cli.utils import probe_kernel as pk

        with pytest.raises(TypeError, match="str"):
            pk.validate_bundled_base(5, {})

    def test_validate_bundled_base_null_byte(self) -> None:
        from soup_cli.utils import probe_kernel as pk

        with pytest.raises(ValueError, match="null byte"):
            pk.validate_bundled_base("a\x00b", {"a\x00b": object()})

    def test_validate_bundled_base_oversize(self) -> None:
        from soup_cli.utils import probe_kernel as pk

        with pytest.raises(ValueError, match="chars"):
            pk.validate_bundled_base("x" * 5000, {})

    def test_validate_bundled_base_unknown(self) -> None:
        from soup_cli.utils import probe_kernel as pk

        with pytest.raises(ValueError, match="no bundled probe"):
            pk.validate_bundled_base("nope", {"a": object()})

    # --- synthetic_probe_weights determinism + salt separation ---
    def test_synthetic_weights_deterministic(self) -> None:
        from soup_cli.utils.probe_kernel import synthetic_probe_weights

        a = synthetic_probe_weights("base/x", 16, salt="truth")
        b = synthetic_probe_weights("base/x", 16, salt="truth")
        assert np.array_equal(a, b)
        assert a.shape == (16,)
        assert float(np.linalg.norm(a)) == pytest.approx(1.0, abs=1e-5)

    def test_synthetic_weights_salt_distinguishes(self) -> None:
        from soup_cli.utils.probe_kernel import synthetic_probe_weights

        truth = synthetic_probe_weights("base/x", 16, salt="truth")
        harm = synthetic_probe_weights("base/x", 16, salt="harm")
        assert not np.array_equal(truth, harm)

    def test_synthetic_weights_empty_salt(self) -> None:
        from soup_cli.utils.probe_kernel import synthetic_probe_weights

        with pytest.raises(ValueError, match="salt"):
            synthetic_probe_weights("base/x", 16, salt="")

    def test_synthetic_weights_bool_hidden_dim(self) -> None:
        from soup_cli.utils.probe_kernel import synthetic_probe_weights

        with pytest.raises(TypeError, match="hidden_dim"):
            synthetic_probe_weights("base/x", True, salt="truth")

    # --- run_linear_probe (generic orchestrator) ---
    def test_run_linear_probe_major(self) -> None:
        from soup_cli.utils.probe_kernel import run_linear_probe

        # w aligned with every row, threshold below all scores → 100% flagged.
        acts = np.ones((10, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        res = run_linear_probe(
            acts, kind="truth", base="b", w=w, threshold=-100.0,
            minor=0.05, major=0.20,
        )
        assert res.verdict == "MAJOR"
        assert res.flag_rate == 1.0
        assert res.num_tokens == 10

    def test_run_linear_probe_non_2d(self) -> None:
        from soup_cli.utils.probe_kernel import run_linear_probe

        with pytest.raises(ValueError, match="2D"):
            run_linear_probe(
                np.ones(4), kind="truth", base="b", w=np.ones(4),
                threshold=0.0, minor=0.05, major=0.20,
            )

    # --- run_bundled_probe: weights path skips the allowlist ---
    def test_run_bundled_weights_skips_allowlist(self) -> None:
        from soup_cli.utils.probe_kernel import run_bundled_probe

        acts = np.ones((5, 4), dtype=np.float32)
        res = run_bundled_probe(
            acts, "totally/unknown/base", kind="harm", bundled={},
            salt="harm", minor=0.05, major=0.20,
            weights=(np.ones(4, dtype=np.float32), -100.0),
        )
        assert res.verdict == "MAJOR"  # arbitrary base accepted with weights

    def test_run_bundled_weights_dim_mismatch(self) -> None:
        from soup_cli.utils.probe_kernel import run_bundled_probe

        acts = np.ones((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="hidden_dim mismatch"):
            run_bundled_probe(
                acts, "b", kind="harm", bundled={}, salt="harm",
                minor=0.05, major=0.20,
                weights=(np.ones(6, dtype=np.float32), 0.0),
            )

    def test_run_bundled_synthetic_requires_bundled_base(self) -> None:
        from soup_cli.utils.probe_kernel import run_bundled_probe

        acts = np.ones((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="no bundled probe"):
            run_bundled_probe(
                acts, "unknown", kind="harm", bundled={}, salt="harm",
                minor=0.05, major=0.20,
            )

    # --- ProbeSpec / ProbeResult __post_init__ + frozen ---
    def test_probespec_rejects_bad_hidden_dim(self) -> None:
        from soup_cli.utils.probe_kernel import ProbeSpec

        with pytest.raises(ValueError, match="hidden_dim"):
            ProbeSpec(base="b", hidden_dim=0, threshold=0.0, description="")

    def test_probespec_frozen(self) -> None:
        import dataclasses

        from soup_cli.utils.probe_kernel import ProbeSpec

        spec = ProbeSpec(base="b", hidden_dim=8, threshold=0.0, description="")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.base = "other"  # type: ignore[misc]

    def test_proberesult_rejects_bad_verdict(self) -> None:
        from soup_cli.utils.probe_kernel import ProbeResult

        with pytest.raises(ValueError, match="verdict"):
            ProbeResult(
                kind="truth", base="b", num_tokens=1, flag_rate=0.0,
                max_score=0.0, verdict="WAT",
            )

    def test_proberesult_rejects_negative_tokens(self) -> None:
        from soup_cli.utils.probe_kernel import ProbeResult

        with pytest.raises(ValueError, match="num_tokens"):
            ProbeResult(
                kind="truth", base="b", num_tokens=-1, flag_rate=0.0,
                max_score=0.0, verdict="OK",
            )

    def test_proberesult_frozen(self) -> None:
        import dataclasses

        from soup_cli.utils.probe_kernel import ProbeResult

        res = ProbeResult(
            kind="truth", base="b", num_tokens=1, flag_rate=0.0,
            max_score=0.0, verdict="OK",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.verdict = "MAJOR"  # type: ignore[misc]

    # --- resolve_layer_module PEFT-wrapper fallback (live smoke caught this) ---
    def test_resolve_layer_module_peft_fallback(self) -> None:
        from soup_cli.utils.live_eval import resolve_layer_module

        inner, _ = _tiny_model_and_tok()

        class _FakePeft:
            # PeftModel-shaped: the natural path does NOT resolve against the
            # wrapper, only against get_base_model().
            def get_base_model(self):
                return inner

        resolved = resolve_layer_module(_FakePeft(), "model.layers.1")
        assert resolved is inner.model.layers[1]

    def test_resolve_layer_module_dunder_blocked_even_with_peft(self) -> None:
        from soup_cli.utils.live_eval import resolve_layer_module

        inner, _ = _tiny_model_and_tok()

        class _FakePeft:
            def get_base_model(self):
                return inner

        # Dunder is rejected up front, before any walk / fallback.
        with pytest.raises(ValueError, match="dunder"):
            resolve_layer_module(_FakePeft(), "model.__class__")

    # --- new checkpoint-layout guard in _capture_activations ---
    def test_capture_rejects_dir_without_config(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.train import _capture_activations

        out_dir = tmp_path / "out"
        out_dir.mkdir()  # neither config.json nor adapter_config.json
        (tmp_path / "p.jsonl").write_text('"x"', encoding="utf-8")
        with pytest.raises(ValueError, match="neither"):
            _capture_activations(
                "model.layers.0", "p.jsonl", "base", str(out_dir)
            )


class TestPatchInvariants:
    def test_version_bumped(self) -> None:
        import soup_cli

        major_minor = tuple(int(x) for x in soup_cli.__version__.split(".")[:3])
        assert major_minor >= (0, 71, 8)

    def test_new_modules_no_top_level_heavy(self) -> None:
        import pathlib

        src = pathlib.Path(__file__).resolve().parent.parent / "src" / "soup_cli"
        for rel in (
            "utils/probe_kernel.py",
            "utils/truth_probe.py",
            "utils/harm_probe.py",
            "utils/interference_live.py",
        ):
            text = (src / rel).read_text(encoding="utf-8")
            assert "\nimport torch" not in text, rel
            assert "\nimport numpy" not in text, rel
            assert "\nimport transformers" not in text, rel

    def test_probe_kernel_no_top_level_numpy(self) -> None:
        import pathlib

        src = pathlib.Path(__file__).resolve().parent.parent / "src" / "soup_cli"
        text = (src / "utils" / "probe_kernel.py").read_text(encoding="utf-8")
        assert "\nimport numpy" not in text
        assert "\nimport torch" not in text
