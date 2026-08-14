"""Issue #305 — exact concat+SVD task-arithmetic for MIXED-rank LoRA adapters.

The v0.71.34 ``soup adapters arithmetic`` does a signed *element-wise* merge over
the intersection of ``lora_A``/``lora_B`` tensors, so it requires all inputs to
share the same LoRA rank — a mixed-rank input is refused.

This suite pins the new concat path (PEFT ``combination_type='cat'`` style):
stack the factors so ``B_out @ A_out = Σ cᵢ·(Bᵢ@Aᵢ)`` exactly, natively handling
mixed ranks and exact negation, with an optional ``--rank`` truncated-SVD refactor
to cap the concatenated rank. The same-rank fast element-wise path is retained.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest


def _lora_keys(stem="base_model.model.layers.0.self_attn.q_proj"):
    return f"{stem}.lora_A.weight", f"{stem}.lora_B.weight"


# ---------------------------------------------------------------------------
# Pure merge math (no torch)
# ---------------------------------------------------------------------------
class TestMergeConcat:
    def test_mixed_rank_add_reconstructs_exactly(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        rng = np.random.default_rng(0)
        ak, bk = _lora_keys()
        # adapter 1: rank 4, adapter 2: rank 8; in=16, out=8
        a1 = rng.standard_normal((4, 16)).astype(np.float32)
        b1 = rng.standard_normal((8, 4)).astype(np.float32)
        a2 = rng.standard_normal((8, 16)).astype(np.float32)
        b2 = rng.standard_normal((8, 8)).astype(np.float32)
        merged, skipped, new_rank = merge_task_arithmetic_concat(
            [{ak: a1, bk: b1}, {ak: a2, bk: b2}], [1.0, 1.0]
        )
        delta_out = merged[bk] @ merged[ak]
        expected = (b1 @ a1) + (b2 @ a2)
        assert np.allclose(delta_out, expected, atol=1e-4)
        assert new_rank == 12  # 4 + 8
        assert skipped == ()

    def test_mixed_rank_negate_reconstructs_exactly(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        rng = np.random.default_rng(1)
        ak, bk = _lora_keys()
        a1 = rng.standard_normal((2, 6)).astype(np.float32)
        b1 = rng.standard_normal((5, 2)).astype(np.float32)
        a2 = rng.standard_normal((3, 6)).astype(np.float32)
        b2 = rng.standard_normal((5, 3)).astype(np.float32)
        merged, _, _ = merge_task_arithmetic_concat(
            [{ak: a1, bk: b1}, {ak: a2, bk: b2}], [1.0, -0.5]
        )
        delta_out = merged[bk] @ merged[ak]
        expected = (b1 @ a1) - 0.5 * (b2 @ a2)
        assert np.allclose(delta_out, expected, atol=1e-4)

    def test_zero_scaling_zeroes_that_adapters_contribution(self):
        # scalings=[0.0, 1.0] must zero the first adapter's contribution (proves
        # the CLI's explicit `is None` fallback threads a real 0.0, not `or 1.0`).
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        rng = np.random.default_rng(20)
        ak, bk = _lora_keys()
        a1 = rng.standard_normal((4, 6)).astype(np.float32)
        b1 = rng.standard_normal((5, 4)).astype(np.float32)
        a2 = rng.standard_normal((4, 6)).astype(np.float32)
        b2 = rng.standard_normal((5, 4)).astype(np.float32)
        merged, _, _ = merge_task_arithmetic_concat(
            [{ak: a1, bk: b1}, {ak: a2, bk: b2}], [1.0, 1.0], scalings=[0.0, 1.0]
        )
        # adapter-1 zeroed -> only adapter-2 contributes
        assert np.allclose(merged[bk] @ merged[ak], b2 @ a2, atol=1e-4)

    def test_concat_rank_cap_boundary(self, monkeypatch):
        # sum == cap is accepted; cap+1 is rejected (mutation-tight on `>`).
        import soup_cli.utils.adapter_arithmetic as mod
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        monkeypatch.setattr(mod, "_MAX_CONCAT_RANK", 8)
        ak, bk = _lora_keys()
        # exactly at cap: 4 + 4 == 8 -> accepted
        ok = merge_task_arithmetic_concat(
            [
                {ak: np.ones((4, 6), np.float32), bk: np.ones((5, 4), np.float32)},
                {ak: np.ones((4, 6), np.float32), bk: np.ones((5, 4), np.float32)},
            ],
            [1.0, 1.0],
        )
        assert ok[2] == 8
        # cap + 1: 4 + 5 == 9 -> rejected
        with pytest.raises(ValueError, match="concatenated rank"):
            merge_task_arithmetic_concat(
                [
                    {ak: np.ones((4, 6), np.float32), bk: np.ones((5, 4), np.float32)},
                    {ak: np.ones((5, 6), np.float32), bk: np.ones((5, 5), np.float32)},
                ],
                [1.0, 1.0],
            )

    def test_non_lora_shape_mismatch_skipped(self):
        # A shared non-LoRA tensor with mismatched shapes falls into `skipped`,
        # not a crash / silent mismerge.
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        ak, bk = _lora_keys()
        bias = "base_model.model.layers.0.self_attn.q_proj.bias"
        w1 = {ak: np.ones((4, 8), np.float32), bk: np.ones((8, 4), np.float32),
              bias: np.ones((8,), np.float32)}
        w2 = {ak: np.ones((4, 8), np.float32), bk: np.ones((8, 4), np.float32),
              bias: np.ones((16,), np.float32)}  # mismatched shape
        merged, skipped, _ = merge_task_arithmetic_concat([w1, w2], [1.0, 1.0])
        assert bias not in merged
        assert bias in skipped

    def test_scalings_baked_into_a(self):
        # Per-adapter scaling (alpha/r) baked into A block so a single-scaling
        # output adapter reproduces Σ cᵢ·sᵢ·(Bᵢ@Aᵢ).
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        rng = np.random.default_rng(2)
        ak, bk = _lora_keys()
        a1 = rng.standard_normal((4, 6)).astype(np.float32)
        b1 = rng.standard_normal((5, 4)).astype(np.float32)
        merged, _, _ = merge_task_arithmetic_concat(
            [{ak: a1, bk: b1}], [2.0], scalings=[3.0]
        )
        delta_out = merged[bk] @ merged[ak]
        assert np.allclose(delta_out, 2.0 * 3.0 * (b1 @ a1), atol=1e-4)

    def test_rank_truncation_svd(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        rng = np.random.default_rng(3)
        ak, bk = _lora_keys()
        a1 = rng.standard_normal((4, 16)).astype(np.float32)
        b1 = rng.standard_normal((8, 4)).astype(np.float32)
        a2 = rng.standard_normal((8, 16)).astype(np.float32)
        b2 = rng.standard_normal((8, 8)).astype(np.float32)
        full = (b1 @ a1) + (b2 @ a2)
        merged, _, new_rank = merge_task_arithmetic_concat(
            [{ak: a1, bk: b1}, {ak: a2, bk: b2}], [1.0, 1.0], rank=6
        )
        assert new_rank == 6
        assert merged[ak].shape == (6, 16)
        assert merged[bk].shape == (8, 6)
        # Truncated reconstruction == best rank-6 SVD approx of the full delta.
        u, s, vt = np.linalg.svd(full, full_matrices=False)
        best = (u[:, :6] * s[:6]) @ vt[:6, :]
        assert np.allclose(merged[bk] @ merged[ak], best, atol=1e-3)

    def test_rank_larger_than_concat_no_truncation(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        rng = np.random.default_rng(4)
        ak, bk = _lora_keys()
        a1 = rng.standard_normal((2, 6)).astype(np.float32)
        b1 = rng.standard_normal((5, 2)).astype(np.float32)
        merged, _, new_rank = merge_task_arithmetic_concat(
            [{ak: a1, bk: b1}], [1.0], rank=999
        )
        assert new_rank == 2  # capped at the actual concatenated rank

    def test_rank_must_be_positive(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        ak, bk = _lora_keys()
        a1 = np.ones((2, 3), dtype=np.float32)
        b1 = np.ones((4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="rank"):
            merge_task_arithmetic_concat([{ak: a1, bk: b1}], [1.0], rank=0)

    def test_length_mismatch_rejected(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        ak, bk = _lora_keys()
        with pytest.raises(ValueError, match="length"):
            merge_task_arithmetic_concat(
                [{ak: np.ones((2, 3)), bk: np.ones((4, 2))}], [1.0, 2.0]
            )

    def test_concat_rank_cap_rejects_ballooning_module(self, monkeypatch):
        # A single abnormally-high-rank module is refused before the concat
        # allocation (DoS cap), pointing the operator at --rank.
        import soup_cli.utils.adapter_arithmetic as mod
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        monkeypatch.setattr(mod, "_MAX_CONCAT_RANK", 8)
        ak, bk = _lora_keys()
        # rank 6 + rank 6 = 12 > cap 8
        a1 = np.ones((6, 4), dtype=np.float32)
        b1 = np.ones((4, 6), dtype=np.float32)
        a2 = np.ones((6, 4), dtype=np.float32)
        b2 = np.ones((4, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="concatenated rank|--rank"):
            merge_task_arithmetic_concat(
                [{ak: a1, bk: b1}, {ak: a2, bk: b2}], [1.0, 1.0]
            )

    def test_output_element_cap_rejects_amplified_output(self, monkeypatch):
        # Many modules padded to a uniform rank must be refused when the total
        # emitted element count exceeds the aggregate cap.
        import soup_cli.utils.adapter_arithmetic as mod
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        monkeypatch.setattr(mod, "_MAX_OUTPUT_ELEMENTS", 100)
        w1, w2 = {}, {}
        for i in range(5):
            ak, bk = _lora_keys(f"m.{i}")
            w1[ak] = np.ones((2, 8), dtype=np.float32)
            w1[bk] = np.ones((8, 2), dtype=np.float32)
            w2[ak] = np.ones((2, 8), dtype=np.float32)
            w2[bk] = np.ones((8, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="tensor elements|cap"):
            merge_task_arithmetic_concat([w1, w2], [1.0, 1.0])

    def test_incompatible_inner_dims_rejected(self):
        # Same module but in-dim (A cols) differs → cannot concat.
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        ak, bk = _lora_keys()
        a1 = np.ones((2, 3), dtype=np.float32)
        b1 = np.ones((4, 2), dtype=np.float32)
        a2 = np.ones((2, 5), dtype=np.float32)  # in-dim 5 != 3
        b2 = np.ones((4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="dim|shape"):
            merge_task_arithmetic_concat(
                [{ak: a1, bk: b1}, {ak: a2, bk: b2}], [1.0, 1.0]
            )

    def test_heterogeneous_per_module_rank_padded_to_uniform(self):
        # Two modules with different concat ranks must emit a UNIFORM-rank,
        # loadable adapter (zero-padded), not tensors of mismatched rank vs the
        # single config `r`. Reconstruction stays exact for both modules.
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        rng = np.random.default_rng(7)
        # module q: rank 4+8=12 ; module k: rank 2+2=4
        q_a1 = rng.standard_normal((4, 16)).astype(np.float32)
        q_b1 = rng.standard_normal((8, 4)).astype(np.float32)
        q_a2 = rng.standard_normal((8, 16)).astype(np.float32)
        q_b2 = rng.standard_normal((8, 8)).astype(np.float32)
        k_a1 = rng.standard_normal((2, 16)).astype(np.float32)
        k_b1 = rng.standard_normal((8, 2)).astype(np.float32)
        k_a2 = rng.standard_normal((2, 16)).astype(np.float32)
        k_b2 = rng.standard_normal((8, 2)).astype(np.float32)
        qak, qbk = _lora_keys("m.q")
        kak, kbk = _lora_keys("m.k")
        w1 = {qak: q_a1, qbk: q_b1, kak: k_a1, kbk: k_b1}
        w2 = {qak: q_a2, qbk: q_b2, kak: k_a2, kbk: k_b2}
        merged, skipped, new_rank = merge_task_arithmetic_concat([w1, w2], [1.0, 1.0])
        assert new_rank == 12
        # Every A tensor has the SAME first dim == new_rank (uniform / loadable)
        assert merged[qak].shape == (12, 16)
        assert merged[kak].shape == (12, 16)
        assert merged[qbk].shape == (8, 12)
        assert merged[kbk].shape == (8, 12)
        # Reconstruction still exact for both modules despite the padding
        assert np.allclose(merged[qbk] @ merged[qak], (q_b1 @ q_a1) + (q_b2 @ q_a2), atol=1e-4)
        assert np.allclose(merged[kbk] @ merged[kak], (k_b1 @ k_a1) + (k_b2 @ k_a2), atol=1e-4)
        assert skipped == ()

    def test_shared_non_lora_tensor_combined_linearly(self):
        # bias / modules_to_save present in every adapter must be combined
        # linearly (Σ cᵢ·tᵢ), not silently dropped (mirrors element-wise path).
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        ak, bk = _lora_keys()
        a1 = np.ones((4, 8), dtype=np.float32)
        b1 = np.ones((8, 4), dtype=np.float32)
        a2 = np.ones((8, 8), dtype=np.float32)
        b2 = np.ones((8, 8), dtype=np.float32)
        bias = "base_model.model.layers.0.self_attn.q_proj.bias"
        w1 = {ak: a1, bk: b1, bias: np.full((8,), 2.0, dtype=np.float32)}
        w2 = {ak: a2, bk: b2, bias: np.full((8,), 4.0, dtype=np.float32)}
        merged, skipped, _ = merge_task_arithmetic_concat([w1, w2], [1.0, -1.0])
        assert bias in merged
        # 1.0*2 + (-1.0)*4 = -2
        assert np.allclose(merged[bias], -2.0)
        assert skipped == ()

    def test_disjoint_pair_stems_skipped(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic_concat

        a_ak, a_bk = _lora_keys("mod.a")
        b_ak, b_bk = _lora_keys("mod.b")
        w1 = {a_ak: np.ones((2, 3), dtype=np.float32), a_bk: np.ones((4, 2), dtype=np.float32)}
        w2 = {b_ak: np.ones((2, 3), dtype=np.float32), b_bk: np.ones((4, 2), dtype=np.float32)}
        merged, skipped, _ = merge_task_arithmetic_concat([w1, w2], [1.0, 1.0])
        assert merged == {}
        assert set(skipped) == {a_ak, a_bk, b_ak, b_bk}


class TestReadAdapterScaling:
    def test_reads_scaling(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_arithmetic import read_adapter_lora_scaling

        monkeypatch.chdir(tmp_path)
        d = tmp_path / "ad"
        d.mkdir()
        (d / "adapter_config.json").write_text(
            json.dumps({"r": 8, "lora_alpha": 16}), encoding="utf-8"
        )
        assert read_adapter_lora_scaling("ad") == pytest.approx(2.0)

    def test_missing_returns_none(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_arithmetic import read_adapter_lora_scaling

        monkeypatch.chdir(tmp_path)
        d = tmp_path / "ad"
        d.mkdir()
        assert read_adapter_lora_scaling("ad") is None

    def test_zero_rank_returns_none(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_arithmetic import read_adapter_lora_scaling

        monkeypatch.chdir(tmp_path)
        d = tmp_path / "ad"
        d.mkdir()
        (d / "adapter_config.json").write_text(
            json.dumps({"r": 0, "lora_alpha": 16}), encoding="utf-8"
        )
        assert read_adapter_lora_scaling("ad") is None

    def test_zero_alpha_returns_zero_not_none(self, tmp_path, monkeypatch):
        # lora_alpha=0 with r>0 is a valid (intentionally-zeroed) task vector —
        # scaling is 0.0, NOT None. The CLI must not `or 1.0` it back to full.
        from soup_cli.utils.adapter_arithmetic import read_adapter_lora_scaling

        monkeypatch.chdir(tmp_path)
        d = tmp_path / "ad"
        d.mkdir()
        (d / "adapter_config.json").write_text(
            json.dumps({"r": 8, "lora_alpha": 0}), encoding="utf-8"
        )
        assert read_adapter_lora_scaling("ad") == 0.0

    def test_read_adapter_base_still_works(self, tmp_path, monkeypatch):
        # Regression: the shared-config refactor must not break read_adapter_base.
        from soup_cli.utils.adapter_arithmetic import read_adapter_base

        monkeypatch.chdir(tmp_path)
        d = tmp_path / "ad"
        d.mkdir()
        (d / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "meta/x", "r": 8}),
            encoding="utf-8",
        )
        assert read_adapter_base("ad") == "meta/x"


# ---------------------------------------------------------------------------
# CLI routing (mixed rank no longer refused; same-rank unchanged)
# ---------------------------------------------------------------------------
def _make_adapter(directory, base, tensors, r=8, alpha=16, extra_config=None):
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: np.asarray(v, dtype=np.float32) for k, v in tensors.items()},
        str(directory / "adapter_model.safetensors"),
    )
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": base,
        "r": r,
        "lora_alpha": alpha,
    }
    if extra_config:
        cfg.update(extra_config)
    (directory / "adapter_config.json").write_text(
        json.dumps(cfg), encoding="utf-8"
    )
    return str(directory)


class TestWriteMergedAdapterConfigOverrides:
    def test_config_overrides_patch_r_and_alpha(self, tmp_path, monkeypatch):
        from safetensors.numpy import load_file

        from soup_cli.utils.adapter_merge import write_merged_adapter

        monkeypatch.chdir(tmp_path)
        src = _make_adapter(
            tmp_path / "src", "meta/x",
            {"base_model.model.layers.0.q.lora_A.weight": np.ones((4, 8), np.float32)},
            r=4, alpha=8, extra_config={"rank_pattern": {"q": 2}},
        )
        weights = {"base_model.model.layers.0.q.lora_A.weight": np.zeros((6, 8), np.float32)}
        write_merged_adapter(
            str(tmp_path / "out"), src, weights,
            config_overrides={"r": 6, "lora_alpha": 6, "rank_pattern": {}},
        )
        cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
        assert cfg["r"] == 6
        assert cfg["lora_alpha"] == 6
        assert cfg["rank_pattern"] == {}  # stale per-module override cleared
        out_keys = list(load_file(str(tmp_path / "out" / "adapter_model.safetensors")))
        assert any("lora_A" in k for k in out_keys)

    def test_config_overrides_none_leaves_config_verbatim(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_merge import write_merged_adapter

        monkeypatch.chdir(tmp_path)
        src = _make_adapter(
            tmp_path / "src", "meta/x",
            {"m.lora_A.weight": np.ones((4, 8), np.float32)}, r=4, alpha=8,
        )
        write_merged_adapter(
            str(tmp_path / "out"), src,
            {"m.lora_A.weight": np.ones((4, 8), np.float32)},
        )
        cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
        assert cfg["r"] == 4 and cfg["lora_alpha"] == 8  # unchanged


def _rng_tensor(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


class TestArithmeticMixedRankCli:
    def _run(self, args, cwd):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        runner = CliRunner()
        old = os.getcwd()
        os.chdir(cwd)
        try:
            return runner.invoke(app, args)
        finally:
            os.chdir(old)

    def test_mixed_rank_merges_not_refused(self, tmp_path):
        ak, bk = _lora_keys()
        a = _make_adapter(
            tmp_path / "coder",
            "meta/x",
            {ak: _rng_tensor((4, 16), 1), bk: _rng_tensor((8, 4), 2)},
            r=4,
            alpha=8,
        )
        b = _make_adapter(
            tmp_path / "math",
            "meta/x",
            {ak: _rng_tensor((8, 16), 3), bk: _rng_tensor((8, 8), 4)},
            r=8,
            alpha=16,
        )
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
        assert cfg["r"] == 12  # 4 + 8, and config updated to match tensors

    def test_mixed_rank_with_rank_truncation(self, tmp_path):
        ak, bk = _lora_keys()
        a = _make_adapter(
            tmp_path / "coder",
            "meta/x",
            {ak: _rng_tensor((4, 16), 5), bk: _rng_tensor((8, 4), 6)},
            r=4,
            alpha=8,
        )
        b = _make_adapter(
            tmp_path / "math",
            "meta/x",
            {ak: _rng_tensor((8, 16), 7), bk: _rng_tensor((8, 8), 8)},
            r=8,
            alpha=16,
        )
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "--rank", "5", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
        assert cfg["r"] == 5
        from safetensors.numpy import load_file

        merged = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
        assert merged[ak].shape == (5, 16)

    def test_mixed_rank_clears_stale_rank_pattern(self, tmp_path):
        # A source adapter carrying a per-module rank_pattern must NOT leak it
        # into the uniform-rank concat output.
        ak, bk = _lora_keys()
        a = _make_adapter(
            tmp_path / "coder", "meta/x",
            {ak: _rng_tensor((4, 16), 21), bk: _rng_tensor((8, 4), 22)},
            r=4, alpha=8,
            extra_config={"rank_pattern": {"q_proj": 4}, "alpha_pattern": {"q_proj": 8}},
        )
        b = _make_adapter(
            tmp_path / "math", "meta/x",
            {ak: _rng_tensor((8, 16), 23), bk: _rng_tensor((8, 8), 24)}, r=8, alpha=16,
        )
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
        assert cfg["rank_pattern"] == {}
        assert cfg["alpha_pattern"] == {}

    def test_mixed_rank_zero_alpha_contribution_zeroed(self, tmp_path):
        # An adapter with lora_alpha=0 (scaling 0.0) must contribute nothing —
        # the CLI must thread the real 0.0, not `or 1.0` it back to full.
        ak, bk = _lora_keys()
        a1 = _rng_tensor((4, 16), 25)
        b1 = _rng_tensor((8, 4), 26)
        a2 = _rng_tensor((8, 16), 27)
        b2 = _rng_tensor((8, 8), 28)
        a = _make_adapter(tmp_path / "coder", "meta/x", {ak: a1, bk: b1}, r=4, alpha=0)
        b = _make_adapter(tmp_path / "math", "meta/x", {ak: a2, bk: b2}, r=8, alpha=8)
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        from safetensors.numpy import load_file

        merged = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
        delta = merged[bk].astype("float64") @ merged[ak].astype("float64")
        # coder scaling 0/4 = 0 -> only math contributes: (8/8=1.0)*b2@a2
        assert np.allclose(delta, b2.astype("float64") @ a2.astype("float64"), atol=1e-3)

    def test_same_rank_unchanged_element_wise(self, tmp_path):
        # Same-rank inputs still route through the fast element-wise path
        # (√|c| split) — config r unchanged at 8.
        ak, bk = _lora_keys()
        a = _make_adapter(
            tmp_path / "coder",
            "meta/x",
            {ak: _rng_tensor((8, 16), 9), bk: _rng_tensor((8, 8), 10)},
            r=8,
        )
        b = _make_adapter(
            tmp_path / "math",
            "meta/x",
            {ak: _rng_tensor((8, 16), 11), bk: _rng_tensor((8, 8), 12)},
            r=8,
        )
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
        assert cfg["r"] == 8  # unchanged — element-wise path

    def test_same_rank_with_rank_flag_truncates_via_concat(self, tmp_path):
        # --rank on same-rank inputs is NOT ignored: it routes through the concat
        # path and truncates the output rank (help text now says so).
        ak, bk = _lora_keys()
        a = _make_adapter(
            tmp_path / "coder",
            "meta/x",
            {ak: _rng_tensor((8, 16), 13), bk: _rng_tensor((8, 8), 14)},
            r=8,
        )
        b = _make_adapter(
            tmp_path / "math",
            "meta/x",
            {ak: _rng_tensor((8, 16), 15), bk: _rng_tensor((8, 8), 16)},
            r=8,
        )
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "--rank", "4", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
        assert cfg["r"] == 4  # truncated to --rank via SVD, not "ignored"
        from safetensors.numpy import load_file

        merged = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
        assert merged[ak].shape == (4, 16)
