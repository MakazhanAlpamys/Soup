"""LoRA scaling across ``soup adapters merge`` / ``adapters arithmetic`` inputs.

The merged adapter is written with the first input's ``adapter_config.json``, so
PEFT decodes every merged module at that adapter's scaling. Inputs trained with a
different ``lora_alpha`` / ``use_rslora`` / ``alpha_pattern`` must be rescaled to
it, or they contribute at the wrong strength.

Each fixture adapter puts its delta in its own rank slot (``A[s, s] = 1``,
``B[s, s] = value``), so the element-wise sum ``(B_a + B_b) @ (A_a + A_b)`` has
no cross terms and the expected effective delta is exact.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

pytest.importorskip("safetensors")

_STEM = "base_model.model.model.layers.0.self_attn.q_proj"
_A = f"{_STEM}.lora_A.weight"
_B = f"{_STEM}.lora_B.weight"
_DIM = 4


def _make_adapter(path, *, r, alpha, slot, value, **extra):
    from safetensors.numpy import save_file

    path.mkdir(parents=True)
    a = np.zeros((r, _DIM), dtype=np.float32)
    b = np.zeros((_DIM, r), dtype=np.float32)
    a[slot, slot] = 1.0
    b[slot, slot] = value
    save_file({_A: a, _B: b}, str(path / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "tiny/base",
        "r": r,
        "lora_alpha": alpha,
        "target_modules": ["q_proj"],
        **extra,
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return path.name


def _effective_delta(path, scaling):
    """``scaling * B @ A`` of q_proj, read straight from the written adapter."""
    from safetensors.numpy import load_file

    tensors = load_file(str(path / "adapter_model.safetensors"))
    return scaling * (tensors[_B].astype(np.float64) @ tensors[_A].astype(np.float64))


def _written_scaling(path):
    """lora_alpha / r of a plain (no rsLoRA, no pattern) written config."""
    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    assert not cfg.get("use_rslora")
    assert not cfg.get("alpha_pattern") and not cfg.get("rank_pattern")
    return cfg["lora_alpha"] / cfg["r"]


def _diag(*values):
    out = np.zeros((_DIM, _DIM))
    for i, v in enumerate(values):
        out[i, i] = v
    return out


def _run(args, cwd):
    from typer.testing import CliRunner

    from soup_cli.commands.adapters import app

    old = os.getcwd()
    os.chdir(cwd)
    try:
        return CliRunner().invoke(app, args)
    finally:
        os.chdir(old)


@pytest.fixture
def pair(tmp_path):
    """a: scaling 1, delta e0 e0^T.  b: scaling 4 (alpha 8 / r 2), delta 4 e1 e1^T."""
    _make_adapter(tmp_path / "a", r=2, alpha=2, slot=0, value=1.0)
    _make_adapter(tmp_path / "b", r=2, alpha=8, slot=1, value=1.0)
    return tmp_path


class TestArithmeticSameRank:
    def test_sum_keeps_each_adapters_scaling(self, pair):
        res = _run(
            ["arithmetic", "a + b", "--adapter", "a=a", "--adapter", "b=b",
             "-o", "out", "--allow-unscanned"],
            pair,
        )
        assert res.exit_code == 0, res.output
        out = pair / "out"
        got = _effective_delta(out, _written_scaling(out))
        np.testing.assert_allclose(got, _diag(1.0, 4.0), atol=1e-6)

    def test_result_does_not_depend_on_operand_order(self, pair):
        for expr, out in (("a + b", "ab"), ("b + a", "ba")):
            res = _run(
                ["arithmetic", expr, "--adapter", "a=a", "--adapter", "b=b",
                 "-o", out, "--allow-unscanned"],
                pair,
            )
            assert res.exit_code == 0, res.output
        ab = _effective_delta(pair / "ab", _written_scaling(pair / "ab"))
        ba = _effective_delta(pair / "ba", _written_scaling(pair / "ba"))
        np.testing.assert_allclose(ab, ba, atol=1e-6)

    def test_negation_removes_the_adapters_own_delta(self, pair):
        res = _run(
            ["arithmetic", "a - b", "--adapter", "a=a", "--adapter", "b=b",
             "-o", "out", "--allow-unscanned"],
            pair,
        )
        assert res.exit_code == 0, res.output
        out = pair / "out"
        got = _effective_delta(out, _written_scaling(out))
        np.testing.assert_allclose(got, _diag(1.0, -4.0), atol=1e-6)


class TestMerge:
    def test_weight_one_reproduces_the_selected_adapter(self, pair):
        res = _run(
            ["merge", "a", "b", "-o", "out", "--weights", "0,1", "--allow-unscanned"],
            pair,
        )
        assert res.exit_code == 0, res.output
        out = pair / "out"
        got = _effective_delta(out, _written_scaling(out))
        np.testing.assert_allclose(got, _diag(0.0, 4.0), atol=1e-6)

    def test_rslora_input_is_rescaled(self, tmp_path):
        _make_adapter(tmp_path / "a", r=2, alpha=2, slot=0, value=1.0)
        # rsLoRA: scaling 2 / sqrt(2)
        _make_adapter(tmp_path / "e", r=2, alpha=2, slot=1, value=1.0, use_rslora=True)
        res = _run(
            ["merge", "a", "e", "-o", "out", "--weights", "0,1", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 0, res.output
        out = tmp_path / "out"
        got = _effective_delta(out, _written_scaling(out))
        np.testing.assert_allclose(got, _diag(0.0, math.sqrt(2.0)), atol=1e-6)

    def test_template_alpha_pattern_is_honoured(self, tmp_path):
        # The template decodes q_proj at alpha_pattern 4 / r 2 = 2, not 2 / 2 = 1.
        _make_adapter(
            tmp_path / "a", r=2, alpha=2, slot=0, value=1.0, alpha_pattern={"q_proj": 4},
        )
        _make_adapter(tmp_path / "b", r=2, alpha=2, slot=1, value=1.0)
        res = _run(
            ["merge", "a", "b", "-o", "out", "--weights", "0,1", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 0, res.output
        cfg = json.loads(
            (tmp_path / "out" / "adapter_config.json").read_text(encoding="utf-8")
        )
        assert cfg["alpha_pattern"] == {"q_proj": 4}
        got = _effective_delta(tmp_path / "out", 4 / 2)
        np.testing.assert_allclose(got, _diag(0.0, 1.0), atol=1e-6)

    def test_equal_scaling_merge_is_unchanged(self, tmp_path):
        # Same lora_alpha: nothing to fold, so the output is the plain average.
        _make_adapter(tmp_path / "a", r=2, alpha=8, slot=0, value=3.0)
        _make_adapter(tmp_path / "b", r=2, alpha=8, slot=1, value=5.0)
        res = _run(["merge", "a", "b", "-o", "out", "--allow-unscanned"], tmp_path)
        assert res.exit_code == 0, res.output
        from safetensors.numpy import load_file

        got = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
        expected_b = np.zeros((_DIM, 2), dtype=np.float32)
        expected_b[0, 0], expected_b[1, 1] = 1.5, 2.5
        assert np.array_equal(got[_B], expected_b)


class TestEveryStrategy:
    """The fold runs before the strategy, so linear, ties, dare and svd all see it.

    ``a`` and ``b`` hold the SAME effective delta, ``4 e1 e1^T``, under two scalings:
    ``a`` as ``B = 4`` at scaling 1, ``b`` as ``B = 1`` at scaling 4. Any of the four
    strategies merging two equal deltas must give that delta back. Without the fold the
    raw ``B`` tensors average to 2.5, decoded at ``a``'s scaling 1.
    """

    @pytest.mark.parametrize(
        "extra",
        [
            ["--strategy", "linear"],
            ["--strategy", "ties"],
            # density 1.0 keeps every element, so the random drop cannot move the result.
            ["--strategy", "dare", "--density", "1.0"],
            ["--strategy", "svd", "--rank", "1"],
        ],
        ids=["linear", "ties", "dare", "svd"],
    )
    def test_equal_deltas_under_two_scalings_merge_to_that_delta(self, tmp_path, extra):
        _make_adapter(tmp_path / "a", r=2, alpha=2, slot=1, value=4.0)
        _make_adapter(tmp_path / "b", r=2, alpha=8, slot=1, value=1.0)
        res = _run(["merge", "a", "b", "-o", "out", "--allow-unscanned", *extra], tmp_path)
        assert res.exit_code == 0, res.output
        out = tmp_path / "out"
        got = _effective_delta(out, _written_scaling(out))
        np.testing.assert_allclose(got, _diag(0.0, 4.0), atol=1e-5)


class TestZeroScaling:
    def test_a_template_with_zero_scaling_passes_every_input_through(self, tmp_path):
        """``lora_alpha: 0`` on the template: the merged adapter decodes every module at
        scaling 0, whatever its tensors hold, so there is no ratio to fold to. The inputs
        are merged as they are, as before, and the output keeps the template's 0."""
        _make_adapter(tmp_path / "a", r=2, alpha=0, slot=0, value=1.0)
        _make_adapter(tmp_path / "b", r=2, alpha=8, slot=1, value=3.0)
        res = _run(
            ["merge", "a", "b", "-o", "out", "--weights", "0,1", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 0, res.output
        from safetensors.numpy import load_file

        got = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
        expected_b = np.zeros((_DIM, 2), dtype=np.float32)
        expected_b[1, 1] = 3.0
        assert np.array_equal(got[_B], expected_b)
        cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text(encoding="utf-8"))
        assert cfg["lora_alpha"] == 0

    def test_an_input_with_zero_scaling_contributes_nothing(self, tmp_path):
        """An input whose own scaling is 0 decoded to no delta, and keeps contributing none
        under the template's scaling: its ``lora_B`` is multiplied by 0 / template."""
        _make_adapter(tmp_path / "a", r=2, alpha=2, slot=0, value=1.0)
        _make_adapter(tmp_path / "z", r=2, alpha=0, slot=1, value=5.0)
        res = _run(
            ["merge", "a", "z", "-o", "out", "--weights", "0,1", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 0, res.output
        out = tmp_path / "out"
        got = _effective_delta(out, _written_scaling(out))
        np.testing.assert_allclose(got, np.zeros((_DIM, _DIM)), atol=1e-12)


class TestOtherLoraTensors:
    def _configs(self, tmp_path, monkeypatch, **extra):
        # The config reader refuses a directory outside the working directory, as
        # the CLI does, so these direct calls run from tmp_path like the CLI tests.
        monkeypatch.chdir(tmp_path)
        a, b = tmp_path / "a", tmp_path / "b"
        for path, alpha in ((a, 2), (b, 8)):
            path.mkdir()
            (path / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": alpha, **extra}),
                encoding="utf-8",
            )
        return [str(a), str(b)]

    def test_lora_embedding_b_is_rescaled_like_lora_b(self, tmp_path, monkeypatch):
        """PEFT's LoRA embedding adds ``(x @ A^T) @ B^T * scaling``, the same scaling as a
        linear layer, so ``lora_embedding_B`` is folded and ``lora_embedding_A`` is not."""
        from soup_cli.utils.adapter_merge import fold_lora_scaling

        stem = "base_model.model.model.embed_tokens"
        emb_a = np.ones((2, 10), dtype=np.float32)
        emb_b = np.ones((_DIM, 2), dtype=np.float32)
        weights = {f"{stem}.lora_embedding_A": emb_a, f"{stem}.lora_embedding_B": emb_b}
        folded = fold_lora_scaling([weights, weights], self._configs(tmp_path, monkeypatch))

        assert folded[0] is weights
        np.testing.assert_array_equal(folded[1][f"{stem}.lora_embedding_B"], emb_b * 4.0)
        np.testing.assert_array_equal(folded[1][f"{stem}.lora_embedding_A"], emb_a)

    def test_dora_magnitude_vector_is_left_as_it_is(self, tmp_path, monkeypatch):
        """DoRA's ``lora_magnitude_vector`` is not multiplied by the LoRA scaling in PEFT's
        forward: it rescales ``W + scaling * B @ A`` column by column. Folding ``lora_B``
        keeps each input's ``scaling * B @ A``, so the magnitude stays as it was and goes
        through the strategy as before (an average for ``linear``)."""
        from soup_cli.utils.adapter_merge import fold_lora_scaling

        magnitude = np.full((_DIM,), 7.0, dtype=np.float32)
        b = np.ones((_DIM, 2), dtype=np.float32)
        weights = {_B: b, f"{_STEM}.lora_magnitude_vector": magnitude}
        dirs = self._configs(tmp_path, monkeypatch, use_dora=True)
        folded = fold_lora_scaling([weights, weights], dirs)

        np.testing.assert_array_equal(folded[1][_B], b * 4.0)
        assert folded[1][f"{_STEM}.lora_magnitude_vector"] is magnitude


class TestCmaesCandidates:
    def test_candidate_is_scored_with_the_rescaled_merge(self, pair):
        from soup_cli.utils.cmaes_merge import build_cmaes_eval_fn, build_cmaes_plan

        (pair / "suite.jsonl").write_text(
            json.dumps({"prompt": "p", "expected": "e"}) + "\n", encoding="utf-8"
        )
        seen = {}

        def scorer(merged_dir, _suite):
            from pathlib import Path

            seen["delta"] = _effective_delta(Path(merged_dir), _written_scaling(Path(merged_dir)))
            return 0.5

        old = os.getcwd()
        os.chdir(pair)
        try:
            plan = build_cmaes_plan(
                adapters=["a", "b"], eval_suite="suite.jsonl", budget_spec="60s",
                population_size=2, max_generations=1, seed=0,
            )
            build_cmaes_eval_fn(plan, scorer=scorer)((0.0, 1.0))
        finally:
            os.chdir(old)
        np.testing.assert_allclose(seen["delta"], _diag(0.0, 4.0), atol=1e-6)


class TestModuleScaling:
    def test_plain_and_rslora(self):
        from soup_cli.utils.adapter_diff import lora_module_scaling

        assert lora_module_scaling({"r": 8, "lora_alpha": 16}, "m.q_proj") == 2.0
        assert lora_module_scaling(
            {"r": 16, "lora_alpha": 16, "use_rslora": True}, "m.q_proj"
        ) == pytest.approx(4.0)

    def test_patterns_match_like_peft(self):
        from soup_cli.utils.adapter_diff import lora_module_scaling

        cfg = {
            "r": 8, "lora_alpha": 8,
            "rank_pattern": {"v_proj": 4}, "alpha_pattern": {"layers.0.self_attn.q_proj": 32},
        }
        assert lora_module_scaling(cfg, "model.layers.0.self_attn.q_proj") == 4.0
        assert lora_module_scaling(cfg, "model.layers.1.self_attn.q_proj") == 1.0
        assert lora_module_scaling(cfg, "model.layers.0.self_attn.v_proj") == 2.0
        # A suffix that is not a whole dotted component does not match.
        assert lora_module_scaling(cfg, "model.layers.0.self_attn.xv_proj") == 1.0

    def test_unusable_config_gives_none(self):
        from soup_cli.utils.adapter_diff import lora_module_scaling

        assert lora_module_scaling({}, "m") is None
        assert lora_module_scaling({"r": 0, "lora_alpha": 8}, "m") is None
        assert lora_module_scaling({"r": True, "lora_alpha": 8}, "m") is None
