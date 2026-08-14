"""v0.71.30 — PRM-guided GRPO + bundled rollout envs.

Tests the pure PRM reward kernels (split_steps / aggregate_step_scores), the
schema fields + cross-validators, the torch-lazy PRMScorer (safetensors head
load + scoring), the GRPO wiring, the bundled rollout envs, and the recipes.
"""
import ast
import inspect

import pytest

from soup_cli.utils.prm_reward import (
    AGGREGATE_MODES,
    aggregate_step_scores,
    split_steps,
)


# ---------------------------------------------------------------------------
# Task 1 — pure kernels
# ---------------------------------------------------------------------------
class TestSplitSteps:
    def test_splits_on_newlines(self):
        assert split_steps("a\nb\nc") == ["a", "b", "c"]

    def test_drops_empty_and_whitespace(self):
        assert split_steps("a\n\n  \nb\n") == ["a", "b"]

    def test_strips_each_step(self):
        assert split_steps("  a  \n\tb\t") == ["a", "b"]

    def test_non_string_returns_empty(self):
        assert split_steps(None) == []
        assert split_steps(123) == []

    def test_empty_returns_empty(self):
        assert split_steps("") == []
        assert split_steps("   \n  ") == []

    def test_caps_step_count(self):
        from soup_cli.utils.prm_reward import _MAX_STEPS

        text = "\n".join(str(i) for i in range(_MAX_STEPS + 50))
        assert len(split_steps(text)) == _MAX_STEPS

    def test_caps_step_chars(self):
        from soup_cli.utils.prm_reward import _MAX_STEP_CHARS

        long = "x" * (_MAX_STEP_CHARS + 100)
        out = split_steps(long)
        assert len(out) == 1
        assert len(out[0]) == _MAX_STEP_CHARS


class TestAggregate:
    def test_min(self):
        assert aggregate_step_scores([0.9, 0.2, 0.7], "min") == pytest.approx(0.2)

    def test_last(self):
        assert aggregate_step_scores([0.9, 0.2, 0.7], "last") == pytest.approx(0.7)

    def test_prod(self):
        assert aggregate_step_scores([0.5, 0.5, 0.5], "prod") == pytest.approx(0.125)

    def test_empty_returns_zero(self):
        assert aggregate_step_scores([], "min") == 0.0
        assert aggregate_step_scores([], "prod") == 0.0
        assert aggregate_step_scores([], "last") == 0.0

    def test_single(self):
        assert aggregate_step_scores([0.42], "min") == pytest.approx(0.42)

    def test_non_finite_is_safe(self):
        # NaN / inf must not propagate — coerced to 0.0
        out = aggregate_step_scores([float("nan"), 0.5], "min")
        assert out == 0.0

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError, match="min|prod|last"):
            aggregate_step_scores([0.5], "mean")

    def test_bool_mode_rejected(self):
        with pytest.raises(ValueError):
            aggregate_step_scores([0.5], True)

    def test_aggregate_modes_constant(self):
        assert set(AGGREGATE_MODES) == {"min", "prod", "last"}


# ---------------------------------------------------------------------------
# Task 2 — schema fields + cross-validators
# ---------------------------------------------------------------------------
def _prm_yaml(
    *,
    task: str = "grpo",
    backend: str = "transformers",
    modality: str = "text",
    prm_reward: str | None = "./prm",
    prm_aggregate: str | None = None,
) -> str:
    lines = [
        "base: HuggingFaceTB/SmolLM2-135M",
        f"task: {task}",
        f"backend: {backend}",
        f"modality: {modality}",
        "data:",
        "  train: ./data/train.jsonl",
        "  format: chatml",
        "training:",
    ]
    if prm_reward is not None:
        lines.append(f"  prm_reward: {prm_reward}")
    if prm_aggregate is not None:
        lines.append(f"  prm_aggregate: {prm_aggregate}")
    return "\n".join(lines) + "\n"


class TestPrmSchema:
    def test_default_fields(self):
        from soup_cli.config.schema import TrainingConfig

        tc = TrainingConfig()
        assert tc.prm_reward is None
        assert tc.prm_aggregate == "min"

    def test_happy_grpo_parses(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_prm_yaml(prm_aggregate="prod"))
        assert cfg.training.prm_reward == "./prm"
        assert cfg.training.prm_aggregate == "prod"

    def test_rejects_non_grpo_task(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="task='grpo'"):
            load_config_from_string(_prm_yaml(task="sft"))

    def test_rejects_mlx_backend(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="transformers"):
            load_config_from_string(_prm_yaml(backend="mlx"))

    def test_rejects_unsloth_backend(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="transformers"):
            load_config_from_string(_prm_yaml(backend="unsloth"))

    def test_rejects_non_text_modality(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="modality='text'"):
            load_config_from_string(_prm_yaml(modality="vision"))

    def test_aggregate_without_prm_reward_is_footgun(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="prm_reward"):
            load_config_from_string(
                _prm_yaml(prm_reward=None, prm_aggregate="prod")
            )

    def test_default_aggregate_without_prm_reward_ok(self):
        from soup_cli.config.loader import load_config_from_string

        # prm_aggregate at its default is fine even without prm_reward.
        cfg = load_config_from_string(
            _prm_yaml(task="sft", prm_reward=None, prm_aggregate="min")
        )
        assert cfg.training.prm_reward is None

    def test_rejects_empty_string(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError, match="empty"):
            TrainingConfig(prm_reward="")

    def test_rejects_null_byte(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError, match="null"):
            TrainingConfig(prm_reward="./prm\x00evil")

    def test_rejects_oversize(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError, match="512|chars"):
            TrainingConfig(prm_reward="x" * 5000)

    def test_prm_aggregate_invalid_literal_rejected_by_schema(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError, match="prm_aggregate"):
            TrainingConfig(prm_aggregate="mean")


# ---------------------------------------------------------------------------
# Task 3 — PRMScorer + build_prm_reward_fn
# ---------------------------------------------------------------------------
class TestLoadRewardHeadWeights:
    def test_loads_head_from_safetensors(self, tmp_path):
        import torch
        from safetensors.torch import save_file

        from soup_cli.utils.prm_reward import load_reward_head_weights

        save_file(
            {
                "model.embed": torch.zeros(4, 4),
                "reward_head.weight": torch.ones(1, 8),
                "reward_head.bias": torch.zeros(1),
            },
            str(tmp_path / "model.safetensors"),
        )
        head = load_reward_head_weights(str(tmp_path))
        assert set(head.keys()) == {"weight", "bias"}
        assert tuple(head["weight"].shape) == (1, 8)

    def test_missing_head_rejected(self, tmp_path):
        import torch
        from safetensors.torch import save_file

        from soup_cli.utils.prm_reward import load_reward_head_weights

        save_file({"model.embed": torch.zeros(4, 4)}, str(tmp_path / "model.safetensors"))
        with pytest.raises(ValueError, match="reward_head|Soup-trained PRM"):
            load_reward_head_weights(str(tmp_path))

    def test_non_directory_rejected(self, tmp_path):
        from soup_cli.utils.prm_reward import load_reward_head_weights

        with pytest.raises(ValueError, match="directory"):
            load_reward_head_weights(str(tmp_path / "does_not_exist"))


def _make_fake_scorer(aggregate="min"):
    """A PRMScorer with an injected fake model + tokenizer (no network).

    Fake model: hidden_states[-1] = arange(T) broadcast over H, and
    reward_head averages H → each step boundary scores its token position.
    Fake tokenizer: one token per whitespace word (>=1).
    """
    from types import SimpleNamespace

    import torch
    from torch import nn

    from soup_cli.utils.prm_reward import PRMScorer

    hidden = 8

    class _FakeTok:
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __call__(self, text, add_special_tokens=False):
            n = max(1, len(text.split())) if text else 0
            return {"input_ids": [1] * n}

    head = nn.Linear(hidden, 1, bias=True)
    with torch.no_grad():
        head.weight.copy_(torch.ones(1, hidden) / hidden)
        head.bias.zero_()

    class _FakeModel:
        reward_head = head

        def __call__(self, input_ids, output_hidden_states=False, **kwargs):
            seq_len = input_ids.shape[1]
            batch = input_ids.shape[0]
            hs = torch.arange(seq_len).float().reshape(1, seq_len, 1).expand(batch, seq_len, hidden)
            return SimpleNamespace(hidden_states=[hs])

    scorer = PRMScorer("./prm", aggregate=aggregate, device="cpu")
    scorer._model = _FakeModel()
    scorer._tokenizer = _FakeTok()
    return scorer


class TestPRMScorer:
    def test_name_is_prm_reward(self):
        s = _make_fake_scorer()
        assert s.__name__ == "prm_reward"

    def test_bad_aggregate_rejected(self):
        from soup_cli.utils.prm_reward import PRMScorer

        with pytest.raises(ValueError, match="min|prod|last"):
            PRMScorer("./prm", aggregate="mean")

    def test_scores_step_boundaries_min(self):
        s = _make_fake_scorer("min")
        # completion "a b\nc d e" -> steps ["a b"(2 tok), "c d e"(3 tok)]
        # boundaries at positions 1 and 4 -> scores [1.0, 4.0] -> min 1.0
        out = s([[{"role": "assistant", "content": "a b\nc d e"}]])
        assert out == pytest.approx([1.0])

    def test_scores_last(self):
        s = _make_fake_scorer("last")
        out = s([[{"role": "assistant", "content": "a b\nc d e"}]])
        assert out == pytest.approx([4.0])

    def test_multiple_completions_len(self):
        s = _make_fake_scorer("min")
        out = s(["a\nb", "c\nd\ne", "x"])
        assert len(out) == 3
        assert all(isinstance(v, float) for v in out)

    def test_empty_completion_scores_zero(self):
        s = _make_fake_scorer("min")
        out = s([""])
        assert out == [0.0]

    def test_prompt_prepended_as_context(self):
        # With a prompt prefix of 3 tokens, boundaries shift by +3.
        s = _make_fake_scorer("min")
        out = s(
            [[{"role": "assistant", "content": "a b\nc d e"}]],
            prompts=[[{"role": "user", "content": "one two three"}]],
        )
        # prefix len 3 -> boundaries at 4 and 7 -> min 4.0
        assert out == pytest.approx([4.0])

    def test_prompt_multi_message_joined(self):
        # Two non-empty prompt messages join with "\n" -> 1 + 2 = 3 prefix tokens.
        s = _make_fake_scorer("min")
        out = s(
            [[{"role": "assistant", "content": "a"}]],
            prompts=[
                [
                    {"role": "system", "content": "one"},
                    {"role": "user", "content": "two three"},
                ]
            ],
        )
        # prefix 3 tokens, single step "a" (1 tok) -> boundary at position 3
        assert out == pytest.approx([3.0])

    def test_string_prompt_context(self):
        s = _make_fake_scorer("min")
        out = s(["a"], prompts=["one two"])
        # prefix 2 tokens + step "a" -> boundary at position 2
        assert out == pytest.approx([2.0])

    def test_bare_dict_completion(self):
        # _completion_text dict branch (completion not wrapped in a list).
        s = _make_fake_scorer("min")
        out = s([{"role": "assistant", "content": "a b\nc d e"}])
        assert out == pytest.approx([1.0])

    def test_tuple_completion(self):
        # _completion_text tuple branch -> "".join -> single step.
        s = _make_fake_scorer("last")
        out = s([("a", "b")])
        assert len(out) == 1
        assert isinstance(out[0], float)

    def test_batched_parity_with_per_completion(self):
        s = _make_fake_scorer("min")
        completions = ["a\nb", "c\nd\ne"]
        batched = s(completions)
        for i, c in enumerate(completions):
            single = s([c])
            assert abs(batched[i] - single[0]) < 1e-5, f"mismatch at {i}"

    def test_mixed_length_no_cross_row_contamination(self):
        s = _make_fake_scorer("min")
        short = s(["a\nb"])
        long_ = s(["c\nd\ne\nf\ng"])
        both = s(["a\nb", "c\nd\ne\nf\ng"])
        assert abs(both[0] - short[0]) < 1e-5
        assert abs(both[1] - long_[0]) < 1e-5

def _make_capped_scorer(max_pos, aggregate="min"):
    """PRMScorer whose fake model advertises a tiny max_position_embeddings."""
    from types import SimpleNamespace

    import torch
    from torch import nn

    from soup_cli.utils.prm_reward import PRMScorer

    hidden = 8

    class _FakeTok:
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __call__(self, text, add_special_tokens=False):
            n = max(1, len(text.split())) if text else 0
            return {"input_ids": [1] * n}

    head = nn.Linear(hidden, 1, bias=True)
    with torch.no_grad():
        head.weight.copy_(torch.ones(1, hidden) / hidden)
        head.bias.zero_()

    class _FakeModelCapped:
        reward_head = head
        config = SimpleNamespace(max_position_embeddings=max_pos)

        def __call__(self, input_ids, output_hidden_states=False, **kwargs):
            seq_len = input_ids.shape[1]
            batch = input_ids.shape[0]
            hs = torch.arange(seq_len).float().reshape(1, seq_len, 1).expand(batch, seq_len, hidden)
            return SimpleNamespace(hidden_states=[hs])

    scorer = PRMScorer("./prm", aggregate=aggregate, device="cpu")
    scorer._model = _FakeModelCapped()
    scorer._tokenizer = _FakeTok()
    return scorer

class TestPRMScorerInputCap:
    def test_truncates_to_max_position_embeddings(self):
        # steps "a b"(2 tok)->boundary 1, "c d"(2 tok)->boundary 3; cap=3 keeps
        # only positions < 3, dropping boundary 3.
        s = _make_capped_scorer(max_pos=3, aggregate="min")
        out = s([[{"role": "assistant", "content": "a b\nc d"}]])
        assert out == pytest.approx([1.0])

    def test_cap_below_first_boundary_returns_zero(self):
        # cap=0 -> no boundary survives -> 0.0
        s = _make_capped_scorer(max_pos=0, aggregate="min")
        out = s([[{"role": "assistant", "content": "a b\nc d"}]])
        assert out == [0.0]


class TestBuildPrmRewardFn:
    def test_returns_named_scorer(self, monkeypatch):
        import soup_cli.utils.prm_reward as mod

        monkeypatch.setattr(mod, "_resolve_trust", lambda *a, **k: False)

        class _T:
            prm_reward = "some/hf-id-not-on-disk"
            prm_aggregate = "prod"

        fn = mod.build_prm_reward_fn(_T(), device="cpu", trust_remote_code=False)
        assert fn.__name__ == "prm_reward"
        assert fn.aggregate == "prod"

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        import soup_cli.utils.prm_reward as mod

        monkeypatch.setattr(mod, "_resolve_trust", lambda *a, **k: False)
        # tmp_path is outside cwd and exists on disk -> containment reject.

        class _T:
            prm_reward = str(tmp_path)
            prm_aggregate = "min"

        with pytest.raises(ValueError, match="working directory"):
            mod.build_prm_reward_fn(_T(), device="cpu", trust_remote_code=False)

    def test_none_path_rejected(self):
        import soup_cli.utils.prm_reward as mod

        class _T:
            prm_reward = None
            prm_aggregate = "min"

        with pytest.raises(ValueError, match="prm_reward=None"):
            mod.build_prm_reward_fn(_T(), device="cpu", trust_remote_code=False)

    def test_local_existing_dir_under_cwd_accepted(self, tmp_path, monkeypatch):
        import os

        import soup_cli.utils.prm_reward as mod

        monkeypatch.setattr(mod, "_resolve_trust", lambda *a, **k: False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "my_prm").mkdir()

        class _T:
            prm_reward = "./my_prm"
            prm_aggregate = "min"

        fn = mod.build_prm_reward_fn(_T(), device="cpu", trust_remote_code=False)
        assert fn.prm_path == os.path.realpath(str(tmp_path / "my_prm"))


class TestResolveTrust:
    def test_delegates_with_requires_flag(self, monkeypatch):
        import soup_cli.utils.trust_remote as trust_mod
        from soup_cli.utils.prm_reward import _resolve_trust

        captured = {}
        monkeypatch.setattr(
            trust_mod, "model_requires_trust_remote_code", lambda base: True
        )

        def _fake_resolve(base, requested, console, requires_remote_code):
            captured["requires"] = requires_remote_code
            captured["requested"] = requested
            return True

        monkeypatch.setattr(trust_mod, "resolve_trust_remote_code", _fake_resolve)
        assert _resolve_trust("some/base", False, console=object()) is True
        assert captured["requires"] is True
        assert captured["requested"] is False


class TestLoadRewardHeadMultiShard:
    def test_collects_head_across_shards(self, tmp_path):
        import torch
        from safetensors.torch import save_file

        from soup_cli.utils.prm_reward import load_reward_head_weights

        # Head split across two shards (weight in one, bias in the other).
        save_file(
            {"model.a": torch.zeros(2, 2), "reward_head.weight": torch.ones(1, 8)},
            str(tmp_path / "model-00001-of-00002.safetensors"),
        )
        save_file(
            {"model.b": torch.zeros(2, 2), "reward_head.bias": torch.zeros(1)},
            str(tmp_path / "model-00002-of-00002.safetensors"),
        )
        head = load_reward_head_weights(str(tmp_path))
        assert set(head.keys()) == {"weight", "bias"}


# ---------------------------------------------------------------------------
# Task 5 — bundled rollout envs
# ---------------------------------------------------------------------------
_ENV_MODULES = ["calculator", "retrieval_qa", "guess_number"]
_NEW_RECIPES = ["grpo-env-calculator", "grpo-env-retrieval-qa", "grpo-env-guess-number"]


class TestEnvs:
    @pytest.mark.parametrize("modname", _ENV_MODULES)
    def test_rows_normalise(self, modname):
        import importlib

        from soup_cli.utils.agent_rollout import _normalise_rollout_rows

        mod = importlib.import_module(f"soup_cli.envs.{modname}")
        rows = mod.rollout([])
        assert rows, "env must produce a non-empty row set"
        norm = _normalise_rollout_rows(rows, "openenv")
        assert len(norm) == len(rows)
        for row in rows:
            assert isinstance(row["prompt"], str) and row["prompt"]
            assert isinstance(row["answer"], str) and row["answer"]

    @pytest.mark.parametrize("modname", _ENV_MODULES)
    def test_deterministic(self, modname):
        import importlib

        mod = importlib.import_module(f"soup_cli.envs.{modname}")
        assert mod.rollout([]) == mod.rollout([])

    @pytest.mark.parametrize("modname", _ENV_MODULES)
    def test_rollout_signature_ignores_prompt_content(self, modname):
        import importlib

        mod = importlib.import_module(f"soup_cli.envs.{modname}")
        # Passing seed prompts must not crash and stays deterministic.
        assert mod.rollout(["seed a", "seed b"]) == mod.rollout(["x", "y"])

    def test_calculator_answers_correct(self):
        import re

        from soup_cli.envs.calculator import rollout

        for row in rollout([]):
            m = re.search(r"(-?\d+)\s*([+\-*])\s*(-?\d+)", row["prompt"])
            assert m is not None, row["prompt"]
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            expected = {"+": a + b, "-": a - b, "*": a * b}[op]
            assert row["answer"] == str(expected)

    def test_guess_number_answer_matches_product(self):
        import re

        from soup_cli.envs.guess_number import rollout

        for row in rollout([]):
            m = re.search(r"equals (\d+) times (\d+)", row["prompt"])
            assert m is not None, row["prompt"]
            a, b = int(m.group(1)), int(m.group(2))
            assert row["answer"] == str(a * b)

    def test_retrieval_qa_answer_matches_asked_fact(self):
        import re

        from soup_cli.envs.retrieval_qa import rollout

        for row in rollout([]):
            # Tie the answer to the SPECIFIC asked fact, not just "any embedded
            # value" — a wrong-index pairing bug must fail here.
            m = re.search(r"completes '(.+?) ___'", row["prompt"])
            assert m is not None, row["prompt"]
            entity_attr = m.group(1)
            assert f"{entity_attr} {row['answer']}." in row["prompt"], row

    @pytest.mark.parametrize("modname", _ENV_MODULES)
    def test_row_count_is_default(self, modname):
        import importlib

        from soup_cli.envs._common import DEFAULT_ROWS

        mod = importlib.import_module(f"soup_cli.envs.{modname}")
        assert len(mod.rollout([])) == DEFAULT_ROWS

    def test_envs_produce_distinct_output(self):
        # Guard against an accidental seed / module copy-paste across envs.
        from soup_cli.envs import calculator, guess_number, retrieval_qa

        outs = [
            tuple(r["prompt"] for r in calculator.rollout([])),
            tuple(r["prompt"] for r in guess_number.rollout([])),
            tuple(r["prompt"] for r in retrieval_qa.rollout([])),
        ]
        assert len(set(outs)) == 3


class TestRecipeRolloutFunc:
    _EXPECTED_REWARD = {
        "grpo-env-calculator": ("verifiable", "math"),
        "grpo-env-retrieval-qa": ("accuracy", None),
        "grpo-env-guess-number": ("verifiable", "math"),
    }

    @pytest.mark.parametrize("name", _NEW_RECIPES)
    def test_rollout_func_resolves_and_runs(self, name):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe
        from soup_cli.utils.agent_rollout import resolve_rollout_func

        cfg = load_config_from_string(get_recipe(name).yaml_str)
        fn = resolve_rollout_func(cfg.training.rollout_func)
        rows = fn([])
        assert rows, "resolved rollout_func must produce rows"

    @pytest.mark.parametrize("name", _NEW_RECIPES)
    def test_recipe_reward_matches_env(self, name):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        cfg = load_config_from_string(get_recipe(name).yaml_str)
        expected_fn, expected_domain = self._EXPECTED_REWARD[name]
        assert cfg.training.reward_fn == expected_fn
        assert cfg.training.verifiable_domain == expected_domain


# ---------------------------------------------------------------------------
# Task 4 — GRPO wiring
# ---------------------------------------------------------------------------
class TestGrpoPrmWiring:
    def test_prm_reward_selected(self, monkeypatch):
        import soup_cli.trainer.grpo as grpo

        sentinel = object()
        monkeypatch.setattr(
            "soup_cli.utils.prm_reward.build_prm_reward_fn",
            lambda tcfg, device, trust_remote_code: sentinel,
        )

        class _T:
            prm_reward = "./prm"
            prm_aggregate = "min"
            reward_fn = "accuracy"
            verifiable_domain = None

        out = grpo._select_reward_fn(_T(), "cpu", False)
        assert out is sentinel

    def test_standard_reward_selected(self, monkeypatch):
        import soup_cli.trainer.grpo as grpo

        captured = {}

        def _fake_load(spec, verifiable_domain=None):
            captured["spec"] = spec
            return "REWARD_FN"

        monkeypatch.setattr("soup_cli.trainer.rewards.load_reward_fn", _fake_load)

        class _T:
            prm_reward = None
            prm_aggregate = "min"
            reward_fn = "accuracy"
            verifiable_domain = None

        out = grpo._select_reward_fn(_T(), "cpu", False)
        assert out == "REWARD_FN"
        assert captured["spec"] == "accuracy"

    def test_verifiable_domain_threaded(self, monkeypatch):
        import soup_cli.trainer.grpo as grpo

        captured = {}

        def _fake_load(spec, verifiable_domain=None):
            captured["spec"] = spec
            captured["domain"] = verifiable_domain
            return "REWARD_FN"

        monkeypatch.setattr("soup_cli.trainer.rewards.load_reward_fn", _fake_load)

        class _T:
            prm_reward = None
            prm_aggregate = "min"
            reward_fn = "verifiable"
            verifiable_domain = "math"

        grpo._select_reward_fn(_T(), "cpu", False)
        assert captured["spec"] == "verifiable"
        assert captured["domain"] == "math"


# ---------------------------------------------------------------------------
# Task 6 — recipes
# ---------------------------------------------------------------------------
class TestRecipes:
    @pytest.mark.parametrize("name", _NEW_RECIPES)
    def test_recipe_resolves(self, name):
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe(name)
        assert recipe is not None
        assert recipe.task == "grpo"

    @pytest.mark.parametrize("name", _NEW_RECIPES)
    def test_recipe_yaml_parses(self, name):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe(name)
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.task == "grpo"
        assert cfg.training.rollout_backend == "openenv"
        assert cfg.training.rollout_func.startswith("soup_cli.envs.")

    def test_catalog_size_is_142(self):
        from soup_cli.recipes.catalog import RECIPES

        assert len(RECIPES) == 142


# ---------------------------------------------------------------------------
# PRM producer fixes surfaced by the live smoke (train-result shape + tokenizer)
# ---------------------------------------------------------------------------
class TestBuildPrmTrainResult:
    def test_has_all_summary_keys(self):
        from soup_cli.trainer.prm import build_prm_train_result

        out = build_prm_train_result(
            log_history=[{"loss": 3.2}, {"loss": 1.1}],
            metrics={"train_loss": 2.0},
            global_step=6,
            duration_secs=125.0,
            output_dir="./out",
        )
        # These are exactly the keys commands/train.py's summary indexes.
        for key in (
            "initial_loss",
            "final_loss",
            "duration",
            "duration_secs",
            "total_steps",
            "output_dir",
        ):
            assert key in out, key
        assert out["initial_loss"] == pytest.approx(3.2)
        assert out["final_loss"] == pytest.approx(1.1)
        assert out["total_steps"] == 6
        assert out["duration"] == "2m"

    def test_empty_log_history_falls_back_to_metrics(self):
        from soup_cli.trainer.prm import build_prm_train_result

        out = build_prm_train_result(
            log_history=[],
            metrics={"train_loss": 2.5},
            global_step=0,
            duration_secs=3700.0,
            output_dir="./out",
        )
        assert out["initial_loss"] == pytest.approx(2.5)
        assert out["final_loss"] == pytest.approx(2.5)
        assert out["duration"] == "1h 1m"

    def test_train_saves_tokenizer(self):
        # Guard: PRMTrainerWrapper.train() must persist the tokenizer so the
        # PRM checkpoint is loadable standalone by PRMScorer.
        import inspect

        from soup_cli.trainer.prm import PRMTrainerWrapper

        src = inspect.getsource(PRMTrainerWrapper.train)
        assert "self.tokenizer.save_pretrained" in src


class TestNoTopLevelTorch:
    def test_prm_reward_has_no_top_level_torch(self):
        import soup_cli.utils.prm_reward as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        heavy = {"torch", "transformers", "peft", "safetensors"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in heavy, alias.name
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in heavy, node.module
