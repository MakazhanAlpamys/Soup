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

    def test_rejects_null_byte(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError, match="null"):
            TrainingConfig(prm_reward="./prm\x00evil")

    def test_rejects_oversize(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError):
            TrainingConfig(prm_reward="x" * 5000)


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

        def __call__(self, input_ids, output_hidden_states=False):
            seq_len = input_ids.shape[1]
            hs = torch.arange(seq_len).float().reshape(1, seq_len, 1).repeat(1, 1, hidden)
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
