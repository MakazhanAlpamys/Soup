"""Tests for RLVR (RL from Verifiable Rewards) — Part C of v0.25.0."""

import pytest

# ---------------------------------------------------------------------------
# Math verification reward
# ---------------------------------------------------------------------------


class TestMathVerifyReward:
    def test_exact_numeric_match(self):
        from soup_cli.trainer.rewards import math_verify_reward

        completions = [
            [{"role": "assistant", "content": "The answer is #### 42"}],
            [{"role": "assistant", "content": "\\boxed{3.14}"}],
        ]
        rewards = math_verify_reward(completions, answer=["42", "3.14"])
        assert rewards == [1.0, 1.0]

    def test_tolerance_exact_match_to_fourth_decimal(self):
        from soup_cli.trainer.rewards import math_verify_reward

        completions = [
            [{"role": "assistant", "content": "#### 3.14159"}],
        ]
        # Difference = 1e-5, well below default 1e-4 tolerance → full credit
        rewards = math_verify_reward(completions, answer=["3.14160"])
        assert rewards[0] == 1.0

    def test_tolerance_partial_credit(self):
        from soup_cli.trainer.rewards import math_verify_reward

        # Difference = 0.001, outside 1e-4 tolerance but inside 1e-2 → 0.6
        completions = [
            [{"role": "assistant", "content": "#### 3.141"}],
        ]
        rewards = math_verify_reward(completions, answer=["3.142"])
        assert rewards[0] == 0.6

    def test_tolerance_far_miss(self):
        from soup_cli.trainer.rewards import math_verify_reward

        completions = [
            [{"role": "assistant", "content": "#### 3.0"}],
        ]
        rewards = math_verify_reward(completions, answer=["3.14159"])
        assert rewards[0] == 0.0

    def test_wrong_answer(self):
        from soup_cli.trainer.rewards import math_verify_reward

        completions = [
            [{"role": "assistant", "content": "#### 100"}],
        ]
        rewards = math_verify_reward(completions, answer=["42"])
        assert rewards[0] == 0.0

    def test_no_answer_extraction(self):
        from soup_cli.trainer.rewards import math_verify_reward

        completions = [
            [{"role": "assistant", "content": "no answer here"}],
        ]
        rewards = math_verify_reward(completions, answer=["42"])
        assert rewards[0] == 0.0

    def test_empty_completions(self):
        from soup_cli.trainer.rewards import math_verify_reward

        assert math_verify_reward([], answer=[]) == []

    def test_no_eval_on_user_content(self):
        """Security: math_verify must never call eval() or exec() on model output."""
        from soup_cli.trainer.rewards import math_verify_reward

        completions = [
            [{"role": "assistant", "content": "#### __import__('os').system('rm')"}],
        ]
        # Should not crash, should not match a numeric answer
        rewards = math_verify_reward(completions, answer=["42"])
        assert rewards[0] == 0.0


# ---------------------------------------------------------------------------
# Code execution reward
# ---------------------------------------------------------------------------


class TestCodeExecReward:
    def test_correct_code(self):
        from soup_cli.trainer.rewards import code_exec_reward

        completions = [
            [{"role": "assistant", "content": "print(2 + 2)"}],
        ]
        rewards = code_exec_reward(completions, expected=["4"])
        assert rewards[0] == 1.0

    def test_wrong_output(self):
        from soup_cli.trainer.rewards import code_exec_reward

        completions = [
            [{"role": "assistant", "content": "print(1 + 1)"}],
        ]
        rewards = code_exec_reward(completions, expected=["4"])
        assert rewards[0] == 0.0

    def test_code_extracted_from_markdown(self):
        from soup_cli.trainer.rewards import code_exec_reward

        completions = [
            [{"role": "assistant", "content": "```python\nprint(5 * 5)\n```"}],
        ]
        rewards = code_exec_reward(completions, expected=["25"])
        assert rewards[0] == 1.0

    def test_infinite_loop_caught_by_timeout(self):
        from soup_cli.trainer.rewards import code_exec_reward

        completions = [
            [{"role": "assistant", "content": "while True: pass"}],
        ]
        rewards = code_exec_reward(completions, expected=["4"])
        assert rewards[0] == 0.0

    def test_network_blocked(self):
        """Security: code_exec must prevent network access."""
        from soup_cli.trainer.rewards import code_exec_reward

        completions = [
            [
                {
                    "role": "assistant",
                    "content": "import urllib.request\nprint(urllib.request.urlopen('http://example.com'))",
                }
            ],
        ]
        rewards = code_exec_reward(completions, expected=["hello"])
        assert rewards[0] == 0.0

    def test_output_size_capped(self):
        """Security: output capped at 10KB."""
        from soup_cli.trainer.rewards import MAX_CODE_OUTPUT_BYTES, code_exec_reward

        # Generate output bigger than cap
        completions = [
            [{"role": "assistant", "content": "print('x' * 50000)"}],
        ]
        rewards = code_exec_reward(completions, expected=["x" * 50000])
        # Reward is 0 since truncated output won't match
        assert rewards[0] == 0.0
        assert MAX_CODE_OUTPUT_BYTES == 10_000


# ---------------------------------------------------------------------------
# JSON schema reward
# ---------------------------------------------------------------------------


class TestJsonSchemaReward:
    def test_valid_matching_schema(self):
        from soup_cli.trainer.rewards import json_schema_reward

        completions = [
            [{"role": "assistant", "content": '{"name": "alice", "age": 30}'}],
        ]
        schemas = [
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            }
        ]
        rewards = json_schema_reward(completions, schema=schemas)
        assert rewards[0] == 1.0

    def test_missing_required_field(self):
        from soup_cli.trainer.rewards import json_schema_reward

        completions = [
            [{"role": "assistant", "content": '{"name": "alice"}'}],
        ]
        schemas = [
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            }
        ]
        rewards = json_schema_reward(completions, schema=schemas)
        assert rewards[0] < 1.0

    def test_invalid_json(self):
        from soup_cli.trainer.rewards import json_schema_reward

        completions = [
            [{"role": "assistant", "content": "not json at all"}],
        ]
        schemas = [{"type": "object", "properties": {}, "required": []}]
        rewards = json_schema_reward(completions, schema=schemas)
        assert rewards[0] == 0.0

    def test_wrong_type_for_field(self):
        """Field present but wrong type counts as missing."""
        from soup_cli.trainer.rewards import json_schema_reward

        completions = [
            [{"role": "assistant", "content": '{"name": "alice", "age": "thirty"}'}],
        ]
        schemas = [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            }
        ]
        rewards = json_schema_reward(completions, schema=schemas)
        # name matches (0.5), age wrong type (0.0) — score 0.5
        assert rewards[0] == 0.5

    def test_integer_field_rejects_bool(self):
        """bool is not a valid int for JSON schema integer fields."""
        from soup_cli.trainer.rewards import json_schema_reward

        completions = [
            [{"role": "assistant", "content": '{"enabled": true}'}],
        ]
        schemas = [
            {
                "type": "object",
                "properties": {"enabled": {"type": "integer"}},
                "required": ["enabled"],
            }
        ]
        rewards = json_schema_reward(completions, schema=schemas)
        assert rewards[0] == 0.0


# ---------------------------------------------------------------------------
# Verifiable routing via config
# ---------------------------------------------------------------------------


class TestVerifiableRouting:
    def test_verifiable_math_loads(self):
        from soup_cli.trainer.rewards import load_reward_fn

        fn = load_reward_fn("verifiable", verifiable_domain="math")
        assert callable(fn)

    def test_verifiable_code_loads(self):
        from soup_cli.trainer.rewards import load_reward_fn

        fn = load_reward_fn("verifiable", verifiable_domain="code")
        assert callable(fn)

    def test_verifiable_json_schema_loads(self):
        from soup_cli.trainer.rewards import load_reward_fn

        fn = load_reward_fn("verifiable", verifiable_domain="json_schema")
        assert callable(fn)

    def test_verifiable_unknown_domain_raises(self):
        from soup_cli.trainer.rewards import load_reward_fn

        with pytest.raises(ValueError):
            load_reward_fn("verifiable", verifiable_domain="unknown")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestVerifiableConfig:
    def test_verifiable_domain_accepted(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(reward_fn="verifiable", verifiable_domain="math")
        assert cfg.verifiable_domain == "math"

    def test_verifiable_domain_invalid_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(reward_fn="verifiable", verifiable_domain="hacker")

    def test_verifiable_requires_domain(self):
        """reward_fn=verifiable requires verifiable_domain to be set."""
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(reward_fn="verifiable", verifiable_domain=None)


# ---------------------------------------------------------------------------
# Synth data template
# ---------------------------------------------------------------------------


class TestVerifiableTemplate:
    def test_build_prompt_math(self):
        from soup_cli.data.templates.verifiable import build_prompt

        prompt = build_prompt(count=5, fmt="alpaca", format_spec="{...}", domain="math")
        assert "5" in prompt
        assert "math" in prompt.lower()

    def test_build_prompt_code(self):
        from soup_cli.data.templates.verifiable import build_prompt

        prompt = build_prompt(count=3, fmt="alpaca", format_spec="{}", domain="code")
        assert "3" in prompt

    def test_build_prompt_json_schema(self):
        from soup_cli.data.templates.verifiable import build_prompt

        prompt = build_prompt(count=2, fmt="alpaca", format_spec="{}", domain="json_schema")
        assert "2" in prompt
        assert "schema" in prompt.lower()

    def test_domains_constant(self):
        from soup_cli.data.templates.verifiable import TEMPLATE_SPEC

        for key in ("math", "code", "json_schema"):
            assert key in TEMPLATE_SPEC["domains"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
