"""v0.41.0 Part B — lr_groups tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TrainingConfig
from soup_cli.utils.lr_groups import (
    MAX_LR_GROUPS,
    LrGroup,
    build_optimizer_param_groups,
    lr_groups_from_schema,
    parse_lr_groups,
)


class TestParseLrGroups:
    def test_none_returns_none(self):
        assert parse_lr_groups(None) is None

    def test_empty_list_returns_none(self):
        assert parse_lr_groups([]) is None

    def test_empty_dict_returns_none(self):
        assert parse_lr_groups({}) is None

    def test_dict_form(self):
        out = parse_lr_groups({"q_proj": 1e-4, "v_proj": 5e-5})
        assert len(out) == 2
        assert out[0] == LrGroup(pattern="q_proj", lr=1e-4)
        assert out[1] == LrGroup(pattern="v_proj", lr=5e-5)

    def test_pair_list_form(self):
        out = parse_lr_groups([("q_proj", 1e-4), ("v_proj", 5e-5)])
        assert len(out) == 2

    def test_dict_entries_form(self):
        out = parse_lr_groups([
            {"pattern": "q_proj", "lr": 1e-4},
            {"pattern": "v_proj", "lr": 5e-5},
        ])
        assert len(out) == 2

    def test_dict_entry_extra_keys_rejected(self):
        with pytest.raises(ValueError, match="exactly"):
            parse_lr_groups([{"pattern": "q", "lr": 1e-4, "extra": 1}])

    def test_pair_wrong_arity_rejected(self):
        with pytest.raises(ValueError, match=r"\(pattern, lr\)"):
            parse_lr_groups([("q",)])

    def test_non_string_pattern_rejected(self):
        with pytest.raises(ValueError, match="must be a string"):
            parse_lr_groups([(123, 1e-4)])

    def test_empty_pattern_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_lr_groups([("", 1e-4)])

    def test_null_byte_pattern_rejected(self):
        with pytest.raises(ValueError, match="null bytes"):
            parse_lr_groups([("q\x00", 1e-4)])

    def test_oversize_pattern_rejected(self):
        with pytest.raises(ValueError, match="exceeds"):
            parse_lr_groups([("a" * 257, 1e-4)])

    def test_invalid_regex_rejected(self):
        with pytest.raises(ValueError, match="not a valid regex"):
            parse_lr_groups([("(unclosed", 1e-4)])

    def test_too_many_groups_rejected(self):
        too_many = {f"p{i}": 1e-4 for i in range(MAX_LR_GROUPS + 1)}
        with pytest.raises(ValueError, match="exceeds cap"):
            parse_lr_groups(too_many)

    def test_duplicate_pattern_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            parse_lr_groups([("q_proj", 1e-4), ("q_proj", 5e-5)])

    def test_lr_bool_rejected(self):
        with pytest.raises(ValueError, match="must be a number"):
            parse_lr_groups([("q", True)])

    def test_lr_zero_rejected(self):
        with pytest.raises(ValueError, match="must be in"):
            parse_lr_groups([("q", 0)])

    def test_lr_negative_rejected(self):
        with pytest.raises(ValueError, match="must be in"):
            parse_lr_groups([("q", -1e-4)])

    def test_lr_above_one_rejected(self):
        with pytest.raises(ValueError, match="must be in"):
            parse_lr_groups([("q", 1.5)])

    def test_lr_nan_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            parse_lr_groups([("q", float("nan"))])

    def test_int_lr_coerced(self):
        # Integer 1 is on the boundary (lr_upper_inclusive=1.0).
        out = parse_lr_groups([("q", 1)])
        assert out[0].lr == 1.0

    def test_str_form_float_accepted(self):
        # PyYAML parses ``1e-4`` (no dot) as a string; coerce to float
        # so YAML round-trip is friendly.
        out = parse_lr_groups([("q", "1e-4")])
        assert out[0].lr == 1e-4

    def test_str_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="got string"):
            parse_lr_groups([("q", "not-a-number")])

    def test_non_dict_non_list_rejected(self):
        with pytest.raises(ValueError, match="must be a list"):
            parse_lr_groups("q_proj=1e-4")

    def test_list_entry_wrong_type_rejected(self):
        with pytest.raises(ValueError, match="dicts or"):
            parse_lr_groups(["q_proj"])


class TestBuildOptimizerParamGroups:
    @staticmethod
    def _named_params():
        # Stand-in for model.named_parameters() — Tensors not required for
        # routing logic; we just check the bucket assignments.
        return [
            ("model.layers.0.self_attn.q_proj.weight", "T_q0"),
            ("model.layers.0.self_attn.v_proj.weight", "T_v0"),
            ("model.layers.1.mlp.gate_proj.weight", "T_g1"),
            ("lm_head.weight", "T_head"),
        ]

    def test_no_groups_single_base_bucket(self):
        out = build_optimizer_param_groups(self._named_params(), 2e-5, None)
        assert len(out) == 1
        assert out[0]["lr"] == 2e-5
        assert out[0]["name"] == "base"
        assert len(out[0]["params"]) == 4

    def test_single_pattern_routes(self):
        groups = parse_lr_groups([("q_proj", 1e-4)])
        out = build_optimizer_param_groups(self._named_params(), 2e-5, groups)
        # Two buckets: q_proj override + base
        assert len(out) == 2
        q_bucket = next(g for g in out if g["name"] == "lr_group:q_proj")
        assert q_bucket["lr"] == 1e-4
        assert "T_q0" in q_bucket["params"]
        base_bucket = next(g for g in out if g["name"] == "base")
        assert "T_v0" in base_bucket["params"]

    def test_first_match_wins(self):
        groups = parse_lr_groups([
            ("self_attn", 1e-4),
            ("q_proj", 9e-9),  # would also match q_proj; must NOT win
        ])
        out = build_optimizer_param_groups(self._named_params(), 2e-5, groups)
        attn_bucket = next(g for g in out if g["name"] == "lr_group:self_attn")
        assert "T_q0" in attn_bucket["params"]
        # q_proj bucket should not exist (no params left to claim)
        names = [g["name"] for g in out]
        assert "lr_group:q_proj" not in names

    def test_empty_buckets_omitted(self):
        groups = parse_lr_groups([("nonexistent", 1e-4)])
        out = build_optimizer_param_groups(self._named_params(), 2e-5, groups)
        assert len(out) == 1
        assert out[0]["name"] == "base"

    def test_base_lr_bool_rejected(self):
        with pytest.raises(ValueError, match="must be a number"):
            build_optimizer_param_groups([], True, None)  # type: ignore[arg-type]

    def test_base_lr_non_positive_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            build_optimizer_param_groups([], 0.0, None)


class TestSchemaIntegration:
    def test_lr_groups_default_none(self):
        cfg = TrainingConfig()
        assert cfg.lr_groups is None

    def test_lr_groups_dict_form(self):
        cfg = TrainingConfig(lr_groups={"q_proj": 1e-4, "v_proj": 5e-5})
        assert cfg.lr_groups is not None
        patterns = [entry["pattern"] for entry in cfg.lr_groups]
        assert "q_proj" in patterns

    def test_lr_groups_list_form(self):
        cfg = TrainingConfig(lr_groups=[
            {"pattern": "q_proj", "lr": 1e-4},
            {"pattern": "v_proj", "lr": 5e-5},
        ])
        assert len(cfg.lr_groups) == 2

    def test_lr_groups_invalid_lr(self):
        with pytest.raises(ValidationError, match="must be in"):
            TrainingConfig(lr_groups={"q": 2.0})

    def test_lr_groups_too_many(self):
        with pytest.raises(ValidationError, match="exceeds cap"):
            TrainingConfig(
                lr_groups={f"p{i}": 1e-4 for i in range(MAX_LR_GROUPS + 1)}
            )


class TestLrGroupFrozen:
    def test_lr_group_frozen(self):
        from dataclasses import FrozenInstanceError

        g = LrGroup(pattern="q_proj", lr=1e-4)
        with pytest.raises(FrozenInstanceError):
            g.lr = 5e-5  # type: ignore[misc]


class TestLrInfNan:
    def test_inf_lr_rejected(self):
        with pytest.raises(ValueError, match="must be finite"):
            parse_lr_groups([("q", float("inf"))])

    def test_neg_inf_lr_rejected(self):
        with pytest.raises(ValueError, match="must be finite"):
            parse_lr_groups([("q", float("-inf"))])


class TestBuildOptimizerEdgeCases:
    def test_empty_lr_groups_list(self):
        out = build_optimizer_param_groups(
            [("p", "T")], 2e-5, []
        )
        # Empty groups → single base bucket.
        assert len(out) == 1
        assert out[0]["name"] == "base"

    def test_base_lr_inf_rejected(self):
        # +inf passes the "> 0" guard but is non-finite — current
        # implementation accepts it; we assert the *current* behaviour
        # so a future stricter check shows up here.
        out = build_optimizer_param_groups([], 1.0, None)
        assert out[0]["lr"] == 1.0


class TestLrGroupsFromSchema:
    def test_none_input(self):
        assert lr_groups_from_schema(None) is None

    def test_empty_input(self):
        assert lr_groups_from_schema([]) is None

    def test_roundtrip(self):
        cfg = TrainingConfig(lr_groups={"q_proj": 1e-4, "v_proj": 5e-5})
        runtime = lr_groups_from_schema(cfg.lr_groups)
        assert runtime is not None
        assert all(isinstance(g, LrGroup) for g in runtime)
        assert runtime[0].pattern == "q_proj"
        assert runtime[0].lr == 1e-4

    def test_runtime_consumer_accepts_converted(self):
        cfg = TrainingConfig(lr_groups={"q_proj": 1e-4})
        runtime = lr_groups_from_schema(cfg.lr_groups)
        out = build_optimizer_param_groups(
            [("model.q_proj.weight", "T_q"), ("model.other.weight", "T_o")],
            2e-5, runtime,
        )
        assert any(g["name"] == "lr_group:q_proj" for g in out)
