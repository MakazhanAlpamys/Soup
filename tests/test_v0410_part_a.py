"""v0.41.0 Part A — Optimizer Zoo tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TrainingConfig
from soup_cli.utils.optimizer_zoo import (
    SUPPORTED_OPTIMIZERS,
    is_new_v0_41_optimizer,
    required_package,
    validate_optimizer_name,
)


class TestSupportedOptimizers:
    def test_default_in_allowlist(self):
        assert "adamw_torch" in SUPPORTED_OPTIMIZERS

    def test_new_v0_41_entries(self):
        for name in (
            "badam",
            "apollo_adamw",
            "adam_mini",
            "lomo",
            "adalomo",
            "grokadamw",
            "schedule_free_adamw",
            "schedule_free_sgd",
            "muon",
            "dion",
            "came_pytorch",
            "ao_adamw_fp8",
            "ao_adamw_4bit",
            "ao_adamw_8bit",
        ):
            assert name in SUPPORTED_OPTIMIZERS, name
            assert is_new_v0_41_optimizer(name)

    def test_bnb_entries_present(self):
        assert "adamw_bnb_8bit" in SUPPORTED_OPTIMIZERS
        assert "paged_adamw_8bit" in SUPPORTED_OPTIMIZERS

    def test_frozenset_immutable(self):
        with pytest.raises(AttributeError):
            SUPPORTED_OPTIMIZERS.add("evil")  # type: ignore[attr-defined]


class TestValidateOptimizerName:
    def test_default_passes(self):
        assert validate_optimizer_name("adamw_torch") == "adamw_torch"

    def test_uppercase_normalised(self):
        assert validate_optimizer_name("BAdam") == "badam"

    def test_unknown_rejected(self):
        with pytest.raises(ValueError, match="not in the supported allowlist"):
            validate_optimizer_name("magicoptimizer")

    def test_non_string_rejected(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_optimizer_name(123)  # type: ignore[arg-type]

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_optimizer_name("")

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError, match="null bytes"):
            validate_optimizer_name("adamw\x00")

    def test_too_long_rejected(self):
        with pytest.raises(ValueError, match="exceeds"):
            validate_optimizer_name("a" * 65)


class TestRequiredPackage:
    def test_native_returns_none(self):
        assert required_package("adamw_torch") is None

    def test_bnb_returns_none(self):
        assert required_package("adamw_bnb_8bit") is None

    def test_new_returns_pkg(self):
        assert required_package("badam") == "badam"
        assert required_package("apollo_adamw") == "apollo-torch"
        assert required_package("schedule_free_adamw") == "schedulefree"
        assert required_package("ao_adamw_fp8") == "torchao"


class TestSchemaIntegration:
    def test_default_optimizer_passes(self):
        TrainingConfig()

    def test_new_optimizer_accepted(self):
        cfg = TrainingConfig(optimizer="badam")
        assert cfg.optimizer == "badam"

    def test_unknown_optimizer_rejected(self):
        with pytest.raises(ValidationError) as exc:
            TrainingConfig(optimizer="lookmaisnewname")
        assert "not in the supported allowlist" in str(exc.value)

    def test_uppercase_normalised(self):
        cfg = TrainingConfig(optimizer="BADAM")
        assert cfg.optimizer == "badam"

    def test_null_byte_rejected(self):
        with pytest.raises(ValidationError, match="null bytes"):
            TrainingConfig(optimizer="adamw\x00")


class TestIsNewV041Optimizer:
    def test_legacy_returns_false(self):
        assert is_new_v0_41_optimizer("adamw_torch") is False
        assert is_new_v0_41_optimizer("adafactor") is False

    def test_bnb_returns_false(self):
        assert is_new_v0_41_optimizer("adamw_bnb_8bit") is False

    def test_unknown_returns_false(self):
        assert is_new_v0_41_optimizer("not_an_optimizer") is False

    def test_non_string_returns_false(self):
        assert is_new_v0_41_optimizer(123) is False
        assert is_new_v0_41_optimizer(None) is False

    def test_case_insensitive(self):
        assert is_new_v0_41_optimizer("BADAM") is True


class TestRequiredPackageFull:
    @pytest.mark.parametrize("name,pkg", [
        ("adam_mini", "adam-mini"),
        ("lomo", "lomo-optim"),
        ("adalomo", "lomo-optim"),
        ("grokadamw", "grokadamw"),
        ("muon", "muon-optimizer"),
        ("dion", "dion-optimizer"),
        ("came_pytorch", "came-pytorch"),
        ("ao_adamw_4bit", "torchao"),
        ("ao_adamw_8bit", "torchao"),
        ("schedule_free_sgd", "schedulefree"),
    ])
    def test_each_pkg(self, name, pkg):
        assert required_package(name) == pkg

    def test_unknown_returns_none(self):
        assert required_package("not_an_optimizer") is None
