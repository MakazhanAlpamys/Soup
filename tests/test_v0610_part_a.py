"""Tests for v0.61.0 Part A — `soup unlearn` trainer (NPO / SimNPO / RMU).

Schema-only release: tests cover the new ``task='unlearn'`` Literal entry,
the ``unlearn_method`` field, the ``forget_set`` / ``retain_set`` data
schema, and the ``UnlearnTrainerWrapper`` stub. Live wiring is deferred
to v0.61.1 (mirrors v0.50.0 / v0.52.0 / v0.53.0 stub-then-live pattern).
"""

from __future__ import annotations

import dataclasses

import pytest

# ---------- Module surface ----------


class TestModuleSurface:
    def test_imports(self):
        from soup_cli.utils.unlearning import (
            SUPPORTED_UNLEARN_METHODS,
            UnlearnMethodSpec,
            apply_unlearn_loss,
            build_unlearn_trainer,
            get_unlearn_method_spec,
            validate_unlearn_method,
        )
        assert callable(validate_unlearn_method)
        assert callable(get_unlearn_method_spec)
        assert callable(apply_unlearn_loss)
        assert callable(build_unlearn_trainer)
        assert dataclasses.is_dataclass(UnlearnMethodSpec)
        assert isinstance(SUPPORTED_UNLEARN_METHODS, frozenset)

    def test_supported_methods_exact(self):
        from soup_cli.utils.unlearning import SUPPORTED_UNLEARN_METHODS

        assert SUPPORTED_UNLEARN_METHODS == frozenset({"npo", "simnpo", "rmu"})

    def test_metadata_mapping_proxy(self):
        from types import MappingProxyType

        from soup_cli.utils.unlearning import _UNLEARN_METHOD_METADATA  # type: ignore

        assert isinstance(_UNLEARN_METHOD_METADATA, MappingProxyType)


# ---------- validate_unlearn_method ----------


class TestValidateUnlearnMethod:
    def test_happy_path(self):
        from soup_cli.utils.unlearning import validate_unlearn_method

        for name in ("npo", "simnpo", "rmu"):
            assert validate_unlearn_method(name) == name

    def test_case_insensitive(self):
        from soup_cli.utils.unlearning import validate_unlearn_method

        assert validate_unlearn_method("NPO") == "npo"
        assert validate_unlearn_method("SimNPO") == "simnpo"
        assert validate_unlearn_method("RMU") == "rmu"

    def test_unknown_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_method

        with pytest.raises(ValueError, match="unknown unlearn method"):
            validate_unlearn_method("dpo")

    def test_bool_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_method

        with pytest.raises(TypeError):
            validate_unlearn_method(True)

    def test_non_string_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_method

        with pytest.raises(TypeError):
            validate_unlearn_method(123)

    def test_empty_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_method

        with pytest.raises(ValueError):
            validate_unlearn_method("")

    def test_null_byte_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_method

        with pytest.raises(ValueError, match="null"):
            validate_unlearn_method("npo\x00")

    def test_oversize_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_method

        with pytest.raises(ValueError):
            validate_unlearn_method("a" * 100)


# ---------- get_unlearn_method_spec ----------


class TestGetUnlearnMethodSpec:
    def test_happy_path(self):
        from soup_cli.utils.unlearning import get_unlearn_method_spec

        spec = get_unlearn_method_spec("npo")
        assert spec.name == "npo"
        assert isinstance(spec.description, str) and spec.description
        assert spec.live_wired is False  # deferred to v0.61.1

    def test_frozen(self):
        from soup_cli.utils.unlearning import get_unlearn_method_spec

        spec = get_unlearn_method_spec("npo")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "other"  # type: ignore

    def test_unknown_raises(self):
        from soup_cli.utils.unlearning import get_unlearn_method_spec

        with pytest.raises(ValueError):
            get_unlearn_method_spec("zzz")

    def test_all_three_have_specs(self):
        from soup_cli.utils.unlearning import get_unlearn_method_spec

        for name in ("npo", "simnpo", "rmu"):
            spec = get_unlearn_method_spec(name)
            assert spec.name == name


# ---------- apply_unlearn_loss (stub) ----------


class TestApplyUnlearnLoss:
    def test_unknown_rejected_before_notimplemented(self):
        from soup_cli.utils.unlearning import apply_unlearn_loss

        with pytest.raises(ValueError):
            apply_unlearn_loss("zzz")

    def test_deferred_marker_present(self):
        from soup_cli.utils.unlearning import apply_unlearn_loss

        with pytest.raises(NotImplementedError, match="v0.61.1"):
            apply_unlearn_loss("npo")


# ---------- build_unlearn_trainer (stub) ----------


class TestBuildUnlearnTrainer:
    def _make_cfg(self):
        from soup_cli.config.schema import SoupConfig

        return SoupConfig(
            base="test-model",
            task="unlearn",
            data={
                "train": "test.jsonl",
                "forget_set": "f.jsonl",
                "retain_set": "r.jsonl",
            },
            training={"unlearn_method": "npo"},
        )

    def test_returns_wrapper_instance(self):
        from soup_cli.trainer.unlearn import UnlearnTrainerWrapper
        from soup_cli.utils.unlearning import build_unlearn_trainer

        cfg = self._make_cfg()
        wrapper = build_unlearn_trainer(cfg)
        assert isinstance(wrapper, UnlearnTrainerWrapper)

    def test_kwargs_signature_allows_known(self):
        from soup_cli.utils.unlearning import build_unlearn_trainer

        cfg = self._make_cfg()
        # Forward-compat kwargs accepted at construction time. Setup
        # raises NotImplementedError per v0.61.1 stub contract.
        wrapper = build_unlearn_trainer(cfg, device="cpu", trust_remote_code=False)
        with pytest.raises(NotImplementedError, match="v0.61.1"):
            wrapper.setup()

    def test_invalid_config_rejected(self):
        from soup_cli.utils.unlearning import build_unlearn_trainer

        class _Cfg:
            pass

        with pytest.raises(AttributeError, match="SoupConfig"):
            build_unlearn_trainer(_Cfg())


# ---------- validate_unlearn_compat ----------


class TestValidateUnlearnCompat:
    def test_happy_path(self):
        from soup_cli.utils.unlearning import validate_unlearn_compat

        # Should not raise
        validate_unlearn_compat(task="unlearn", backend="transformers")

    def test_mlx_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_compat

        with pytest.raises(ValueError, match="mlx"):
            validate_unlearn_compat(task="unlearn", backend="mlx")

    def test_wrong_task_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_compat

        with pytest.raises(ValueError):
            validate_unlearn_compat(task="sft", backend="transformers")

    def test_bool_task_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_compat

        with pytest.raises(TypeError):
            validate_unlearn_compat(task=True, backend="transformers")  # type: ignore

    def test_bool_backend_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_compat

        with pytest.raises(TypeError):
            validate_unlearn_compat(task="unlearn", backend=True)  # type: ignore

    def test_empty_task_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_compat

        with pytest.raises(ValueError):
            validate_unlearn_compat(task="", backend="transformers")

    def test_null_byte_rejected(self):
        from soup_cli.utils.unlearning import validate_unlearn_compat

        with pytest.raises(ValueError, match="null"):
            validate_unlearn_compat(task="unlearn", backend="transformers\x00")


# ---------- Schema integration: task='unlearn' ----------


class TestSchemaUnlearnTask:
    def test_unlearn_task_accepted(self):
        from soup_cli.config.schema import SoupConfig

        cfg = SoupConfig(
            base="test-model",
            task="unlearn",
            data={
                "train": "test.jsonl",
                "format": "auto",
                "forget_set": "forget.jsonl",
                "retain_set": "retain.jsonl",
            },
            training={"unlearn_method": "npo"},
        )
        assert cfg.task == "unlearn"
        assert cfg.training.unlearn_method == "npo"

    def test_unlearn_method_default_none_for_non_unlearn(self):
        from soup_cli.config.schema import SoupConfig

        cfg = SoupConfig(
            base="test-model",
            task="sft",
            data={"train": "test.jsonl"},
        )
        assert cfg.training.unlearn_method is None

    def test_unlearn_method_unknown_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import SoupConfig

        with pytest.raises(ValidationError):
            SoupConfig(
                base="test-model",
                task="unlearn",
                data={"train": "test.jsonl", "forget_set": "f.jsonl"},
                training={"unlearn_method": "zzz"},
            )

    def test_unlearn_method_case_insensitive(self):
        from soup_cli.config.schema import SoupConfig

        cfg = SoupConfig(
            base="test-model",
            task="unlearn",
            data={
                "train": "test.jsonl",
                "forget_set": "f.jsonl",
                "retain_set": "r.jsonl",
            },
            training={"unlearn_method": "NPO"},
        )
        assert cfg.training.unlearn_method == "npo"

    def test_mlx_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import SoupConfig

        with pytest.raises(ValidationError, match="mlx"):
            SoupConfig(
                base="test-model",
                task="unlearn",
                backend="mlx",
                data={"train": "test.jsonl", "forget_set": "f.jsonl"},
                training={"unlearn_method": "npo"},
            )

    def test_unlearn_method_outside_unlearn_task_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import SoupConfig

        with pytest.raises(ValidationError, match="unlearn"):
            SoupConfig(
                base="test-model",
                task="sft",
                data={"train": "test.jsonl"},
                training={"unlearn_method": "npo"},
            )

    def test_forget_set_field(self):
        from soup_cli.config.schema import DataConfig

        data = DataConfig(train="test.jsonl", forget_set="forget.jsonl")
        assert data.forget_set == "forget.jsonl"

    def test_retain_set_field(self):
        from soup_cli.config.schema import DataConfig

        data = DataConfig(train="test.jsonl", retain_set="retain.jsonl")
        assert data.retain_set == "retain.jsonl"

    def test_forget_set_null_byte_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import DataConfig

        with pytest.raises(ValidationError):
            DataConfig(train="test.jsonl", forget_set="f\x00.jsonl")

    def test_retain_set_oversize_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import DataConfig

        with pytest.raises(ValidationError):
            DataConfig(train="test.jsonl", retain_set="x" * 5000)

    def test_unlearn_requires_forget_set(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import SoupConfig

        with pytest.raises(ValidationError, match="forget_set"):
            SoupConfig(
                base="test-model",
                task="unlearn",
                data={"train": "test.jsonl"},
                training={"unlearn_method": "npo"},
            )

    def test_unlearn_alpha_field(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(unlearn_method="npo", unlearn_alpha=0.5)
        assert tcfg.unlearn_alpha == 0.5

    def test_unlearn_alpha_bounds(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(unlearn_method="npo", unlearn_alpha=-0.1)

        with pytest.raises(ValidationError):
            TrainingConfig(unlearn_method="npo", unlearn_alpha=11.0)

    def test_unlearn_alpha_bool_rejected(self):
        from soup_cli.config.schema import TrainingConfig

        # Bool raises TypeError from the validator; Pydantic v2 does not
        # wrap TypeError in ValidationError (only ValueError /
        # AssertionError / PydanticCustomError).
        with pytest.raises(TypeError, match="bool"):
            TrainingConfig(unlearn_method="npo", unlearn_alpha=True)

    def test_unlearn_alpha_without_method_rejected(self):
        """Review L10 — `unlearn_alpha` without `unlearn_method` is a footgun."""
        from pydantic import ValidationError

        from soup_cli.config.schema import SoupConfig

        with pytest.raises(ValidationError, match="unlearn_method"):
            SoupConfig(
                base="test-model",
                task="sft",
                data={"train": "test.jsonl"},
                training={"unlearn_alpha": 0.5},
            )

    def test_unlearn_alpha_boundary_zero_accepted(self):
        """Review L3 — exact lower boundary (0.0)."""
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(unlearn_method="npo", unlearn_alpha=0.0)
        assert tcfg.unlearn_alpha == 0.0

    def test_unlearn_alpha_boundary_ten_accepted(self):
        """Review L3 — exact upper boundary (10.0)."""
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(unlearn_method="npo", unlearn_alpha=10.0)
        assert tcfg.unlearn_alpha == 10.0


# ---------- Source-grep regression guards (review L7) ----------


class TestSourceWiring:
    def test_cli_registers_edit_typer(self):
        """cli.py must add `_edit_cmd.app` under name='edit'."""
        from pathlib import Path

        cli_src = Path(__file__).resolve().parent.parent / "soup_cli" / "cli.py"
        text = cli_src.read_text(encoding="utf-8")
        assert "_edit_cmd" in text
        assert 'name="edit"' in text

    def test_eval_registers_v0610(self):
        """eval.py must call register(app, console) from _eval_v0610."""
        from pathlib import Path

        eval_src = (
            Path(__file__).resolve().parent.parent
            / "soup_cli" / "commands" / "eval.py"
        )
        text = eval_src.read_text(encoding="utf-8")
        assert "_register_v0610" in text

    def test_pyproject_includes_unlearning_fixtures(self):
        """pyproject.toml artifacts list must include unlearning fixtures."""
        from pathlib import Path

        py = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = py.read_text(encoding="utf-8")
        assert "unlearning/*.jsonl" in text


# ---------- UnlearnTrainerWrapper ----------


class TestUnlearnTrainerWrapper:
    def test_import(self):
        from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

        assert UnlearnTrainerWrapper is not None

    def test_train_before_setup_raises(self):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="unlearn",
            data={
                "train": "test.jsonl",
                "forget_set": "f.jsonl",
                "retain_set": "r.jsonl",
            },
            training={"unlearn_method": "npo"},
        )
        wrapper = UnlearnTrainerWrapper(cfg)
        with pytest.raises(RuntimeError, match="setup"):
            wrapper.train()

    def test_setup_deferred(self):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="unlearn",
            data={
                "train": "test.jsonl",
                "forget_set": "f.jsonl",
                "retain_set": "r.jsonl",
            },
            training={"unlearn_method": "npo"},
        )
        wrapper = UnlearnTrainerWrapper(cfg)
        with pytest.raises(NotImplementedError, match="v0.61.1"):
            wrapper.setup()

    def test_method_attribute(self):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="unlearn",
            data={
                "train": "test.jsonl",
                "forget_set": "f.jsonl",
                "retain_set": "r.jsonl",
            },
            training={"unlearn_method": "simnpo"},
        )
        wrapper = UnlearnTrainerWrapper(cfg)
        assert wrapper.method == "simnpo"
