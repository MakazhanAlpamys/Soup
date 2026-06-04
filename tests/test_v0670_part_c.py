"""v0.67.0 Part C — MoLE per-token adapter routing.

Tests for ``soup_cli/utils/mole_routing.py``:

- Frozen ``MoleGatingConfig`` dataclass
- ``validate_mole_compat`` (task / backend / adapter-count gate)
- ``build_gating_kernel`` stub raises NotImplementedError with v0.67.1 marker
- New ``task='moe_lora_routing'`` Literal on SoupConfig
- SoupConfig cross-validator gates mole config to the correct task/backend
"""

from __future__ import annotations

import dataclasses
import math

import pytest

# -----------------------------------------------------------------------------
# Public surface
# -----------------------------------------------------------------------------


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import mole_routing

        assert hasattr(mole_routing, "MoleGatingConfig")
        assert hasattr(mole_routing, "validate_mole_compat")
        assert hasattr(mole_routing, "build_gating_kernel")
        assert hasattr(mole_routing, "MIN_TASK_ADAPTERS")
        assert hasattr(mole_routing, "MAX_TASK_ADAPTERS")

    def test_constants_immutable(self) -> None:
        from soup_cli.utils import mole_routing

        assert mole_routing.MIN_TASK_ADAPTERS >= 2
        assert mole_routing.MAX_TASK_ADAPTERS <= 64


# -----------------------------------------------------------------------------
# MoleGatingConfig
# -----------------------------------------------------------------------------


class TestMoleGatingConfig:
    def test_construct(self) -> None:
        from soup_cli.utils.mole_routing import MoleGatingConfig

        cfg = MoleGatingConfig(
            num_task_adapters=4,
            hidden_dim=128,
            temperature=1.0,
            top_k=2,
        )
        assert cfg.num_task_adapters == 4
        assert cfg.top_k == 2

    def test_frozen(self) -> None:
        from soup_cli.utils.mole_routing import MoleGatingConfig

        cfg = MoleGatingConfig(
            num_task_adapters=4,
            hidden_dim=128,
            temperature=1.0,
            top_k=2,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.top_k = 99  # type: ignore[misc]

    def test_num_task_adapters_below_floor(self) -> None:
        from soup_cli.utils.mole_routing import MoleGatingConfig

        with pytest.raises(ValueError):
            MoleGatingConfig(
                num_task_adapters=1, hidden_dim=128, temperature=1.0, top_k=1
            )

    def test_num_task_adapters_above_cap(self) -> None:
        from soup_cli.utils.mole_routing import (
            MAX_TASK_ADAPTERS,
            MoleGatingConfig,
        )

        with pytest.raises(ValueError):
            MoleGatingConfig(
                num_task_adapters=MAX_TASK_ADAPTERS + 1,
                hidden_dim=128,
                temperature=1.0,
                top_k=1,
            )

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.mole_routing import MoleGatingConfig

        with pytest.raises(TypeError):
            MoleGatingConfig(
                num_task_adapters=True,  # type: ignore[arg-type]
                hidden_dim=128,
                temperature=1.0,
                top_k=1,
            )

    def test_hidden_dim_must_be_positive(self) -> None:
        from soup_cli.utils.mole_routing import MoleGatingConfig

        with pytest.raises(ValueError):
            MoleGatingConfig(
                num_task_adapters=4,
                hidden_dim=0,
                temperature=1.0,
                top_k=1,
            )

    def test_temperature_non_finite_rejected(self) -> None:
        from soup_cli.utils.mole_routing import MoleGatingConfig

        with pytest.raises(ValueError):
            MoleGatingConfig(
                num_task_adapters=4,
                hidden_dim=128,
                temperature=math.nan,
                top_k=1,
            )

    def test_temperature_non_positive_rejected(self) -> None:
        from soup_cli.utils.mole_routing import MoleGatingConfig

        with pytest.raises(ValueError):
            MoleGatingConfig(
                num_task_adapters=4,
                hidden_dim=128,
                temperature=0.0,
                top_k=1,
            )

    def test_top_k_above_num_adapters_rejected(self) -> None:
        from soup_cli.utils.mole_routing import MoleGatingConfig

        with pytest.raises(ValueError):
            MoleGatingConfig(
                num_task_adapters=3,
                hidden_dim=128,
                temperature=1.0,
                top_k=5,
            )

    def test_top_k_below_one_rejected(self) -> None:
        from soup_cli.utils.mole_routing import MoleGatingConfig

        with pytest.raises(ValueError):
            MoleGatingConfig(
                num_task_adapters=4,
                hidden_dim=128,
                temperature=1.0,
                top_k=0,
            )


# -----------------------------------------------------------------------------
# validate_mole_compat
# -----------------------------------------------------------------------------


class TestValidateMoleCompat:
    def test_happy_path(self) -> None:
        from soup_cli.utils.mole_routing import validate_mole_compat

        validate_mole_compat(
            task="moe_lora_routing",
            backend="transformers",
            num_task_adapters=4,
        )

    def test_wrong_task_rejected(self) -> None:
        from soup_cli.utils.mole_routing import validate_mole_compat

        with pytest.raises(ValueError) as exc_info:
            validate_mole_compat(
                task="sft",
                backend="transformers",
                num_task_adapters=4,
            )
        assert "moe_lora_routing" in str(exc_info.value)

    def test_mlx_rejected(self) -> None:
        from soup_cli.utils.mole_routing import validate_mole_compat

        with pytest.raises(ValueError) as exc_info:
            validate_mole_compat(
                task="moe_lora_routing",
                backend="mlx",
                num_task_adapters=4,
            )
        assert "mlx" in str(exc_info.value).lower()

    def test_too_few_adapters(self) -> None:
        from soup_cli.utils.mole_routing import validate_mole_compat

        with pytest.raises(ValueError):
            validate_mole_compat(
                task="moe_lora_routing",
                backend="transformers",
                num_task_adapters=1,
            )

    def test_bool_args_rejected(self) -> None:
        from soup_cli.utils.mole_routing import validate_mole_compat

        with pytest.raises(TypeError):
            validate_mole_compat(
                task=True,  # type: ignore[arg-type]
                backend="transformers",
                num_task_adapters=4,
            )
        with pytest.raises(TypeError):
            validate_mole_compat(
                task="moe_lora_routing",
                backend=True,  # type: ignore[arg-type]
                num_task_adapters=4,
            )

    def test_null_byte_args_rejected(self) -> None:
        from soup_cli.utils.mole_routing import validate_mole_compat

        with pytest.raises(ValueError):
            validate_mole_compat(
                task="moe_lora_routing\x00",
                backend="transformers",
                num_task_adapters=4,
            )


# -----------------------------------------------------------------------------
# Deferred-live stub
# -----------------------------------------------------------------------------


class TestBuildGatingKernel:
    def test_live_returns_module(self) -> None:
        # v0.71.12 #222 — the v0.67.1 deferred stub is lifted: build_gating_kernel
        # now returns a live torch nn.Module (per-token top-k softmax router).
        import torch

        from soup_cli.utils.mole_routing import MoleGatingConfig, build_gating_kernel

        cfg = MoleGatingConfig(
            num_task_adapters=4, hidden_dim=8, temperature=1.0, top_k=2
        )
        kernel = build_gating_kernel(cfg)
        weights = kernel(torch.randn(2, 3, 8))
        assert weights.shape == (2, 3, 4)
        assert torch.allclose(weights.sum(-1), torch.ones(2, 3), atol=1e-5)

    def test_non_config_rejected(self) -> None:
        from soup_cli.utils.mole_routing import build_gating_kernel

        with pytest.raises(TypeError):
            build_gating_kernel("not-a-config")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# SoupConfig integration — task='moe_lora_routing' Literal
# -----------------------------------------------------------------------------


class TestSchemaIntegration:
    def test_task_accepted_in_literal(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        # v0.71.12 #222 — task='moe_lora_routing' now requires
        # training.mole_task_adapters (>= 2 task-LoRA paths to route over).
        yaml = """
base: meta-llama/Llama-3.1-8B
task: moe_lora_routing
backend: transformers
modality: text
data:
  train: data.jsonl
  format: chatml
training:
  mole_task_adapters: ['./adapter_a', './adapter_b']
"""
        cfg = load_config_from_string(yaml)
        assert cfg.task == "moe_lora_routing"

    def test_task_rejected_on_mlx(self) -> None:
        from soup_cli.config.loader import load_config_from_string

        yaml = """
base: m
task: moe_lora_routing
backend: mlx
modality: text
data:
  train: data.jsonl
  format: chatml
"""
        with pytest.raises(Exception) as exc_info:
            load_config_from_string(yaml)
        assert "mlx" in str(exc_info.value).lower()


# -----------------------------------------------------------------------------
# Source-grep regression
# -----------------------------------------------------------------------------


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        src = (root / "src" / "soup_cli" / "utils" / "mole_routing.py").read_text(
            encoding="utf-8"
        )
        head_lines = [
            line
            for line in src.splitlines()[:50]
            if line.strip() and not line.strip().startswith("#")
        ]
        head = "\n".join(head_lines)
        for forbidden in ("import torch", "import transformers", "import peft"):
            assert forbidden not in head, f"top-level {forbidden!r}"
