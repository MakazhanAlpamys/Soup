"""#1996 - ``training.lora.rank_pattern`` / ``alpha_pattern`` are refused on ``backend: mlx``.

Both fields are real on the transformers backend (peft's per-module rank and alpha
overrides), but every MLX trainer builds its adapter from ONE scalar
``training.lora.r`` and ``training.lora.alpha``: SFT's ``_apply_lora`` calls
``linear_to_lora_layers`` with ``int(lora_cfg.r)``, and the DPO, ORPO and SimPO loops
reuse that method. The only MLX refusal was for the DoRA variant, and the fields are not
in the doctor's ``_MLX_SFT`` table, so a plain-LoRA config that set them loaded,
passed ``soup train --dry-run`` and trained a uniform-rank adapter with no warning.

This is the stop-gap until the MLX trainer honours the patterns (PR #2133). When that
lands, delete this file together with ``SoupConfig._validate_mlx_lora_patterns``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.backend_support import REGISTRY
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from tests.conftest import strip_ansi

_MLX_TASKS = ["sft", "dpo", "orpo", "simpo", "kto", "grpo"]
_REMOVE_OR_SWITCH = "or use backend: transformers."

_YAML = """
base: mlx-community/Qwen2.5-0.5B-Instruct-4bit
task: sft
backend: mlx
data:
  train: ./data/train.jsonl
  format: alpaca
training:
  epochs: 1
  lora:
    r: 8
    alpha: 16
    rank_pattern:
      q_proj: 4
output: ./output
"""


def _plain(text: str) -> str:
    """Strip ANSI escapes and collapse whitespace (Rich colours and wraps on CI)."""
    return " ".join(strip_ansi(text).split())


def _config(task: str = "sft", backend: str = "mlx", **lora: object) -> SoupConfig:
    return SoupConfig(
        base="test/model",
        task=task,
        backend=backend,
        data={"train": "d.jsonl"},
        training={"epochs": 1, "lora": lora},
    )


def _refusal(task: str = "sft", backend: str = "mlx", **lora: object) -> str:
    """The message of the refusal, without pydantic's echo of the input dict."""
    with pytest.raises(ValidationError) as excinfo:
        _config(task, backend, **lora)
    errors = excinfo.value.errors()
    assert len(errors) == 1, errors
    return " ".join(str(errors[0]["msg"]).split())


class TestRefusedOnMlx:
    @pytest.mark.parametrize("task", _MLX_TASKS)
    def test_rank_pattern_is_refused(self, task: str) -> None:
        msg = _refusal(task, rank_pattern={"q_proj": 4})
        assert "training.lora.rank_pattern is not applied by backend: mlx" in msg
        assert "one scalar training.lora.r and training.lora.alpha" in msg
        assert f"Remove training.lora.rank_pattern, {_REMOVE_OR_SWITCH}" in msg
        assert "alpha_pattern" not in msg  # names only what the config set

    @pytest.mark.parametrize("task", _MLX_TASKS)
    def test_alpha_pattern_is_refused(self, task: str) -> None:
        msg = _refusal(task, alpha_pattern={"q_proj": 32})
        assert "training.lora.alpha_pattern is not applied by backend: mlx" in msg
        assert f"Remove training.lora.alpha_pattern, {_REMOVE_OR_SWITCH}" in msg
        assert "rank_pattern" not in msg

    @pytest.mark.parametrize("task", _MLX_TASKS)
    def test_both_patterns_are_named_in_one_refusal(self, task: str) -> None:
        msg = _refusal(task, rank_pattern={"q_proj": 4}, alpha_pattern={"q_proj": 32})
        assert (
            "training.lora.rank_pattern and training.lora.alpha_pattern "
            "are not applied by backend: mlx"
        ) in msg
        assert (
            "Remove training.lora.rank_pattern and training.lora.alpha_pattern, "
            + _REMOVE_OR_SWITCH
        ) in msg

    @pytest.mark.parametrize("value", [{"q_proj": 1}, {"q_proj": 4, "v_proj": 8}, {".*proj": 2}])
    def test_every_non_empty_pattern_is_refused(self, value: dict) -> None:
        # The refusal is about the backend, not about what the pattern says.
        assert "is not applied by backend: mlx" in _refusal(rank_pattern=value)
        assert "is not applied by backend: mlx" in _refusal(alpha_pattern=value)

    def test_a_pattern_equal_to_the_scalar_is_still_refused(self) -> None:
        # The refusal is about the field being set, not about its value: a pattern that
        # repeats lora.r / lora.alpha changes nothing, but the config still asked for it.
        assert "rank_pattern is not applied" in _refusal(r=8, alpha=16, rank_pattern={"q_proj": 8})
        assert "alpha_pattern is not applied" in _refusal(
            r=8, alpha=16, alpha_pattern={"q_proj": 16}
        )

    def test_a_yaml_config_is_refused_at_load(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(_YAML)
        msg = _plain(str(excinfo.value))
        assert "training.lora.rank_pattern is not applied by backend: mlx" in msg
        assert f"Remove training.lora.rank_pattern, {_REMOVE_OR_SWITCH}" in msg

    def test_soup_train_dry_run_stops_with_the_message(self, tmp_path) -> None:
        config = tmp_path / "soup.yaml"
        config.write_text(_YAML, encoding="utf-8")
        result = CliRunner().invoke(app, ["train", "--config", str(config), "--dry-run"])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "training.lora.rank_pattern is not applied by backend: mlx" in _plain(
            result.output
        )


class TestStillLoads:
    @pytest.mark.parametrize("task", _MLX_TASKS)
    @pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
    def test_an_empty_pattern_is_not_an_override(self, task: str, field: str) -> None:
        # peft_wiring reads ``if rank_pattern:``, so {} sets nothing on any backend.
        cfg = _config(task, **{field: {}})
        assert getattr(cfg.training.lora, field) == {}

    @pytest.mark.parametrize("task", _MLX_TASKS)
    def test_unset_and_null_patterns_load(self, task: str) -> None:
        cfg = _config(task, rank_pattern=None, alpha_pattern=None)
        assert cfg.backend == "mlx"
        assert cfg.training.lora.rank_pattern is None
        assert cfg.training.lora.alpha_pattern is None
        assert _config(task).training.lora.rank_pattern is None

    @pytest.mark.parametrize("task", _MLX_TASKS)
    def test_scalar_lora_settings_still_load_on_mlx(self, task: str) -> None:
        cfg = _config(task, r=8, alpha=16, use_rslora=True)
        assert (cfg.training.lora.r, cfg.training.lora.alpha) == (8, 16)

    def test_transformers_keeps_the_patterns(self) -> None:
        # Only transformers applies them (peft_wiring). unsloth refuses them too since
        # #2153 -- that is tests/test_unsloth_lora_pattern_refusal.py.
        cfg = _config(
            "sft", "transformers", rank_pattern={"q_proj": 4}, alpha_pattern={"q_proj": 32}
        )
        assert cfg.training.lora.rank_pattern == {"q_proj": 4}
        assert cfg.training.lora.alpha_pattern == {"q_proj": 32}

    @pytest.mark.parametrize("task", ["dpo", "orpo"])
    def test_transformers_preference_tasks_keep_the_patterns(self, task: str) -> None:
        # The way out the message offers (use backend: transformers) loads for the
        # preference tasks too. It is not unconditional: an MLX-only setting such as
        # training.optimizer: muon is refused on transformers for its own reason.
        cfg = _config(task, "transformers", rank_pattern={"q_proj": 4})
        assert cfg.training.lora.rank_pattern == {"q_proj": 4}

    def test_the_yaml_config_loads_once_the_backend_is_switched(self) -> None:
        cfg = load_config_from_string(
            _YAML.replace("backend: mlx", "backend: transformers").replace(
                "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "Qwen/Qwen2.5-0.5B-Instruct"
            )
        )
        assert cfg.training.lora.rank_pattern == {"q_proj": 4}


class TestOtherRefusalsKeepTheirOwnWords:
    """Each of these configs names a problem that "use backend: transformers" would not
    fix, or that has a more basic answer, so that refusal has to come first."""

    @pytest.mark.parametrize("task", ["ipo", "pretrain"])
    def test_a_task_mlx_cannot_run_is_refused_for_that_first(self, task: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _config(task, rank_pattern={"q_proj": 4})
        msg = " ".join(str(excinfo.value).split())
        assert "MLX backend only ships SFT" in msg
        assert "is not applied by backend: mlx" not in msg

    @pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
    def test_mlx_dora_keeps_its_own_message(self, field: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _config(use_dora=True, **{field: {"q_proj": 4}})
        msg = " ".join(str(excinfo.value).split())
        assert "MLX DoRA does not support" in msg
        assert "is not applied by backend: mlx" not in msg

    def test_full_fine_tuning_keeps_its_own_message(self) -> None:
        # lora.r=0 builds no adapter, so there is nothing for a pattern to override
        # on any backend; MLX is refused for r=0 itself before this check runs.
        with pytest.raises(ValidationError) as excinfo:
            _config("sft", r=0, rank_pattern={"q_proj": 4})
        msg = " ".join(str(excinfo.value).split())
        assert "training.lora.r=0 (full fine-tuning) requires backend='transformers'" in msg
        assert "is not applied by backend: mlx" not in msg


class TestOtherEntryPoints:
    def test_direct_construction_is_refused(self) -> None:
        """``soup sweep``, ``soup rewind`` and autopilot build ``SoupConfig`` directly."""
        with pytest.raises(ValidationError) as excinfo:
            SoupConfig(
                base="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                backend="mlx",
                data={"train": "t.jsonl"},
                training={"lora": {"alpha_pattern": {"q_proj": 32}}},
            )
        assert "training.lora.alpha_pattern is not applied by backend: mlx" in _plain(
            str(excinfo.value)
        )

    def test_a_sweep_arm_that_switches_to_mlx_is_refused(self) -> None:
        """``soup sweep --param backend=mlx`` rebuilds the config from a dump of the
        base config, so patterns on a transformers base must not ride into an mlx arm."""
        from soup_cli.commands.sweep import _set_nested_param

        base = _config("sft", "transformers", rank_pattern={"q_proj": 4})
        arm = base.model_dump()
        _set_nested_param(arm, "backend", "mlx")
        with pytest.raises(ValidationError) as excinfo:
            SoupConfig(**arm)
        assert "training.lora.rank_pattern is not applied by backend: mlx" in _plain(
            str(excinfo.value)
        )

    def test_the_mcp_config_loader_refuses_it(self, tmp_path, monkeypatch) -> None:
        """The MCP profile/plan tools read the YAML through ``load_config_from_string``.

        The loader returns the user only ``invalid config (<ExceptionType>)`` on purpose
        (its text must stay path-free), so the refusal reason cannot reach an MCP client;
        it is kept on ``__cause__``, which is what pins that the YAML was refused for the
        pattern and not for some other ValueError.
        """
        from soup_cli.mcp_server.registry import McpToolError, _load_config_under_cwd

        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(_YAML, encoding="utf-8")
        with pytest.raises(McpToolError, match="invalid config") as excinfo:
            _load_config_under_cwd("soup.yaml")
        assert "training.lora.rank_pattern is not applied by backend: mlx" in _plain(
            str(excinfo.value.__cause__)
        )
        assert "rank_pattern" not in str(excinfo.value)  # the client text stays generic

    def test_the_mlx_docs_state_the_refusal(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "docs" / "backends-and-ops.md").read_text(encoding="utf-8")
        flat = " ".join(text.split())
        assert (
            "`training.lora.rank_pattern` and `training.lora.alpha_pattern` are refused "
            "on `backend: mlx` for SFT, DPO, ORPO, SimPO, KTO and GRPO"
        ) in flat

    def test_the_doctor_table_has_no_pattern_rows(self) -> None:
        # A field refused at load must not also be a doctor row (#958, #1330).
        fields = {entry.field for entries in REGISTRY.values() for entry in entries}
        assert not {"training.lora.rank_pattern", "training.lora.alpha_pattern"} & fields
