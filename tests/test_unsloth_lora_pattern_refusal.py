"""#2153 - ``training.lora.rank_pattern`` / ``alpha_pattern`` are refused on ``backend: unsloth``.

Both fields are real on the transformers backend (peft's per-module rank and alpha
overrides, built by ``peft_wiring.build_lora_config_kwargs``), but every unsloth setup
(``_setup_unsloth`` on SFT, DPO, GRPO, PPO, KTO, ORPO, SimPO, IPO, BCO, pretrain,
embedding and, through SFT, TTS) goes through ``utils/unsloth.py:load_model_and_tokenizer``,
which hands ``FastLanguageModel.get_peft_model`` one scalar ``r`` and ``lora_alpha`` and has
no parameter for a pattern. A config that set one loaded, passed ``soup train --dry-run``
and trained a uniform-rank adapter with no warning.

It is the unsloth twin of ``tests/test_mlx_lora_pattern_refusal.py`` (#1996, #2150) and is
kept in its own file on purpose: when one backend starts applying the patterns, delete that
backend's validator and that backend's file, and the other keeps working. For unsloth:
``SoupConfig._validate_unsloth_lora_patterns`` and this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.backend_support import REGISTRY
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import UNSLOTH_SETUP_TASKS, SoupConfig
from tests.conftest import strip_ansi

#: Every task that reaches an unsloth setup; read from the schema, so a task added to
#: the constant is covered here without an edit.
_UNSLOTH_TASKS = sorted(UNSLOTH_SETUP_TASKS)
_WAY_OUT = "or use backend: transformers."

#: What these tasks need beyond the minimal config to pass their own validators.
_TASK_NEEDS: dict[str, dict] = {
    "preference": {"training": {"preference_loss": "dpo"}},
    "tts": {"modality": "audio_out", "training": {"tts_family": "orpheus"}},
}

_YAML = """
base: unsloth/Llama-3.2-1B-Instruct
task: sft
backend: unsloth
data:
  train: ./data/train.jsonl
  format: alpaca
training:
  epochs: 1
  quantization: 4bit
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


def _config(task: str = "sft", backend: str = "unsloth", **lora: object) -> SoupConfig:
    needs = _TASK_NEEDS.get(task, {})
    return SoupConfig(
        base="test/model",
        task=task,
        backend=backend,
        modality=needs.get("modality", "text"),
        data={"train": "d.jsonl"},
        training={"epochs": 1, "lora": lora, **needs.get("training", {})},
    )


def _refusal(task: str = "sft", backend: str = "unsloth", **lora: object) -> str:
    """The message of the refusal, without pydantic's echo of the input dict."""
    with pytest.raises(ValidationError) as excinfo:
        _config(task, backend, **lora)
    errors = excinfo.value.errors()
    assert len(errors) == 1, errors
    return " ".join(str(errors[0]["msg"]).split())


class TestTheBugOnMain:
    """These configs loaded before #2153, which is the whole defect."""

    @pytest.mark.parametrize("task", _UNSLOTH_TASKS)
    def test_every_unsloth_task_builds_without_a_pattern(self, task: str) -> None:
        # The helper builds a loadable config for each task, so the refusals below are
        # about the pattern and not about a config that never validated.
        assert _config(task).backend == "unsloth"

    def test_the_unsloth_loader_has_no_pattern_parameter(self) -> None:
        """The reason the fields cannot be honoured here: the one loader every
        ``_setup_unsloth`` calls takes a scalar r and alpha and nothing else. If this
        starts failing, the loader grew a pattern parameter: wire it and delete the
        refusal rather than editing this test."""
        import inspect

        from soup_cli.utils.unsloth import load_model_and_tokenizer

        params = set(inspect.signature(load_model_and_tokenizer).parameters)
        assert {"lora_r", "lora_alpha"} <= params
        assert not {name for name in params if "pattern" in name}


class TestRefusedOnUnsloth:
    @pytest.mark.parametrize("task", _UNSLOTH_TASKS)
    def test_rank_pattern_is_refused(self, task: str) -> None:
        msg = _refusal(task, rank_pattern={"q_proj": 4})
        assert "training.lora.rank_pattern is not applied by backend: unsloth" in msg
        assert "one scalar training.lora.r and training.lora.alpha" in msg
        assert f"Remove training.lora.rank_pattern, {_WAY_OUT}" in msg
        assert "alpha_pattern" not in msg  # names only what the config set

    @pytest.mark.parametrize("task", _UNSLOTH_TASKS)
    def test_alpha_pattern_is_refused(self, task: str) -> None:
        msg = _refusal(task, alpha_pattern={"q_proj": 32})
        assert "training.lora.alpha_pattern is not applied by backend: unsloth" in msg
        assert f"Remove training.lora.alpha_pattern, {_WAY_OUT}" in msg
        assert "rank_pattern" not in msg

    @pytest.mark.parametrize("task", _UNSLOTH_TASKS)
    def test_both_patterns_are_named_in_one_refusal(self, task: str) -> None:
        msg = _refusal(task, rank_pattern={"q_proj": 4}, alpha_pattern={"q_proj": 32})
        assert (
            "training.lora.rank_pattern and training.lora.alpha_pattern "
            "are not applied by backend: unsloth"
        ) in msg
        assert (
            "Remove training.lora.rank_pattern and training.lora.alpha_pattern, " + _WAY_OUT
        ) in msg

    def test_the_whole_message_is_pinned(self) -> None:
        assert _refusal("sft", rank_pattern={"q_proj": 4}) == (
            "Value error, training.lora.rank_pattern is not applied by backend: unsloth "
            "— the unsloth loader attaches every adapter from one scalar "
            "training.lora.r and training.lora.alpha, so per-module ranks and alphas "
            "would be dropped. Remove training.lora.rank_pattern, or use backend: "
            "transformers."
        )

    @pytest.mark.parametrize("value", [{"q_proj": 1}, {"q_proj": 4, "v_proj": 8}, {".*proj": 2}])
    def test_every_non_empty_pattern_is_refused(self, value: dict) -> None:
        # The refusal is about the backend, not about what the pattern says.
        assert "is not applied by backend: unsloth" in _refusal(rank_pattern=value)
        assert "is not applied by backend: unsloth" in _refusal(alpha_pattern=value)

    def test_a_pattern_equal_to_the_scalar_is_still_refused(self) -> None:
        # The refusal is about the field being set, not about its value: a pattern that
        # repeats lora.r / lora.alpha changes nothing, but the config still asked for it.
        assert "rank_pattern is not applied" in _refusal(
            r=8, alpha=16, rank_pattern={"q_proj": 8}
        )
        assert "alpha_pattern is not applied" in _refusal(
            r=8, alpha=16, alpha_pattern={"q_proj": 16}
        )

    def test_a_yaml_config_is_refused_at_load(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(_YAML)
        msg = _plain(str(excinfo.value))
        assert "training.lora.rank_pattern is not applied by backend: unsloth" in msg
        assert f"Remove training.lora.rank_pattern, {_WAY_OUT}" in msg

    def test_soup_train_dry_run_stops_with_the_message(self, tmp_path) -> None:
        config = tmp_path / "soup.yaml"
        config.write_text(_YAML, encoding="utf-8")
        result = CliRunner().invoke(app, ["train", "--config", str(config), "--dry-run"])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "training.lora.rank_pattern is not applied by backend: unsloth" in _plain(
            result.output
        )


class TestStillLoads:
    @pytest.mark.parametrize("task", _UNSLOTH_TASKS)
    @pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
    def test_an_empty_pattern_is_not_an_override(self, task: str, field: str) -> None:
        # peft_wiring reads ``if rank_pattern:``, so {} sets nothing on any backend.
        cfg = _config(task, **{field: {}})
        assert getattr(cfg.training.lora, field) == {}

    @pytest.mark.parametrize("task", _UNSLOTH_TASKS)
    def test_unset_and_null_patterns_load(self, task: str) -> None:
        cfg = _config(task, rank_pattern=None, alpha_pattern=None)
        assert cfg.backend == "unsloth"
        assert cfg.training.lora.rank_pattern is None
        assert cfg.training.lora.alpha_pattern is None

    @pytest.mark.parametrize("task", _UNSLOTH_TASKS)
    def test_scalar_lora_settings_still_load_on_unsloth(self, task: str) -> None:
        cfg = _config(task, r=8, alpha=16, use_rslora=True)
        assert (cfg.training.lora.r, cfg.training.lora.alpha) == (8, 16)

    @pytest.mark.parametrize("task", _UNSLOTH_TASKS)
    def test_transformers_keeps_the_patterns(self, task: str) -> None:
        # The way out the message offers must lead to a config that loads, on every
        # task the refusal fires for.
        cfg = _config(
            task, "transformers", rank_pattern={"q_proj": 4}, alpha_pattern={"q_proj": 32}
        )
        assert cfg.training.lora.rank_pattern == {"q_proj": 4}
        assert cfg.training.lora.alpha_pattern == {"q_proj": 32}

    def test_the_yaml_config_loads_once_the_backend_is_switched(self) -> None:
        cfg = load_config_from_string(_YAML.replace("backend: unsloth", "backend: transformers"))
        assert cfg.training.lora.rank_pattern == {"q_proj": 4}

    def test_the_unsloth_tip_is_not_offered_to_a_config_with_patterns(self) -> None:
        """``soup train`` suggests ``backend: unsloth`` only to a config that would still
        load with it (#1736); a transformers config that uses a pattern would not."""
        from soup_cli.commands.train import _loads_on_unsloth

        def loads(**lora: object) -> bool:
            return _loads_on_unsloth(_config("sft", "transformers", **lora))

        assert loads(r=8) is True
        assert loads(rank_pattern={"q_proj": 4}) is False
        assert loads(alpha_pattern={"q_proj": 4}) is False


class TestOtherRefusalsKeepTheirOwnWords:
    """Each of these configs names a problem that "use backend: transformers" fixes
    for a more basic reason, or that has its own words, so that refusal comes first."""

    @pytest.mark.parametrize("task", ["reward_model", "distill", "asr", "online_dpo"])
    def test_a_task_without_an_unsloth_setup_is_refused_for_that_first(self, task: str) -> None:
        def message(**lora: object) -> str:
            with pytest.raises(ValidationError) as excinfo:
                SoupConfig(
                    base="test/model",
                    task=task,
                    backend="unsloth",
                    data={"train": "d.jsonl"},
                    training={"epochs": 1, "lora": lora},
                )
            return " ".join(str(excinfo.value.errors()[0]["msg"]).split())

        with_pattern = message(rank_pattern={"q_proj": 4})
        assert with_pattern == message()  # the pattern does not change which refusal speaks
        assert "is not applied by backend: unsloth" not in with_pattern

    def test_reward_model_names_the_missing_unsloth_setup(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            SoupConfig(
                base="test/model",
                task="reward_model",
                backend="unsloth",
                data={"train": "d.jsonl"},
                training={"epochs": 1, "lora": {"rank_pattern": {"q_proj": 4}}},
            )
        msg = " ".join(str(excinfo.value).split())
        assert "backend='unsloth' is not applied by task='reward_model'" in msg
        assert "is not applied by backend: unsloth" not in msg

    @pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
    def test_multimodal_sft_keeps_its_own_message(self, field: str) -> None:
        # SFT vision / audio load through transformers, which does apply the patterns;
        # the modality refusal is the real problem and says what to do about it.
        with pytest.raises(ValidationError) as excinfo:
            SoupConfig(
                base="test/model",
                task="sft",
                backend="unsloth",
                modality="vision",
                data={"train": "d.jsonl"},
                training={"epochs": 1, "lora": {field: {"q_proj": 4}}},
            )
        msg = " ".join(str(excinfo.value).split())
        assert "task='sft' with backend='unsloth' requires modality='text'" in msg
        assert "is not applied by backend: unsloth" not in msg

    def test_full_fine_tuning_keeps_its_own_message(self) -> None:
        # lora.r=0 builds no adapter, so there is nothing for a pattern to override
        # on any backend; unsloth is refused for r=0 itself before this check runs.
        with pytest.raises(ValidationError) as excinfo:
            _config("sft", r=0, rank_pattern={"q_proj": 4})
        msg = " ".join(str(excinfo.value).split())
        assert "training.lora.r=0 (full fine-tuning) requires backend='transformers'" in msg
        assert "is not applied by backend: unsloth" not in msg


class TestOtherEntryPoints:
    def test_direct_construction_is_refused(self) -> None:
        """``soup sweep``, ``soup rewind`` and autopilot build ``SoupConfig`` directly."""
        with pytest.raises(ValidationError) as excinfo:
            SoupConfig(
                base="unsloth/Llama-3.2-1B-Instruct",
                backend="unsloth",
                data={"train": "t.jsonl"},
                training={"lora": {"alpha_pattern": {"q_proj": 32}}},
            )
        assert "training.lora.alpha_pattern is not applied by backend: unsloth" in _plain(
            str(excinfo.value)
        )

    def test_a_sweep_arm_that_switches_to_unsloth_is_refused(self) -> None:
        """``soup sweep --param backend=unsloth`` rebuilds the config from a dump of the
        base config, so patterns on a transformers base must not ride into an unsloth arm."""
        from soup_cli.commands.sweep import _set_nested_param

        base = _config("sft", "transformers", rank_pattern={"q_proj": 4})
        arm = base.model_dump()
        _set_nested_param(arm, "backend", "unsloth")
        with pytest.raises(ValidationError) as excinfo:
            SoupConfig(**arm)
        assert "training.lora.rank_pattern is not applied by backend: unsloth" in _plain(
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
        assert "training.lora.rank_pattern is not applied by backend: unsloth" in _plain(
            str(excinfo.value.__cause__)
        )
        assert "rank_pattern" not in str(excinfo.value)  # the client text stays generic

    def test_the_unsloth_docs_state_the_refusal(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "docs" / "backends-and-ops.md").read_text(encoding="utf-8")
        flat = " ".join(text.split())
        assert (
            "`training.lora.rank_pattern` and `training.lora.alpha_pattern` are refused "
            "on `backend: unsloth`"
        ) in flat

    def test_the_doctor_table_has_no_unsloth_pattern_rows(self) -> None:
        # A field refused at load must not also be a doctor row (#958, #1330).
        fields = {
            entry.field
            for (_task, backend, _modality), entries in REGISTRY.items()
            if backend == "unsloth"
            for entry in entries
        }
        assert not {"training.lora.rank_pattern", "training.lora.alpha_pattern"} & fields
