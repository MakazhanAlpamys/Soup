"""Tests for v0.40.4 Part A — #63 trust_remote_code multi-trainer.

Extends the v0.36.0 ``--trust-remote-code`` opt-in to all 10 non-SFT
trainer wrappers (DPO / GRPO / KTO / ORPO / SimPO / IPO / PPO /
RewardModel / Pretrain / Embedding / BCO / Preference) and 5 standalone
commands (diff / export / merge / infer / generate).

Strategy: each trainer wrapper now accepts ``trust_remote_code: bool``
on ``__init__`` and resolves it via ``resolve_trust_remote_code`` from
``soup_cli.utils.trust_remote``. We assert source-level invariants
(closes the v0.36.0 known-gap family) plus a live-call test that
exercises the resolver path on each wrapper.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

# Rich help renderer can split a flag like ``--trust-remote-code`` with
# ANSI colour escapes between ``-``, ``-trust``, ``-remote-code`` when
# the terminal is narrow (CI runners hit this; Windows local does not).
# Strip ANSI so substring assertions are robust. Mirrors the helper in
# tests/test_trust_remote_code.py and tests/test_log_level.py.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)

# All 10 trainer modules + their wrapper class names.
# Direct trainers that resolve trust_remote_code in __init__.
TRAINER_TARGETS = [
    ("dpo", "DPOTrainerWrapper"),
    ("grpo", "GRPOTrainerWrapper"),
    ("kto", "KTOTrainerWrapper"),
    ("orpo", "ORPOTrainerWrapper"),
    ("simpo", "SimPOTrainerWrapper"),
    ("ipo", "IPOTrainerWrapper"),
    ("ppo", "PPOTrainerWrapper"),
    ("reward_model", "RewardModelTrainerWrapper"),
    ("pretrain", "PretrainTrainerWrapper"),
    ("embedding", "EmbeddingTrainerWrapper"),
    ("bco", "BCOTrainerWrapper"),
]
# PreferenceTrainerWrapper is a pure dispatcher — it does not resolve
# the flag itself; instead it forwards via kwargs to the inner wrapper.
DISPATCHER_TARGETS = [("preference", "PreferenceTrainerWrapper")]
ALL_TRAINER_TARGETS = TRAINER_TARGETS + DISPATCHER_TARGETS


def _minimal_cfg(base: str = "meta-llama/Llama-3.2-1B"):
    """Minimal SoupConfig stub with required fields for trainer ctor."""
    from soup_cli.config.schema import SoupConfig

    # The ``preference`` task has a cross-validator on preference_loss —
    # for non-preference trainers we use sft as the task.
    return SoupConfig(
        base=base,
        task="sft",
        data={"train": "fake.jsonl"},
        training={"epochs": 1, "lr": 1e-4},
    )


class TestTrainerCtorAcceptsTrustRemoteCode:
    """Every non-SFT trainer wrapper now accepts trust_remote_code in __init__."""

    @pytest.mark.parametrize("module,cls_name", TRAINER_TARGETS)
    def test_init_accepts_kwarg(self, module: str, cls_name: str):
        mod = __import__(f"soup_cli.trainer.{module}", fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        cfg = _minimal_cfg()
        # Must not raise on a meta-llama/* base (allowlisted, opt-in safe).
        instance = cls(cfg, device="cpu", trust_remote_code=False)
        assert hasattr(instance, "trust_remote_code")
        assert instance.trust_remote_code is False
        # Resolver result is cached — defaults to False on safe org.
        assert hasattr(instance, "_trust_remote_code")
        assert instance._trust_remote_code is False

    @pytest.mark.parametrize("module,cls_name", TRAINER_TARGETS)
    def test_init_with_opt_in(self, module: str, cls_name: str):
        mod = __import__(f"soup_cli.trainer.{module}", fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        cfg = _minimal_cfg()
        instance = cls(cfg, device="cpu", trust_remote_code=True)
        # Opt-in propagates to the resolved value.
        assert instance._trust_remote_code is True


class TestUnknownOrgRejectionWithoutOptIn:
    """When ``base`` points at a local dir whose config.json has auto_map,
    construction must fail without --trust-remote-code (closes the
    silent-execute-on-load loophole that v0.36.0 only fixed for SFT).
    """

    @pytest.mark.parametrize("module,cls_name", TRAINER_TARGETS)
    def test_local_auto_map_raises(self, tmp_path, module: str, cls_name: str):
        # Write a fake local checkpoint with auto_map (the marker that
        # triggers ``model_requires_trust_remote_code``).
        local = tmp_path / "fake_model"
        local.mkdir()
        (local / "config.json").write_text(
            '{"model_type": "fake", "auto_map": {"AutoConfig": "x.MyConfig"}}',
            encoding="utf-8",
        )
        mod = __import__(f"soup_cli.trainer.{module}", fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        cfg = _minimal_cfg(base=str(local))
        with pytest.raises(ValueError, match="trust_remote_code"):
            cls(cfg, device="cpu", trust_remote_code=False)


class TestSourceLevelInvariants:
    """The v0.36.0 known-gap family is now closed. No trainer file should
    contain ``trust_remote_code=True`` literally — every site reads from
    the resolved attribute or a parameter.
    """

    @pytest.mark.parametrize("module,_cls", ALL_TRAINER_TARGETS)
    def test_no_hardcoded_true_in_trainer(self, module: str, _cls: str):
        text = Path(f"soup_cli/trainer/{module}.py").read_text(encoding="utf-8")
        # Allow text only in comments / docstrings — assert exact-arg form
        # ``trust_remote_code=True`` is absent.
        assert "trust_remote_code=True" not in text, (
            f"{module}.py still hardcodes trust_remote_code=True — must "
            f"thread the v0.36.0 helper instead"
        )

    def test_no_hardcoded_true_in_sft(self):
        # SFT was already cleaned in v0.36.0 — regression guard.
        text = Path("soup_cli/trainer/sft.py").read_text(encoding="utf-8")
        assert "trust_remote_code=True" not in text


class TestCommandFlagsExist:
    """The 5 commands now expose ``--trust-remote-code`` Typer options.

    ``soup data generate`` is nested under the data subcommand group; the
    other 4 are top-level. Substring assertions go through ``_strip_ansi``
    because Rich splits long flag names with colour escapes on narrow
    terminals (CI runners hit this — same v0.40.3 ANSI-helper-text fix).
    """

    # (cli_path, human_label) — cli_path is the argv list for CliRunner.
    _COMMANDS = [
        (["diff", "--help"], "diff"),
        (["export", "--help"], "export"),
        (["merge", "--help"], "merge"),
        (["infer", "--help"], "infer"),
        (["data", "generate", "--help"], "data generate"),
    ]

    @pytest.mark.parametrize("argv,label", _COMMANDS)
    def test_cli_help_lists_flag(self, argv: list[str], label: str):
        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, argv)
        out = _strip_ansi(result.output)
        assert "--trust-remote-code" in out, (
            f"--trust-remote-code missing from `soup {label} --help` "
            f"(exit_code={result.exit_code}, output={out[:200]!r})"
        )


class TestTrainPyWiresFlagToAllTrainers:
    """The v0.36.0 ``sft_kwargs`` split is now removed — every trainer
    receives ``trust_remote_code`` from the same kwargs dict.
    """

    def test_no_sft_kwargs_split(self):
        text = Path("soup_cli/commands/train.py").read_text(encoding="utf-8")
        # The v0.36.0 vintage line ``sft_kwargs = dict(trainer_kwargs, ...``
        # no longer needs a separate dict — assert it's gone.
        assert "sft_kwargs = dict(trainer_kwargs," not in text
        # And every wrapper instantiation reads from the unified kwargs.
        assert "trust_remote_code=trust_remote_code" in text


class TestPpoLoadRewardModelHelperGated:
    """``_load_reward_model`` is module-level in ppo.py and used to be
    a hard-coded ``trust_remote_code=True`` site. v0.40.4 threads the
    user opt-in through it.
    """

    def test_helper_signature_accepts_flag(self):
        import inspect

        from soup_cli.trainer.ppo import _load_reward_model

        sig = inspect.signature(_load_reward_model)
        assert "trust_remote_code" in sig.parameters
        assert sig.parameters["trust_remote_code"].default is False

    def test_helper_rejects_unknown_org_local_auto_map(self, tmp_path):
        from soup_cli.trainer.ppo import _load_reward_model

        local = tmp_path / "fake_rm"
        local.mkdir()
        (local / "config.json").write_text(
            '{"model_type": "fake", "auto_map": {"AutoConfig": "x.MyConfig"}}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="trust_remote_code"):
            _load_reward_model(str(local), device="cpu", trust_remote_code=False)


class TestExportHelperSignatures:
    """``_merge_adapter`` and the four ``_export_*`` helpers in export.py
    now thread ``trust_remote_code`` from the Typer flag.
    """

    def test_merge_adapter_signature(self):
        import inspect

        from soup_cli.commands.export import _merge_adapter

        params = inspect.signature(_merge_adapter).parameters
        assert "trust_remote_code" in params
        assert params["trust_remote_code"].default is False

    @pytest.mark.parametrize(
        "fn_name", ["_export_onnx", "_export_tensorrt", "_export_awq", "_export_gptq"],
    )
    def test_export_helper_signature(self, fn_name: str):
        import inspect

        from soup_cli.commands import export as export_mod

        fn = getattr(export_mod, fn_name)
        params = inspect.signature(fn).parameters
        assert "trust_remote_code" in params, (
            f"{fn_name} did not get the v0.40.4 flag"
        )
        assert params["trust_remote_code"].default is False


class TestPreferenceForwardsToInner:
    """``PreferenceTrainerWrapper`` forwards ``trust_remote_code`` to the
    inner DPO / SimPO / ORPO / IPO / BCO wrapper.
    """

    def test_preference_inner_kwargs_include_flag(self):
        text = Path("soup_cli/trainer/preference.py").read_text(encoding="utf-8")
        # Both _build_inner and _build_multi_objective build a kwargs dict;
        # both must include the flag forwarding.
        assert text.count('"trust_remote_code": self.trust_remote_code') >= 2


class TestBcoAlreadyOnHelper:
    """Re-verify that BCO (v0.40.0 Part A) is now on the v0.36.0 helper too."""

    def test_bco_uses_resolver(self):
        text = Path("soup_cli/trainer/bco.py").read_text(encoding="utf-8")
        assert "resolve_trust_remote_code" in text
        assert "self._trust_remote_code" in text


class TestResolverLoadedLazily:
    """The resolver import inside ``__init__`` keeps ``import soup_cli.trainer.<x>``
    cheap — heavy deps are still gated behind ``setup()``.
    """

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_init_does_not_call_from_pretrained(
        self, mock_model, mock_tok,
    ):
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        cfg = _minimal_cfg()
        DPOTrainerWrapper(cfg, device="cpu", trust_remote_code=False)
        # Heavy loaders are still deferred to setup().
        assert mock_model.call_count == 0
        assert mock_tok.call_count == 0


class TestCommandHelpersGateRemoteCode:
    """v0.40.4 H1 — negative tests for each of the 5 commands' load-helper
    paths. Each helper resolves trust_remote_code BEFORE calling
    ``from_pretrained``, so a local checkpoint with ``auto_map`` must
    raise ``ValueError`` when ``trust_remote_code=False``.
    """

    def _make_local_auto_map(self, tmp_path):
        local = tmp_path / "fake_model"
        local.mkdir()
        (local / "config.json").write_text(
            '{"model_type": "fake", "auto_map": {"AutoConfig": "x.MyConfig"}}',
            encoding="utf-8",
        )
        return local

    def test_diff_load_model_rejects(self, tmp_path):
        from soup_cli.commands.diff import _load_model

        local = self._make_local_auto_map(tmp_path)
        with pytest.raises(ValueError, match="trust_remote_code"):
            _load_model(str(local), None, "cpu", trust_remote_code=False)

    def test_infer_load_model_rejects(self, tmp_path):
        from soup_cli.commands.infer import _load_model

        local = self._make_local_auto_map(tmp_path)
        with pytest.raises(ValueError, match="trust_remote_code"):
            _load_model(str(local), None, "cpu", trust_remote_code=False)

    def test_export_merge_adapter_rejects(self, tmp_path):
        from soup_cli.commands.export import _merge_adapter

        local = self._make_local_auto_map(tmp_path)
        with pytest.raises(ValueError, match="trust_remote_code"):
            _merge_adapter(
                str(local),
                str(local),
                str(tmp_path / "out"),
                trust_remote_code=False,
            )

    def test_generate_local_rejects(self, tmp_path):
        from soup_cli.commands.generate import _generate_local

        local = self._make_local_auto_map(tmp_path)
        with pytest.raises(ValueError, match="trust_remote_code"):
            _generate_local(
                prompt="x",
                count=1,
                fmt="alpaca",
                model_name=str(local),
                temperature=0.7,
                seed_examples=[],
                trust_remote_code=False,
            )


class TestPreferenceDispatcherLiveForwardsRejection:
    """H2 — beyond the source-grep test, verify that constructing a
    PreferenceTrainerWrapper that targets a local auto_map model
    eventually raises (the resolver fires inside the inner wrapper at
    setup() time).
    """

    def test_inner_wrapper_raises_on_setup(self, tmp_path):
        local = tmp_path / "fake_model"
        local.mkdir()
        (local / "config.json").write_text(
            '{"model_type": "fake", "auto_map": {"AutoConfig": "x.MyConfig"}}',
            encoding="utf-8",
        )

        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = SoupConfig(
            base=str(local),
            task="preference",
            data={"train": "fake.jsonl"},
            training={"epochs": 1, "lr": 1e-4, "preference_loss": "dpo"},
        )
        # The dispatcher itself constructs without resolving (forwarding
        # is deferred until ``_build_inner`` is called from ``setup``).
        wrapper = PreferenceTrainerWrapper(
            cfg, device="cpu", trust_remote_code=False,
        )
        # _build_inner constructs DPOTrainerWrapper which DOES resolve.
        with pytest.raises(ValueError, match="trust_remote_code"):
            wrapper._build_inner()
