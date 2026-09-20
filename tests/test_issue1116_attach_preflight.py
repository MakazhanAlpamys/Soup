"""#1116: can every shipped config actually attach a LoRA adapter?

A config can parse (#330), name a repo that resolves (#677), and still be
untrainable, because `target_modules: auto` resolves to nothing for an
architecture neither Soup nor peft maps. #1070 (every shipped MoE recipe),
#798's dropout refusal and #1074's `templates/moe.yaml` blocker were each found
by a person reading code, not by a check.

Two rules are load-bearing and are pinned hardest here, because breaking either
turns the guard into noise that gets muted:

1. **A failure to LOAD is not a failure to ATTACH.** Gated, remote-code and
   unreadable configs are UNVERIFIED, never CANNOT_ATTACH, and never change the
   exit code. On a box with no `HF_TOKEN` that is most of the gated bases in
   the catalogue, so the rule is what keeps CI from going permanently red.
2. **Remote code is never executed**, so a repo needing it is unverifiable
   rather than run.

The end-to-end tests build real `transformers` configs locally and inject them,
so the real resolution and the real `get_peft_model` run with no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.utils.attach_preflight import (
    PREFLIGHT_LAYERS,
    AttachCheck,
    PreflightReport,
    Verdict,
    check_attach,
    classify_load_failure,
    count_adapted,
    plan_adapter,
    shrink_for_preflight,
)

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
torch = pytest.importorskip("torch")


def _cfg(base="org/m", task="sft", r=8, dropout=0.0, backend=None, targets="auto"):
    """A stand-in for the loaded SoupConfig, carrying only what the check reads."""
    return SimpleNamespace(
        base=base,
        task=task,
        backend=backend,
        training=SimpleNamespace(
            lora=SimpleNamespace(r=r, alpha=16, dropout=dropout, target_modules=targets)
        ),
    )


class TestLoadFailuresAreNotAttachFailures:
    """Rule 1. Each of these is a config this machine could not answer for."""

    @pytest.mark.parametrize(
        "message, expected_fragment",
        [
            ("You are trying to access a gated repo.", "gated"),
            ("contains custom code which must be executed", "trust_remote_code"),
            (
                "requires you to execute the configuration file... trust_remote_code",
                "trust_remote_code",
            ),
            ("Unrecognized model in org/x. Should have a `model_type`", "cannot read"),
            ("org/x does not appear to have a file named config.json", "no config.json"),
            ("org/x is not a local folder and is not a valid model identifier", "#677"),
        ],
    )
    def test_each_known_loader_failure_is_named(self, message, expected_fragment):
        assert expected_fragment in classify_load_failure(ValueError(message))

    def test_an_unrecognised_failure_returns_empty_not_a_guess(self):
        """Empty is how the caller knows to print the exception instead. Filing an
        unknown failure under the last branch is how a guard starts lying."""
        assert classify_load_failure(ValueError("something entirely new")) == ""

    @pytest.mark.parametrize(
        "message",
        [
            "You are trying to access a gated repo.",
            "contains custom code which must be executed",
            "org/x is not a local folder and is not a valid model identifier",
        ],
    )
    def test_they_verdict_unverified_and_do_not_fail_the_run(self, message):
        def _raise(_base):
            raise OSError(message)

        check = check_attach(
            "r", _cfg(), load_hf_config=_raise, build_model=lambda *_: None
        )

        assert check.verdict is Verdict.UNVERIFIED, check
        assert PreflightReport([check]).exit_code == 0, (
            "an unverifiable config failed the run; on a box with no HF_TOKEN "
            "that is 51 recipes, and the guard gets muted"
        )

    def test_an_unknown_loader_failure_still_does_not_claim_it_cannot_attach(self):
        """The conservative direction: we did not reach the attach, so we do not
        get to say it fails. The message carries the exception verbatim."""
        def _raise(_base):
            raise RuntimeError("hub had a bad minute")

        check = check_attach(
            "r", _cfg(), load_hf_config=_raise, build_model=lambda *_: None
        )

        assert check.verdict is Verdict.UNVERIFIED
        assert "hub had a bad minute" in check.detail

    def test_a_recognised_build_failure_is_also_unverified(self):
        """The same rule one stage later, and the stage my first version of these
        tests missed entirely: ``from_config`` raises "Unrecognized model" for a
        config this transformers cannot build (MiniMax-M3 does exactly this).
        Filing that as CANNOT_ATTACH blames the recipe for the library's age."""
        def _boom(_config, _task):
            raise ValueError("Unrecognized model in org/x. Should have a `model_type`")

        check = check_attach(
            "r", _cfg(), load_hf_config=lambda _b: SimpleNamespace(model_type="x"),
            build_model=_boom,
        )

        assert check.verdict is Verdict.UNVERIFIED, check
        assert check.stage == "build"
        assert PreflightReport([check]).exit_code == 0

    def test_an_unrecognised_build_failure_is_an_attach_failure(self):
        """The other side, so the rule is not "never fail". The config loaded and
        the architecture is known; if it will not instantiate, that is real."""
        def _boom(_config, _task):
            raise TypeError("__init__() missing 1 required argument")

        check = check_attach(
            "r", _cfg(), load_hf_config=lambda _b: SimpleNamespace(model_type="x"),
            build_model=_boom,
        )

        assert check.verdict is Verdict.CANNOT_ATTACH
        assert check.stage == "build"


class TestNothingToAttach:
    """Neither of these is a failure, and reporting them as one puts permanent
    red rows in a report meant to go green."""

    def test_full_fine_tuning_has_no_adapter(self):
        check = check_attach(
            "r", _cfg(r=0), load_hf_config=_unreachable, build_model=_unreachable
        )

        assert check.verdict is Verdict.NO_ADAPTER
        assert "lora.r" in check.detail

    def test_the_mlx_backend_is_out_of_scope(self):
        check = check_attach(
            "r", _cfg(backend="mlx"),
            load_hf_config=_unreachable, build_model=_unreachable,
        )

        assert check.verdict is Verdict.NO_ADAPTER
        assert "mlx" in check.detail

    def test_neither_touches_the_network(self):
        """Pinned by `_unreachable` above, which raises if called -- the check
        must decide these before it asks the Hub anything."""
        assert plan_adapter(_cfg(r=0))[0] is False
        assert plan_adapter(_cfg(backend="mlx"))[0] is False
        assert plan_adapter(_cfg())[0] is True


def _unreachable(*_args, **_kwargs):
    raise AssertionError("the check reached the network for a config with no adapter")


class TestShrinking:
    def test_sub_configs_are_cut_too(self):
        """A vision-language wrapper keeps the text tower's depth in
        `text_config`; leaving it builds 61 layers whose names repeat after 2."""
        config = SimpleNamespace(
            num_hidden_layers=61,
            text_config=SimpleNamespace(num_hidden_layers=61),
            vision_config=SimpleNamespace(num_hidden_layers=27),
        )

        shrink_for_preflight(config)

        assert config.num_hidden_layers == PREFLIGHT_LAYERS
        assert config.text_config.num_hidden_layers == PREFLIGHT_LAYERS
        assert config.vision_config.num_hidden_layers == PREFLIGHT_LAYERS

    def test_a_model_already_smaller_is_left_alone(self):
        """Raising a 1-layer test model to 2 would change what is being checked."""
        config = SimpleNamespace(num_hidden_layers=1)

        shrink_for_preflight(config)

        assert config.num_hidden_layers == 1

    def test_a_config_without_layers_is_not_invented(self):
        config = SimpleNamespace(model_type="x")

        shrink_for_preflight(config)

        assert not hasattr(config, "num_hidden_layers")


class TestTheRealAttach:
    """End to end with the real resolution and the real `get_peft_model`, on
    real architectures built locally. No network, no weights."""

    @staticmethod
    def _local(config):
        return lambda _base: config

    @staticmethod
    def _meta(config, _task):
        from soup_cli.utils.attach_preflight import build_on_meta

        return build_on_meta(config, _task)

    def test_a_dense_llama_attaches(self):
        from transformers import LlamaConfig

        config = LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        )
        check = check_attach(
            "llama", _cfg(), load_hf_config=self._local(config), build_model=self._meta
        )

        assert check.verdict is Verdict.ATTACHES, check.detail
        assert check.adapted == 4, "q_proj and v_proj on two layers"
        assert check.expert_modules == 0

    def test_a_phi3_shaped_model_cannot_attach(self):
        """The finding this check exists for, reduced to a test. phi-4 is
        `model_type: phi3`; its attention is a fused `qkv_proj` + `o_proj`, so
        there is no `q_proj`/`v_proj` for a default to hit, peft 0.20 has no
        `phi3` key, and Soup's `auto` returns None. Three shipped recipes name
        that base."""
        from transformers import Phi3Config

        config = Phi3Config(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        )
        check = check_attach(
            "phi", _cfg(), load_hf_config=self._local(config), build_model=self._meta
        )

        assert check.verdict is Verdict.CANNOT_ATTACH, check
        assert check.model_type == "phi3"
        assert check.adapted == 0

    def test_the_configs_own_lora_settings_reach_peft(self):
        """Substituting a tidy ``r=8, dropout=0.0`` would have hidden #798
        completely: the dropout that made every MoE recipe unattachable was the
        schema default those recipes inherited, and a check that normalises it
        away cannot see the defect it exists to find.

        Asserted on the kwargs rather than on a failure, because reaching peft's
        ``ParamWrapper`` (where a non-zero dropout actually raises) needs
        ``target_parameters``, which is #798/#1074 machinery and not this
        module's business.
        """
        from transformers import LlamaConfig

        config = LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        )
        seen = {}

        def _record(model, kwargs):
            seen.update(kwargs)
            return SimpleNamespace(named_modules=lambda: [("a.lora_A", None)])

        check_attach(
            "llama", _cfg(r=64, dropout=0.05),
            load_hf_config=self._local(config), build_model=self._meta,
            attach=_record,
        )

        assert seen["lora_dropout"] == 0.05, "the check normalised the dropout away"
        assert seen["r"] == 64, "the check substituted its own rank"

    def test_an_attach_that_adapts_nothing_is_a_failure(self):
        """The defensive branch: peft normally raises when nothing matches, but if
        it ever returns a model with no adapter, "it attached" is the wrong
        answer -- that is a run that trains no LoRA parameters and looks fine,
        which is the v0.73.3 shape this repo has been bitten by before."""
        from transformers import LlamaConfig

        config = LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        )
        check = check_attach(
            "llama", _cfg(),
            load_hf_config=self._local(config), build_model=self._meta,
            attach=lambda _m, _k: SimpleNamespace(named_modules=lambda: []),
        )

        assert check.verdict is Verdict.CANNOT_ATTACH, check
        assert "no module carries an adapter" in check.detail

    def test_an_explicit_target_that_matches_nothing_fails(self):
        """`auto` is not the only way to reach nothing -- a hand-written list
        naming modules this architecture does not have reaches it too."""
        from transformers import LlamaConfig

        config = LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        )
        check = check_attach(
            "llama", _cfg(targets=["not_a_module_here"]),
            load_hf_config=self._local(config), build_model=self._meta,
        )

        assert check.verdict is Verdict.CANNOT_ATTACH


class TestTheBreakdowns:
    """`adapted=8` says nothing about whether the right 8. These two columns are
    the ones that caught a vision tower being adapted during a text fine-tune."""

    def test_expert_and_vision_modules_are_counted_apart(self):
        model = SimpleNamespace(
            named_modules=lambda: [
                ("model.layers.0.self_attn.q_proj.lora_A", None),
                ("model.layers.0.mlp.experts.gate_up_proj.lora_A", None),
                ("vision_tower.layers.0.self_attn.q_proj.lora_A", None),
                ("model.layers.0.self_attn.q_proj.lora_B", None),
            ]
        )

        assert count_adapted(model) == (3, 1, 1)

    def test_the_inner_adapter_module_is_not_counted_twice(self):
        """peft emits BOTH ``q_proj.lora_A`` (the ModuleDict) and
        ``q_proj.lora_A.<adapter>`` (the Linear inside it) from
        ``named_modules``. My first version matched either and reported 8 for a
        model with 4 adapted modules -- caught by the real llama test below, not
        by a fake."""
        model = SimpleNamespace(
            named_modules=lambda: [
                ("a.lora_A", None), ("a.lora_A.default", None),
                ("a.lora_B", None), ("a.lora_B.default", None),
            ]
        )

        assert count_adapted(model)[0] == 1

    def test_a_non_default_adapter_name_is_still_counted(self):
        """The other half of why ``.lora_A`` is the right anchor: peft names the
        inner Linear after the adapter, so matching ``lora_A.default`` alone
        reports 0 for any run that named its adapter something else."""
        model = SimpleNamespace(
            named_modules=lambda: [("a.lora_A", None), ("a.lora_A.my_adapter", None)]
        )

        assert count_adapted(model)[0] == 1


class TestTheExitCode:
    def test_only_cannot_attach_fails_the_run(self):
        rows = [
            AttachCheck("a", "b", "sft", Verdict.ATTACHES, adapted=4),
            AttachCheck("b", "b", "sft", Verdict.UNVERIFIED, detail="gated"),
            AttachCheck("c", "b", "sft", Verdict.NO_ADAPTER, detail="lora.r: 0"),
        ]

        assert PreflightReport(rows).exit_code == 0

    def test_one_failure_fails_the_run(self):
        rows = [
            AttachCheck("a", "b", "sft", Verdict.ATTACHES, adapted=4),
            AttachCheck("c", "b", "sft", Verdict.CANNOT_ATTACH, detail="nothing"),
        ]
        report = PreflightReport(rows)

        assert report.exit_code == 1
        assert [c.name for c in report.failures] == ["c"]


class TestTheCommand:
    """The CLI is a thin wrapper, so this pins only what the wrapper decides:
    which configs are in scope, and the exit code."""

    def _run(self, tmp_path, monkeypatch, verdict):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        def _fake(name, cfg, **_kwargs):
            return AttachCheck(name, cfg.base, cfg.task, verdict, detail="stubbed")

        monkeypatch.setattr("soup_cli.utils.attach_preflight.check_attach", _fake)
        path = tmp_path / "soup.yaml"
        path.write_text(
            "base: org/m\ntask: sft\ndata:\n  train: ./x.jsonl\n  format: alpaca\n"
            "training:\n  lora:\n    r: 8\n",
            encoding="utf-8",
        )
        return CliRunner().invoke(app, ["recipes", "verify", "--config", str(path)])

    def test_a_failing_config_exits_1(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, monkeypatch, Verdict.CANNOT_ATTACH)

        assert result.exit_code == 1, result.output

    def test_an_unverified_config_exits_0(self, tmp_path, monkeypatch):
        """The rule again, through the command: CI must not go red because the
        runner has no token."""
        result = self._run(tmp_path, monkeypatch, Verdict.UNVERIFIED)

        assert result.exit_code == 0, result.output

    def test_a_missing_config_is_refused(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(
            app, ["recipes", "verify", "--config", str(tmp_path / "nope.yaml")]
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()
