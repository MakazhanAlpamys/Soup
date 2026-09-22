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


def _cfg(
    base="org/m", task="sft", r=8, dropout=0.0, backend=None, targets="auto",
    modality="text",
):
    """A stand-in for the loaded SoupConfig, carrying only what the check reads."""
    return SimpleNamespace(
        base=base,
        task=task,
        modality=modality,
        backend=backend,
        training=SimpleNamespace(
            lora=SimpleNamespace(r=r, alpha=16, dropout=dropout, target_modules=targets)
        ),
    )


def _real_cfg(r=8, dropout=0.0, targets="auto", moe_lora=False):
    """A config loaded through the real schema, for every test that reaches the
    attach. The check now builds the adapter with the trainer's own
    ``build_lora_config``, which reads the whole LoRA block (``use_dora``,
    ``rank_pattern``, the #1037 variants ...), so a hand-built stand-in would be
    testing a config no user can write."""
    import json

    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(
        "base: org/m\ntask: sft\n"
        "data:\n  train: ./x.jsonl\n  format: alpaca\n"
        f"training:\n  moe_lora: {'true' if moe_lora else 'false'}\n"
        f"  lora:\n    r: {r}\n    alpha: 16\n    dropout: {dropout}\n"
        f"    target_modules: {json.dumps(targets)}\n"
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


class TestTheLoaderMatchesTheTrainer:
    """The class a config is built with has to be the one its trainer loads, or
    the verdict is about a different model. SFT picks by MODALITY
    (``trainer/sft.py``: causal LM for text, ``AutoModelForImageTextToText`` for
    vision, ``AutoModel`` for audio); my first version keyed on the task and
    built every vision recipe as a causal LM. It was caught by re-verifying the
    multimodal rows of #1116 against the real loaders, not by these tests --
    which is why these now exist."""

    def test_vision_sft_uses_the_image_text_class(self):
        from soup_cli.utils.attach_preflight import loader_for

        assert loader_for(_cfg(modality="vision")) == ("AutoModelForImageTextToText",)

    def test_audio_sft_uses_the_bare_model(self):
        from soup_cli.utils.attach_preflight import loader_for

        assert loader_for(_cfg(modality="audio")) == ("AutoModel",)

    def test_text_sft_uses_a_causal_lm(self):
        from soup_cli.utils.attach_preflight import loader_for

        # Exact, not [0]: a fallback at [1] is the bug this pins (see
        # test_a_text_config_the_causal_lm_class_refuses_is_not_attached).
        assert loader_for(_cfg()) == ("AutoModelForCausalLM",)

    def test_a_task_with_its_own_head_wins_over_modality(self):
        """An embedding run is ``AutoModel`` whatever the modality says --
        ``trainer/embedding.py`` never loads a causal LM."""
        from soup_cli.utils.attach_preflight import loader_for

        assert loader_for(_cfg(task="embedding")) == ("AutoModel",)
        assert loader_for(_cfg(task="reward_model")) == (
            "AutoModelForSequenceClassification",
        )


class TestRemoteCodeIsNeverOffered:
    """The module promises remote code is never executed. The first version kept
    that promise for the config and broke it for the model: ``from_config`` was
    called without ``trust_remote_code``, and for a custom-code architecture
    transformers does not refuse -- it PROMPTS on stdin, "Do you wish to run the
    custom code? [y/N]". That hangs a terminal, corrupts ``--json``, and runs
    remote code for anyone who types y. Kimi-K2.5 is the live case: its config
    loads with ``trust_remote_code=False`` while its modeling code is remote.
    Found by running the real command end to end, not by any test here."""

    def test_every_build_passes_trust_remote_code_false(self, monkeypatch):
        from soup_cli.utils import attach_preflight

        seen = []

        def _spy(_config, **kwargs):
            seen.append(kwargs)
            return SimpleNamespace()

        # Patch the method on the real class: ``build_on_meta`` resolves the
        # class through transformers' lazy module, which a module-level setattr
        # does not reliably intercept.
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", _spy)
        attach_preflight.build_on_meta(SimpleNamespace(), ("AutoModelForCausalLM",))

        assert seen == [{"trust_remote_code": False}], (
            "from_config was called without trust_remote_code=False; transformers "
            "will prompt on stdin for a custom-code architecture"
        )

    def test_the_config_load_refuses_remote_code_too(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            transformers.AutoConfig, "from_pretrained",
            lambda base, **kwargs: seen.append(kwargs) or SimpleNamespace(),
        )
        from soup_cli.utils.attach_preflight import load_hf_config

        load_hf_config("org/m")

        assert seen == [{"trust_remote_code": False}]


class TestJsonIsParseable:
    """``--json`` exists for a program to read, so stdout must be the document
    and nothing else."""

    def _invoke(self, tmp_path, monkeypatch, *, force_colour):
        import json as _json

        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.setattr(
            "soup_cli.utils.attach_preflight.check_attach",
            lambda name, cfg, **_k: AttachCheck(
                name, cfg.base, cfg.task, Verdict.ATTACHES, adapted=4
            ),
        )
        if force_colour:
            # The module-level Console is built at import time, so FORCE_COLOR set
            # here would never reach it and the assertion below would pass
            # without testing anything -- which is exactly what my first version
            # did (the print_json mutant survived). Pin a colour-forced console
            # into the module instead; it resolves sys.stdout at write time, so
            # CliRunner still captures it.
            from rich.console import Console

            import soup_cli.commands.recipes as recipes_cmd

            monkeypatch.setattr(recipes_cmd, "console", Console(force_terminal=True))
        good = tmp_path / "soup.yaml"
        good.write_text(
            "base: org/m\ntask: sft\ndata:\n  train: ./x.jsonl\n  format: alpaca\n"
            "training:\n  lora:\n    r: 8\n",
            encoding="utf-8",
        )
        result = CliRunner().invoke(app, ["recipes", "verify", "--config", str(good), "--json"])
        return result, _json

    def test_stdout_parses_as_json(self, tmp_path, monkeypatch):
        result, json_mod = self._invoke(tmp_path, monkeypatch, force_colour=False)

        rows = json_mod.loads(result.stdout)
        assert rows[0]["verdict"] == "attaches"

    def test_it_still_parses_when_colour_is_forced(self, tmp_path, monkeypatch):
        """Rich's ``print_json`` syntax-colours its output, and under
        ``FORCE_COLOR`` -- which many CI runners set -- the escape codes make the
        document unparseable. Measured: ``json.loads`` raised JSONDecodeError on
        ``print_json`` output with a forced terminal."""
        result, json_mod = self._invoke(tmp_path, monkeypatch, force_colour=True)

        assert "\x1b[" not in result.stdout, "ANSI colour codes in --json output"
        json_mod.loads(result.stdout)


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
    def _meta(config, classes):
        from soup_cli.utils.attach_preflight import build_on_meta

        return build_on_meta(config, classes)

    def test_a_dense_llama_attaches(self):
        from transformers import LlamaConfig

        config = LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        )
        check = check_attach(
            "llama", _real_cfg(), load_hf_config=self._local(config), build_model=self._meta
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
            "phi", _real_cfg(), load_hf_config=self._local(config), build_model=self._meta
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

        def _record(model, peft_config):
            seen["config"] = peft_config
            return SimpleNamespace(named_modules=lambda: [("a.lora_A", None)])

        check_attach(
            "llama", _real_cfg(r=64, dropout=0.05),
            load_hf_config=self._local(config), build_model=self._meta,
            attach=_record,
        )

        # The config handed to peft IS the trainer's -- build_lora_config's output.
        assert seen["config"].lora_dropout == 0.05, "the check normalised the dropout away"
        assert seen["config"].r == 64, "the check substituted its own rank"

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
            "llama", _real_cfg(),
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
            "llama", _real_cfg(targets=["not_a_module_here"]),
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


class TestTheVerdictIsTheTrainers:
    """The check must answer for what the TRAINER does, not for a lookalike.
    Two ways the first version answered for a lookalike, each now pinned."""

    @staticmethod
    def _qwen2_moe_config():
        from transformers import Qwen2MoeConfig

        return Qwen2MoeConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            moe_intermediate_size=16, shared_expert_intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            num_experts=4, num_experts_per_tok=2,
        )

    def _check(self, cfg, hf_config, classes=None):
        from soup_cli.utils.attach_preflight import build_on_meta

        return check_attach(
            "r", cfg, load_hf_config=lambda _b: hf_config,
            build_model=lambda c, _cls: build_on_meta(c, classes or _cls),
        )

    def test_moe_lora_rescues_an_unmapped_moe_as_it_does_in_the_trainer(self):
        """``qwen2_moe`` is mapped by neither Soup nor peft, so ``auto`` resolves to
        nothing -- and the trainer then replaces ``target_modules`` from the
        ``moe_lora`` scan. The first version skipped that step and called such a
        recipe broken: measured on ``qwen3-30b-a3b-sft`` it said FAILS while the
        trainer's path attached 12 modules."""
        check = self._check(_real_cfg(moe_lora=True), self._qwen2_moe_config())

        assert check.verdict is Verdict.ATTACHES, check.detail
        assert check.adapted > 0

    def test_without_moe_lora_the_same_model_cannot_attach(self):
        """The control: it is the ``moe_lora`` step that attached it above, not a
        check that has stopped failing anything."""
        check = self._check(_real_cfg(moe_lora=False), self._qwen2_moe_config())

        assert check.verdict is Verdict.CANNOT_ATTACH, check

    def test_the_attach_sees_the_keys_the_trainer_hands_peft(self):
        """The maintainer's review ask on #1116: run against the object the trainer
        hands ``get_peft_model``, INCLUDING the ``model.`` prefix peft matches on.

        peft fullmatches a string target against the whole module key, and under
        ``AutoModelForImageTextToText`` -- what vision SFT loads -- the key is
        ``model.language_model....``. So a pattern anchored at ``language_model``
        genuinely cannot attach, and a check that built the model at a different
        level (``AutoModel``, which drops the ``model.`` wrapper) would report it
        green. That exact regex shipped in #1102 and was caught at review."""
        from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig

        vl = LlavaConfig(
            text_config=LlamaConfig(
                vocab_size=64, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                pad_token_id=0,
            ),
            vision_config=CLIPVisionConfig(
                hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=2, image_size=32, patch_size=16,
            ),
            image_token_index=63,
        )
        classes = ("AutoModelForImageTextToText",)
        anchored = _real_cfg(targets=r"language_model\..*\.self_attn\.(q_proj|v_proj)")
        prefixed = _real_cfg(targets=r".*language_model\..*\.self_attn\.(q_proj|v_proj)")

        broken = self._check(anchored, vl, classes)
        working = self._check(prefixed, vl, classes)

        assert broken.verdict is Verdict.CANNOT_ATTACH, (
            "an anchored regex attached -- the check is not seeing the model. prefix"
        )
        assert working.verdict is Verdict.ATTACHES, working.detail
        assert working.vision_modules == 0, "and it stayed out of the vision tower"


    def test_a_text_config_the_causal_lm_class_refuses_is_not_attached(self):
        """No trainer falls back from ``AutoModelForCausalLM`` to ``AutoModel``;
        SFT's text path loads exactly one class. My first version did fall back,
        so a config the causal-LM class refuses was built one level down and
        reported as ATTACHING -- MiniMax-M3 with no ``modality`` read green, while
        in SFT it cannot even load (#1145). A LLaVA reproduces the shape locally:
        ``AutoModelForCausalLM`` refuses it, ``AutoModel`` builds ``LlavaModel``."""
        from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig

        from soup_cli.utils.attach_preflight import build_on_meta, loader_for

        vl = LlavaConfig(
            text_config=LlamaConfig(
                vocab_size=64, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                pad_token_id=0,
            ),
            vision_config=CLIPVisionConfig(
                hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=2, image_size=32, patch_size=16,
            ),
            image_token_index=63,
        )
        text_cfg = _real_cfg()                       # no modality -> the text path

        check = check_attach(
            "vl-as-text", text_cfg, load_hf_config=lambda _b: vl,
            build_model=lambda c, _cls: build_on_meta(c, loader_for(text_cfg)),
        )

        assert check.verdict is Verdict.CANNOT_ATTACH, (
            f"reported {check.verdict.value} -- the check built it with a class "
            "the trainer never uses"
        )
        assert check.stage == "build"
