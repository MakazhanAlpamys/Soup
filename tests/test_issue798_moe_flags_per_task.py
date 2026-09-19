"""#798 — the MoE flags only ever worked on SFT.

``moe_lora`` picks LoRA targets that cover the expert FFNs. Only ``sft`` and
``pretrain`` (and ``tts`` through ``super()``) read it, yet 8 DPO and 7 GRPO
shipped recipes set it, so those recipes trained attention-only LoRA while
saying otherwise. ``moe_expert_quant`` and ``train_router_only`` are read by
the SFT path only, and ``moe_aux_loss_coeff`` by SFT and pretrain.

The ruling on the issue was (b): fix the recipes rather than document that they
do not work. So the five preference/RL trainers wire ``moe_lora``, and the
flags no trainer outside a named list reads are refused at config load.

The wiring tests drive each trainer's real ``_setup_transformers`` over a real
tiny ``Qwen3MoeForCausalLM`` with real peft. Only the loaders are stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.config.loader import load_config_from_string

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

_MOE_TASKS = ("dpo", "kto", "orpo", "simpo", "grpo")

#: The per-task minimum a config needs, beyond base/data.
_TASK_YAML = {
    "dpo": ("dpo", ""),
    "kto": ("kto", ""),
    "orpo": ("dpo", ""),
    "simpo": ("dpo", ""),
    "grpo": ("alpaca", "  reward_fn: length\n"),
    # sft is not in _MOE_TASKS (it always read the flag); it is here so the
    # dropout refusal can be driven through the path #798 assumed worked.
    "sft": ("alpaca", ""),
}


def _tiny_moe():
    """A real Qwen3-MoE, ~10k parameters. Experts are fused 3-D weights here
    (``mlp.experts.gate_up_proj``), which is what peft adapts."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            max_position_embeddings=64,
        )
    )


def _config(task: str, *, moe_lora: bool, extra: str = "", dropout: float = 0.0) -> object:
    data_format, training_extra = _TASK_YAML[task]
    return load_config_from_string(
        f"base: org/tiny-moe\n"
        f"task: {task}\n"
        f"backend: transformers\n"
        f"data:\n  train: x.jsonl\n  format: {data_format}\n"
        f"training:\n"
        f"  quantization: none\n"
        f"  moe_lora: {'true' if moe_lora else 'false'}\n"
        f"  lora:\n    r: 4\n    alpha: 8\n    dropout: {dropout}\n"
        f"{training_extra}{extra}"
    )


_WRAPPERS = {
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper"),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper"),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper"),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper"),
    "grpo": ("soup_cli.trainer.grpo", "GRPOTrainerWrapper"),
    "sft": ("soup_cli.trainer.sft", "SFTTrainerWrapper"),
}


_SMALL = dict(
    vocab_size=64,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
)


def _stand_in(arch):
    """A tiny real model per MoE family, for the per-architecture evidence.

    Only families with a config class in the installed transformers can be built;
    `mistral-large-3` and `kimi-k2.x` have none here, and the PR names them as
    untested rather than implying coverage.
    """
    import transformers as tf

    if arch == "qwen3_moe":
        return _tiny_moe()
    if arch == "mixtral":
        return tf.MixtralForCausalLM(tf.MixtralConfig(
            **_SMALL, intermediate_size=32, num_local_experts=4, num_experts_per_tok=2))
    if arch == "minimax":
        return tf.MiniMaxForCausalLM(tf.MiniMaxConfig(
            **_SMALL, intermediate_size=32, num_local_experts=4, num_experts_per_tok=2))
    if arch == "deepseek_v3":
        return tf.DeepseekV3ForCausalLM(tf.DeepseekV3Config(
            **_SMALL, intermediate_size=32, moe_intermediate_size=16, n_routed_experts=4,
            num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=0,
            n_group=1, topk_group=1))
    if arch == "glm4_moe":
        return tf.Glm4MoeForCausalLM(tf.Glm4MoeConfig(
            **_SMALL, intermediate_size=32, moe_intermediate_size=16, n_routed_experts=4,
            num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=0,
            n_group=1, topk_group=1))
    raise KeyError(arch)


_MOE_STAND_INS = {
    arch: (lambda a=arch: _stand_in(a))
    for arch in ("qwen3_moe", "deepseek_v3", "glm4_moe", "mixtral", "minimax")
}


def _tiny_dense():
    """A dense Qwen3 of the same size: the control for "MoE only"."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
        )
    )


def _attach_lora(
    task: str, monkeypatch, *, moe_lora: bool, dense=False, expect_failure=False,
    dropout: float = 0.0,
):
    """Run the trainer's own ``_setup_transformers`` far enough to attach LoRA.

    The model loader and tokenizer are stubbed; ``get_peft_model`` and the
    target-module resolution are the real ones, which is the point.
    """
    import importlib

    module = importlib.import_module(_WRAPPERS[task][0])
    wrapper_cls = getattr(module, _WRAPPERS[task][1])

    tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>", pad_token_id=0)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tokenizer
    )
    build = _tiny_dense if dense else _tiny_moe
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: build()
    )

    cfg = _config(task, moe_lora=moe_lora, dropout=dropout)
    wrapper = object.__new__(wrapper_cls)
    wrapper.config = cfg
    wrapper.device = "cpu"
    wrapper._trust_remote_code = False
    wrapper.model = None
    wrapper.tokenizer = None
    try:
        wrapper._setup_transformers(cfg, cfg.training)
    except ValueError:
        # peft's "no target_modules" refusal: the control asserts it.
        if expect_failure:
            raise
        raise
    except Exception as exc:  # noqa: BLE001 — setup continues past the LoRA attach
        if not _adapted(wrapper.model or object()):
            raise AssertionError(f"{task}: setup failed before LoRA attach: {exc!r}") from exc
    return wrapper.model


def _adapted(model) -> list[str]:
    if not hasattr(model, "named_modules"):
        return []
    return sorted(
        name.rsplit(".lora_A", 1)[0]
        for name, _ in model.named_modules()
        if name.endswith("lora_A")
    )


class TestMoeLoraReachesTheExpertFfns:
    @pytest.mark.parametrize("task", _MOE_TASKS)
    def test_moe_lora_true_adapts_the_experts(self, task, monkeypatch):
        """The 15 shipped DPO/GRPO recipes say moe_lora: true and trained
        attention-only LoRA."""
        adapted = _adapted(_attach_lora(task, monkeypatch, moe_lora=True))
        experts = [name for name in adapted if "experts" in name]
        assert experts, f"{task}: no expert module carries an adapter: {adapted}"

    @pytest.mark.parametrize("task", _MOE_TASKS)
    def test_without_the_flag_the_attach_fails_on_this_model(self, task, monkeypatch):
        """The control the ruling asked for, and it is stronger than expected: on a
        fused-expert Qwen3-MoE, ``target_modules: auto`` (what all 15 recipes use)
        resolves to None, and peft refuses. So these recipes could not attach LoRA
        at all, rather than quietly training attention-only."""
        with pytest.raises(ValueError, match="target_modules"):
            _attach_lora(task, monkeypatch, moe_lora=False, expect_failure=True)

    @pytest.mark.parametrize("task", _MOE_TASKS)
    def test_a_non_moe_base_is_untouched_by_the_flag(self, task, monkeypatch):
        """The other control: the wiring must not change a dense model's targets,
        whatever the flag says."""
        adapted_off = _adapted(_attach_lora(task, monkeypatch, moe_lora=False, dense=True))
        adapted_on = _adapted(_attach_lora(task, monkeypatch, moe_lora=True, dense=True))
        assert adapted_off == adapted_on, (adapted_off, adapted_on)
        assert adapted_on, "the dense control adapted nothing"


class TestTheFlagsNoTrainerReadsAreRefused:
    @pytest.mark.parametrize("task", ["dpo", "grpo", "kto", "pretrain"])
    @pytest.mark.parametrize(
        "knob", ["  moe_expert_quant: nf4\n", "  train_router_only: true\n"]
    )
    def test_sft_only_knobs_are_refused_off_sft(self, task, knob):
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(
                f"base: org/m\ntask: {task}\n"
                f"data:\n  train: x.jsonl\n  format: {_TASK_YAML.get(task, ('alpaca', ''))[0]}\n"
                # dropout 0 so the #798 dropout refusal does not answer first:
                # this test is about the task, not about the dropout.
                f"training:\n  moe_lora: true\n  lora:\n    dropout: 0.0\n{knob}"
                + ("  reward_fn: length\n" if task == "grpo" else "")
            )
        message = str(excinfo.value)
        assert task in message
        assert "sft" in message

    @pytest.mark.parametrize("knob", ["moe_expert_quant: nf4", "train_router_only: true"])
    def test_sft_still_accepts_them(self, knob):
        cfg = load_config_from_string(
            "base: org/m\ntask: sft\ndata: {train: x.jsonl}\n"
            f"training: {{moe_lora: true, lora: {{dropout: 0.0}}, {knob}}}\n"
        )
        assert cfg.task == "sft"

    @pytest.mark.parametrize("task", ["dpo", "grpo"])
    def test_a_non_default_aux_loss_coeff_is_refused(self, task):
        with pytest.raises(ValueError, match="moe_aux_loss_coeff"):
            load_config_from_string(
                f"base: org/m\ntask: {task}\n"
                f"data:\n  train: x.jsonl\n  format: {_TASK_YAML[task][0]}\n"
                "training:\n  moe_aux_loss_coeff: 0.05\n"
                + ("  reward_fn: length\n" if task == "grpo" else "")
            )

    @pytest.mark.parametrize("task", ["dpo", "grpo"])
    def test_the_default_aux_loss_coeff_still_loads(self, task):
        """Every shipped DPO/GRPO MoE recipe writes the default 0.01 explicitly,
        and a dumped config writes it too: refusing that would break them."""
        cfg = load_config_from_string(
            f"base: org/m\ntask: {task}\n"
            f"data:\n  train: x.jsonl\n  format: {_TASK_YAML[task][0]}\n"
            "training:\n  moe_aux_loss_coeff: 0.01\n"
            + ("  reward_fn: length\n" if task == "grpo" else "")
        )
        assert cfg.training.moe_aux_loss_coeff == 0.01

    @pytest.mark.parametrize("task", ["sft", "pretrain"])
    def test_a_non_default_aux_loss_coeff_is_accepted_where_it_is_read(self, task):
        cfg = load_config_from_string(
            f"base: org/m\ntask: {task}\ndata: {{train: x.jsonl}}\n"
            "training: {moe_aux_loss_coeff: 0.05}\n"
        )
        assert cfg.training.moe_aux_loss_coeff == 0.05


class TestShippedConfigs:
    def test_every_recipe_still_loads(self):
        from soup_cli.recipes.catalog import RECIPES

        for name, recipe in RECIPES.items():
            load_config_from_string(recipe.yaml_str), name

    def test_the_moe_recipes_are_the_ones_this_issue_names(self):
        """15, not the 8 the issue estimated: 8 dpo + 7 grpo."""
        import yaml

        from soup_cli.recipes.catalog import RECIPES

        tasks = []
        for recipe in RECIPES.values():
            cfg = yaml.safe_load(recipe.yaml_str)
            training = cfg.get("training") or {}
            if training.get("moe_lora") and cfg.get("task") in _MOE_TASKS:
                tasks.append(cfg["task"])
        assert sorted(tasks) == ["dpo"] * 8 + ["grpo"] * 7, sorted(tasks)



class TestTheDropoutConstraint:
    """peft's ParamWrapper refuses dropout on FUSED MoE experts.

    Checked against the model, not at config load. The first version of this
    branch refused ``moe_lora`` + non-zero dropout in the schema, which is wrong
    in two directions: it fired for a dense base, where the flag is a documented
    no-op and nothing would have broken, and it fired ahead of more specific
    validators, so ``stream_layers`` conflicts started reporting a dropout
    problem instead (tests/test_v07200.py::TestStreamMutualExclusions). At load
    time the model is a string; whether its experts are fused is not knowable.
    """

    def _tcfg(self, dropout):
        return SimpleNamespace(moe_lora=True, lora=SimpleNamespace(dropout=dropout))

    def test_fused_experts_plus_dropout_is_refused(self):
        from soup_cli.utils.moe import resolve_moe_lora_targets

        with pytest.raises(ValueError) as excinfo:
            resolve_moe_lora_targets(_tiny_moe(), self._tcfg(0.05), ["q_proj"])
        message = str(excinfo.value)
        assert "lora.dropout: 0.0" in message
        # peft's own words, or the rule reads as arbitrary and the first thing a
        # user does is set the dropout back (#798 ruling).
        assert "lora.ParamWrapper does not work with lora_dropout != 0" in message

    def test_the_same_model_with_dropout_zero_gets_the_expert_targets(self):
        from soup_cli.utils.moe import resolve_moe_lora_targets

        targets = resolve_moe_lora_targets(_tiny_moe(), self._tcfg(0.0), ["q_proj"])
        assert any("proj" in t for t in targets) and targets != ["q_proj"]

    def test_a_dense_model_with_dropout_is_not_refused(self):
        """The control: a dense base has no fused experts, peft is happy, and
        refusing it was the bug in the first version of this branch."""
        from soup_cli.utils.moe import resolve_moe_lora_targets

        assert resolve_moe_lora_targets(
            _tiny_dense(), self._tcfg(0.05), ["q_proj"]
        ) == ["q_proj"]

    def test_the_predicate_follows_peft_not_the_parameter_shapes(self):
        """Mixtral and MiniMax have the SAME fused 3-D expert parameters and the
        same module structure as Qwen3-MoE, and peft attaches to them happily at
        dropout 0.05. A model-wide "are there fused expert params" scan -- the
        first version of this check, and what #1074's review caught -- refuses
        those two for a reason that does not apply to them.

        What actually differs is inside peft: its v4->v5 checkpoint conversion
        rewrites target_modules into target_parameters only for model types that
        have a conversion mapping, and only that rewrite produces the
        ParamWrapper which refuses dropout.
        """
        from soup_cli.utils.moe import (
            get_moe_target_modules,
            peft_routes_lora_to_fused_params,
        )

        routed, fused_params = {}, {}
        for name, build in _MOE_STAND_INS.items():
            model = build()
            targets = get_moe_target_modules(model)
            routed[name] = peft_routes_lora_to_fused_params(model, targets)
            fused_params[name] = sorted({
                param_name.rsplit(".", 1)[-1]
                for param_name, param in model.named_parameters()
                if "expert" in param_name.lower() and param.ndim == 3
            })

        assert routed == {
            "qwen3_moe": True,
            "deepseek_v3": True,
            "glm4_moe": True,
            "mixtral": False,
            "minimax": False,
        }, routed
        # The control that makes the point: every one of them HAS fused 3-D
        # expert parameters, under the same names.
        assert all(names == ["down_proj", "gate_up_proj"] for names in fused_params.values()), (
            fused_params
        )

    @pytest.mark.parametrize("arch", ["mixtral", "minimax"])
    def test_an_architecture_peft_does_not_route_is_not_refused(self, arch):
        """The over-refusal this replaced: peft accepts these at dropout 0.05,
        so Soup must not refuse them. Verified against the real attach below."""
        from soup_cli.utils.moe import resolve_moe_lora_targets

        model = _MOE_STAND_INS[arch]()
        assert resolve_moe_lora_targets(model, self._tcfg(0.05), ["q_proj"]) is not None

    @pytest.mark.parametrize("arch", ["mixtral", "minimax"])
    def test_peft_really_accepts_those_two_at_dropout(self, arch):
        """Not taken from the review on trust: the attach Soup now allows."""
        from peft import LoraConfig, TaskType, get_peft_model

        from soup_cli.utils.moe import get_moe_target_modules

        model = _MOE_STAND_INS[arch]()
        get_peft_model(model, LoraConfig(
            r=4, lora_dropout=0.05, target_modules=get_moe_target_modules(model),
            task_type=TaskType.CAUSAL_LM,
        ))

    def test_a_stub_model_with_peft_patched_out_is_never_refused(self, monkeypatch):
        """The shape `tests/test_pretrain.py` actually uses, which caught this.

        That suite stubs the model AND patches `peft.LoraConfig`, so the probe
        built a MagicMock config whose `target_parameters` is truthy for any
        model -- and a stub with no experts got refused. Patching LoraConfig is
        the load-bearing half: with the real class a bare MagicMock returns None
        and the bug hides, which is how my first version of this test passed
        while proving nothing.
        """
        from unittest.mock import MagicMock

        import peft

        from soup_cli.utils.moe import peft_routes_lora_to_fused_params

        monkeypatch.setattr(peft, "LoraConfig", MagicMock())
        assert peft_routes_lora_to_fused_params(MagicMock(), ["down_proj"]) is False

    @pytest.mark.parametrize("arch", ["qwen3_moe", "deepseek_v3", "glm4_moe"])
    def test_the_routed_architectures_are_refused(self, arch):
        from soup_cli.utils.moe import resolve_moe_lora_targets

        with pytest.raises(ValueError, match="ParamWrapper"):
            resolve_moe_lora_targets(_MOE_STAND_INS[arch](), self._tcfg(0.05), ["q_proj"])

    @pytest.mark.parametrize("task", ["dpo", "sft"] )
    def test_the_refusal_reaches_a_real_trainer_setup(self, task, monkeypatch):
        """End to end through the trainer's own setup, not just the helper.

        Asserts SOUP's wording, not peft's: with the helper's refusal removed,
        peft raises its own ParamWrapper error from the same line, so matching
        on "ParamWrapper" alone passes whether or not the wiring is there.
        Found by mutation N2.
        """
        if task == "sft":
            pytest.importorskip("trl")
        with pytest.raises(ValueError) as excinfo:
            _attach_lora(task, monkeypatch, moe_lora=True, dropout=0.05,
                         expect_failure=True)
        message = str(excinfo.value)
        assert "training.moe_lora=true needs training.lora.dropout: 0.0" in message
        assert "lora.ParamWrapper does not work with lora_dropout != 0" in message

    def test_the_config_itself_still_loads(self):
        """Deliberate: no load-time refusal. If someone re-adds one, this fails
        and the trade-off above gets read again rather than rediscovered."""
        cfg = load_config_from_string(
            "base: org/m\ntask: sft\ndata: {train: x.jsonl}\n"
            "training: {moe_lora: true, lora: {dropout: 0.05}}\n"
        )
        assert cfg.training.lora.dropout == 0.05

    def test_dropout_is_untouched_without_moe_lora(self):
        """The rule is about moe_lora, not about dropout."""
        cfg = load_config_from_string(
            "base: org/m\ntask: sft\ndata: {train: x.jsonl}\n"
            "training: {lora: {dropout: 0.05}}\n"
        )
        assert cfg.training.lora.dropout == 0.05

    def test_peft_really_refuses_it(self):
        """Not taken on trust: the constraint this rule exists for, reproduced.

        A dropout LoRA over the fused experts raises inside peft; the same attach
        with dropout 0 succeeds and adapts the expert module."""
        from peft import LoraConfig, TaskType, get_peft_model

        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        with pytest.raises(ValueError, match="ParamWrapper"):
            get_peft_model(
                _tiny_moe(),
                LoraConfig(r=4, lora_dropout=0.05, target_modules=targets,
                           task_type=TaskType.CAUSAL_LM),
            )
        model = get_peft_model(
            _tiny_moe(),
            LoraConfig(r=4, lora_dropout=0.0, target_modules=targets,
                       task_type=TaskType.CAUSAL_LM),
        )
        assert [n for n in _adapted(model) if "experts" in n]


class TestTheSweepInteraction:
    """`soup sweep` can generate the combination this PR refuses (#1074 ask 4).

    `commands/sweep.py`'s parameter map exposes `moe_aux_loss_coeff`, so a grid
    over it on `dpo` or `grpo` now hard-fails at config load for every non-
    default point. The values are user-typed rather than Soup-generated, so the
    warn-then-refuse staging for generated values does not cover it -- but the
    interaction should be a known fact rather than a surprise.
    """

    def test_the_sweep_shortcut_writes_the_refused_knob(self):
        """Driven through the function rather than reading its dict: the map is
        a local in `_set_nested_param`, and what matters is where a sweep point
        lands, not how the table is spelled."""
        from soup_cli.commands.sweep import _set_nested_param

        written = _set_nested_param({}, "moe_aux_loss_coeff", 0.05)
        assert written["training"]["moe_aux_loss_coeff"] == 0.05

    def test_a_non_default_point_on_dpo_is_refused_and_the_default_is_not(self):
        yaml_ = ("base: org/m\ntask: dpo\n"
                 "data:\n  train: x.jsonl\n  format: dpo\n"
                 "training:\n  moe_aux_loss_coeff: {}\n")
        assert load_config_from_string(yaml_.format("0.01")).training.moe_aux_loss_coeff == 0.01
        with pytest.raises(ValueError, match="moe_aux_loss_coeff"):
            load_config_from_string(yaml_.format("0.05"))


class TestPerArchitectureCoverage:
    """Which families the wiring actually reaches, measured (#1074 review ask 3).

    The wiring is not uniform across MoE families, and saying so is the point:
    a recipe that looks fixed and is not is worse than one that visibly fails.
    """

    #: What `get_moe_target_modules` + a real peft attach does per family, on
    #: tiny stand-ins. `expert adapters > 0` is the thing `moe_lora` promises.
    EXPECTED = {
        "qwen3_moe": True,
        "deepseek_v3": True,
        "glm4_moe": True,
        "mixtral": False,
        "minimax": False,
    }

    @pytest.mark.parametrize("arch", sorted(EXPECTED))
    def test_expert_coverage_is_what_the_pr_claims(self, arch):
        from peft import LoraConfig, TaskType, get_peft_model

        from soup_cli.utils.moe import get_moe_target_modules

        model = _MOE_STAND_INS[arch]()
        attached = get_peft_model(model, LoraConfig(
            r=4, lora_dropout=0.0, target_modules=get_moe_target_modules(model),
            task_type=TaskType.CAUSAL_LM,
        ))
        experts = [name for name in _adapted(attached) if "expert" in name.lower()]
        assert bool(experts) is self.EXPECTED[arch], (arch, experts)

    def test_the_uncovered_families_are_named_in_the_docs(self):
        """MiniMax gets zero expert adapters, so `minimax-m3-sft` and
        `minimax-m3-dpo` still train attention-only after this PR. That has to be
        written down where a user looks, not only in a test."""
        from pathlib import Path

        docs = (Path(__file__).resolve().parents[1]
                / "docs" / "performance-and-quantization.md").read_text(encoding="utf-8")
        lowered = docs.lower()
        assert "minimax-m3-sft" in lowered and "minimax-m3-dpo" in lowered, (
            "the two recipes the wiring does not reach must be named in the docs"
        )
        assert "attention-only" in lowered, "and what they do instead"


class TestEveryMoeRecipeCanAttach:
    """Recipe 32 must not arrive broken on the day it ships (#798 ruling)."""

    def test_every_recipe_that_sets_moe_lora_pins_dropout_zero(self):
        import yaml

        from soup_cli.recipes.catalog import RECIPES

        offenders = []
        for name, recipe in RECIPES.items():
            training = yaml.safe_load(recipe.yaml_str).get("training") or {}
            if not training.get("moe_lora"):
                continue
            dropout = (training.get("lora") or {}).get("dropout")
            if dropout != 0.0:
                offenders.append(f"{name}: lora.dropout={dropout!r}")
        assert offenders == [], (
            "a recipe with moe_lora: true and a non-zero lora.dropout cannot "
            "attach LoRA at all (peft ParamWrapper): " + "; ".join(offenders)
        )

    def test_the_guard_sees_the_recipes_it_is_guarding(self):
        """A control: the loop above passes vacuously if nothing sets moe_lora."""
        import yaml

        from soup_cli.recipes.catalog import RECIPES

        counted = sum(
            1 for r in RECIPES.values()
            if (yaml.safe_load(r.yaml_str).get("training") or {}).get("moe_lora")
        )
        assert counted >= 31, counted
