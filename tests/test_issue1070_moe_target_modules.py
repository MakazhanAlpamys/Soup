"""#1070: ``target_modules: auto`` resolved to ``None`` for every MoE architecture.

``resolve_lora_target_modules`` mapped the Qwen3.5 and Qwen4-Exp families and
returned ``None`` for everything else, delegating to peft. peft has no default
for any MoE ``model_type`` Soup ships a recipe for, so the attach did not fall
back -- it raised ``No target_modules passed but also no target_parameters
found``, and every one of the 31 shipped MoE recipes uses ``target_modules:
auto``.

Measured on peft 0.20 / transformers 5.16.1, shrunk real models on CPU, before
and after this change:

    model_type      main            with the table
    qwen3_moe       ValueError      8 adapted modules
    deepseek_v3     ValueError      10
    deepseek_v4     ValueError      10
    glm_moe_dsa     ValueError      10
    minimax_m3_vl   ValueError      8, none of them in the vision tower
    mixtral         ValueError      ValueError  (unchanged control)

The tests here use fake config objects, because the resolver only ever reads
``model_type``; the live attach is in the PR.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import pytest
import yaml

from soup_cli.recipes.catalog import list_recipes
from soup_cli.utils.peft_wiring import (
    MOE_TEXT_LORA_TARGETS,
    QWEN35_TEXT_LORA_TARGETS,
    resolve_lora_target_modules,
)


def _model(model_type, text_type=None):
    """The only thing the resolver reads is ``config.model_type``."""
    text_config = types.SimpleNamespace(model_type=text_type) if text_type else None
    return types.SimpleNamespace(
        config=types.SimpleNamespace(model_type=model_type, text_config=text_config)
    )


class TestTheTableResolves:
    @pytest.mark.parametrize("model_type", sorted(MOE_TEXT_LORA_TARGETS))
    def test_every_entry_resolves_to_something_peft_can_use(self, model_type):
        """Not ``None``, which is what the bug was, and not empty -- an empty list
        reaches peft as "no targets" just as ``None`` does."""
        resolved = resolve_lora_target_modules(_model(model_type), "auto")

        assert resolved, f"{model_type} resolved to {resolved!r}"
        assert isinstance(resolved, (list, str))

    def test_the_attention_lists_are_the_measured_ones(self):
        """Pinned literally. These came from instantiating each architecture and
        listing its ``nn.Linear`` modules; a plausible-looking edit to one of
        them is a guess unless the probe is re-run."""
        assert MOE_TEXT_LORA_TARGETS["qwen3_moe"] == (
            "q_proj", "k_proj", "v_proj", "o_proj"
        )
        assert MOE_TEXT_LORA_TARGETS["deepseek_v3"] == (
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"
        )
        assert MOE_TEXT_LORA_TARGETS["deepseek_v4"] == (
            "q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj"
        )
        assert MOE_TEXT_LORA_TARGETS["glm_moe_dsa"] == MOE_TEXT_LORA_TARGETS["deepseek_v3"]
        assert MOE_TEXT_LORA_TARGETS["kimi_k25"] == MOE_TEXT_LORA_TARGETS["deepseek_v3"]

    def test_no_entry_names_a_routed_expert_parameter(self):
        """Routed experts are 3-D ``nn.Parameter`` tensors on several of these, so
        ``target_modules`` cannot reach them at all; naming one would silently
        target nothing. Expert adaptation is ``target_parameters`` work (#798)."""
        for model_type, entry in MOE_TEXT_LORA_TARGETS.items():
            names = [entry] if isinstance(entry, str) else list(entry)
            for name in names:
                assert "expert" not in name, (model_type, name)

    def test_a_returned_list_is_a_copy(self):
        """A caller that mutates its targets must not edit the table. ``sft.py``
        reassigns ``target_modules`` for ``moe_lora``, and a shared list is how
        that kind of edit leaks into the next model in the same process."""
        first = resolve_lora_target_modules(_model("qwen3_moe"), "auto")
        first.append("mutated")

        assert "mutated" not in resolve_lora_target_modules(_model("qwen3_moe"), "auto")


class TestTheVisionTowerIsNotAdapted:
    """MiniMax-M3 is a VL wrapper whose ``vision_tower`` has its own ``q_proj`` /
    ``k_proj`` / ``v_proj``, so a suffix list would adapt the image encoder during a
    text fine-tune. peft reads a string target as a regex and ``fullmatch``es it
    against the WHOLE module key.

    My first version of these tests checked the regex against hand-written keys
    like ``language_model.layers.0.self_attn.q_proj``. The real keys, under the
    class vision SFT loads (``AutoModelForImageTextToText``), carry a ``model.``
    prefix, so the shipped regex matched **nothing** and the attach raised -- and
    the tests stayed green because they never touched a model (#1102 review, F2).
    So every key here is read off a real model's ``named_modules()``, and the
    decisive assertion is a real ``get_peft_model``.

    The stand-in is a tiny LLaVA, not MiniMax-M3 itself, because MiniMax's
    config is on the Hub and the suite does not touch the network. What matters
    is the shape the regex has to survive, and LLaVA has exactly it: a
    ``model.language_model.*.self_attn.{q,k,v,o}_proj`` tower beside a
    ``model.vision_tower.*.self_attn.{q,k,v}_proj`` one.
    """

    @staticmethod
    def _vl_model():
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            CLIPVisionConfig,
            LlamaConfig,
            LlavaConfig,
        )

        config = LlavaConfig(
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
        with torch.device("meta"):
            return AutoModelForImageTextToText.from_config(config)

    @staticmethod
    def _linear_keys(model):
        import torch

        return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]

    def test_the_keys_are_prefixed_the_way_peft_sees_them(self):
        """The premise, read off the model rather than assumed: the language
        tower's keys start with ``model.``, and the vision tower has attention
        projections of its own for a careless target to hit."""
        keys = self._linear_keys(self._vl_model())

        language = [k for k in keys if "language_model" in k and "self_attn" in k]
        vision = [k for k in keys if "vision_tower" in k and "self_attn" in k]
        assert language and all(k.startswith("model.language_model.") for k in language)
        assert vision, "sanity: the vision tower has attention projections"

    def test_the_regex_matches_every_language_attention_projection(self):
        pattern = resolve_lora_target_modules(_model("minimax_m3_vl"), "auto")
        keys = self._linear_keys(self._vl_model())
        language = [
            k for k in keys if "language_model" in k
            and re.search(r"self_attn\.(q|k|v|o)_proj$", k)
        ]

        assert isinstance(pattern, str), "peft reads a string target as a regex"
        assert language, "sanity"
        assert all(re.fullmatch(pattern, k) for k in language), [
            k for k in language if not re.fullmatch(pattern, k)
        ]

    def test_the_regex_matches_nothing_outside_the_language_tower(self):
        pattern = resolve_lora_target_modules(_model("minimax_m3_vl"), "auto")
        keys = self._linear_keys(self._vl_model())
        outside = [k for k in keys if "language_model" not in k]

        assert outside, "sanity: there are non-language linears to exclude"
        assert not [k for k in outside if re.fullmatch(pattern, k)]

    def test_a_real_attach_adapts_the_language_tower_only(self):
        """The assertion that would have failed on the shipped regex: peft raised
        ``Target modules ... not found in the base model``."""
        from peft import LoraConfig, get_peft_model

        pattern = resolve_lora_target_modules(_model("minimax_m3_vl"), "auto")
        attached = get_peft_model(
            self._vl_model(), LoraConfig(r=4, lora_alpha=8, target_modules=pattern)
        )
        adapted = [n for n, _ in attached.named_modules() if n.endswith(".lora_A")]

        assert adapted, "the regex attached to nothing"
        assert all("language_model" in n for n in adapted), adapted
        assert not any("vision" in n for n in adapted)

    def test_the_text_only_config_uses_plain_suffixes(self):
        """Loaded without the wrapper there is no ``language_model.`` segment and
        no vision tower, so the regex would match nothing at all."""
        resolved = resolve_lora_target_modules(_model("minimax_m3_vl_text"), "auto")

        assert resolved == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_the_wrapper_wins_over_its_own_text_config(self):
        """A VL model reports both types; the wrapper's entry is the one that knows
        about the vision tower, so it must not be decided by set order."""
        pattern = resolve_lora_target_modules(
            _model("minimax_m3_vl", text_type="minimax_m3_vl_text"), "auto"
        )

        assert isinstance(pattern, str)

    def test_the_wrapper_wins_even_when_the_text_type_sorts_first(self):
        resolved = resolve_lora_target_modules(
            _model("minimax_m3_vl", text_type="deepseek_v3"), "auto"
        )

        assert isinstance(resolved, str), f"took the text_config entry: {resolved!r}"


class TestWhichConfigTheTypeComesFrom:
    def test_a_type_only_on_the_text_config_still_resolves(self):
        """A wrapper Soup has never seen around a text tower it has. Reading only
        the outer ``model_type`` misses it, which is the shape that made Qwen3.5
        need both configs in the first place."""
        resolved = resolve_lora_target_modules(
            _model("some_unseen_wrapper", text_type="deepseek_v3"), "auto"
        )

        assert resolved == list(MOE_TEXT_LORA_TARGETS["deepseek_v3"])


class TestWhatIsLeftAlone:
    def test_an_explicit_list_is_returned_unchanged(self):
        assert resolve_lora_target_modules(_model("qwen3_moe"), ["q_proj"]) == ["q_proj"]

    def test_the_qwen35_family_is_unchanged(self):
        for model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
            assert resolve_lora_target_modules(_model(model_type), "auto") == list(
                QWEN35_TEXT_LORA_TARGETS
            )

    def test_qwen4_exp_is_unchanged(self):
        assert resolve_lora_target_modules(_model("qwen4_exp_text"), "auto") == "all-linear"

    def test_an_architecture_peft_maps_is_still_delegated(self):
        """The #1074 lesson one axis over: refuse only where the facts are known.
        peft maps llama, so Soup must not answer for it."""
        pytest.importorskip("peft")

        assert resolve_lora_target_modules(_model("llama"), "auto") is None


def _lora(target_modules="auto"):
    """The LoRA block ``build_lora_config`` reads, as the schema defaults it."""
    return types.SimpleNamespace(
        r=4, alpha=8, dropout=0.0, target_modules=target_modules, use_dora=False,
        use_rslora=False, rank_pattern=None, alpha_pattern=None,
        init_lora_weights=True, variant="lora",
    )


class TestTheRefusal:
    """The refusal lives in ``build_lora_config``, the last step every trainer
    takes, not in the resolver. Refusing in the resolver fired BEFORE a later
    step could supply targets -- ``moe_lora`` replaces ``target_modules`` after
    the resolver at every MoE-wired call site (#1102 review, F1). The resolver
    now hands back a falsy :class:`UnmappedTargets` and lets the end decide."""

    def test_the_resolver_no_longer_raises_it_reports(self):
        from soup_cli.utils.peft_wiring import UnmappedTargets

        resolved = resolve_lora_target_modules(_model("not_a_real_arch_9000"), "auto")

        assert isinstance(resolved, UnmappedTargets)
        assert resolved.model_types == ["not_a_real_arch_9000"]
        assert not resolved, "falsy, like the None it stands in for"

    def test_an_architecture_nobody_maps_is_refused_by_name(self):
        """It raised before this change too -- as peft's ``No target_modules
        passed``, which names neither the architecture nor the fix."""
        pytest.importorskip("peft")
        from soup_cli.utils.peft_wiring import build_lora_config

        resolved = resolve_lora_target_modules(_model("not_a_real_arch_9000"), "auto")
        with pytest.raises(ValueError) as excinfo:
            build_lora_config(_lora(), target_modules=resolved, task_type="CAUSAL_LM")
        message = str(excinfo.value)

        assert "not_a_real_arch_9000" in message and "#1070" in message

    def test_the_message_does_not_send_a_dense_model_to_the_moe_table(self):
        """The refusal reaches dense architectures too (phi3, smollm3, lfm2 ...),
        so naming ``MOE_TEXT_LORA_TARGETS`` told a Phi user to add their model to
        a MoE-only table (#1102 review, F5)."""
        pytest.importorskip("peft")
        from soup_cli.utils.peft_wiring import build_lora_config

        resolved = resolve_lora_target_modules(_model("phi3"), "auto")
        with pytest.raises(ValueError) as excinfo:
            build_lora_config(_lora(), target_modules=resolved, task_type="CAUSAL_LM")

        assert "MOE_TEXT_LORA_TARGETS" not in str(excinfo.value)
        assert "target_modules" in str(excinfo.value)

    def test_target_parameters_suppress_the_refusal(self):
        """peft accepts ``target_parameters`` with no ``target_modules``, so a
        caller supplying them has something to attach and must not be refused.
        ``build_lora_config`` sees them directly -- no call site has to pass a
        flag, which is what left the trainer half unpinned before (F4)."""
        pytest.importorskip("peft")
        from soup_cli.utils.peft_wiring import build_lora_config

        resolved = resolve_lora_target_modules(_model("not_a_real_arch_9000"), "auto")
        config = build_lora_config(
            _lora(), target_modules=resolved, task_type="CAUSAL_LM",
            target_parameters=["model.layers.0.mlp.experts.gate_up_proj"],
        )

        assert config.target_modules is None


class TestMoeLoraStillSuppliesTargets:
    """F1, end to end. ``target_modules: auto`` + ``moe_lora: true`` on a MoE
    architecture neither table maps must attach, exactly as it does on ``main``,
    because ``moe_lora`` supplies the targets after the resolver. Measured on a
    real ``qwen2_moe`` -- the architecture #1070 reproduced on -- through the same
    two calls every MoE-wired trainer makes, in the same order: ``main`` attached
    7 modules; the raise-in-the-resolver version of this PR refused it."""

    @staticmethod
    def _qwen2_moe():
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

        return Qwen2MoeForCausalLM(
            Qwen2MoeConfig(
                vocab_size=64, hidden_size=16, intermediate_size=32,
                moe_intermediate_size=16, shared_expert_intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                num_experts=4, num_experts_per_tok=2,
            )
        )

    def test_the_moe_lora_override_attaches_on_an_unmapped_moe(self):
        pytest.importorskip("peft")
        from peft import get_peft_model

        from soup_cli.utils.moe import resolve_moe_lora_targets
        from soup_cli.utils.peft_wiring import build_lora_config

        model = self._qwen2_moe()
        tcfg = types.SimpleNamespace(moe_lora=True, lora=_lora())
        targets = resolve_lora_target_modules(model, "auto")
        targets = resolve_moe_lora_targets(model, tcfg, targets, None)
        attached = get_peft_model(
            model, build_lora_config(_lora(), target_modules=targets, task_type="CAUSAL_LM")
        )

        assert [n for n, _ in attached.named_modules() if n.endswith(".lora_A")]

    def test_without_moe_lora_the_same_model_is_refused_by_name(self):
        """The control: the refusal still happens when nothing supplies targets,
        so the override above is what saved the attach, not a disabled guard."""
        pytest.importorskip("peft")
        from soup_cli.utils.moe import resolve_moe_lora_targets
        from soup_cli.utils.peft_wiring import build_lora_config

        model = self._qwen2_moe()
        tcfg = types.SimpleNamespace(moe_lora=False, lora=_lora())
        targets = resolve_lora_target_modules(model, "auto")
        targets = resolve_moe_lora_targets(model, tcfg, targets, None)

        with pytest.raises(ValueError, match="qwen2_moe"):
            build_lora_config(_lora(), target_modules=targets, task_type="CAUSAL_LM")


class TestWhatIsDelegated:
    def test_a_model_with_no_declared_model_type_is_delegated(self):
        """The refusal needs a name to put in the message. A config that declares
        no ``model_type`` -- or a test double standing in for a model, which is
        how several suites drive this -- is not an architecture we know to be
        unmappable, so it is delegated exactly as before #1070."""
        pytest.importorskip("peft")
        nameless = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type=None, text_config=None)
        )

        assert resolve_lora_target_modules(nameless, "auto") is None

    def test_a_non_string_model_type_is_delegated(self):
        """`test_embedding.py` and `test_pretrain.py` drive the resolver with
        MagicMock models, whose ``model_type`` is a Mock, not a name."""
        pytest.importorskip("peft")
        from unittest.mock import MagicMock

        assert resolve_lora_target_modules(MagicMock(), "auto") is None

    def test_an_explicit_list_is_never_refused(self):
        assert resolve_lora_target_modules(_model("not_a_real_arch_9000"), ["q_proj"]) == [
            "q_proj"
        ]


# Every base a shipped recipe names, with its ``model_type`` and routed-expert
# count read off its raw ``config.json`` -- written by
# ``scripts/record_recipe_base_architectures.py``, never by hand.
#
# The first version of this ratchet found MoE recipes by their MoE *flags* and
# then patched in a hand-kept list of the flagless ones it knew about: the same
# flag-blindness one level down. A synthetic flagless MoE recipe with a new base
# passed straight through it (#1102 review, F3). A MoE base is now one whose
# config DECLARES experts, and a new base fails the completeness test until it is
# recorded, whatever its recipe sets.
_RECORD_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "recipe_base_architectures.json"
)

#: MoE ``model_type`` values covered before this table existed.
_ALREADY_MAPPED = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}

#: MoE bases deliberately NOT covered, each with the reason. Pinned by count, so a
#: new exclusion is a reviewed edit rather than a way to make the ratchet go green.
EXCLUDED_MOE_BASES = {
    "deepseek-ai/DeepSeek-OCR": (
        "deepseek_vl_v2 needs trust_remote_code to build, so its module paths "
        "cannot be measured without executing Hub code, and this table takes only "
        "measured entries. It matters here, not in principle: q_lora_rank is None, "
        "so its language tower uses plain q_proj -- the same name as its CLIP "
        "vision encoder -- and a guessed suffix list would adapt the image encoder "
        "(MiniMax-M3's failure). The recipe itself is fine: Soup enables remote "
        "code by CLI flag, not by config field, so `soup train --trust-remote-code` "
        "loads it -- and then reaches the named refusal instead of peft's."
    ),
}

#: Bases a MoE-flagged recipe names whose config.json cannot be read. Their
#: recipes keep failing, now with Soup's named refusal instead of peft's.
UNRESOLVABLE_MOE_RECIPE_BASES = {
    "mistralai/Mistral-Large-3-675B-Instruct-2512": (
        "the repo exists and lists consolidated-*.safetensors but has no "
        "config.json (404), so AutoConfig cannot load it at all"
    ),
    "moonshotai/Kimi-K2": (
        "401 unauthenticated, while moonshotai/Kimi-K2.5 and Kimi-K2.6 resolve "
        "from the same org -- gated or gone, not decidable from here"
    ),
}


def _record():
    import json

    return json.loads(_RECORD_PATH.read_text(encoding="utf-8"))


def _catalogue_bases():
    return {yaml.safe_load(r.yaml_str).get("base") for r in list_recipes()}


def _flagged_bases():
    """Bases of recipes that set a MoE knob. No longer a selector -- only a
    cross-check on the record, which is where MoE-ness is decided now."""
    flags = ("moe_lora", "moe_aux_loss_coeff", "train_router_only", "moe_expert_quant")
    bases = set()
    for recipe in list_recipes():
        config = yaml.safe_load(recipe.yaml_str)
        training = config.get("training") or {}
        if any(key in training for key in flags):
            bases.add(config.get("base"))
    return bases


def _moe_bases():
    return {b: v for b, v in _record().items() if v.get("routed_experts", 0) > 1}


class TestTheRatchet:
    """Fails when a recipe arrives whose base is MoE and uncovered -- by what its
    config declares, not by what its recipe happens to set."""

    def test_every_recipe_base_is_recorded(self):
        """Completeness, and the assertion that closes F3: a base missing from the
        record fails here before any MoE question is asked, so a flagless MoE
        recipe cannot pass by being invisible."""
        missing = sorted(_catalogue_bases() - set(_record()))

        assert missing == [], (
            "These recipe bases are not in tests/fixtures/"
            "recipe_base_architectures.json:\n  " + "\n  ".join(missing)
            + "\nRun scripts/record_recipe_base_architectures.py and review the diff."
        )

    def test_the_record_names_no_base_the_catalogue_dropped(self):
        stale = sorted(set(_record()) - _catalogue_bases())

        assert stale == [], f"recorded but no recipe uses them: {stale}"

    def test_every_moe_base_is_covered_or_excluded_with_a_reason(self):
        uncovered = sorted(
            f"{base} ({info['model_type']}, {info['routed_experts']} experts)"
            for base, info in _moe_bases().items()
            if info["model_type"] not in MOE_TEXT_LORA_TARGETS
            and info["model_type"] not in _ALREADY_MAPPED
            and base not in EXCLUDED_MOE_BASES
        )

        assert uncovered == [], (
            "These shipped MoE bases resolve to nothing:\n  " + "\n  ".join(uncovered)
            + "\nAdd each model_type to MOE_TEXT_LORA_TARGETS with module names "
            "MEASURED from the architecture, or to EXCLUDED_MOE_BASES with a reason."
        )

    def test_every_flagged_recipe_base_is_moe_or_unresolvable(self):
        """The flags cross-check the record rather than select from it: a recipe
        that sets a MoE knob on a base the record calls dense means one of them
        is wrong, and a hand edit to the record would show up here."""
        record = _record()
        disagree = sorted(
            base for base in _flagged_bases()
            if record.get(base, {}).get("routed_experts", 0) <= 1
            and "unresolvable" not in record.get(base, {})
        )

        assert disagree == [], f"flagged MoE but recorded dense: {disagree}"

    def test_the_exclusions_and_unresolvables_are_pinned(self):
        """A new exclusion or a newly unreadable MoE base is a deliberate edit.
        Both counts are MoE bases only -- the record holds far more unreadable
        bases overall, almost all gated dense models."""
        record = _record()
        for base, reason in {**EXCLUDED_MOE_BASES, **UNRESOLVABLE_MOE_RECIPE_BASES}.items():
            assert len(reason) > 40, base
        assert set(EXCLUDED_MOE_BASES) <= set(_moe_bases())
        assert all("unresolvable" in record[b] for b in UNRESOLVABLE_MOE_RECIPE_BASES)
        assert set(UNRESOLVABLE_MOE_RECIPE_BASES) <= _flagged_bases()
        assert len(EXCLUDED_MOE_BASES) == 1
        assert len(UNRESOLVABLE_MOE_RECIPE_BASES) == 2

    def test_the_record_has_the_measured_shape(self):
        """If regeneration silently lost bases or experts, the tests above would
        pass while checking less. Pinned to the numbers the reviewer derived
        independently: 117 bases, 20 MoE, 6 of them flagless, 11 model types."""
        moe = _moe_bases()
        flagless = [b for b in moe if b not in _flagged_bases()]

        assert len(_record()) == 117
        assert len(moe) == 20
        assert len(flagless) == 6, sorted(flagless)
        assert len({info["model_type"] for info in moe.values()}) == 11


#: Text SFT/pretrain and the preference/RL trainers; #1148's six are driven in test_issue1099.
_TRAINERS = {
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper", "dpo"),
    "grpo": ("soup_cli.trainer.grpo", "GRPOTrainerWrapper", "alpaca"),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper", "kto"),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper", "dpo"),
    "pretrain": ("soup_cli.trainer.pretrain", "PretrainTrainerWrapper", "plaintext"),
    "sft": ("soup_cli.trainer.sft", "SFTTrainerWrapper", "alpaca"),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper", "dpo"),
}


def _drive_trainer(task, monkeypatch, *, moe_lora):
    """Run a real trainer's LoRA setup on a real ``qwen2_moe`` -- an architecture
    neither Soup's table nor peft maps. Only the loaders are stubbed;
    ``resolve_lora_target_modules``, ``resolve_moe_lora_targets``,
    ``build_lora_config`` and ``get_peft_model`` are the real ones, in the order
    the trainer calls them. That order is the whole of F1."""
    import importlib

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from soup_cli.config.loader import load_config_from_string

    module_path, class_name, data_format = _TRAINERS[task]
    wrapper_cls = getattr(importlib.import_module(module_path), class_name)
    tokenizer = types.SimpleNamespace(pad_token=None, eos_token="</s>", pad_token_id=0)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tokenizer
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained",
        lambda *_a, **_k: TestMoeLoraStillSuppliesTargets._qwen2_moe(),
    )
    cfg = load_config_from_string(
        f"base: org/tiny-qwen2-moe\ntask: {task}\nbackend: transformers\n"
        f"data:\n  train: x.jsonl\n  format: {data_format}\n"
        f"training:\n  quantization: none\n"
        f"  moe_lora: {'true' if moe_lora else 'false'}\n"
        f"  lora:\n    r: 4\n    alpha: 8\n    dropout: 0.0\n    target_modules: auto\n"
    )
    wrapper = object.__new__(wrapper_cls)
    wrapper.config, wrapper.device, wrapper._trust_remote_code = cfg, "cpu", False
    wrapper.model = wrapper.tokenizer = None
    try:
        wrapper._setup_transformers(cfg, cfg.training)
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 -- setup carries on past the attach
        pass
    return wrapper.model


@pytest.mark.parametrize("task", sorted(_TRAINERS))
class TestThroughARealTrainer:
    """The trainer half, pinned by behaviour rather than by a call kwarg: the
    review found nothing drove a trainer, so reverting its call sites was
    invisible (F4). With the refusal in ``build_lora_config`` there is no
    call-site change left to revert -- but the attach is still asserted here,
    through the trainer, because that is where F1 lived."""

    def test_moe_lora_attaches_an_unmapped_moe(self, task, monkeypatch):
        model = _drive_trainer(task, monkeypatch, moe_lora=True)

        adapted = [n for n, _ in model.named_modules() if n.endswith(".lora_A")]
        assert adapted, f"{task}: moe_lora supplied no adapter on an unmapped MoE"

    def test_without_moe_lora_it_is_refused_by_name(self, task, monkeypatch):
        with pytest.raises(ValueError, match="qwen2_moe"):
            _drive_trainer(task, monkeypatch, moe_lora=False)
