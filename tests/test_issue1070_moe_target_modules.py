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
    ``k_proj`` / ``v_proj``. A suffix list would adapt the image encoder during a
    text fine-tune -- the "trains something nobody chose" failure #1070's own
    constraints warn about. peft treats a string target as a regex."""

    # Real module paths, from instantiating MiniMaxAI/MiniMax-M3 on the meta device.
    LANGUAGE = [
        "language_model.layers.0.self_attn.q_proj",
        "language_model.layers.0.self_attn.k_proj",
        "language_model.layers.1.self_attn.v_proj",
        "language_model.layers.1.self_attn.o_proj",
    ]
    NOT_LANGUAGE = [
        "vision_tower.layers.0.self_attn.q_proj",
        "vision_tower.layers.0.self_attn.k_proj",
        "vision_tower.layers.0.self_attn.out_proj",
        "vision_tower.layers.0.mlp.fc1",
        "multi_modal_projector.linear_1",
        "language_model.layers.0.mlp.down_proj",
        "language_model.layers.0.mlp.gate_up_proj",
    ]

    def test_the_regex_matches_every_language_attention_projection(self):
        pattern = resolve_lora_target_modules(_model("minimax_m3_vl"), "auto")

        assert isinstance(pattern, str), "peft reads a string target as a regex"
        for name in self.LANGUAGE:
            assert re.fullmatch(pattern, name), name

    def test_the_regex_matches_nothing_in_the_vision_tower(self):
        pattern = resolve_lora_target_modules(_model("minimax_m3_vl"), "auto")

        for name in self.NOT_LANGUAGE:
            assert not re.fullmatch(pattern, name), name

    def test_the_text_only_config_uses_plain_suffixes(self):
        """Loaded without the wrapper there is no ``language_model.`` prefix and
        no vision tower, so the regex would match nothing at all."""
        resolved = resolve_lora_target_modules(_model("minimax_m3_vl_text"), "auto")

        assert resolved == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_the_wrapper_wins_over_its_own_text_config(self):
        """A VL model reports both types. The wrapper's entry is the one that
        knows about the vision tower, so it must not be decided by set order."""
        pattern = resolve_lora_target_modules(
            _model("minimax_m3_vl", text_type="minimax_m3_vl_text"), "auto"
        )

        assert isinstance(pattern, str)
        assert not re.fullmatch(pattern, self.NOT_LANGUAGE[0])

    def test_the_wrapper_wins_even_when_the_text_type_sorts_first(self):
        """The real pair (``minimax_m3_vl`` / ``minimax_m3_vl_text``) happens to
        sort wrapper-first, so iterating the types alphabetically gets the right
        answer by accident. Kimi's pair is ``kimi_k25`` / ``kimi_k2``, which sorts
        the other way; this pins the rule rather than the alphabet."""
        resolved = resolve_lora_target_modules(
            _model("minimax_m3_vl", text_type="deepseek_v3"), "auto"
        )

        assert isinstance(resolved, str), (
            f"took the text_config entry: {resolved!r}"
        )


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


class TestTheRefusal:
    def test_an_architecture_nobody_maps_is_refused_by_name(self):
        """It raised before this change too -- as peft's ``No target_modules
        passed``, which names neither the architecture nor where to fix it."""
        pytest.importorskip("peft")

        with pytest.raises(ValueError) as excinfo:
            resolve_lora_target_modules(_model("not_a_real_arch_9000"), "auto")
        message = str(excinfo.value)

        assert "not_a_real_arch_9000" in message
        assert "MOE_TEXT_LORA_TARGETS" in message and "#1070" in message

    def test_target_parameters_suppress_the_refusal(self):
        """peft accepts ``target_parameters`` with no ``target_modules``, so a
        caller supplying them has something to attach and must not be refused --
        this is the over-refusal the guard exists to avoid."""
        pytest.importorskip("peft")

        assert (
            resolve_lora_target_modules(
                _model("not_a_real_arch_9000"), "auto", has_target_parameters=True
            )
            is None
        )

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


# The ``model_type`` each shipped MoE recipe's base reports, read from its
# config.json on the Hub. Checked in because the ratchet below has to run
# offline and in the required matrix; the network check that this stays true is
# the probe in the PR, re-run when a base is added.
MOE_RECIPE_BASE_MODEL_TYPES = {
    "MiniMaxAI/MiniMax-M3": "minimax_m3_vl",
    "Qwen/Qwen3-30B-A3B": "qwen3_moe",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "qwen3_moe",
    "Qwen/Qwen3.5-122B-A10B": "qwen3_5_moe",
    "Qwen/Qwen3.5-35B-A3B": "qwen3_5_moe",
    "Qwen/Qwen3.5-397B-A17B": "qwen3_5_moe",
    "Qwen/Qwen3.6-35B-A3B": "qwen3_5_moe",
    "deepseek-ai/DeepSeek-V3": "deepseek_v3",
    "deepseek-ai/DeepSeek-V3-0324": "deepseek_v3",
    "deepseek-ai/DeepSeek-V4-Flash": "deepseek_v4",
    "deepseek-ai/DeepSeek-V4-Pro": "deepseek_v4",
    "moonshotai/Kimi-K2.5": "kimi_k25",
    "moonshotai/Kimi-K2.6": "kimi_k25",
    "zai-org/GLM-5.1": "glm_moe_dsa",
}

#: Bases whose ``model_type`` cannot be read, so the table cannot cover them and
#: their recipes keep raising. Both were checked against the Hub, unauthenticated,
#: with resolvable siblings from the same org as the control.
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

_ALREADY_MAPPED = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}


def _moe_recipe_bases():
    """Every shipped recipe that sets a MoE knob, by base."""
    bases = {}
    for recipe in list_recipes():
        config = yaml.safe_load(recipe.yaml_str)
        training = config.get("training") or {}
        if any(
            key in training
            for key in (
                "moe_lora",
                "moe_aux_loss_coeff",
                "train_router_only",
                "moe_expert_quant",
            )
        ):
            bases.setdefault(config.get("base"), []).append(recipe.model)
    return bases


class TestTheRatchet:
    """Fails when MoE recipe 32 arrives with a ``model_type`` the table does not
    cover -- the shape the maintainer asked for, one level up from #798's dropout
    guard and the same idea as #1033's recipe-count ratchet."""

    def test_every_moe_recipe_base_has_a_recorded_model_type(self):
        known = set(MOE_RECIPE_BASE_MODEL_TYPES) | set(UNRESOLVABLE_MOE_RECIPE_BASES)
        unrecorded = sorted(set(_moe_recipe_bases()) - known)

        assert unrecorded == [], (
            "A MoE recipe names a base whose model_type is not recorded:\n  "
            + "\n  ".join(unrecorded)
            + "\nRead its config.json from the Hub and add it to "
            "MOE_RECIPE_BASE_MODEL_TYPES, then cover the model_type in "
            "MOE_TEXT_LORA_TARGETS."
        )

    def test_every_recorded_model_type_is_covered(self):
        uncovered = sorted(
            {
                model_type
                for model_type in MOE_RECIPE_BASE_MODEL_TYPES.values()
                if model_type not in MOE_TEXT_LORA_TARGETS
                and model_type not in _ALREADY_MAPPED
            }
        )

        assert uncovered == [], (
            "These MoE model_types are shipped in a recipe and resolve to nothing:\n  "
            + "\n  ".join(uncovered)
            + "\nAdd each to MOE_TEXT_LORA_TARGETS in utils/peft_wiring.py, with "
            "the module names measured from the architecture, not guessed."
        )

    def test_the_unresolvable_set_does_not_grow_silently(self):
        """A base nobody can resolve is a recipe that cannot train. Two is the
        number today, each with a reason; a third is a deliberate edit here."""
        for base, reason in UNRESOLVABLE_MOE_RECIPE_BASES.items():
            assert len(reason) > 20, base
        assert set(UNRESOLVABLE_MOE_RECIPE_BASES) <= set(_moe_recipe_bases()), (
            "an entry names a base no recipe uses any more; remove it"
        )
        assert len(UNRESOLVABLE_MOE_RECIPE_BASES) == 2

    def test_the_recipe_sweep_finds_the_expected_shape(self):
        """Sanity for the two tests above: if the sweep silently found nothing,
        they would both pass while checking nothing at all."""
        bases = _moe_recipe_bases()

        assert len(bases) == 16, sorted(bases)
        assert sum(len(models) for models in bases.values()) == 31
