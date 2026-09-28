"""#1395: mistral-medium-3-5-sft cannot load as written.

The recipe base ``mistralai/Mistral-Medium-3.5-128B`` has Hugging Face config.json
with ``model_type: mistral3``, which is a vision-language wrapper containing a
Pixtral vision tower and a Mistral language model.

Without ``modality: vision``, SFT loads ``AutoModelForCausalLM``, which refuses
``Mistral3Config`` with:
    ValueError: Unrecognized configuration class ... Mistral3Config for this kind of
    AutoModel: AutoModelForCausalLM

With ``modality: vision``, SFT loads ``AutoModelForImageTextToText``, but
``target_modules: auto`` fails because peft has no mapping for ``mistral3``,
raising:
    ValueError: target_modules='auto' has no mapping for model_type=['ministral3', 'mistral3']

Furthermore, a plain suffix list would match ``q_proj`` / ``k_proj`` / ``v_proj`` /
``o_proj`` in the Pixtral vision tower
(``model.vision_tower.transformer.layers.<N>.attention.(q|k|v|o)_proj``), silently
adapting the vision encoder for a text fine-tune.

This test suite validates:
1. ``MOE_TEXT_LORA_TARGETS`` contains ``mistral3`` mapped to the language-scoped regex.
2. In-memory stand-in model on the meta device (zero network calls) discriminates
   module names, matching all 8 language attention projections and 0 vision modules.
3. Mutation-killing tests: dropping the leading ``.*``, using a plain suffix list,
   unscoped attention regex, or dropping individual projections all fail decisively.
4. Real ``get_peft_model`` attach on the meta device adapts 8 language modules and
   0 vision modules.
5. Recipe invariants and snapshot consistency.
6. Preflight attach check: reports ``ATTACHES`` with 0 vision modules; controls prove
   that omitting ``modality: vision`` or omitting the table entry fails preflight.
"""

from __future__ import annotations

import re
import types
from unittest.mock import patch

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.recipes.catalog import RECIPES, get_recipe
from soup_cli.utils.attach_preflight import Verdict, build_on_meta, check_attach
from soup_cli.utils.peft_wiring import (
    MOE_TEXT_LORA_TARGETS,
    resolve_lora_target_modules,
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

_EXPECTED_REGEX = r".*language_model\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _model(model_type: str, text_type: str | None = None) -> types.SimpleNamespace:
    """Mock model object exposing config with model_type and optional text_config."""
    text_config = types.SimpleNamespace(model_type=text_type) if text_type else None
    return types.SimpleNamespace(
        config=types.SimpleNamespace(model_type=model_type, text_config=text_config)
    )


def _standin_mistral3_config(num_layers: int = 2) -> transformers.Mistral3Config:
    """Create an offline in-memory Mistral3Config with small dimensions."""
    from transformers import Mistral3Config, MistralConfig, PixtralVisionConfig

    return Mistral3Config(
        text_config=MistralConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=num_layers,
            num_attention_heads=2,
            num_key_value_heads=2,
            pad_token_id=0,
        ),
        vision_config=PixtralVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=num_layers,
            num_attention_heads=2,
            image_size=32,
            patch_size=16,
        ),
    )


def _standin_mistral3_model(num_layers: int = 2):
    """Instantiate a tiny Mistral3 model on the meta device."""
    from transformers import AutoModelForImageTextToText

    cfg = _standin_mistral3_config(num_layers=num_layers)
    with torch.device("meta"):
        return AutoModelForImageTextToText.from_config(cfg)


def _linear_keys(model) -> list[str]:
    """Return names of all nn.Linear modules."""
    return [name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)]


class TestMistral3TableResolution:
    """Pin the entry in MOE_TEXT_LORA_TARGETS and its resolution logic."""

    def test_mistral3_is_in_moe_text_lora_targets(self) -> None:
        assert "mistral3" in MOE_TEXT_LORA_TARGETS

    def test_mistral3_targets_pinned_literally(self) -> None:
        """Exact string match: must be the language-scoped regex."""
        assert MOE_TEXT_LORA_TARGETS["mistral3"] == _EXPECTED_REGEX

    def test_mistral3_does_not_name_expert_parameters(self) -> None:
        """Target modules must never name routed expert parameters."""
        assert "expert" not in MOE_TEXT_LORA_TARGETS["mistral3"]

    def test_resolves_to_regex_string(self) -> None:
        resolved = resolve_lora_target_modules(_model("mistral3"), "auto")
        assert resolved == _EXPECTED_REGEX
        assert isinstance(resolved, str)

    def test_wrapper_wins_over_inner_text_config(self) -> None:
        """When outer is mistral3 and inner is mistral, outer wrapper entry wins."""
        resolved = resolve_lora_target_modules(
            _model("mistral3", text_type="mistral"), "auto"
        )
        assert resolved == _EXPECTED_REGEX

    def test_unseen_wrapper_resolves_from_text_config(self) -> None:
        """An unseen wrapper with mistral3 text_config resolves via mistral3."""
        resolved = resolve_lora_target_modules(
            _model("unseen_custom_wrapper", text_type="mistral3"), "auto"
        )
        assert resolved == _EXPECTED_REGEX

    def test_explicit_target_modules_list_passed_through(self) -> None:
        """Explicit targets are returned unchanged without touching the table."""
        assert resolve_lora_target_modules(_model("mistral3"), ["q_proj"]) == ["q_proj"]


@pytest.fixture(scope="module")
def mistral3_model_and_keys():
    """Module-scoped model and linear keys to avoid recreating the meta model."""
    model = _standin_mistral3_model(num_layers=2)
    keys = _linear_keys(model)
    return model, keys


class TestMistral3TargetDiscrimination:
    """Validate regex target discrimination against the real module hierarchy."""

    def test_module_structure_has_expected_prefixes(self, mistral3_model_and_keys) -> None:
        _, keys = mistral3_model_and_keys
        lang_attn = [k for k in keys if "language_model" in k and "self_attn" in k]
        vision_attn = [k for k in keys if "vision_tower" in k and "attention" in k]
        projector = [k for k in keys if "multi_modal_projector" in k]
        mlp = [k for k in keys if "language_model" in k and ".mlp." in k]

        assert len(lang_attn) == 8, f"Expected 8 language attention linears, got {len(lang_attn)}"
        assert all(k.startswith("model.language_model.layers.") for k in lang_attn)
        assert len(vision_attn) == 8, f"Expected 8 vision attention linears, got {len(vision_attn)}"
        assert all(k.startswith("model.vision_tower.transformer.layers.") for k in vision_attn)
        assert projector, "Expected multi_modal_projector linear layers"
        assert mlp, "Expected language MLP linear layers"

    def test_regex_matches_every_language_attention_projection(
        self, mistral3_model_and_keys
    ) -> None:
        _, keys = mistral3_model_and_keys
        pattern = MOE_TEXT_LORA_TARGETS["mistral3"]
        lang_attn = [
            k for k in keys
            if "language_model" in k and re.search(r"self_attn\.(q|k|v|o)_proj$", k)
        ]
        matched = [k for k in keys if re.fullmatch(pattern, k)]

        assert len(lang_attn) == 8
        assert sorted(matched) == sorted(lang_attn)

    @pytest.mark.parametrize("proj", _PROJECTIONS)
    def test_regex_matches_each_projection_name(
        self, mistral3_model_and_keys, proj: str
    ) -> None:
        """Pin completeness: every one of q_proj, k_proj, v_proj, o_proj is matched."""
        _, keys = mistral3_model_and_keys
        pattern = MOE_TEXT_LORA_TARGETS["mistral3"]
        matched_proj = [
            k for k in keys
            if re.fullmatch(pattern, k) and k.endswith(f".{proj}")
        ]
        assert len(matched_proj) == 2, f"Expected 2 layers for {proj}, got {matched_proj}"

    def test_regex_matches_nothing_in_vision_tower(self, mistral3_model_and_keys) -> None:
        _, keys = mistral3_model_and_keys
        pattern = MOE_TEXT_LORA_TARGETS["mistral3"]
        vision_keys = [k for k in keys if "vision_tower" in k]
        matched_vision = [k for k in vision_keys if re.fullmatch(pattern, k)]
        assert vision_keys, "Sanity: vision keys exist"
        assert matched_vision == [], f"Vision tower keys matched: {matched_vision}"

    def test_regex_matches_nothing_in_multimodal_projector(
        self, mistral3_model_and_keys
    ) -> None:
        _, keys = mistral3_model_and_keys
        pattern = MOE_TEXT_LORA_TARGETS["mistral3"]
        proj_keys = [k for k in keys if "multi_modal_projector" in k]
        matched_proj = [k for k in proj_keys if re.fullmatch(pattern, k)]
        assert proj_keys, "Sanity: projector keys exist"
        assert matched_proj == [], f"Projector keys matched: {matched_proj}"

    def test_regex_matches_no_mlp_projections(self, mistral3_model_and_keys) -> None:
        _, keys = mistral3_model_and_keys
        pattern = MOE_TEXT_LORA_TARGETS["mistral3"]
        mlp_keys = [k for k in keys if ".mlp." in k]
        matched_mlp = [k for k in mlp_keys if re.fullmatch(pattern, k)]
        assert mlp_keys, "Sanity: MLP keys exist"
        assert matched_mlp == [], f"MLP keys matched: {matched_mlp}"

    def test_regex_matches_nothing_on_lm_head(self, mistral3_model_and_keys) -> None:
        _, keys = mistral3_model_and_keys
        pattern = MOE_TEXT_LORA_TARGETS["mistral3"]
        head_keys = [k for k in keys if "lm_head" in k]
        matched_head = [k for k in head_keys if re.fullmatch(pattern, k)]
        assert head_keys, "Sanity: lm_head exists"
        assert matched_head == []


class TestMutationKilling:
    """Can-it-fail tests: deliberate mutations to the regex or mapping must fail."""

    def test_mutation_dropping_leading_wildcard_matches_zero_keys(
        self, mistral3_model_and_keys
    ) -> None:
        """Mutating `.*language_model` -> `language_model` matches 0 keys due to `model.` prefix.

        This reproduces the exact failure mode from #1102 review (F2).
        """
        _, keys = mistral3_model_and_keys
        mutated_pattern = r"language_model\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        matched = [k for k in keys if re.fullmatch(mutated_pattern, k)]
        assert matched == [], "Dropping .* must fail to match any keys"

    def test_mutation_unscoped_suffix_tuple_hits_vision_tower(
        self, mistral3_model_and_keys
    ) -> None:
        """If target_modules were plain suffixes ('q_proj', ...), vision tower would be hit."""
        _, keys = mistral3_model_and_keys
        vision_hits = [
            k for k in keys
            if "vision_tower" in k and any(k.endswith(f".{p}") for p in _PROJECTIONS)
        ]
        assert len(vision_hits) == 8, (
            "Plain suffix matching would adapt 8 vision tower linears"
        )

    def test_mutation_unscoped_attention_regex_hits_vision_tower(
        self, mistral3_model_and_keys
    ) -> None:
        """Mutating pattern to match any attention projection hits vision tower."""
        _, keys = mistral3_model_and_keys
        mutated_pattern = r".*attention\.(q_proj|k_proj|v_proj|o_proj)"
        vision_matched = [
            k for k in keys
            if "vision_tower" in k and re.fullmatch(mutated_pattern, k)
        ]
        assert len(vision_matched) == 8

    @pytest.mark.parametrize("omitted_proj", _PROJECTIONS)
    def test_mutation_omitting_any_projection_fails_completeness(
        self, mistral3_model_and_keys, omitted_proj: str
    ) -> None:
        """Dropping any single projection reduces match count from 8 to 6."""
        _, keys = mistral3_model_and_keys
        surviving = [p for p in _PROJECTIONS if p != omitted_proj]
        mutated_pattern = rf".*language_model\..*\.self_attn\.({'|'.join(surviving)})"
        matched = [k for k in keys if re.fullmatch(mutated_pattern, k)]
        assert len(matched) == 6


class TestRealPeftAttachOnMetaDevice:
    """Verify peft get_peft_model on the meta-device model adapts only language modules."""

    def test_meta_device_peft_attach(self) -> None:
        from peft import LoraConfig, get_peft_model

        model = _standin_mistral3_model(num_layers=2)
        pattern = MOE_TEXT_LORA_TARGETS["mistral3"]
        attached = get_peft_model(
            model, LoraConfig(r=4, lora_alpha=8, target_modules=pattern)
        )
        adapted = [name for name, _ in attached.named_modules() if name.endswith(".lora_A")]

        assert len(adapted) == 8, f"Expected 8 adapted modules, got {len(adapted)}"
        assert all("language_model" in n for n in adapted), (
            f"Non-language module adapted: {[n for n in adapted if 'language_model' not in n]}"
        )
        assert not any("vision" in n or "visual" in n for n in adapted), (
            f"Vision module adapted: {[n for n in adapted if 'vision' in n or 'visual' in n]}"
        )
        assert not any("projector" in n for n in adapted)
        assert not any(".mlp." in n for n in adapted)


class TestMistralMediumRecipe:
    """Two-surface pin and invariant checks for mistral-medium-3-5-sft."""

    def test_recipe_metadata_and_yaml_invariants(self) -> None:
        assert "mistral-medium-3-5-sft" in RECIPES
        recipe = get_recipe("mistral-medium-3-5-sft")
        assert recipe is not None
        assert recipe.model == "mistralai/Mistral-Medium-3.5-128B"
        assert recipe.task == "sft"

        parsed = yaml.safe_load(recipe.yaml_str)
        assert parsed["base"] == recipe.model
        assert parsed["task"] == recipe.task
        assert parsed.get("modality") == "vision", "Recipe must declare modality: vision"
        assert parsed.get("training", {}).get("lora", {}).get("target_modules") == "auto"

        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.modality == "vision"
        assert cfg.base == "mistralai/Mistral-Medium-3.5-128B"
        assert cfg.task == "sft"

    def test_recipe_snapshot_contains_modality_vision(self) -> None:
        import json
        from pathlib import Path

        snapshot_path = (
            Path(__file__).resolve().parent / "fixtures" / "recipe_config_snapshots.json"
        )
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        recipe_delta = data["recipes"].get("mistral-medium-3-5-sft", {})
        assert recipe_delta.get("modality") == "vision"


class TestPreflightAttach:
    """End-to-end check_attach verification with controls."""

    def test_preflight_verifies_attaches(self) -> None:
        recipe = get_recipe("mistral-medium-3-5-sft")
        cfg = load_config_from_string(recipe.yaml_str)
        standin_config = _standin_mistral3_config(num_layers=2)

        check = check_attach(
            "mistral-medium-3-5-sft",
            cfg,
            load_hf_config=lambda _base: standin_config,
            build_model=build_on_meta,
        )

        assert check.verdict == Verdict.ATTACHES
        assert check.adapted == 8
        assert check.vision_modules == 0
        assert check.expert_modules == 0
        assert check.model_type == "mistral3"

    def test_preflight_control_without_modality_fails_at_build(self) -> None:
        """Regression control: without modality: vision, causal LM class rejects Mistral3Config."""
        recipe = get_recipe("mistral-medium-3-5-sft")
        # Strip modality line
        yaml_without_modality = "\n".join(
            line for line in recipe.yaml_str.splitlines() if not line.startswith("modality:")
        )
        cfg = load_config_from_string(yaml_without_modality)
        standin_config = _standin_mistral3_config(num_layers=2)

        check = check_attach(
            "mistral-medium-3-5-sft-no-modality",
            cfg,
            load_hf_config=lambda _base: standin_config,
            build_model=build_on_meta,
        )

        assert check.verdict == Verdict.CANNOT_ATTACH
        assert check.stage == "build"
        assert "Mistral3Config" in check.detail

    def test_preflight_control_without_table_entry_fails_at_attach(self) -> None:
        """Regression control: without mistral3 in MOE_TEXT_LORA_TARGETS, peft refuses attach."""
        recipe = get_recipe("mistral-medium-3-5-sft")
        cfg = load_config_from_string(recipe.yaml_str)
        standin_config = _standin_mistral3_config(num_layers=2)

        stripped_table = {k: v for k, v in MOE_TEXT_LORA_TARGETS.items() if k != "mistral3"}
        with patch("soup_cli.utils.peft_wiring.MOE_TEXT_LORA_TARGETS", stripped_table):
            check = check_attach(
                "mistral-medium-3-5-sft-unmapped",
                cfg,
                load_hf_config=lambda _base: standin_config,
                build_model=build_on_meta,
            )

        assert check.verdict == Verdict.CANNOT_ATTACH
        assert check.stage == "attach"
        assert "target_modules" in check.detail or "target_parameters" in check.detail

