"""v0.71.24 — 2026 model-family recipe expansion (catalog 116 -> 133).

Pure config/data release: 17 new ready-made SFT recipes for the open-weight
models released Feb-Jun 2026 (Qwen3.5 / Qwen3.6 / DeepSeek-V4 / GLM-5.1 /
Kimi-K2.5/K2.6 / MiniMax-M3 / Mistral-Large-3), plus a stale-repo-ID fix for
``glm-5-sft`` (``THUDM/glm-5`` -> ``zai-org/GLM-5``). CI validates each recipe by
``load_config_from_string`` parse only (no network).
"""
from __future__ import annotations

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from soup_cli.recipes.catalog import RECIPES, get_recipe, list_recipes, search_recipes

# ---------------------------------------------------------------------------
# Step A — 17 new SFT recipes (every base verified to resolve on Hugging Face)
# ---------------------------------------------------------------------------

V07124_RECIPE_NAMES = [
    # Qwen3.5 family (Apache-2.0)
    "qwen3.5-0.8b-sft",
    "qwen3.5-2b-sft",
    "qwen3.5-4b-sft",
    "qwen3.5-9b-sft",
    "qwen3.5-27b-sft",
    "qwen3.5-35b-a3b-sft",
    "qwen3.5-122b-a10b-sft",
    "qwen3.5-397b-a17b-sft",
    # Qwen3.6 (Apache-2.0)
    "qwen3.6-27b-sft",
    "qwen3.6-35b-a3b-sft",
    # DeepSeek-V4 (MIT)
    "deepseek-v4-flash-sft",
    "deepseek-v4-pro-sft",
    # GLM (MIT, zai-org)
    "glm-5.1-sft",
    # Kimi (Modified MIT)
    "kimi-k2.5-sft",
    "kimi-k2.6-sft",
    # MiniMax (MiniMax Community License)
    "minimax-m3-sft",
    # Mistral Large 3 (Apache-2.0)
    "mistral-large-3-sft",
]

# name -> expected base repo-ID (the verified Hugging Face repo)
V07124_RECIPE_BASES = {
    "qwen3.5-0.8b-sft": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b-sft": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b-sft": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b-sft": "Qwen/Qwen3.5-9B",
    "qwen3.5-27b-sft": "Qwen/Qwen3.5-27B",
    "qwen3.5-35b-a3b-sft": "Qwen/Qwen3.5-35B-A3B",
    "qwen3.5-122b-a10b-sft": "Qwen/Qwen3.5-122B-A10B",
    "qwen3.5-397b-a17b-sft": "Qwen/Qwen3.5-397B-A17B",
    "qwen3.6-27b-sft": "Qwen/Qwen3.6-27B",
    "qwen3.6-35b-a3b-sft": "Qwen/Qwen3.6-35B-A3B",
    "deepseek-v4-flash-sft": "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-v4-pro-sft": "deepseek-ai/DeepSeek-V4-Pro",
    "glm-5.1-sft": "zai-org/GLM-5.1",
    "kimi-k2.5-sft": "moonshotai/Kimi-K2.5",
    "kimi-k2.6-sft": "moonshotai/Kimi-K2.6",
    "minimax-m3-sft": "MiniMaxAI/MiniMax-M3",
    "mistral-large-3-sft": "mistralai/Mistral-Large-3-675B-Instruct-2512",
}

# The MoE bases added in v0.71.24 — each recipe must enable moe_lora + aux loss.
# (Scope is intentionally limited to recipes added in THIS release, not every MoE
# recipe in the catalog.)
V07124_MOE_RECIPES = [
    "qwen3.5-35b-a3b-sft",
    "qwen3.5-122b-a10b-sft",
    "qwen3.5-397b-a17b-sft",
    "qwen3.6-35b-a3b-sft",
    "deepseek-v4-flash-sft",
    "deepseek-v4-pro-sft",
    "glm-5.1-sft",
    "kimi-k2.5-sft",
    "kimi-k2.6-sft",
    "minimax-m3-sft",
    "mistral-large-3-sft",
]


class TestV07124Recipes:
    def test_recipe_count_target(self) -> None:
        assert len(V07124_RECIPE_NAMES) == 17

    @pytest.mark.parametrize("name", V07124_RECIPE_NAMES)
    def test_recipe_registered(self, name: str) -> None:
        assert name in RECIPES, f"Recipe {name!r} missing from catalog"
        assert get_recipe(name) is not None

    @pytest.mark.parametrize("name", V07124_RECIPE_NAMES)
    def test_recipe_base_matches_verified_repo(self, name: str) -> None:
        assert RECIPES[name].model == V07124_RECIPE_BASES[name]

    @pytest.mark.parametrize("name", V07124_RECIPE_NAMES)
    def test_recipe_is_sft(self, name: str) -> None:
        assert RECIPES[name].task == "sft"

    @pytest.mark.parametrize("name", V07124_RECIPE_NAMES)
    def test_recipe_metadata_well_formed(self, name: str) -> None:
        meta = RECIPES[name]
        assert meta.model, f"{name}: model is empty"
        assert meta.size, f"{name}: size is empty"
        assert meta.tags, f"{name}: tags is empty"
        assert meta.description, f"{name}: description is empty"
        assert meta.yaml_str, f"{name}: yaml_str is empty"

    @pytest.mark.parametrize("name", V07124_RECIPE_NAMES)
    def test_recipe_yaml_parses_as_soup_config(self, name: str) -> None:
        cfg = load_config_from_string(RECIPES[name].yaml_str)
        assert isinstance(cfg, SoupConfig)
        assert cfg.base == RECIPES[name].model
        assert cfg.task == "sft"

    @pytest.mark.parametrize("name", V07124_RECIPE_NAMES)
    def test_recipe_yaml_is_safe_loadable(self, name: str) -> None:
        parsed = yaml.safe_load(RECIPES[name].yaml_str)
        assert isinstance(parsed, dict)
        assert "base" in parsed and "task" in parsed
        # The inline ``base:`` / ``task:`` in the YAML must match the RecipeMeta.
        assert parsed["base"] == RECIPES[name].model
        assert parsed["task"] == RECIPES[name].task

    @pytest.mark.parametrize("name", V07124_RECIPE_NAMES)
    def test_recipe_model_id_no_null_or_whitespace(self, name: str) -> None:
        meta = RECIPES[name]
        assert "\x00" not in meta.model
        assert " " not in meta.model
        # No path-traversal segments or shell metacharacters in a repo id.
        assert ".." not in meta.model, f"path-traversal segment in {meta.model!r}"
        assert not any(c in meta.model for c in ';|&$`<>(){}*?!\\\n\t'), (
            f"shell metacharacter in {meta.model!r}"
        )
        parts = meta.model.split("/")
        assert len(parts) == 2
        assert all(p for p in parts), f"empty component in {meta.model!r}"

    @pytest.mark.parametrize("name", V07124_RECIPE_NAMES)
    def test_recipe_max_length_within_bounds(self, name: str) -> None:
        cfg = load_config_from_string(RECIPES[name].yaml_str)
        assert 64 <= cfg.data.max_length <= 1_048_576

    @pytest.mark.parametrize("name", V07124_MOE_RECIPES)
    def test_moe_recipes_enable_moe_lora(self, name: str) -> None:
        cfg = load_config_from_string(RECIPES[name].yaml_str)
        assert cfg.training.moe_lora is True, f"{name} should set moe_lora: true"

    @pytest.mark.parametrize("name", V07124_MOE_RECIPES)
    def test_moe_recipes_set_aux_loss_coeff(self, name: str) -> None:
        # Every new MoE recipe sets the aux-loss coefficient explicitly (0.01) so
        # the whole batch is uniformly self-documenting (review HIGH-1).
        cfg = load_config_from_string(RECIPES[name].yaml_str)
        assert abs(cfg.training.moe_aux_loss_coeff - 0.01) < 1e-9, (
            f"{name} should set moe_aux_loss_coeff: 0.01, got "
            f"{cfg.training.moe_aux_loss_coeff}"
        )

    @pytest.mark.parametrize("name", sorted(set(V07124_RECIPE_NAMES) - set(V07124_MOE_RECIPES)))
    def test_dense_recipes_do_not_set_moe_lora(self, name: str) -> None:
        # The complement of the MoE list: a dense recipe must NOT enable moe_lora
        # (guards against pasting a MoE block into a dense recipe).
        cfg = load_config_from_string(RECIPES[name].yaml_str)
        assert cfg.training.moe_lora is not True, (
            f"{name} is a dense recipe but has moe_lora: true"
        )

    @pytest.mark.parametrize("name", V07124_MOE_RECIPES)
    def test_moe_recipes_tagged_moe(self, name: str) -> None:
        assert "moe" in RECIPES[name].tags, (
            f"{name} is a MoE recipe but lacks the 'moe' tag"
        )

    @pytest.mark.parametrize("name", [n for n in V07124_RECIPE_NAMES if n.startswith("qwen")])
    def test_qwen_recipes_tagged_qwen(self, name: str) -> None:
        assert "qwen" in RECIPES[name].tags, f"{name} lacks the 'qwen' tag"

    def test_search_by_task_sft_includes_all_new_recipes(self) -> None:
        sft_models = {r.model for r in search_recipes(task="sft")}
        for name in V07124_RECIPE_NAMES:
            assert RECIPES[name].model in sft_models, (
                f"{name} not returned by search_recipes(task='sft')"
            )

    def test_kimi_k2_5_size_known(self) -> None:
        # K2.5 is a ~1T/32B MoE (verified on the model card) — not "N/A".
        assert RECIPES["kimi-k2.5-sft"].size == "1T"

    def test_kimi_k2_6_size_known(self) -> None:
        # K2.6 is a ~1T/32B MoE (verified on the model card).
        assert RECIPES["kimi-k2.6-sft"].size == "1T"

    def test_minimax_recipe_notes_community_license(self) -> None:
        desc = RECIPES["minimax-m3-sft"].description.lower()
        assert "minimax" in desc
        assert "license" in desc
        # commercial-use caveat must be surfaced
        assert "commercial" in desc

    def test_mistral_large_3_notes_apache_license(self) -> None:
        # Plan guessed MRL; the published card is Apache-2.0 — describe accurately.
        desc = RECIPES["mistral-large-3-sft"].description.lower()
        assert "apache" in desc

    @pytest.mark.parametrize("name", ["kimi-k2.5-sft", "kimi-k2.6-sft"])
    def test_kimi_recipes_note_modified_mit(self, name: str) -> None:
        assert "mit" in RECIPES[name].description.lower()

    def test_deepseek_v4_sizes_intentionally_na(self) -> None:
        # DeepSeek has not publicly confirmed the V4 parameter counts; "N/A" is the
        # honest value. This documents the deliberate choice (vs Kimi K2.5/K2.6,
        # whose ~1T sizes ARE on the card).
        assert RECIPES["deepseek-v4-flash-sft"].size == "N/A"
        assert RECIPES["deepseek-v4-pro-sft"].size == "N/A"

    @pytest.mark.parametrize("name", V07124_RECIPE_NAMES)
    def test_recipe_searchable(self, name: str) -> None:
        # The recipe key is part of the searchable text, so a query for the recipe's
        # own name must return that exact recipe (deterministic — no token heuristic).
        results = search_recipes(query=name)
        assert any(r.model == RECIPES[name].model for r in results), (
            f"{name} not returned by search_recipes(query={name!r})"
        )


# ---------------------------------------------------------------------------
# Step B — stale repo-ID fix: glm-5-sft (THUDM/glm-5 -> zai-org/GLM-5)
# ---------------------------------------------------------------------------


class TestGlm5RepoIdFix:
    def test_glm5_uses_zai_org(self) -> None:
        meta = RECIPES["glm-5-sft"]
        assert meta.model == "zai-org/GLM-5"
        assert "THUDM" not in meta.model

    def test_glm5_yaml_base_updated(self) -> None:
        meta = RECIPES["glm-5-sft"]
        parsed = yaml.safe_load(meta.yaml_str)
        assert parsed["base"] == "zai-org/GLM-5"
        assert "THUDM/glm-5" not in meta.yaml_str

    def test_glm5_old_key_still_exists(self) -> None:
        # The fix RENAMES the base field; it must NOT delete the old recipe key.
        assert "glm-5-sft" in RECIPES, "glm-5-sft was accidentally removed"
        assert "glm-5.1-sft" in RECIPES, "glm-5.1-sft is the new v0.71.24 recipe"

    def test_glm5_and_glm51_are_distinct_entries(self) -> None:
        assert RECIPES["glm-5-sft"].model != RECIPES["glm-5.1-sft"].model

    def test_no_recipe_references_thudm_glm5(self) -> None:
        for name, meta in RECIPES.items():
            assert "THUDM/glm-5" not in meta.model, (
                f"{name}: model still references THUDM/glm-5"
            )
            assert "THUDM/glm-5" not in meta.yaml_str, (
                f"{name}: yaml_str still references THUDM/glm-5"
            )


# ---------------------------------------------------------------------------
# Catalog count — 116 -> 133
# ---------------------------------------------------------------------------


class TestCatalogCount:
    def test_total_recipe_count_is_133(self) -> None:
        assert len(RECIPES) == 133

    def test_list_recipes_matches_dict(self) -> None:
        assert len(list_recipes()) == len(RECIPES)

    def test_all_recipes_parse(self) -> None:
        # Defence-in-depth — the whole catalog must still parse after the additions.
        for name, meta in RECIPES.items():
            cfg = load_config_from_string(meta.yaml_str)
            assert cfg.base, name

    def test_all_recipes_yaml_base_matches_model(self) -> None:
        # Catalog-wide integrity invariant — the inline ``base:`` must equal
        # RecipeMeta.model for EVERY recipe (catches a half-applied repo-ID fix).
        for name, meta in RECIPES.items():
            parsed = yaml.safe_load(meta.yaml_str)
            assert parsed.get("base") == meta.model, (
                f"{name}: yaml base {parsed.get('base')!r} != model {meta.model!r}"
            )
