"""Tests for v0.31.0 — Model & Recipe Breadth.

Adds 34 new recipes across vision, audio, reasoning, small/edge, domain
specialists, and multimodal reasoning groups. Total catalog: 46 -> 80.

Each Part group has its own test class. The final ``TestRecipeCatalog80``
asserts overall catalog invariants and parameterised per-recipe validity.
"""
from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.recipes.catalog import RECIPES, get_recipe, search_recipes

runner = CliRunner()


# ---------------------------------------------------------------------------
# Part A: Vision recipes (6 new)
# ---------------------------------------------------------------------------

PART_A_VISION = [
    ("llama3.2-vision-90b-sft", "sft", "meta-llama/Llama-3.2-90B-Vision-Instruct"),
    ("pixtral-12b-sft", "sft", "mistralai/Pixtral-12B-2409"),
    ("qwen2-vl-7b-sft", "sft", "Qwen/Qwen2-VL-7B-Instruct"),
    ("qwen2-vl-72b-sft", "sft", "Qwen/Qwen2-VL-72B-Instruct"),
    ("internvl-2.5-8b-sft", "sft", "OpenGVLab/InternVL2_5-8B"),
    ("minicpm-v-2.6-sft", "sft", "openbmb/MiniCPM-V-2_6"),
]


class TestPartAVision:
    """6 new vision/multimodal recipes."""

    @pytest.mark.parametrize("name,task,model", PART_A_VISION)
    def test_recipe_registered(self, name: str, task: str, model: str) -> None:
        assert name in RECIPES, f"Missing recipe: {name}"
        recipe = get_recipe(name)
        assert recipe is not None
        assert recipe.task == task
        assert recipe.model == model

    @pytest.mark.parametrize("name,task,model", PART_A_VISION)
    def test_recipe_loads_as_soupconfig(self, name: str, task: str, model: str) -> None:
        recipe = get_recipe(name)
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.base == model
        assert cfg.task == task
        assert cfg.modality == "vision", f"{name} must declare modality: vision"

    @pytest.mark.parametrize("name,task,model", PART_A_VISION)
    def test_recipe_has_vision_format(self, name: str, task: str, model: str) -> None:
        """Vision recipes must use llava or sharegpt4v format."""
        recipe = get_recipe(name)
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.data.format in ("llava", "sharegpt4v")

    @pytest.mark.parametrize("query,expected_model", [
        ("Pixtral", "mistralai/Pixtral-12B-2409"),
        ("Qwen2-VL", "Qwen/Qwen2-VL-7B-Instruct"),
        ("Qwen2-VL", "Qwen/Qwen2-VL-72B-Instruct"),
        ("InternVL", "OpenGVLab/InternVL2_5-8B"),
        ("MiniCPM", "openbmb/MiniCPM-V-2_6"),
        ("Llama-3.2-90B", "meta-llama/Llama-3.2-90B-Vision-Instruct"),
    ])
    def test_vision_search_returns_each_new_recipe(
        self, query: str, expected_model: str
    ) -> None:
        """Each new vision model surfaces via keyword search (per-model assertion)."""
        results = search_recipes(query=query)
        models = {r.model for r in results}
        assert expected_model in models, (
            f"search('{query}') missed {expected_model}; got {models}"
        )

    @pytest.mark.parametrize("name,_task,_model", PART_A_VISION)
    def test_vision_recipe_sets_image_dir(
        self, name: str, _task: str, _model: str
    ) -> None:
        """Vision recipes must declare image_dir or a non-empty fallback path."""
        cfg = load_config_from_string(get_recipe(name).yaml_str)
        assert cfg.data.image_dir, f"{name} missing image_dir for vision modality"


# ---------------------------------------------------------------------------
# Part B: Audio recipes (3 new)
# ---------------------------------------------------------------------------

PART_B_AUDIO = [
    ("qwen2-audio-7b-sft", "sft", "Qwen/Qwen2-Audio-7B-Instruct"),
    ("seamlessm4t-v2-sft", "sft", "facebook/seamless-m4t-v2-large"),
    ("whisper-large-v3-ft", "sft", "openai/whisper-large-v3"),
]


class TestPartBAudio:
    """3 new audio fine-tuning recipes."""

    @pytest.mark.parametrize("name,task,model", PART_B_AUDIO)
    def test_recipe_registered(self, name: str, task: str, model: str) -> None:
        assert name in RECIPES
        recipe = get_recipe(name)
        assert recipe.task == task
        assert recipe.model == model

    @pytest.mark.parametrize("name,task,model", PART_B_AUDIO)
    def test_recipe_loads_with_audio_modality(self, name: str, task: str, model: str) -> None:
        recipe = get_recipe(name)
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.modality == "audio", f"{name} must declare modality: audio"
        assert cfg.data.format == "audio"

    @pytest.mark.parametrize("name,_task,_model", PART_B_AUDIO)
    def test_audio_recipe_sets_audio_dir(
        self, name: str, _task: str, _model: str
    ) -> None:
        """Audio recipes must declare audio_dir."""
        cfg = load_config_from_string(get_recipe(name).yaml_str)
        assert cfg.data.audio_dir, f"{name} missing audio_dir for audio modality"


# ---------------------------------------------------------------------------
# Part C: Reasoning recipes (7 new)
# ---------------------------------------------------------------------------

PART_C_REASONING = [
    ("r1-distill-qwen-1.5b-grpo", "grpo", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("r1-distill-qwen-7b-grpo", "grpo", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("r1-distill-qwen-14b-grpo", "grpo", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
    ("r1-distill-llama-70b-grpo", "grpo", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
    ("qwen3-coder-30b-sft", "sft", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
    ("qwen3-30b-a3b-reasoning-grpo", "grpo", "Qwen/Qwen3-30B-A3B"),
    ("phi4-reasoning-grpo", "grpo", "microsoft/phi-4"),
]


class TestPartCReasoning:
    """7 new reasoning recipes (R1 distills + Qwen3-Coder + Qwen3 reasoning + Phi-4)."""

    @pytest.mark.parametrize("name,task,model", PART_C_REASONING)
    def test_recipe_registered(self, name: str, task: str, model: str) -> None:
        assert name in RECIPES
        recipe = get_recipe(name)
        assert recipe.task == task
        assert recipe.model == model

    @pytest.mark.parametrize("name,task,model", PART_C_REASONING)
    def test_recipe_loads(self, name: str, task: str, model: str) -> None:
        cfg = load_config_from_string(get_recipe(name).yaml_str)
        assert cfg.base == model
        if task == "grpo":
            assert cfg.training.reward_fn is not None, f"{name} must define reward_fn"

    @pytest.mark.parametrize("model_id", [
        # New in v0.31.0
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        # Already in catalog from earlier releases
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ])
    def test_all_r1_distill_model_ids_registered(self, model_id: str) -> None:
        """All 6 DeepSeek-R1-Distill sizes appear somewhere in the catalog.

        4 are added by v0.31.0; 2 ship from earlier releases (32B-Qwen and
        8B-Llama via ``deepseek-r1-32b-grpo`` / ``deepseek-r1-8b-grpo``).
        """
        registered = {r.model for r in RECIPES.values()}
        assert model_id in registered


# ---------------------------------------------------------------------------
# Part D: Small / edge recipes (8 new)
# ---------------------------------------------------------------------------

PART_D_EDGE = [
    ("qwen2.5-0.5b-sft", "sft", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-1.5b-sft", "sft", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen2.5-3b-sft", "sft", "Qwen/Qwen2.5-3B-Instruct"),
    ("gemma2-2b-sft", "sft", "google/gemma-2-2b-it"),
    ("smollm2-135m-sft", "sft", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("smollm2-360m-sft", "sft", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("smollm2-1.7b-sft", "sft", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("phi3.5-mini-sft", "sft", "microsoft/Phi-3.5-mini-instruct"),
]


class TestPartDEdge:
    """8 new small / edge model recipes."""

    @pytest.mark.parametrize("name,task,model", PART_D_EDGE)
    def test_recipe_registered(self, name: str, task: str, model: str) -> None:
        assert name in RECIPES
        recipe = get_recipe(name)
        assert recipe.task == task
        assert recipe.model == model

    @pytest.mark.parametrize("name,task,model", PART_D_EDGE)
    def test_recipe_loads(self, name: str, task: str, model: str) -> None:
        cfg = load_config_from_string(get_recipe(name).yaml_str)
        assert cfg.base == model

    @pytest.mark.parametrize("name,task,model", PART_D_EDGE)
    def test_edge_recipes_tagged_edge(self, name: str, task: str, model: str) -> None:
        """Edge recipes should advertise themselves with an 'edge' or 'small'/'tiny' tag."""
        recipe = get_recipe(name)
        joined = " ".join(recipe.tags).lower()
        assert any(tag in joined for tag in ("edge", "tiny", "small", "mobile"))


# ---------------------------------------------------------------------------
# Part E: Domain specialists (8 new)
# ---------------------------------------------------------------------------

PART_E_DOMAIN = [
    ("biomistral-7b-sft", "sft", "BioMistral/BioMistral-7B"),
    ("meditron-7b-sft", "sft", "epfl-llm/meditron-7b"),
    ("codellama-70b-sft", "sft", "codellama/CodeLlama-70b-Instruct-hf"),
    ("codellama-13b-sft", "sft", "codellama/CodeLlama-13b-Instruct-hf"),
    ("magicoder-7b-sft", "sft", "ise-uiuc/Magicoder-S-DS-6.7B"),
    ("nemotron-4-340b-sft", "sft", "nvidia/Nemotron-4-340B-Instruct"),
    ("llama2-13b-finance-sft", "sft", "meta-llama/Llama-2-13b-hf"),
    ("mathstral-7b-sft", "sft", "mistralai/Mathstral-7B-v0.1"),
]


class TestPartEDomain:
    """8 new domain specialist recipes (medical, code, finance, math)."""

    @pytest.mark.parametrize("name,task,model", PART_E_DOMAIN)
    def test_recipe_registered(self, name: str, task: str, model: str) -> None:
        assert name in RECIPES
        recipe = get_recipe(name)
        assert recipe.task == task
        assert recipe.model == model

    @pytest.mark.parametrize("name,task,model", PART_E_DOMAIN)
    def test_recipe_loads(self, name: str, task: str, model: str) -> None:
        cfg = load_config_from_string(get_recipe(name).yaml_str)
        assert cfg.base == model

    def test_domain_tags_present(self) -> None:
        """Each discipline-specific recipe carries its domain tag.

        ``nemotron-4-340b-sft`` is intentionally omitted: it is a
        general-purpose large model with no single discipline tag.
        """
        recipe_to_domain = {
            "biomistral-7b-sft": "medical",
            "meditron-7b-sft": "medical",
            "codellama-70b-sft": "code",
            "codellama-13b-sft": "code",
            "magicoder-7b-sft": "code",
            "llama2-13b-finance-sft": "finance",
            "mathstral-7b-sft": "math",
        }
        for name, domain in recipe_to_domain.items():
            recipe = get_recipe(name)
            assert domain in recipe.tags, f"{name} missing '{domain}' tag (got {recipe.tags})"


# ---------------------------------------------------------------------------
# Part F: Multimodal reasoning (2 new)
# ---------------------------------------------------------------------------

PART_F_MM_REASONING = [
    ("llama3.2-vision-grpo", "grpo", "meta-llama/Llama-3.2-11B-Vision-Instruct"),
    ("pixtral-dpo", "dpo", "mistralai/Pixtral-12B-2409"),
]

# Single source of truth across the file — concatenated for catalog-wide
# parametrizations. If any group above is renamed, every consumer below
# updates automatically (closes the divergence vector flagged by TDD M2).
ALL_V031_RECIPES = (
    PART_A_VISION + PART_B_AUDIO + PART_C_REASONING
    + PART_D_EDGE + PART_E_DOMAIN + PART_F_MM_REASONING
)


class TestPartFMultimodalReasoning:
    """2 new multimodal alignment / reasoning recipes."""

    @pytest.mark.parametrize("name,task,model", PART_F_MM_REASONING)
    def test_recipe_registered(self, name: str, task: str, model: str) -> None:
        assert name in RECIPES
        recipe = get_recipe(name)
        assert recipe.task == task
        assert recipe.model == model

    @pytest.mark.parametrize("name,task,model", PART_F_MM_REASONING)
    def test_recipe_loads_with_vision_modality(self, name: str, task: str, model: str) -> None:
        cfg = load_config_from_string(get_recipe(name).yaml_str)
        assert cfg.modality == "vision"

    def test_vision_grpo_recipe_sets_reward_fn(self) -> None:
        """The multimodal GRPO recipe must define a reward_fn (not silent stub)."""
        cfg = load_config_from_string(get_recipe("llama3.2-vision-grpo").yaml_str)
        assert cfg.training.reward_fn is not None
        assert cfg.training.num_generations is not None


# ---------------------------------------------------------------------------
# Part G: Per-recipe parameterised validation (catalog-wide)
# ---------------------------------------------------------------------------

class TestRecipeCatalog80:
    """Catalog-wide invariants after v0.31.0 expansion (46 -> 80 recipes)."""

    def test_total_catalog_size_is_80(self) -> None:
        assert len(RECIPES) == 80

    @pytest.mark.parametrize("name", sorted(RECIPES.keys()))
    def test_every_recipe_loads_as_soupconfig(self, name: str) -> None:
        recipe = RECIPES[name]
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.base == recipe.model, (
            f"Recipe '{name}' base mismatch: meta={recipe.model}, yaml={cfg.base}"
        )
        assert cfg.task == recipe.task

    @pytest.mark.parametrize("name", sorted(RECIPES.keys()))
    def test_every_recipe_meta_matches_yaml(self, name: str) -> None:
        recipe = RECIPES[name]
        parsed = yaml.safe_load(recipe.yaml_str)
        assert parsed.get("base") == recipe.model
        # Require an explicit task: line in the YAML so RecipeMeta.task and the
        # yaml never silently disagree (task=sft is the schema default but the
        # recipe yaml must spell it out for review-ability).
        assert "task" in parsed, f"Recipe '{name}' yaml missing explicit task: line"
        assert recipe.task == parsed["task"]

    @pytest.mark.parametrize("name", sorted(RECIPES.keys()))
    def test_every_recipe_has_non_empty_tags_and_description(self, name: str) -> None:
        recipe = RECIPES[name]
        assert recipe.tags, f"Recipe '{name}' has no tags"
        assert recipe.description, f"Recipe '{name}' has no description"
        assert len(recipe.description) >= 10, f"Recipe '{name}' description too short"

    @pytest.mark.parametrize("name", sorted(RECIPES.keys()))
    def test_every_recipe_model_id_is_safe(self, name: str) -> None:
        """Model ids must be HF-style 'org/name' or local-safe (no traversal markers)."""
        model = RECIPES[name].model
        # Reject path traversal / null-byte / scheme injection in model strings.
        assert "\x00" not in model
        assert ".." not in model
        assert "://" not in model
        # Reject windows-style backslash separators (defense in depth).
        assert "\\" not in model

    @pytest.mark.parametrize("name", sorted(RECIPES.keys()))
    def test_every_recipe_default_data_path_is_relative(self, name: str) -> None:
        """Default training data path must be a non-empty relative path."""
        recipe = RECIPES[name]
        cfg = load_config_from_string(recipe.yaml_str)
        train_path = cfg.data.train
        assert train_path, f"{name} has empty/missing data.train"
        # Paths starting with / on POSIX or with a Windows drive letter are absolute.
        assert not train_path.startswith("/"), f"{name} uses absolute data path"
        assert not (len(train_path) >= 2 and train_path[1] == ":"), (
            f"{name} uses Windows drive-letter absolute path"
        )

    @pytest.mark.parametrize("name", sorted(RECIPES.keys()))
    def test_every_recipe_lora_target_modules_set(self, name: str) -> None:
        """``lora.target_modules`` must be 'auto' or a non-empty list — never empty.

        An empty target_modules list passes Pydantic validation but produces a
        no-op LoRA adapter at training time with no error: a silent failure
        mode the catalog must guard against.
        """
        cfg = load_config_from_string(RECIPES[name].yaml_str)
        modules = cfg.training.lora.target_modules
        if isinstance(modules, list):
            assert modules, f"{name} has empty lora.target_modules list"
        else:
            assert modules == "auto", (
                f"{name} lora.target_modules must be 'auto' or a non-empty list, "
                f"got {modules!r}"
            )

    @pytest.mark.parametrize("name", sorted(RECIPES.keys()))
    def test_every_recipe_max_length_within_bounds(self, name: str) -> None:
        """max_length must be within the schema bounds [64, 1_048_576]."""
        cfg = load_config_from_string(RECIPES[name].yaml_str)
        assert 64 <= cfg.data.max_length <= 1_048_576, (
            f"{name} max_length={cfg.data.max_length} out of [64, 1M]"
        )

    @pytest.mark.parametrize("name", sorted(
        n for n, r in RECIPES.items() if r.task == "grpo"
    ))
    def test_every_grpo_recipe_has_reward_fn_and_num_generations(
        self, name: str
    ) -> None:
        """GRPO recipes must wire reward_fn and num_generations — required for the trainer."""
        cfg = load_config_from_string(RECIPES[name].yaml_str)
        assert cfg.training.reward_fn is not None, (
            f"GRPO recipe '{name}' missing reward_fn"
        )
        assert cfg.training.num_generations is not None, (
            f"GRPO recipe '{name}' missing num_generations"
        )
        assert cfg.training.num_generations >= 2, (
            f"GRPO recipe '{name}' num_generations={cfg.training.num_generations} "
            "must be >= 2"
        )

# ---------------------------------------------------------------------------
# Part H: Recipe verification CI workflow file present
# ---------------------------------------------------------------------------

class TestPartHVerificationWorkflow:
    """v0.31.0 ships a CI workflow that validates every recipe at PR time.

    The 100-step live smoke train requires GPU and is deferred to v0.31.1; for
    v0.31.0 the workflow runs the parameterised pytest validation in
    test_recipes_v031.py to catch regressions from upstream model id changes.
    """

    def test_recipe_validation_workflow_exists(self) -> None:
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        workflow = repo_root / ".github" / "workflows" / "recipe-validation.yml"
        assert workflow.exists(), (
            f"Expected CI workflow at {workflow} to validate recipes on every PR"
        )
        content = workflow.read_text(encoding="utf-8")
        # Must invoke the per-recipe test file
        assert "test_recipes_v031.py" in content

    def test_recipe_validation_workflow_triggers(self) -> None:
        """Workflow YAML structure: triggered on push + pull_request to main."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        workflow_path = repo_root / ".github" / "workflows" / "recipe-validation.yml"
        # Note: GitHub Actions parses 'on:' as Python True (yaml 1.1 boolean
        # alias) — so look up via the truthy key, falling back to 'on'.
        parsed = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        triggers = parsed.get(True) or parsed.get("on") or {}
        assert "push" in triggers, "workflow must trigger on push"
        assert "pull_request" in triggers, "workflow must trigger on pull_request"
        for evt in ("push", "pull_request"):
            branches = triggers[evt].get("branches", [])
            assert "main" in branches, f"{evt} trigger must filter to main branch"


# ---------------------------------------------------------------------------
# CLI smoke (v0.31.0 recipes are reachable via `soup recipes`)
# ---------------------------------------------------------------------------

class TestRecipesCLIv031:
    """v0.31.0 recipes are reachable through `soup recipes show`."""

    @pytest.mark.parametrize("name,_task,_model", ALL_V031_RECIPES)
    def test_show_recipe(self, name: str, _task: str, _model: str) -> None:
        result = runner.invoke(app, ["recipes", "show", name])
        assert result.exit_code == 0, (
            f"`soup recipes show {name}` failed: {result.output!r}"
        )
        # The model id appears verbatim in the YAML output.
        assert _model in result.output

    def test_list_runs_clean(self) -> None:
        """`soup recipes list` exits 0 (Rich table rendering is terminal-width-dependent)."""
        result = runner.invoke(app, ["recipes", "list"])
        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize("query,expected", [
        ("pixtral", "pixtral-12b-sft"),
        ("smollm", "smollm2-135m-sft"),
        ("biomistral", "biomistral-7b-sft"),
        ("meditron", "meditron-7b-sft"),
        ("mathstral", "mathstral-7b-sft"),
    ])
    def test_search_finds_v031_recipes(self, query: str, expected: str) -> None:
        """`soup recipes search <kw>` surfaces v0.31.0 recipes by keyword."""
        result = runner.invoke(app, ["recipes", "search", query])
        assert result.exit_code == 0, result.output
        assert expected in result.output, f"{expected} missing from search '{query}'"
