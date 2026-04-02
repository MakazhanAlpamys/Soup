"""Tests for soup recipes — ready-made configs for popular models."""


import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Recipe catalog tests
# ---------------------------------------------------------------------------

class TestRecipeCatalog:
    """Tests for recipe catalog and search."""

    def test_list_recipes(self):
        """list_recipes returns all recipes."""
        from soup_cli.recipes.catalog import RECIPES, list_recipes

        recipes = list_recipes()
        assert len(recipes) > 0
        assert len(recipes) == len(RECIPES)

    def test_get_recipe_exists(self):
        """get_recipe returns a known recipe."""
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("llama3.1-8b-sft")
        assert recipe is not None
        assert recipe.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert recipe.task == "sft"

    def test_get_recipe_not_exists(self):
        """get_recipe returns None for unknown recipe."""
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("nonexistent-recipe")
        assert recipe is None

    def test_search_by_task(self):
        """search_recipes filters by task."""
        from soup_cli.recipes.catalog import search_recipes

        results = search_recipes(task="grpo")
        assert len(results) > 0
        assert all(r.task == "grpo" for r in results)

    def test_search_by_keyword(self):
        """search_recipes matches by keyword."""
        from soup_cli.recipes.catalog import search_recipes

        results = search_recipes(query="reasoning")
        assert len(results) > 0

    def test_search_by_size(self):
        """search_recipes filters by model size."""
        from soup_cli.recipes.catalog import search_recipes

        results = search_recipes(size="7b")
        assert len(results) > 0
        for recipe in results:
            assert "7b" in recipe.size.lower() or "7b" in recipe.model.lower()

    def test_search_no_results(self):
        """search_recipes returns empty list for no matches."""
        from soup_cli.recipes.catalog import search_recipes

        results = search_recipes(query="zzzznonexistent")
        assert results == []

    def test_all_recipes_valid_yaml(self):
        """All recipes contain valid YAML that loads as SoupConfig."""
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import RECIPES

        for name, recipe in RECIPES.items():
            try:
                config = load_config_from_string(recipe.yaml_str)
                assert config.base is not None, f"Recipe {name} has no base model"
            except Exception as exc:
                pytest.fail(f"Recipe '{name}' is invalid: {exc}")

    def test_recipe_has_required_fields(self):
        """All recipes have model, task, size, tags, yaml_str."""
        from soup_cli.recipes.catalog import RECIPES

        for name, recipe in RECIPES.items():
            assert recipe.model, f"Recipe {name} missing model"
            assert recipe.task, f"Recipe {name} missing task"
            assert recipe.size, f"Recipe {name} missing size"
            assert recipe.yaml_str, f"Recipe {name} missing yaml_str"

    def test_recipe_tasks_match_yaml(self):
        """Recipe.task matches the task in the YAML content."""
        import yaml

        from soup_cli.recipes.catalog import RECIPES

        for name, recipe in RECIPES.items():
            parsed = yaml.safe_load(recipe.yaml_str)
            yaml_task = parsed.get("task", "sft")
            assert recipe.task == yaml_task, (
                f"Recipe '{name}' task mismatch: meta={recipe.task}, yaml={yaml_task}"
            )


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestRecipesCLI:
    """CLI tests for soup recipes command."""

    def test_list_command(self):
        """soup recipes list shows all recipes."""
        result = runner.invoke(app, ["recipes", "list"])
        assert result.exit_code == 0
        assert "llama3.1-8b-sft" in result.output

    def test_list_default(self):
        """soup recipes (no subcommand) shows help."""
        result = runner.invoke(app, ["recipes"])
        assert result.exit_code == 0

    def test_show_recipe(self):
        """soup recipes show <name> prints YAML."""
        result = runner.invoke(app, ["recipes", "show", "llama3.1-8b-sft"])
        assert result.exit_code == 0
        assert "meta-llama" in result.output

    def test_show_unknown_recipe(self):
        """soup recipes show <unknown> shows error."""
        result = runner.invoke(app, ["recipes", "show", "nonexistent"])
        assert result.exit_code != 0

    def test_use_recipe(self, tmp_path, monkeypatch):
        """soup recipes use <name> writes soup.yaml."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "recipes", "use", "llama3.1-8b-sft",
            "--yes",
        ])
        assert result.exit_code == 0
        out_path = tmp_path / "soup.yaml"
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "meta-llama/Llama-3.1-8B-Instruct" in content

    def test_use_custom_output(self, tmp_path, monkeypatch):
        """soup recipes use <name> -o custom.yaml writes to custom path."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "recipes", "use", "qwen2.5-7b-sft",
            "-o", "my_config.yaml",
            "--yes",
        ])
        assert result.exit_code == 0
        assert (tmp_path / "my_config.yaml").exists()

    def test_use_overwrite_confirmation(self, tmp_path, monkeypatch):
        """Existing file asks for confirmation."""
        monkeypatch.chdir(tmp_path)
        out_file = tmp_path / "soup.yaml"
        out_file.write_text("existing content", encoding="utf-8")
        # Deny confirmation
        runner.invoke(app, [
            "recipes", "use", "llama3.1-8b-sft",
        ], input="n\n")
        assert out_file.read_text(encoding="utf-8") == "existing content"

    def test_search_command(self):
        """soup recipes search <query> shows results."""
        result = runner.invoke(app, ["recipes", "search", "reasoning"])
        assert result.exit_code == 0

    def test_search_by_task_flag(self):
        """soup recipes search --task grpo shows GRPO recipes."""
        result = runner.invoke(app, ["recipes", "search", "--task", "grpo"])
        assert result.exit_code == 0
        assert "grpo" in result.output.lower()

    def test_search_by_size_flag(self):
        """soup recipes search --size 7b shows 7B recipes."""
        result = runner.invoke(app, ["recipes", "search", "--size", "7b"])
        assert result.exit_code == 0

    def test_use_output_path_traversal(self, tmp_path, monkeypatch):
        """Output path traversal is blocked."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "recipes", "use", "llama3.1-8b-sft",
            "-o", "../../../tmp/evil.yaml",
            "--yes",
        ])
        assert result.exit_code != 0

    def test_use_unknown_recipe(self, tmp_path, monkeypatch):
        """soup recipes use <unknown> shows error."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "recipes", "use", "nonexistent-recipe",
            "--yes",
        ])
        assert result.exit_code != 0

    def test_help(self):
        """soup recipes --help shows usage."""
        result = runner.invoke(app, ["recipes", "--help"])
        assert result.exit_code == 0
        assert "recipes" in result.output.lower()
