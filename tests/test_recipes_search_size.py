"""Tests verifying recipe search by declared model size (#2930)."""

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.recipes.catalog import RECIPES, get_recipe, search_recipes

runner = CliRunner()


def test_all_catalog_recipes_findable_by_declared_size():
    """Every catalog recipe with a declared size is found when searched by that size."""
    for name, recipe in RECIPES.items():
        if not recipe.size or recipe.size == "N/A":
            continue
        results = search_recipes(size=recipe.size)
        assert recipe in results, (
            f"Recipe {name} ({recipe.size!r}) missing from search_recipes(size={recipe.size!r})"
        )


def test_previously_missed_recipes_findable_by_declared_size():
    """Recipes whose declared size was missing from name and model are found (#2930)."""
    probe_cases = [
        ("llama4-scout-17b-sft", "109B"),
        ("llama4-scout-17b-dpo", "109B"),
        ("llama4-scout-17b-grpo", "109B"),
        ("deepseek-v3-7b-sft", "685B"),
        ("llama4-scout-tools", "109B"),
        ("deepseek-v3-pipeline", "671B"),
        ("minicpm-v-2.6-sft", "8B"),
        ("seamlessm4t-v2-sft", "2.3B"),
        ("whisper-large-v3-ft", "1.5B"),
        ("phi4-reasoning-grpo", "14B"),
        ("whisper-tiny-asr", "39M"),
        ("whisper-base-asr", "74M"),
        ("whisper-large-v3-asr", "1.5B"),
        ("phi3.5-mini-sft", "3.8B"),
        ("glm-4.6-sft", "357B"),
        ("glm-5-sft", "754B"),
        ("minimax-m2-sft", "230B"),
        ("granite-4-sft", "7B"),
        ("magistral-small-sft", "24B"),
        ("devstral-sft", "24B"),
        ("voxtral-sft", "4.7B"),
        ("oute-tts", "0.5B"),
        ("glm-5.1-sft", "754B"),
        ("glm-5.1-dpo", "754B"),
        ("glm-5.1-grpo", "754B"),
        ("kimi-k2.5-sft", "1T"),
        ("kimi-k2.5-dpo", "1T"),
        ("kimi-k2.5-grpo", "1T"),
        ("kimi-k2.6-sft", "1T"),
        ("kimi-k2.6-grpo", "1T"),
        ("kimi-k2.6-dpo", "1T"),
        ("minimax-m3-sft", "428B"),
        ("minimax-m3-dpo", "428B"),
    ]
    for recipe_name, declared_size in probe_cases:
        recipe = get_recipe(recipe_name)
        assert recipe is not None, f"Recipe {recipe_name} not found in catalog"
        results = search_recipes(size=declared_size)
        assert recipe in results, (
            f"Recipe {recipe_name} ({declared_size}) missing from results: {results}"
        )


def test_case_insensitive_size_matching():
    """Size search is case-insensitive across uppercase and lowercase variants."""
    upper_109 = search_recipes(size="109B")
    lower_109 = search_recipes(size="109b")
    assert upper_109 == lower_109
    assert len(upper_109) >= 4

    upper_8 = search_recipes(size="8B")
    lower_8 = search_recipes(size="8b")
    assert upper_8 == lower_8
    assert len(upper_8) > 0


def test_non_size_query_fallback():
    """Non-size text queries match against name, model id and description (#2930)."""
    results = search_recipes(size="llama")
    assert len(results) > 0
    for r in results:
        names = [n for n, rec in RECIPES.items() if rec is r]
        matched = (
            any("llama" in n.lower() for n in names)
            or "llama" in r.model.lower()
            or "llama" in r.description.lower()
        )
        assert matched


def test_compound_size_and_task_filtering():
    """Combining size and task filters applies both constraints (positive control)."""
    results_sft = search_recipes(size="109B", task="sft")
    assert len(results_sft) > 0
    assert all(r.task == "sft" for r in results_sft)
    assert all("109b" in r.size.lower() for r in results_sft)

    results_dpo = search_recipes(size="109B", task="dpo")
    assert len(results_dpo) > 0
    assert all(r.task == "dpo" for r in results_dpo)
    assert all("109b" in r.size.lower() for r in results_dpo)

    # Disjoint combination returns empty
    results_disjoint = search_recipes(size="109B", task="pretrain")
    assert results_disjoint == []


def test_compound_size_and_query_filtering():
    """Combining size and query filters applies both constraints."""
    results = search_recipes(size="109B", query="scout")
    assert len(results) > 0
    for r in results:
        assert "109b" in r.size.lower()
        searchable = f"{r.model} {r.task} {r.description} {' '.join(r.tags)}"
        assert "scout" in searchable.lower()


def test_unmatched_size_returns_empty_list():
    """Querying an unknown size token cleanly returns an empty list."""
    results = search_recipes(size="999999Z")
    assert results == []


def test_decimal_size_queries():
    """Decimal size tokens (1.5B, 0.5B, 2.3B) match corresponding catalog recipes."""
    res_15 = search_recipes(size="1.5B")
    names_15 = {n for n, r in RECIPES.items() if r in res_15}
    assert "whisper-large-v3-ft" in names_15
    assert "whisper-large-v3-asr" in names_15

    res_05 = search_recipes(size="0.5B")
    names_05 = {n for n, r in RECIPES.items() if r in res_05}
    assert "oute-tts" in names_05

    res_23 = search_recipes(size="2.3B")
    names_23 = {n for n, r in RECIPES.items() if r in res_23}
    assert "seamlessm4t-v2-sft" in names_23


def test_size_query_excludes_longer_token_substrings():
    """A size query does not match recipes whose longer size token contains the text (#2930)."""
    # 7B must find 7B recipes (e.g. granite-4-sft) but NOT 17B, 27B, or 70B
    results_7b = search_recipes(size="7B")
    granite = get_recipe("granite-4-sft")
    assert granite in results_7b
    for r in results_7b:
        assert r.size.lower() == "7b"
    scout_17b = get_recipe("llama4-scout-17b-sft")
    assert scout_17b not in results_7b

    # 1B must NOT match 11B recipes
    results_1b = search_recipes(size="1B")
    assert len(results_1b) > 0
    for r in results_1b:
        assert r.size.lower() == "1b"


def test_mcp_tool_recipes_search_by_declared_size():
    """MCP tool_recipes_search resolves recipes matching declared model size."""
    from soup_cli.mcp_server.registry import tool_recipes_search

    res = tool_recipes_search({"size": "109B"})
    assert res["count"] >= 4
    names = [r["name"] for r in res["results"]]
    assert "llama4-scout-17b-sft" in names


def test_cli_recipes_search_size_flag():
    """CLI 'soup recipes search --size <size>' displays matching recipes."""
    cli_runner = CliRunner(env={"COLUMNS": "200"})
    result = cli_runner.invoke(app, ["recipes", "search", "--size", "109B"])
    assert result.exit_code == 0
    assert "llama4-scout-17b-sft" in result.output
    assert "109B" in result.output
