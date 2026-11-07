"""Resident K2 fails at configuration, before remote-code/model construction."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.recipes.catalog import RECIPES
from soup_cli.utils.exit_codes import EXIT_USAGE_ERROR
from tests.conftest import strip_ansi

BASES = ("moonshotai/Kimi-K2-Base", "moonshotai/Kimi-K2-Instruct",
         "moonshotai/Kimi-K2-Instruct-0905", "moonshotai/Kimi-K2-Thinking")
REASON = "Resident Kimi K2 is unsupported on Transformers 5.x"
BLOCKED = ("kimi-k2-sft", "kimi-k2-thinking-grpo")


def _assert_route_text(text, base):
    if base.casefold() in {"moonshotai/kimi-k2-base", "moonshotai/kimi-k2-instruct"}:
        assert "training.stream_layers: true" in text
        assert "guarded" in text.lower()
        assert "not available until it merges" not in text
        assert "pending" not in text.lower()
        assert "--trust-remote-code" in text and "tokenizer" in text
        assert "pip install tiktoken" in text
        assert "docs/performance-and-quantization.md" in text
        assert "CPU-only parity on a tiny K2-like checkpoint" in text
        assert "no supported route yet" not in text.lower()
    else:
        assert "no supported route yet" in text.lower()
        assert "stream_layers" not in text and "#2107" not in text


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("task", ["sft", "grpo"])
def test_resident_k2_refuses_with_a_specific_remedy(base, task):
    with pytest.raises(ValueError, match=REASON) as error:
        load_config_from_string(f"base: {base}\ntask: {task}\ndata: {{train: data.jsonl}}\n")
    assert base in str(error.value)
    _assert_route_text(str(error.value), base)
    assert "remote code" in str(error.value)


@pytest.mark.parametrize("base", ["moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6",
                                 "deepseek-ai/DeepSeek-V3", "other/Kimi-K2-Base", "./local-k2"])
def test_unrelated_models_are_not_reclassified(base):
    cfg = load_config_from_string(f"base: {base}\ndata: {{train: data.jsonl}}\n")
    assert cfg.base == base


def test_a_relative_local_directory_is_not_mistaken_for_the_public_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / BASES[0]).mkdir(parents=True)
    cfg = load_config_from_string(f"base: {BASES[0]}\ndata: {{train: data.jsonl}}\n")
    assert cfg.base == BASES[0]


@pytest.mark.parametrize("base", BASES)
def test_hub_id_case_does_not_bypass_the_guard(base):
    with pytest.raises(ValueError, match=REASON) as error:
        load_config_from_string(f"base: {base.upper()}\ndata: {{train: data.jsonl}}\n")
    _assert_route_text(str(error.value), base)


@pytest.mark.parametrize("override", ["training: {stream_layers: true, batch_size: 1}",
                                     "backend: unsloth", "training: {hub: modelscope}"])
def test_guard_does_not_claim_other_loading_routes(override):
    # This is a scope control, not evidence these routes can train K2 today.
    cfg = load_config_from_string(
        "base: moonshotai/Kimi-K2-Base\ndata: {train: data.jsonl}\n" + override + "\n",
    )
    assert cfg.base == "moonshotai/Kimi-K2-Base"


@pytest.mark.parametrize("base", [BASES[0], BASES[-1]])
@pytest.mark.parametrize("mode", ["--dry-run", "--yes"])
def test_train_cli_refuses_before_any_model_or_config_download(tmp_path, monkeypatch, base, mode):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        f"base: {base}\ndata: {{train: data.jsonl, val_split: 0}}\n", encoding="utf-8",
    )
    (tmp_path / "data.jsonl").write_text(
        '{"instruction":"Q", "output":"A"}\n', encoding="utf-8",
    )
    build = Mock(side_effect=AssertionError("refused K2 reached trainer construction"))
    metadata = Mock(side_effect=AssertionError("refused K2 requested model metadata"))
    monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", build)
    monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", metadata)
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", mode])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 1, (output, result.exception)
    assert REASON in output and base in output
    _assert_route_text(output, base)
    build.assert_not_called()
    metadata.assert_not_called()


@pytest.mark.parametrize("name", BLOCKED)
def test_catalog_marks_refused_recipes_and_does_not_write_them(tmp_path, monkeypatch, name):
    from soup_cli.cli import app

    recipe = RECIPES[name]
    assert recipe.unavailable_reason.startswith(REASON)
    _assert_route_text(recipe.unavailable_reason, recipe.model)
    if name == "kimi-k2-sft":
        assert len(recipe.description) <= 100
        assert "resident unsupported" in recipe.description
        assert "guarded streaming only" in recipe.description
        assert "recipes use" in recipe.description
    else:
        _assert_route_text(recipe.description, recipe.model)
    assert "unsupported" in recipe.description.lower()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "1000")
    runner = CliRunner()
    shown = runner.invoke(app, ["recipes", "show", name])
    assert shown.exit_code == 0
    assert "unsupported" in strip_ansi(shown.output).lower()
    listed = runner.invoke(app, ["recipes", "list"])
    assert listed.exit_code == 0
    assert "unsupported" in strip_ansi(listed.output).lower()
    used = runner.invoke(app, ["recipes", "use", name, "--yes"])
    assert used.exit_code == EXIT_USAGE_ERROR == 3
    output = " ".join(strip_ansi(used.output).split())
    assert recipe.unavailable_reason in output
    _assert_route_text(output, recipe.model)
    assert not (tmp_path / "soup.yaml").exists()


def test_catalog_unavailability_is_limited_to_the_two_affected_recipes():
    assert {name for name, recipe in RECIPES.items() if recipe.unavailable_reason} == set(BLOCKED)


def test_snapshot_records_each_recipes_own_route_reason():
    from scripts.generate_recipe_snapshot import build_snapshot

    snapshot = build_snapshot()
    for name in BLOCKED:
        record = snapshot["recipes"][name]
        assert record == {"$unavailable": RECIPES[name].unavailable_reason}
        _assert_route_text(record["$unavailable"], RECIPES[name].model)


def test_snapshot_does_not_hide_an_unrelated_config_error(monkeypatch):
    from scripts.generate_recipe_snapshot import build_snapshot
    from soup_cli.recipes import catalog

    invalid = replace(RECIPES[BLOCKED[0]], yaml_str=RECIPES[BLOCKED[0]].yaml_str + "typo: true\n")
    monkeypatch.setattr(catalog, "RECIPES", {BLOCKED[0]: invalid})
    with pytest.raises(ValueError, match="unknown config key"):
        build_snapshot()


def test_snapshot_refuses_a_stale_unavailable_marker(monkeypatch):
    from scripts.generate_recipe_snapshot import build_snapshot
    from soup_cli.recipes import catalog

    stale = replace(RECIPES["qwen3.5-4b-sft"], unavailable_reason=REASON)
    monkeypatch.setattr(catalog, "RECIPES", {"qwen3.5-4b-sft": stale})
    with pytest.raises(ValueError, match="marked unavailable but its config loads"):
        build_snapshot()
