"""Resident K2 is refused at configuration, before remote-code/model construction.

#2104 task 2: the refusal lifts for Kimi-K2-Base and Kimi-K2-Instruct on sft/grpo
(the resident native route); those two IDs are therefore admitted there and still
refused for the other tasks and the other loading routes. Kimi-K2-Thinking and
Kimi-K2-Instruct-0905 stay refused for every task (streamed remedy).
"""

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
ADMITTED_BASES = ("moonshotai/Kimi-K2-Base", "moonshotai/Kimi-K2-Instruct")
REFUSED_BASES = ("moonshotai/Kimi-K2-Instruct-0905", "moonshotai/Kimi-K2-Thinking")
REASON = "Resident Kimi K2 is unsupported on Transformers 5.x"
BLOCKED = ("kimi-k2-thinking-grpo",)
K2_TASKS = ("sft", "grpo")


def _assert_route_text(text, base):
    if base.casefold() in {"moonshotai/kimi-k2-base", "moonshotai/kimi-k2-instruct"}:
        # Refused for a task outside {sft, grpo}: the streamed route is the remedy there.
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


@pytest.mark.parametrize("base", REFUSED_BASES)
@pytest.mark.parametrize("task", ["sft", "grpo", "dpo"])
def test_resident_k2_refuses_with_a_specific_remedy(base, task):
    with pytest.raises(ValueError, match=REASON) as error:
        load_config_from_string(f"base: {base}\ntask: {task}\ndata: {{train: data.jsonl}}\n")
    assert base in str(error.value)
    _assert_route_text(str(error.value), base)
    assert "remote code" in str(error.value)


@pytest.mark.parametrize("base", ADMITTED_BASES)
@pytest.mark.parametrize("task", K2_TASKS)
def test_resident_k2_admits_base_and_instruct_on_sft_and_grpo(base, task):
    # Mutation check (a): re-adding Base/Instruct to the refused set fails this test.
    # Review round 2: the admission holds at training.quantization: none only — the
    # resident route loads bf16 only (test_resident_k2_refuses_non_none_quantization_*).
    cfg = load_config_from_string(
        f"base: {base}\ntask: {task}\ndata: {{train: data.jsonl}}\n"
        "training: {quantization: none}\n"
    )
    assert cfg.base == base
    assert cfg.task == task


@pytest.mark.parametrize("base", ADMITTED_BASES)
def test_resident_k2_refuses_base_and_instruct_for_other_tasks(base):
    with pytest.raises(ValueError, match=REASON) as error:
        load_config_from_string(f"base: {base}\ntask: dpo\ndata: {{train: data.jsonl}}\n")
    assert base in str(error.value)
    _assert_route_text(str(error.value), base)


def test_schema_admission_depends_on_the_task():
    # Same base, two tasks: dpo is refused naming the base, sft loads at
    # quantization: none (config only, no model download happens at config validation).
    yaml_base = "base: moonshotai/Kimi-K2-Base\ndata: {train: data.jsonl}\n"
    with pytest.raises(ValueError, match="moonshotai/Kimi-K2-Base"):
        load_config_from_string(yaml_base + "task: dpo\n")
    cfg = load_config_from_string(yaml_base + "task: sft\ntraining: {quantization: none}\n")
    assert cfg.base == "moonshotai/Kimi-K2-Base"


# ---------------------------------------------------------------------------
# #2104 review round 2, ask 1: a non-none training.quantization is refused at
# config validation, not inside load_resident_model. The resident route loads
# bf16 only; the refusal names training.stream_layers: true as the 4-bit path.
# ---------------------------------------------------------------------------

QUANT_REFUSAL = (
    "resident Kimi K2 route: training.quantization '4bit' is refused — the resident "
    "route loads bf16 only; set training.quantization: none for this route, or train "
    "4-bit through training.stream_layers: true."
)


def _assert_quant_refusal(exc, base="moonshotai/Kimi-K2-Base", value="'4bit'"):
    message = str(exc)
    assert f"training.quantization {value}" in message
    assert "bf16 only" in message
    assert "training.quantization: none" in message
    assert "training.stream_layers: true" in message
    assert base in message or "resident Kimi K2 route" in message


def test_resident_k2_refuses_the_default_quantization_at_config_validation():
    """Mutation check: deleting the quantization clause in
    resident_kimi_k2_refusal makes this test fail — today the default
    (training.quantization unset -> '4bit') sails past config validation and
    only dies inside load_resident_model, after the tokenizer is fetched."""
    with pytest.raises(ValueError) as caught:
        load_config_from_string(
            "base: moonshotai/Kimi-K2-Base\ntask: sft\ndata: {train: data.jsonl}\n"
        )
    _assert_quant_refusal(caught.value)


def test_resident_k2_admits_an_explicit_quantization_none():
    cfg = load_config_from_string(
        "base: moonshotai/Kimi-K2-Base\ntask: sft\ndata: {train: data.jsonl}\n"
        "training: {quantization: none}\n"
    )
    assert cfg.training.quantization == "none"


@pytest.mark.parametrize("base", ADMITTED_BASES)
@pytest.mark.parametrize("task", K2_TASKS)
@pytest.mark.parametrize("value", ["4bit", "8bit"])
def test_resident_k2_refuses_non_none_quantization_at_config_validation(base, task, value):
    with pytest.raises(ValueError) as caught:
        load_config_from_string(
            f"base: {base}\ntask: {task}\ndata: {{train: data.jsonl}}\n"
            f"training: {{quantization: {value}}}\n"
        )
    _assert_quant_refusal(caught.value, base=base, value=repr(value))


@pytest.mark.parametrize("base", REFUSED_BASES)
@pytest.mark.parametrize("quant", ["4bit", "none"])
def test_resident_k2_quantization_clause_does_not_mask_the_thinking_refusal(base, quant):
    """Thinking/0905 stay refused for every task: the route refusal fires before the
    quantization clause, so the message is the no-supported-route one, not quantization."""
    with pytest.raises(ValueError, match=REASON) as caught:
        load_config_from_string(
            f"base: {base}\ntask: sft\ndata: {{train: data.jsonl}}\n"
            f"training: {{quantization: {quant}}}\n"
        )
    assert "no supported route yet" in str(caught.value).lower()
    assert "training.quantization" not in str(caught.value)


@pytest.mark.parametrize("base", ["deepseek-ai/DeepSeek-V3", "other/Kimi-K2-Base"])
def test_resident_k2_quantization_clause_does_not_touch_other_bases(base):
    cfg = load_config_from_string(
        f"base: {base}\ntask: sft\ndata: {{train: data.jsonl}}\n"
        "training: {quantization: 4bit}\n"
    )
    assert cfg.base == base


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


@pytest.mark.parametrize("base", REFUSED_BASES)
def test_hub_id_case_does_not_bypass_the_guard(base):
    with pytest.raises(ValueError, match=REASON) as error:
        load_config_from_string(f"base: {base.upper()}\ndata: {{train: data.jsonl}}\n")
    _assert_route_text(str(error.value), base)


@pytest.mark.parametrize("base", ADMITTED_BASES)
@pytest.mark.parametrize("task", K2_TASKS)
def test_hub_id_case_does_not_block_the_admission(base, task):
    cfg = load_config_from_string(f"base: {base.upper()}\ntask: {task}\n"
                                  f"data: {{train: data.jsonl}}\n"
                                  "training: {quantization: none}\n")
    assert cfg.base == base.upper()


@pytest.mark.parametrize("override", ["training: {stream_layers: true, batch_size: 1}",
                                     "backend: unsloth", "training: {hub: modelscope}"])
def test_guard_does_not_claim_other_loading_routes(override):
    # This is a scope control, not evidence these routes can train K2 today.
    cfg = load_config_from_string(
        "base: moonshotai/Kimi-K2-Base\ndata: {train: data.jsonl}\n" + override + "\n",
    )
    assert cfg.base == "moonshotai/Kimi-K2-Base"


@pytest.mark.parametrize("base", REFUSED_BASES)
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


@pytest.mark.parametrize("mode", ["--dry-run", "--yes"])
def test_train_cli_admits_base_sft_past_config_validation(tmp_path, monkeypatch, mode):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        f"base: {BASES[0]}\ndata: {{train: data.jsonl, val_split: 0}}\n"
        "training: {quantization: none}\n",
        encoding="utf-8",
    )
    (tmp_path / "data.jsonl").write_text(
        '{"instruction":"Q", "output":"A"}\n', encoding="utf-8",
    )
    build = Mock(side_effect=AssertionError("admitted K2 reached trainer construction"))
    metadata = Mock(side_effect=AssertionError("admitted K2 requested model metadata"))
    monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", build)
    monkeypatch.setattr("soup_cli.utils.hf.get_hf_api", metadata)
    result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", mode])
    output = " ".join(strip_ansi(result.output).split())
    assert REASON not in output, (output, result.exception)
    if mode == "--dry-run":
        # A dry run validates the config and stops before trainer construction.
        assert result.exit_code == 0, (output, result.exception)
        build.assert_not_called()
    else:
        assert isinstance(result.exception, AssertionError)
        assert "reached trainer construction" in str(result.exception)
        build.assert_called_once()
    metadata.assert_not_called()


@pytest.mark.parametrize("name", BLOCKED)
def test_catalog_marks_refused_recipes_and_does_not_write_them(tmp_path, monkeypatch, name):
    from soup_cli.cli import app

    recipe = RECIPES[name]
    assert recipe.unavailable_reason.startswith(REASON)
    _assert_route_text(recipe.unavailable_reason, recipe.model)
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


def test_catalog_keeps_the_sft_recipe_unavailable_this_round(tmp_path, monkeypatch):
    """Review round 2, ask 6: the recipe restore is not part of this round. The row stays
    unavailable: its reason is the config-validation refusal the reverted yaml
    (quantization: 4bit) meets, plus the sentence naming what landed and reserving
    enablement for review."""
    from soup_cli.cli import app
    from soup_cli.recipes.catalog import KIMI_K2_SFT_RECIPE_RESERVED_REASON

    recipe = RECIPES["kimi-k2-sft"]
    assert recipe.unavailable_reason == KIMI_K2_SFT_RECIPE_RESERVED_REASON
    assert "bf16 only" in recipe.unavailable_reason
    assert "training.stream_layers: true" in recipe.unavailable_reason
    assert "guarded native resident route" in recipe.unavailable_reason
    assert "reserved for review" in recipe.unavailable_reason
    assert len(recipe.description) <= 100
    assert "unsupported" in recipe.description.lower()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "1000")
    runner = CliRunner()
    shown = runner.invoke(app, ["recipes", "show", "kimi-k2-sft"])
    assert shown.exit_code == 0
    assert "reserved for review" in strip_ansi(shown.output)
    listed = runner.invoke(app, ["recipes", "list"])
    assert listed.exit_code == 0
    assert "unsupported" in strip_ansi(listed.output).lower()
    used = runner.invoke(app, ["recipes", "use", "kimi-k2-sft", "--yes"])
    assert used.exit_code == EXIT_USAGE_ERROR == 3
    output = " ".join(strip_ansi(used.output).split())
    assert recipe.unavailable_reason in output
    assert not (tmp_path / "soup.yaml").exists()


def test_catalog_unavailability_is_limited_to_the_kimi_k2_rows():
    # Both K2 rows stay unavailable this round: Thinking/0905 (no supported route yet)
    # and the sft recipe (the guarded native route exists; enablement is reserved for
    # review).
    assert {name for name, recipe in RECIPES.items() if recipe.unavailable_reason} == (
        set(BLOCKED) | {"kimi-k2-sft"}
    )


def test_snapshot_records_each_recipes_own_route_reason():
    from scripts.generate_recipe_snapshot import build_snapshot

    snapshot = build_snapshot()
    grpo = snapshot["recipes"]["kimi-k2-thinking-grpo"]
    assert grpo == {"$unavailable": RECIPES["kimi-k2-thinking-grpo"].unavailable_reason}
    _assert_route_text(grpo["$unavailable"], RECIPES["kimi-k2-thinking-grpo"].model)
    # The sft row is unavailable again this round (ask 6): the recorded reason is the
    # reservation wording, and it starts from the refusal the reverted yaml meets.
    sft = snapshot["recipes"]["kimi-k2-sft"]
    assert sft == {"$unavailable": RECIPES["kimi-k2-sft"].unavailable_reason}
    assert "reserved for review" in sft["$unavailable"]
    assert "bf16 only" in sft["$unavailable"]


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
