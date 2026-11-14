"""The sft recipe the loop trains with (target modules, LoRA r / alpha, lr, epochs) is
configurable in ``loop.yaml``.

The recipe used to be five hard-coded numbers in ``_render_train_yaml``. Now:

- ``DEFAULT_SFT_RECIPE`` (loop_state.py) holds the defaults, final (candidate B in bf16, the
  fit runs of 2026-11-06); one test below pins its numbers (TestConstant), the rest
  read the constant;
- the keys are ``target_modules``, ``lora_r``, ``lora_alpha``, ``lr``, ``epochs``,
  ``batch_size``, ``gradient_accumulation_steps``, ``seed`` (the last three are pinned:
  tests/test_loop_sft_batch_and_seed.py) and ``quantization`` (``4bit``, ``none`` or ``auto``;
  ``auto`` is measured when the train stage renders the config:
  tests/test_loop_sft_quantization.py);
- ``LoopState.sft_recipe`` (last field) is ``None`` (the constant) or a dict that overrides
  only the keys it names; unknown keys, out-of-range values and bool-as-int are refused,
  by the state itself and so also when a hand-edited ``loop.yaml`` is read;
- ``soup loop init --sft-target-modules / --sft-lora-r / --sft-lora-alpha / --sft-lr /
  --sft-epochs / --sft-quantization`` store a key only when given, refuse (exit 2, naming
  the flag) a value out of range, and refuse all of them for a loop that does not train sft;
- ``_render_train_yaml`` writes the merged recipe for sft (dpo and kto are unchanged), and
  every rendered config is one ``SoupConfig`` accepts;
- ``soup loop status`` shows the effective sft recipe in one row and says which keys come
  from ``loop.yaml``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
import soup_cli.utils.loop_state as loop_state
from soup_cli.config.schema import SoupConfig
from soup_cli.utils.loop_state import (
    DEFAULT_SFT_RECIPE,
    SFT_RECIPE_KEYS,
    LoopState,
    effective_sft_recipe,
    init_state,
    read_state,
    validate_sft_recipe,
    validate_sft_recipe_value,
    write_state,
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

#: The two candidate recipes of the A/B fit runs, spelled as loop.yaml keys.
RECIPE_A = {
    "target_modules": "all-linear", "lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 5,
}
RECIPE_B = {
    "target_modules": "all-linear", "lora_r": 64, "lora_alpha": 128, "lr": 5e-4, "epochs": 8,
}
#: The same two as flags.
FULL_RECIPE = {**RECIPE_A, "quantization": "none"}
FLAGS_A = (
    "--sft-target-modules", "all-linear", "--sft-lora-r", "16", "--sft-lora-alpha", "32",
    "--sft-lr", "2e-4", "--sft-epochs", "5",
)
FLAGS_B = (
    "--sft-target-modules", "all-linear", "--sft-lora-r", "64", "--sft-lora-alpha", "128",
    "--sft-lr", "5e-4", "--sft-epochs", "8",
)
FLAGS_FULL = (*FLAGS_A, "--sft-quantization", "none")
_INIT = ("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base")
_LOOP_YAML = os.path.join(".soup", "loop.yaml")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _panel_text(text: str) -> str:
    """Output with the box characters removed, so a wrapped sentence reads whole."""
    return _clean(text.replace("│", " ").replace("|", " "))


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    for name in (
        "SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_SERVE_ENDPOINT",
        "SOUP_LOOP_FEEDBACK_LOG",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRACE_DIR_RESOLVER", "_TRAIN_RUNNER"):
        monkeypatch.setattr(ls, name, None)
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    return tmp_path


def _state(**kwargs) -> LoopState:
    kwargs.setdefault("served_model", "./model")
    kwargs.setdefault("base_model", "org/base")
    kwargs.setdefault("train_method", "sft")
    return LoopState(eval_suite="e", baseline="b", **kwargs)


def _render(state: LoopState) -> dict:
    return yaml.safe_load(ls._render_train_yaml(state, "d.jsonl", "out"))


def _cli(*args):
    from soup_cli.commands.loop import app

    result = CliRunner().invoke(app, list(args))
    return result, _clean(result.output), _panel_text(result.output)


def _recipe_row(panel: str) -> str:
    """The text of the status table's sft_recipe row, to the next row or the table's end."""
    match = re.search(r"sft_recipe (.*?)(?= verify | production | canary_traffic |$)", panel)
    assert match is not None, panel
    return match.group(1)


# ---------------------------------------------------------------------------
# The constant
# ---------------------------------------------------------------------------


class TestConstant:
    def test_it_holds_exactly_the_nine_keys(self):
        assert set(DEFAULT_SFT_RECIPE) == {
            "target_modules", "lora_r", "lora_alpha", "lr", "epochs", "batch_size",
            "gradient_accumulation_steps", "seed", "quantization",
        }
        assert set(SFT_RECIPE_KEYS) == set(DEFAULT_SFT_RECIPE)

    def test_it_is_read_only(self):
        assert isinstance(DEFAULT_SFT_RECIPE, MappingProxyType)
        with pytest.raises(TypeError):
            DEFAULT_SFT_RECIPE["lr"] = 0.5  # type: ignore[index]

    def test_every_key_passes_the_validator_that_guards_a_loop_yaml(self):
        assert validate_sft_recipe(dict(DEFAULT_SFT_RECIPE)) == dict(DEFAULT_SFT_RECIPE)

    def test_it_renders_to_a_config_soup_train_accepts(self, cwd):
        config = SoupConfig(**_render(_state()))

        assert config.task == "sft"
        assert config.training.lr == DEFAULT_SFT_RECIPE["lr"]
        assert config.training.epochs == DEFAULT_SFT_RECIPE["epochs"]
        assert config.training.lora.r == DEFAULT_SFT_RECIPE["lora_r"]
        assert config.training.lora.alpha == DEFAULT_SFT_RECIPE["lora_alpha"]

    def test_the_values_are_candidate_b(self):
        """The one pin: change it together with the constant."""
        assert dict(DEFAULT_SFT_RECIPE) == {
            "target_modules": "all-linear",
            "lora_r": 64,
            "lora_alpha": 128,
            "lr": 5e-4,
            "epochs": 8,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "seed": 42,
            "quantization": "auto",
        }

    def test_the_source_says_where_the_values_come_from_and_not_that_they_are_provisional(self):
        with open(loop_state.__file__, encoding="utf-8") as handle:
            source = handle.read()

        assert "# Fit runs, 2026-11-06: candidate B in bf16" in source
        assert "provisional" not in source.lower()


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


GOOD_VALUES = [
    ("target_modules", "all-linear"),
    ("target_modules", None),
    ("target_modules", ["q_proj"]),
    ("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]),
    ("target_modules", ["model.layers.0.self_attn.q_proj"]),
    ("lora_r", 1),
    ("lora_r", 1024),
    ("lora_alpha", 1),
    ("lora_alpha", 4096),
    ("lr", 1e-9),
    ("lr", 2e-4),
    ("lr", 1.0),
    ("lr", 1),
    ("epochs", 1),
    ("epochs", 100),
    ("quantization", "4bit"),
    ("quantization", "none"),
    ("quantization", "auto"),
]


class TestValidator:
    @pytest.mark.parametrize(("key", "value"), GOOD_VALUES)
    def test_a_valid_value_is_accepted(self, key, value):
        validate_sft_recipe_value(key, value)
        assert validate_sft_recipe({key: value}) == {key: value}

    @pytest.mark.parametrize("recipe", [None, {}])
    def test_nothing_to_override_is_valid(self, recipe):
        assert validate_sft_recipe(recipe) == recipe

    def test_both_candidate_recipes_are_valid(self):
        assert validate_sft_recipe(RECIPE_A) == RECIPE_A
        assert validate_sft_recipe(RECIPE_B) == RECIPE_B
        assert validate_sft_recipe(FULL_RECIPE) == FULL_RECIPE

    @pytest.mark.parametrize("name", ["lora_rank", "learning_rate", "Epochs", "", "dropout"])
    def test_an_unknown_key_is_refused_and_named(self, name):
        with pytest.raises(ValueError, match="unknown sft_recipe key") as err:
            validate_sft_recipe({"lr": 1e-4, name: 3})

        assert repr(name) in str(err.value)
        assert (
            "allowed: batch_size, epochs, gradient_accumulation_steps, lora_alpha, lora_r, lr, "
            "quantization, seed, target_modules"
            in str(err.value)
        )

    def test_every_unknown_key_is_listed_not_only_the_first(self):
        with pytest.raises(ValueError) as err:
            validate_sft_recipe({"aaa": 1, "bbb": 2})

        assert "'aaa'" in str(err.value) and "'bbb'" in str(err.value)

    @pytest.mark.parametrize("recipe", [[], "lr=1e-4", 5, (("lr", 1e-4),), True])
    def test_a_recipe_that_is_not_a_dict_is_refused(self, recipe):
        with pytest.raises(ValueError, match="sft_recipe must be a dict or None"):
            validate_sft_recipe(recipe)

    @pytest.mark.parametrize("value", [0, -1, 1025, 100000, 16.0, 16.5, "16", None, [16], True,
                                        False])
    def test_lora_r_is_an_int_from_1_to_1024(self, value):
        with pytest.raises(ValueError, match=r"lora_r must be an int in \[1, 1024\]"):
            validate_sft_recipe_value("lora_r", value)

    @pytest.mark.parametrize("value", [0, -1, 4097, 10**9, 32.0, "32", None, [32], True, False])
    def test_lora_alpha_is_an_int_from_1_to_4096(self, value):
        with pytest.raises(ValueError, match=r"lora_alpha must be an int in \[1, 4096\]"):
            validate_sft_recipe_value("lora_alpha", value)

    @pytest.mark.parametrize("value", [0, 0.0, -1e-4, 1.0000001, 2, 100, float("nan"),
                                        float("inf"), float("-inf"), "1e-4", None, [1e-4], True,
                                        False])
    def test_lr_is_a_finite_number_above_0_and_at_most_1(self, value):
        with pytest.raises(ValueError, match=r"lr must be a number in \(0, 1\]"):
            validate_sft_recipe_value("lr", value)

    @pytest.mark.parametrize("value", [0, -1, 101, 10**6, 5.0, "5", None, [5], True, False])
    def test_epochs_is_an_int_from_1_to_100(self, value):
        with pytest.raises(ValueError, match=r"epochs must be an int in \[1, 100\]"):
            validate_sft_recipe_value("epochs", value)

    @pytest.mark.parametrize(
        "value",
        ["8bit", "gptq", "fp8", "AUTO", "None", "4-bit", "nf4", "", " auto", None, True, False,
         4, 0, ["4bit"], {"4bit": 1}],
    )
    def test_quantization_is_4bit_none_or_auto(self, value):
        with pytest.raises(ValueError, match='quantization must be one of "4bit", "none", "auto"'):
            validate_sft_recipe_value("quantization", value)

    @pytest.mark.parametrize(
        "value",
        [
            "q_proj", "auto", "ALL-LINEAR", "all_linear", "", [], [""], [" "], [1], [None],
            ["q_proj", None], ["q proj"], ["q_proj,v_proj"], ["q_proj\x00"], [["q_proj"]],
            ["x" * 129], ["q_proj"] * 65, ("q_proj",), {"q_proj": 1}, 5, True, ["q\nproj"],
            ["../q_proj"], ["q_proj;"],
        ],
    )
    def test_target_modules_is_all_linear_or_a_list_of_module_names(self, value):
        with pytest.raises(ValueError, match="target_modules must be"):
            validate_sft_recipe_value("target_modules", value)

    def test_all_linear_as_a_list_entry_says_to_use_the_string(self):
        with pytest.raises(ValueError, match='"all-linear" is a string'):
            validate_sft_recipe_value("target_modules", ["all-linear"])

    def test_the_value_validator_refuses_an_unknown_key_too(self):
        with pytest.raises(ValueError, match="unknown sft_recipe key"):
            validate_sft_recipe_value("rank", 8)

    def test_a_bad_value_in_a_recipe_is_refused_with_its_message(self):
        with pytest.raises(ValueError, match="lr must be a number"):
            validate_sft_recipe({"epochs": 5, "lr": 2})


# ---------------------------------------------------------------------------
# The state field
# ---------------------------------------------------------------------------


class TestStateField:
    def test_it_follows_the_thresholds_and_defaults_to_none(self):
        # The paraphraser fields (tests/test_loop_paraphrases.py) are the two after it.
        names = list(LoopState.__dataclass_fields__)
        assert names[names.index("min_new_rated") + 1] == "sft_recipe"
        assert _state().sft_recipe is None

    def test_a_valid_recipe_is_kept(self):
        assert _state(sft_recipe=RECIPE_A).sft_recipe == RECIPE_A

    def test_an_unknown_key_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="unknown sft_recipe key 'lora_rank'"):
            _state(sft_recipe={"lora_rank": 8})

    @pytest.mark.parametrize(
        ("recipe", "message"),
        [
            ({"lora_r": 0}, "lora_r must be"),
            ({"lora_r": 1025}, "lora_r must be"),
            ({"lora_alpha": 0}, "lora_alpha must be"),
            ({"lora_alpha": 4097}, "lora_alpha must be"),
            ({"lr": 0}, "lr must be"),
            ({"lr": 1.5}, "lr must be"),
            ({"lr": float("nan")}, "lr must be"),
            ({"epochs": 0}, "epochs must be"),
            ({"epochs": 101}, "epochs must be"),
            ({"epochs": True}, "epochs must be"),
            ({"lora_r": True}, "lora_r must be"),
            ({"target_modules": []}, "target_modules must be"),
            ({"target_modules": "q_proj"}, "target_modules must be"),
            ({"quantization": "8bit"}, "quantization must be"),
            ({"quantization": None}, "quantization must be"),
            ({"quantization": True}, "quantization must be"),
        ],
    )
    def test_a_bad_value_is_refused_at_construction(self, recipe, message):
        with pytest.raises(ValueError, match=message):
            _state(sft_recipe=recipe)

    def test_a_recipe_that_is_not_a_dict_is_refused(self):
        with pytest.raises(ValueError, match="sft_recipe must be a dict or None"):
            _state(sft_recipe=["lr"])

    def test_it_survives_a_write_and_a_read(self, cwd):
        write_state(_state(sft_recipe={**RECIPE_A, "target_modules": ["q_proj", "v_proj"]}))

        reread = read_state()

        assert reread.sft_recipe == {**RECIPE_A, "target_modules": ["q_proj", "v_proj"]}

    def test_an_old_loop_yaml_loads_with_none(self, cwd):
        write_state(_state())
        with open(_LOOP_YAML, encoding="utf-8") as handle:
            data = json.load(handle)
        del data["sft_recipe"]
        with open(_LOOP_YAML, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        assert read_state().sft_recipe is None

    def test_a_loop_yaml_with_an_unknown_recipe_key_is_refused(self, cwd):
        write_state(_state())
        with open(_LOOP_YAML, encoding="utf-8") as handle:
            data = json.load(handle)
        data["sft_recipe"] = {"lr": 2e-4, "warmup": 3}
        with open(_LOOP_YAML, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        with pytest.raises(ValueError, match="state file has invalid contents.*'warmup'"):
            read_state()

    def test_a_loop_yaml_with_an_out_of_range_value_is_refused(self, cwd):
        write_state(_state())
        with open(_LOOP_YAML, encoding="utf-8") as handle:
            data = json.load(handle)
        data["sft_recipe"] = {"epochs": 1000}
        with open(_LOOP_YAML, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        with pytest.raises(ValueError, match="state file has invalid contents.*epochs must be"):
            read_state()

    def test_a_loop_yaml_with_a_bool_for_a_number_is_refused(self, cwd):
        write_state(_state())
        with open(_LOOP_YAML, encoding="utf-8") as handle:
            data = json.load(handle)
        data["sft_recipe"] = {"lora_r": True}
        with open(_LOOP_YAML, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        with pytest.raises(ValueError, match="lora_r must be"):
            read_state()

    def test_replace_revalidates_it(self):
        with pytest.raises(ValueError, match="unknown sft_recipe key"):
            replace(_state(), sft_recipe={"nope": 1})

    def test_init_state_stores_it_only_when_given(self, cwd):
        plain, _ = init_state("./m", "e", "b", path=str(cwd / "a.json"))
        given, _ = init_state(
            "./m", "e", "b", sft_recipe={"lr": 3e-4}, path=str(cwd / "b.json"), force=True
        )

        assert plain.sft_recipe is None
        assert given.sft_recipe == {"lr": 3e-4}


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------


class TestEffectiveRecipe:
    def test_none_is_the_constant(self):
        assert effective_sft_recipe(None) == dict(DEFAULT_SFT_RECIPE)
        assert _state().effective_sft_recipe() == dict(DEFAULT_SFT_RECIPE)

    def test_a_partial_dict_overrides_only_the_keys_it_names(self):
        merged = _state(sft_recipe={"lr": 3e-4, "epochs": 2}).effective_sft_recipe()

        assert merged["lr"] == 3e-4 and merged["epochs"] == 2
        for key in ("target_modules", "lora_r", "lora_alpha", "quantization"):
            assert merged[key] == DEFAULT_SFT_RECIPE[key], key

    def test_a_full_dict_overrides_everything(self):
        full = {**FULL_RECIPE, "batch_size": 3, "gradient_accumulation_steps": 2, "seed": 7}

        assert _state(sft_recipe=full).effective_sft_recipe() == full

    def test_a_dict_without_the_pinned_keys_keeps_the_pinned_values(self):
        merged = _state(sft_recipe=FULL_RECIPE).effective_sft_recipe()

        assert merged == {
            **FULL_RECIPE, "batch_size": 2, "gradient_accumulation_steps": 4, "seed": 42,
        }

    def test_a_recipe_without_quantization_keeps_the_constants(self):
        merged = _state(sft_recipe=RECIPE_A).effective_sft_recipe()

        assert merged["quantization"] == DEFAULT_SFT_RECIPE["quantization"]

    def test_the_merge_follows_the_constant(self, monkeypatch):
        """A partial override keeps the constant's CURRENT value for the keys it leaves."""
        monkeypatch.setattr(
            loop_state, "DEFAULT_SFT_RECIPE",
            MappingProxyType({**RECIPE_A, "lr": 7e-4, "quantization": "4bit"}),
        )

        merged = _state(sft_recipe={"epochs": 9}).effective_sft_recipe()

        assert merged == {**RECIPE_A, "lr": 7e-4, "epochs": 9, "quantization": "4bit"}

    def test_an_explicit_none_for_target_modules_overrides_the_constant(self, monkeypatch):
        monkeypatch.setattr(
            loop_state, "DEFAULT_SFT_RECIPE",
            MappingProxyType({**RECIPE_A, "quantization": "auto"}),
        )

        merged = _state(sft_recipe={"target_modules": None}).effective_sft_recipe()

        assert merged["target_modules"] is None
        assert merged["lora_r"] == 16

    def test_the_result_is_a_copy(self):
        state = _state(sft_recipe={"target_modules": ["q_proj"]})
        merged = state.effective_sft_recipe()

        merged["target_modules"].append("v_proj")
        merged["lr"] = 0.5

        assert state.sft_recipe == {"target_modules": ["q_proj"]}
        assert state.effective_sft_recipe()["lr"] == DEFAULT_SFT_RECIPE["lr"]
        assert DEFAULT_SFT_RECIPE["lr"] != 0.5


# ---------------------------------------------------------------------------
# The rendered config
# ---------------------------------------------------------------------------


class TestRender:
    def test_a_loop_without_a_recipe_renders_the_constant(self, cwd):
        training = _render(_state())["training"]

        assert training["lr"] == DEFAULT_SFT_RECIPE["lr"]
        assert training["epochs"] == DEFAULT_SFT_RECIPE["epochs"]
        assert training["batch_size"] == DEFAULT_SFT_RECIPE["batch_size"] == 2
        assert training["gradient_accumulation_steps"] == 4
        assert training["seed"] == 42
        assert training["lora"]["r"] == DEFAULT_SFT_RECIPE["lora_r"]
        assert training["lora"]["alpha"] == DEFAULT_SFT_RECIPE["lora_alpha"]
        assert ("target_modules" in training["lora"]) == (
            DEFAULT_SFT_RECIPE["target_modules"] is not None
        )
        # "auto" is resolved to a value; with no CUDA device (the suite's default) it is 4bit.
        assert DEFAULT_SFT_RECIPE["quantization"] == "auto"
        assert training["quantization"] == "4bit"

    def test_a_none_target_leaves_the_key_out_so_the_schema_default_applies(self, cwd):
        state = _state(sft_recipe={"target_modules": None})

        rendered = _render(state)
        config = SoupConfig(**rendered)

        assert "target_modules" not in rendered["training"]["lora"]
        assert config.training.lora.target_modules == "auto"

    def test_all_linear_is_written_and_accepted(self, cwd):
        rendered = _render(_state(sft_recipe={"target_modules": "all-linear"}))
        config = SoupConfig(**rendered)

        assert rendered["training"]["lora"]["target_modules"] == "all-linear"
        assert config.training.lora.target_modules == "all-linear"

    def test_an_explicit_module_list_is_written_and_accepted(self, cwd):
        modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        rendered = _render(_state(sft_recipe={"target_modules": modules}))
        config = SoupConfig(**rendered)

        assert rendered["training"]["lora"]["target_modules"] == modules
        assert config.training.lora.target_modules == modules

    def test_the_rendered_list_is_not_the_states_list(self, cwd):
        state = _state(sft_recipe={"target_modules": ["q_proj"]})
        rendered = yaml.safe_load(ls._render_train_yaml(state, "d.jsonl", "out"))

        rendered["training"]["lora"]["target_modules"].append("x")

        assert state.sft_recipe == {"target_modules": ["q_proj"]}

    def test_lr_and_epochs_are_the_recipes(self, cwd):
        rendered = _render(_state(sft_recipe={"lr": 3e-4, "epochs": 2}))
        config = SoupConfig(**rendered)

        assert (rendered["training"]["lr"], rendered["training"]["epochs"]) == (3e-4, 2)
        assert (config.training.lr, config.training.epochs) == (3e-4, 2)

    def test_auto_quantization_is_resolved_not_left_for_the_schema(self, cwd):
        """No CUDA device here, so ``auto`` is 4bit (the rule: test_loop_sft_quantization)."""
        rendered = _render(_state(sft_recipe={"quantization": "auto"}))
        config = SoupConfig(**rendered)

        assert rendered["training"]["quantization"] == "4bit"
        assert config.training.quantization == "4bit"

    @pytest.mark.parametrize("value", ["4bit", "none"])
    def test_an_explicit_quantization_is_written_and_accepted(self, cwd, value):
        rendered = _render(_state(sft_recipe={"quantization": value}))
        config = SoupConfig(**rendered)

        assert rendered["training"]["quantization"] == value
        assert config.training.quantization == value
        assert config.training.lora.r == DEFAULT_SFT_RECIPE["lora_r"]

    @pytest.mark.parametrize("quantization", ["auto", "4bit", "none"])
    @pytest.mark.parametrize("target", ["all-linear", ["q_proj", "v_proj"], None])
    def test_every_quantization_and_target_combination_is_a_config_soup_train_accepts(
        self, cwd, quantization, target
    ):
        recipe = {**RECIPE_B, "quantization": quantization, "target_modules": target}

        config = SoupConfig(**_render(_state(sft_recipe=recipe)))

        assert config.task == "sft"
        assert config.training.lora.alpha == 128

    def test_an_int_lr_is_written_as_a_float(self, cwd):
        rendered = _render(_state(sft_recipe={"lr": 1}))

        assert rendered["training"]["lr"] == 1.0
        assert isinstance(rendered["training"]["lr"], float)

    def test_lora_r_and_alpha_are_the_recipes(self, cwd):
        rendered = _render(_state(sft_recipe={"lora_r": 8, "lora_alpha": 24}))
        config = SoupConfig(**rendered)

        assert rendered["training"]["lora"]["r"] == 8
        assert rendered["training"]["lora"]["alpha"] == 24
        assert (config.training.lora.r, config.training.lora.alpha) == (8, 24)

    def test_a_partial_recipe_keeps_the_constant_for_the_rest(self, cwd):
        training = _render(_state(sft_recipe={"lr": 3e-4}))["training"]

        assert training["lr"] == 3e-4
        assert training["epochs"] == DEFAULT_SFT_RECIPE["epochs"]
        assert training["lora"]["r"] == DEFAULT_SFT_RECIPE["lora_r"]
        assert training["lora"]["alpha"] == DEFAULT_SFT_RECIPE["lora_alpha"]

    @pytest.mark.parametrize("name", ["A", "B"])
    def test_the_two_candidate_recipes_render_to_configs_soup_train_accepts(self, cwd, name):
        recipe = {"A": RECIPE_A, "B": RECIPE_B}[name]
        rendered = _render(_state(sft_recipe=recipe))
        config = SoupConfig(**rendered)

        assert config.task == "sft"
        assert config.data.val_split == 0
        assert config.training.lora.target_modules == "all-linear"
        assert config.training.lora.r == recipe["lora_r"]
        assert config.training.lora.alpha == recipe["lora_alpha"]
        assert config.training.lr == recipe["lr"]
        assert config.training.epochs == recipe["epochs"]

    @pytest.mark.parametrize(("method", "batch"), [("dpo", 1), ("kto", 2)])
    def test_dpo_and_kto_ignore_a_recipe(self, cwd, method, batch):
        state = replace(_state(sft_recipe=FULL_RECIPE), train_method=method)

        rendered = _render(state)

        assert rendered["training"] == {"epochs": 1, "batch_size": batch, "lr": 5.0e-5}
        SoupConfig(**rendered)

    def test_the_train_stage_writes_the_recipe_into_the_config_soup_train_reads(
        self, cwd, monkeypatch
    ):
        configs = []

        def trainer(argv, **kwargs):
            with open(argv[argv.index("--config") + 1], encoding="utf-8") as handle:
                configs.append(yaml.safe_load(handle))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        (cwd / "rows.jsonl").write_text('{"messages": []}\n', "utf-8")
        ctx = {"pairs_harvested": 4, "new_pairs": 4, "pairs_path": "rows.jsonl"}

        out = ls.train_dpo_from_pairs(_state(sft_recipe=RECIPE_A), ctx)

        assert out["skipped"] is False, out
        (config,) = configs
        assert config["training"]["lora"] == {
            "r": 16, "alpha": 32, "target_modules": "all-linear"
        }
        assert (config["training"]["lr"], config["training"]["epochs"]) == (2e-4, 5)


# ---------------------------------------------------------------------------
# soup loop init
# ---------------------------------------------------------------------------


class TestInitFlags:
    def test_without_flags_nothing_is_stored(self, cwd):
        result, _, _ = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().sft_recipe is None

    def test_each_flag_stores_only_its_own_key(self, cwd):
        cases = [
            ("--sft-lora-r", "32", {"lora_r": 32}),
            ("--sft-lora-alpha", "48", {"lora_alpha": 48}),
            ("--sft-lr", "3e-4", {"lr": 3e-4}),
            ("--sft-epochs", "7", {"epochs": 7}),
            ("--sft-target-modules", "all-linear", {"target_modules": "all-linear"}),
            ("--sft-quantization", "none", {"quantization": "none"}),
        ]
        for flag, value, expected in cases:
            result, _, _ = _cli(*_INIT, flag, value, "--force")
            assert result.exit_code == 0, (flag, result.output, repr(result.exception))
            assert read_state().sft_recipe == expected, flag

    def test_all_six_flags_store_the_whole_recipe(self, cwd):
        result, _, _ = _cli(*_INIT, *FLAGS_FULL)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().sft_recipe == FULL_RECIPE

    def test_five_flags_store_five_keys_and_no_quantization(self, cwd):
        result, _, _ = _cli(*_INIT, *FLAGS_A)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().sft_recipe == RECIPE_A

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("4bit", "4bit"), ("none", "none"), ("auto", "auto"), ("NONE", "none"),
         (" 4BIT ", "4bit"), ("Auto", "auto")],
    )
    def test_quantization_is_4bit_none_or_auto_in_any_case(self, cwd, raw, expected):
        result, _, _ = _cli(*_INIT, "--sft-quantization", raw)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().sft_recipe == {"quantization": expected}

    @pytest.mark.parametrize(
        "raw", ["8bit", "gptq", "nf4", "fast", "", "   ", "4 bit", "none,auto"]
    )
    def test_a_bad_quantization_is_refused_naming_the_flag(self, cwd, raw):
        result, text, _ = _cli(*_INIT, "--sft-quantization", raw)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "invalid --sft-quantization:" in text
        assert 'quantization must be one of "4bit", "none", "auto"' in text
        assert not os.path.exists(_LOOP_YAML)

    @pytest.mark.parametrize("name", ["A", "B"])
    @pytest.mark.parametrize("quantization", ["auto", "4bit", "none"])
    def test_a_candidate_recipe_with_each_quantization_renders_a_config_soup_train_accepts(
        self, cwd, name, quantization
    ):
        flags = {"A": FLAGS_A, "B": FLAGS_B}[name]
        result, _, _ = _cli(*_INIT, *flags, "--sft-quantization", quantization)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        config = SoupConfig(**_render(read_state()))

        assert config.training.quantization == ("4bit" if quantization == "auto" else quantization)

    def test_the_flags_work_with_the_default_method_and_with_sft_named(self, cwd):
        result, _, _ = _cli(*_INIT, "--train-method", "SFT", "--sft-epochs", "3")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        state = read_state()
        assert state.train_method == "sft" and state.sft_recipe == {"epochs": 3}

    @pytest.mark.parametrize("name", ["A", "B"])
    def test_a_candidate_recipe_from_the_flags_renders_a_config_soup_train_accepts(
        self, cwd, name
    ):
        flags = {"A": FLAGS_A, "B": FLAGS_B}[name]
        recipe = {"A": RECIPE_A, "B": RECIPE_B}[name]
        result, _, _ = _cli(*_INIT, *flags)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        config = SoupConfig(**_render(read_state()))

        assert config.training.lora.target_modules == "all-linear"
        assert (config.training.lora.r, config.training.lora.alpha) == (
            recipe["lora_r"], recipe["lora_alpha"]
        )
        assert (config.training.lr, config.training.epochs) == (recipe["lr"], recipe["epochs"])

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("all-linear", "all-linear"),
            ("ALL-LINEAR", "all-linear"),
            (" All-Linear ", "all-linear"),
            ("auto", None),
            (" AUTO ", None),
            ("q_proj", ["q_proj"]),
            ("q_proj,v_proj", ["q_proj", "v_proj"]),
            (" q_proj , v_proj ", ["q_proj", "v_proj"]),
            ("q_proj,k_proj,v_proj,o_proj", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ],
    )
    def test_target_modules_is_all_linear_auto_or_a_comma_separated_list(
        self, cwd, raw, expected
    ):
        """``auto`` is the schema default: stored as None, so the config names no modules."""
        result, _, _ = _cli(*_INIT, "--sft-target-modules", raw)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().sft_recipe == {"target_modules": expected}

    @pytest.mark.parametrize(
        "raw", ["", "   ", ",", "q_proj,", ",q_proj", "q_proj,,v_proj", "q_proj, ,v_proj"]
    )
    def test_an_empty_entry_is_refused_naming_the_flag_and_the_forms_that_work(self, cwd, raw):
        result, text, _ = _cli(*_INIT, "--sft-target-modules", raw)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "invalid --sft-target-modules:" in text
        assert "target_modules has an empty entry" in text
        assert "all-linear, auto, or a comma-separated list such as q_proj,v_proj" in text
        assert not os.path.exists(_LOOP_YAML)

    @pytest.mark.parametrize(
        "raw",
        ["all-linear,q_proj", "q proj", "q_proj;v_proj", "x" * 129,
         ",".join(["q_proj"] * 65), "../q_proj", "q_proj.", "model..q_proj"],
    )
    def test_a_bad_target_modules_is_refused_naming_the_flag(self, cwd, raw):
        result, text, _ = _cli(*_INIT, "--sft-target-modules", raw)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "invalid --sft-target-modules:" in text
        assert "target_modules must be" in text
        assert not os.path.exists(_LOOP_YAML)

    @pytest.mark.parametrize(
        ("flag", "value", "message"),
        [
            ("--sft-lora-r", "0", "lora_r must be an int in [1, 1024]"),
            ("--sft-lora-r", "-4", "lora_r must be an int in [1, 1024]"),
            ("--sft-lora-r", "1025", "lora_r must be an int in [1, 1024]"),
            ("--sft-lora-alpha", "0", "lora_alpha must be an int in [1, 4096]"),
            ("--sft-lora-alpha", "-1", "lora_alpha must be an int in [1, 4096]"),
            ("--sft-lora-alpha", "4097", "lora_alpha must be an int in [1, 4096]"),
            ("--sft-lr", "0", "lr must be a number in (0, 1]"),
            ("--sft-lr", "-1e-4", "lr must be a number in (0, 1]"),
            ("--sft-lr", "1.5", "lr must be a number in (0, 1]"),
            ("--sft-lr", "nan", "lr must be a number in (0, 1]"),
            ("--sft-lr", "inf", "lr must be a number in (0, 1]"),
            ("--sft-epochs", "0", "epochs must be an int in [1, 100]"),
            ("--sft-epochs", "-3", "epochs must be an int in [1, 100]"),
            ("--sft-epochs", "101", "epochs must be an int in [1, 100]"),
        ],
    )
    def test_a_value_out_of_range_is_refused_naming_the_flag(self, cwd, flag, value, message):
        result, text, _ = _cli(*_INIT, flag, value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert f"invalid {flag}:" in text
        assert message in text
        assert not os.path.exists(_LOOP_YAML)

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--sft-lora-r", "x"), ("--sft-lora-r", "16.5"), ("--sft-lora-alpha", "many"),
            ("--sft-lr", "fast"), ("--sft-epochs", "five"), ("--sft-epochs", "2.5"),
        ],
    )
    def test_a_value_that_is_not_a_number_is_refused_naming_the_flag(self, cwd, flag, value):
        result, text, _ = _cli(*_INIT, flag, value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert flag in text
        assert not os.path.exists(_LOOP_YAML)

    @pytest.mark.parametrize("method", ["dpo", "kto"])
    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--sft-target-modules", "all-linear"), ("--sft-lora-r", "16"),
            ("--sft-lora-alpha", "32"), ("--sft-lr", "2e-4"), ("--sft-epochs", "5"),
            ("--sft-quantization", "none"),
        ],
    )
    def test_a_flag_is_refused_for_a_loop_that_does_not_train_sft(
        self, cwd, method, flag, value
    ):
        result, text, _ = _cli(*_INIT, "--train-method", method, flag, value)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert flag in text
        assert "--train-method sft" in text
        assert method in text
        assert not os.path.exists(_LOOP_YAML)

    def test_the_refusal_names_every_flag_given(self, cwd):
        result, text, _ = _cli(
            *_INIT, "--train-method", "dpo", "--sft-lr", "2e-4", "--sft-epochs", "5"
        )

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--sft-lr" in text and "--sft-epochs" in text
        assert "--sft-lora-r" not in text

    def test_a_non_sft_loop_without_the_flags_is_unchanged(self, cwd):
        result, _, _ = _cli(*_INIT, "--train-method", "dpo")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().sft_recipe is None

    def test_a_refused_init_leaves_an_existing_loop_yaml_alone(self, cwd):
        _cli(*_INIT, "--sft-epochs", "3")
        before = Path(_LOOP_YAML).read_bytes()

        result, _, _ = _cli(*_INIT, "--sft-epochs", "0", "--force")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert Path(_LOOP_YAML).read_bytes() == before

    def test_the_user_input_in_a_refusal_cannot_restyle_the_terminal(self, cwd):
        result, text, _ = _cli(*_INIT, "--sft-target-modules", "[bold red]x[/],\x1b[31m")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "\x1b[31m" not in result.output
        assert "invalid --sft-target-modules:" in text
        assert "[bold red]x[/]" in text  # printed as typed, not read as markup

    def test_help_names_the_six_flags(self, cwd):
        result, _, panel = _cli("init", "--help")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        for flag in (
            "--sft-target-modules", "--sft-lora-r", "--sft-lora-alpha", "--sft-lr",
            "--sft-epochs", "--sft-quantization",
        ):
            assert flag in panel, flag

    def test_the_init_panel_says_which_recipe_keys_came_from_the_flags(self, cwd):
        result, _, panel = _cli(*_INIT, "--sft-lr", "2e-4", "--sft-epochs", "7")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert "sft recipe from loop.yaml:" in panel
        assert "lr 0.0002" in panel and "epochs 7" in panel
        assert "lora_r" not in panel

    def test_the_init_panel_says_nothing_about_a_recipe_when_none_was_given(self, cwd):
        result, _, panel = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert "sft recipe from loop.yaml" not in panel


# ---------------------------------------------------------------------------
# soup loop status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_an_sft_loop_shows_the_constant_in_one_row(self, cwd):
        _cli(*_INIT)

        result, _, panel = _cli("status")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        row = _recipe_row(panel)
        assert panel.count("sft_recipe") == 1
        assert f"lora_r {DEFAULT_SFT_RECIPE['lora_r']}" in row
        assert f"lora_alpha {DEFAULT_SFT_RECIPE['lora_alpha']}" in row
        assert f"lr {DEFAULT_SFT_RECIPE['lr']:g}" in row
        assert f"epochs {DEFAULT_SFT_RECIPE['epochs']}" in row
        assert "target_modules" in row
        assert "quantization auto (none if the base fits in free VRAM, else 4bit)" in row
        assert "all built-in defaults" in row
        assert "from loop.yaml" not in row

    def test_a_target_modules_of_none_says_it_is_the_schema_default(self, cwd):
        write_state(_state(sft_recipe={"target_modules": None}))

        _, _, panel = _cli("status")

        assert "target_modules auto (schema default)" in _recipe_row(panel)

    def test_the_effective_values_and_the_keys_from_loop_yaml_are_shown(self, cwd):
        _cli(*_INIT, "--sft-lr", "2e-4", "--sft-lora-r", "16")

        result, _, panel = _cli("status")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        row = _recipe_row(panel)
        assert "lr 0.0002" in row and "lora_r 16" in row
        assert f"epochs {DEFAULT_SFT_RECIPE['epochs']}" in row
        assert f"lora_alpha {DEFAULT_SFT_RECIPE['lora_alpha']}" in row
        assert "from loop.yaml: lora_r, lr" in row
        assert "the rest built-in defaults" in row
        assert "all built-in defaults" not in row

    def test_a_recipe_naming_every_key_says_so(self, cwd):
        _cli(
            *_INIT, *FLAGS_B, "--sft-batch-size", "2", "--sft-grad-accum", "4",
            "--sft-seed", "42", "--sft-quantization", "4bit",
        )

        _, _, panel = _cli("status")

        row = _recipe_row(panel)
        assert "target_modules all-linear" in row
        assert "lora_r 64" in row and "lora_alpha 128" in row
        assert "lr 0.0005" in row and "epochs 8" in row
        assert "batch_size 2" in row and "gradient_accumulation_steps 4" in row
        assert "seed 42" in row
        assert "quantization 4bit" in row
        assert "all from loop.yaml" in row

    def test_a_quantization_from_the_flag_is_shown_and_named(self, cwd):
        _cli(*_INIT, "--sft-quantization", "none")

        _, _, panel = _cli("status")

        row = _recipe_row(panel)
        assert "quantization none" in row
        assert "from loop.yaml: quantization;" in row

    def test_auto_from_the_flag_is_shown_as_the_schema_default_and_as_from_loop_yaml(self, cwd):
        _cli(*_INIT, "--sft-target-modules", "auto")

        _, _, panel = _cli("status")

        row = _recipe_row(panel)
        assert "target_modules auto (schema default)" in row
        assert "from loop.yaml: target_modules" in row

    def test_a_module_list_is_shown_in_full(self, cwd):
        _cli(*_INIT, "--sft-target-modules", "q_proj,v_proj")

        _, _, panel = _cli("status")

        assert "target_modules q_proj, v_proj" in _recipe_row(panel)

    @pytest.mark.parametrize("method", ["dpo", "kto"])
    def test_a_loop_that_does_not_train_sft_has_no_recipe_row(self, cwd, method):
        _cli(*_INIT, "--train-method", method)

        result, _, panel = _cli("status")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert "sft_recipe" not in panel

    def test_a_loop_yaml_written_before_the_field_shows_the_constant(self, cwd):
        write_state(_state())
        with open(_LOOP_YAML, encoding="utf-8") as handle:
            data = json.load(handle)
        del data["sft_recipe"]
        with open(_LOOP_YAML, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        result, _, panel = _cli("status")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert "all built-in defaults" in _recipe_row(panel)

    def test_a_loop_yaml_with_a_bad_recipe_stops_status_with_exit_2(self, cwd):
        write_state(_state())
        with open(_LOOP_YAML, encoding="utf-8") as handle:
            data = json.load(handle)
        data["sft_recipe"] = {"epochs": 0}
        with open(_LOOP_YAML, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        result, text, _ = _cli("status")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "loop state invalid" in text and "epochs must be" in text
