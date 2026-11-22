"""The loop's sft recipe pins the micro-batch, gradient accumulation and seed.

A live run trained on byte-identical rows but inherited ``batch_size`` 1, an accumulation of
the schema's own default and no seed: 152 optimizer steps instead of 80 at lr 5e-4, and quality
fell (format_json 1.00 to 0.775). So the sft recipe carries three more keys, written into every
rendered config whatever else is configured, so nothing is inherited from a default:

- ``batch_size`` 2 (``training.batch_size``, the per-device micro-batch);
- ``gradient_accumulation_steps`` 4 (``training.gradient_accumulation_steps``);
- ``seed`` 42 (``training.seed``).

Each is validated like the other keys (a positive int; the seed an int from 0, the schema's
bound; a bool refused), overridable in ``loop.yaml`` and through ``--sft-batch-size`` /
``--sft-grad-accum`` / ``--sft-seed`` (refused for a loop that does not train sft), and shown
in ``soup loop status``. dpo and kto are unchanged.
"""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.config.schema import SoupConfig, TrainingConfig
from soup_cli.utils.loop_state import (
    DEFAULT_SFT_RECIPE,
    SFT_RECIPE_KEYS,
    LoopState,
    read_state,
    validate_sft_recipe,
    validate_sft_recipe_value,
    write_state,
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

#: The three pinned values, under the names of the SoupConfig fields they reach.
PINNED = {"batch_size": 2, "gradient_accumulation_steps": 4, "seed": 42}
_INIT = ("init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--base-model", "org/base")
_LOOP_YAML = os.path.join(".soup", "loop.yaml")
_SEED_MAX = 2**32 - 1


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


def _panel_text(text: str) -> str:
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
    match = re.search(r"sft_recipe (.*?)(?= verify | production | canary_traffic |$)", panel)
    assert match is not None, panel
    return match.group(1)


# ---------------------------------------------------------------------------
# The constant
# ---------------------------------------------------------------------------


class TestConstant:
    @pytest.mark.parametrize(("key", "value"), sorted(PINNED.items()))
    def test_it_pins_each_value(self, key, value):
        assert DEFAULT_SFT_RECIPE[key] == value
        assert isinstance(DEFAULT_SFT_RECIPE[key], int)
        assert not isinstance(DEFAULT_SFT_RECIPE[key], bool)

    @pytest.mark.parametrize("key", sorted(PINNED))
    def test_the_key_is_a_recipe_key_and_a_field_of_the_training_config(self, key):
        assert key in SFT_RECIPE_KEYS
        assert key in TrainingConfig.model_fields

    def test_the_recipe_keys_and_the_constant_agree(self):
        assert set(SFT_RECIPE_KEYS) == set(DEFAULT_SFT_RECIPE)

    def test_the_whole_constant_still_passes_the_validator(self):
        assert validate_sft_recipe(dict(DEFAULT_SFT_RECIPE)) == dict(DEFAULT_SFT_RECIPE)


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


class TestValidator:
    @pytest.mark.parametrize("key", ["batch_size", "gradient_accumulation_steps"])
    @pytest.mark.parametrize("value", [1, 2, 8, 1024])
    def test_a_positive_int_is_accepted(self, key, value):
        validate_sft_recipe_value(key, value)
        assert validate_sft_recipe({key: value}) == {key: value}

    @pytest.mark.parametrize("key", ["batch_size", "gradient_accumulation_steps"])
    @pytest.mark.parametrize(
        "value",
        [0, -1, -8, 1025, 10**9, 2.0, 1.5, "2", "auto", None, True, False, [2], {"a": 1}],
    )
    def test_anything_but_a_positive_int_is_refused(self, key, value):
        with pytest.raises(ValueError, match=f"^{key} must be an int in \\[1, 1024\\]"):
            validate_sft_recipe_value(key, value)

    @pytest.mark.parametrize("value", [0, 1, 42, 123456789, _SEED_MAX])
    def test_a_seed_from_zero_to_the_schemas_bound_is_accepted(self, value):
        validate_sft_recipe_value("seed", value)
        assert validate_sft_recipe({"seed": value}) == {"seed": value}

    @pytest.mark.parametrize(
        "value",
        [-1, -42, _SEED_MAX + 1, 10**12, 1.5, 42.0, "42", None, True, False, [42]],
    )
    def test_anything_but_an_int_from_zero_is_refused_as_a_seed(self, value):
        with pytest.raises(ValueError, match=f"^seed must be an int in \\[0, {_SEED_MAX}\\]"):
            validate_sft_recipe_value("seed", value)

    def test_a_bool_is_no_int_for_any_of_the_three(self):
        for key in PINNED:
            for value in (True, False):
                with pytest.raises(ValueError, match=f"^{key} must be"):
                    validate_sft_recipe({key: value})

    def test_the_unknown_key_message_lists_the_new_keys(self):
        with pytest.raises(ValueError, match="unknown sft_recipe key") as err:
            validate_sft_recipe({"warmup": 3})

        for key in PINNED:
            assert key in str(err.value)

    def test_every_bound_the_loop_accepts_is_one_the_config_accepts(self):
        config = TrainingConfig(
            batch_size=1024, gradient_accumulation_steps=1024, seed=_SEED_MAX
        )
        assert (config.batch_size, config.gradient_accumulation_steps, config.seed) == (
            1024, 1024, _SEED_MAX
        )
        assert TrainingConfig(seed=0).seed == 0


# ---------------------------------------------------------------------------
# loop.yaml
# ---------------------------------------------------------------------------


class TestLoopYaml:
    def test_a_recipe_with_the_keys_round_trips(self, cwd):
        recipe = {"batch_size": 3, "gradient_accumulation_steps": 2, "seed": 7}
        write_state(_state(sft_recipe=recipe))

        assert read_state().sft_recipe == recipe

    @pytest.mark.parametrize(
        ("recipe", "message"),
        [
            ({"batch_size": 0}, "batch_size must be an int"),
            ({"batch_size": True}, "batch_size must be an int"),
            ({"gradient_accumulation_steps": -1}, "gradient_accumulation_steps must be an int"),
            ({"gradient_accumulation_steps": "4"}, "gradient_accumulation_steps must be an int"),
            ({"seed": -1}, "seed must be an int"),
            ({"seed": True}, "seed must be an int"),
            ({"seed": 2**32}, "seed must be an int"),
        ],
    )
    def test_a_hand_edited_loop_yaml_with_a_bad_value_is_refused(self, cwd, recipe, message):
        write_state(_state())
        with open(_LOOP_YAML, encoding="utf-8") as handle:
            data = json.load(handle)
        data["sft_recipe"] = recipe
        with open(_LOOP_YAML, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        with pytest.raises(ValueError, match=f"state file has invalid contents.*{message}"):
            read_state()

    def test_a_loop_yaml_from_before_the_keys_loads_with_the_pinned_values(self, cwd):
        write_state(_state(sft_recipe={"lr": 2e-4}))

        effective = read_state().effective_sft_recipe()

        assert {key: effective[key] for key in PINNED} == PINNED


# ---------------------------------------------------------------------------
# The rendered config
# ---------------------------------------------------------------------------

#: Other recipe keys changed, in every way the recipe allows.
OTHER_RECIPES = [
    None,
    {"lr": 3e-4, "epochs": 2},
    {"target_modules": "all-linear", "lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 5},
    {"target_modules": ["q_proj", "v_proj"], "lora_r": 8, "lora_alpha": 16,
     "quantization": "4bit"},
    {"quantization": "none", "epochs": 1},
    {"target_modules": None, "lora_r": 128, "lora_alpha": 256, "lr": 1e-4, "quantization": "auto"},
]
#: ``min_new_*`` at their defaults and far from them.
MIN_NEW = [
    {},
    {"min_new_pairs": 7, "min_new_edits": 3, "min_new_rated": 11},
    {"min_new_pairs": 50, "min_new_edits": 100, "min_new_rated": 1},
]


class TestRender:
    def test_the_default_recipe_writes_the_three_values_explicitly(self, cwd):
        training = _render(_state())["training"]

        for key, value in PINNED.items():
            assert key in training, f"{key} must be written, not inherited from the schema"
            assert training[key] == value

    @pytest.mark.parametrize("min_new", MIN_NEW, ids=["default", "low-edits", "high-edits"])
    @pytest.mark.parametrize("recipe", OTHER_RECIPES)
    def test_the_rendered_config_carries_exactly_the_pinned_values_whatever_else_is_set(
        self, cwd, recipe, min_new
    ):
        rendered = _render(_state(sft_recipe=recipe, **min_new))
        config = SoupConfig(**rendered)

        assert rendered["training"]["batch_size"] == 2
        assert rendered["training"]["gradient_accumulation_steps"] == 4
        assert rendered["training"]["seed"] == 42
        assert config.training.batch_size == 2
        assert config.training.gradient_accumulation_steps == 4
        assert config.training.seed == 42
        assert config.task == "sft"

    @pytest.mark.parametrize("min_new", MIN_NEW, ids=["default", "low-edits", "high-edits"])
    def test_the_train_stage_writes_them_into_the_config_soup_train_reads(
        self, cwd, monkeypatch, min_new
    ):
        configs = []

        def trainer(argv, **kwargs):
            with open(argv[argv.index("--config") + 1], encoding="utf-8") as handle:
                configs.append(yaml.safe_load(handle))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(ls, "_TRAIN_RUNNER", trainer)
        (cwd / "rows.jsonl").write_text('{"messages": []}\n', "utf-8")
        ctx = {"pairs_harvested": 60, "new_pairs": 60, "pairs_path": "rows.jsonl"}
        state = _state(sft_recipe={"lr": 3e-4, "lora_r": 8, "lora_alpha": 16}, **min_new)

        out = ls.train_dpo_from_pairs(state, ctx)

        assert out["skipped"] is False, out
        (config,) = configs
        assert {key: config["training"][key] for key in PINNED} == PINNED
        SoupConfig(**config)

    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            ({"batch_size": 1}, {"batch_size": 1, "gradient_accumulation_steps": 4, "seed": 42}),
            ({"gradient_accumulation_steps": 8},
             {"batch_size": 2, "gradient_accumulation_steps": 8, "seed": 42}),
            ({"seed": 7}, {"batch_size": 2, "gradient_accumulation_steps": 4, "seed": 7}),
            ({"seed": 0}, {"batch_size": 2, "gradient_accumulation_steps": 4, "seed": 0}),
            ({"batch_size": 4, "gradient_accumulation_steps": 1, "seed": _SEED_MAX},
             {"batch_size": 4, "gradient_accumulation_steps": 1, "seed": _SEED_MAX}),
        ],
    )
    def test_a_key_loop_yaml_names_overrides_only_itself(self, cwd, override, expected):
        rendered = _render(_state(sft_recipe=override))
        config = SoupConfig(**rendered)

        for key, value in expected.items():
            assert rendered["training"][key] == value
            assert getattr(config.training, key) == value

    def test_the_effective_batch_is_eight_and_the_steps_follow_from_it(self, cwd):
        config = SoupConfig(**_render(_state()))

        effective = config.training.batch_size * config.training.gradient_accumulation_steps

        assert effective == 8

    @pytest.mark.parametrize(("method", "batch"), [("dpo", 1), ("kto", 2)])
    def test_dpo_and_kto_keep_their_own_settings(self, cwd, method, batch):
        state = _state(
            train_method=method, sft_recipe=None, min_new_pairs=7, min_new_edits=3
        )

        training = _render(state)["training"]

        assert training == {"epochs": 1, "batch_size": batch, "lr": 5.0e-5}


# ---------------------------------------------------------------------------
# soup loop init
# ---------------------------------------------------------------------------

_FLAGS = (
    ("--sft-batch-size", "batch_size", "3", 3),
    ("--sft-grad-accum", "gradient_accumulation_steps", "2", 2),
    ("--sft-seed", "seed", "7", 7),
)


class TestInitFlags:
    @pytest.mark.parametrize(("flag", "key", "raw", "value"), _FLAGS)
    def test_each_flag_stores_its_key_and_only_that_key(self, cwd, flag, key, raw, value):
        result, _, _ = _cli(*_INIT, flag, raw)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().sft_recipe == {key: value}

    def test_the_three_flags_together_store_the_three_keys(self, cwd):
        result, _, _ = _cli(
            *_INIT, "--sft-batch-size", "1", "--sft-grad-accum", "16", "--sft-seed", "0"
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().sft_recipe == {
            "batch_size": 1, "gradient_accumulation_steps": 16, "seed": 0
        }

    def test_no_flag_stores_no_recipe(self, cwd):
        result, _, _ = _cli(*_INIT)
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert read_state().sft_recipe is None

    def test_the_flags_render_a_config_soup_train_accepts(self, cwd):
        result, _, _ = _cli(
            *_INIT, "--sft-batch-size", "4", "--sft-grad-accum", "2", "--sft-seed", "123"
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

        config = SoupConfig(**_render(read_state()))

        assert (
            config.training.batch_size,
            config.training.gradient_accumulation_steps,
            config.training.seed,
        ) == (4, 2, 123)

    @pytest.mark.parametrize(
        ("flag", "key", "raw"),
        [
            ("--sft-batch-size", "batch_size", "0"),
            ("--sft-batch-size", "batch_size", "-2"),
            ("--sft-batch-size", "batch_size", "1025"),
            ("--sft-grad-accum", "gradient_accumulation_steps", "0"),
            ("--sft-grad-accum", "gradient_accumulation_steps", "-1"),
            ("--sft-grad-accum", "gradient_accumulation_steps", "5000"),
            ("--sft-seed", "seed", "-1"),
            ("--sft-seed", "seed", str(_SEED_MAX + 1)),
        ],
    )
    def test_a_value_out_of_range_is_refused_naming_the_flag(self, cwd, flag, key, raw):
        result, text, _ = _cli(*_INIT, flag, raw)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert f"invalid {flag}:" in text
        assert f"{key} must be an int" in text
        assert not os.path.exists(_LOOP_YAML)

    @pytest.mark.parametrize(
        ("flag", "raw"),
        [("--sft-batch-size", "x"), ("--sft-grad-accum", "1.5"), ("--sft-seed", "auto")],
    )
    def test_a_value_that_is_not_an_int_is_refused_naming_the_flag(self, cwd, flag, raw):
        result, text, _ = _cli(*_INIT, flag, raw)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert flag in text
        assert not os.path.exists(_LOOP_YAML)

    @pytest.mark.parametrize("method", ["dpo", "kto"])
    @pytest.mark.parametrize(("flag", "key", "raw", "value"), _FLAGS)
    def test_a_flag_is_refused_for_a_loop_that_does_not_train_sft(
        self, cwd, method, flag, key, raw, value
    ):
        result, text, _ = _cli(*_INIT, "--train-method", method, flag, raw)

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert flag in text
        assert "--train-method sft" in text
        assert not os.path.exists(_LOOP_YAML)

    def test_the_refusal_names_every_new_flag_given(self, cwd):
        result, text, _ = _cli(
            *_INIT, "--train-method", "dpo", "--sft-batch-size", "2", "--sft-seed", "1"
        )

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--sft-batch-size" in text and "--sft-seed" in text
        assert "--sft-grad-accum" not in text

    def test_help_names_the_three_flags(self, cwd):
        result, _, panel = _cli("init", "--help")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        for flag in ("--sft-batch-size", "--sft-grad-accum", "--sft-seed"):
            assert flag in panel, flag

    def test_the_init_panel_says_which_keys_came_from_the_flags(self, cwd):
        result, _, panel = _cli(*_INIT, "--sft-seed", "9", "--sft-grad-accum", "3")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        assert "sft recipe from loop.yaml:" in panel
        assert "gradient_accumulation_steps 3" in panel and "seed 9" in panel
        assert "batch_size" not in panel


# ---------------------------------------------------------------------------
# soup loop status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_the_default_row_shows_the_three_pinned_values(self, cwd):
        _cli(*_INIT)

        result, _, panel = _cli("status")
        assert result.exit_code == 0, (result.output, repr(result.exception))

        row = _recipe_row(panel)
        assert "batch_size 2" in row
        assert "gradient_accumulation_steps 4" in row
        assert "seed 42" in row
        assert "all built-in defaults" in row

    def test_an_override_is_shown_and_named_as_from_loop_yaml(self, cwd):
        _cli(*_INIT, "--sft-batch-size", "3", "--sft-seed", "7")

        _, _, panel = _cli("status")

        row = _recipe_row(panel)
        assert "batch_size 3" in row and "seed 7" in row
        assert "gradient_accumulation_steps 4" in row
        assert "from loop.yaml: batch_size, seed;" in row
