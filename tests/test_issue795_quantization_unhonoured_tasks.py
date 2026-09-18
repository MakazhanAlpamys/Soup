"""#795: ``training.quantization`` defaults to ``4bit`` and six trainers never read it.

``distill``, ``classifier`` / ``reranker`` / ``cross_encoder``, ``prm``,
``moe_lora_routing``, ``unlearn`` and ``asr`` load the base at checkpoint precision
whatever ``quantization`` says. The schema accepted ``4bit`` (the default) and an
explicit ``8bit`` for all of them, the stored config recorded a quantisation that
never happened, and the VRAM pre-flight budgeted e.g. a full-precision PRM fine-tune
as a ~4 GB QLoRA job -- under-prediction, the unsafe direction.

Because the default is ``4bit``, an UNSET ``quantization`` resolves to ``none`` for
these tasks so a minimal config still parses and the stored config is truthful.

An explicit bitsandbytes value -- ``4bit``, ``8bit``, or the ``load_in_8bit`` alias --
loads with a warning and resolves to ``none`` (#1043 review): every config Soup
dumped on the old default carries ``quantization: 4bit`` literally, and refusing it
broke files Soup wrote itself. A quant-menu value was never a default, so it is
still refused.
"""

import io

import pytest
from rich.console import Console

from soup_cli.config import loader
from soup_cli.config.deprecation import SoupConfigDeprecationWarning
from soup_cli.config.loader import load_config_from_string

_MINIMAL = {
    "distill": ("", "  teacher_model: org/teacher\n"),
    "classifier": ("", "  num_labels: 2\n"),
    "reranker": ("", "  num_labels: 1\n"),
    "cross_encoder": ("", "  num_labels: 1\n"),
    "prm": ("", ""),
    "moe_lora_routing": ("", "  mole_task_adapters: [./a1, ./a2]\n"),
    "unlearn": ("  forget_set: forget.jsonl\n", "  unlearn_method: npo\n"),
    "asr": ("", ""),
}
_UNHONOURED = sorted(_MINIMAL)


def _yaml(task, training_extra="", *, base="org/model"):
    data_extra, training = _MINIMAL[task]
    body = training + training_extra
    return (
        f"base: {base}\ntask: {task}\n"
        f"data:\n  train: x.jsonl\n{data_extra}"
        + (f"training:\n{body}" if body else "")
    )


class TestTheDefaultResolves:
    @pytest.mark.parametrize("task", _UNHONOURED)
    def test_an_unset_quantization_is_none(self, task):
        """On ``main`` every one of these loaded as ``4bit``: a stored config, and
        the registry entry and config hash built from it, described a quantised run
        that never happened."""
        cfg = load_config_from_string(_yaml(task))
        assert cfg.training.quantization == "none"

    @pytest.mark.parametrize("task", _UNHONOURED)
    def test_an_explicit_none_is_accepted(self, task):
        cfg = load_config_from_string(_yaml(task, "  quantization: none\n"))
        assert cfg.training.quantization == "none"

    @pytest.mark.parametrize("task", _UNHONOURED)
    def test_the_resolved_config_survives_dump_and_reload(self, task):
        """#321's warning about fields-set gates: a dumped config writes the value
        out explicitly. It must carry ``none`` so the reload is not refused."""
        import yaml

        cfg = load_config_from_string(_yaml(task))
        reloaded = load_config_from_string(yaml.safe_dump(cfg.model_dump(mode="json")))
        assert reloaded.training.quantization == "none"


@pytest.fixture
def printed(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(
        loader, "console", Console(file=buffer, width=500, force_terminal=False, color_system=None)
    )
    return buffer


class TestAnExplicitBnbValueWarnsAndResolves:
    @pytest.mark.parametrize("task", _UNHONOURED)
    @pytest.mark.parametrize("quant", ["4bit", "8bit"])
    def test_bnb_values_name_the_task(self, task, quant, printed):
        """Loads, says so once naming the task and the value, and trains as none."""
        cfg = load_config_from_string(_yaml(task, f"  quantization: {quant}\n"))
        assert cfg.training.quantization == "none"
        out = printed.getvalue()
        assert out.count(f"training.quantization: {quant} has no effect") == 1, out
        assert f"task='{task}'" in out
        assert "will refuse it" in out

    @pytest.mark.parametrize("task", ["prm", "distill"])
    def test_the_warning_is_a_deprecation_warning_outside_the_loader(self, task):
        """Code that builds SoupConfig directly sees an ordinary warning."""
        import yaml

        from soup_cli.config.schema import SoupConfig

        raw = yaml.safe_load(_yaml(task, "  quantization: 4bit\n"))
        with pytest.warns(SoupConfigDeprecationWarning, match=f"task='{task}'"):
            cfg = SoupConfig(**raw)
        assert cfg.training.quantization == "none"

    @pytest.mark.parametrize("task", _UNHONOURED)
    def test_an_unset_value_does_not_warn(self, task, printed):
        """Control: only an explicit value warns. The default resolves quietly."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", SoupConfigDeprecationWarning)
            cfg = load_config_from_string(_yaml(task))
        assert cfg.training.quantization == "none"
        assert printed.getvalue() == ""

    @pytest.mark.parametrize("task", ["prm", "unlearn"])
    def test_the_load_in_8bit_alias_warns_too(self, task, printed):
        """``load_in_8bit: true`` rewrites ``quantization`` to ``8bit``; for these
        tasks that is the same unhonoured value by another spelling."""
        cfg = load_config_from_string(_yaml(task, "  load_in_8bit: true\n"))
        assert cfg.training.quantization == "none"
        assert f"task='{task}'" in printed.getvalue()

    @pytest.mark.parametrize("task", _UNHONOURED)
    @pytest.mark.parametrize("exclude_none", [False, True], ids=["full", "exclude_none"])
    def test_a_config_dumped_on_the_old_default_still_loads(self, task, exclude_none, printed):
        """The regression that revised the ruling: a run-config dump, or an
        autopilot-style ``exclude_none`` dump, written while ``4bit`` was the
        default carries it literally. Built here by dumping and writing the old
        value back, which is what those files contain."""
        import yaml

        from soup_cli.config.schema import SoupConfig

        dumped = SoupConfig(**yaml.safe_load(_yaml(task))).model_dump(
            mode="json", exclude_none=exclude_none
        )
        dumped["training"]["quantization"] = "4bit"
        cfg = load_config_from_string(yaml.safe_dump(dumped, sort_keys=False))
        assert cfg.training.quantization == "none"
        assert printed.getvalue().count("has no effect") == 1


class TestAnExplicitValueIsRefused:

    @pytest.mark.parametrize("task", ["prm", "distill"])
    def test_a_quant_menu_value_is_refused_too(self, task):
        with pytest.raises(ValueError, match=f"task='{task}'"):
            load_config_from_string(_yaml(task, "  quantization: gptq\n"))

    @pytest.mark.parametrize(
        "knob",
        [
            "  bnb_4bit_quant_storage: bfloat16\n",
            "  bnb_4bit_use_double_quant: true\n",
            "  llm_int8: true\n",
        ],
    )
    def test_a_bnb_only_setting_is_not_left_meaning_nothing(self, knob):
        """A 4-bit/8-bit-only knob on an unset quantization must not parse into a
        ``none`` run where the knob silently does nothing. ``bnb_4bit_quant_storage``
        is validated inside TrainingConfig before the task is known, so it passed
        on the unresolved ``4bit`` default."""
        with pytest.raises(ValueError):
            load_config_from_string(_yaml("prm", knob))

    def test_explicit_4bit_with_a_4bit_storage_setting_is_still_refused(self):
        """Only the bare value Soup's dumps carry is forgiven. ``bnb_4bit_quant_storage``
        was never written by a dump (it defaults to ``None``), so setting it beside
        ``4bit`` is a deliberate 4-bit request the trainer cannot honour."""
        with pytest.raises(ValueError, match="task='prm'"):
            load_config_from_string(
                _yaml("prm", "  quantization: 4bit\n  bnb_4bit_quant_storage: bfloat16\n")
            )

    @pytest.mark.parametrize("task", ["prm", "asr"])
    def test_load_in_16bit_is_accepted(self, task):
        cfg = load_config_from_string(_yaml(task, "  load_in_16bit: true\n"))
        assert cfg.training.quantization == "none"


class TestTheResolverRunsFirst:
    def test_it_is_the_first_after_validator(self):
        """Every later SoupConfig validator that reads ``quantization`` must see the
        resolved ``none``. In particular #1037's
        ``_validate_peft_variant_backend_and_quantization`` would refuse a distill +
        LoftQ config with ``quantization`` unset as 4-bit if it ran first. The order
        lived only in a docstring; this pins it, so a conflict resolution that
        pastes a validator above this one fails here (#1043 review)."""
        from soup_cli.config.schema import SoupConfig

        validators = SoupConfig.__pydantic_decorators__.model_validators
        after = [name for name, dec in validators.items() if dec.info.mode == "after"]
        assert after, "SoupConfig has no after-validators to order"
        assert after[0] == "_resolve_quantization_for_unhonouring_tasks", after[:3]


class TestHonouringTasksAreUnchanged:
    """Controls: the gate must not reach the trainers that DO read the field."""

    @pytest.mark.parametrize("task", ["sft", "dpo", "pretrain", "embedding"])
    def test_default_stays_4bit(self, task):
        cfg = load_config_from_string(f"base: org/model\ntask: {task}\ndata: {{train: x.jsonl}}\n")
        assert cfg.training.quantization == "4bit"

    @pytest.mark.parametrize("quant", ["4bit", "8bit"])
    def test_explicit_values_still_load_on_sft(self, quant):
        cfg = load_config_from_string(
            f"base: org/model\ntask: sft\ndata: {{train: x.jsonl}}\n"
            f"training: {{quantization: {quant}}}\n"
        )
        assert cfg.training.quantization == quant


class TestPRMLora:
    def test_an_explicit_lora_block_is_refused(self):
        """``prm.py`` never reads ``lora.*``: every base parameter trains, so the
        block is decorative."""
        with pytest.raises(ValueError, match="task='prm'.*lora"):
            load_config_from_string(_yaml("prm", "  lora:\n    r: 16\n"))

    def test_r_zero_states_what_prm_does_and_is_allowed(self):
        cfg = load_config_from_string(_yaml("prm", "  lora:\n    r: 0\n"))
        assert cfg.training.lora.r == 0

    def test_the_schema_default_lora_is_not_intent(self):
        cfg = load_config_from_string(_yaml("prm"))
        assert cfg.task == "prm"

    def test_a_dumped_prm_config_still_reloads(self):
        """A dumped config writes the default ``lora`` block out; that is not intent
        and must not be refused on reload."""
        import yaml

        cfg = load_config_from_string(_yaml("prm"))
        load_config_from_string(yaml.safe_dump(cfg.model_dump(mode="json")))

    @pytest.mark.parametrize("task", ["classifier", "asr"])
    def test_tasks_that_read_lora_keep_it(self, task):
        """classifier_lora / asr_lora read ``training.lora`` for r/alpha/dropout."""
        cfg = load_config_from_string(_yaml(task, "  lora:\n    r: 16\n"))
        assert cfg.training.lora.r == 16


class TestThePreflight:
    def test_prm_with_an_int_batch_size_is_not_predicted_as_qlora(self):
        """The issue's own case: Qwen2.5-1.5B PRM, batch 4, predicted as a ~4 GB
        QLoRA job while a full-precision fine-tune trains."""
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = load_config_from_string(
            _yaml("prm", "  batch_size: 4\n", base="Qwen/Qwen2.5-1.5B-Instruct")
        )
        fit = _build_hardware_fit_input(cfg)
        assert fit is not None
        assert fit.quant == "none"
        assert fit.peft == "full", "PRM fine-tunes every parameter; LoRA under-predicts"

    @pytest.mark.parametrize(
        "task, extra, expected",
        [
            ("classifier", "", "full"),
            ("classifier", "  classifier_lora: true\n", "lora"),
            ("asr", "", "full"),
            ("asr", "  asr_lora: true\n", "lora"),
        ],
    )
    def test_opt_in_lora_tasks_follow_their_own_flag(self, task, extra, expected):
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = load_config_from_string(
            _yaml(task, "  batch_size: 4\n" + extra, base="Qwen/Qwen2.5-1.5B-Instruct")
        )
        fit = _build_hardware_fit_input(cfg)
        assert fit is not None and fit.peft == expected

    def test_sft_default_is_still_qlora(self):
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = load_config_from_string(
            "base: Qwen/Qwen2.5-1.5B-Instruct\ntask: sft\ndata: {train: x.jsonl}\n"
            "training: {batch_size: 4}\n"
        )
        fit = _build_hardware_fit_input(cfg)
        assert fit is not None and fit.peft == "qlora"


class TestShippedConfigs:
    def test_every_recipe_still_loads(self):
        """#983's lesson: a gate is only safe if nothing Soup ships trips it. The
        three ASR recipes set ``quantization: none``."""
        from soup_cli.recipes.catalog import RECIPES

        for recipe in RECIPES.values():
            load_config_from_string(recipe.yaml_str)

    def test_soup_draft_distill_config_loads_as_none(self):
        from soup_cli.commands.draft import _build_distill_config_yaml

        text = _build_distill_config_yaml(
            draft_base="org/draft", target="org/target", data="d.jsonl",
            out_dir="out", steps=10, data_rows=100,
        )
        assert load_config_from_string(text).training.quantization == "none"

    def test_soup_shrink_heal_config_loads_as_none(self):
        """It never sets ``quantization``: on ``main`` it stored ``4bit``."""
        from soup_cli.commands.shrink import _build_heal_config_yaml

        text = _build_heal_config_yaml(
            pruned_dir="pruned", teacher="org/teacher", heal_data="h.jsonl",
            steps=10, out_dir="out", heal_rows=100,
        )
        assert "quantization" not in text
        assert load_config_from_string(text).training.quantization == "none"
