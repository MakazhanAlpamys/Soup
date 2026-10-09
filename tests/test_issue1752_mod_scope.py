"""#1752 — ``training.use_mod`` is refused at load where nothing installs a router.

The flag loaded on all 23 tasks. Two call sites install a Mixture-of-Depths
router: the LoRA arm of ``SFTTrainerWrapper._setup_transformers`` (``tts``
reaches it through ``super()``) and ``PretrainTrainerWrapper._setup_transformers``
after its LISA/LoRA split. Everywhere else the run trained without one and
said nothing. Those configs now stop at load.

Nothing is trained here: the gate is a config validator.
"""

from __future__ import annotations

import typing

import pytest
import yaml

from soup_cli.config.loader import load_config, load_config_from_string
from soup_cli.config.schema import MOD_APPLYING_TASKS, SoupConfig

_TASKS = tuple(typing.get_args(SoupConfig.model_fields["task"].annotation))
# Written out, not read from MOD_APPLYING_TASKS: a task dropped from or added
# to the constant must not move its own tests with it.
_APPLYING = ("sft", "tts", "pretrain")
_REFUSED = tuple(task for task in _TASKS if task not in _APPLYING)

# The minimum each task needs to load at all (as in the #1697 tests).
_TOP = {"tts": {"modality": "audio_out"}}
_DATA = {
    "dpo": {"format": "dpo"}, "kto": {"format": "kto"}, "pretrain": {"format": "plaintext"},
    "unlearn": {"forget_set": "./f.jsonl"},
}
_TRAINING = {
    "preference": {"preference_loss": "dpo"},
    "tts": {"tts_family": "orpheus"},
    "classifier": {"num_labels": 2}, "reranker": {"num_labels": 2},
    "cross_encoder": {"num_labels": 2},
    "unlearn": {"unlearn_method": "npo"},
    "moe_lora_routing": {"mole_task_adapters": ["./a", "./b"]},
    "distill": {"teacher_model": "org/t"},
    "online_dpo": {"online_dpo_judge": "pairrm"},
}

_SFT_WHERE = (
    "on the resident text LoRA path with backend: transformers, "
    "no layer streaming, lora.r >= 1, and neither LISA nor Spectrum"
)
_TTS_WHERE = (
    "on the transformers LoRA path tts reaches through the SFT "
    "trainer (lora.r >= 1, not unsloth)"
)
_PRETRAIN_WHERE = "on backend: transformers, including a LISA run"

# Paths that load today and never reach the router call:
# (task, top-level keys, training keys, the setting named, the change, where).
_OFF_PATH = {
    "sft-mlx": (
        "sft", {"backend": "mlx"}, {},
        "backend='mlx'", "backend: transformers", _SFT_WHERE,
    ),
    "sft-vision": (
        "sft", {"modality": "vision"}, {},
        "modality='vision'", "modality: text", _SFT_WHERE,
    ),
    "sft-audio": (
        "sft", {"modality": "audio"}, {},
        "modality='audio'", "modality: text", _SFT_WHERE,
    ),
    "sft-stream": (
        "sft", {}, {"stream_layers": True, "batch_size": 1},
        "training.stream_layers=true", "stream_layers: false", _SFT_WHERE,
    ),
    "sft-unsloth": (
        "sft", {"backend": "unsloth"}, {},
        "backend='unsloth'", "backend: transformers", _SFT_WHERE,
    ),
    "sft-full-ft": (
        "sft", {}, {"quantization": "none", "lora": {"r": 0}},
        "training.lora.r=0", "lora.r >= 1", _SFT_WHERE,
    ),
    "sft-lisa": (
        "sft", {}, {"quantization": "none", "lisa_enabled": True},
        "training.lisa_enabled=true", "lisa_enabled: false", _SFT_WHERE,
    ),
    "sft-spectrum": (
        "sft", {}, {
            "quantization": "none",
            "unfrozen_parameters": ["model.layers.0.mlp.down_proj"],
        },
        "training.unfrozen_parameters", "unfrozen_parameters: null", _SFT_WHERE,
    ),
    "tts-unsloth": (
        "tts", {"backend": "unsloth"}, {},
        "backend='unsloth'", "backend: transformers", _TTS_WHERE,
    ),
    "tts-full-ft": (
        "tts", {}, {"lora": {"r": 0}},
        "training.lora.r=0", "lora.r >= 1", _TTS_WHERE,
    ),
    "pretrain-unsloth": (
        "pretrain", {"backend": "unsloth"}, {},
        "backend='unsloth'", "backend: transformers", _PRETRAIN_WHERE,
    ),
}


def _yaml(task="sft", top=None, **training):
    raw = {
        "base": "org/m", "task": task, **_TOP.get(task, {}), **(top or {}),
        "data": {"train": "./x.jsonl", **_DATA.get(task, {})},
        "training": {**_TRAINING.get(task, {}), **training},
    }
    return yaml.safe_dump(raw)


def _refusal(text):
    with pytest.raises(ValueError) as info:
        load_config_from_string(text)
    return " ".join(str(info.value).split())


def _assert_off_path(message, task, settings, where, change):
    assert (
        f"training.use_mod is not applied by task={task!r} with {settings}: "
        f"a Mixture-of-Depths router is installed only {where}, so"
    ) in message, message
    assert "would train without one" in message
    assert message.endswith(f"Use {change}, or set use_mod: false.")


def test_the_task_list_is_the_one_this_file_was_written_for():
    assert len(_TASKS) == 23
    assert set(_APPLYING) <= set(_TASKS)
    assert MOD_APPLYING_TASKS == frozenset({"sft", "tts", "pretrain"})


class TestRefusedOutsideTheThreeTasks:
    @pytest.mark.parametrize("task", _TASKS)
    def test_without_the_flag_the_task_still_loads(self, task):
        assert load_config_from_string(_yaml(task)).training.use_mod is False

    @pytest.mark.parametrize("task", _REFUSED)
    def test_the_message_names_the_field_the_task_and_the_fix(self, task):
        message = _refusal(_yaml(task, use_mod=True))

        assert (
            f"training.use_mod is not applied by task={task!r}: only "
            "task='sft', 'tts' and 'pretrain' install a Mixture-of-Depths router"
        ) in message, message
        assert "would train without one" in message
        assert message.endswith("Use one of those tasks, or set use_mod: false.")

    @pytest.mark.parametrize("backend", ["transformers", "unsloth"])
    def test_the_task_refusal_does_not_depend_on_the_backend(self, backend):
        """``dpo`` on unsloth is refused for the task, not for the path."""
        message = _refusal(_yaml("dpo", {"backend": backend}, use_mod=True))

        assert "is not applied by task='dpo': only task='sft'" in message, message

    def test_clearing_the_flag_loads(self):
        raw = yaml.safe_load(_yaml("dpo", use_mod=True))
        raw["training"]["use_mod"] = False

        assert load_config_from_string(yaml.safe_dump(raw)).task == "dpo"


class TestRefusedOffTheApplyingPath:
    @pytest.mark.parametrize("case", list(_OFF_PATH))
    def test_the_message_names_the_field_the_path_and_the_fix(self, case):
        task, top, training, setting, change, where = _OFF_PATH[case]

        message = _refusal(_yaml(task, top, **training, use_mod=True))

        _assert_off_path(message, task, setting, where, change)

    @pytest.mark.parametrize("case", list(_OFF_PATH))
    @pytest.mark.parametrize("flag", [None, False])
    def test_omitted_or_false_still_loads(self, case, flag):
        """The refusal is about the request, not about the path."""
        task, top, training, _, _, _ = _OFF_PATH[case]
        if flag is not None:
            training = {**training, "use_mod": flag}

        cfg = load_config_from_string(_yaml(task, top, **training))

        assert cfg.training.use_mod is False

    @pytest.mark.parametrize(
        ("top", "training", "settings", "change"),
        [
            (
                {"backend": "unsloth", "modality": "vision"},
                {},
                "modality='vision' and backend='unsloth'",
                "modality: text and backend: transformers",
            ),
            (
                {"backend": "unsloth", "modality": "audio"},
                {},
                "modality='audio' and backend='unsloth'",
                "modality: text and backend: transformers",
            ),
            (
                {"backend": "mlx", "modality": "vision"},
                {},
                "backend='mlx' and modality='vision'",
                "backend: transformers and modality: text",
            ),
        ],
    )
    def test_every_sft_setting_off_the_path_is_named(self, top, training, settings, change):
        """Changing one of two settings would still leave the run off the LoRA
        arm, so both are named, in the order the trainer checks them."""
        message = _refusal(_yaml("sft", top, **training, use_mod=True))

        _assert_off_path(message, "sft", settings, _SFT_WHERE, change)

    def test_tts_unsloth_and_full_finetune_are_named_together(self):
        """Both are legal on tts and both skip the inherited LoRA arm."""
        message = _refusal(_yaml("tts", {"backend": "unsloth"}, lora={"r": 0}, use_mod=True))

        _assert_off_path(
            message, "tts",
            "backend='unsloth' and training.lora.r=0",
            _TTS_WHERE,
            "backend: transformers and lora.r >= 1",
        )

    def test_following_the_sft_message_loads(self):
        raw = yaml.safe_load(
            _yaml("sft", {"modality": "vision", "backend": "unsloth"}, use_mod=True)
        )
        raw["modality"] = "text"
        raw["backend"] = "transformers"

        cfg = load_config_from_string(yaml.safe_dump(raw))

        assert cfg.training.use_mod is True
        assert cfg.modality == "text"
        assert cfg.backend == "transformers"


class TestTheApplyingPathsStillLoad:
    @pytest.mark.parametrize("task", ["sft", "tts", "pretrain"])
    def test_transformers_lora(self, task):
        cfg = load_config_from_string(
            _yaml(task, {"backend": "transformers"}, use_mod=True)
        )

        assert cfg.training.use_mod is True

    def test_pretrain_lisa(self):
        cfg = load_config_from_string(
            _yaml("pretrain", quantization="none", lisa_enabled=True, use_mod=True)
        )

        assert cfg.training.use_mod is True
        assert cfg.training.lisa_enabled is True

    @pytest.mark.parametrize("modality", ["vision", "audio"])
    def test_pretrain_modality_is_not_an_sft_gate(self, modality):
        """``PretrainTrainerWrapper.setup`` branches on the backend only."""
        cfg = load_config_from_string(
            _yaml("pretrain", {"modality": modality}, use_mod=True)
        )

        assert cfg.training.use_mod is True

    def test_pretrain_stream_layers_false_is_not_streaming(self):
        cfg = load_config_from_string(
            _yaml("pretrain", stream_layers=False, use_mod=True)
        )

        assert cfg.training.use_mod is True

    def test_pretrain_lora_r_zero_is_not_the_sft_full_finetune_gate(self):
        """Pretrain has no ``lora.r == 0`` branch; the router call still runs."""
        cfg = load_config_from_string(
            _yaml("pretrain", quantization="none", lora={"r": 0}, use_mod=True)
        )

        assert cfg.training.lora.r == 0
        assert cfg.training.use_mod is True

    def test_sft_stream_layers_false_is_not_streaming(self):
        cfg = load_config_from_string(_yaml("sft", stream_layers=False, use_mod=True))

        assert cfg.training.use_mod is True

    def test_capacity_factor_still_loads_on_an_applying_path(self):
        cfg = load_config_from_string(
            _yaml("sft", use_mod=True, mod_capacity_factor=0.25)
        )

        assert cfg.training.mod_capacity_factor == pytest.approx(0.25)


class TestAlreadyRefusedForAnotherReason:
    """These never reached the new check, before or after. They are listed so
    a baseline failure is not reported as a MoD refusal."""

    @pytest.mark.parametrize(
        ("text", "needle"),
        [
            (
                _yaml("pretrain", {"backend": "mlx"}, use_mod=True),
                "MLX backend only ships SFT",
            ),
            (
                _yaml("tts", {"backend": "mlx"}, use_mod=True),
                "not supported on backend=mlx",
            ),
            (
                _yaml("pretrain", stream_layers=True, batch_size=1, use_mod=True),
                "training.stream_layers supports task",
            ),
            (
                _yaml("tts", stream_layers=True, batch_size=1, use_mod=True),
                "training.stream_layers supports task",
            ),
            (
                _yaml("sft", stream_layers=True, use_mod=True),
                "batch_size='auto'",
            ),
            (
                _yaml("sft", lisa_enabled=True, use_mod=True),
                "quantization='none'",
            ),
            (
                _yaml(
                    "sft", {"backend": "unsloth"},
                    quantization="none", lora={"r": 0}, use_mod=True,
                ),
                "full fine-tuning",
            ),
            (
                _yaml(
                    "tts", quantization="none", lisa_enabled=True, use_mod=True,
                ),
                "training.lisa_enabled",
            ),
            (
                _yaml("dpo", use_mod=True, mod_capacity_factor=2.0),
                "mod_capacity_factor",
            ),
            (
                _yaml("sft", use_mod=True, mod_capacity_factor=2.0),
                "mod_capacity_factor",
            ),
        ],
    )
    def test_the_earlier_error_is_not_replaced(self, text, needle):
        message = _refusal(text)

        assert needle in message, message
        assert "Mixture-of-Depths" not in message


class TestDumpAndFileLoad:
    def test_a_default_dump_round_trips(self):
        dumped = yaml.safe_dump(
            load_config_from_string(_yaml("sft")).model_dump(mode="json")
        )

        assert "use_mod: false" in dumped
        assert load_config_from_string(dumped).training.use_mod is False

    def test_load_config_exits_on_a_refused_file(self, tmp_path, monkeypatch):
        from soup_cli.config import loader

        printed: list[str] = []

        def record(*args, **kwargs):
            printed.append(str(args[0]) if args else "")

        monkeypatch.setattr(loader.console, "print", record)
        path = tmp_path / "soup.yaml"
        path.write_text(_yaml("dpo", use_mod=True), encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)

        assert excinfo.value.code == 1
        joined = " ".join(" ".join(printed).split())
        assert "training.use_mod is not applied by task='dpo'" in joined

    def test_load_config_accepts_an_applying_file(self, tmp_path):
        path = tmp_path / "soup.yaml"
        path.write_text(_yaml("tts", use_mod=True), encoding="utf-8")

        cfg = load_config(path)

        assert cfg.task == "tts"
        assert cfg.training.use_mod is True


class TestValidatorOrder:
    def test_it_sits_with_the_route_checks_and_before_the_backend_refusal(self):
        """#1357's backend refusal stays last. Earlier task gates stay in
        front, so they keep their own errors."""
        validators = SoupConfig.__pydantic_decorators__.model_validators
        after = [name for name, dec in validators.items() if dec.info.mode == "after"]

        assert (
            after.index("_validate_freeze_is_applied")
            < after.index("_validate_rope_scaling_is_applied")
            < after.index("_validate_use_mod_is_applied")
            < after.index("_validate_unsloth_has_a_setup")
        ), after[-5:]
        assert after[-1] == "_validate_unsloth_has_a_setup"
        for earlier in (
            "_validate_tts_compat",
            "_validate_unfrozen_parameters",
            "_validate_lisa_compat",
            "_validate_full_finetune",
            "_validate_stream_layers_compat",
            "_validate_mlx_task_support",
        ):
            assert after.index(earlier) < after.index("_validate_use_mod_is_applied")
