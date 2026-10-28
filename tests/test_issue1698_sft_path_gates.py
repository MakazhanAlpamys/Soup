"""#1698: the four sft fields are refused on the paths that never read them.

#983 refused ``use_flash_attn`` / ``use_liger`` outside ``{sft, tts}`` and #1587
did the same for ``reasoning_effort`` / ``train_on_eot`` outside
``{sft, tts, distill}``, both on the task name alone. Inside ``task: sft`` the
trainer takes one of five model setups and one of six dataset branches
(``SFTTrainerWrapper.setup``), and most of them read none of the four fields, so
the config loaded, the run trained, and nothing said the setting was dropped.
Those cells now stop at load, naming the field, the path and what to change.

The cells come from one table, ``soup_cli.config.schema.SFT_PATHS``, which
``_validate_freeze_is_applied`` reads too (#1581, #1649). Streaming keeps the
kernel flags refused by #1692, and MLX keeps their own warning, so neither is
duplicated here.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SFT_PATH_FIELDS, SFT_PATHS
from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
from soup_cli.trainer.sft import SFTTrainerWrapper

_FIELD_VALUE = {
    "reasoning_effort": "high",
    "train_on_eot": True,
    "use_flash_attn": True,
    "use_liger": True,
}

#: Each path that drops a field: (top-level keys, data keys, the setting the
#: message names, the change it asks for, a phrase from its reason).
_PATHS = {
    "vision": (
        {"modality": "vision"}, {}, "modality='vision'", "modality: text",
        "the vision setup builds its own dataset and model",
    ),
    "audio": (
        {"modality": "audio"}, {}, "modality='audio'", "modality: text",
        "the audio setup builds its own dataset and model",
    ),
    "raft": (
        {}, {"format": "raft"}, "data.format='raft'", "data.format: chatml",
        "RAFT rows are tokenised by the RAFT builder",
    ),
    "pre_tokenized": (
        {}, {"format": "pre_tokenized", "tokenized_path": "./tok"},
        "data.format='pre_tokenized'", "data.format: chatml",
        "rows are read from the pre-tokenised cache",
    ),
    "mlx": (
        {"backend": "mlx"}, {}, "backend='mlx'", "backend: transformers",
        "the MLX trainer has its own loop",
    ),
    "unsloth": (
        {"backend": "unsloth"}, {}, "backend='unsloth'", "backend: transformers",
        "the unsloth setup loads its own model",
    ),
}

#: The cells the issue marks "no", one entry per path and field.
_REFUSED = {
    "vision": ["reasoning_effort", "train_on_eot", "use_flash_attn", "use_liger"],
    "audio": ["reasoning_effort", "train_on_eot", "use_flash_attn", "use_liger"],
    "raft": ["reasoning_effort", "train_on_eot"],
    "pre_tokenized": ["reasoning_effort"],
    "mlx": ["reasoning_effort", "train_on_eot"],
    "unsloth": ["use_flash_attn", "use_liger"],
}


def _yaml(top=None, data=None, **training):
    raw = {"base": "org/m", "task": "sft", **(top or {})}
    raw["data"] = {"train": "./x.jsonl", **(data or {})}
    if raw["task"] == "tts":
        raw.setdefault("modality", "audio_out")
        training = {"tts_family": "orpheus", **training}
    if raw["task"] == "distill":
        training = {"teacher_model": "org/teacher", **training}
    raw["training"] = training
    return yaml.safe_dump(raw)


def _refusal(text):
    with pytest.raises(ValueError) as info:
        load_config_from_string(text)
    return " ".join(str(info.value).split())


def _cells():
    return [(path, field) for path, fields in _REFUSED.items() for field in fields]


class TestEveryNoCellIsRefused:
    @pytest.mark.parametrize(("path", "field"), _cells())
    def test_the_message_names_the_field_the_path_and_the_fix(self, path, field):
        top, data, setting, change, reason = _PATHS[path]

        message = _refusal(_yaml(top, data, **{field: _FIELD_VALUE[field]}))

        assert f"training.{field} is not applied by task='sft' with {setting}:" in message, message
        assert reason in message, message
        assert f"Use {change}, or remove the key." in message, message

    @pytest.mark.parametrize("path", list(_PATHS))
    def test_without_the_field_the_path_still_loads(self, path):
        """The refusal is about the field, not about the path."""
        top, data, _, _, _ = _PATHS[path]

        cfg = load_config_from_string(_yaml(top, data))

        assert cfg.task == "sft"

    def test_the_issue_vision_reproduction_is_refused(self):
        message = _refusal(
            "base: Qwen/Qwen2-VL-2B-Instruct\ntask: sft\nmodality: vision\n"
            "data: {train: data.jsonl, format: chatml}\n"
            "training: {use_liger: true}\n"
        )

        assert "with modality='vision'" in message, message

    @pytest.mark.parametrize("path", list(_PATHS))
    def test_every_field_the_path_drops_is_named_in_one_refusal(self, path):
        top, data, setting, change, _ = _PATHS[path]
        fields = _REFUSED[path]
        flags = {field: _FIELD_VALUE[field] for field in fields}

        message = _refusal(_yaml(top, data, **flags))

        for field in fields:
            assert f"training.{field}" in message, message
        assert f"with {setting}:" in message, message
        assert f"Use {change}, or remove" in message, message
        if len(fields) > 1:
            assert "are not applied by task='sft'" in message, message

    def test_every_setting_is_named_in_the_order_the_setup_checks_them(self):
        """Changing one setting would still leave the run off the text path."""
        message = _refusal(
            _yaml({"backend": "unsloth", "modality": "vision"}, use_liger=True)
        )

        assert (
            "training.use_liger is not applied by task='sft' with "
            "modality='vision' and backend='unsloth':"
        ) in message, message
        assert "Use modality: text and backend: transformers, or remove the key." in message


class TestTheControlsStillLoad:
    def test_the_text_path_reads_all_four(self):
        cfg = load_config_from_string(
            _yaml({"backend": "transformers"}, **_FIELD_VALUE)
        )

        assert cfg.training.reasoning_effort == "high"
        assert cfg.training.use_flash_attn is True
        assert cfg.training.use_liger is True
        assert cfg.training.train_on_eot is True

    @pytest.mark.parametrize("backend", ["unsloth"])
    def test_unsloth_reads_the_formatter_fields(self, backend):
        cfg = load_config_from_string(
            _yaml({"backend": backend}, reasoning_effort="high", train_on_eot=True)
        )

        assert cfg.training.reasoning_effort == "high"
        assert cfg.training.train_on_eot is True

    def test_streaming_reads_the_formatter_fields(self):
        cfg = load_config_from_string(
            _yaml(stream_layers=True, batch_size=1, reasoning_effort="high", train_on_eot=True)
        )

        assert cfg.training.reasoning_effort == "high"
        assert cfg.training.train_on_eot is True

    def test_train_on_eot_keeps_loading_on_a_pretokenized_cache(self):
        """Honoured through the ``soup data preprocess`` cache key (#1054)."""
        cfg = load_config_from_string(
            _yaml(data={"format": "pre_tokenized", "tokenized_path": "./tok"}, train_on_eot=True)
        )

        assert cfg.training.train_on_eot is True

    @pytest.mark.parametrize("path", ["raft", "pre_tokenized"])
    def test_the_kernel_flags_still_load_where_the_resident_setup_runs(self, path):
        _, data, _, _, _ = _PATHS[path]

        cfg = load_config_from_string(_yaml(data=data, use_flash_attn=True, use_liger=True))

        assert cfg.training.use_flash_attn is True
        assert cfg.training.use_liger is True

    @pytest.mark.parametrize("path", ["raft", "pre_tokenized"])
    def test_freezing_still_loads_where_the_resident_setup_runs(self, path):
        _, data, _, _, _ = _PATHS[path]

        cfg = load_config_from_string(_yaml(data=data, freeze_layers=2))

        assert cfg.training.freeze_layers == 2

    def test_mlx_keeps_its_own_kernel_flag_warning(self):
        """MLX names both flags as ignored; a refusal here would be a second message."""
        cfg = load_config_from_string(
            _yaml({"backend": "mlx"}, use_flash_attn=True, use_liger=True)
        )

        assert cfg.training.use_flash_attn is True

    def test_streaming_keeps_the_1692_message_for_the_kernel_flags(self):
        with pytest.raises(ValueError) as info:
            load_config_from_string(_yaml(stream_layers=True, batch_size=1, use_liger=True))

        assert "mutually exclusive with use_liger" in str(info.value)

    def test_streaming_leaves_the_kernel_flags_to_1692(self):
        """#1692's validator raises first on a load, so call this one directly."""
        cfg = load_config_from_string(_yaml(stream_layers=True, batch_size=1))
        cfg.training.use_liger = True
        cfg.training.use_flash_attn = True

        assert cfg._validate_sft_path_is_read() is cfg

    @pytest.mark.parametrize("task", ["tts", "distill"])
    def test_the_other_formatter_tasks_keep_the_1587_behaviour(self, task):
        cfg = load_config_from_string(
            _yaml({"task": task}, reasoning_effort="high", train_on_eot=True)
        )

        assert cfg.training.reasoning_effort == "high"

    @pytest.mark.parametrize(
        ("top", "data", "flags"),
        [
            ({"task": "distill"}, {"format": "raft"}, {"reasoning_effort": "high"}),
            ({"task": "tts", "backend": "unsloth"}, {}, {"use_liger": True}),
        ],
    )
    def test_the_path_gate_is_for_task_sft_only(self, top, data, flags):
        """Checks the message, not a clean load, so it stays true if these tasks
        get path gates of their own."""
        try:
            load_config_from_string(_yaml(top, data, **flags))
        except ValueError as exc:
            assert "not applied by task='sft'" not in str(exc), str(exc)

    def test_a_task_outside_sft_keeps_the_1587_message(self):
        message = _refusal(_yaml({"task": "dpo"}, reasoning_effort="high"))

        assert "requires task in" in message, message
        assert "not applied by task='sft'" not in message, message


class TestTheTableIsTheDispatch:
    def test_the_rows_are_named_in_dispatch_order(self):
        assert [path.key for path in SFT_PATHS] == [
            "mlx",
            "vision",
            "audio",
            "stream",
            "unsloth",
            "pre_tokenized",
            "raft",
        ]

    def test_every_refused_cell_has_a_row_that_drops_it(self):
        rows = {path.key: path for path in SFT_PATHS}

        for path, fields in _REFUSED.items():
            for field in fields:
                assert field not in rows[path].reads, (path, field)
                assert field not in rows[path].warned, (path, field)
                assert field not in rows[path].refused_elsewhere, (path, field)

    def test_the_controls_are_read_by_the_row_that_keeps_them(self):
        rows = {path.key: path for path in SFT_PATHS}

        assert "reasoning_effort" in rows["stream"].reads
        assert "train_on_eot" in rows["pre_tokenized"].reads
        assert "use_liger" in rows["raft"].reads
        assert rows["stream"].refused_elsewhere == {"use_flash_attn", "use_liger"}
        assert rows["mlx"].warned == {"use_flash_attn", "use_liger"}

    def test_the_issue_fields_and_the_freeze_fields_share_one_table(self):
        assert SFT_PATH_FIELDS == (
            "reasoning_effort",
            "train_on_eot",
            "use_flash_attn",
            "use_liger",
        )
        # The freeze validator reads the same rows for its own two fields.
        assert "freeze_layers" in SFT_PATHS[-1].reads
        assert "freeze_layers" not in SFT_PATHS[0].reads


def _reads_any(source, names):
    """:func:`_reads_the_freeze_fields` from #1581, for any field name."""
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.Attribute) and node.attr in names:
            return True
    return False


class TestTheRowsFollowTheSetups:
    """Each row says what the code does. A setup that starts reading a field it
    is refused on fails here until its row is updated; a formatter call that
    appears in a second dataset branch does too."""

    @pytest.mark.parametrize(
        ("key", "method"),
        [
            ("vision", "_setup_vision_transformers"),
            ("audio", "_setup_audio_transformers"),
            ("unsloth", "_setup_unsloth"),
        ],
    )
    def test_these_setups_do_not_read_the_fields(self, key, method):
        source = inspect.getsource(getattr(SFTTrainerWrapper, method))
        names = set(_REFUSED[key])

        assert not _reads_any(source, names), (
            f"{method} now reads {sorted(names)}: drop its row from SFT_PATHS "
            "in _validate_sft_path_is_read"
        )

    def test_the_mlx_trainer_does_not_read_the_formatter_fields(self):
        source = inspect.getsource(MLXSFTTrainerWrapper)

        assert not _reads_any(source, {"reasoning_effort", "train_on_eot"}), (
            "MLXSFTTrainerWrapper now reads reasoning_effort / train_on_eot: "
            "drop the mlx row from SFT_PATHS"
        )

    @pytest.mark.parametrize("method", ["_prepare_raft_dataset", "_prepare_raft_raw_dataset"])
    def test_the_raft_builders_do_not_run_the_formatter(self, method):
        source = inspect.getsource(getattr(SFTTrainerWrapper, method))

        assert not _reads_any(source, {"reasoning_effort", "train_on_eot"}), (
            f"{method} now reads the formatter fields: drop the raft row from SFT_PATHS"
        )

    def test_the_formatter_has_one_call_site_in_the_text_branch(self):
        """``pre_tokenized`` and ``raft`` skip ``build_format_row``; the text
        branch is the only place it is built."""
        source = inspect.getsource(SFTTrainerWrapper.setup)

        assert source.count("build_format_row(") == 1, (
            "build_format_row is built more than once in setup: the rows that "
            "say a dataset branch skips it are no longer true"
        )
