"""Staged config fields must warn in v0.76 and refuse in v0.77 (#808).

17 config fields are declared in schema.py for features that have not landed,
accepted silently, and read by nothing. Following the #627 warn-then-refuse
pattern across two releases:
- While warning, each staged field prints:
    <field> is accepted but read by nothing; v<STAGED_FIELD_REJECTION_VERSION> will refuse it
- The deadline version is stated in STAGED_FIELD_REJECTION_VERSION ("0.77").
- TestTheDeadline asserts against soup_cli.__version__ in both directions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from soup_cli import __version__
from soup_cli.config import loader
from soup_cli.config.deprecation import DEPRECATED_VALUE_REJECTION_VERSION
from soup_cli.config.staged_fields import (
    STAGED_FIELD_REJECTION_VERSION,
    STAGED_FIELDS,
    find_staged_config_fields,
    format_staged_fields,
)

_VALID = """
base: hf/model
task: sft
data:
  train: ./t.jsonl
  format: auto
output: ./o
"""

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _raw(extra_data: str = "", extra_training: str = "") -> dict:
    import yaml

    doc = yaml.safe_load(_VALID)
    if extra_data:
        doc["data"].update(yaml.safe_load(extra_data))
    if extra_training:
        doc["training"] = yaml.safe_load(extra_training)
    return doc


def _valid_with_data(**kwargs) -> str:
    lines = [
        "base: hf/model",
        "task: sft",
        "data:",
        "  train: ./t.jsonl",
        "  format: auto",
    ]
    for k, v in kwargs.items():
        lines.append(f"  {k}: {v}")
    lines.append("output: ./o")
    return "\n".join(lines) + "\n"


class TestTheDeadline:
    """The warning must be a deadline, not a permanent decoration (#808).

    Mirrors the #627 pattern:
    - Named in the message
    - Stated in ONE constant
    - Enforced against soup_cli.__version__ in both directions
    """

    def test_the_warning_names_the_version_rather_than_a_vague_future(self) -> None:
        staged = find_staged_config_fields(_raw(extra_training="{long_context_grpo: true}"))
        assert len(staged) == 1
        msg = format_staged_fields(staged, include_deadline=True)
        assert f"v{STAGED_FIELD_REJECTION_VERSION}" in msg
        assert "future release" not in msg.lower()

    def test_the_version_is_written_out_in_exactly_one_source_file(self) -> None:
        """The constant is defined in exactly one source file."""
        assert STAGED_FIELD_REJECTION_VERSION == "0.77"

        pattern = re.compile(
            r"^\s*STAGED_FIELD_REJECTION_VERSION\s*(?::\s*str\s*)?=", re.MULTILINE
        )
        src = Path(__file__).parents[1] / "src" / "soup_cli"
        holders = sorted(
            p.relative_to(src).as_posix()
            for p in src.rglob("*.py")
            if pattern.search(p.read_text(encoding="utf-8"))
        )
        assert holders == ["config/staged_fields.py"], (
            f"the deadline constant is defined in more than one place: {holders}"
        )

    def test_the_deadline_is_enforced_against_the_declared_version(self) -> None:
        """While version < deadline, severity is 'warn'; once reached, 'error'."""
        declared = tuple(int(p) for p in __version__.split(".")[:2])
        deadline = tuple(int(p) for p in STAGED_FIELD_REJECTION_VERSION.split(".")[:2])

        if declared < deadline:
            assert loader.STAGED_FIELD_SEVERITY == "warn", (
                f"v{__version__} is before the v{STAGED_FIELD_REJECTION_VERSION} "
                "deadline, so staged fields must warn, not refuse"
            )
        else:
            assert loader.STAGED_FIELD_SEVERITY == "error", (
                f"v{__version__} has reached the v{STAGED_FIELD_REJECTION_VERSION} "
                "deadline: set STAGED_FIELD_SEVERITY to 'error'"
            )

    def test_the_deadline_wording_changes_once_the_switch_rejects(self) -> None:
        """Under refusal, message says 'refused as of v...' rather than 'will refuse it'."""
        staged = find_staged_config_fields(_raw(extra_training="{long_context_grpo: true}"))
        msg = format_staged_fields(staged, include_deadline=False)
        assert f"refused as of v{STAGED_FIELD_REJECTION_VERSION}" in msg
        assert "will refuse it" not in msg


class TestStagedFieldsInventory:
    """Validate inventory size and default handling across all 31 staged fields."""

    def test_exactly_31_staged_fields_registered(self) -> None:
        # 17 from #808 initial, 3 from #761, 9 Group B knobs and 2 RL checkpoint flags (#808).
        assert len(STAGED_FIELDS) == 31, (
            f"Expected exactly 31 staged fields, got {len(STAGED_FIELDS)}"
        )

    @pytest.mark.parametrize(
        "section,name,non_default_value",
        [
            ("training", "long_context_grpo", True),
            ("training", "vision_grpo", True),
            ("training", "quantize_ref_model", True),
            ("training", "grace_codebook", True),
            ("training", "grace_codebook_size", 1024),
            ("training", "grace_codebook_dim", 64),
            ("training", "lr_groups", [{"pattern": "lm_head", "lr": 1e-5}]),
            ("training", "early_stop_patience", 5),
            ("training", "citation_recall_threshold", 0.8),
            ("training", "forgetting_eval_steps", 250),
            ("training", "forgetting_benchmark", "mini_common_sense"),
            ("training", "forgetting_stop", True),
            ("training", "checkpoint_eval_steps", 500),
            ("training", "checkpoint_eval_metric", "judge"),
            ("training", "checkpoint_eval_tasks", "arc,hellaswag"),
            ("training", "checkpoint_keep_top", 5),
            ("training", "convergence_window", 100),
            ("training", "convergence_rel_tol", 0.01),
            ("data", "video_dir", "./videos"),
            ("data", "video_fps", 2.0),
            ("data", "video_maxlen", 64),
            ("data", "image_min_pixels", 256),
            ("data", "image_max_pixels", 1024),
            ("data", "image_resize_algorithm", "bilinear"),
            ("data", "split_thinking", True),
            ("data", "eval_on_each_dataset", True),
            ("data", "resize_vocab", True),
            ("data", "extend_conversation", True),
            ("data", "skip_prepare_dataset", True),
            ("training", "rl_checkpoint_include_ref_model", True),
            ("training", "rl_checkpoint_include_rollout_buffer", True),
        ],
    )
    def test_each_staged_field_detected_when_non_default(
        self, section: str, name: str, non_default_value
    ) -> None:
        raw = {"training": {}, "data": {}}
        raw[section][name] = non_default_value
        found = find_staged_config_fields(raw)
        assert len(found) == 1
        assert found[0].section == section
        assert found[0].name == name
        assert found[0].path == f"{section}.{name}"
        assert found[0].value == non_default_value

    def test_staged_fields_are_subset_of_known_unconsumed(self) -> None:
        """Every staged field must be recorded in #748's KNOWN_UNCONSUMED allowlist."""
        from tests.test_issue748_config_fields_reach_a_consumer import KNOWN_UNCONSUMED

        staged_paths = {f"{s}.{n}" for s, n in STAGED_FIELDS}
        assert staged_paths <= set(KNOWN_UNCONSUMED), (
            f"Staged fields not in KNOWN_UNCONSUMED: {staged_paths - set(KNOWN_UNCONSUMED)}"
        )

    def test_schema_defaults_match_pydantic_model_fields(self) -> None:
        """Anti-circular ratchet: default in STAGED_FIELDS must match schema model_fields (#808)."""
        from soup_cli.config.schema import DataConfig, TrainingConfig

        configs = {
            "training": TrainingConfig.model_fields,
            "data": DataConfig.model_fields,
        }
        for (section, name), table_default in STAGED_FIELDS.items():
            assert name in configs[section], f"{section}.{name} not found in schema"
            schema_default = configs[section][name].default
            assert schema_default == table_default, (
                f"STAGED_FIELDS[{section}.{name}] default {table_default!r} does not "
                f"match Pydantic schema default {schema_default!r}"
            )

    def test_schema_defaults_produce_zero_staged_findings(self) -> None:
        raw = {"training": {}, "data": {}}
        for (sec, name), default_val in STAGED_FIELDS.items():
            raw[sec][name] = default_val
        found = find_staged_config_fields(raw)
        assert found == [], f"Default values should not trigger warnings: {found}"

    def test_scientific_notation_and_coerced_defaults_produce_zero_staged_findings(
        self,
    ) -> None:
        """Raw values like 'false' matching schema defaults produce no findings."""
        raw = {
            "training": {
                "grace_codebook": "false",
                "long_context_grpo": "false",
            },
            "data": {
                "split_thinking": "false",
                "resize_vocab": "false",
            },
        }
        found = find_staged_config_fields(raw)
        assert found == []

    def test_root_level_misplaced_keys_remapped_before_staged_check(self) -> None:
        """Root-level misplaced keys are normalised via remap_root_level_misplaced_keys (#808)."""
        from unittest.mock import patch

        import soup_cli.config.schema as schema_mod

        raw = {"staged_dummy": 999}
        with patch.object(schema_mod, "ROOT_LEVEL_MISPLACED_KEYS", ("staged_dummy",)):
            with patch.dict(STAGED_FIELDS, {("training", "staged_dummy"): 0}):
                found = find_staged_config_fields(raw)
                assert len(found) == 1
                assert found[0].path == "training.staged_dummy"
                assert found[0].value == 999

    @pytest.mark.parametrize(
        "name", ["rl_checkpoint_include_ref_model", "rl_checkpoint_include_rollout_buffer"]
    )
    def test_the_staged_rl_checkpoint_flags_say_nothing_is_saved(self, name: str) -> None:
        """The descriptions must not drift back to claiming the state is saved (#808)."""
        from soup_cli.config.schema import TrainingConfig

        assert "Not saved yet" in TrainingConfig.model_fields[name].description


class TestLoaderStagedFieldIntegration:
    """Test loader behavior under warning vs error severity."""

    def test_staged_field_warns_and_loads_under_warn_severity(
        self, capsys, monkeypatch
    ) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")
        yaml_str = _valid_with_data(split_thinking="true")
        cfg = loader.load_config_from_string(yaml_str)
        assert cfg.data.split_thinking is True
        plain_out = _plain(capsys.readouterr().out)
        assert "data.split_thinking" in plain_out
        assert "accepted but read by nothing" in plain_out
        assert STAGED_FIELD_REJECTION_VERSION in plain_out
        assert "will refuse it" in plain_out

    def test_staged_field_refuses_under_error_severity(self, monkeypatch) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
        yaml_str = _valid_with_data(split_thinking="true")
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(yaml_str)
        msg = str(excinfo.value)
        assert "data.split_thinking is accepted but read by nothing" in msg
        assert f"v{STAGED_FIELD_REJECTION_VERSION}" in msg

    def test_multiple_staged_fields_reported_together(self, monkeypatch) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
        yaml_str = _valid_with_data(split_thinking="true", video_fps="24.0")
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(yaml_str)
        msg = str(excinfo.value)
        assert "data.split_thinking is accepted but read by nothing" in msg
        assert "data.video_fps is accepted but read by nothing" in msg

    def test_load_config_file_warns(self, tmp_path, capsys, monkeypatch) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(_valid_with_data(split_thinking="true"))
        cfg = loader.load_config(cfg_file)
        assert cfg.data.split_thinking is True
        plain_out = _plain(capsys.readouterr().out)
        assert "data.split_thinking" in plain_out
        assert "accepted but read by nothing" in plain_out
        assert STAGED_FIELD_REJECTION_VERSION in plain_out
        assert "will refuse it" in plain_out

    def test_load_config_file_refuses_under_error(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(_valid_with_data(split_thinking="true"))
        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(cfg_file)
        assert excinfo.value.code == 1

    @pytest.mark.parametrize("severity", ["warn", "error"])
    @pytest.mark.parametrize(
        "field,spelling",
        [("split_thinking", "'false'"), ("resize_vocab", "'off'"), ("video_dir", "''")],
    )
    def test_default_equivalent_spellings_load_silently(
        self, severity: str, field: str, spelling: str, monkeypatch, capsys
    ) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", severity)
        loader.load_config_from_string(_valid_with_data(**{field: spelling}))
        assert "read by nothing" not in _plain(capsys.readouterr().out)

    @pytest.mark.parametrize("severity", ["warn", "error"])
    def test_load_config_file_default_equivalent_spelling_loads_silently(
        self, severity: str, tmp_path, capsys, monkeypatch
    ) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", severity)
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(_valid_with_data(split_thinking="'false'", video_dir="''"))
        loader.load_config(cfg_file)
        assert "read by nothing" not in _plain(capsys.readouterr().out)

    @pytest.mark.parametrize("severity", ["warn", "error"])
    def test_convergence_rel_tol_scientific_notation_staged_field_loads_silently(
        self, severity: str, monkeypatch, capsys
    ) -> None:
        """convergence_rel_tol: 5e-3 matching schema default loads cleanly as a staged field."""
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", severity)
        yaml_str = _VALID + "\ntraining:\n  convergence_rel_tol: 5e-3\n"
        cfg = loader.load_config_from_string(yaml_str)
        assert cfg.training.convergence_rel_tol == 0.005
        assert capsys.readouterr().out == ""

    def test_group_b_tunables_no_longer_listed_in_train_command(
        self, monkeypatch
    ) -> None:
        """Group B tunables are staged fields; train command lists only enabled flags."""
        from soup_cli.commands.train import _nondefault_unwired_training_settings

        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")
        yaml_str = _VALID + "\ntraining:\n  convergence_rel_tol: 5e-3\n  convergence_window: 50\n"
        cfg = loader.load_config_from_string(yaml_str)
        assert _nondefault_unwired_training_settings(cfg.training) == []

        yaml_nondef = (
            _VALID
            + "\ntraining:\n  convergence_detection: true\n"
            + "  convergence_window: 100\n  convergence_rel_tol: 0.01\n"
            + "  forgetting_eval_steps: 250\n  checkpoint_keep_top: 5\n"
        )
        cfg_nondef = loader.load_config_from_string(yaml_nondef)
        assert _nondefault_unwired_training_settings(cfg_nondef.training) == [
            "convergence_detection",
        ]


class TestGroupBTunablesIntegration:
    """Group B knobs warn at load and refuse under v0.77 deadline (#808)."""

    _GROUP_B_CASES = {
        "training.forgetting_eval_steps": "training:\n  forgetting_eval_steps: 250\n",
        "training.forgetting_benchmark": "training:\n  forgetting_benchmark: mini_common_sense\n",
        "training.forgetting_stop": "training:\n  forgetting_stop: true\n",
        "training.checkpoint_eval_steps": "training:\n  checkpoint_eval_steps: 500\n",
        "training.checkpoint_eval_metric": "training:\n  checkpoint_eval_metric: judge\n",
        "training.checkpoint_eval_tasks": "training:\n  checkpoint_eval_tasks: arc,hellaswag\n",
        "training.checkpoint_keep_top": "training:\n  checkpoint_keep_top: 5\n",
        "training.convergence_window": "training:\n  convergence_window: 100\n",
        "training.convergence_rel_tol": "training:\n  convergence_rel_tol: 0.01\n",
    }

    def test_soup_train_note_names_no_staged_field(self, monkeypatch) -> None:
        """Each knob is reported once, by the loader; soup train's note adds none (#808)."""
        from soup_cli.commands.train import (
            _UNWIRED_TRAINING_TUNABLES,
            _nondefault_unwired_training_settings,
        )

        staged = {name for section, name in STAGED_FIELDS if section == "training"}
        assert not staged & set(_UNWIRED_TRAINING_TUNABLES)

        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")
        body = "".join(case.split("\n", 1)[1] for case in self._GROUP_B_CASES.values())
        cfg = loader.load_config_from_string(_VALID + "\ntraining:\n" + body)
        assert _nondefault_unwired_training_settings(cfg.training) == []

    @pytest.mark.parametrize("path", sorted(_GROUP_B_CASES))
    def test_group_b_knob_warns_with_deadline_under_warn_severity(
        self, path: str, capsys, monkeypatch
    ) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")
        yaml_str = _VALID + "\n" + self._GROUP_B_CASES[path]
        loader.load_config_from_string(yaml_str)
        plain_out = _plain(capsys.readouterr().out)
        assert f"{path} is accepted but read by nothing" in plain_out
        assert f"v{STAGED_FIELD_REJECTION_VERSION} will refuse it" in plain_out

    @pytest.mark.parametrize("path", sorted(_GROUP_B_CASES))
    def test_group_b_knob_refuses_under_error_severity(
        self, path: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
        yaml_str = _VALID + "\n" + self._GROUP_B_CASES[path]
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(yaml_str)
        assert f"{path} is accepted but read by nothing" in str(excinfo.value)
        assert f"v{STAGED_FIELD_REJECTION_VERSION}" in str(excinfo.value)

    @pytest.mark.parametrize("severity", ["warn", "error"])
    def test_group_b_defaults_stay_silent(
        self, severity: str, capsys, monkeypatch
    ) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", severity)
        yaml_defaults = (
            _VALID
            + "\ntraining:\n"
            + "  forgetting_eval_steps: 100\n"
            + "  forgetting_benchmark: mini_mmlu\n"
            + "  forgetting_stop: false\n"
            + "  checkpoint_eval_steps: 200\n"
            + "  checkpoint_eval_metric: composite\n"
            + "  checkpoint_eval_tasks: null\n"
            + "  checkpoint_keep_top: 3\n"
            + "  convergence_window: 50\n"
            + "  convergence_rel_tol: 0.005\n"
        )
        loader.load_config_from_string(yaml_defaults)
        assert "read by nothing" not in _plain(capsys.readouterr().out)


_LATE_STAGED_RL_FLAGS = ["rl_checkpoint_include_ref_model", "rl_checkpoint_include_rollout_buffer"]


def _valid_with_training(**kwargs) -> str:
    lines = ["base: hf/model", "task: sft", "data:", "  train: ./t.jsonl", "training:"]
    lines += [f"  {k}: {v}" for k, v in kwargs.items()]
    lines.append("output: ./o")
    return "\n".join(lines) + "\n"


class TestTheLateStagedRlCheckpointFlags:
    """#1980 staged these two after v0.76.0, so no release warned about them before
    v0.77 refuses them (ruling 2026-10-21, one deadline constant). Only a value that
    ever claimed something is refused: an explicit ``false``, the default that never
    made the manifest lie, still loads, silently."""

    @pytest.mark.parametrize("name", _LATE_STAGED_RL_FLAGS)
    def test_an_explicit_false_loads_silently(self, name: str, capsys) -> None:
        cfg = loader.load_config_from_string(_valid_with_training(**{name: "false"}))
        assert getattr(cfg.training, name) is False
        assert "read by nothing" not in _plain(capsys.readouterr().out)

    @pytest.mark.parametrize("name", _LATE_STAGED_RL_FLAGS)
    def test_true_is_refused_with_the_one_line_fix(self, name: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(_valid_with_training(**{name: "true"}))
        assert (
            f"training.{name} is accepted but read by nothing; "
            f"refused as of v{STAGED_FIELD_REJECTION_VERSION}. Delete the line."
        ) in str(excinfo.value)


# ---------------------------------------------------------------------------
# The staged refusal comes before a staged field's own validator (#808)
# ---------------------------------------------------------------------------

_SFT_ALPACA = (
    "base: hf/model\ntask: sft\ndata:\n  train: ./t.jsonl\n  format: alpaca\n{data}"
    "training:\n  epochs: 1\n{training}output: ./o\n"
)

#: refused field -> (data lines, training lines, the advice its own validator gave)
_OWN_VALIDATOR_CASES = {
    "training.grace_codebook": ("", "  grace_codebook: true\n", "requires BOTH"),
    "training.long_context_grpo": ("", "  long_context_grpo: true\n", "requires task="),
    "data.video_fps": ("  video_fps: 2.0\n", "", "require format in"),
}

_RUC_REFUSAL_LINE = (
    "  data.remove_unused_columns: true never had an effect "
    "(no trainer reads it; the trainers that set it pass false). Delete the line. "
    f"Refused as of Soup v{DEPRECATED_VALUE_REJECTION_VERSION}."
)


def _staged_line(path: str) -> str:
    return (
        f"{path} is accepted but read by nothing; "
        f"refused as of v{STAGED_FIELD_REJECTION_VERSION}. Delete the line."
    )


@pytest.fixture
def printed(monkeypatch):
    """What the loader prints, uncoloured and unwrapped."""
    import io

    from rich.console import Console

    buffer = io.StringIO()
    monkeypatch.setattr(
        loader, "console", Console(file=buffer, width=500, force_terminal=False, color_system=None)
    )
    return buffer


class TestTheRefusalComesBeforeTheFieldsOwnValidator:
    """Some staged fields have a cross-validator of their own. When it failed, the
    loader reported it alone: ``grace_codebook: true`` asked for
    ``grace_codebook_size`` and ``grace_codebook_dim``, ``long_context_grpo: true``
    for ``task: grpo``, ``video_fps`` for a video format -- and the next load
    refused every line the user had just added. The staged refusal is now the first
    line and the validator's advice is not given."""

    @pytest.mark.parametrize("path", sorted(_OWN_VALIDATOR_CASES))
    def test_the_string_load_gives_only_the_staged_line(self, path: str) -> None:
        data, training, _advice = _OWN_VALIDATOR_CASES[path]
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(_SFT_ALPACA.format(data=data, training=training))
        assert str(excinfo.value) == _staged_line(path)

    @pytest.mark.parametrize("path", sorted(_OWN_VALIDATOR_CASES))
    def test_soup_train_prints_the_staged_line_and_not_the_advice(
        self, path: str, tmp_path, monkeypatch
    ) -> None:
        from typer.testing import CliRunner

        from soup_cli.cli import app

        data, training, advice = _OWN_VALIDATOR_CASES[path]
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(
            _SFT_ALPACA.format(data=data, training=training), encoding="utf-8"
        )
        result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
        out = _plain(result.output)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert f"Config validation error: {_staged_line(path)}" in out, out
        assert advice not in out, out

    def test_a_value_refusal_follows_the_staged_line_in_the_same_report(
        self, printed, tmp_path
    ) -> None:
        path = tmp_path / "soup.yaml"
        path.write_text(
            _SFT_ALPACA.format(
                data="  remove_unused_columns: true\n", training="  grace_codebook: true\n"
            ),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)
        assert excinfo.value.code == 1
        assert printed.getvalue().splitlines() == [
            "Config validation error:",
            "",
            f"  {_staged_line('training.grace_codebook')}",
            _RUC_REFUSAL_LINE,
        ]

    def test_the_string_load_puts_an_unrelated_error_on_the_next_line(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(
                _SFT_ALPACA.format(data="", training="  lr: -1\n  early_stop_patience: 5\n")
            )
        assert str(excinfo.value) == (
            _staged_line("training.early_stop_patience")
            + "\ntraining -> lr: Input should be greater than 0"
        )

    def test_a_longer_field_name_is_not_taken_for_the_refused_one(self) -> None:
        names = ("grace_codebook",)
        assert loader._names_a_field({"loc": ("training", "grace_codebook"), "msg": "x"}, names)
        assert loader._names_a_field({"loc": (), "msg": "training.grace_codebook=true"}, names)
        assert not loader._names_a_field(
            {"loc": ("training", "grace_codebook_size"), "msg": "grace_codebook_size: 0"}, names
        )

    def test_under_warn_the_validator_is_reported_as_before(self, monkeypatch) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(
                _SFT_ALPACA.format(data="", training="  grace_codebook: true\n")
            )
        assert str(excinfo.value) == (
            ": Value error, training.grace_codebook=true requires BOTH "
            "training.grace_codebook_size and training.grace_codebook_dim."
        )


# ---------------------------------------------------------------------------
# soup sweep --param applies values after the load (#808)
# ---------------------------------------------------------------------------


def _sweep_refusal(paths: str) -> str:
    return (
        f"sweep parameter sets a field that is accepted but read by nothing: {paths}; "
        f"refused as of v{STAGED_FIELD_REJECTION_VERSION}. Remove it from --param."
    )


class TestASweepParameterCannotSetAStagedField:
    """``soup sweep`` applies each ``--param`` to a ``model_dump()`` of the loaded
    config and builds ``SoupConfig`` from it directly, so a staged field set there
    skipped the loader's refusal."""

    def _swept(self, **params: object) -> dict:
        """A config dict as ``_run_single`` builds it: dump plus the --param values."""
        from soup_cli.commands.sweep import _set_nested_param
        from soup_cli.config.schema import SoupConfig

        config_dict = SoupConfig(**_raw()).model_dump()
        for key, value in params.items():
            _set_nested_param(config_dict, key, value)
        return config_dict

    def test_a_staged_field_away_from_its_default_is_refused(self) -> None:
        from soup_cli.commands.sweep import _reject_staged_sweep_params

        with pytest.raises(ValueError) as excinfo:
            _reject_staged_sweep_params(
                self._swept(**{"training.early_stop_patience": 5, "data.split_thinking": True})
            )
        assert str(excinfo.value) == _sweep_refusal(
            "training.early_stop_patience, data.split_thinking"
        )

    def test_its_default_and_a_real_field_are_not_refused(self) -> None:
        """The control: a guard that refused every sweep would pass the test above."""
        from soup_cli.commands.sweep import _reject_staged_sweep_params

        _reject_staged_sweep_params(self._swept(**{"training.early_stop_patience": 2}))
        _reject_staged_sweep_params(self._swept(**{"training.lr": 1e-5}))

    def test_under_warn_nothing_is_refused(self, monkeypatch) -> None:
        from soup_cli.commands.sweep import _reject_staged_sweep_params

        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "warn")
        _reject_staged_sweep_params(self._swept(**{"training.early_stop_patience": 5}))

    def test_run_single_refuses_before_it_builds_the_config(self) -> None:
        from soup_cli.commands.sweep import _run_single
        from soup_cli.config.schema import SoupConfig

        with pytest.raises(ValueError) as excinfo:
            _run_single(
                SoupConfig(**_raw()),
                {"training.early_stop_patience": 5},
                "sweep_1",
                Path("soup.yaml"),
            )
        assert str(excinfo.value) == _sweep_refusal("training.early_stop_patience")

    def test_the_command_exits_1_when_only_a_later_arm_sets_it(self, tmp_path) -> None:
        """The first combination holds the default, so a probe of ``combinations[0]``
        alone would let the grid start; the second arm would then fail inside the
        arm loop, which records it and exits 0."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        cfg = tmp_path / "soup.yaml"
        cfg.write_text("base: test-model\ndata:\n  train: ./data.jsonl\n", encoding="utf-8")
        result = CliRunner().invoke(app, [
            "sweep",
            "--config", str(cfg),
            "--param", "training.early_stop_patience=2,5",
            "--yes",
        ])
        out = _plain(result.output)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert _sweep_refusal("training.early_stop_patience") in out, out
        assert "--- Run 1/2" not in out, out
